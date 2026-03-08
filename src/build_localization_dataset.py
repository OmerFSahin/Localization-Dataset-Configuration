from __future__ import annotations

import argparse
import json
import shutil
import gzip
from pathlib import Path
from typing import Tuple

import numpy as np
import nrrd


VALID_EXTS = (".nrrd", ".nhdr", ".gz")


def load_nrrd(path: Path) -> Tuple[np.ndarray, dict]:
    """Load plain or gzip-compressed NRRD."""
    try:
        return nrrd.read(str(path))
    except Exception as e_plain:
        try:
            with gzip.open(path, "rb") as gzf:
                return nrrd.read(gzf)
        except Exception as e_gzip:
            raise IOError(f"{path.name} not valid NRRD: {e_plain} / {e_gzip}")


def bbox_mm_from_mask(mask: np.ndarray, header: dict, margin: int) -> list:
    """Compute bounding box in millimeter space from mask."""
    if mask.ndim != 3:
        raise ValueError("Mask must be 3D.")

    nz = np.where(mask > 0)
    if nz[0].size == 0:
        raise ValueError("Mask is empty.")

    z0, y0, x0 = [int(c.min()) - margin for c in nz]
    z1, y1, x1 = [int(c.max()) + margin for c in nz]

    D, H, W = mask.shape
    x0, x1 = np.clip([x0, x1], 0, W - 1)
    y0, y1 = np.clip([y0, y1], 0, H - 1)
    z0, z1 = np.clip([z0, z1], 0, D - 1)

    directions = np.asarray(header["space directions"])
    origin = np.asarray(header["space origin"])

    p_min = origin + directions.T @ np.array([x0, y0, z0])
    p_max = origin + directions.T @ np.array([x1, y1, z1])

    bbox = [
        float(min(p_min[i], p_max[i])) for i in range(3)
    ] + [
        float(max(p_min[i], p_max[i])) for i in range(3)
    ]

    return bbox


def process_patient(pdir: Path, output_root: Path, margin: int) -> None:
    scan = next(
        (f for f in pdir.iterdir() if "(scan)" in f.name.lower() and f.suffix in VALID_EXTS),
        None,
    )
    mask = next(
        (f for f in pdir.iterdir() if "(mask)" in f.name.lower() and f.suffix in VALID_EXTS),
        None,
    )

    if not scan or not mask:
        raise FileNotFoundError("Scan or mask missing.")

    mask_data, mask_hdr = load_nrrd(mask)
    bbox = bbox_mm_from_mask(mask_data, mask_hdr, margin)

    out_dir = output_root / pdir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(scan, out_dir / scan.name)

    with open(out_dir / "meta.json", "w") as f:
        json.dump({"bbox_mm": bbox}, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate localization dataset configuration (bbox in mm) from segmentation masks."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--margin", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    patients = [p for p in sorted(args.input_dir.iterdir()) if p.is_dir()]
    print(f"Found {len(patients)} patient folders")

    for pdir in patients:
        try:
            process_patient(pdir, args.output_dir, args.margin)
        except Exception as e:
            print(f"[SKIP] {pdir.name}: {e}")
            continue

        print(f"[OK] {pdir.name}: meta.json written")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

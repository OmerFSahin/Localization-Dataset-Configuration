# Localization Dataset Configuration & Packaging

Generates bounding box metadata (in millimeter space) from segmentation masks to prepare datasets for localization model training.

The script computes the 3D bounding box of foreground voxels and exports a `meta.json` file for each patient.

Tested on 3D cardiac MRI segmentation datasets.
The dataset is not included due to licensing and size considerations.

---

## Expected Structure

```
Localization-Dataset-Configuration/
├─ README.md
├─ requirements.txt
├─ src/
│   └─ build_localization_dataset.py
└─ data/
    ├─ input/
    └─ output/
```

Input:
```
data/input/
  Patient_001/
    Patient_001(scan).nrrd
    Patient_001(mask).nrrd
```
Output:
```
data/output/Patient_001/
  Patient_001(scan).nrrd
  meta.json
```

---

## Usage

python src/build_localization_dataset.py \
  --input-dir data/input \
  --output-dir data/output \
  --margin 10

---

## What It Does

- Reads 3D segmentation masks
- Computes bounding box of non-zero voxels
- Converts voxel coordinates to millimeter space
- Saves bbox as:
```
{
  "bbox_mm": [xmin, ymin, zmin, xmax, ymax, zmax]
}
```
- Copies scan file to output folder

---

## Requirements

numpy 
pynrrd 

Install:
```
pip install -r requirements.txt
```
## Output Example

```
 data/output/pat0/meta.json
{
    "bbox_mm": [
        -149.7547378540039,
        -164.9680023193359,
        -55.59241485595706,
        -38.50486630201338,
        -34.81170558929438,
        117.94931411743164
    ]
}

```

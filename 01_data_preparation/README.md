# Stage 1: Data Preparation & Annotation Pipeline

This module handles parsing raw CVAT XML annotations, mapping plant labels to a unified 4-class taxonomy, extracting polygon instance segmentations and root keypoints, and generating normalized YOLO-formatted datasets.

---

## 📋 Directory Structure

```
01_data_preparation/
├── scripts/
│   ├── 01_analyze_xml.py          # Analyze XML annotation statistics & bounds
│   ├── 02_convert_xml_to_yolo.py  # Convert CVAT XML -> YOLO (4 classes + root point)
│   ├── 03_visualize_dataset.py    # Overlay bounding boxes, masks & root keypoints
│   └── 04_resplit_dataset.py      # Split dataset (80% Train, 5% Val, 15% Test)
└── README.md
```

---

## 🏷️ Class Taxonomy Mapping

| Class ID | Target Class Name | Original CVAT XML Labels | Description |
|---|---|---|---|
| `0` | `crop_small_leaf` | `crop6` | Young crop seedlings |
| `1` | `crop_large_leaf` | `crop1`, `crop2`, `crop3`, `crop4`, `crop5` | Mature crops with broad canopy |
| `2` | `weed_small_leaf` | `weed1`, `weed2`, `weed3`, `weed4` | Early-stage competitive weeds |
| `3` | `weed_large_leaf` | `weed5` through `weed14` | Mature weed clusters requiring removal |

---

## 💻 Step-by-Step Usage Guide

### Step 1: Analyze Raw XML Annotations
Parses raw CVAT XML export files to inspect annotation counts, element tags (`points`, `polygon`, `polyline`), and check for out-of-bounds coordinates.

```bash
python 01_data_preparation/scripts/01_analyze_xml.py data/raw/xml_takeo-annotation-done/fifth.xml
```

### Step 2: Convert CVAT XML to YOLO Segmentation + Root Point Format
Extracts instance polygons and pairs each plant with its stem root keypoint coordinate. Normalizes coordinates relative to image width and height.

```bash
python 01_data_preparation/scripts/02_convert_xml_to_yolo.py \
    --xml data/raw/xml_takeo-annotation-done/fifth.xml \
    --img-dir data/raw/dataset \
    --out-dir data/processed/yolo_dataset_4classes
```

**Output Label Format (`.txt`)**:
Each object line is formatted as:
```
class_id  root_x  root_y  x1 y1 x2 y2 ... xN yN
```

### Step 3: Visualize Converted Dataset
Renders bounding boxes, translucent polygon masks, class labels, and root markers directly onto images to visually verify label accuracy.

```bash
python 01_data_preparation/scripts/03_visualize_dataset.py \
    --dataset data/processed/yolo_dataset_4classes \
    --output output/dataset_previews
```

### Step 4: Re-Split Dataset into Train / Val / Test
Gathers all image-label pairs and splits them cleanly into **80% Train**, **5% Validation**, and **15% Test** sets. Generates the dataset configuration file `data.yaml`.

```bash
python 01_data_preparation/scripts/04_resplit_dataset.py \
    --src data/processed/yolo_dataset_4classes \
    --out data/processed
```

---

## 💡 Developer & Maintainer Notes

* **Splitting Strategy**: Dataset partitioning uses random image shuffling with a fixed random seed (`seed=42`) to guarantee 100% reproducible splits across experiment runs.
* **Image-Level Isolation**: Splitting is executed strictly at the image level so that no individual image appears in multiple splits (preventing data leakage).
* **Future Work Note for New Maintainers**: If you significantly expand the dataset in the future and notice class distribution imbalance in the validation set, consider updating `04_resplit_dataset.py` to use **Multi-Label Stratified Splitting** (`scikit-multilearn` or `iterative-stratification`).

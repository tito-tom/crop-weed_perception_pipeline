# Stage 1 Documentation: Data Collection & Annotation Protocol

## 1. Taxonomic Classification Scheme

The dataset distinguishes crops and weeds based on leaf morphology into 4 target classes:

| Class ID | Class Name | Plant Category | Description |
|---|---|---|---|
| `0` | `crop_small_leaf` | Crop | Seedlings / young crops with small leaf surface area |
| `1` | `crop_large_leaf` | Crop | Mature crops with broad, developed canopy leaves |
| `2` | `weed_small_leaf` | Weed | Early-stage weeds competing for nutrients |
| `3` | `weed_large_leaf` | Weed | Dense/mature weed clusters requiring destruction |

## 2. CVAT Annotation Rules

Annotators use **CVAT (Computer Vision Annotation Tool)** following strict rules:
1. **Polygon Outer Boundary**: Trace the complete visible outer leaf contour for instance segmentation.
2. **Root Keypoint Tag**: Place a single `points` annotation at the stem root origin touching the soil plane.
3. **Point-to-Polygon Pairing**: Automated conversion script pairs each root keypoint with its enclosing plant polygon contour.

## 3. YOLO Label File Specification

Normalized line format for each object in `.txt` files:
$$ \text{class\_id} \quad \hat{x}_{root} \quad \hat{y}_{root} \quad \hat{x}_1 \quad \hat{y}_1 \quad \hat{x}_2 \quad \hat{y}_2 \quad \dots \quad \hat{x}_N \quad \hat{y}_N $$

Coordinates are normalized relative to image width $W$ and height $H$:
$$ \hat{x} = \frac{x}{W}, \quad \hat{y} = \frac{y}{H} $$

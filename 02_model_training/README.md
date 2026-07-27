# Stage 2: Dual-Head Deep Learning Model Training & ONNX Export

This stage defines the custom joint **YOLOv11 Instance Segmentation + Root Keypoint Regression** model architecture, PyTorch dataset loaders, multi-task loss formulation, training loop, evaluation metrics, and ONNX export utility.

---

## 📋 Module Overview

```
02_model_training/
├── config.py              # Central configuration (paths, learning rate, loss gains)
├── yolo_seg_root.yaml     # Model architecture blueprint (backbone, neck, custom head)
├── modules.py             # CustomSegmentHead & cv5 root keypoint regression branch
├── model_builder.py       # Model constructor & Ultralytics parser registration
├── dataset.py             # PyTorch dataset loader with Albumentations
├── loss.py                # Multi-task loss (Box CIoU, Seg BCE, Cls BCE, DFL, Root L1)
├── metrics.py             # Evaluation metrics (mAP50, mAP50-95, PCK@10)
├── train.py               # Main training script
├── val.py                 # Standalone model evaluation script
├── export_onnx.py         # Export PyTorch (.pt) weights to ONNX format
└── README.md
```

---

## ⚙️ Configuration (`config.py`)

All hyperparameters, paths, and loss multipliers are centralized in `config.py`. Key settings include:

```python
# Model & Dataset Settings
NUM_CLASSES   = 4
IMG_SIZE      = 640
BATCH_SIZE    = 8
EPOCHS        = 50
LEARNING_RATE = 5e-4
DEVICE        = "cuda"  # Auto-detects GPU, falls back to CPU

# Multi-Task Loss Gains
LOSS_BOX = 7.5    # Bounding box regression gain
LOSS_CLS = 0.5    # Classification gain
LOSS_DFL = 1.5    # Distribution focal loss gain
LOSS_KPT = 12.0   # Root keypoint loss gain
```

---

## 💻 Step-by-Step Usage Guide

### Step 1: Prepare Environment & Install Dependencies
Ensure PyTorch and Ultralytics are installed:

```bash
pip install torch torchvision ultralytics albumentations onnx onnxruntime
```

### Step 2: Run Model Training
Launches training for the dual-head YOLOv11 model. Training checkpoints (`best.pt` and periodic epoch checkpoints) are automatically saved to `runs/train/`.

```bash
cd 02_model_training
python train.py
```

* **Monitoring**: Epoch training loss, elapsed time, and best model checkpoints will be printed to stdout.

### Step 3: Run Standalone Evaluation
Evaluates a trained checkpoint on the validation/test split. Calculates bounding box precision, recall, mAP50, mAP50-95, and Root Keypoint **PCK@10** (Percentage of Correct Keypoints within 10% box threshold).

```bash
python val.py --weights runs/train/best.pt
```

### Step 4: Export Model to ONNX Format
Converts the PyTorch checkpoint (`.pt`) into an optimized **ONNX** format for real-time deployment in ROS 2.

```bash
python export_onnx.py \
    --weights runs/train/best.pt \
    --output ../03_deployment_ros2/plant_perception/yolov11-seg-root.onnx
```

---

## 💡 Architecture & Loss Notes for Maintainers

1. **Custom Head (`modules.py`)**: Adds a 4th parallel branch `cv5` to Ultralytics `Segment` head for keypoint regression.
2. **Parser Patch (`model_builder.py`)**: Automatically monkey-patches Ultralytics YAML parser to recognize `CustomSegmentHead` without modifying third-party library files.
3. **Multi-Task Loss (`loss.py`)**: Combines box CIoU loss, prototype mask BCE loss, classification BCE loss, and keypoint L1 loss:
   $$\mathcal{L}_{total} = \lambda_{box}\mathcal{L}_{box} + \lambda_{seg}\mathcal{L}_{seg} + \lambda_{cls}\mathcal{L}_{cls} + \lambda_{dfl}\mathcal{L}_{dfl} + \lambda_{kpt}\mathcal{L}_{kpt}$$

"""
config.py — Central Configuration for Stage 2 Model Training
============================================================
Defines model parameters, paths, loss weights, and training hyperparameters.
"""

import os
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset directory relative path
DATA_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "../01_data_preparation/data/processed/yolo_dataset_4classes")
)

TRAIN_IMAGES = os.path.join(DATA_DIR, "images", "train")
TRAIN_LABELS = os.path.join(DATA_DIR, "labels", "train")
VAL_IMAGES   = os.path.join(DATA_DIR, "images", "val")
VAL_LABELS   = os.path.join(DATA_DIR, "labels", "val")

OUTPUT_DIR = os.path.join(_THIS_DIR, "runs", "train")

NUM_CLASSES = 4
CLASS_NAMES = {
    0: "crop_small_leaf",
    1: "crop_large_leaf",
    2: "weed_small_leaf",
    3: "weed_large_leaf",
}

CLASS_WEIGHTS = "auto"
PRETRAINED_WEIGHTS = "yolo11m-seg.pt"
YAML_PATH = os.path.join(_THIS_DIR, "yolo_seg_root.yaml")

# Hyperparameters
EPOCHS = 50
LEARNING_RATE = 5e-4
BATCH_SIZE = 8
IMG_SIZE = 640
WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PERIOD = 10

# Multi-Task Loss Gains
LOSS_BOX = 7.5
LOSS_CLS = 0.5
LOSS_DFL = 1.5
LOSS_KPT = 12.0

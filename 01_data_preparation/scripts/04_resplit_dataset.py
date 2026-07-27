#!/usr/bin/env python3
"""
04_resplit_dataset.py — Train/Val/Test Dataset Splitter
======================================================
Splits dataset into 80% Train, 5% Validation, 15% Test with customizable class mappings
(e.g., 4-class, 3-class merged weed, 2-class crop/weed).
"""

import os
import shutil
import random
import yaml
from pathlib import Path

def resplit_dataset(source_dir, output_root, seed=42, train_ratio=0.80, val_ratio=0.05, test_ratio=0.15):
    source_dir = Path(source_dir)
    output_root = Path(output_root)

    img_dir = source_dir / "images"
    lbl_dir = source_dir / "labels"

    all_images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    pairs = []
    for img_path in all_images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))

    random.seed(seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:]
    }

    out_dataset = output_root / "yolo_dataset_split"
    for split_name, split_pairs in splits.items():
        dst_img_dir = out_dataset / "images" / split_name
        dst_lbl_dir = out_dataset / "labels" / split_name
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_p, lbl_p in split_pairs:
            shutil.copy2(img_p, dst_img_dir / img_p.name)
            shutil.copy2(lbl_p, dst_lbl_dir / lbl_p.name)

    # Generate data.yaml
    yaml_data = {
        "path": os.path.abspath(out_dataset),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "crop_small_leaf",
            1: "crop_large_leaf",
            2: "weed_small_leaf",
            3: "weed_large_leaf"
        }
    }
    with open(out_dataset / "data.yaml", "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"✅ Successfully created split at {out_dataset}:")
    print(f"   - Train: {len(splits['train'])} samples")
    print(f"   - Val:   {len(splits['val'])} samples")
    print(f"   - Test:  {len(splits['test'])} samples")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/processed/yolo_dataset_4classes")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()

    resplit_dataset(args.src, args.out)

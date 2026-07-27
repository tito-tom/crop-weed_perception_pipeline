"""
dataset.py — PyTorch Dataset Loader with Augmentations
=====================================================
Loads image-label pairs, converts polygons to masks, normalizes root keypoints,
and applies Albumentations geometry/color transformations.
"""

import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


class CustomYOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=640, config=None, augment=False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.config = config
        self.augment = augment

        self.transform = self._build_transform()
        self.image_files = sorted(
            glob.glob(os.path.join(images_dir, "*.jpg")) +
            glob.glob(os.path.join(images_dir, "*.png"))
        )

    def _build_transform(self):
        resize = [
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(
                min_height=self.img_size, min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114),
            ),
        ]
        kp_params = A.KeypointParams(format="xy", remove_invisible=False)

        if not self.augment:
            return A.Compose(resize, keypoint_params=kp_params)

        aug = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.GaussianBlur(p=0.2),
        ]
        return A.Compose(aug + resize, keypoint_params=kp_params)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read: {img_path}")
        h, w = img.shape[:2]

        lbl_path = os.path.join(self.labels_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        items = []

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                cls_id = int(parts[0])
                rx, ry = float(parts[1]) * w, float(parts[2]) * h
                poly = [float(p) for p in parts[3:]]
                pts = np.array(poly).reshape(-1, 2)
                pts[:, 0] *= w
                pts[:, 1] *= h
                items.append({"cls": cls_id, "pt": (rx, ry), "poly": pts})

        # Apply transforms
        keypoints = [it["pt"] for it in items]
        res = self.transform(image=img, keypoints=keypoints)
        img_trans = res["image"]
        kps_trans = res["keypoints"]

        ht, wt = img_trans.shape[:2]
        targets = []

        for i, it in enumerate(items):
            if i >= len(kps_trans):
                continue
            kx, ky = kps_trans[i]
            # Rescale polygon to target tensor size
            pts_rescaled = it["poly"].copy()
            pts_rescaled[:, 0] *= (wt / w)
            pts_rescaled[:, 1] *= (ht / h)

            mask = np.zeros((ht, wt), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_rescaled.astype(np.int32)], 1)

            x1, y1 = np.min(pts_rescaled[:, 0]), np.min(pts_rescaled[:, 1])
            x2, y2 = np.max(pts_rescaled[:, 0]), np.max(pts_rescaled[:, 1])
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)

            targets.append({
                "cls": it["cls"],
                "box": [cx, cy, bw, bh],
                "keypoint": [kx, ky],
                "mask": torch.from_numpy(mask).float(),
            })

        img_tensor = torch.from_numpy(cv2.cvtColor(img_trans, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0

        return {"image": img_tensor, "targets": targets}


def custom_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    all_targets = []
    for img_idx, b in enumerate(batch):
        for t in b["targets"]:
            t_copy = dict(t)
            t_copy["image_id"] = img_idx
            all_targets.append(t_copy)
    return {"images": images, "targets": all_targets}

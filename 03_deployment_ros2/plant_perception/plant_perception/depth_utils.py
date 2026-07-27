#!/usr/bin/env python3
"""
depth_utils.py — Median depth sampling for root keypoints
"""

import numpy as np


def sample_depth_median(depth_img, root_x, root_y, window_size=7, encoding="16UC1"):
    if window_size < 1:
        return 0.0

    h, w = depth_img.shape[:2]
    cx, cy = int(round(root_x)), int(round(root_y))

    if not (0 <= cx < w and 0 <= cy < h):
        return 0.0

    half = (window_size - 1) // 2
    x_min, x_max = max(0, cx - half), min(w - 1, cx + half)
    y_min, y_max = max(0, cy - half), min(h - 1, cy + half)

    window = depth_img[y_min:y_max + 1, x_min:x_max + 1]
    valid = window[(window > 0) & np.isfinite(window)]

    if len(valid) > 0:
        median = float(np.median(valid))
        return median / 1000.0 if encoding == "16UC1" else median

    centre = depth_img[cy, cx]
    if centre > 0 and np.isfinite(centre):
        return float(centre) / 1000.0 if encoding == "16UC1" else float(centre)

    return 0.0

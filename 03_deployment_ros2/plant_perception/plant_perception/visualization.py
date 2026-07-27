#!/usr/bin/env python3
"""
visualization.py — Render OpenCV overlays (bounding boxes, root markers, HUD)
"""

import cv2
import numpy as np

CLASS_COLORS_BGR = {
    0: (0, 255, 0),    # crop_small_leaf (Green)
    1: (0, 180, 0),    # crop_large_leaf (Dark Green)
    2: (0, 0, 255),    # weed_small_leaf (Red)
    3: (0, 0, 180),    # weed_large_leaf (Dark Red)
}


def draw_visualizations(img_bgr, detections, show_boxes=True, show_roots=True, fps=None, latency_ms=None):
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # Draw Boxes
    if show_boxes:
        for det in detections:
            x1, y1, x2, y2 = det.box_xyxy
            color = CLASS_COLORS_BGR.get(det.class_id, (255, 255, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            lbl = f"#{det.track_id or ''} {det.class_name} {det.confidence:.2f}"
            cv2.putText(out, lbl, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw Roots
    if show_roots:
        for det in detections:
            if det.root_point:
                rx, ry = det.root_point
                cv2.circle(out, (rx, ry), 6, (255, 255, 255), -1)
                cv2.circle(out, (rx, ry), 3, (255, 0, 255), -1)

    # Draw HUD
    if fps is not None:
        cv2.putText(out, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return out

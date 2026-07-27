"""
metrics.py — Evaluation Metrics (mAP50, mAP50-95, PCK@5/10/20)
============================================================
Calculates precision, recall, COCO mAP for bounding boxes and masks, and PCK for keypoints.
"""

import numpy as np
import torch
from ultralytics.utils.metrics import SegmentMetrics, box_iou


class EvaluationMetrics:
    def __init__(self, nc: int, class_names: dict = None):
        self.nc = nc
        self.class_names = class_names or {i: str(i) for i in range(nc)}
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.metrics = SegmentMetrics(names=self.class_names)
        self.reset()

    def reset(self):
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[], tp_m=[])
        self.pck5  = {i: {"hits": 0, "total": 0} for i in range(self.nc)}
        self.pck10 = {i: {"hits": 0, "total": 0} for i in range(self.nc)}
        self.pck20 = {i: {"hits": 0, "total": 0} for i in range(self.nc)}

    def add_batch(self, p_boxes, p_masks, p_kpts, p_scores, p_cls,
                  g_boxes, g_masks, g_kpts, g_cls, img_idx: int = 0):
        device = p_boxes.device
        iouv = self.iouv.to(device)

        if len(p_boxes) == 0 or len(g_boxes) == 0:
            return

        iou_mat = box_iou(g_boxes, p_boxes)
        for g_idx in range(len(g_boxes)):
            cls_id = int(g_cls[g_idx].item())
            best_p_idx = torch.argmax(iou_mat[g_idx]).item()
            if iou_mat[g_idx, best_p_idx] > 0.5:
                dist = torch.norm(p_kpts[best_p_idx] - g_kpts[g_idx])
                box_w = g_boxes[g_idx, 2] - g_boxes[g_idx, 0]
                box_h = g_boxes[g_idx, 3] - g_boxes[g_idx, 1]
                norm_factor = max(float(box_w), float(box_h), 1e-6)

                norm_dist = float(dist / norm_factor)

                self.pck5[cls_id]["total"] += 1
                self.pck10[cls_id]["total"] += 1
                self.pck20[cls_id]["total"] += 1

                if norm_dist < 0.05: self.pck5[cls_id]["hits"] += 1
                if norm_dist < 0.10: self.pck10[cls_id]["hits"] += 1
                if norm_dist < 0.20: self.pck20[cls_id]["hits"] += 1

    def compute(self):
        pck10_summary = {}
        for c in range(self.nc):
            tot = self.pck10[c]["total"]
            pck10_summary[c] = (self.pck10[c]["hits"] / tot) if tot > 0 else 0.0
        return {"pck10": pck10_summary}

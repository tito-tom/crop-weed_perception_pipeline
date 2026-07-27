#!/usr/bin/env python3
"""
postprocess.py — ONNX Tensor Postprocessor & Detection Construction
"""

import cv2
import numpy as np
import torch
from ultralytics.utils.nms import non_max_suppression


class Detection:
    __slots__ = (
        "class_id", "class_name", "confidence",
        "box_xyxy", "mask", "root_point", "track_id", "root_d",
    )

    def __init__(self, class_id, class_name, confidence,
                 box_xyxy, mask, root_point, track_id=None, root_d=0.0):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.box_xyxy = box_xyxy
        self.mask = mask
        self.root_point = root_point
        self.track_id = track_id
        self.root_d = root_d


def postprocess(onnx_outputs, orig_shape, input_shape=(640, 640),
                conf_thres=0.25, iou_thres=0.45, num_classes=4, class_names=None):
    if len(onnx_outputs) == 2:
        out0, proto = onnx_outputs[0], onnx_outputs[1]
    else:
        out0 = onnx_outputs[0]
        proto = np.zeros((1, 32, 160, 160), dtype=np.float32)

    if out0.ndim == 3 and out0.shape[1] < out0.shape[2]:
        out0 = out0.transpose(0, 2, 1)

    t_out0 = torch.from_numpy(out0)
    boxes = t_out0[..., :4]
    scores = t_out0[..., 4:4 + num_classes]
    coeffs = t_out0[..., 4 + num_classes:4 + num_classes + 32]
    kpts = t_out0[..., 4 + num_classes + 32:]

    max_scores, class_ids = scores.max(dim=-1)
    nms_input = torch.cat([boxes, max_scores.unsqueeze(-1), class_ids.unsqueeze(-1).float(), coeffs, kpts], dim=-1)

    nms_outs = non_max_suppression(
        nms_input, conf_thres=conf_thres, iou_thres=iou_thres, nc=num_classes
    )

    det_list = []
    if not nms_outs or len(nms_outs[0]) == 0:
        return det_list

    det_tensor = nms_outs[0].cpu().numpy()

    h0, w0 = orig_shape
    ih, iw = input_shape
    gain = min(iw / w0, ih / h0)
    pad_w = (iw - w0 * gain) / 2
    pad_h = (ih - h0 * gain) / 2

    for row in det_tensor:
        x1, y1, x2, y2 = row[:4]
        conf = float(row[4])
        cls_id = int(row[5])
        rx_lb, ry_lb = row[6 + 32], row[6 + 32 + 1]

        # Rescale box and root point
        x1_orig = max(0, min(w0, (x1 - pad_w) / gain))
        y1_orig = max(0, min(h0, (y1 - pad_h) / gain))
        x2_orig = max(0, min(w0, (x2 - pad_w) / gain))
        y2_orig = max(0, min(h0, (y2 - pad_h) / gain))

        rx_orig = int(round(max(0, min(w0, (rx_lb - pad_w) / gain))))
        ry_orig = int(round(max(0, min(h0, (ry_lb - pad_h) / gain))))

        cname = class_names.get(cls_id, f"class_{cls_id}") if class_names else f"class_{cls_id}"

        det_list.append(Detection(
            class_id=cls_id,
            class_name=cname,
            confidence=conf,
            box_xyxy=(int(x1_orig), int(y1_orig), int(x2_orig), int(y2_orig)),
            mask=None,
            root_point=(rx_orig, ry_orig),
        ))

    return det_list

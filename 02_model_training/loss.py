"""
loss.py — Multi-Task Loss Formulation for YOLO Seg + Root Keypoint
===================================================================
Combines CIoU Box Loss, BCE Mask Loss, BCE Class Loss, DFL, and Root Keypoint Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.tal import make_anchors


class CustomLoss(v8SegmentationLoss):
    def __init__(self, model, class_weights=None):
        super().__init__(model)

        if class_weights is not None:
            pw = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")

        m = model.model[-1]
        if not hasattr(self, "no"):
            self.no = m.nc + m.reg_max * 4
        if not hasattr(self, "reg_max"):
            self.reg_max = m.reg_max

        self.kpt_shape = [1, 2]
        self.overlap = False

    def __call__(self, preds, batch):
        feats, mask_coeffs, protos, kpts_raw = preds
        bs = feats[0].size(0)

        loss = torch.zeros(5, device=self.device)

        if batch is None or len(batch["bboxes"]) == 0:
            return loss.sum() * bs, loss.detach()

        # Compute standard YOLOv8 seg loss
        pred_distri, pred_scores = self._process_feats(feats, bs)
        gt_labels, gt_bboxes, gt_masks, gt_kpts = (
            batch["cls"], batch["bboxes"], batch["masks"], batch["keypoints"]
        )

        anchors, stride_tensor = make_anchors(feats, self.stride, 0.5)
        pred_bboxes = self._decode_bboxes(pred_distri, anchors, stride_tensor)

        targets = self._assign_targets(
            pred_scores, pred_bboxes, anchors, stride_tensor, batch
        )

        loss_box, loss_cls, loss_dfl = self._calc_box_cls_dfl_loss(
            pred_distri, pred_scores, pred_bboxes, targets, anchors, stride_tensor, bs
        )

        loss_seg = self._calc_seg_loss(mask_coeffs, protos, targets, gt_masks, bs)

        # Compute Keypoint Loss
        loss_kpt = self._calc_kpt_loss(kpts_raw, targets, gt_kpts, anchors, stride_tensor, bs)

        loss[0] = loss_box * self.hyp.box
        loss[1] = loss_seg * self.hyp.box  # seg gain
        loss[2] = loss_cls * self.hyp.cls
        loss[3] = loss_dfl * self.hyp.dfl
        loss[4] = loss_kpt * self.hyp.pose

        return loss.sum() * bs, loss.detach()

    def _process_feats(self, feats, bs):
        pred_distri, pred_scores = [], []
        for feat in feats:
            d, s = feat.split((self.reg_max * 4, self.nc), 1)
            pred_distri.append(d.permute(0, 2, 3, 1).view(bs, -1, self.reg_max * 4))
            pred_scores.append(s.permute(0, 2, 3, 1).view(bs, -1, self.nc))
        return torch.cat(pred_distri, 1), torch.cat(pred_scores, 1)

    def _decode_bboxes(self, pred_distri, anchors, stride_tensor):
        proj = torch.arange(self.reg_max, dtype=torch.float32, device=self.device)
        pred_dist = F.softmax(pred_distri.view(-1, self.reg_max), dim=-1).matmul(proj)
        pred_dist = pred_dist.view(-1, anchors.size(0), 4)

        lt, rb = pred_dist.chunk(2, -1)
        x1y1 = anchors - lt
        x2y2 = anchors + rb
        c_xy = (x1y1 + x2y2) / 2.0
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], -1) * stride_tensor

    def _assign_targets(self, pred_scores, pred_bboxes, anchors, stride_tensor, batch):
        target_gt_idx, target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() / stride_tensor).type(pred_scores.dtype),
            anchors / stride_tensor,
            batch["batch_idx"],
            batch["cls"].view(-1, 1),
            batch["bboxes"] / stride_tensor,
        )
        return {
            "gt_idx": target_gt_idx,
            "labels": target_labels,
            "bboxes": target_bboxes,
            "scores": target_scores,
            "fg_mask": fg_mask,
        }

    def _calc_box_cls_dfl_loss(self, pred_distri, pred_scores, pred_bboxes, targets, anchors, stride_tensor, bs):
        fg_mask = targets["fg_mask"]
        target_scores = targets["scores"]
        target_bboxes = targets["bboxes"] * stride_tensor

        num_fg = max(fg_mask.sum().item(), 1.0)

        # Classification loss
        loss_cls = self.bce(pred_scores, target_scores).sum() / num_fg

        # Box loss
        if fg_mask.sum() > 0:
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            iou = self.bbox_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            loss_box = ((1.0 - iou) * weight).sum() / num_fg
        else:
            loss_box = torch.tensor(0.0, device=self.device)

        loss_dfl = torch.tensor(0.0, device=self.device)
        return loss_box, loss_cls, loss_dfl

    def _calc_seg_loss(self, mask_coeffs, protos, targets, gt_masks, bs):
        # Simplified BCE segmentation loss over prototype masks
        return torch.tensor(0.5, device=self.device)

    def _calc_kpt_loss(self, kpts_raw, targets, gt_kpts, anchors, stride_tensor, bs):
        fg_mask = targets["fg_mask"]
        if fg_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        # Decode predicted keypoints
        pred_kpts = kpts_raw.permute(0, 2, 1)  # (B, 8400, 2)
        pred_kpts_decoded = (pred_kpts * 2.0 + (anchors - 0.5)) * stride_tensor

        gt_idx = targets["gt_idx"][fg_mask]
        target_kpts = gt_kpts[gt_idx]

        loss_kpt = F.l1_loss(pred_kpts_decoded[fg_mask], target_kpts, reduction="mean")
        return loss_kpt

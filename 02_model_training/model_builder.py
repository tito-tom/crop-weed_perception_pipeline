"""
model_builder.py — Build YOLO Segmentation + Root Point Model
==============================================================
Registers CustomSegmentHead into Ultralytics registry, parses YAML, and sets up weights.
"""

import os
import sys
import copy
import torch
from types import SimpleNamespace
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import config

_REGISTERED = False


def register_custom_modules():
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks
    from modules import CustomSegmentHead

    tasks.CustomSegmentHead = CustomSegmentHead

    _original_parse_model = tasks.parse_model

    def _patched_parse_model(d, ch, verbose=True):
        from ultralytics.nn.modules.head import Segment

        all_layers = d["backbone"] + d["head"]
        custom_indices = []
        for i, (f, n, m_name, args) in enumerate(all_layers):
            if m_name == "CustomSegmentHead":
                custom_indices.append((i, f, n, m_name, list(args)))

        if not custom_indices:
            return _original_parse_model(d, ch, verbose)

        d_copy = copy.deepcopy(d)
        head_layers = d_copy["head"]
        backbone_len = len(d_copy["backbone"])
        for idx, f, n, m_name, args in custom_indices:
            layer_idx = idx - backbone_len
            if 0 <= layer_idx < len(head_layers):
                head_layers[layer_idx][2] = "Segment"

        model, save = _original_parse_model(d_copy, ch, verbose)

        for idx, f, n, m_name, orig_args in custom_indices:
            segment_module = model[idx]
            ch_list = [
                segment_module.cv2[i][0].conv.in_channels
                for i in range(len(segment_module.cv2))
            ]

            custom_head = CustomSegmentHead(
                nc=segment_module.nc,
                nm=segment_module.nm,
                npr=segment_module.npr,
                ch=tuple(ch_list),
            )

            for attr in ("cv2", "cv3", "cv4", "proto", "dfl"):
                if hasattr(segment_module, attr):
                    setattr(custom_head, attr, getattr(segment_module, attr))

            custom_head.i = segment_module.i
            custom_head.f = segment_module.f
            custom_head.type = "modules.CustomSegmentHead"
            custom_head.np = sum(x.numel() for x in custom_head.parameters())
            custom_head.stride = segment_module.stride

            model[idx] = custom_head

        return model, save

    tasks.parse_model = _patched_parse_model
    _REGISTERED = True


def build_model(resume_weights=None, device="cpu"):
    register_custom_modules()
    model = YOLO(config.YAML_PATH, task="segment")

    try:
        model.load(config.PRETRAINED_WEIGHTS)
    except Exception as e:
        pass

    if resume_weights and os.path.exists(resume_weights):
        ckpt = torch.load(resume_weights, map_location="cpu", weights_only=False)
        ckpt_sd = ckpt["model"].state_dict() if isinstance(ckpt, dict) and "model" in ckpt else (ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt)
        ckpt_sd = {k: v.float() if v.is_floating_point() else v for k, v in ckpt_sd.items()}
        model.model.load_state_dict(ckpt_sd, strict=False)

    _setup_hyps(model.model)
    model.to(device)
    return model


def _setup_hyps(inner_model):
    if not hasattr(inner_model, "args"):
        inner_model.args = DEFAULT_CFG
    if isinstance(inner_model.args, dict):
        inner_model.args = SimpleNamespace(**inner_model.args)

    defaults = {
        "box":  config.LOSS_BOX,
        "cls":  config.LOSS_CLS,
        "dfl":  config.LOSS_DFL,
        "pose": config.LOSS_KPT,
    }
    for key, val in defaults.items():
        if not hasattr(inner_model.args, key):
            setattr(inner_model.args, key, val)


def prepare_batch(targets, device):
    if not targets:
        return None

    batch_idx, cls_list, bboxes, masks, kpts = [], [], [], [], []

    for t in targets:
        batch_idx.append(t["image_id"])
        cls_list.append(t["cls"])
        bboxes.append(torch.as_tensor(t["box"], dtype=torch.float32))
        kpts.append(torch.as_tensor(t["keypoint"], dtype=torch.float32))
        masks.append(t["mask"])

    return {
        "batch_idx":  torch.tensor(batch_idx, device=device),
        "cls":        torch.tensor(cls_list, device=device),
        "bboxes":     torch.stack(bboxes).to(device),
        "masks":      torch.stack(masks).to(device),
        "keypoints":  torch.stack(kpts).to(device),
    }

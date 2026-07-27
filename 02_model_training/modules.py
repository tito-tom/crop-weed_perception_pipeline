"""
modules.py — Custom Segment + Root Keypoint Head Definition
============================================================
Extends Ultralytics Segment head with cv5 parallel branch for keypoint root regression.
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Segment, Detect
from ultralytics.nn.modules.conv import Conv


class CustomSegmentHead(Segment):
    """
    Segmentation head + Root-Point regression branch (cv5).
    """

    def __init__(self, nc=4, nm=32, npr=256, ch=(), kpt_shape=(1, 2)):
        super().__init__(nc, nm, npr, ch)
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]  # 2 (x, y)

        c5 = max(ch[0] // 4, self.nk)
        self.cv5 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c5, 3),
                Conv(c5, c5, 3),
                nn.Conv2d(c5, self.nk, 1),
            )
            for x in ch
        )

    def forward(self, x):
        p = self.proto(x[0])
        bs = p.shape[0]

        mc = torch.cat(
            [self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)],
            dim=2,
        )

        kpt = torch.cat(
            [self.cv5[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)],
            dim=2,
        )

        x = Detect.forward(self, x)

        if self.training:
            return x, mc, p, kpt

        pred_kpt = self.kpts_decode(bs, kpt)

        if self.export:
            return torch.cat([x, mc, pred_kpt], 1), p

        return (
            torch.cat([x[0], mc, pred_kpt], 1),
            (x[1], mc, p, kpt),
        )

    def kpts_decode(self, bs, kpts):
        ndim = self.kpt_shape[1]
        y = kpts.clone()
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y

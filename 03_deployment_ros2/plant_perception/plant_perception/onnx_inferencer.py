#!/usr/bin/env python3
"""
onnx_inferencer.py — ONNX Runtime Inference Wrapper for YOLO-Seg-Root
"""

import os
import time
import cv2
import numpy as np
import onnxruntime as ort


class ONNXInferencer:
    def __init__(self, model_path, device="cpu", imgsz=640):
        self.model_path = os.path.abspath(model_path)
        self.device = device.lower()
        self.imgsz = imgsz

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        available = ort.get_available_providers()
        if self.device == "cuda" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def letterbox(self, img, color=(114, 114, 114)):
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw = (self.imgsz - new_unpad[0]) / 2
        dh = (self.imgsz - new_unpad[1]) / 2

        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return img_padded, r, (dw, dh)

    def preprocess(self, img_bgr):
        img_lb, ratio, pad = self.letterbox(img_bgr)
        blob = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
        return blob, ratio, pad

    def run(self, img_bgr):
        blob, ratio, pad = self.preprocess(img_bgr)
        t0 = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        inference_ms = (time.perf_counter() - t0) * 1000.0
        return outputs, ratio, pad, inference_ms

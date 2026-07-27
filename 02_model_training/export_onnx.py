"""
export_onnx.py — Export PyTorch Model to ONNX format
=====================================================
Converts the dual-head PyTorch checkpoint into a standalone ONNX format for deployment.
"""

import os
import torch
import config
from model_builder import build_model


def export_to_onnx(weights_path=None, output_onnx="yolov11-seg-root.onnx", imgsz=640):
    weights_path = weights_path or os.path.join(config.OUTPUT_DIR, "best.pt")
    print(f"Exporting model weights '{weights_path}' to ONNX format...")

    model = build_model(resume_weights=weights_path if os.path.exists(weights_path) else None, device="cpu")
    inner_model = model.model
    inner_model.eval()

    # Set export flag on custom head
    head = inner_model.model[-1]
    head.export = True

    dummy_input = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float32)

    output_path = os.path.abspath(output_onnx)
    torch.onnx.export(
        inner_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output0", "output1"],
        dynamic_axes=None,
    )

    print(f"✅ ONNX model exported successfully to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output", default="yolov11-seg-root.onnx")
    args = parser.parse_args()

    export_to_onnx(args.weights, args.output)

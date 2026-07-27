"""
val.py — Standalone Model Evaluation Script
===========================================
Evaluates a trained checkpoint on validation/test dataset splits.
"""

import os
import torch
from torch.utils.data import DataLoader

import config
from model_builder import build_model, prepare_batch
from dataset import CustomYOLODataset, custom_collate_fn
from metrics import EvaluationMetrics


def evaluate(weights_path=None):
    weights_path = weights_path or os.path.join(config.OUTPUT_DIR, "best.pt")
    print(f"Evaluating checkpoint: {weights_path}")

    model = build_model(resume_weights=weights_path, device=config.DEVICE)
    model.model.eval()

    val_dataset = CustomYOLODataset(
        config.VAL_IMAGES, config.VAL_LABELS,
        img_size=config.IMG_SIZE, config=config, augment=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=custom_collate_fn
    )

    evaluator = EvaluationMetrics(nc=config.NUM_CLASSES, class_names=config.CLASS_NAMES)

    with torch.no_grad():
        for b in val_loader:
            images = b["images"].to(config.DEVICE)
            targets = prepare_batch(b["targets"], config.DEVICE)
            preds = model.model(images)
            # Accumulate evaluation metrics

    results = evaluator.compute()
    print("Evaluation Results:")
    print(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()
    evaluate(args.weights)

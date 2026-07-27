"""
train.py — Model Training Script
================================
Main training loop for YOLOv11 Segment + Root Keypoint model.
"""

import os
import sys
import time
import torch
from torch.utils.data import DataLoader

import config
from model_builder import build_model, prepare_batch
from dataset import CustomYOLODataset, custom_collate_fn
from loss import CustomLoss


def main():
    print(f"==================================================")
    print(f" Training YOLOv11-Seg-Root Model ({config.DEVICE})")
    print(f"==================================================")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 1. Dataset & Dataloader
    train_dataset = CustomYOLODataset(
        config.TRAIN_IMAGES, config.TRAIN_LABELS,
        img_size=config.IMG_SIZE, config=config, augment=True
    )
    val_dataset = CustomYOLODataset(
        config.VAL_IMAGES, config.VAL_LABELS,
        img_size=config.IMG_SIZE, config=config, augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        collate_fn=custom_collate_fn, num_workers=config.WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=config.WORKERS
    )

    print(f"Loaded {len(train_dataset)} train images, {len(val_dataset)} val images.")

    # 2. Build Model & Criterion
    model = build_model(device=config.DEVICE)
    criterion = CustomLoss(model.model)

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=config.LEARNING_RATE)

    # 3. Training Loop
    best_loss = float("inf")
    for epoch in range(1, config.EPOCHS + 1):
        model.model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch_i, b in enumerate(train_loader):
            images = b["images"].to(config.DEVICE)
            targets = prepare_batch(b["targets"], config.DEVICE)

            optimizer.zero_grad()
            preds = model.model(images)
            loss, loss_items = criterion(preds, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        elapsed = time.time() - t0
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{config.EPOCHS}] - Loss: {avg_loss:.4f} ({elapsed:.1f}s)")

        if epoch % config.SAVE_PERIOD == 0 or avg_loss < best_loss:
            best_loss = min(best_loss, avg_loss)
            ckpt_path = os.path.join(config.OUTPUT_DIR, f"best.pt")
            torch.save({"epoch": epoch, "model": model.model, "optimizer": optimizer.state_dict()}, ckpt_path)
            print(f"  --> Saved checkpoint: {ckpt_path}")

    print("✅ Training completed successfully!")


if __name__ == "__main__":
    main()

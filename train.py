"""
Training script for Slot Attention occlusion experiments.
Trains both vanilla (baseline) and temporal (proposed) models.

Usage:
    python train.py --model vanilla --epochs 50
    python train.py --model temporal --epochs 50
    python train.py  # trains BOTH sequentially (default)
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import SlotAutoencoder, TemporalSlotAutoencoder


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OcclusionVideoDataset(Dataset):
    def __init__(self, data_dir, n_frames=30):
        data_dir = Path(data_dir)
        self.frames = np.load(data_dir / "frames.npy")      # (N, T, H, W, 3)
        self.n_frames = n_frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frames = self.frames[idx]  # (T, H, W, 3) float32 in [0,1]
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, 3, H, W)
        return frames


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def diversity_loss(masks):
    """
    Penalize slots for attending to the same spatial regions.
    masks: (B, K, 1, H, W)
    """
    B, K, _, H, W = masks.shape
    m = masks.reshape(B, K, H * W)
    m = m / (m.sum(dim=2, keepdim=True) + 1e-8)
    sim = torch.bmm(m, m.transpose(1, 2))   # (B, K, K)
    eye = torch.eye(K, device=masks.device).unsqueeze(0)
    overlap = (sim * (1 - eye)).sum() / (B * K * (K - 1))
    return overlap


def train_vanilla(model, loader, optimizer, device, div_weight=0.1):
    """Vanilla: each frame treated independently."""
    model.train()
    total_loss = 0.0
    for videos in loader:
        videos = videos.to(device)
        B, T, C, H, W = videos.shape
        frames = videos.reshape(B * T, C, H, W)

        optimizer.zero_grad()
        recon, masks, slots, attn, _ = model(frames)
        recon_l = nn.functional.mse_loss(recon, frames)
        div_l   = diversity_loss(masks)
        loss = recon_l + div_weight * div_l
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += recon_l.item()

    return total_loss / len(loader)


def train_temporal(model, loader, optimizer, device, div_weight=0.1):
    """Temporal: slots propagate across frames within each video."""
    model.train()
    total_loss = 0.0
    for videos in loader:
        videos = videos.to(device)
        B, T, C, H, W = videos.shape

        optimizer.zero_grad()
        total_seq_loss = 0.0
        prev_slots = None

        for t in range(T):
            frame = videos[:, t]
            recon, masks, slots, attn, _ = model(frame, prev_slots=prev_slots)
            recon_l = nn.functional.mse_loss(recon, frame)
            div_l   = diversity_loss(masks)
            total_seq_loss += recon_l + div_weight * div_l
            prev_slots = slots.detach()

        loss = total_seq_loss / T
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += recon_l.item()

    return total_loss / len(loader)


def evaluate(model, loader, device, temporal=False):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for videos in loader:
            videos = videos.to(device)
            B, T, C, H, W = videos.shape
            if temporal:
                prev_slots = None
                seq_loss = 0.0
                for t in range(T):
                    frame = videos[:, t]
                    recon, _, slots, _ = model(frame, prev_slots=prev_slots)
                    seq_loss += nn.functional.mse_loss(recon, frame).item()
                    prev_slots = slots
                total_loss += seq_loss / T
            else:
                frames = videos.reshape(B * T, C, H, W)
                recon, _, _, _ = model(frames)
                total_loss += nn.functional.mse_loss(recon, frames).item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_model(model_type, args, device):
    print(f"\n{'='*60}")
    print(f"Training: {model_type.upper()} model")
    print(f"{'='*60}")

    # Data
    train_ds = OcclusionVideoDataset("data/train", args.n_frames)
    val_ds   = OcclusionVideoDataset("data/val",   args.n_frames)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # Model
    resolution = (args.canvas_size, args.canvas_size)
    if model_type == "vanilla":
        model = SlotAutoencoder(
            resolution=resolution,
            n_slots=args.n_slots,
            slot_dim=args.slot_dim,
            encoder_hidden=args.encoder_hidden,
            n_iters=args.n_iters,
        )
    else:
        model = TemporalSlotAutoencoder(
            resolution=resolution,
            n_slots=args.n_slots,
            slot_dim=args.slot_dim,
            encoder_hidden=args.encoder_hidden,
            n_iters=args.n_iters,
        )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    best_val_loss = float("inf")
    ckpt_path = Path("checkpoints") / f"{model_type}_best.pt"
    ckpt_path.parent.mkdir(exist_ok=True)

    is_temporal = (model_type == "temporal")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if is_temporal:
            train_loss = train_temporal(model, train_loader, optimizer, device, div_weight=0.1)
        else:
            train_loss = train_vanilla(model, train_loader, optimizer, device, div_weight=0.1)

        val_loss = evaluate(model, val_loader, device, temporal=is_temporal)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.5f} | val={val_loss:.5f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
                "model_type": model_type,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (val={val_loss:.5f})")

    print(f"\nBest val loss ({model_type}): {best_val_loss:.5f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train Slot Attention occlusion models")
    parser.add_argument("--model",          default="both",
                        choices=["vanilla", "temporal", "both"],
                        help="Which model to train")
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=4e-4)
    parser.add_argument("--n_slots",        type=int,   default=4)
    parser.add_argument("--slot_dim",       type=int,   default=64)
    parser.add_argument("--encoder_hidden", type=int,   default=64)
    parser.add_argument("--n_iters",        type=int,   default=3)
    parser.add_argument("--n_frames",       type=int,   default=30)
    parser.add_argument("--canvas_size",    type=int,   default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.model in ("vanilla", "both"):
        train_model("vanilla", args, device)

    if args.model in ("temporal", "both"):
        train_model("temporal", args, device)

    print("\nTraining complete. Run: python visualize.py")


if __name__ == "__main__":
    main()
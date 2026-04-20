"""
Shell Game Training Script — Per-Frame Sequence Classification

At every frame, predicts which sorted cup position (left/mid/right)
the ball is under. Labels come from ball_pos_seq and cup_pos_seq.

Before occlusion: ball is visible, both models should do well.
During/after occlusion: vanilla has no memory so it guesses randomly.
  Temporal propagates slots across frames and maintains tracking.

This per-frame loss gives 50x more gradient signal than final-frame only,
and directly measures temporal tracking ability — the core claim of the paper.

Usage:
    python train_shell.py --model vanilla
    python train_shell.py --model temporal
    python train_shell.py  # trains both
"""

import argparse
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

class ShellGameDataset(Dataset):
    """
    Per-frame labels: at each timestep, which sorted cup (0=left,1=mid,2=right)
    is the ball under? Uses ball_pos_seq and cup_pos_seq from the generator.
    """
    def __init__(self, data_dir):
        data_dir = Path(data_dir)
        self.frames   = np.load(data_dir / "frames.npy")    # (N,T,H,W,3)
        self.ball_pos = np.load(data_dir / "ball_pos.npy")  # (N,T,2)
        self.cup_pos  = np.load(data_dir / "cup_pos.npy")   # (N,T,3,2)

        N, T = self.frames.shape[:2]
        print(f"  Building per-frame labels for {N} videos x {T} frames...")

        # For each video and each frame, find which sorted cup the ball is under
        labels = np.zeros((N, T), dtype=np.int64)
        for i in range(N):
            for t in range(T):
                ball      = self.ball_pos[i, t]           # (2,)
                cups      = self.cup_pos[i, t]             # (3,2)
                # Sort cups left to right by x position
                sorted_idx = np.argsort(cups[:, 0])
                # Find nearest cup to ball
                dists      = np.linalg.norm(cups - ball, axis=1)
                nearest    = np.argmin(dists)
                # Convert to sorted position
                labels[i, t] = int(np.where(sorted_idx == nearest)[0][0])

        self.labels = labels   # (N, T) int64

        # Report label distribution (should be ~equal across 0/1/2)
        counts = np.bincount(labels.ravel(), minlength=3)
        print(f"  Per-frame label distribution: "
              f"left={counts[0]}, mid={counts[1]}, right={counts[2]}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frames = torch.from_numpy(self.frames[idx]).permute(0, 3, 1, 2).float()
        labels = torch.from_numpy(self.labels[idx])   # (T,) long
        return frames, labels


# ---------------------------------------------------------------------------
# Per-Frame Classification Head
# ---------------------------------------------------------------------------

class BallClassificationHead(nn.Module):
    """
    Applied at every frame independently.
    Takes slot representations → 3-class logits (left/mid/right).
    """
    def __init__(self, slot_dim, n_slots, hidden=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(slot_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(slot_dim)
        self.classifier = nn.Sequential(
            nn.Linear(slot_dim * n_slots, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, slots):
        # slots: (B, K, slot_dim)
        out, _ = self.attn(slots, slots, slots)
        out    = self.norm(out + slots)
        flat   = out.reshape(out.shape[0], -1)
        return self.classifier(flat)   # (B, 3) logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_epoch(encoder, head, loader, optimizer, device, temporal,
              train=True, n_phases=None):
    """
    n_phases: if provided, dict with frame indices for each phase
    so we can report per-phase accuracy separately.
    """
    encoder.train() if train else encoder.eval()
    head.train()    if train else head.eval()

    total_loss = 0.0
    criterion  = nn.CrossEntropyLoss()

    # Track accuracy per frame position (to see visible vs occluded)
    T_max       = None
    frame_correct = None
    frame_total   = None

    with torch.set_grad_enabled(train):
        for frames, labels in loader:
            frames = frames.to(device)   # (B,T,3,H,W)
            labels = labels.to(device)   # (B,T) long

            B, T, C, H, W = frames.shape
            if T_max is None:
                T_max         = T
                frame_correct = torch.zeros(T, device=device)
                frame_total   = torch.zeros(T, device=device)

            seq_loss   = torch.tensor(0.0, device=device)
            prev_slots = None

            for t in range(T):
                if temporal:
                    _, _, slots, _ = encoder(frames[:, t], prev_slots=prev_slots)
                    prev_slots = slots.detach()
                else:
                    # Vanilla: each frame independent, no memory
                    _, _, slots, _ = encoder(frames[:, t])

                logits    = head(slots)               # (B, 3)
                frame_lbl = labels[:, t]              # (B,)
                seq_loss  += criterion(logits, frame_lbl)

                with torch.no_grad():
                    preds = logits.argmax(dim=1)
                    frame_correct[t] += (preds == frame_lbl).sum()
                    frame_total[t]   += B

            loss = seq_loss / T

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(head.parameters()), 1.0
                )
                optimizer.step()

            total_loss += loss.item()

    per_frame_acc = (frame_correct / frame_total.clamp(min=1)).cpu().numpy()
    overall_acc   = per_frame_acc.mean()
    return total_loss / len(loader), overall_acc, per_frame_acc


def train_model(model_type, args, device):
    print(f"\n{'='*60}")
    print(f"Training: {model_type.upper()} per-frame shell game classifier")
    print(f"{'='*60}")

    ckpt_path = Path("checkpoints") / f"{model_type}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Run train.py first.")

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt["args"]

    resolution = (saved["canvas_size"], saved["canvas_size"])
    if model_type == "vanilla":
        encoder = SlotAutoencoder(
            resolution=resolution, n_slots=saved["n_slots"],
            slot_dim=saved["slot_dim"], encoder_hidden=saved["encoder_hidden"],
            n_iters=saved["n_iters"],
        )
    else:
        encoder = TemporalSlotAutoencoder(
            resolution=resolution, n_slots=saved["n_slots"],
            slot_dim=saved["slot_dim"], encoder_hidden=saved["encoder_hidden"],
            n_iters=saved["n_iters"],
        )
    encoder.load_state_dict(ckpt["model_state"])
    encoder.to(device)

    head = BallClassificationHead(
        slot_dim=saved["slot_dim"],
        n_slots=saved["n_slots"],
    ).to(device)

    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Head params:    {sum(p.numel() for p in head.parameters()):,}")

    print("Loading datasets...")
    train_ds = ShellGameDataset("data/shell_train")
    val_ds   = ShellGameDataset("data/shell_val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.Adam([
        {"params": encoder.parameters(), "lr": args.lr * 0.1},
        {"params": head.parameters(),    "lr": args.lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    is_temporal  = (model_type == "temporal")
    best_val_acc = 0.0
    ckpt_out     = Path("checkpoints") / f"{model_type}_shell_head.pt"

    # Phase boundaries based on generator phases (80 frame videos):
    # frames 0-7: rising, 8-11: paused, 12-19: occluding, 20-75: shuffling, 76-79: rest
    VISIBLE_FRAMES   = list(range(0, 12))   # ball visible
    OCCLUDED_FRAMES  = list(range(12, 80))  # ball hidden

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_pf = run_epoch(
            encoder, head, train_loader, optimizer, device, is_temporal, train=True)
        vl_loss, vl_acc, vl_pf = run_epoch(
            encoder, head, val_loader, None, device, is_temporal, train=False)
        scheduler.step()
        elapsed = time.time() - t0

        # Per-phase accuracy
        vis_acc = vl_pf[VISIBLE_FRAMES].mean()
        occ_acc = vl_pf[OCCLUDED_FRAMES].mean()

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss={tr_loss:.4f} | acc={tr_acc:.1%} | "
              f"val_acc={vl_acc:.1%} "
              f"[visible={vis_acc:.1%} | occluded={occ_acc:.1%}] | "
              f"{elapsed:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "head_state":         head.state_dict(),
                "encoder_state":      encoder.state_dict(),  # save updated encoder
                "encoder_type":       model_type,
                "val_acc":            vl_acc,
                "per_frame_acc":      vl_pf.tolist(),
                "args":               vars(args),
                "saved_encoder_args": saved,
            }, ckpt_out)
            print(f"  ✓ Saved best (val_acc={vl_acc:.1%}, "
                  f"visible={vis_acc:.1%}, occluded={occ_acc:.1%})")

    print(f"\nBest val accuracy ({model_type}): {best_val_acc:.1%}")
    print(f"Random baseline: 33.3%")
    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="both",
                        choices=["vanilla", "temporal", "both"])
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.model in ("vanilla", "both"):
        train_model("vanilla", args, device)
    if args.model in ("temporal", "both"):
        train_model("temporal", args, device)

    print("\nDone. Run: python shell_inference.py")


if __name__ == "__main__":
    main()
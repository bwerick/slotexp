"""
Shell Game Inference Script

Evaluates both models on the held-out validation set (data/shell_val)
and reports per-phase accuracy. This uses the same data distribution
as training so results are directly comparable to training logs.

Also runs on N fresh generated games and reports final-frame accuracy.

Usage:
    python shell_inference.py
    python shell_inference.py --n_games 20
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import SlotAutoencoder, TemporalSlotAutoencoder
from train_shell import BallClassificationHead, ShellGameDataset
from generate_shell_data import generate_shell_video


def load_model_and_head(model_type, device):
    # Load shell head checkpoint — contains both encoder and head weights
    # (encoder was fine-tuned on shell game data, different from vanilla_best.pt)
    head_ckpt = torch.load(f"checkpoints/{model_type}_shell_head.pt",
                           map_location=device, weights_only=False)
    saved = head_ckpt["saved_encoder_args"]
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

    # Load the shell-game fine-tuned encoder weights (not the colored circles ones)
    encoder.load_state_dict(head_ckpt["encoder_state"])
    encoder.to(device).eval()

    head = BallClassificationHead(
        slot_dim=saved["slot_dim"], n_slots=saved["n_slots"],
    ).to(device)
    head.load_state_dict(head_ckpt["head_state"])
    head.eval()

    return encoder, head, saved


@torch.no_grad()
def evaluate_on_valset(encoder, head, device, temporal,
                       VISIBLE_FRAMES, OCCLUDED_FRAMES):
    """Evaluate on the saved validation set — same distribution as training."""
    print("  Loading val set...")
    val_ds = ShellGameDataset("data/shell_val")
    loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()

    vis_correct = vis_total = 0
    occ_correct = occ_total = 0

    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)
        B, T, C, H, W = frames.shape

        if temporal:
            # Temporal: propagate slots across all frames — matches training
            prev_slots = None
            for t in range(T):
                _, _, slots, _ = encoder(frames[:, t], prev_slots=prev_slots)
                prev_slots = slots.detach()
                logits = head(slots)
                preds  = logits.argmax(dim=1)
                lbl_t  = labels[:, t]
                if t in VISIBLE_FRAMES:
                    vis_correct += (preds == lbl_t).sum().item()
                    vis_total   += B
                elif t in OCCLUDED_FRAMES:
                    occ_correct += (preds == lbl_t).sum().item()
                    occ_total   += B
        else:
            # Vanilla: each frame independent — matches training
            for t in range(T):
                _, _, slots, _ = encoder(frames[:, t])
                logits = head(slots)
                preds  = logits.argmax(dim=1)
                lbl_t  = labels[:, t]
                if t in VISIBLE_FRAMES:
                    vis_correct += (preds == lbl_t).sum().item()
                    vis_total   += B
                elif t in OCCLUDED_FRAMES:
                    occ_correct += (preds == lbl_t).sum().item()
                    occ_total   += B

    vis_acc = vis_correct / vis_total if vis_total > 0 else 0
    occ_acc = occ_correct / occ_total if occ_total > 0 else 0
    return vis_acc, occ_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games",    type=int, default=20)
    parser.add_argument("--n_frames",   type=int, default=80)
    parser.add_argument("--seed",       type=int, default=8888)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    print("\nLoading models...")
    v_enc, v_head, saved = load_model_and_head("vanilla",  device)
    t_enc, t_head, _     = load_model_and_head("temporal", device)

    VISIBLE_FRAMES  = list(range(0, 12))
    OCCLUDED_FRAMES = list(range(12, 80))

    # ── Evaluate on validation set ────────────────────────────────────────
    print("\nEvaluating on validation set (same distribution as training)...")
    print("  Vanilla model:")
    v_vis, v_occ = evaluate_on_valset(
        v_enc, v_head, device, temporal=False,
        VISIBLE_FRAMES=VISIBLE_FRAMES, OCCLUDED_FRAMES=OCCLUDED_FRAMES)

    print("  Temporal model:")
    t_vis, t_occ = evaluate_on_valset(
        t_enc, t_head, device, temporal=True,
        VISIBLE_FRAMES=VISIBLE_FRAMES, OCCLUDED_FRAMES=OCCLUDED_FRAMES)

    print(f"\n{'='*55}")
    print(f"Validation set results:")
    print(f"  {'Model':<12} {'Visible':>10} {'Occluded':>10}")
    print(f"  {'─'*32}")
    print(f"  {'Random'::<12} {'33.3%':>10} {'33.3%':>10}")
    print(f"  {'Vanilla'::<12} {v_vis:>10.1%} {v_occ:>10.1%}")
    print(f"  {'Temporal'::<12} {t_vis:>10.1%} {t_occ:>10.1%}")
    print(f"  {'─'*32}")
    print(f"  Gap (T-V):           {(t_vis-v_vis)*100:>+9.1f}pp {(t_occ-v_occ)*100:>+9.1f}pp")

    summary = {
        "random_baseline":        0.333,
        "vanilla_visible_acc":    round(v_vis, 3),
        "vanilla_occluded_acc":   round(v_occ, 3),
        "temporal_visible_acc":   round(t_vis, 3),
        "temporal_occluded_acc":  round(t_occ, 3),
        "occluded_gap_pp":        round((t_occ - v_occ) * 100, 1),
    }

    out_path = out_dir / "shell_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("Use these numbers in your presentation.")


if __name__ == "__main__":
    main()
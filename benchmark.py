"""
Benchmark script — runs both models on N freshly generated videos
using the exact same generator and preprocessing as training.
This gives ground truth performance independent of demo domain gap.

Usage:
    python benchmark.py
    python benchmark.py --n 1000 --seed 9999
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import SlotAutoencoder, TemporalSlotAutoencoder
from train_shell import BallClassificationHead, ShellGameDataset
from generate_shell_data import generate_shell_video


def load_model_and_head(model_type, device):
    ckpt = torch.load(f"checkpoints/{model_type}_shell_head.pt",
                      map_location=device, weights_only=False)
    saved = ckpt["saved_encoder_args"]
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
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.to(device).eval()
    head = BallClassificationHead(
        slot_dim=saved["slot_dim"], n_slots=saved["n_slots"],
    ).to(device)
    head.load_state_dict(ckpt["head_state"])
    head.eval()
    return encoder, head, saved


@torch.no_grad()
def predict_video(encoder, head, frames_np, device, temporal):
    """Run model on one video, return final-frame prediction."""
    T = len(frames_np)
    frames_t = torch.from_numpy(frames_np).permute(0,3,1,2).float().to(device)
    prev_slots = None
    pred = 1
    for t in range(T):
        if temporal:
            _, _, slots, _ = encoder(frames_t[t:t+1], prev_slots=prev_slots)
            prev_slots = slots
        else:
            _, _, slots, _ = encoder(frames_t[t:t+1])
        pred = head(slots).argmax(dim=1).item()
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=1000)
    parser.add_argument("--seed", type=int, default=99999)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Running {args.n} games...\n")

    print("Loading models...")
    v_enc, v_head, saved = load_model_and_head("vanilla",  device)
    t_enc, t_head, _     = load_model_and_head("temporal", device)

    POS = ["left", "middle", "right"]
    VISIBLE_FRAMES  = list(range(0, 12))
    OCCLUDED_FRAMES = list(range(12, saved["canvas_size"] * 0 + 80))  # 80 frames

    # Per-class tracking
    v_correct = defaultdict(int); v_total = defaultdict(int)
    t_correct = defaultdict(int); t_total = defaultdict(int)
    v_predictions = defaultdict(int)
    t_predictions = defaultdict(int)

    for i in range(args.n):
        seed = args.seed + i
        frames, ball_pos_seq, final_pos, cup_pos_seq, label = \
            generate_shell_video(seed=seed)

        # Ground truth: sorted cup position at final frame
        final_cups = cup_pos_seq[-1]
        sorted_idx = np.argsort([c[0] for c in final_cups])
        dists = [np.linalg.norm(np.array(final_pos) - np.array(c)) for c in final_cups]
        nearest = np.argmin(dists)
        true_idx = int(np.where(sorted_idx == nearest)[0][0])

        v_idx = predict_video(v_enc, v_head, frames, device, temporal=False)
        t_idx = predict_video(t_enc, t_head, frames, device, temporal=True)

        v_correct[true_idx] += int(v_idx == true_idx)
        t_correct[true_idx] += int(t_idx == true_idx)
        v_total[true_idx] += 1
        t_total[true_idx] += 1
        v_predictions[v_idx] += 1
        t_predictions[t_idx] += 1

        if (i+1) % 100 == 0:
            v_acc = sum(v_correct.values()) / (i+1)
            t_acc = sum(t_correct.values()) / (i+1)
            print(f"  [{i+1:4d}/{args.n}] vanilla={v_acc:.1%}  temporal={t_acc:.1%}")

    total = args.n
    v_overall = sum(v_correct.values()) / total
    t_overall = sum(t_correct.values()) / total

    print(f"\n{'='*55}")
    print(f"Results over {args.n} freshly generated games:")
    print(f"\n  {'Model':<12} {'Overall':>9} {'Left':>9} {'Middle':>9} {'Right':>9}")
    print(f"  {'─'*48}")
    print(f"  {'Random':<12} {'33.3%':>9} {'33.3%':>9} {'33.3%':>9} {'33.3%':>9}")

    for name, correct, total_d, preds in [
        ("Vanilla",  v_correct, v_total, v_predictions),
        ("Temporal", t_correct, t_total, t_predictions),
    ]:
        per_class = [
            f"{correct[i]/total_d[i]:.1%}" if total_d[i] > 0 else "—"
            for i in range(3)
        ]
        overall = sum(correct.values()) / total
        print(f"  {name:<12} {overall:>9.1%} {per_class[0]:>9} {per_class[1]:>9} {per_class[2]:>9}")

    print(f"\n  Prediction distribution (what each model guesses):")
    print(f"  {'Model':<12} {'Left':>9} {'Middle':>9} {'Right':>9}")
    print(f"  {'─'*40}")
    for name, preds in [("Vanilla", v_predictions), ("Temporal", t_predictions)]:
        print(f"  {name:<12} {preds[0]/total:>9.1%} {preds[1]/total:>9.1%} {preds[2]/total:>9.1%}")

    print(f"\n  Gap (temporal - vanilla): {(t_overall - v_overall)*100:+.1f}pp")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
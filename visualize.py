"""
Visualization script — generates all figures for the paper.

Produces:
  outputs/occlusion_comparison.png  — side-by-side slot masks: vanilla vs temporal
  outputs/slot_drift_analysis.png   — per-slot attention entropy over time
  outputs/reconstruction_grid.png   — per-slot reconstructions across frames
  outputs/occlusion_summary.gif     — animated GIF of the occlusion event

Run after training:
    python visualize.py
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from model import SlotAutoencoder, TemporalSlotAutoencoder
from generate_data import generate_video


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_type, device, args):
    ckpt_path = Path("checkpoints") / f"{model_type}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Run train.py first.")

    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt["args"]

    resolution = (saved_args["canvas_size"], saved_args["canvas_size"])
    if model_type == "vanilla":
        model = SlotAutoencoder(
            resolution=resolution,
            n_slots=saved_args["n_slots"],
            slot_dim=saved_args["slot_dim"],
            encoder_hidden=saved_args["encoder_hidden"],
            n_iters=saved_args["n_iters"],
        )
    else:
        model = TemporalSlotAutoencoder(
            resolution=resolution,
            n_slots=saved_args["n_slots"],
            slot_dim=saved_args["slot_dim"],
            encoder_hidden=saved_args["encoder_hidden"],
            n_iters=saved_args["n_iters"],
        )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, saved_args


@torch.no_grad()
def run_inference(model, frames_np, device, temporal=False):
    """
    Run model on a single video.
    frames_np: (T, H, W, 3) float32
    Returns:
        all_masks:  (T, K, H, W)
        all_recons: (T, 3, H, W)
        all_slots:  (T, K, D)
        all_attns:  (T, K, N)
    """
    T = len(frames_np)
    frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).to(device)  # (T,3,H,W)

    all_masks, all_recons, all_slots, all_attns = [], [], [], []
    prev_slots = None

    for t in range(T):
        frame = frames_t[t:t+1]  # (1, 3, H, W)

        if temporal:
            recon, masks, slots, attn = model(frame, prev_slots=prev_slots)
            prev_slots = slots
        else:
            recon, masks, slots, attn = model(frame)

        all_masks.append(masks[0].cpu().numpy())   # (K, 1, H, W)
        all_recons.append(recon[0].cpu().numpy())  # (3, H, W)
        all_slots.append(slots[0].cpu().numpy())   # (K, D)
        all_attns.append(attn[0].cpu().numpy())    # (K, N)

    all_masks  = np.stack(all_masks)   # (T, K, 1, H, W)
    all_recons = np.stack(all_recons)  # (T, 3, H, W)
    all_slots  = np.stack(all_slots)   # (T, K, D)
    all_attns  = np.stack(all_attns)   # (T, K, N)

    return all_masks, all_recons, all_slots, all_attns


def img(x):
    """Convert (C,H,W) or (H,W,C) tensor/array to (H,W,3) uint8."""
    if x.ndim == 3 and x.shape[0] == 3:
        x = x.transpose(1, 2, 0)
    return np.clip(x, 0, 1)


# ---------------------------------------------------------------------------
# Figure 1: Side-by-side occlusion comparison
# ---------------------------------------------------------------------------

def plot_occlusion_comparison(frames, occ_mask, vanilla_masks, temporal_masks,
                               frame_indices, out_path):
    """
    For selected frames, show:
      Row 0: original frame
      Rows 1..K: vanilla slot masks
      Rows K+1..2K: temporal slot masks
    """
    K = vanilla_masks.shape[1]
    n_frames = len(frame_indices)
    slot_colors = ["#FF4444", "#4499FF", "#44FF66", "#FFD700"]

    fig_h = 2 + K * 1.2 * 2
    fig, axes = plt.subplots(1 + K * 2, n_frames,
                              figsize=(n_frames * 2, fig_h))
    fig.patch.set_facecolor("#1a1a2e")

    for col, fi in enumerate(frame_indices):
        frame = img(frames[fi])
        is_occ = occ_mask[fi, 0]  # object 0 occluded?

        # Original frame
        ax = axes[0, col]
        ax.imshow(frame)
        ax.set_xticks([]); ax.set_yticks([])
        title = f"t={fi}"
        if is_occ:
            title += "\n[OCCLUDED]"
            for spine in ax.spines.values():
                spine.set_edgecolor("red"); spine.set_linewidth(2)
        ax.set_title(title, color="white", fontsize=8)

        # Vanilla masks
        for k in range(K):
            ax = axes[1 + k, col]
            mask = vanilla_masks[fi, k, 0]  # (H, W)
            ax.imshow(mask, cmap="inferno", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"V-Slot {k+1}", color=slot_colors[k % len(slot_colors)],
                              fontsize=7, rotation=0, labelpad=35)

        # Temporal masks
        for k in range(K):
            ax = axes[1 + K + k, col]
            mask = temporal_masks[fi, k, 0]
            ax.imshow(mask, cmap="inferno", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"T-Slot {k+1}", color=slot_colors[k % len(slot_colors)],
                              fontsize=7, rotation=0, labelpad=35)

    # Row labels
    axes[0, 0].set_ylabel("Input", color="white", fontsize=8, rotation=0, labelpad=35)

    # Section labels
    fig.text(0.01, 0.72, "Vanilla\n(no memory)", color="#FF8888",
             fontsize=9, va="center", rotation=90, fontweight="bold")
    fig.text(0.01, 0.30, "Temporal\n(with memory)", color="#88FF88",
             fontsize=9, va="center", rotation=90, fontweight="bold")

    fig.suptitle("Slot Attention: Vanilla vs Temporal Propagation\nunder Occlusion",
                 color="white", fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Slot attention entropy over time
# ---------------------------------------------------------------------------

def plot_slot_entropy(frames_np, occ_mask, vanilla_attns, temporal_attns, out_path):
    """
    Attention entropy for slot 0 (which tracks the occluded object).
    High entropy = slot is confused / attending everywhere.
    Low entropy  = slot is tightly bound to one region.
    Binding drift appears as entropy SPIKE during occlusion in vanilla model.
    """
    T = len(frames_np)
    K = vanilla_attns.shape[1]

    def entropy(attn_tk):
        # attn_tk: (T, N) for slot k
        p = attn_tk + 1e-8
        p = p / p.sum(axis=1, keepdims=True)
        return -(p * np.log(p)).sum(axis=1)  # (T,)

    fig, axes = plt.subplots(1, K, figsize=(4 * K, 3.5))
    fig.patch.set_facecolor("#1a1a2e")
    if K == 1:
        axes = [axes]

    t = np.arange(T)
    occ_frames = np.where(occ_mask[:, 0])[0]

    for k, ax in enumerate(axes):
        ax.set_facecolor("#16213e")
        v_ent = entropy(vanilla_attns[:, k, :])
        t_ent = entropy(temporal_attns[:, k, :])

        ax.plot(t, v_ent, color="#FF6B6B", linewidth=2,
                label="Vanilla (no memory)", zorder=3)
        ax.plot(t, t_ent, color="#6BCB77", linewidth=2,
                label="Temporal (with memory)", zorder=3)

        if len(occ_frames) > 0:
            ax.axvspan(occ_frames[0], occ_frames[-1], alpha=0.15,
                       color="red", label="Occlusion window")

        ax.set_title(f"Slot {k+1} Attention Entropy", color="white", fontsize=10)
        ax.set_xlabel("Frame", color="white", fontsize=9)
        ax.set_ylabel("Entropy (nats)", color="white", fontsize=9)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")

    fig.suptitle("Attention Entropy During Occlusion\n"
                 "(High entropy = slot confused, binding drift occurring)",
                 color="white", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Per-slot reconstruction grid
# ---------------------------------------------------------------------------

def plot_reconstruction_grid(frames_np, vanilla_masks, temporal_masks,
                               vanilla_recons, temporal_recons,
                               frame_indices, out_path):
    K = vanilla_masks.shape[1]
    n_f = len(frame_indices)

    fig = plt.figure(figsize=(n_f * 2, (K + 1) * 2 * 2 + 1))
    fig.patch.set_facecolor("#1a1a2e")
    gs = GridSpec(2 * (K + 1), n_f, figure=fig, hspace=0.05, wspace=0.05)

    def ax_img(ax, data, title=None, border_color=None):
        ax.imshow(img(data))
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title, color="white", fontsize=7)
        if border_color:
            for s in ax.spines.values():
                s.set_edgecolor(border_color); s.set_linewidth(2)

    for col, fi in enumerate(frame_indices):
        # Row 0: input frame
        ax = fig.add_subplot(gs[0, col])
        ax_img(ax, frames_np[fi], f"t={fi}")

        # Vanilla per-slot reconstructions
        for k in range(K):
            ax = fig.add_subplot(gs[1 + k, col])
            slot_recon = img(frames_np[fi]) * vanilla_masks[fi, k, 0, :, :, np.newaxis]
            ax_img(ax, slot_recon)

        # Separator row — temporal section
        ax = fig.add_subplot(gs[K + 1, col])
        ax_img(ax, frames_np[fi])
        if col == 0:
            ax.set_ylabel("──── Temporal ────", color="#88FF88", fontsize=7)

        for k in range(K):
            ax = fig.add_subplot(gs[K + 2 + k, col])
            slot_recon = img(frames_np[fi]) * temporal_masks[fi, k, 0, :, :, np.newaxis]
            ax_img(ax, slot_recon)

    fig.suptitle("Per-Slot Reconstructions: Vanilla vs Temporal",
                 color="white", fontsize=11, y=1.005)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Animated GIF
# ---------------------------------------------------------------------------

def save_gif(frames_np, vanilla_masks, temporal_masks, occ_mask, out_path):
    try:
        import imageio
    except ImportError:
        print("imageio not installed, skipping GIF. Run: pip install imageio")
        return

    T, K = len(frames_np), vanilla_masks.shape[1]
    gif_frames = []
    slot_colors_rgb = [
        (1.0, 0.3, 0.3), (0.3, 0.6, 1.0), (0.3, 1.0, 0.5), (1.0, 0.9, 0.2)
    ]

    for t in range(T):
        fig, axes = plt.subplots(3, K + 1, figsize=((K + 1) * 2, 6))
        fig.patch.set_facecolor("#1a1a2e")

        # Input
        for row in range(3):
            ax = axes[row, 0]
            ax.imshow(img(frames_np[t]))
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title("Input", color="white", fontsize=8)
                if occ_mask[t, 0]:
                    ax.set_title("Input\n⚠ OCCLUDED", color="red", fontsize=8)

        axes[1, 0].set_ylabel("Vanilla", color="#FF8888", fontsize=9)
        axes[2, 0].set_ylabel("Temporal", color="#88FF88", fontsize=9)

        for k in range(K):
            # Vanilla
            ax = axes[1, k + 1]
            ax.imshow(vanilla_masks[t, k, 0], cmap="inferno", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"Slot {k+1}", color="white", fontsize=8)

            # Temporal
            ax = axes[2, k + 1]
            ax.imshow(temporal_masks[t, k, 0], cmap="inferno", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])

        axes[0, 0].axis("off")
        fig.suptitle(f"Frame {t:02d}/{T-1}", color="white", fontsize=10)
        plt.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        gif_frames.append(buf.reshape(h, w, 3))
        plt.close()

    imageio.mimsave(out_path, gif_frames, fps=5, loop=0)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",         type=int, default=1337,
                        help="Seed for the test video (try different seeds)")
    parser.add_argument("--n_objects",    type=int, default=3)
    parser.add_argument("--canvas_size",  type=int, default=64)
    parser.add_argument("--n_frames",     type=int, default=30)
    parser.add_argument("--gif",          action="store_true",
                        help="Also generate animated GIF (requires imageio)")
    args = parser.parse_args()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load both models
    print("Loading models...")
    vanilla_model,  saved_args = load_model("vanilla",  device, args)
    temporal_model, _          = load_model("temporal", device, args)

    # Generate a fresh test video (not seen during training)
    print(f"Generating test video (seed={args.seed})...")
    frames_np, positions, occ_mask = generate_video(
        canvas_size=args.canvas_size,
        n_frames=args.n_frames,
        n_objects=args.n_objects,
        radius=8,
        seed=args.seed,
    )
    print(f"  Occlusion frames: {np.where(occ_mask[:, 0])[0].tolist()}")

    # Run inference
    print("Running vanilla inference...")
    v_masks, v_recons, v_slots, v_attns = run_inference(
        vanilla_model, frames_np, device, temporal=False
    )
    print("Running temporal inference...")
    t_masks, t_recons, t_slots, t_attns = run_inference(
        temporal_model, frames_np, device, temporal=True
    )

    # Select representative frames: before, during, after occlusion
    occ_frames = np.where(occ_mask[:, 0])[0]
    if len(occ_frames) > 0:
        mid_occ = occ_frames[len(occ_frames) // 2]
        frame_indices = sorted(set([
            max(0, occ_frames[0] - 5),
            max(0, occ_frames[0] - 2),
            mid_occ,
            min(args.n_frames - 1, occ_frames[-1] + 2),
            min(args.n_frames - 1, occ_frames[-1] + 5),
        ]))
    else:
        step = args.n_frames // 5
        frame_indices = list(range(0, args.n_frames, step))[:5]

    print(f"Visualization frames: {frame_indices}")

    # Generate figures
    print("\nGenerating figures...")
    plot_occlusion_comparison(
        frames_np, occ_mask, v_masks, t_masks,
        frame_indices,
        out_dir / "occlusion_comparison.png"
    )
    plot_slot_entropy(
        frames_np, occ_mask, v_attns, t_attns,
        out_dir / "slot_drift_analysis.png"
    )
    plot_reconstruction_grid(
        frames_np, v_masks, t_masks, v_recons, t_recons,
        frame_indices,
        out_dir / "reconstruction_grid.png"
    )
    if args.gif:
        save_gif(frames_np, v_masks, t_masks, occ_mask,
                 out_dir / "occlusion_summary.gif")

    print(f"\nAll outputs saved to {out_dir}/")
    print("Figures for your paper:")
    print("  occlusion_comparison.png  → main comparison figure")
    print("  slot_drift_analysis.png   → entropy/drift analysis")
    print("  reconstruction_grid.png   → per-slot reconstruction grid")


if __name__ == "__main__":
    main()

"""
Shell Game Sample Inspector

Generates animated GIFs of N random videos from the dataset
so you can visually verify the data looks correct before training.

Usage:
    python inspect_shell_samples.py                    # 10 random samples
    python inspect_shell_samples.py --n 20             # 20 samples
    python inspect_shell_samples.py --seed 42          # fixed random seed
    python inspect_shell_samples.py --generate         # generate fresh data first
"""

import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def make_gif(frames, ball_pos_seq, cup_pos_seq, label, video_idx, out_path, canvas_size):
    """
    Create an annotated GIF for one video.
    Shows: frame, which phase we're in, ball position dot overlay, final label.
    """
    try:
        import imageio
    except ImportError:
        print("imageio not installed. Run: pip install imageio")
        return

    T = len(frames)
    gif_frames = []

    # Detect phases by ball visibility change
    # Ball is visible when no cup overlaps it
    CUP_R_APPROX = canvas_size * 0.156  # matches generator (cup_r=10 on 64px canvas)

    def cup_covers_ball(t):
        ball = ball_pos_seq[t]
        for cx, cy in cup_pos_seq[t]:
            if np.hypot(ball[0] - cx, ball[1] - cy) < CUP_R_APPROX * 0.9:
                return True
        return False

    for t in range(T):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
        fig.patch.set_facecolor('#0d0d14')

        # Left: raw frame (what the model sees)
        ax = axes[0]
        ax.imshow(np.clip(frames[t], 0, 1), interpolation='nearest')
        ax.set_title('model input', color='#a89fff', fontsize=9, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#3a3560')

        # Right: annotated view
        ax = axes[1]
        ax.imshow(np.clip(frames[t], 0, 1), interpolation='nearest')

        # Draw ball position as dot even when hidden
        bx, by = ball_pos_seq[t]
        hidden = cup_covers_ball(t)
        dot_color = '#f5c842' if not hidden else '#ff6b6b'
        dot_style = 'o' if not hidden else 'x'
        ax.plot(bx, by, dot_style, color=dot_color,
                markersize=6, markeredgewidth=1.5,
                label='ball (true pos)')

        ax.set_title('annotated', color='#a89fff', fontsize=9, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#3a3560')

        # Phase label
        if t < 8:
            phase_txt = 'rising'
        elif t < 12:
            phase_txt = 'paused'
        elif t < 20:
            phase_txt = 'occluding'
        elif t < T - 4:
            phase_txt = 'shuffling'
        else:
            phase_txt = 'rest'

        hidden_txt = ' [hidden]' if hidden else ' [visible]'
        fig.suptitle(
            f'video {video_idx} | frame {t+1}/{T} | {phase_txt}{hidden_txt} | final: {label}',
            color='white', fontsize=8, y=1.01
        )

        plt.tight_layout(pad=0.5)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        gif_frames.append(buf.reshape(h, w, 4)[:, :, :3])
        plt.close()

    imageio.mimsave(out_path, gif_frames, fps=8, loop=0)


def make_contact_sheet(samples, out_path, canvas_size):
    """
    Single PNG showing first, mid-occlusion, and final frame
    for all sampled videos — quick at-a-glance overview.
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(10, n * 2.2))
    fig.patch.set_facecolor('#0d0d14')

    if n == 1:
        axes = [axes]

    col_labels = ['start', 'occluding', 'shuffling', 'final']

    for row, (idx, frames, ball_pos_seq, cup_pos_seq, label) in enumerate(samples):
        T = len(frames)
        key_frames = [0, 16, T // 2, T - 1]

        for col, fi in enumerate(key_frames):
            ax = axes[row][col]
            ax.imshow(np.clip(frames[fi], 0, 1), interpolation='nearest')

            # Annotate ball position
            bx, by = ball_pos_seq[fi]
            ax.plot(bx, by, 'x', color='#ff6b6b', markersize=5, markeredgewidth=1.5)

            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('#3a3560')

            if row == 0:
                ax.set_title(col_labels[col], color='#a89fff', fontsize=9)
            if col == 0:
                ax.set_ylabel(f'v{idx}\n{label}', color='#f5c842', fontsize=8,
                              rotation=0, labelpad=40, va='center')

    fig.suptitle('Shell game dataset — sample contact sheet\n(x marks true ball position)',
                 color='white', fontsize=10, y=1.01)
    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved contact sheet: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  default="data/shell_train",
                        help="Directory containing frames.npy etc.")
    parser.add_argument("--out_dir",   default="outputs/shell_samples")
    parser.add_argument("--n",         type=int, default=10,
                        help="Number of random videos to sample")
    parser.add_argument("--seed",      type=int, default=None,
                        help="Random seed for sampling (omit for truly random)")
    parser.add_argument("--generate",  action="store_true",
                        help="Generate fresh data before inspecting")
    parser.add_argument("--no_gif",    action="store_true",
                        help="Skip GIFs, only make contact sheet (faster)")
    args = parser.parse_args()

    # Optionally regenerate data first
    if args.generate:
        import subprocess
        print("Generating fresh shell game data...")
        subprocess.run(["python", "generate_shell_data.py",
                        "--n_videos", "2000"], check=True)

    data_dir = Path(args.data_dir)
    if not (data_dir / "frames.npy").exists():
        print(f"No data found at {data_dir}/")
        print("Run: python generate_shell_data.py  (or use --generate flag)")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {data_dir}/...")
    frames_all   = np.load(data_dir / "frames.npy")    # (N,T,H,W,3)
    ball_pos_all = np.load(data_dir / "ball_pos.npy")  # (N,T,2)
    cup_pos_all  = np.load(data_dir / "cup_pos.npy")   # (N,T,3,2)
    canvas_size  = frames_all.shape[2]

    with open(data_dir / "labels.txt") as f:
        labels = [l.strip() for l in f.readlines()]

    N = len(frames_all)
    print(f"Dataset: {N} videos, {frames_all.shape[1]} frames each")

    # Sample random indices
    rng = random.Random(args.seed)
    indices = rng.sample(range(N), min(args.n, N))
    print(f"Sampled video indices: {indices}")

    # Label distribution in sample
    sample_labels = [labels[i] for i in indices]
    from collections import Counter
    print(f"Label distribution in sample: {dict(Counter(sample_labels))}")

    # Load samples
    samples = []
    for idx in indices:
        samples.append((
            idx,
            frames_all[idx],
            ball_pos_all[idx],
            cup_pos_all[idx],
            labels[idx],
        ))

    # Contact sheet — quick overview
    make_contact_sheet(samples, out_dir / "contact_sheet.png", canvas_size)

    # Animated GIFs — one per video
    if not args.no_gif:
        try:
            import imageio
            print(f"\nGenerating {len(samples)} animated GIFs...")
            for idx, frames, ball_pos_seq, cup_pos_seq, label in samples:
                out_path = out_dir / f"video_{idx:04d}_{label}.gif"
                make_gif(frames, ball_pos_seq, cup_pos_seq, label,
                         idx, out_path, canvas_size)
                print(f"  Saved: {out_path.name}")
        except ImportError:
            print("imageio not installed — skipping GIFs. Run: pip install imageio")
            print("Contact sheet was saved successfully.")
    else:
        print("Skipped GIFs (--no_gif). Contact sheet saved.")

    print(f"\nAll outputs in: {out_dir}/")
    print("Check contact_sheet.png first for a quick overview.")
    print("Open individual GIFs to watch each video play frame by frame.")


if __name__ == "__main__":
    main()
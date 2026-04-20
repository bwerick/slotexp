"""
Shell Game Dataset Generator

Matches the browser demo exactly:
  - Ball starts at rest at a RANDOM cup's position (not always middle)
  - That cup rises, pauses, then descends to occlude the ball
  - No labels on cups at any point
  - Shuffling uses arc-based non-overlapping swaps
  - Labels are final (x,y) position of ball + left/middle/right category

Usage:
    python generate_shell_data.py --out_dir data/shell_train --n_videos 2000
"""

import numpy as np
import argparse
from pathlib import Path


CUP_COLOR  = np.array([0.48, 0.42, 1.0])
BALL_COLOR = np.array([0.96, 0.78, 0.26])


def make_circle(canvas_size, cx, cy, r, color):
    img = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
    y, x = np.ogrid[:canvas_size, :canvas_size]
    mask = (x - cx)**2 + (y - cy)**2 <= r**2
    img[mask] = color
    return img


def lerp(a, b, t):
    return a + (b - a) * t


def eased(t):
    return t * t * (3 - 2 * t)


def render_frame(canvas_size, cup_positions, cup_r, ball_pos, ball_on_top, ball_r):
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
    def paint(cx, cy, r, color):
        layer = make_circle(canvas_size, cx, cy, r, color)
        mask = layer.sum(axis=2) > 0
        canvas[mask] = layer[mask]
    if not ball_on_top:
        paint(ball_pos[0], ball_pos[1], ball_r, BALL_COLOR)
        for cx, cy in cup_positions:
            paint(cx, cy, cup_r, CUP_COLOR)
    else:
        for cx, cy in cup_positions:
            paint(cx, cy, cup_r, CUP_COLOR)
        paint(ball_pos[0], ball_pos[1], ball_r, BALL_COLOR)
    return canvas


def pos_label(x, canvas_size):
    if x < canvas_size * 0.35: return 'left'
    if x > canvas_size * 0.65: return 'right'
    return 'middle'


def generate_shell_video(canvas_size=64, n_frames=80, cup_r=6, ball_r=3,
                          n_shuffles=6, seed=None):
    if seed is not None:
        np.random.seed(seed)

    cy_center = canvas_size // 2
    # Perfectly symmetric positions around center
    # Using 25/50/75% gives exact symmetry: gaps of 16px each side
    center = canvas_size // 2
    spread = canvas_size // 4  # 16px from center
    start_xs = [center - spread, center, center + spread]
    cup_positions = [[float(x), float(cy_center)] for x in start_xs]

    # Random cup hides the ball
    hiding_cup = np.random.randint(0, 3)
    ball_pos = [float(start_xs[hiding_cup]), float(cy_center)]
    cup_up_y = cy_center - cup_r * 3.0
    min_sep = cup_r * 2.5  # generous margin

    frames, ball_pos_seq, cup_pos_seq = [], [], []

    def snap(ball, cups, ball_top=False):
        frame = render_frame(canvas_size, cups, cup_r, ball, ball_top, ball_r)
        frames.append(frame)
        ball_pos_seq.append(list(ball))
        cup_pos_seq.append([list(p) for p in cups])

    # Phase 1: cup rises (8 frames)
    for fi in range(8):
        t = eased(fi / 7)
        cup_positions[hiding_cup][1] = lerp(cy_center, cup_up_y, t)
        snap(ball_pos, cup_positions)

    # Hold raised (4 frames)
    for _ in range(4):
        snap(ball_pos, cup_positions)

    # Phase 2: cup descends (8 frames)
    for fi in range(8):
        t = eased(fi / 7)
        cup_positions[hiding_cup][1] = lerp(cup_up_y, cy_center, t)
        snap(ball_pos, cup_positions)

    cup_positions[hiding_cup][1] = cy_center
    # Track ball by which cup array INDEX holds it.
    # IMPORTANT: indices NEVER change — cup_positions[i] always refers to the
    # same physical cup. Swaps just move where that cup is located in space.
    # So ball_cup_idx never needs to change after a swap.
    ball_cup_idx = hiding_cup

    # Phase 3: shuffle
    # 16 frames per swap minimum — keeps per-frame delta small for slot tracking
    frames_per_shuffle = max(16, (n_frames - 24) // n_shuffles)
    last_pair = [-1, -1]

    for _ in range(n_shuffles):
        while True:
            a, b = np.random.choice(3, size=2, replace=False).tolist()
            if [a, b] != last_pair:
                break
        last_pair = [a, b]

        ax, ay = cup_positions[a]
        bx, by = cup_positions[b]
        c = [i for i in range(3) if i != a and i != b][0]
        cx, cy_s = cup_positions[c]
        arc_h = max(min_sep * 2.0, cup_r * 5.0)
        sign_a = -1 if ax <= bx else 1

        for fi in range(frames_per_shuffle):
            t = eased(fi / max(frames_per_shuffle - 1, 1))
            np_ = [list(p) for p in cup_positions]
            np_[a][0] = lerp(ax, bx, t)
            np_[a][1] = ay + sign_a * arc_h * np.sin(np.pi * t)
            np_[b][0] = lerp(bx, ax, t)
            np_[b][1] = by - sign_a * arc_h * np.sin(np.pi * t)

            for moving in [a, b]:
                dx = np_[moving][0] - cx
                dy = np_[moving][1] - cy_s
                dist = (dx**2 + dy**2) ** 0.5
                if dist < min_sep and dist > 0:
                    s = min_sep / dist
                    np_[moving][0] = cx + dx * s
                    np_[moving][1] = cy_s + dy * s

            dax = np_[a][0] - np_[b][0]
            day = np_[a][1] - np_[b][1]
            dab = (dax**2 + day**2) ** 0.5
            if dab < min_sep and dab > 0:
                s = min_sep / dab
                mx = (np_[a][0] + np_[b][0]) / 2
                my = (np_[a][1] + np_[b][1]) / 2
                np_[a][0] = mx + dax * s / 2; np_[a][1] = my + day * s / 2
                np_[b][0] = mx - dax * s / 2; np_[b][1] = my - day * s / 2

            # Ball always follows its cup by reading np_[ball_cup_idx] directly
            # ball_cup_idx never changes — the same array index always = same cup
            snap(list(np_[ball_cup_idx]), np_)

        # Commit: update where each cup physically is in space
        # The cup at index a moved to where b was, and vice versa
        cup_positions[a] = list(np_[a])
        cup_positions[b] = list(np_[b])
        cup_positions[c] = list(np_[c])
        # ball_cup_idx stays the same — indices are permanent

    # Phase 4: rest (4 frames)
    ball_final = list(cup_positions[ball_cup_idx])
    for _ in range(4):
        snap(ball_final, cup_positions)

    while len(frames) > n_frames:
        frames.pop(); ball_pos_seq.pop(); cup_pos_seq.pop()
    while len(frames) < n_frames:
        frames.append(frames[-1])
        ball_pos_seq.append(ball_pos_seq[-1])
        cup_pos_seq.append(cup_pos_seq[-1])

    frames_np      = np.stack(frames).astype(np.float32)
    ball_pos_np    = np.array(ball_pos_seq, dtype=np.float32)
    cup_pos_np     = np.array(cup_pos_seq, dtype=np.float32)
    final_ball_pos = ball_pos_np[-1]
    label          = pos_label(final_ball_pos[0], canvas_size)

    return frames_np, ball_pos_np, final_ball_pos, cup_pos_np, label


def generate_dataset(n_videos, canvas_size, n_frames, n_shuffles, out_dir, seed=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_frames, all_ball_pos, all_final_pos, all_cup_pos, all_labels = [], [], [], [], []
    label_counts = {'left': 0, 'middle': 0, 'right': 0}

    print(f"Generating {n_videos} shell game videos...")
    for i in range(n_videos):
        frames, ball_pos, final_pos, cup_pos, label = generate_shell_video(
            canvas_size=canvas_size, n_frames=n_frames,
            n_shuffles=n_shuffles, seed=seed + i,
        )
        all_frames.append(frames); all_ball_pos.append(ball_pos)
        all_final_pos.append(final_pos); all_cup_pos.append(cup_pos)
        all_labels.append(label); label_counts[label] += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_videos}")

    np.save(out_dir / "frames.npy",    np.stack(all_frames))
    np.save(out_dir / "ball_pos.npy",  np.stack(all_ball_pos))
    np.save(out_dir / "final_pos.npy", np.stack(all_final_pos))
    np.save(out_dir / "cup_pos.npy",   np.stack(all_cup_pos))
    with open(out_dir / "labels.txt", "w") as f:
        f.write("\n".join(all_labels) + "\n")

    print(f"Saved to {out_dir}/")
    print(f"  frames: {np.stack(all_frames).shape}")
    print(f"  Label distribution: {label_counts}  (should be roughly equal)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",     default="data/shell_train")
    parser.add_argument("--n_videos",    type=int, default=2000)
    parser.add_argument("--canvas_size", type=int, default=64)  # cup_r=6 fits well on 64px
    parser.add_argument("--n_frames",    type=int, default=80)
    parser.add_argument("--n_shuffles",  type=int, default=6)
    parser.add_argument("--seed",        type=int, default=0)
    args = parser.parse_args()

    generate_dataset(
        n_videos=args.n_videos, canvas_size=args.canvas_size,
        n_frames=args.n_frames, n_shuffles=args.n_shuffles,
        out_dir=args.out_dir, seed=args.seed,
    )
    val_dir = args.out_dir.replace("train", "val")
    generate_dataset(
        n_videos=200, canvas_size=args.canvas_size,
        n_frames=args.n_frames, n_shuffles=args.n_shuffles,
        out_dir=val_dir, seed=args.seed + 999999,
    )
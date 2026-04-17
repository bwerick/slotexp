"""
Synthetic Occlusion Dataset Generator
Generates videos of colored shapes moving across a canvas,
with deliberate occlusion events for slot attention experiments.
"""

import numpy as np
import os
import argparse
from pathlib import Path


def make_circle(canvas_size, cx, cy, r, color):
    """Draw a filled circle on a black canvas."""
    img = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
    y, x = np.ogrid[:canvas_size, :canvas_size]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    img[mask] = color
    return img


def generate_video(canvas_size=64, n_frames=30, n_objects=3, radius=8, seed=None):
    """
    Generate a single video with n_objects colored circles.
    One occlusion event is guaranteed: object 0 passes behind object 1.

    Returns:
        frames: (T, H, W, 3) float32 array in [0, 1]
        positions: (T, N, 2) float32 array of (cx, cy) per object per frame
        occlusion_mask: (T, N) bool array, True when object i is occluded
    """
    if seed is not None:
        np.random.seed(seed)

    # Fixed distinct colors for each object
    colors = [
        np.array([1.0, 0.2, 0.2]),   # red
        np.array([0.2, 0.6, 1.0]),   # blue
        np.array([0.2, 1.0, 0.4]),   # green
        np.array([1.0, 0.9, 0.1]),   # yellow
        np.array([1.0, 0.4, 1.0]),   # magenta
    ][:n_objects]

    margin = radius + 2
    lo, hi = margin, canvas_size - margin

    # --- Design occlusion: object 0 travels across object 1 ---
    # Object 1 stays near center
    obj1_x = canvas_size // 2
    obj1_y = canvas_size // 2

    # Object 0 starts left, ends right, crossing through object 1's position
    obj0_start = np.array([lo, obj1_y + np.random.randint(-4, 5)], dtype=float)
    obj0_end   = np.array([hi, obj1_y + np.random.randint(-4, 5)], dtype=float)

    # Other objects move randomly
    other_starts = np.random.uniform(lo, hi, (n_objects - 2, 2)) if n_objects > 2 else np.zeros((0, 2))
    other_vels   = np.random.uniform(-1.5, 1.5, (n_objects - 2, 2)) if n_objects > 2 else np.zeros((0, 2))

    frames = []
    positions = []
    occlusion_mask = []

    for t in range(n_frames):
        alpha = t / max(n_frames - 1, 1)

        # Object 0 position (the traveler)
        p0 = obj0_start + alpha * (obj0_end - obj0_start)

        # Object 1 position (the occluder, fixed)
        p1 = np.array([obj1_x, obj1_y], dtype=float)

        # Other objects bounce around
        other_pos = []
        for i, (s, v) in enumerate(zip(other_starts, other_vels)):
            pos = s + v * t
            # Bounce off walls
            for d in range(2):
                if pos[d] < lo or pos[d] > hi:
                    other_vels[i, d] *= -1
                pos[d] = np.clip(pos[d], lo, hi)
            other_pos.append(pos.copy())

        all_positions = [p0, p1] + other_pos
        positions.append(np.array(all_positions))

        # Determine draw order: object 1 (occluder) drawn on top of object 0
        # So render order is: others, object 0, object 1
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
        render_order = list(range(2, n_objects)) + [0, 1]  # 1 always on top of 0

        occ = np.zeros(n_objects, dtype=bool)
        for idx in render_order:
            px, py = all_positions[idx]
            layer = make_circle(canvas_size, px, py, radius, colors[idx])
            # Wherever this object paints, it overwrites
            mask = layer.sum(axis=2) > 0
            canvas[mask] = layer[mask]

        # Check occlusion: object 0 is occluded if it overlaps with object 1
        dist_01 = np.linalg.norm(p0 - p1)
        occ[0] = dist_01 < (radius * 1.8)  # partially or fully behind obj 1

        frames.append(canvas)
        occlusion_mask.append(occ)

    frames = np.stack(frames, axis=0)           # (T, H, W, 3)
    positions = np.stack(positions, axis=0)     # (T, N, 2)
    occlusion_mask = np.stack(occlusion_mask)   # (T, N)

    return frames, positions, occlusion_mask


def generate_dataset(n_videos, canvas_size, n_frames, n_objects, radius, out_dir, seed=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    all_positions = []
    all_occlusions = []

    print(f"Generating {n_videos} videos ({n_frames} frames each, {n_objects} objects)...")
    for i in range(n_videos):
        frames, positions, occ = generate_video(
            canvas_size=canvas_size,
            n_frames=n_frames,
            n_objects=n_objects,
            radius=radius,
            seed=seed + i
        )
        all_frames.append(frames)
        all_positions.append(positions)
        all_occlusions.append(occ)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_videos}")

    np.save(out_dir / "frames.npy",     np.stack(all_frames))      # (N, T, H, W, 3)
    np.save(out_dir / "positions.npy",  np.stack(all_positions))   # (N, T, K, 2)
    np.save(out_dir / "occlusions.npy", np.stack(all_occlusions))  # (N, T, K)
    print(f"Saved dataset to {out_dir}/")
    print(f"  frames.npy:     {np.stack(all_frames).shape}")
    print(f"  positions.npy:  {np.stack(all_positions).shape}")
    print(f"  occlusions.npy: {np.stack(all_occlusions).shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",     default="data/train")
    parser.add_argument("--n_videos",    type=int, default=2000)
    parser.add_argument("--canvas_size", type=int, default=64)
    parser.add_argument("--n_frames",    type=int, default=30)
    parser.add_argument("--n_objects",   type=int, default=3)
    parser.add_argument("--radius",      type=int, default=8)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        n_videos=args.n_videos,
        canvas_size=args.canvas_size,
        n_frames=args.n_frames,
        n_objects=args.n_objects,
        radius=args.radius,
        out_dir=args.out_dir,
        seed=args.seed,
    )

    # Also generate a small validation set
    val_dir = args.out_dir.replace("train", "val")
    generate_dataset(
        n_videos=200,
        canvas_size=args.canvas_size,
        n_frames=args.n_frames,
        n_objects=args.n_objects,
        radius=args.radius,
        out_dir=val_dir,
        seed=args.seed + 99999,
    )

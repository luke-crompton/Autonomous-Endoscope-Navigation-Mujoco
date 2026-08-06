"""
Compute a per-pixel relative-error map from DA3 predictions vs Blender GT depth.

The map captures the spatial structure of CNN prediction errors — which pixels
are systematically more or less uncertain — and is used to inject realistic
observation noise into the RL training environment (Option A, scope_colon_env.py).

Option A injects: depth_obs[i,j] += N(0, error_map[i,j]) * depth_obs[i,j]
so noise is proportional to depth value (close walls get small absolute noise,
open lumen gets larger absolute noise), matching how depth estimation error scales.

Future options (more realistic, see memory/future_optimisations.md):
  Option B — also compute per-pixel bias (signed mean error) and inject that too
  Option C — bucket errors by GT depth value instead of image position
  Option D — combined spatial + depth-value model

Usage:
    python compute_error_map.py
    python compute_error_map.py --pred_dir predictions_eval_500 --gt_dir dataset_eval_500/depth

Outputs (all in Perception/RD_V2/):
    error_map_full.npy   (H, W) = (180, 320) mean relative error per pixel
    error_map_64.npy     (64, 64) downsampled — copy this to navigation/v7_shaft/
    error_map_vis.png    heatmap for visual inspection (bright = high uncertainty)
    error_map_stats.json summary statistics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pred_dir",
        default=str(HERE / "predictions_eval_500"),
        help="Output dir from run_da3_inference.py (contains depth_pred/ subfolder).",
    )
    p.add_argument(
        "--gt_dir",
        default=str(HERE / "dataset_eval_500" / "depth"),
        help="GT depth folder from blender_depth_dataset_eval.py.",
    )
    p.add_argument(
        "--out_dir",
        default=str(HERE),
        help="Where to write error_map_*.npy and error_map_vis.png.",
    )
    p.add_argument(
        "--depth_res",
        type=int,
        default=64,
        help="RL depth obs resolution — error map is downsampled to (N, N).",
    )
    p.add_argument(
        "--noise_floor",
        type=float,
        default=0.02,
        help="Minimum relative error std added everywhere (avoids zero-noise "
             "regions where the CNN happens to be perfect on this eval set). "
             "0.02 = 2%% of depth value.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    pred_depth_dir = Path(args.pred_dir) / "depth_pred"
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_depth_dir.exists():
        sys.exit(
            f"Prediction folder not found: {pred_depth_dir}\n"
            "Run run_da3_inference.py --dataset_dir dataset_eval_500 "
            "--output_dir predictions_eval_500 --checkpoint checkpoints/da3small_finetune_head.pth"
        )
    if not gt_dir.exists():
        sys.exit(
            f"GT depth folder not found: {gt_dir}\n"
            "Run blender_depth_dataset_eval.py -- --n_geoms 5 --frames_per_geom 100 "
            "--out_dir dataset_eval_500"
        )

    pred_paths = sorted(pred_depth_dir.glob("depth_*.npy"))
    if not pred_paths:
        sys.exit(f"No depth_*.npy found in {pred_depth_dir}")

    print(f"Found {len(pred_paths)} prediction frames", flush=True)

    # Accumulate per-pixel relative error across all frames.
    # error_sum[i,j]  = sum of |relative_error| at pixel (i,j) across valid frames
    # error_count[i,j] = number of frames where pixel (i,j) had valid GT depth
    error_sum = None
    error_count = None
    H_full = W_full = None

    n_skipped = 0
    for k, pred_path in enumerate(pred_paths):
        stem = pred_path.stem          # depth_00000
        gt_path = gt_dir / (stem + ".npy")
        if not gt_path.exists():
            n_skipped += 1
            continue

        pred = np.load(str(pred_path)).astype(np.float64)
        gt = np.load(str(gt_path)).astype(np.float64)

        # Initialise accumulator on first valid frame
        if error_sum is None:
            H_full, W_full = gt.shape
            error_sum = np.zeros((H_full, W_full), dtype=np.float64)
            error_count = np.zeros((H_full, W_full), dtype=np.int32)

        # Resize prediction to GT shape if needed (DA3 may output at a different res)
        if pred.shape != gt.shape:
            pred = np.asarray(
                Image.fromarray(pred.astype(np.float32)).resize(
                    (W_full, H_full), Image.BILINEAR
                ),
                dtype=np.float64,
            )

        # Valid mask: GT must be positive (0 = invalid/open-end pixel in Blender output)
        valid = (gt > 0) & np.isfinite(pred) & (pred > 0)
        if not valid.any():
            n_skipped += 1
            continue

        # Scale-align prediction to GT using per-frame median ratio.
        # RD_V2 uses metric_weight=0.0 so output is relative — needs alignment.
        ratio = float(np.median(gt[valid] / pred[valid]))
        if not np.isfinite(ratio) or ratio <= 0:
            n_skipped += 1
            continue
        aligned = pred * ratio

        # Relative error: |aligned - gt| / gt, clipped to [0, 2] to prevent
        # outlier frames (bad GT pixels near clip boundary) from dominating the map.
        rel_err = np.where(
            valid,
            np.clip(np.abs(aligned - gt) / (gt + 1e-9), 0.0, 2.0),
            0.0,
        )
        error_sum += rel_err
        error_count += valid.astype(np.int32)

        if (k + 1) % 50 == 0 or k == 0:
            print(f"  {k+1}/{len(pred_paths)} frames processed", flush=True)

    if error_sum is None:
        sys.exit("No valid frame pairs found — check pred/GT folder alignment.")

    print(f"Processed {len(pred_paths) - n_skipped} frames ({n_skipped} skipped).")

    # Mean relative error per pixel; where count==0 fall back to global mean.
    with np.errstate(divide="ignore", invalid="ignore"):
        error_map_full = np.where(
            error_count > 0,
            error_sum / np.maximum(error_count, 1),
            np.nan,
        )
    global_mean = float(np.nanmean(error_map_full))
    error_map_full = np.where(np.isnan(error_map_full), global_mean, error_map_full)

    # Add noise floor so every pixel has at least some noise (avoids zero-gradient
    # regions where the RL has no signal about DA3 uncertainty).
    error_map_full = np.maximum(error_map_full, args.noise_floor).astype(np.float32)

    # Save full-resolution map
    full_path = out_dir / "error_map_full.npy"
    np.save(str(full_path), error_map_full)
    print(f"Saved full-res error map ({H_full}x{W_full}): {full_path}")

    # Downsample to RL depth_res
    N = args.depth_res
    error_map_small = np.asarray(
        Image.fromarray(error_map_full).resize((N, N), Image.BILINEAR),
        dtype=np.float32,
    )
    small_path = out_dir / f"error_map_{N}.npy"
    np.save(str(small_path), error_map_small)
    print(f"Saved {N}x{N} error map: {small_path}")

    # Visualisation: normalise to [0,255] and save as PNG (bright = high uncertainty)
    vis = error_map_full
    lo, hi = float(np.percentile(vis, 2)), float(np.percentile(vis, 98))
    if hi <= lo:
        hi = lo + 1e-6
    vis_norm = np.clip((vis - lo) / (hi - lo), 0.0, 1.0)
    # Inferno-ish colormap: black -> dark red -> orange -> yellow
    r = np.clip(vis_norm * 2.0,        0, 1)
    g = np.clip(vis_norm * 2.0 - 0.5,  0, 1)
    b = np.clip(vis_norm * 2.0 - 1.5,  0, 1)
    vis_rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    vis_path = out_dir / "error_map_vis.png"
    Image.fromarray(vis_rgb).save(str(vis_path))
    print(f"Saved visualisation: {vis_path}")

    # Stats
    stats = {
        "n_frames_used": int(len(pred_paths) - n_skipped),
        "n_frames_skipped": int(n_skipped),
        "error_map_shape_full": [int(H_full), int(W_full)],
        "error_map_shape_rl": [N, N],
        "noise_floor": args.noise_floor,
        "global_mean_relative_error": float(global_mean),
        "percentiles": {
            "p10": float(np.percentile(error_map_full, 10)),
            "p25": float(np.percentile(error_map_full, 25)),
            "p50": float(np.percentile(error_map_full, 50)),
            "p75": float(np.percentile(error_map_full, 75)),
            "p90": float(np.percentile(error_map_full, 90)),
        },
    }
    stats_path = out_dir / "error_map_stats.json"
    with open(str(stats_path), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats: {stats_path}")

    print(
        f"\nDone.\n"
        f"  Global mean relative error : {global_mean*100:.1f}%\n"
        f"  p50 (median)               : {stats['percentiles']['p50']*100:.1f}%\n"
        f"  p90                        : {stats['percentiles']['p90']*100:.1f}%\n"
        f"\nNext steps:\n"
        f"  1. Inspect error_map_vis.png — bright regions = high CNN uncertainty\n"
        f"  2. Copy error_map_{N}.npy to navigation/v7_shaft/\n"
        f"  3. The RL env loads it automatically at __init__\n"
    )


if __name__ == "__main__":
    main()

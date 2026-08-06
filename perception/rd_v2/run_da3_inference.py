"""
Zero-shot Depth Anything 3 (SMALL) inference on the realistic colon dataset.

What this script does:
  - Loads RGB frames written by blender_depth_dataset.py.
  - Runs the pretrained depth-anything/DA3-SMALL model on each frame
    (one image per inference call -- DA3 treats a batch as multi-view of
    a single scene, so independent mono-depth per frame must be called
    one at a time).
  - Saves predicted depth + confidence maps as .npy.
  - Optionally writes side-by-side RGB | predicted-depth | GT-depth PNGs.
  - Compares prediction vs ground truth and writes summary.json:
    raw MAE, scale-aligned MAE, and median pred/GT ratio. The ratio answers
    "is DA3 output already metric, or relative inverse depth?".

Install (first run only):
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3
    cd Depth-Anything-3
    pip install -e .
    pip install transformers

Usage:
    python run_da3_inference.py --limit 10 --save_vis
    python run_da3_inference.py            # process all frames in dataset/
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from PIL import Image


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_dir', default=os.path.join(here, 'dataset_finetune'),
                   help='Folder containing rgb/ and depth/ subfolders.')
    p.add_argument('--output_dir', default=os.path.join(here, 'predictions'),
                   help='Where predicted depth/conf/vis/summary go.')
    p.add_argument('--model', default='depth-anything/DA3-SMALL',
                   help='HuggingFace model repo id.')
    p.add_argument('--checkpoint', default=None,
                   help='Optional path to a fine-tuned checkpoint (.pth) to load '
                        'on top of the pretrained weights.')
    p.add_argument('--limit', type=int, default=0,
                   help='Process only the first N frames (0 = all).')
    p.add_argument('--device', default='auto',
                   help='cuda | cpu | auto.')
    p.add_argument('--save_vis', action='store_true',
                   help='Write RGB | pred | GT side-by-side PNGs.')
    return p.parse_args()


def resolve_device(arg):
    import torch
    if arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return arg


def load_model(model_name, device, checkpoint_path=None):
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as e:
        sys.exit(
            f"depth_anything_3 is not installed.\n"
            f"  git clone https://github.com/ByteDance-Seed/Depth-Anything-3\n"
            f"  cd Depth-Anything-3 && pip install -e .\n"
            f"Original error: {e}"
        )
    import torch
    print(f"Loading {model_name} on {device}...")
    t0 = time.time()
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)
    if checkpoint_path is not None:
        print(f"  applying checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        if isinstance(ckpt, dict) and 'val_scale_aligned_mae_m' in ckpt:
            print(f"  checkpoint reports val MAE = {ckpt['val_scale_aligned_mae_m']*1000:.2f} mm "
                  f"at epoch {ckpt.get('epoch', '?')}")
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")
    return model


def predict_one(model, image_path):
    """Run DA3 on a single image, returning (depth, conf) as (H, W) float32.

    DA3's inference() takes a list of images. For independent mono-depth per
    frame we pass exactly one image so the model doesn't try to fuse views.

    DA3 resizes inputs internally to a multiple of its patch size, so the raw
    prediction shape doesn't match the input. We resize back to the input
    image's (H, W) so saved .npy, metrics, and visualisations all line up.
    """
    pred = model.inference([image_path])
    # pred.depth: [N, H, W] -- N==1 here.
    depth = np.asarray(pred.depth[0], dtype=np.float32)
    conf = None
    if hasattr(pred, 'conf') and pred.conf is not None:
        conf = np.asarray(pred.conf[0], dtype=np.float32)

    with Image.open(image_path) as im:
        W_in, H_in = im.size
    if depth.shape != (H_in, W_in):
        depth = np.asarray(
            Image.fromarray(depth).resize((W_in, H_in), Image.BILINEAR),
            dtype=np.float32,
        )
        if conf is not None:
            conf = np.asarray(
                Image.fromarray(conf).resize((W_in, H_in), Image.BILINEAR),
                dtype=np.float32,
            )
    return depth, conf


def save_side_by_side(out_path, rgb_path, pred_depth, gt_depth):
    """Stack RGB | pred-depth | GT-depth horizontally for visual sanity check.

    Both depth maps are normalised to their own valid-range max for display.
    Invalid GT pixels (0) and invalid pred pixels (nan / 0) shown as black.
    """
    rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
    H, W = rgb.shape[:2]

    def colorize(depth):
        d = depth.copy()
        if d.shape != (H, W):
            d = np.asarray(Image.fromarray(d).resize((W, H), Image.BILINEAR),
                           dtype=np.float32)
        valid = (d > 0) & np.isfinite(d)
        out = np.zeros((H, W, 3), dtype=np.uint8)
        if valid.any():
            dv = d[valid]
            lo, hi = float(np.percentile(dv, 2)), float(np.percentile(dv, 98))
            if hi <= lo:
                hi = lo + 1e-6
            norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
            # Simple inferno-ish ramp: black -> red -> yellow -> white.
            r = np.clip(norm * 2.0, 0, 1)
            g = np.clip(norm * 2.0 - 0.5, 0, 1)
            b = np.clip(norm * 2.0 - 1.5, 0, 1)
            rgb_d = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
            rgb_d[~valid] = 0
            out = rgb_d
        return out

    panel = np.concatenate([rgb, colorize(pred_depth), colorize(gt_depth)], axis=1)
    Image.fromarray(panel).save(out_path)


def compute_metrics(pred, gt):
    """Compare pred vs GT only where GT is valid (>0)."""
    valid = (gt > 0) & np.isfinite(pred)
    if not valid.any():
        return None
    p = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)
    raw_mae = float(np.mean(np.abs(p - g)))
    raw_rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    # Scale-align by median ratio: handles relative-vs-metric mismatch.
    p_pos = p[p > 0]
    g_pos = g[p > 0]
    if len(p_pos) > 0:
        ratio = float(np.median(g_pos / p_pos))
    else:
        ratio = float('nan')
    aligned_mae = float(np.mean(np.abs(p * ratio - g))) if np.isfinite(ratio) else float('nan')
    return {
        'raw_mae_m': raw_mae,
        'raw_rmse_m': raw_rmse,
        'median_gt_over_pred': ratio,
        'scale_aligned_mae_m': aligned_mae,
        'valid_pixels': int(valid.sum()),
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)

    rgb_dir = os.path.join(args.dataset_dir, 'rgb')
    gt_dir = os.path.join(args.dataset_dir, 'depth')
    if not os.path.isdir(rgb_dir):
        sys.exit(f"RGB folder not found: {rgb_dir}")

    out_depth = os.path.join(args.output_dir, 'depth_pred')
    out_conf = os.path.join(args.output_dir, 'conf')
    out_vis = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(out_depth, exist_ok=True)
    os.makedirs(out_conf, exist_ok=True)
    if args.save_vis:
        os.makedirs(out_vis, exist_ok=True)

    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, 'frame_*.png')))
    if args.limit > 0:
        rgb_paths = rgb_paths[:args.limit]
    if not rgb_paths:
        sys.exit(f"No frames matched {rgb_dir}/frame_*.png")

    model = load_model(args.model, device, checkpoint_path=args.checkpoint)

    print(f"Running inference on {len(rgb_paths)} frames -> {args.output_dir}")
    per_frame = []
    t_start = time.time()
    for i, rgb_path in enumerate(rgb_paths):
        stem = os.path.splitext(os.path.basename(rgb_path))[0]      # frame_00000
        idx = stem.split('_')[-1]                                   # 00000

        pred_depth, conf = predict_one(model, rgb_path)
        np.save(os.path.join(out_depth, f'depth_{idx}.npy'), pred_depth)
        if conf is not None:
            np.save(os.path.join(out_conf, f'conf_{idx}.npy'), conf)

        gt_path = os.path.join(gt_dir, f'depth_{idx}.npy')
        if os.path.exists(gt_path):
            gt = np.load(gt_path)
            metrics = compute_metrics(pred_depth, gt)
            if metrics is not None:
                metrics['frame'] = int(idx)
                per_frame.append(metrics)
            if args.save_vis:
                save_side_by_side(
                    os.path.join(out_vis, f'vis_{idx}.png'),
                    rgb_path, pred_depth, gt,
                )

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(rgb_paths) - i - 1) / fps if fps > 0 else 0
            print(f"  {i+1}/{len(rgb_paths)}  ({fps:.2f} fps, ETA {eta/60:.1f} min)")

    # Aggregate summary.
    summary = {
        'model': args.model,
        'device': device,
        'n_frames': len(rgb_paths),
        'total_seconds': time.time() - t_start,
    }
    if per_frame:
        ratios = np.array([m['median_gt_over_pred'] for m in per_frame
                           if np.isfinite(m['median_gt_over_pred'])])
        raw_maes = np.array([m['raw_mae_m'] for m in per_frame])
        aln_maes = np.array([m['scale_aligned_mae_m'] for m in per_frame
                             if np.isfinite(m['scale_aligned_mae_m'])])
        summary['mean_raw_mae_m'] = float(raw_maes.mean()) if len(raw_maes) else None
        summary['mean_scale_aligned_mae_m'] = float(aln_maes.mean()) if len(aln_maes) else None
        summary['median_gt_over_pred_overall'] = float(np.median(ratios)) if len(ratios) else None
        summary['per_frame'] = per_frame

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Summary -> {args.output_dir}/summary.json")
    if 'median_gt_over_pred_overall' in summary and summary['median_gt_over_pred_overall']:
        ratio = summary['median_gt_over_pred_overall']
        print(f"  Median GT/pred ratio = {ratio:.4f}")
        if 0.5 < ratio < 2.0:
            print("  -> prediction is approximately metric.")
        else:
            print("  -> prediction appears relative/inverse depth; "
                  "scale-align before use (expected if metric_weight=0).")


if __name__ == '__main__':
    main()

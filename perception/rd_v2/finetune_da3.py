"""
Fine-tune DA3-SMALL on the realistic Blender colon dataset.

What this script does:
  - Loads DA3-SMALL pretrained weights from HuggingFace.
  - Freezes the DinoV2 backbone and camera encoder/decoder.
  - Trains only the DualDPT head (depth + ray branches; we read only the
    depth branch, ray branch receives no gradient and stays at pretrained
    values).
  - Loss: Eigen-style scale-invariant log loss (SI-only by default, matching
    DA3's zero-shot relative-depth output). Pass --metric_weight >0 to add
    an L1-in-metres term and pull toward absolute metric depth.
  - Validation: holds out geometry 5 (filename frame_01200..01499) and
    reports scale-aligned MAE in metres -- directly comparable to the
    zero-shot baseline number (7.1 mm).
  - Saves the best-by-val-MAE checkpoint to checkpoints/.

Phase 2 of the fine-tune plan (writing only). Running this is Phase 3
and needs separate approval -- it takes ~15-25 min on the RTX 4060.

Bypass note: DA3's public `forward()` is `@torch.inference_mode()`-decorated.
We call `model.model(...)` (the inner DepthAnything3Net) directly so gradients
flow.

Usage:
    python finetune_da3.py
    python finetune_da3.py --epochs 30 --batch_size 2 --metric_weight 0.05

Loading the checkpoint later for inference:
    ckpt = torch.load('checkpoints/da3small_finetune_head.pth')
    model = DepthAnything3.from_pretrained('depth-anything/DA3-SMALL')
    model.load_state_dict(ckpt['state_dict'])
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ------------------------- Config defaults --------------------------------

DEFAULT_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(DEFAULT_HERE, 'dataset_finetune')
DEFAULT_CKPT_DIR = os.path.join(DEFAULT_HERE, 'checkpoints')
DEFAULT_CKPT_NAME = 'da3small_finetune_head.pth'

# DA3 ViT patch size is 14. The pretrained model expects inputs sized to
# (504, H) where H is the nearest multiple of 14 to the aspect-preserved
# value. For 320x180 source: 504 x 280 (= 36 x 20 patches). Matches the
# preprocessing seen in run_da3_inference.py logs.
TARGET_W = 504
TARGET_H = 280

# Generator wrote 5 geoms * 6 scenarios * 50 frames = 300 frames per geometry,
# numbered 0..299, 300..599, ... in filename order. Geom 5 -> 1200..1499.
VAL_GEOM_START = 1200
VAL_GEOM_END = 1500  # exclusive

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ------------------------- Dataset ----------------------------------------

class ColonDepthDataset(Dataset):
    """Reads RGB + metric-depth pairs from the Blender-rendered dataset.

    RGB is resized to (TARGET_W, TARGET_H), ImageNet-normalised.
    Depth is resized via nearest neighbour to preserve the 0=invalid mask
    (bilinear would blend invalid pixels into valid neighbours).
    """

    def __init__(self, rgb_paths, depth_paths):
        assert len(rgb_paths) == len(depth_paths)
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_pil = Image.open(self.rgb_paths[idx]).convert('RGB')
        rgb_resized = rgb_pil.resize((TARGET_W, TARGET_H), Image.BILINEAR)
        rgb_arr = np.asarray(rgb_resized, dtype=np.float32) / 255.0
        rgb_arr = (rgb_arr - IMAGENET_MEAN) / IMAGENET_STD
        rgb_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1)  # (3, H, W)

        depth_np = np.load(self.depth_paths[idx]).astype(np.float32)
        depth_pil = Image.fromarray(depth_np)
        depth_resized = np.array(
            depth_pil.resize((TARGET_W, TARGET_H), Image.NEAREST),
            dtype=np.float32,
        )
        depth_tensor = torch.from_numpy(depth_resized)  # (H, W), metres, 0=invalid
        return rgb_tensor, depth_tensor


def split_train_val(dataset_dir):
    """Split rgb files by original filename number: geom 5 -> val."""
    rgb_dir = os.path.join(dataset_dir, 'rgb')
    depth_dir = os.path.join(dataset_dir, 'depth')
    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, 'frame_*.png')))

    train_rgb, train_depth, val_rgb, val_depth = [], [], [], []
    for p in rgb_paths:
        m = re.search(r'frame_(\d+)\.png$', os.path.basename(p))
        if not m:
            continue
        idx = int(m.group(1))
        d = os.path.join(depth_dir, f'depth_{m.group(1)}.npy')
        if not os.path.exists(d):
            continue
        if VAL_GEOM_START <= idx < VAL_GEOM_END:
            val_rgb.append(p)
            val_depth.append(d)
        else:
            train_rgb.append(p)
            train_depth.append(d)
    return (train_rgb, train_depth), (val_rgb, val_depth)


# ------------------------- Loss -------------------------------------------

def mixed_loss(pred, gt, valid_mask, metric_weight):
    """Eigen-style scale-invariant log loss + small L1 in metres.

    SI term: well-established shape-only loss. Differences in log space,
    with variance term to remove the global shift (= scale).
    Metric term: direct |pred - gt|, drags the global scale toward metres.
    """
    losses_si, losses_met = [], []
    for b in range(pred.shape[0]):
        v = valid_mask[b]
        if v.sum() < 10:
            continue
        p_v = pred[b][v].clamp(min=1e-6)
        g_v = gt[b][v].clamp(min=1e-6)
        log_diff = torch.log(p_v) - torch.log(g_v)
        # Eigen SI: lam=0.85 is the original paper's setting.
        si = (log_diff ** 2).mean() - 0.85 * (log_diff.mean() ** 2)
        met = torch.abs(p_v - g_v).mean()
        losses_si.append(si)
        losses_met.append(met)
    if not losses_si:
        return None, None, None
    si_loss = torch.stack(losses_si).mean()
    met_loss = torch.stack(losses_met).mean()
    return si_loss + metric_weight * met_loss, si_loss, met_loss


# ------------------------- Eval -------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    """Per-sample scale-aligned MAE in metres. Matches the metric reported by
    run_da3_inference.py so val numbers are directly comparable to the
    zero-shot baseline (mean_scale_aligned_mae_m = 7.1 mm)."""
    model.eval()
    maes = []
    for rgb, gt in loader:
        rgb = rgb.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            image_5d = rgb.unsqueeze(1)
            out = model.model(
                image_5d, extrinsics=None, intrinsics=None,
                export_feat_layers=[], infer_gs=False,
                use_ray_pose=False, ref_view_strategy="saddle_balanced",
            )
            pred = out['depth'].squeeze(1).float()
        for b in range(pred.shape[0]):
            v = gt[b] > 0
            if v.sum() < 10:
                continue
            p = pred[b][v].clamp(min=1e-6).double()
            g = gt[b][v].double()
            ratio = (g / p).median()
            maes.append(torch.abs(ratio * p - g).mean().item())
    return float(np.mean(maes)) if maes else float('nan')


# ------------------------- Main -------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_dir', default=DEFAULT_DATASET)
    p.add_argument('--ckpt_dir', default=DEFAULT_CKPT_DIR)
    p.add_argument('--ckpt_name', default=DEFAULT_CKPT_NAME)
    p.add_argument('--model', default='depth-anything/DA3-SMALL')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--metric_weight', type=float, default=0.0,
                   help='Weight on the L1-in-metres term (Option B mixed loss). '
                        '0 = pure scale-invariant (Option A fallback).')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--print_every', type=int, default=20)
    p.add_argument('--grad_clip', type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    # Line-buffer stdout so progress prints appear in real time when the
    # script is invoked in the background with stdout piped to a file.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass  # Python <3.7 or unusual stdout; just live with block-buffering.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f'  {props.name}, {props.total_memory / 1e9:.2f} GB')

    # Dataset split ---------------------------------------------------------
    (tr_rgb, tr_d), (va_rgb, va_d) = split_train_val(args.dataset_dir)
    print(f'Train: {len(tr_rgb)} frames   Val: {len(va_rgb)} frames')
    if len(tr_rgb) < 1 or len(va_rgb) < 1:
        sys.exit('Empty train or val split.')

    train_loader = DataLoader(
        ColonDepthDataset(tr_rgb, tr_d),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ColonDepthDataset(va_rgb, va_d),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model -----------------------------------------------------------------
    from depth_anything_3.api import DepthAnything3
    print(f'Loading {args.model} ...')
    model = DepthAnything3.from_pretrained(args.model).to(device)

    # Head-only fine-tune: freeze backbone + cam_enc + cam_dec
    for p in model.model.backbone.parameters():
        p.requires_grad = False
    if model.model.cam_enc is not None:
        for p in model.model.cam_enc.parameters():
            p.requires_grad = False
    if model.model.cam_dec is not None:
        for p in model.model.cam_dec.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {n_trainable:,} / {n_total:,} '
          f'({100*n_trainable/n_total:.1f}%)')

    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    log_path = os.path.join(args.ckpt_dir, args.ckpt_name.replace('.pth', '_log.json'))

    # Baseline val (zero-shot, before any training) -------------------------
    val_mae_0 = evaluate(model, val_loader, device)
    print(f'\nBaseline (zero-shot) val_scale_aligned_mae = {val_mae_0*1000:.2f} mm')
    print(f'(For reference: 7.1 mm reported on the 120-frame pilot)\n')

    # Training loop ---------------------------------------------------------
    best_val_mae = float('inf')
    log = [{'epoch': 0, 'val_scale_aligned_mae_m': val_mae_0, 'note': 'baseline'}]
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = {'total': 0.0, 'si': 0.0, 'metric': 0.0, 'n': 0}
        for batch_i, (rgb, gt) in enumerate(train_loader):
            rgb = rgb.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            valid = gt > 0
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                image_5d = rgb.unsqueeze(1)
                out = model.model(
                    image_5d, extrinsics=None, intrinsics=None,
                    export_feat_layers=[], infer_gs=False,
                    use_ray_pose=False, ref_view_strategy="saddle_balanced",
                )
                pred = out['depth'].squeeze(1).float()
                loss, si, metric = mixed_loss(pred, gt, valid, args.metric_weight)
            if loss is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()

            running['total'] += loss.item()
            running['si'] += si.item()
            running['metric'] += metric.item()
            running['n'] += 1

            if (batch_i + 1) % args.print_every == 0:
                n = running['n']
                print(f'  ep {epoch:02d} batch {batch_i+1:04d}/{len(train_loader)}  '
                      f'total={running["total"]/n:.4f}  '
                      f'si={running["si"]/n:.4f}  '
                      f'metric_l1={running["metric"]/n:.4f}')

        train_total = running['total'] / max(running['n'], 1)
        val_mae = evaluate(model, val_loader, device)
        epoch_t = time.time() - t0
        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_scale_aligned_mae_m': val_mae,
                'args': vars(args),
            }, ckpt_path)

        print(f'epoch {epoch:02d}/{args.epochs}  '
              f'train_total={train_total:.4f}  '
              f'val_scale_aligned_mae={val_mae*1000:.2f} mm  '
              f'epoch_time={epoch_t/60:.1f} min  '
              f'{"BEST -> saved" if is_best else ""}')

        log.append({
            'epoch': epoch,
            'train_total_loss': train_total,
            'train_si_loss': running['si'] / max(running['n'], 1),
            'train_metric_l1': running['metric'] / max(running['n'], 1),
            'val_scale_aligned_mae_m': val_mae,
            'epoch_time_sec': epoch_t,
            'is_best': bool(is_best),
        })
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)

    total_t = time.time() - t_start
    print(f'\nDone in {total_t/60:.1f} min.')
    print(f'Baseline (zero-shot) val MAE: {val_mae_0*1000:.2f} mm')
    print(f'Best val MAE:                 {best_val_mae*1000:.2f} mm')
    print(f'Checkpoint: {ckpt_path}')


if __name__ == '__main__':
    main()

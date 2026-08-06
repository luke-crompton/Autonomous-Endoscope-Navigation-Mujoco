"""
filter_void_frames.py -- drop dataset frames with too many pure-black pixels.

Scans dataset_finetune/rgb/ for the fraction of pure-black pixels per frame
(black happens where camera rays missed the colon mesh and saw the
strength=0 world background). Frames above the threshold get moved to
dataset_finetune/_rejected/{rgb,depth}/ and poses.npy is re-saved with
the same rows dropped.

Position invariant after filtering:
    poses.npy[i] aligns with the i-th file in
    sorted(glob('rgb/frame_*.png'))
NOT with the filename number -- file numbers will have gaps where
rejects were removed. A poses.npy.before_filter backup is kept once.

Default behaviour is dry-run (prints what it would do, touches nothing).
Pass --apply to actually move files and re-save poses.npy.

Usage:
    python filter_void_frames.py                 # dry-run, 10% threshold
    python filter_void_frames.py --threshold 0.05
    python filter_void_frames.py --apply         # actually filter
"""

import argparse
import json
import os
import shutil
import sys
from glob import glob

import numpy as np
from PIL import Image


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_dir',
                   default=os.path.join(here, 'dataset_finetune'),
                   help='Folder containing rgb/, depth/, poses.npy, metadata.json.')
    p.add_argument('--threshold', type=float, default=0.10,
                   help='Fraction of pure-black pixels above which a frame is rejected (default 0.10).')
    p.add_argument('--apply', action='store_true',
                   help='Actually move files and re-save poses.npy. Without this, only prints what would happen.')
    return p.parse_args()


def black_fraction(rgb_path):
    a = np.asarray(Image.open(rgb_path).convert('RGB'))
    return float(((a[:, :, 0] == 0) & (a[:, :, 1] == 0) & (a[:, :, 2] == 0)).mean())


def main():
    args = parse_args()
    rgb_dir = os.path.join(args.dataset_dir, 'rgb')
    depth_dir = os.path.join(args.dataset_dir, 'depth')
    poses_path = os.path.join(args.dataset_dir, 'poses.npy')
    meta_path = os.path.join(args.dataset_dir, 'metadata.json')

    rgb_paths = sorted(glob(os.path.join(rgb_dir, 'frame_*.png')))
    if not rgb_paths:
        sys.exit(f'No frames matched {rgb_dir}/frame_*.png')

    print(f'Scanning {len(rgb_paths)} frames in {args.dataset_dir} ...')
    fractions = np.array([black_fraction(p) for p in rgb_paths])

    reject_mask = fractions > args.threshold
    n_reject = int(reject_mask.sum())
    n_keep = len(rgb_paths) - n_reject

    print()
    print(f'Threshold:    > {args.threshold*100:.1f}% pure-black')
    print(f'Would reject: {n_reject} frames ({100*n_reject/len(rgb_paths):.1f}%)')
    print(f'Would keep:   {n_keep} frames')

    rejected_idx = np.where(reject_mask)[0]
    if len(rejected_idx) > 0:
        print()
        print(f'Rejected frames (showing up to 25):')
        for i in rejected_idx[:25]:
            bn = os.path.basename(rgb_paths[i])
            print(f'  {bn}  ({fractions[i]*100:.1f}% black)')
        if len(rejected_idx) > 25:
            print(f'  ... and {len(rejected_idx)-25} more')

    if not args.apply:
        print()
        print('(dry-run; nothing moved. pass --apply to actually filter.)')
        return

    # Apply mode --------------------------------------------------------------
    rej_dir = os.path.join(args.dataset_dir, '_rejected')
    rej_rgb = os.path.join(rej_dir, 'rgb')
    rej_depth = os.path.join(rej_dir, 'depth')
    os.makedirs(rej_rgb, exist_ok=True)
    os.makedirs(rej_depth, exist_ok=True)

    moved_rgb = 0
    moved_depth = 0
    for i in rejected_idx:
        bn = os.path.basename(rgb_paths[i])
        idx = os.path.splitext(bn)[0].split('_')[-1]
        depth_bn = f'depth_{idx}.npy'
        shutil.move(os.path.join(rgb_dir, bn), os.path.join(rej_rgb, bn))
        moved_rgb += 1
        d_src = os.path.join(depth_dir, depth_bn)
        if os.path.exists(d_src):
            shutil.move(d_src, os.path.join(rej_depth, depth_bn))
            moved_depth += 1

    print()
    print(f'Moved {moved_rgb} RGB and {moved_depth} depth files to {rej_dir}/')

    # Re-save poses.npy with rejected rows dropped
    if os.path.exists(poses_path):
        poses = np.load(poses_path)
        if len(poses) != len(rgb_paths):
            print(f'WARN: poses.npy ({len(poses)}) and frames ({len(rgb_paths)}) length differ;')
            print('      not modifying poses.npy. You may need to fix this manually.')
        else:
            backup = poses_path + '.before_filter'
            if not os.path.exists(backup):
                shutil.copy2(poses_path, backup)
                print(f'Backed up original poses to {os.path.basename(backup)}')
            kept_poses = poses[~reject_mask]
            np.save(poses_path, kept_poses)
            print(f'Re-saved poses.npy: {len(poses)} -> {len(kept_poses)} entries')

    # Update metadata.json
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        meta['total_frames'] = int(n_keep)
        meta['filtered_void_threshold'] = float(args.threshold)
        meta['n_rejected_void'] = int(n_reject)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'Updated metadata.json: total_frames = {n_keep}, n_rejected_void = {n_reject}')

    print('\nDone.')


if __name__ == '__main__':
    main()

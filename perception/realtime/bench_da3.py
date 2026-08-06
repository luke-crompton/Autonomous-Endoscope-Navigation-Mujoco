"""
Latency benchmark for the fine-tuned DA3-SMALL depth model.

Replaces the project's undocumented "~20 FPS" with a number that has its
conditions attached. Times the four deployment stages separately --
preprocess (incl. host->device), forward, device->host, and the 64x64 nav-obs
downsample -- so we know which one to attack before optimising anything.

Reports MEDIAN and p95, not mean fps. A mean over a loop that includes a
cuDNN autotune spike or a compile pause is not a latency you will ever see.

Two passes are run per configuration:
  - attributed: a CUDA sync after every stage, so each stage's cost is real.
    The syncs themselves cost a little, so the total is slightly pessimistic.
  - end-to-end: one sync per frame. This is the honest per-frame latency.

Usage -- just Run it; no arguments needed, and it relaunches itself under
Python 3.11 if the IDE is pointed at a different interpreter.
    python bench_da3.py                             # bf16, synthetic frames
    python bench_da3.py --source camera             # real frames off the scope
    python bench_da3.py --source dataset --frames 200
    python bench_da3.py --sweep                     # fp32 vs bf16 vs fp16
    python bench_da3.py --verify                    # accuracy sanity check
"""

import argparse
import glob
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _py311  # noqa: E402,F401  -- relaunches under Python 3.11 if needed

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from da3_runtime import (  # noqa: E402
    Da3Depth, DEFAULT_CHECKPOINT, SRC_W, SRC_H, TARGET_W, TARGET_H, sync,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RD_V2 = os.path.join(os.path.dirname(HERE), 'rd_v2')
VAL_START, VAL_END = 1200, 1500     # held-out geometry, per finetune_da3.py


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                   help="Fine-tuned .pth. Pass '' for zero-shot.")
    p.add_argument('--source', default='synthetic',
                   choices=['synthetic', 'camera', 'dataset', 'dir'],
                   help='synthetic = random frames at --cam_w/--cam_h; '
                        'camera = grab real frames first, then time on those; '
                        'dataset = rd_v2/dataset_finetune/rgb; dir = --frames_dir.')
    p.add_argument('--frames_dir', default=None)
    p.add_argument('--cam_index', type=int, default=1,
                   help='--source camera: which camera to grab from. '
                        'Default: 1 (the scope).')
    p.add_argument('--cam_backend', default='auto',
                   choices=['auto', 'dshow', 'msmf', 'any'])
    p.add_argument('--cam_w', type=int, default=1280,
                   help='Frame width: the synthetic frame size, or the mode '
                        'requested from the camera. Set it to what the camera '
                        'really delivers so preprocess cost is honest.')
    p.add_argument('--cam_h', type=int, default=720)
    p.add_argument('--frames', type=int, default=100, help='Timed frames.')
    p.add_argument('--warmup', type=int, default=20,
                   help='Untimed frames first. Covers cuDNN autotune and, if '
                        '--compile is set, the one-off graph compile.')
    p.add_argument('--precision', default='bf16', choices=['fp32', 'bf16', 'fp16'])
    p.add_argument('--sweep', action='store_true',
                   help='Benchmark fp32, bf16 and fp16 back to back.')
    p.add_argument('--channels_last', action='store_true')
    p.add_argument('--compile', action='store_true',
                   help='torch.compile the inner net. Raise --warmup with this.')
    p.add_argument('--no_train_blur', action='store_true',
                   help='Skip the 320x180 bottleneck that reproduces the '
                        "fine-tune data's sharpness budget.")
    p.add_argument('--device', default='auto')
    p.add_argument('--verify', action='store_true',
                   help='Also compute scale-aligned MAE on held-out val frames '
                        'and compare to the 1.43 mm the checkpoint reports. '
                        'Needs rd_v2/dataset_finetune, which is NOT in the repo '
                        '-- regenerate with blender_depth_dataset.py if wanted.')
    p.add_argument('--verify_frames', type=int, default=50)
    p.add_argument('--out', default=os.path.join(HERE, 'bench_results.json'))
    return p.parse_args()


# ------------------------- Frame sources ----------------------------------

def load_frames(args, n):
    """Return a list of (H, W, 3) uint8 RGB arrays, cycled to length n.

    Frames are held in RAM as arrays so disk I/O and PNG decode never land
    inside the timed region -- the live camera will not pay them either.
    """
    if args.source == 'synthetic':
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, (args.cam_h, args.cam_w, 3), dtype=np.uint8)
        return [base] * n

    if args.source == 'camera':
        # Grab real frames up front, then time on them. Capture is deliberately
        # OUTSIDE the timed region -- this benchmark measures the depth
        # pipeline, and mixing in the camera's own frame interval would cap
        # the result at the camera's fps regardless of how fast DA3 is.
        # probe_camera.py measures capture separately; preview_camera_depth.py
        # measures them combined.
        from probe_camera import open_camera, require_cv2
        cv2 = require_cv2()
        cap, backend = open_camera(cv2, args.cam_index, args.cam_backend,
                                   args.cam_w, args.cam_h)
        if cap is None:
            sys.exit(f'Could not open camera index {args.cam_index}.')
        print(f'  grabbing {n} frames from camera {args.cam_index} [{backend}]...')
        for _ in range(10):          # let auto-exposure settle
            cap.read()
        frames = []
        while len(frames) < n:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(np.ascontiguousarray(frame[:, :, ::-1]))   # BGR -> RGB
        cap.release()
        if not frames:
            sys.exit('Camera opened but delivered no frames.')
        while len(frames) < n:       # short read: cycle what we got
            frames.extend(frames[:n - len(frames)])
        return frames[:n]

    if args.source == 'dataset':
        pattern = os.path.join(RD_V2, 'dataset_finetune', 'rgb', 'frame_*.png')
    else:
        if not args.frames_dir:
            sys.exit('--source dir needs --frames_dir')
        pattern = os.path.join(args.frames_dir, '*.png')

    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f'No frames matched {pattern}')
    paths = paths[:max(n, 1)]
    frames = [np.asarray(Image.open(p).convert('RGB')) for p in paths]
    while len(frames) < n:
        frames.extend(frames[:n - len(frames)])
    return frames[:n]


# ------------------------- Timing -----------------------------------------

def stats(samples_ms):
    a = np.asarray(samples_ms, dtype=np.float64)
    return {
        'median_ms': round(float(np.median(a)), 3),
        'p95_ms': round(float(np.percentile(a, 95)), 3),
        'mean_ms': round(float(a.mean()), 3),
        'min_ms': round(float(a.min()), 3),
        'max_ms': round(float(a.max()), 3),
    }


def bench_attributed(engine, frames, warmup):
    """Per-stage timing. A sync after each stage, so nothing hides."""
    dev = engine.device
    for f in frames[:warmup]:
        engine.to_obs_64(engine(f))
    sync(dev)

    pre, fwd, d2h, obs = [], [], [], []
    for f in frames:
        sync(dev)
        t0 = time.perf_counter()
        x = engine.preprocess(f)
        sync(dev)
        t1 = time.perf_counter()
        d = engine.forward(x)
        sync(dev)
        t2 = time.perf_counter()
        dm = engine.postprocess(d)
        t3 = time.perf_counter()
        engine.to_obs_64(dm)
        t4 = time.perf_counter()

        pre.append((t1 - t0) * 1e3)
        fwd.append((t2 - t1) * 1e3)
        d2h.append((t3 - t2) * 1e3)
        obs.append((t4 - t3) * 1e3)

    total = [a + b + c + d for a, b, c, d in zip(pre, fwd, d2h, obs)]
    return {
        'preprocess_h2d': stats(pre),
        'forward': stats(fwd),
        'device_to_host': stats(d2h),
        'obs_64': stats(obs),
        'total_attributed': stats(total),
    }


def bench_end_to_end(engine, frames, warmup):
    """One sync per frame -- the latency you would actually observe."""
    dev = engine.device
    for f in frames[:warmup]:
        engine.to_obs_64(engine(f))
    sync(dev)

    per_frame = []
    for f in frames:
        t0 = time.perf_counter()
        engine.to_obs_64(engine(f))
        sync(dev)
        per_frame.append((time.perf_counter() - t0) * 1e3)

    s = stats(per_frame)
    s['fps_at_median'] = round(1000.0 / s['median_ms'], 2)
    s['fps_at_p95'] = round(1000.0 / s['p95_ms'], 2)
    return s


# ------------------------- Accuracy check ---------------------------------

def verify(engine, n_frames):
    """Scale-aligned MAE on held-out val frames, matching finetune_da3.evaluate.

    Proves this runtime's preprocessing is equivalent to the training one
    before any real-camera number measured through it is trusted.
    """
    rgb_dir = os.path.join(RD_V2, 'dataset_finetune', 'rgb')
    depth_dir = os.path.join(RD_V2, 'dataset_finetune', 'depth')
    if not os.path.isdir(rgb_dir):
        return {'error': f'dataset not found at {rgb_dir}'}

    maes = []
    for idx in range(VAL_START, min(VAL_END, VAL_START + n_frames)):
        rgb_p = os.path.join(rgb_dir, f'frame_{idx:05d}.png')
        gt_p = os.path.join(depth_dir, f'depth_{idx:05d}.npy')
        if not (os.path.exists(rgb_p) and os.path.exists(gt_p)):
            continue
        pred = engine(rgb_p)
        gt = np.load(gt_p).astype(np.float32)
        if gt.shape != (TARGET_H, TARGET_W):
            # NEAREST, so the 0=invalid mask is not blended into valid pixels.
            gt = np.asarray(
                Image.fromarray(gt).resize((TARGET_W, TARGET_H), Image.NEAREST),
                dtype=np.float32,
            )
        v = gt > 0
        if v.sum() < 10:
            continue
        p = np.clip(pred[v], 1e-6, None).astype(np.float64)
        g = gt[v].astype(np.float64)
        ratio = float(np.median(g / p))
        maes.append(float(np.mean(np.abs(ratio * p - g))))

    if not maes:
        return {'error': 'no val frames with GT found'}
    return {
        'n_frames': len(maes),
        'scale_aligned_mae_mm': round(float(np.mean(maes)) * 1000, 3),
        'reference_ckpt_val_mae_mm': round(engine.ckpt_val_mae_mm, 3),
    }


# ------------------------- Main -------------------------------------------

def run_one(args, precision, frames):
    print(f'\n=== precision={precision} '
          f'channels_last={args.channels_last} compile={args.compile} ===')
    engine = Da3Depth(
        checkpoint=args.checkpoint or None,
        device=args.device,
        precision=precision,
        channels_last=args.channels_last,
        compile_model=args.compile,
        match_train_blur=not args.no_train_blur,
    )
    info = engine.describe()
    print(f"  {info['gpu']}  torch {info['torch']}  "
          f"model loaded in {info['load_seconds']}s")

    print(f'  warm-up {args.warmup}, timing {len(frames)} frames...')
    attributed = bench_attributed(engine, frames, args.warmup)
    e2e = bench_end_to_end(engine, frames, args.warmup)

    print(f"  {'stage':<20}{'median':>10}{'p95':>10}")
    for k, v in attributed.items():
        print(f"  {k:<20}{v['median_ms']:>9.2f}{v['p95_ms']:>10.2f}")
    print(f"  {'END-TO-END':<20}{e2e['median_ms']:>9.2f}{e2e['p95_ms']:>10.2f}"
          f"   -> {e2e['fps_at_median']} fps")

    result = {'config': info, 'attributed': attributed, 'end_to_end': e2e}

    if args.verify:
        print(f'  verifying accuracy on {args.verify_frames} val frames...')
        result['verify'] = verify(engine, args.verify_frames)
        vr = result['verify']
        if 'error' in vr:
            print(f"    {vr['error']}")
        else:
            print(f"    scale-aligned MAE = {vr['scale_aligned_mae_mm']:.2f} mm "
                  f"(checkpoint reports {vr['reference_ckpt_val_mae_mm']:.2f} mm)")

    del engine
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def main():
    args = parse_args()
    n = args.frames
    frames = load_frames(args, n)
    src_hw = frames[0].shape[:2]

    print(f'Source: {args.source}  frame {src_hw[1]}x{src_hw[0]} '
          f'-> {"320x180 -> " if not args.no_train_blur else ""}'
          f'{TARGET_W}x{TARGET_H}')
    print(f'Checkpoint: {args.checkpoint or "(zero-shot)"}')

    precisions = ['fp32', 'bf16', 'fp16'] if args.sweep else [args.precision]
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'platform': platform.platform(),
        'python': sys.version.split()[0],
        'source': args.source,
        'source_frame_hw': list(src_hw),
        'train_blur_applied': not args.no_train_blur,
        'timed_frames': n,
        'warmup_frames': args.warmup,
        'runs': {},
    }
    for prec in precisions:
        results['runs'][prec] = run_one(args, prec, frames)

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nWritten -> {args.out}')

    # The number that matters for closed-loop control.
    best = min(results['runs'].values(),
               key=lambda r: r['end_to_end']['median_ms'])
    ms = best['end_to_end']['median_ms']
    print(f"\nBest median end-to-end: {ms:.2f} ms "
          f"({best['end_to_end']['fps_at_median']} fps, "
          f"{best['config']['precision']})")
    print(f'  The nav env steps at 10 ms (100 Hz) -- this is {ms/10:.1f}x that '
          f'budget.')


if __name__ == '__main__':
    main()

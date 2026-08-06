"""
Press Run to generate the CNN error map for v6 RL training.

Runs three steps in sequence:
  1. Blender renders 500 eval frames (new geometries the CNN never saw)
  2. DA3 inference on those frames using the fine-tuned checkpoint
  3. Per-pixel error analysis -> error_map_64.npy

Output: Perception/RD_V2/error_map_64.npy
After this finishes, copy that file to navigation/v7_shaft/.

Each step is skipped automatically if its output already exists,
so you can re-run safely if a step fails partway through.

Edit the constants below if any paths have changed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — edit these if anything has moved
# ---------------------------------------------------------------------------

HERE       = Path(__file__).parent
BLENDER    = Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")
PYTHON     = Path(r"C:\Users\lukec\AppData\Local\Programs\Python\Python311\python.exe")
CHECKPOINT = HERE / "checkpoints" / "da3small_finetune_head.pth"

DATASET_DIR     = HERE / "dataset_eval_500"
PREDICTIONS_DIR = HERE / "predictions_eval_500"
ERROR_MAP_OUT   = HERE  # error_map_64.npy lands here

# ---------------------------------------------------------------------------
# Render config
# ---------------------------------------------------------------------------

N_GEOMS         = 5    # number of unseen geometries
FRAMES_PER_GEOM = 100  # frames per geometry  -> 500 total (~6 min on RTX 4060 OPTIX)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(step: int, title: str) -> None:
    print(f"\n{'=' * 64}", flush=True)
    print(f"  STEP {step}/3 — {title}", flush=True)
    print(f"{'=' * 64}", flush=True)


def _run(cmd: list, step: int, title: str) -> None:
    _header(step, title)
    print("  " + " ".join(str(c) for c in cmd), flush=True)
    print(flush=True)
    # Inherit stdout/stderr so output appears in real time in the IDE console.
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        print(
            f"\n[ERROR] Step {step} exited with code {result.returncode}. "
            "Fix the error above and re-run (completed steps will be skipped).",
            flush=True,
        )
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nError-map pipeline starting.", flush=True)
    print(f"  Blender  : {BLENDER}", flush=True)
    print(f"  Python   : {PYTHON}", flush=True)
    print(f"  Dataset  : {DATASET_DIR}", flush=True)
    print(f"  Preds    : {PREDICTIONS_DIR}", flush=True)
    print(f"  Output   : {ERROR_MAP_OUT}", flush=True)

    # ------------------------------------------------------------------
    # Step 1 — Blender render
    # Skip if the rgb/ folder already has the expected number of frames.
    # ------------------------------------------------------------------
    expected_frames = N_GEOMS * FRAMES_PER_GEOM
    existing_frames = list((DATASET_DIR / "rgb").glob("frame_*.png")) if (DATASET_DIR / "rgb").exists() else []

    if len(existing_frames) >= expected_frames:
        _header(1, "Blender render — SKIPPED (frames already exist)")
        print(f"  Found {len(existing_frames)} frames in {DATASET_DIR / 'rgb'}", flush=True)
    else:
        _run(
            [
                BLENDER, "--background",
                "--python", HERE / "blender_depth_dataset_eval.py",
                "--",
                "--n_geoms",         str(N_GEOMS),
                "--frames_per_geom", str(FRAMES_PER_GEOM),
                "--out_dir",         DATASET_DIR,
            ],
            step=1,
            title=f"Blender render — {expected_frames} frames into {DATASET_DIR.name}/",
        )

    # ------------------------------------------------------------------
    # Step 2 — DA3 inference
    # Skip if summary.json already exists in the predictions dir.
    # ------------------------------------------------------------------
    summary_path = PREDICTIONS_DIR / "summary.json"
    if summary_path.exists():
        _header(2, "DA3 inference — SKIPPED (predictions already exist)")
        print(f"  Found {summary_path}", flush=True)
    else:
        _run(
            [
                PYTHON, "-u",
                HERE / "run_da3_inference.py",
                "--dataset_dir", DATASET_DIR,
                "--output_dir",  PREDICTIONS_DIR,
                "--checkpoint",  CHECKPOINT,
            ],
            step=2,
            title=f"DA3 inference -> {PREDICTIONS_DIR.name}/",
        )

    # ------------------------------------------------------------------
    # Step 3 — Compute error map
    # Skip if error_map_64.npy already exists.
    # ------------------------------------------------------------------
    error_map_path = ERROR_MAP_OUT / "error_map_64.npy"
    if error_map_path.exists():
        _header(3, "Error map — SKIPPED (already exists)")
        print(f"  Found {error_map_path}", flush=True)
    else:
        _run(
            [
                PYTHON, "-u",
                HERE / "compute_error_map.py",
                "--pred_dir", PREDICTIONS_DIR,
                "--gt_dir",   DATASET_DIR / "depth",
                "--out_dir",  ERROR_MAP_OUT,
            ],
            step=3,
            title="Compute per-pixel error map",
        )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'=' * 64}", flush=True)
    print("  ALL STEPS DONE", flush=True)
    print(f"{'=' * 64}", flush=True)
    print(f"\n  error_map_64.npy  ->  {ERROR_MAP_OUT / 'error_map_64.npy'}", flush=True)
    print(f"  error_map_vis.png ->  {ERROR_MAP_OUT / 'error_map_vis.png'}", flush=True)
    print(f"\n  Next: copy error_map_64.npy to navigation/v7_shaft/", flush=True)
    print(f"  Then transfer v6_dr/ to Linux and run: python train_sf.py\n", flush=True)


if __name__ == "__main__":
    main()

> ## ⚠️ SUPERSEDED — HISTORICAL ARCHIVE, DO NOT FOLLOW
> This document describes the **old Mujuco_V2 layout** (`Stage4/vN/`, `transfer/`,
> `Perception/`). Those paths no longer exist in this project, and many of the datasets,
> checkpoints, and CNN weights it references have been deleted. **Do not use it for
> commands, file paths, or "how to run" instructions.**
> - To run something: see the root `README.md`.
> - For the project's history and design rationale: see `../ITERATION_HISTORY.md`.
>
> Kept verbatim only as a record of what was written at the time. Nothing here should be
> edited or acted on.

# User Guide — Mujuco_V2 (Simulated Videoscope Navigation)

A practical "how to run it / what each file does" guide for the repository. For the
*story* of what was built and why (results, design decisions), see `REPORT.md` and
`PROJECT_CONTEXT.md`. This guide is the operational complement.

**Authorship note.** Two parts of this repo were **provided by my supervisor** and are
marked *(supervisor-provided)* below — `Videoscopesimulations/` (the scope tip) and
`colon_realistic_geometry_handoff_2026-05-26/` (the photorealistic Blender colon).
Everything else (my procedural colon, the CNNs, the Stage 4 V1 loop, the DA3 fine-tune,
and the collision RL pipeline) is my own work.

---

## 1. Environment & prerequisites

This project uses **two different Python interpreters** — using the wrong one is the
single most common mistake.

| Purpose | Interpreter | Notes |
|---|---|---|
| **All training / inference / RL** | `C:\Users\lukec\AppData\Local\Programs\Python\Python311\python.exe` | Python 3.11.1. Has GPU `torch` (CUDA, RTX 4060), Stable-Baselines3, Gymnasium, Depth-Anything-3. **Use this for everything.** |
| (avoid) | `python` on PATH | Python 3.14, **CPU-only torch** — silently slow / wrong for training. Do not use. |

Other dependencies:

- **Blender 5.1** — `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`
  (use `blender.exe`, **not** `blender-launcher.exe`). Needed only for the
  photorealistic colon renders.
- **Depth-Anything-3 source** — editable install at
  `C:\Users\lukec\PycharmProjects\Depth-Anything-3`. Do not delete; the DA3 scripts
  import from it.
- **Pandoc** (optional) — installed; converts `REPORT.md`/this guide to HTML/Word if
  wanted.

> **Tip:** in the examples below, `$PY311` stands for the 3.11 interpreter. Set it once
> per PowerShell session:
> ```powershell
> $PY311 = "C:\Users\lukec\AppData\Local\Programs\Python\Python311\python.exe"
> ```

**Constraint:** Depth-Anything-3 pins Python `>=3.9, <=3.13`, so do not upgrade the 3.11
interpreter past 3.13 for DA3 work.

---

## 2. Datasets are not in the repo (regenerable)

To keep the project small, the large training datasets and inference outputs were
**deleted** — they are all regenerable from the scripts here. If a script complains a
folder is missing, regenerate it (see §4 workflows):

| Deleted folder | Regenerate with |
|---|---|
| `Perception/depth_regression/depth_dataset/` | `Depth_dataset.py` (workflow C) |
| `Perception/realistic_depth_field/dataset_finetune/` | `blender_depth_dataset.py` (workflow E) |
| `Perception/realistic_depth_field/dataset/` (pilot) | `blender_depth_dataset.py` |
| `Perception/realistic_depth_field/predictions*/` | `run_da3_inference.py` (workflow F) |

Trained models (`*.pth`, PPO `*.zip`) and run videos **are** kept.

---

## 3. Repository map

| Folder | What it is |
|---|---|
| `Perception/` | My procedural colon model + the perception CNNs + the DA3 depth work. |
| `Perception/depth_regression/` | "Deepest depth blob" steering CNN + its dataset generator. |
| `Perception/centreline_regression/` | Centreline-based steering CNNs, incl. `look_ahead_10mm/` (**current best**) and `look_ahead_15mm/` (superseded). |
| `Perception/realistic_depth_field/` | DA3-SMALL fine-tune + inference + the (dropped) Blender-in-the-loop nav test. |
| `Videoscopesimulations/` | *(supervisor-provided)* the 25-hinge tendon-driven scope tip. |
| `Stage4/v1/` | My working **collision-free** closed-loop controller (the V1 baseline). |
| `Stage4/v2_collision/` | My **collision-enabled RL** pipeline (current frontier). |
| `colon_realistic_geometry_handoff_2026-05-26/` | *(supervisor-provided)* photorealistic Blender colon code; I added `blender/blender_depth_dataset.py` for the DA3 dataset. |

---

## 4. Common workflows (copy-paste)

### A. Run the Stage 4 V1 closed loop (the headline demo)
Drives the scope through my colon using the best steering CNN, opens the MuJoCo viewer +
a camera overlay, and logs annotated frames.
```powershell
& $PY311 "Stage4\v1\controller_loop.py"
# then compact the frames into a timestamped video:
& $PY311 "Stage4\v1\frames_to_video.py"
```
To switch perception model, edit `CNN_FLAVOUR` near the top of `controller_loop.py`
(default `"centreline_10mm"`; options listed in the file).

### B. Train a PPO collision policy
Edit the run constants near the top of `train_ppo.py` first (`RUN_NAME`, `N_ENVS`,
`ent_coef`, etc.), then:
```powershell
& $PY311 "Stage4\v2_collision\train_ppo.py"                 # full run (TOTAL_TIMESTEPS)
& $PY311 "Stage4\v2_collision\train_ppo.py" --smoke         # short smoke + resource report
& $PY311 "Stage4\v2_collision\train_ppo.py" --total_timesteps 500000
```
Use **8 vec envs** (strictly safe on 16 GB RAM); 10 is tolerable; 12 pages to disk.
A 1M-step run is ~75 min on the RTX 4060. Output → `Stage4/v2_collision/runs/<RUN_NAME>/`.

### C. Evaluate / watch a PPO checkpoint
```powershell
& $PY311 "Stage4\v2_collision\eval_ppo.py" --model "Stage4\v2_collision\runs\ppo_v6_entcoef015\final.zip" --episodes 10
& $PY311 "Stage4\v2_collision\viewer_ppo.py"     # live MuJoCo viewer (set checkpoint near top of file)
```
> Note: the kept checkpoints `ppo_v3` and `ppo_v6` both *cheated* (see REPORT.md §5) —
> useful to watch, **not** honest baselines.

### D. Regenerate the steering-CNN dataset, then train
```powershell
& $PY311 "Perception\depth_regression\Depth_dataset.py" --n_samples 2000 --seed 2
& $PY311 "Perception\depth_regression\Training_point_regression.py"          # depth-blob CNN
& $PY311 "Perception\centreline_regression\look_ahead_10mm\relabel_centreline.py"   # re-derive 10mm labels (no re-render)
& $PY311 "Perception\centreline_regression\look_ahead_10mm\Training_centreline_regression.py"
```

### E. Render the photorealistic colon dataset (Blender, GPU)
```powershell
$BLENDER = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
& $BLENDER --background --python "colon_realistic_geometry_handoff_2026-05-26\blender\blender_depth_dataset.py" -- `
    --output "Perception\realistic_depth_field\dataset_finetune" --total 1500 --samples 32
# then drop void/wall-clip frames (dry-run first, then --apply):
& $PY311 "Perception\realistic_depth_field\filter_void_frames.py" --dataset_dir "Perception\realistic_depth_field\dataset_finetune"
& $PY311 "Perception\realistic_depth_field\filter_void_frames.py" --dataset_dir "Perception\realistic_depth_field\dataset_finetune" --apply
```

### F. Fine-tune DA3 and run inference
```powershell
& $PY311 "Perception\realistic_depth_field\finetune_da3.py" --batch_size 16 --epochs 20
& $PY311 "Perception\realistic_depth_field\run_da3_inference.py" `
    --checkpoint "Perception\realistic_depth_field\checkpoints\da3small_finetune_head.pth" `
    --output_dir "Perception\realistic_depth_field\predictions_finetune" --save_vis
```
(`run_da3_inference.py` without `--checkpoint` runs zero-shot DA3.)

---

## 5. File-by-file reference

Main files get a full description; helpers are one-liners.

### `Perception/` (root)
- **`Colon_mesh_xml.py`** — *(main)* My procedural colon generator. Builds the 500 mm
  anatomical centreline (ascending → flexures → descending), a ~15 mm-radius tube with
  35 haustral folds and a soft 3-lobed cross-section, and writes `colon.obj`,
  `colon_centreline.csv`, and `colon_test.xml` (pink material + endoscope tip light).
  Exposes `build_frames()` for camera placement. Seeded by `FOLD_SEED` for
  reproducibility. Everything downstream consumes its outputs.
- `Colon viewr.py` — one-off MuJoCo viewer for the colon mesh.
- `view_test.py` — renders 4 fixed camera views to sanity-check the camera maths.

### `Perception/depth_regression/`
- **`Depth_dataset.py`** — *(main)* Dataset generator. Picks random centreline points,
  perturbs the camera pose, moves the tip light to the camera, renders RGB + depth, and
  computes **two** labels per frame: deepest-depth-blob (`target_x/y`) and
  deepest-visible-centreline (`target_centre_x/y`). Writes `labels.csv` + RGB + depth
  `.npy` + debug overlays to `depth_dataset/`. CLI: `--n_samples`, `--seed` (default 2),
  `--out_dir`, `--width/--height`, plus pose-perturbation ranges. Run with `--help` to
  see all.
- **`Training_point_regression.py`** — *(main)* Trains a ResNet-18 (ImageNet-pretrained,
  head → `Linear→ReLU→Linear(2)→Tanh`) on `target_x/y`. 80/20 split, 30 epochs, Adam,
  MSE. Writes `colon_target_cnn.pth`.
- `Testing_point_regression.py` — loads weights, plots true vs predicted on a random
  frame.

### `Perception/centreline_regression/`
- **`Training_centreline_regression.py`** — *(main)* Same architecture as the
  depth-flavour trainer but reads `target_centre_x/y` (deepest-visible centreline).
  Writes `colon_centreline_cnn.pth`. **Superseded** by the look-ahead variants.
- `Testing_centreline_regression.py` — sibling tester.
- **`look_ahead_10mm/`** — *(main, current best)* Self-contained 10 mm pure-pursuit
  variant: `relabel_centreline.py` (re-derives labels from saved camera quats + depth,
  no re-render), its own `Training_/Testing_centreline_regression.py`, and trained
  `colon_centreline_cnn.pth`. This is the model the V1 loop uses.
- `look_ahead_15mm/` — same pattern at 15 mm; **superseded** (clipped walls at flexures).

### `Perception/realistic_depth_field/` (the DA3 work)
- **`finetune_da3.py`** — *(main)* Head-only fine-tune of DA3-SMALL (freezes the DinoV2
  backbone + cam enc/dec; trains only the ~11% DualDPT depth head). Mixed loss
  (scale-invariant log + `--metric_weight`×L1-in-metres). CLI: `--dataset_dir`,
  `--epochs` (20), `--batch_size` (default 4; the documented run used **16**), `--lr`,
  `--metric_weight` (0.1), `--seed`. Writes `checkpoints/da3small_finetune_head.pth`.
- **`run_da3_inference.py`** — *(main)* Runs DA3 (zero-shot, or fine-tuned via
  `--checkpoint`) over a dataset, saving predicted depth/conf, optional `RGB|pred|GT`
  vis (`--save_vis`), and `summary.json` (raw + scale-aligned MAE, GT/pred ratio). CLI:
  `--dataset_dir`, `--output_dir`, `--checkpoint`, `--limit`, `--device`, `--save_vis`.
- **`filter_void_frames.py`** — drops frames that are mostly black (rays missing into
  void). Dry-run by default; pass `--apply` to actually move files + rewrite `poses.npy`.
  CLI: `--dataset_dir`, `--threshold` (0.10).
- `test_navigation.py` — *(main, but dropped)* the hybrid MuJoCo+Blender nav test
  (depth-centroid aim + PD). Kept as reusable infrastructure; did **not** traverse
  reliably (no collisions). CLI: `--seed`, `--max_steps`, `--samples`, `--checkpoint`,
  `--out_dir`.
- `build_nav_scene.py` — generates the scope-only MuJoCo XML for `test_navigation.py`
  (Blender renders the colon, MuJoCo only does scope physics).
- `blender_render_worker.py` — persistent Blender subprocess (stdin JSON pose → render);
  driven by `test_navigation.py`, not run directly.

### `Videoscopesimulations/` *(supervisor-provided)*
- **`generate_videoscope_one_section.py`** — XML builder for the scope tip: 26 disks,
  25 alternating-axis hinges, 4 tendons + 4 actuators. Imported by both Stage 4 folders.
- `manual_tendon_viewer.py` — interactive viewer with antagonistic tendon control.
- `videoscope_one_section.xml`, `new8.stl`, `README.md` — generated model + mesh + the
  folder's own run notes.

### `Stage4/v1/` (my collision-free baseline)
- **`controller_loop.py`** — *(main)* The closed loop: regenerate scene → load CNN →
  per step render tip-cam → CNN inference → pixel error → rate-limited proportional
  tendon update → advance base along centreline → physics step. Writes annotated frames
  + `run_output/control_log.csv`. **No CLI** — tune via constants at the top
  (`CNN_FLAVOUR`, `K_P`, `RATE_MAX`, `DEADZONE`, `ADVANCE_RATE_M_PER_STEP`, `SIGN_X/Y`).
- **`build_combined_scene.py`** — *(main)* Generator that writes `combined_scene.xml`:
  places the scope inside my colon, sets the base mocap, adds `tip_cam`/`tip_light`.
  Called by the loop; can be run standalone to regenerate the scene.
- `frames_to_video.py` — compacts `run_output/frames/*.png` into a timestamped MP4
  (`--output x.gif` for GIF, `--delete` to remove PNGs).

### `Stage4/v2_collision/` (my collision RL pipeline)
- **`scope_colon_env.py`** — *(main)* The Gymnasium env `ScopeColonEnv`. Action
  `Box(-1,1,(3,))` = two tendon-pair commands + base advance. Observation = depth image
  `(1,64,64)` + small state vector. Reward = progress (high-water-marked) + goal −
  collision + waypoints. **Note:** the aim/look reward machinery from the v7 experiments
  is still present here and is slated for removal in the reframe (see REPORT.md §6).
- **`build_collision_scene.py`** — *(main)* Scene XML generator: visual mesh +
  per-triangle prism collision + scope + mocap target + tip cam/light. CLI:
  `--seed N`. Returns `(xml_path, centreline)`.
- **`colon_generator.py`** — *(main)* Pure-NumPy procedural colon (a port of the
  supervisor's Blender `random_centerline`+`build_tube`). One seed → high-res visual
  STL + low-res collision STL.
- **`mesh_to_thin_prisms.py`** — *(main)* Turns each wall triangle into a thin extruded
  6-vertex prism `<geom>` so a hollow tube collides correctly (the default convex-hull
  mesh would be a solid blob).
- **`train_ppo.py`** — *(main)* SB3 PPO `MultiInputPolicy` (NatureCNN on depth + MLP on
  state), `SubprocVecEnv`. CLI: `--smoke`, `--total_timesteps`, `--single_env`. Run
  constants (`RUN_NAME`, `N_ENVS`, `ent_coef`, reward weights) are edited at the top.
- **`eval_ppo.py`** — *(main)* Loads a checkpoint, runs N deterministic episodes per
  seed, prints a per-seed completion table. CLI: `--model`, `--episodes`, `--seeds`.
- `viewer_ppo.py` — live MuJoCo viewer + cv2 depth window for a checkpoint (watch for
  cheats).
- `view_scene.py` — viewer for the benchmark scenes without a policy (set `SCENE` at top).
- `baselines.py` — random / forward / centre-on-depth scripted policies for reference.
- `benchmark_collisions.py`, `_check_stl_axis.py`, `_smoketest_env.py`,
  `_smoketest_viz.py` — early-session benchmark/smoke helpers.
- `runs/` — PPO checkpoints. `ppo_v6_entcoef015/final.zip` (85% but cheated),
  `ppo_v3/final.zip` (66% but cheated), plus failed v1/v2/v4/v5/v7* runs kept for
  progress review.

### `colon_realistic_geometry_handoff_2026-05-26/` *(supervisor-provided)*
- `blender/blender_depth_dataset.py` — *(my derivative)* the DA3 dataset generator I
  wrote on top of the supervisor's Blender colon (adds intrinsics/pose dumps + GPU
  device enable). CLI (after `--`): `--output`, `--total`, `--samples`, `--width/height`.
- Other `blender/`, `mujoco/`, `colon3d/` files — supervisor's colon-generation and
  rendering code. Treat as a provided library.

---

## 6. Key artifacts on disk

| File | What |
|---|---|
| `Perception/centreline_regression/look_ahead_10mm/colon_centreline_cnn.pth` | Best steering CNN (used by V1 loop). |
| `Perception/depth_regression/colon_target_cnn.pth` | Depth-blob steering CNN. |
| `Perception/realistic_depth_field/checkpoints/da3small_finetune_head.pth` | Fine-tuned DA3 depth model (val MAE 3.28 mm, metric). |
| `Stage4/v2_collision/runs/ppo_v6_entcoef015/final.zip` | 85% PPO policy — **cheated**, evidence only. |
| `Stage4/v1/run_output/*.mp4` | Closed-loop run videos. |

---

## 7. Gotchas

- **Wrong Python** is the #1 trap — always use the 3.11 interpreter for torch/RL/DA3.
- **RAM** on the 16 GB laptop: 8 PPO vec envs safe, 12 pages to disk.
- **Viewer lighting affects the camera:** the MuJoCo passive viewer and the offscreen
  renderer share lighting state, so toggling the viewer headlight changes what the
  tip-cam (and thus the CNN) sees. Don't change viewer lighting during a controlled run.
- **Cheated checkpoints** (`ppo_v3`, `ppo_v6`) score well but are not honest baselines.
- **Datasets are absent by design** (§2) — regenerate rather than assuming a path exists.

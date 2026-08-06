> ## ⚠️ SUPERSEDED — HISTORICAL ARCHIVE, DO NOT FOLLOW
> This document describes the **old Mujuco_V2 layout** (`Stage4/vN/`, `transfer/`,
> `Perception/`) and its session-by-session state as of July 2026. Those paths no longer
> exist in this project. Sections describing datasets, checkpoints, and "next steps" are
> stale — several were already stale before the migration.
> - Current layout and how to run things: root `README.md`.
> - Consolidated, current history of v1→v7_shaft: `../ITERATION_HISTORY.md`.
> - The v8 plan: `../V8_PLAN.md`.
>
> Kept verbatim as the primary source behind `ITERATION_HISTORY.md`. Read for context,
> never for paths or instructions. Do not edit.

# Project Context

## Overall aim

Use a **simulated colon** and a **simulated videoscope** to design a controller and run simulated tests on it. Everything in this repo is a step toward that controller-design-and-evaluation loop in MuJoCo.

## Project stages

| Stage | Status | What it gives us |
|---|---|---|
| 1. Simulated colon environment + camera/depth rendering | Done | A scene the controller will eventually navigate, plus the ability to render what an onboard camera would see. |
| 2. Minimal image-recognition CNN (target-point regressor) | Done (minimal version) | The "perception" half — from a single RGB frame, predict where to aim. |
| 3. Simulated videoscope (25-hinge tendon-driven section) | Done | The "actuator" half — a physical tip that bends via 4 tendons. |
| 4. **Basic feedback-loop controller — keep the predicted point at the image centre** | **In progress** | First real closed-loop test: perception → error → tendon command. |
| 5. Onward from there | Future | Open — depends on how stage 4 behaves. See "Open questions" below. |

So far stages 1–3 are independent prototypes; stage 4 is the first time they have to talk to each other.

---

## Stage 1–3 inventory (what currently exists)

The two folders are still **decoupled** at this point — no shared imports, no shared assets. Stage 4 is what will couple them.

### `Perception/` — colon environment + CNN(s) (stages 1 & 2)

End-to-end pipeline: generate a synthetic colon, render frames inside it, auto-label each frame with a "where to drive next" pixel, train a small CNN to predict that pixel from RGB alone.

**Folder layout:** the shared mesh code and the generated mesh artefacts sit at the top of `Perception/`. CNN flavours live in subfolders so they can be iterated on independently:
- `Perception/depth_regression/` — original approach: label = centroid of the deepest open region in the depth image. Working on straight sections; struggles at corners (deepest direction is occluded behind the bend, so the model locks onto a haustral fold edge) and can also drive at protruding ribs when deep tube remains visible past them.
- `Perception/centreline_regression/` — second flavour family: labels derived from the ground-truth centreline rather than from depth. The dataset generator (`depth_regression/Depth_dataset.py`) writes the **deepest visible centreline point** label per frame so the two label types can be A/B-tested on identical RGB. Top-level training/testing scripts here train on that label. Variants of the centreline approach (different label algorithms) get their own subfolders so multiple can co-exist; first such variant is `look_ahead_30mm/`.

- **`Colon_mesh_xml.py`** — Procedural colon mesh. 500 mm anatomical centreline in the xz plane: ascending straight → 90° hepatic flexure (40 mm radius) → transverse straight → 90° splenic flexure → descending straight, parameterised so position and tangent stay continuous across segments. 15 mm base internal radius with slow longitudinal variation and a slight non-circular cross-section; a 40 mm clean inlet keeps the entrance circular. 35 evenly-spaced haustral folds (~14 mm apart, 6 mm peak depth, 1.5 mm gaussian width); each fold is a single partial arc at a randomly drawn angular centre with half-extent in [60°, 170°], fading to zero across a 60° edge band so it tapers smoothly into the wall instead of dropping vertically. RNG seeded by `FOLD_SEED` for reproducibility. Ring density is `N_AROUND=72` so the 60° fade resolves cleanly (~12 vertices across the taper). A small continuous 1.5 mm puckering between the three taeniae coli (`0.5 * (1 - cos(3·theta))`) keeps the cross-section softly 3-lobed even between folds. Per-vertex normals are written into the OBJ so MuJoCo smooth-shades the surface instead of flat-shading each triangle. Both open ends are capped with two-sided fans. Writes `colon.obj`, `colon_centreline.csv`, and `colon_test.xml` — the XML has a pink skin-tone material and an endoscope-style `camera_light` (warm-white positional light with attenuation/cutoff fall-off) as the only illumination. Exposes `build_frames()` (tangent / normal / binormal at every centreline point) for camera placement.
- **`depth_regression/Depth_dataset.py`** — Dataset generator. Picks a random centreline point, perturbs camera pose (radial offset + random yaw/pitch/roll), moves the `camera_light` to the camera each frame so the endoscope torch-light follows the view, renders RGB and depth, then computes **two** labels per frame: (a) the centroid of the deepest connected depth region (`target_x/y`) and (b) the deepest visible centreline point projected into the image (`target_centre_x/y`) via a walk-forward visibility test against the depth buffer. Saves RGB, depth `.npy`, debug overlay (with both labels drawn — red = depth, yellow = centreline, cyan = visible centreline trace), and `labels.csv` (both label pairs + camera quat for re-labelling). One-off sanity assert at startup confirms the projection math by projecting a straight-ahead centreline point onto the image centre. By default, frames where either labeller can't produce a target are skipped (`--require_both_labels`) so the depth and centreline datasets stay row-aligned. Output goes to `depth_regression/depth_dataset/`. Reads shared `colon_test.xml` + `colon_centreline.csv` from `Perception/`.
- **`depth_regression/Training_point_regression.py`** — ResNet18 (ImageNet-pretrained) with the classifier replaced by `Linear(512→128) → ReLU → Linear(128→2) → Tanh`. 80/20 split, 30 epochs, Adam (lr=1e-4), MSE. Reads `target_x/y` from `depth_regression/depth_dataset/`, writes `depth_regression/colon_target_cnn.pth`.
- **`depth_regression/Testing_point_regression.py`** — Loads the weights, picks a random labelled image, plots true vs predicted target.
- **`centreline_regression/Training_centreline_regression.py`** — Trains the *deepest-visible-centreline-point* variant. Same architecture and loop as the depth-flavour trainer, but reads `target_centre_x/y` from the shared dataset at `../depth_regression/depth_dataset/` and writes `centreline_regression/colon_centreline_cnn.pth`. **Status:** trained; controller test 2026-05-25 showed it over-rotates at flexures because the label sits on the wall corner where the centreline disappears — superseded by `look_ahead_30mm/` below.
- **`centreline_regression/Testing_centreline_regression.py`** — Sibling tester for the deepest-visible model.
- **`centreline_regression/look_ahead_<N>mm/`** — Look-ahead (pure-pursuit-style) centreline variants. Each variant folder is self-contained: `relabel_centreline.py` (with `LOOK_AHEAD_M` set for this variant), `labels.csv` (output of relabel), `Training_centreline_regression.py` / `Testing_centreline_regression.py` (point at this folder's labels), and `colon_centreline_cnn.pth` (trained weights). Relabel is fast (no re-rendering) — it reuses the camera quat saved in the shared `labels.csv` plus each frame's depth `.npy`. Walk-forward picks the centreline point that's `LOOK_AHEAD_M` from the camera, falling back to the last visible point if that look-ahead is occluded. New variants follow the same self-contained-folder pattern.
  - **`look_ahead_30mm/`** — **Status:** trained but over-aggressive in practice. At every corner approach the fallback fires (the centreline is occluded by the inside wall well before 30 mm), so the label collapses to the wall-corner pixel — same failure mode as the deepest-visible variant. Superseded.
  - **`look_ahead_15mm/`** — **Status (2026-05-25): superseded by 10 mm.** Trained and tested end-to-end through the full colon. The scope *did* turn at the flexures, but corner-handling was *too tight* — the scope over-bent and slightly clipped the wall on both flexures, recovered without a complete breakdown, and reached the descending segment. Reviewing this run suggested a shorter look-ahead would give gentler corner commands.
  - **`look_ahead_10mm/`** — **Status (2026-05-26): current best.** Same look-ahead labeller as the other variants but `LOOK_AHEAD_M = 0.010`. Trained and tested end-to-end; the scope traverses the full colon under closed-loop control with no visible tip-cam crash. Confirms the prediction from the 15 mm video that a shorter look-ahead softens the bend command.
- **`Colon viewr.py`** — One-off MuJoCo viewer for the colon mesh.
- **`view_test.py`** — Renders four fixed camera views into `camera_pose_tests/` to sanity-check the camera-axis maths.

**Rendering note (2026-05-25):** `camera_light` currently has `castshadow="false"` because shadow acne on the haustral ribs was producing black speckles all over the interior and corrupting both RGB and depth. The `<map>` znear/zfar was also tightened from `0.0001 / 50` to `0.001 / 1.0` for depth-buffer precision, and the duplicate-winding end caps were collapsed to single-sided. Probably worth re-enabling shadows in the future for realism — when doing so, first bump `<visual><quality shadowsize="2048"/>` (or `4096`) before flipping `castshadow` back on.

**Realism pass (2026-05-25):** ribs sharpened (`FOLD_WIDTH` 3 → 1.5 mm, `FOLD_STRENGTH` 8 → 6 mm) and `N_RINGS` doubled to 280 so the sharper Gaussian still samples cleanly. Smooth shading is now enabled via per-vertex normals in the OBJ. Material bumped to `specular=0.15`, `shininess=0.30`, `emission=0.005` for a wet-mucosa look. The haustra angular pattern went through two iterations — the final form is: each fold is a single soft top-hat arc at a randomly drawn angular centre, half-extent in [60°, 170°], 20° edge fade, layered on top of a small continuous 3-taenia puckering that keeps the cross-section softly 3-lobed even between folds. An earlier attempt that multiplied every fold by the same 3-lobe taenia mask produced a too-symmetric "fidget-spinner" look and has been removed. **CNN retraining is required** — the lumen cross-section is meaningfully different from stage-2-era frames. Not yet done: procedural mucosa texture (would need a PNG asset and `<texture>` setup). If the camera-mounted specular highlight becomes a problem, dial `specular` back to ~0.08.

### `Videoscopesimulations/` — videoscope tip (stage 3)

A programmatically-generated MuJoCo model of one bending section of an articulating videoscope, plus an interactive viewer with antagonistic tendon control.

- **`generate_videoscope_one_section.py`** — XML builder. 26 disks chained through 25 hinge joints with alternating X/Z hinge axes so the section can bend in any direction. 9.0 mm OD disks at 2.5 mm spacing; `new8.stl` visual mesh per disk; 4 tendon-hole sites per disk at 0°/90°/180°/270°, 3.2 mm from axis; 4 spatial tendons; 4 affine-gain actuators (`pull_px/nx/pz/nz`) commanding tendon length in metres.
- **`manual_tendon_viewer.py`** — Loads the XML, launches the passive viewer, enforces antagonistic pairing (pulling one tendon by +d automatically sets its opposite to −d). Pair range ±24 mm. Prints live `tip_site` position in mm.
- **`videoscope_one_section.xml`** — Generated model (safe to delete and regenerate).
- **`new8.stl`** — Per-disk visual mesh.
- **`README.md`** — Self-contained run instructions for this folder.

### `Stage4/` — combined scene + controller (stage 4, in progress)

The first folder that imports from both `Perception/` and `Videoscopesimulations/`. It builds the combined MuJoCo scene and runs the closed-loop controller.

- **`build_combined_scene.py`** — Generator that writes `combined_scene.xml` next to itself. Calls `generate_colon()` (so the latest mesh tweaks always flow through), reads `colon_centreline.csv`, places the scope at `s = 0.05 m` (5 cm into the ascending segment, just before the first hepatic flexure), and rotates the scope body by quat `0.7071 0.7071 0 0` (= +90° about world-X) so its local +Y forward axis aligns with the colon tangent (+Z) at the anchor. `scope_anchor` is marked `mocap="true"` so the controller can drive the base pose kinematically during simulation; joints/tendons still apply to the disk chain underneath. Adds a `tip_cam` and `tip_light` as children of the last disk (`disk_25`): camera `xyaxes="-1 0 0 0 0 -1"` chosen so that when the scope is straight, the image orientation matches the convention the CNN was trained on (image-up = world +Y, image-right = world -X, forward = world +Z); light duplicates the colon's `camera_light` parameters so lighting matches training. The colon mesh keeps `contype=0 conaffinity=0` in v1 — no scope↔colon collisions yet.
- **`controller_loop.py`** — Runs the closed loop: regenerates the scene, loads the trained CNN, opens the MuJoCo passive viewer plus an optional cv2 overlay window. Each control step: render the tip-camera RGB → ResNet-18 inference → `(x_norm, y_norm)` pixel error → deadzone check → rate-limited proportional update of two antagonistic tendon pairs (`pull_px ↔ pull_nx`, `pull_pz ↔ pull_nz`) → advance the base mocap along the centreline (with the base re-oriented tangent to the local centreline direction; bending state is left untouched so the scope's shape just translates+rotates with the base) → physics step. Tunable constants live at the top of the file (`DEADZONE`, `K_P`, `RATE_MAX`, `PHYSICS_STEPS_PER_CONTROL`, `SIGN_X`, `SIGN_Y`, `ADVANCE_RATE_M_PER_STEP`, `MAX_S`, `CNN_FLAVOUR`). `CNN_FLAVOUR` selects from `"depth"`, `"centreline"` (deepest-visible — superseded), `"centreline_30mm"` (30 mm look-ahead — superseded), `"centreline_15mm"` (15 mm look-ahead — superseded) and `"centreline_10mm"` (current best). The corresponding `.pth` path is looked up from a dict; add a row to the dict for each new variant. Per-step pose is interpolated between the two nearest centreline indices (linear position; t/b linearly interpolated then Gram-Schmidt re-orthogonalised) and converted to a MuJoCo quaternion via `mju_mat2Quat`. Writes annotated frames to `run_output/frames/` and a per-step CSV (now including `s_current`) to `run_output/control_log.csv`. The sign convention (`SIGN_X`, `SIGN_Y`) may need flipping after the first run if the scope moves the wrong way for an axis.
- **`frames_to_video.py`** — Utility (run after `controller_loop.py` exits) that compacts the per-step PNG frames in `run_output/frames/` into a single video file with a timestamped name so consecutive runs don't overwrite each other. Defaults to MP4 via cv2 (smaller + faster than GIF); pass `--output something.gif` to get a GIF instead, or `--delete` to remove the source PNGs once the video is written successfully.
- **`combined_scene.xml`** — Generated; safe to delete and regenerate.
- **`run_output/`** — Generated per-run logs, debug frames, and any compacted video files.

---

## Stage 4 — In progress: basic feedback-loop controller ("centre the target")

**Current status (2026-05-26):** end-to-end working with `CNN_FLAVOUR = "centreline_10mm"`. The scope traverses the full colon under closed-loop control with no visible tip-cam crash. The 15 mm variant ran end-to-end too but clipped the wall slightly at both flexures; the shorter 10 mm look-ahead gives the gentler corner commands predicted from the 15 mm video. The depth, deepest-visible-centreline, 30 mm and 15 mm look-ahead flavours have all been tried and superseded. **Stage 4 baseline is paused awaiting supervisor input.**

**Goal:** close the loop between the CNN and the tendons. The CNN outputs a target pixel; the controller's job is to drive the tendons so that target moves to (and stays at) the image centre.

This is the first stage where the two folders need to talk to each other. Outline of what stage 4 will need:

1. **A combined scene** — the videoscope tip placed inside the colon mesh, with the videoscope's tip camera being the one that renders the frame the CNN consumes.
2. **A per-step loop** — render frame → CNN inference → compute pixel error from centre → map error to tendon commands → step MuJoCo.
3. **A simple controller first** — e.g. proportional control: `pull_px - pull_nx ∝ x_error`, `pull_pz - pull_nz ∝ y_error` (with antagonistic pairing already enforced by the viewer code). No advance planning, no learning yet — just "see point off-centre, pull toward it."
4. **A way to evaluate it** — record pixel error over time, tip pose over time, and whether the scope progresses along the colon centreline or stalls.

Stages after this are intentionally undefined until we see how stage 4 behaves (better controller? add a forward-advance axis? swap proportional for something smarter? retrain CNN on tip-camera frames specifically?).

---

## Planned next stage (decided 2026-05-27)

Predict a **depth map from RGB** using the realistic Blender-rendered colon images in `colon_realistic_geometry_handoff_2026-05-26/`. Depth is the simpler, well-defined subproblem; steering can build on it later.

Approach is **Depth Anything 3 (SMALL) zero-shot inference** — no fine-tuning, no from-scratch model. DA3-LARGE / fine-tuning were ruled out as too heavy for this purpose. If zero-shot results are poor we'll revisit.

### What's done this session (2026-05-27)

- **Dataset generator written:** `colon_realistic_geometry_handoff_2026-05-26/blender/blender_depth_dataset.py`. Derivative of `blender_diverse_dataset.py` (5 random geometries × 6 camera scenarios × randomised material/lighting × pixel-perfect BVHTree depth) with three additions: default output path under `Perception/realistic_depth_field/dataset/`, per-frame pose dump to `poses.npy`, and `intrinsics.json` dump.
- **Pilot dataset rendered:** 120 frames at 320×256, 32 Cycles samples, 3.6 min total. Lives at `Perception/realistic_depth_field/dataset/`. Outputs: `rgb/frame_XXXXX.png`, `depth/depth_XXXXX.npy` (float32 metres, 0 = invalid mask, clamped at 0.5 m), `poses.npy` (120, 4, 4) with OpenGL-style `[right, -up, fwd, pos]` columns, `intrinsics.json` (fov=140°, fx=fy=58.2, cx=160, cy=128). Sanity-checked: depth ranges 2–250 mm across scenarios, 55–100% coverage (the 55% is `near_wall`/`sideways` frames where rays miss into open lumen).
- **DA3 inference script written:** `Perception/realistic_depth_field/run_da3_inference.py`. Loads DA3-SMALL from HuggingFace, iterates one image per call (DA3 fuses batched images into one multi-view scene — wrong for independent mono-depth), saves predicted depth + confidence `.npy`, optional side-by-side `RGB | pred | GT` PNGs via `--save_vis`, writes `summary.json` with raw MAE / scale-aligned MAE / median GT-over-pred ratio. The ratio answers "is DA3 output metric or relative inverse depth?".
- **Full DA3 stack installed** into the project's Python 3.11.1 (`C:\Users\lukec\AppData\Local\Programs\Python\Python311\python.exe`):
  - `torch==2.11.0+cu128` (was CPU-only; CUDA now sees the RTX 4060 Laptop, 8.59 GB VRAM)
  - `torchvision==0.26.0+cu128`
  - `depth-anything-3` editable install from `C:\Users\lukec\PycharmProjects\Depth-Anything-3` (zip-downloaded from GitHub — git isn't on PATH on this machine)
  - `transformers 5.9.0`, `xformers 0.0.35`, `pycolmap 4.0.4`, `open3d 0.19.0`, `e3nn`, `trimesh`, etc.
  - `addict 2.4.0` — DA3's `pyproject.toml` forgets to list it; added manually
  - One harmless warning at import: `gsplat` not installed. Only needed for 3D Gaussian Splatting export, not for depth inference.
- **Final env verification:** `torch.cuda.is_available()=True` on RTX 4060, `from depth_anything_3.api import DepthAnything3` imports cleanly. Ready to run.

### Immediate next step (on resume)

Run the 5-frame validation:

```powershell
& "C:\Users\lukec\AppData\Local\Programs\Python\Python311\python.exe" `
  "C:\Users\lukec\PycharmProjects\Mujuco_V2\Perception\realistic_depth_field\run_da3_inference.py" `
  --limit 5 --save_vis
```

First run downloads DA3-SMALL weights from HuggingFace (~hundreds of MB). Then inference on 5 frames. Total ~1–3 min on the 4060.

After it finishes, inspect:
1. `Perception/realistic_depth_field/predictions/visualizations/vis_*.png` — does DA3's depth shape look at all like the GT?
2. `predictions/summary.json` → `median_gt_over_pred` — if ~1.0 the model is metric; if very different (e.g. 0.001 or 1000), it's relative inverse depth and needs scale alignment.

If validation looks reasonable, rerun without `--limit` for all 120 pilot frames, then decide whether to scale the dataset up (current pilot is 120 frames; the script defaults to 1200 for the "full" run — that would take ~30 min at 320×256).

### Environment notes (carry forward)

- **Python for DA3:** `C:\Users\lukec\AppData\Local\Programs\Python\Python311\python.exe` (3.11.1). DA3 pins `requires-python >=3.9, <=3.13` — must NOT upgrade past 3.13 for DA3 work. The system also has Python 3.14.5 at `C:\Python314\python.exe` (default `python` on PATH); don't use this for DA3.
- **Blender:** `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe` (5.1.2). The Start Menu shortcut points at `blender-launcher.exe` (GUI launcher); for headless rendering use `blender.exe` in the same folder.
- **DA3 source:** `C:\Users\lukec\PycharmProjects\Depth-Anything-3` — editable install, don't delete.
- **Disposable smoke test output:** `colon_realistic_geometry_handoff_2026-05-26/_smoketest/` — safe to delete; was only used during pipeline verification.

### DA3-SMALL zero-shot pilot results (2026-05-27)

Ran `run_da3_inference.py` over the full 120-frame pilot (15.5 s wall-clock on the 4060, ~20 FPS — well within real-time budget).

- **Output type:** relative inverse depth. `median_gt_over_pred ≈ 0.029` (consistent across frames, range 0.022–0.044). DA3-SMALL is NOT metric; values are roughly `metres ≈ pred * 0.029`.
- **Overall accuracy:** `mean_scale_aligned_mae_m = 7.1 mm`. After applying one scalar per frame, mean error is ~7 mm against GT depth ranging 2–250 mm.
- **Bimodal quality by frame type:**
  - Frontal lumen views (e.g. frames 23, 92, 95): aligned MAE ~1–4 mm. Pred correctly puts the deep region at the lumen centre, slightly fuzzy halo rather than a sharp disc. Acceptable for aim point; soft for mapping.
  - Sideways / flexure-approach views (e.g. frames 36, 60, 101, 28): aligned MAE ~10–14 mm and **structurally wrong**. See failure mode below.
  - Near-wall views (e.g. frames 15, 39, 108): low MAE numbers are partly artefactual (fewer valid GT pixels because most rays miss into void); on the visible wall portion the gradient is roughly correct.
- **Bug fixed during pilot:** DA3 returns predictions at 504×406 (multiple of its ViT patch size) instead of the 320×256 input shape. `predict_one()` in `run_da3_inference.py` now resizes depth and confidence back to the input `(H, W)` with PIL bilinear before saving/comparison.

**Failure mode (confirmed visually 2026-05-27):** "Lit wall surface confused with depth." On sideways/flexure-approach views where the lumen hole sits off-centre against a brightly-lit wall, DA3 places the peak "deep" region on the well-lit wall surface instead of on the actual lumen hole. The model has no domain prior that "dark = deep" inside a tube. This is precisely the worst failure mode for navigation — the steering controller most needs depth to be correct at flexures, which is where DA3 fails.

**Implication for use cases:**
- **Steering / aim point:** zero-shot DA3-SMALL is dominated by the existing `centreline_10mm` CNN, which was trained specifically to find off-centre lumens at flexures. DA3 is not a steering upgrade.
- **Mapping (sim):** shape is roughly correct on frontal views, washed out on folds and unreliable at flexures. Scale recovery via known scope pose (multi-view consistency) is straightforward; in deployment, scope insertion depth + tip gyro give the same pose, so the recovery path transfers. But the underlying shape fidelity is the limit, not scale.
- **RL observation vector:** smeared "where is open space" is still a useful low-dimensional feature even with the flexure failure, provided the policy is robust to mis-detection on those frames.

### DA3-SMALL fine-tuning DONE (2026-05-27 late)

Mixed-loss fine-tune (Eigen scale-invariant log loss + 0.1 × L1 in metres) succeeded. Files:
- `Perception/realistic_depth_field/finetune_da3.py` — training script. Head-only fine-tune; freezes DinoV2 backbone, cam_enc, cam_dec; only the DualDPT depth-branch trainable (3.87M / 34.3M params = 11.3%). Bypasses DA3's `@torch.inference_mode()`-decorated `forward()` by calling `model.model(...)` (inner `DepthAnything3Net`) directly. PyTorch DataLoader, batch=16 in bf16 autocast, Adam lr=1e-4, ~15 min for 20 epochs on the 4060.
- `Perception/realistic_depth_field/checkpoints/da3small_finetune_head.pth` — best checkpoint (epoch 18). Loaded by `run_da3_inference.py --checkpoint ...`.
- Training data: `Perception/realistic_depth_field/dataset_finetune/` (1469 frames after rendering 1500 in 17.4 min on GPU OPTIX and filtering 31 void frames via `filter_void_frames.py`). Geometry 5 (filename 1200..1499) held out as val.

Results:
- **Held-out val (geom 5):** zero-shot 9.49 mm → fine-tuned **3.28 mm** scale-aligned MAE (≈2.9× better).
- **120-frame pilot dataset (completely unseen during training):** zero-shot scale-aligned MAE 7.1 mm → fine-tuned **2.4 mm** (≈3× better). Mean raw MAE = 4.1 mm.
- **Mixed loss converged to metric output:** median GT/pred ratio = **1.014** (was 0.029 zero-shot). The L1-in-metres term in the mixed loss did its job — no scale-recovery step needed downstream.
- **Flexure failure mode FIXED.** Visual inspection of vis_00036, vis_00060, vis_00101 (the sideways/flexure-approach frames where zero-shot put the "deep" region on the wall): fine-tuned puts the bright region exactly on the lumen hole, matching GT. Good frontal cases are sharper too.
- Inference cost unchanged at ~20 FPS — head-only fine-tune doesn't affect architecture.

`run_da3_inference.py` now accepts `--checkpoint <path>` to load fine-tuned weights on top of the HuggingFace pretrained. Output to a separate `--output_dir` (e.g. `predictions_finetune/`) to preserve the zero-shot results.

### Realistic-perception nav test — attempted and dropped (2026-05-28)

Built a hybrid MuJoCo + Blender closed-loop test driving the fine-tuned DA3 as perception. Files (kept; reusable for the next attempt):

- **`Perception/realistic_depth_field/test_navigation.py`** — main controller. Auto-finds an `s_start` where the rigid 65 mm scope chain naturally fits inside the colon (`find_good_s_start`). PD controller on the centroid-of-top-5%-deepest aim point. Hybrid advance: full rate 0.0005 m/step when on-target, 0.0001 m/step when off-target. Reverse-when-stuck (rate 0.0003 m/step) with cap at s_start. Mild tendon-relax (×0.95) while stuck. Stuck detector = fraction of pixels with depth > 50 mm < 5%. Defaults: K_P = K_D = 0.0008, RATE_MAX = 0.0003, DEADZONE = 0.08, SIGN_X = SIGN_Y = +1, samples = 8, seed = 42.
- **`Perception/realistic_depth_field/build_nav_scene.py`** — programmatic MuJoCo XML generator. Same scope geometry as Stage 4 (imports `Videoscopesimulations/generate_videoscope_one_section.py`), mocap-anchored, tip_cam + tip_light at the last disk. **No colon mesh in MuJoCo** — Blender renders it; MuJoCo only handles scope physics + mocap base advancement.
- **`Perception/realistic_depth_field/blender_render_worker.py`** — persistent Blender subprocess. Builds the procedural colon mesh (seed=42) once at startup, then loops on stdin reading JSON poses and rendering each request to a fixed temp PNG. Stdout protocol: `"READY\n"` after setup, `"DONE\n"` after each render. OPTIX/CUDA GPU at samples=8 ≈ 0.5 s/frame.

**What worked:**
- Auto-find s_start picked 176 mm for seed 42 with rigid-tip offset only 4.7 mm — well inside the ~20 mm tube radius. Initial frame is a textbook "looking down the lumen" view.
- Stuck detector (deep-pixel fraction) correctly identifies wall-jam vs lumen view.
- Reverse-with-cap prevents the doom spiral where reversing past s_start lands the scope in curvier territory.
- In intermittent good frames the perception + aim-point algorithm correctly localises the lumen.

**What didn't:**
- Scope cannot reliably traverse end-to-end. It oscillates between "looking down lumen" and "jammed against the wall", dwelling near s_start without advancing.
- Each controller knob (deadzone, K_P/K_D, RATE_MAX, relax factor, reverse policy, stuck threshold) trades one failure mode for another.
- Fundamental issue: **no scope ↔ colon collisions** + a too-simple per-pixel-target + PD controller. The scope has to centre the lumen purely from depth feedback in free space; on this colon's curvature, the simple controller can't.

**User conclusion (2026-05-28):** "target pixel only, no collisions, basic controller logic is not cutting it" — wrap this experiment. Do not iterate further on bare-pixel-target + PD.

**Likely next directions (pending supervisor input):**
- **Collisions:** import the Blender colon mesh into MuJoCo as a collision-only geom. Let the walls physically constrain the scope shape — the controller wouldn't need to do all the aiming from depth feedback alone. This is the most plausible "real" fix.
- **Better controller** (MPC / RL on fine-tuned depth as observation). Substantial new work; probably needs collisions enabled first to be tractable.
- **Phase 2.5 backbone unfreeze of DA3-SMALL** (still on the table; orthogonal): ~30 min training, could push val MAE from 3.28 mm toward ~1.5-2 mm.
- **Stage 4 + fine-tuned DA3:** swap the `centreline_10mm` CNN in `Stage4/controller_loop.py` for the fine-tuned DA3 in the existing MuJoCo colon. Domain gap: model trained on Blender renders, MuJoCo uses different shading/materials, so generalisation isn't guaranteed.

---

## Stage 4 v2 — Collision-enabled procedural RL (this session, 2026-05-28)

Took the "Collisions" next direction from the bullets above and built a full RL training pipeline. The existing `Stage4/` baseline was preserved under `Stage4/v1/` (paths updated to `.parent.parent.parent` from the deeper folder) and the new work lives in `Stage4/v2_collision/`.

### Files in `Stage4/v2_collision/`

- **`colon_generator.py`** — pure-NumPy port of `colon_realistic_geometry_handoff_2026-05-26/blender/blender_depth_dataset.py`'s `random_centerline()` + `build_tube()`. Same procedural distribution DA3 was fine-tuned on, no Blender dependency at training time. One seed produces a high-res visual mesh (n_axial=500, n_radial=36, ~36k tris) + a low-res collision mesh (n_axial=125, n_radial=20, ~5k tris). Inward-facing triangle winding so MuJoCo's tip-light lights the inner wall correctly. End caps on visual only — collision is open-ended so the rigid scope can enter freely from outside.
- **`mesh_to_thin_prisms.py`** — STL → `<geom>` emitter. Two flavours: `emit_thin_prisms()` writes OBB boxes per triangle (fast, but visually "shard" looking from the inside); `emit_triangle_prism_meshes()` writes a 6-vertex `<mesh>` asset + `<geom type="mesh">` per triangle (extrudes the triangle along its normal). Per-triangle prism is the one used — adjacent triangles share edges, walls look like the smooth mesh, no shards.
- **`build_collision_scene.py`** — programmatic scene XML. Generates the colon for a seed, writes both STLs alongside, emits visual mesh + collision prisms + scope + mocap target + tip cam + tip light. Returns `(xml_path, centreline)`. CLI: `python build_collision_scene.py --seed N`.
- **`scope_colon_env.py`** — Gymnasium `ScopeColonEnv`. Action is `Box(-1, 1, (3,))` for `(x, y, z)` where x/y drive rate-limited tendon-pair commands and z drives target advance along the centreline. Observation is a Dict with `depth` (1, 64, 64) float32 metric-depth from `tip_cam` (rendered via `mujoco.Renderer.enable_depth_rendering()`) and `state` (9,) for [cmd_x_n, cmd_y_n, progress, last_action[3], prev_action[3]]. Reward = W_PROGRESS × new-territory-advance + W_GOAL × 1[goal] − W_COLLISION × ncon + W_WAYPOINT × milestones-crossed. New-territory bookkeeping (`max_actual_base_s`) prevents the policy from gaming reward by oscillating; waypoints at 10/20/.../90% of the goal span give partial-progress credit so a stuck-at-50% episode is still rewarding.
- **`train_ppo.py`** — Stable-Baselines3 PPO with `MultiInputPolicy` (NatureCNN on depth + small MLP on state), 8 or 12 SubprocVecEnv workers (seeds 0..N-1, one colon per worker for the whole run), 1M timesteps default. Custom `ResourceMonitor` callback samples RAM/CPU/GPU during training and prints a summary at exit.
- **`eval_ppo.py`** — load a checkpoint, run N deterministic episodes per seed, print a per-seed completion-rate table sorted best→worst.
- **`viewer_ppo.py`** — load a checkpoint, open MuJoCo passive viewer + cv2 depth window, run the policy live so you can watch and inspect for cheats.
- **`view_scene.py`** — viewer for the benchmark scenes (B0/B1/B2/B4/B5/main) without a policy. Set `SCENE` and `SCENE_SEED` at the top.
- **`baselines.py`** — random / sin-cos-forward / centre-on-depth scripted policies for floor/ceiling reference numbers.
- **`benchmark_collisions.py`** + `_check_stl_axis.py` + `_smoketest_env.py` + `_smoketest_viz.py` — early-session helpers. Benchmark established that per-triangle thin-prism collision on 5k triangles runs at ~1500 phys-steps/sec on this RTX 4060 laptop, plenty for RL.
- **`runs/`** — `ppo_v3/final.zip`, `ppo_v6_entcoef015/final.zip` + per-100k checkpoints for each run. Failed runs (v4/v5) overwrote each other; v6 is the only "succeeded on numbers" model still on disk.

### Architecture decisions (in order they were discovered)

1. **Per-triangle thin-prism collision** beats `<geom type="mesh">` for a hollow tube. MuJoCo's default mesh collision uses the convex hull, which for a tube is a solid blob (scope ignores the inner wall). Per-triangle prism = each triangle becomes a 6-vertex convex mesh extruded along its normal by `WALL_THICKNESS_M`. Adjacent prisms share edges exactly → looks like the smooth mesh, fully captures the inner wall.
2. **Scope base attached via 1-DOF slide joint** to a kinematic mocap "target" body, NOT a 6-DOF free joint + soft weld. The slide axis = target's local +Y = centreline tangent at the current target position. Allows the chain to lag behind target when blocked by walls; forbids rotation, lateral drift, and "flip-around" cheats.
3. **Dynamic slide-joint floor** — each env step writes `model.jnt_range[slide_joint_id, 0] = base_s_init - target_base_s`. Forces `actual_base_s ≥ base_s_init` at all times. The scope physically cannot escape its spawn position backward through the entrance.
4. **Spawn 10 mm into the colon** (was 5 mm) so the slide-floor enforcement doesn't immediately pop the tip out through the entrance cap.
5. **0.5 mm collision wall thickness** (was 0.1 mm) plus `margin="0.0005"`. Tunneling threshold rises from ~0.2 m/s to ~1.0 m/s of disk velocity per physics step. Still beatable by hard tendon yanks but much harder.
6. **Reward shape** — `W_PROGRESS=1.0 × max(0, Δhigh_water_mark)`, `W_COLLISION=0.00001 × ncon`, `W_GOAL=1.0` one-shot, `W_WAYPOINT=0.1 × each of 9 fractional milestones crossed`. High-water-mark progress + no-double-credit waypoints means oscillating forward/back doesn't farm reward.
7. **`ent_coef`** in PPO needs to be calibrated to roughly match task-reward magnitude. v5 at 0.03 had entropy term ≈ 0.145/step vs task ≈ 0.001/step → policy maximised entropy and ignored task. v6 at 0.015 was the sweet spot.

### Training run history

| run | key changes from prior | mean@1M | outcome |
|---|---|---|---|
| v1 | `W_COLL=0.0002 ent=0.0` | ~0 % | reward made any motion net-negative; collapsed to do-nothing |
| v2 | `W_COLL=0.0001` | ~0 % | still too punitive |
| v3 | `W_COLL=0.00002 ent=0.01` | **66 %** | **but cheated** via 6-DOF free joint + soft weld → "drag-base-backward, flip orientation" |
| v4 | 1-DOF slide + slide floor + 0.5mm walls + 10mm spawn (cheat closed) | flat 0.5 % | task became sparse-reward, PPO couldn't find a path |
| v5 | + waypoint rewards, `W_COLL` halved, `ent=0.03` | flat 1 % | entropy term overwhelmed task signal (~100×) |
| **v6** | `ent=0.015` (back from 0.03) | **85 %** | **honest 85 % mean, max 100 % since iter 32 — BUT viewer revealed a new cheat** |

### v6's cheat — "curl into a ball"

Watched v6 in the viewer; the policy uses the scope's 25 hinges × 0.55 rad each = ~770° of available bend to fold the chain into a tight ball and push that ball through the prism wall at curved sections. The 0.5 mm wall thickness + finite contact-force capacity can't fully prevent penetration when concentrated through a tightly-curled chain.

Real endoscope articulating sections only flex about 90–180° total. Our chain is **~4× too floppy**.

### Next step — v7

Fix the scope flexibility in `Videoscopesimulations/generate_videoscope_one_section.py` (affects both `Stage4/v1/` and `Stage4/v2_collision/` since both import this generator):

| | Current | Proposed for v7 |
|---|---|---|
| `range` per hinge | `-0.55 0.55` rad (±31°) | `-0.12 0.12` rad (±7°) |
| `stiffness` per hinge | `0.10` | `0.5` (5× stiffer) |
| Max total chain bend | ~770° (multiple full loops) | ~175° (realistic articulating section) |

Steps for the next session:
1. Make the two-attribute edit in `add_disk()` inside `generate_videoscope_one_section.py`.
2. Re-run `Stage4/v1/controller_loop.py` with `CNN_FLAVOUR="centreline_10mm"` as a sanity check that the existing baseline still works with the stiffer scope. Might need to retune the controller's `K_P` / `RATE_MAX` — flag if it does.
3. Smoke-test the v2_collision env with the stiffer scope, watching in `_smoketest_viz.py` (forward action mode) — the scope should physically refuse to curl into a ball even at MAX_PULL.
4. Train v7 from scratch using v6's hyperparameters (`W_COLLISION=0.00001`, `ent_coef=0.015`, waypoints, 8 vec envs, 1M timesteps, 12 envs OOMs at 0.5 mm walls). `RUN_NAME = "ppo_v7_stiffscope"`. Expect lower mean than v6 (real task is harder without the curl exploit) but the resulting policy will be a real-physics solution.
5. After training, run `eval_ppo.py` and the viewer to confirm no further cheats. If a third cheat shows up, the next likely candidates are: solver-instability exploitation at high contact forces (would need `solref`/`solimp` tuning), or non-adjacent disk self-collisions creating tunnel paths through the chain itself (could add wider `<exclude>` block in `<contact>`).

Side notes for future-self:
- v6's checkpoint at `Stage4/v2_collision/runs/ppo_v6_entcoef015/final.zip` is preserved but the cheat means the numbers are misleading; do NOT report v6 as a "working baseline".
- v3's `runs/ppo_v3/final.zip` is the original cheat-trained model; same caveat.
- RAM is tight on this 16 GB laptop. **8 vec envs + 0.5 mm walls works**; 12 vec envs pages.
- Training fps on the RTX 4060: ~225 fps aggregate at 8 envs. 1M timesteps ≈ 75 min wall-clock. Tail the run via `Get-Content -Wait <task-output-file>`.
- Python 3.11 has the GPU torch (`C:\Users\lukec\AppData\Local\Programs\Python\Python311\python.exe`). Python 3.14 (default on PATH) has CPU-only torch + a duplicate SB3 install — use the 3.11 path explicitly when launching training.

---

## Stage 4 v7 — "lumen-gate" reward experiments + REFRAME (2026-05-28 → 05-29)

**The planned v7 hinge-stiffening was NOT done.** The user proposed a better direction: instead of crippling the scope's articulation to stop v6's curl-into-a-ball cheat, remove the *incentive* and the *physical possibility* of cheating while leaving articulation full. We built and ran three variants of a "forward-look gated reward" (collectively v7/v7b/v7c). **All three failed to learn**, but the failures were diagnostic and led to a clean reframing (below). The scope generator (`generate_videoscope_one_section.py`) was left UNTOUCHED — so the v1 baseline is unaffected.

### What was built (all in `Stage4/v2_collision/`, currently still in the code)
- **`scope_colon_env.py`** gained a forward-look "aim" signal: `aim = open_factor × fwd_factor`, both in [0,1].
  - `open_factor` = mean normalised depth in the central 25% of the tip-cam view (1 = open lumen, 0 = wall). Read from a cached depth image (no 2nd render).
  - `fwd_factor` = `dot(camera_forward_world, centreline_forward_tangent)` at the tip — 1 looking down the tube ahead, 0 sideways, 0 backward. This is what blocks the "curl round and look BACKWARD down covered tube to farm progress" exploit.
  - Helpers added: `_compute_aim()`, precomputed `self.centreline_tangents` in `__init__`, `self._last_depth` cache, `self.adv_cmd` (accel-limited advance), `self.advance_slew`.
- **Reward changes (v7b/v7c form currently in the file):** progress gated — `r_progress = W_PROGRESS × new_territory × gate`, where `gate = clip((aim − 0.3)/(0.6 − 0.3), 0, 1)` is a steep ramp that reaches 1.0 when properly aimed (so legitimate forward motion gets FULL credit, mis-aimed/curled/backward motion gets ~0). Forward advance also scaled by `gate` (reverse left ungated). Small dense `r_look = W_LOOK × aim`. Waypoints register only while `gate ≥ 0.5`. Constants: `AIM_OPEN_LO=0.05`, `AIM_OPEN_HI=0.10` (=5 cm), `AIM_FWD_LO=0.0`, `AIM_FWD_HI=0.7`, `AIM_GATE_LO=0.3`, `AIM_GATE_HI=0.6`, `W_LOOK=0.0005`, `DEFAULT_ADVANCE_RATE_MAX=0.001` (1 mm), `DEFAULT_ADVANCE_SLEW=0.1`.
- **`build_collision_scene.py`:** slide-joint `damping` `1.0 → 0.3` (softer base — chain lags/buckles at a wall instead of the kinematic base forcing it through). Everything else (0.5 mm walls, kinematic mocap drag, 1-DOF slide) unchanged.
- **`train_ppo.py`:** `N_ENVS` 8→10 (10 envs paged only *lightly* on the 16 GB laptop — ~86 pages/s, throughput held at ~410 fps; NOT the throughput-collapsing thrash that 12 envs caused). `TOTAL_TIMESTEPS=500_000`. `ent_coef` 0.015→0.003 (in v7c).

### Run history (this session)
| run | change | result | diagnosis |
|---|---|---|---|
| v7 | aim machinery, raw `aim` multiplier, 0.5 mm advance | progress flat ~0.3%, std rising 1.01→1.08 | `AIM_OPEN_HI=10 cm` too deep for this curved tube → `aim` capped ~0.5 → progress reward a permanent trickle |
| v7b | `AIM_OPEN_HI`=5 cm, steep `gate`→full credit, 1 mm advance, `W_LOOK` 0.0002→0.0005 | reward shape verified correct by smoke test, but still flat ~0.3%, std rising 1.01→1.18 | `ent_coef=0.015` too high for honest rewards (entropy term ~0.064/step vs task ~0.001/step) → entropy-dominated, never commits |
| v7c | `ent_coef` 0.015→0.003 | **std now FALLS** 0.992→0.957 (committing — entropy problem solved), BUT progress STILL flat ~0% through 143k | not entropy anymore → **hard exploration / credit assignment**; policy isn't discovering forward traversal |

All three were killed early (user ended v7c at ~143k). Smoke tests confirmed the aim/gate math works correctly: driving straight gives `aim=gate=1.0` (full 1 mm/step progress credit, `ncon=0`); reaching the first bend with no steering drops `gate→0` and cuts advance cleanly (no force-through), exactly as designed.

### THE REFRAME (user-driven, 2026-05-29) — this is the go-forward plan
The user critiqued the whole aim-shaping approach and is **correct**:
1. **Aim-shaping defeats the purpose.** Rewarding "point at the open lumen ~5 cm ahead" is hand-engineering the perception→aim mapping — exactly the analytic depth-centroid aim-point (the old `centre_on_depth` baseline). If that's baked into the reward, there's no reason to use RL. **Steering must EMERGE from the depth observation, not be told via reward.** → Remove `r_look`, the aim-gate, and aim-gated advance.
2. **Cheat-prevention belongs in PHYSICS, not reward.** The architecture already has the right mechanism: kinematic base drag + 1-DOF slide joint that lags up to 100 mm. A jammed/mis-aimed tip → chain lags → `actual_base_s` stops advancing → zero progress reward, automatically. **The only hole is wall PENETRATION** (v6's curled ball punched through the 0.5 mm prisms and kept advancing). Close that in physics and a pure-progress reward becomes uncheatable.
3. **Progress already counts base motion from outside the colon** (`base_s_init = tip_s_init − REST_LENGTH ≈ −52 mm`), so feeding the body in counts — keep that; removing the gate makes the easy entry-progress a clean bootstrap signal for "forward is good."

### Proposed next plan (agreed, NOT yet started)
1. **Strip the reward to pure outcome:** `W_PROGRESS × new_territory (high-water-marked) + W_GOAL − W_COLLISION × ncon` (+ optional waypoints). Delete the aim signal, `r_look`, the gate, and the aim-gated advance. Advance action drives the base directly; a mis-aimed tip jams and earns nothing. Aiming emerges.
2. **FIRST, verify the physics actually closes the cheat (do this before any retrain — it was skipped every prior run):** scripted test commanding max curl + max advance through a bend, measuring whether `actual_base_s` advances *through* the wall (cheat open) or the chain lags and stalls (cheat closed). Equivalent to watching v6's checkpoint in `viewer_ppo.py` under the new (damping 0.3) physics.
3. **If it still penetrates, harden the physics** (the genuine lever — none of these touch articulation): thicker collision walls and/or stiffer contacts (`solref`/`solimp`), or — most robust — make base advance **force-limited** instead of an infinitely-strong kinematic drag, so a jammed chain physically cannot be shoved forward regardless of wall thickness.

**Open decisions for next session:** (a) keep the **kinematic base drag** (and just harden walls) vs move to a **force-limited advance** (jams impossible by construction); (b) whether to keep waypoint shaping or go strictly progress+goal+collision.

### Housekeeping / disk state after this session
- Failed run dirs under `Stage4/v2_collision/runs/`: `ppo_v7_lumengate/` (ckpt_100000), `ppo_v7b_lumengate/` (ckpts to ~300k), `ppo_v7c_lumengate/` (ckpt_100000). All FAILED (non-learning) — safe to delete; do NOT report as baselines.
- Log files in `Stage4/v2_collision/`: `_v7c_train_stdout.log` (last run), plus `_v7*_train_*.OLD.log` archives and `_v7*_*.pid` files — all throwaway, safe to delete.
- v3/v6 cheat checkpoints still preserved (see earlier notes) — still NOT baselines.
- Launch note: training was run via `Start-Process` to a detached/visible console window with `python -u` (unbuffered) piped through `Tee-Object` to a log, so the user can watch live AND the log is monitorable. 10 envs is tolerable (light paging, fps held); 8 is the strictly-safe count.

---

## Session 2026-06-24 — v8 reward redesign + force-based termination

### What changed in `Stage4/v2_collision/scope_colon_env.py`

**Removed (all v7 aim-shaping machinery):**
- `_compute_aim()` method and all its supporting state (`centreline_tangents`, `_last_depth`, `adv_cmd`, `advance_slew`)
- All `AIM_*` constants, `W_COLLISION`, `W_WAYPOINT`, `W_LOOK`, `WAYPOINT_FRACTIONS`
- Aim-gated advance, aim-gated progress reward, waypoint milestones
- `slide_dof_id` (superseded by the contact force signal below)

**Added — insertion-force signal:**
- Force measured as the **sum of `mj_contactForce` magnitudes** across all active contacts per step. Chosen over `qfrc_constraint[slide_dof]` after calibration (see below).
- `MAX_INSERTION_FORCE = 30 N` — calibrated via `_calibrate_force.py`
- `force_norm = clip(total_contact_force / 30, 0, 1)` computed every step
- `force_norm` added to `state` obs at index 3 → state is now **(10,)** (was 9)
- Termination: `force_norm >= 1.0` → `termination_reason = "force_limit"`
- Per-step penalty: `-W_FORCE × force_norm` (W_FORCE = 0.01), linear ramp so grazing costs a little, jamming costs a lot, curl cheat terminates

**Reward (v8):**
```
r = W_PROGRESS × new_territory   (high-water-marked, no gating)
  + W_GOAL × 1[goal]
  - W_FORCE × force_norm         (linear 0 → 0.01 across force range)
```
No waypoints, no contact count, no aim signal. Steering must emerge from the depth observation.

### Force signal calibration (`_calibrate_force.py`)

Three candidate signals were tested under max-curl + max-advance (the v6 cheat):

| Signal | curl peak | gentle peak | ratio | Chosen? |
|---|---|---|---|---|
| `qfrc_constraint[slide_dof]` | 0.77 N | 0.11 N | 7× | No |
| `mj_contactForce` sum | 215 N | 0.17 N | **1242×** | **Yes** |
| `cfrc_ext` on scope_anchor | 0 N | 0 N | — | No |

Why the others fail: `cfrc_ext` is zero because contacts land on disk bodies, not the anchor. The slide constraint only fires when the chain lags behind the target (floor limit active) — the curl cheat moves the chain *ahead* of the target so the floor never fires and the signal reads ~0 N throughout. Total contact force is the only signal that sees the cheat.

With `MAX_INSERTION_FORCE = 30 N`: max curl terminates at step 8 (`force_norm = 215/30 >> 1`); gentle straight advance stays at `force_norm ≈ 0.006` throughout.

### `train_ppo.py` changes

- `N_ENVS = 20` (20 of 24 cores on the training machine; 4 left for main process + OS)
- `BATCH_SIZE = 256`
- `TOTAL_TIMESTEPS = 2_000_000`
- `CHECKPOINT_EVERY = 200_000`
- `RUN_NAME = "ppo_v8_force_progress"`
- `ent_coef = 0.003` (carried from v7c — calibrated for this reward scale)

### Transfer package

`Mujuco_V2/transfer/` contains the minimum 7 files needed to train on a remote machine:

```
transfer/
├── Stage4/v2_collision/
│   ├── train_ppo.py
│   ├── scope_colon_env.py
│   ├── build_collision_scene.py
│   ├── colon_generator.py
│   └── mesh_to_thin_prisms.py
└── Videoscopesimulations/
    ├── generate_videoscope_one_section.py
    └── new8.stl
```

Run from `Stage4/v2_collision/`. The `scenes/` subdirectory is created automatically. Required packages: `mujoco gymnasium stable-baselines3[extra] torch(+CUDA) numpy psutil`.

### Status at end of session

- v8 env + training config **complete and smoke-tested** (211 env-steps/sec on laptop, obs shape (10,) confirmed, force termination fires correctly)
- Training **NOT yet started** — transferred to 24-core machine
- v8 is the first run with no hand-engineered aiming — policy must infer steering purely from the depth observation, with the curl cheat closed by force termination rather than physics hardening

---

## Session 2026-06-01 — deliverable docs + dataset cleanup (no technical change)

No code or technical state changed this session. Two things happened:

1. **Deliverable docs written** at the repo root:
   - `REPORT.md` — concise (~3.5 page) supervisor progress report. **Attribution:** the
     scope sim (`Videoscopesimulations/`) and the photorealistic Blender colon
     (`colon_realistic_geometry_handoff_2026-05-26/`) are credited as *supervisor-provided*;
     the procedural colon model, CNNs, Stage 4 V1 loop, DA3 fine-tune, and collision RL
     are credited as the user's.
   - `USER_GUIDE.md` — operational "how to run / what each file does" guide.
   - A deliverable zip `..\Mujuco_V2_deliverable.zip` was built (NB: it predates the
     final doc edits — rebuild before sending).

2. **Large regenerable datasets/outputs DELETED from disk** (~1.4 GB) to slim the
   deliverable. The following folders no longer exist but are regenerable from the
   seeded scripts that remain in the repo (see USER_GUIDE.md §2/§4):
   - `Perception/realistic_depth_field/dataset_finetune/` (incl. `_rejected/`)
   - `Perception/depth_regression/depth_dataset/`
   - `Perception/realistic_depth_field/predictions/` and `predictions_finetune/`
   - `Perception/realistic_depth_field/dataset/` (120-frame pilot)

   **KEPT:** all `.py`, all PPO run checkpoints under `Stage4/v2_collision/runs/`, the
   fine-tuned DA3 model (`checkpoints/da3small_finetune_head.pth`), the steering CNN
   weights, and the V1 run videos.

   ⚠️ Earlier sections of this document still describe the deleted dataset folders as
   present on disk — that is now stale. Regenerate (don't assume a dataset path exists).

---

## Future work / optimisation backlog

### Training throughput — port to Sample Factory

**Current bottleneck:** SB3's `SubprocVecEnv` uses a synchronous rollout. All workers must finish one step before any worker gets the next action. Fast workers block waiting for the slowest one, giving 30–60% per-core utilisation even with 20+ envs. Observed on the 24-core training machine at 492 fps / ~25 fps per env.

**Fix:** Port to **Sample Factory** (https://github.com/alex-petrenko/sample-factory). Workers run fully asynchronously — no sync barrier, shared memory instead of pipes, GPU update overlaps with env stepping. Typical result: 3–5× throughput improvement, 80–95% core utilisation.

**Scope of change:**
- `ScopeColonEnv` itself is unchanged — all env logic, reward, obs space stays identical
- Write a ~30-line Sample Factory env registration wrapper
- Replace `train_ppo.py` with a ~60-line Sample Factory training script (translate `ent_coef`, `n_steps`, `batch_size` to SF params)
- Estimated effort: 2–4 hours

**When to do it:** after v8 results are known. If v8 shows promising learning, port before v9 so iteration cycles drop from ~68 min to ~15–20 min. If v8 is flat and major env/reward changes are needed, the port fits naturally into that rework anyway.

### Other deferred optimisations
- **Depth resolution 64→48:** clean speedup (~1.7× env step), low risk. One-line change in `scope_colon_env.py`.
- **`--resume` flag in `train_ppo.py`:** add CLI args for `--resume`, `--n_envs`, `--batch_size` so training can be reconfigured without editing the file.
- **DA3 backbone unfreeze (Phase 2.5):** ~30 min training, could push depth val MAE from 3.28 mm toward ~1.5–2 mm. Orthogonal to RL work.

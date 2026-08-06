> ⚠️ **Harvested 2026-08-06 from the v6_dr portfolio snapshot (`colonoscope-rl`, last touched
> 2026-06-30). NOT yet refreshed for v8_p1.** It describes the kinematic-base v6 line — no
> physical flexible shaft, no roller/capstan feeder, and the pre-v7 observation design. Anything
> here is superseded by [`../CURRENT_PLAN.md`](../CURRENT_PLAN.md) and
> [`../ITERATION_HISTORY.md`](../ITERATION_HISTORY.md), which win in any disagreement.

# System Architecture

## Overview

The system couples three independently developed pipelines — perception, simulation, and control — into a single end-to-end RL training loop, then deploys the trained policy using only onboard sensor data.

### Perception pipeline (run once to produce error_map_64.npy)

```
blender_depth_dataset.py       →  1,500 Blender RGB + depth pairs  →  dataset_finetune/
        ↓
filter_void_frames.py          →  cleaned training dataset (removes black-pixel frames)
        ↓
finetune_da3.py                →  da3small_finetune_head.pth

blender_depth_dataset_eval.py  →  500 unseen RGB + depth pairs     →  dataset_eval_500/
        ↓
run_da3_inference.py \
  --dataset_dir dataset_eval_500 \
  --output_dir predictions_eval_500 \
  --checkpoint checkpoints/da3small_finetune_head.pth
        ↓
compute_error_map.py           →  error_map_64.npy  (64×64 spatial noise map)
```

### RL training pipeline (consumes error_map_64.npy)

```
colon_generator.py         →  procedural 700 mm colon meshes (visual + collision STL)
        ↓
build_collision_scene.py   →  assembled MuJoCo XML scene
        ↓
scope_colon_env.py         →  Gymnasium environment
  · renders 64×64 depth from tip_cam
  · injects error_map_64.npy noise into depth obs each step
  · state obs: [cmd, force_norm, net_fx, net_fz, last_action, tip_contact]
        ↓
train_sf.py  (APPO, 80 parallel envs, GRU 512D)
        ↓
policy checkpoint  →  best_000000728_2981888_reward_1.238.pth
```

### Deployment

```
Real colonoscope camera
        ↓  RGB frame
DA3-SMALL  (base weights + da3small_finetune_head.pth)
        ↓  metric depth image
APPO policy  (GRU 512D + NatureCNN encoder)
        ↓  action: [steer_x, steer_y, advance]
PD tendon controller  (Kp=0.06, Kd=0.30)
        ↓  motor commands
Scope tip
```

---

## Component detail

### Procedural colon environment (`navigation/colon_generator.py`)

Each training seed produces a unique 700 mm colon geometry with 2–4 bends (80–150°), a random tube radius (18–30 mm), and 8–24 haustral folds. Two meshes are generated from the same parametric surface:

- **Visual mesh** (500 axial × 36 radial rings) — what the tip camera renders against; includes a 200 mm straight distal tail so the camera does not see an artificial void near completion.
- **Collision mesh** (125 axial × 20 radial rings) — used for contact physics. Coarser to hit the ~1,500 physics steps/sec target. Converted to per-triangle thin prisms by `mesh_to_thin_prisms.py`, which lets MuJoCo resolve contacts against the inner wall surface rather than approximating it with convex hulls.

### Videoscope model (`simulation/videoscope/`, bundled into `navigation/`)

25-hinge tendon-driven articulating section (9 mm OD, 2.5 mm disk spacing). Four antagonistic tendon pairs controlled via affine-gain actuators. A 1-DOF slide joint with dynamic floor constraint prevents the scope backing out of the colon entrance, which caused a "drag-and-flip" orientation exploit during early training.

Run the interactive manual viewer to explore the scope mechanics:
```bash
cd simulation/videoscope
python manual_tendon_viewer.py
```

### Depth perception pipeline (`perception/`)

**Dataset generation** (`blender_depth_dataset.py`) — self-contained Blender Python script (runs as `blender --background --python blender_depth_dataset.py`). Procedurally generates colon geometry inside Blender, applies a physically-based Cycles material (subsurface scattering, specular IOR, mucosal texture), places a camera on a random centreline path, and renders RGB + metric depth pairs using an emission-shader depth pass. Requires Blender 4.0+ for the Principled BSDF node API.

**Fine-tuning** (`finetune_da3.py`) — head-only fine-tune of DA3-SMALL (DualDPT head, 3.87M / 34.3M params). Backbone frozen. Mixed scale-invariant log + L1 loss. Reduces zero-shot val MAE from 8.30 mm to 1.60 mm, validated on the held-out final geometry of the training set (frames 1200–1499).

**Error map** (`compute_error_map.py`) — computes a 64×64 spatial map of DA3's mean relative per-pixel error on held-out eval frames. This map (`error_map_64.npy`) is the bridge between the perception and RL pipelines: it is injected multiplicatively into MuJoCo depth observations during training so the RL policy learns to be robust to the same spatial noise pattern it will encounter when DA3 runs on real hardware.

### RL controller (`navigation/`)

**Environment** (`scope_colon_env.py`) — Gymnasium environment wrapping the MuJoCo scene. Observation space: `Dict{depth: (1, 64, 64), state: (10,)}`. Action space: `Box(-1, 1, (3,))` — steer X, steer Y, advance rate. Depth is rendered from `tip_cam`, normalised by the max value in the frame (not a fixed clip), and noise-injected via `error_map_64.npy`.

**Encoder** (`sf_encoder.py`) — custom NatureCNN processing the 64×64 depth image into a 512D feature vector. This is registered as a SampleFactory custom encoder and replaces SF's default CNN. The depth features are concatenated with the 10D state obs and fed into the GRU.

**Policy** (`train_sf.py`) — SampleFactory APPO with a 512D single-layer GRU. The GRU is necessary because the task is partially observable: a single depth frame does not disambiguate lumen direction at a haustral ring. The GRU integrates frames across the approach to build a directional estimate.

---

## How `error_map_64.npy` connects perception and navigation

This file is the key sim-to-real bridge and appears in both `navigation/` (bundled for the RL env) and is generated by `perception/compute_error_map.py`.

The pipeline that creates it:
1. Fine-tune DA3 on 1,200 Blender training frames (`finetune_da3.py`)
2. Generate 500 unseen eval frames on different geometry (`blender_depth_dataset_eval.py` → `dataset_eval_500/`)
3. Run inference on those 500 frames (`run_da3_inference.py --dataset_dir dataset_eval_500 --output_dir predictions_eval_500`)
4. Compute per-pixel mean relative error: `mean(|pred − gt| / gt)` across all 500 frames (`compute_error_map.py`)
5. Resize to 64×64 to match the RL depth resolution → `error_map_64.npy`

During RL training, at every observation step:
```python
noise = N(0, error_map_64) × depth_obs
depth_obs += noise
```

Noise is multiplicative so close walls get small absolute noise and open lumen gets larger absolute noise — matching how monocular depth error scales with distance.

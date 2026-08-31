# System architecture

How the whole thing fits together, at the level of "what talks to what and why". For the exact
constants a hardware implementation must match, see
[../hardware/README.md](../hardware/README.md). For *why* it ended up this shape, see
[ITERATION_HISTORY.md](ITERATION_HISTORY.md).

**This describes the `v8_p1` line**, the live one. `v7_shaft/` and `v6_dr/` are earlier
generations kept runnable; `navigation/README.md` says which is which.

---

## The three pipelines

Three pipelines were developed independently and meet at two artefacts: a **depth model** and a
**spatial error map**.

### 1. Perception — run once, offline

Produces the depth model used at deployment, and the error map used during training.

```
blender_depth_dataset.py      →  1,500 Blender RGB + depth pairs
        ↓
filter_void_frames.py         →  cleaned training set (drops black-pixel frames)
        ↓
finetune_da3.py               →  da3small_finetune_head.pth
                                 (head-only fine-tune of DA3-SMALL; backbone frozen)

blender_depth_dataset_eval.py →  500 unseen pairs, different geometry
        ↓
run_da3_inference.py          →  predictions
        ↓
compute_error_map.py          →  error_map_64.npy      ← the sim-to-real bridge
```

### 2. Reinforcement learning — consumes the error map

```
colon_generator.py         →  procedural colon, visual + collision STL
        ↓
build_collision_scene.py   →  assembled MuJoCo scene XML  (never cached — always regenerated)
        ↓
scope_colon_env.py         →  Gymnasium env, 25 Hz decisions
                              · renders 54×96 depth from the tip camera
                              · normalises per-frame, injects error-map noise
                              · state obs: 6 floats (§ Observation)
        ↓
train_sf.py                →  Sample Factory APPO, recurrent (GRU 512)
        ↓
policy checkpoint          →  best_000000940_3850240_reward_1.204.pth
```

### 3. Deployment — the loop that runs on hardware

```
scope tip camera
        ↓  RGB frame
DA3-SMALL + fine-tuned head          ⚠️ output is scale-RELATIVE, not metric
        ↓  depth → resize 54×96 → divide by frame max → clip [0,1]
APPO policy   (depth CNN + state MLP → GRU 512 → Gaussian head)
        ↓  take the MEAN, not a sample
        ↓  a = [bend_x, bend_z, feed_rate] ∈ [-1, 1]³
PD command shaper                     ← its output is also fed back as policy input
        ↓
4 antagonistic tendons  +  insertion feed
        ↓
scope tip
```

Two things in that diagram are easy to get wrong and both have bitten this project:

- **The depth is relative, not metric.** Each frame is divided by its own maximum, so the
  farthest visible point is always exactly 1.0. The policy reads depth *structure* and has no
  access to absolute distance. The fine-tuned model's 1.43 mm figure is scale-aligned against
  ground truth — it is not a distance accuracy available at deploy time.
- **Deployment must take the Gaussian's mean.** The network emits distribution *parameters*.
  Sampling is PPO exploration noise, and measured σ ≈ 0.627 on a ±1 action space dominates the
  command. Taking the mean cuts tip-command jitter by 91%.

---

## The control loop

One decision every **40 ms (25 Hz)**, matching the rate the real depth model can sustain
(measured: 31 fps). Inside each decision the simulator runs **80 physics substeps** at 0.5 ms.

| | |
|---|---|
| **Observation** | `depth (1, 54, 96)` + `state (6,)` |
| **Action** | `Box(-1, 1, (3,))` — X bend setpoint, Z bend setpoint, feed rate |
| **Episode cap** | 2500 steps = 100 s |
| **Max advance** | 1.2 mm/step = 30 mm/s |

### Observation

**Depth** — 54 rows × 96 columns, 16:9, from a `fovy = 67.7°` tip camera, which on 16:9 is
**100.0° horizontal**. Normalised by the per-frame maximum, then multiplicative spatially
structured noise is injected from the error map. Noise scales with depth value, so near walls
get small absolute error and open lumen gets more — matching how monocular depth error grows
with distance.

**State** — six floats, and it matters what they are:

| idx | Value | Kind |
|---|---|---|
| 0–1 | PD filter state, normalised by max pull per axis | software, not a sensor |
| 2–4 | previous **commanded** action | the policy's own echo |
| 5 | `tip_contact`, binary | the only exogenous signal |

Five of the six are software state that a deployment must **reproduce identically, not
measure**. Only `tip_contact` says anything about the world beyond the image.

### What is deliberately *not* observed

This is the design rule that shapes the whole system:

> **Only give the policy signals the simulator can model with genuine fidelity.** Otherwise the
> policy either overfits a fantasy signal or learns to ignore it as noise.

Removed under that rule, each for a specific reason:

- **Contact and actuator forces** (`force_norm`, `net_fx`, `net_fz`, removed 2026-07-10) — clean
  in sim; any physical equivalent is dominated by transmission friction the sim does not model.
- **Cable-pair length differential** (`ten_x_n`, `ten_z_n`, removed 2026-08-04) — the interesting
  case is *inverted* between sim and hardware. In sim a dead zone zeroes the actuator target, so
  the tendon never moves and the policy learns "inside the dead zone, nothing happened." If a
  real dead zone is cable slack, the motor turns and the encoder reads the full command while
  the tip does not move. A reversed signal is worse than a missing one.
- **True colon progress** — privileged sim information, used for reward and diagnostics only.

### Action, and what it drives

`a[0]` and `a[1]` are **absolute bend setpoints, not rates** (×6.795 mm and ×7.361 mm — the two
axes differ because they span 12 vs 13 joints). `a[2]` *is* a rate, accumulating into a
monotonic insertion-depth target.

The setpoints pass through a PD command shaper (τ = 214 ms, 95% settle 640 ms, no overshoot)
whose output is **also observation slots 0–1** — the filter sits inside the observation loop,
not downstream of it. Below that, each of the four tendons runs a length-tracking integral loop
every physics substep, with anti-windup, against a 10 N pull-only force rail. Cables cannot
push.

⚠️ **The PD gains do not rescale arithmetically with control rate.** Moving 100 Hz → 25 Hz by
multiplying the old gains by 4 puts the poles at |z| = 1.337 — a diverging oscillation. This was
verified, not hypothesised. Any future rate change must re-fit against the wall-clock step
response.

Insertion is friction-driven: position-controlled rollers grip the shaft and **can slip**.
Progress is therefore never read from the commanded value — it is measured from the true
physical position of the base link projected onto the colon centreline.

### Reward

```
reward = 4.0 · new_territory            (per metre, high-water-mark only)
       − 0.004/step while the tip is in wall contact
       − 0.05 on a no-progress ending
```

`new_territory` is high-water-mark advance, so going forward, back, and forward over the same
ground earns nothing the second time. **There is no action-smoothness term** — an action-rate
penalty was examined and deliberately rejected, because the per-step progress reward is only
~0.0048 and any weight large enough to matter would train a passive policy.

Episodes end on `shaft_exhausted` or `ejected` (terminated), or `stuck`, `tip_stuck`,
`max_steps` (truncated).

---

## The network

```
depth (1, 54, 96) ──> 3× Conv+ReLU → Flatten 13440 → Linear 512 ─┐
                                                                 ├─ concat 640 ─> GRU(512) ─> Gaussian head (3-D)
state (6,) ─────────> Linear 64 → Linear 128 ────────────────────┘                        └─> value head
```

Both input widths are **derived at construction**, not hardcoded — which is why the 64×64 →
54×96 and 8-D → 6-D changes needed no encoder edit.

**The GRU is load-bearing.** The task is partially observable: a single depth frame does not
disambiguate lumen direction at a haustral fold, so the recurrent state integrates frames across
the approach. Any deployment that re-instantiates the network per frame, or forgets the hidden
state between steps, **is not running the trained policy**.

*Forensic tip:* a checkpoint's `encoder.depth_cnn.net.1.weight` shape tells you which
observation geometry it was trained on — 13440 = current 54×96, 10816 = the old 64×64.

---

## The simulated hardware

**Colon** — each seed is a unique procedural geometry: bends, tube radius and haustral folds all
randomised. Two meshes come from the same parametric surface: a fine **visual** mesh the tip
camera renders against, and a coarser **collision** mesh converted to per-triangle thin prisms
by `mesh_to_thin_prisms.py`, so MuJoCo resolves contacts against the true inner wall rather than
a convex-hull approximation.

**Scope** — a 25-hinge tendon-driven articulating tip (9 mm OD) on a 44-link flexible shaft
(432.5 mm shaft, 495 mm total), fed by an entrance roller/capstan mechanism. This is the
defining difference from the `v6_dr` generation, which moved the scope base kinematically: here
push force must travel around bends through wall contact, so **buckling, slip and jamming are
real failure modes**.

Joint stiffness is `0.055 N·m/rad`, measured at **2.92 N** of cable tension for a full bend.
That figure is a correction: it was `0.5`, needing ~27 N, against a bench measurement of ~3 N.

**Domain randomisation**, drawn once per episode: actuator force gain (0.45–1.0), dead zone
(0–3% of max pull), command lag (0–1 steps). Two honest gaps, both recorded in code — DR never
randomises the command→deflection map (every episode still reaches 98–99% of commanded bend),
and the dead zone is stateless and zero-centred, modelling no backlash or slack take-up on
reversal.

---

## `error_map_64.npy` — the sim-to-real bridge

The single artefact connecting the two pipelines, and the project's main sim-to-real idea:
rather than injecting generic noise, inject *the measured spatial error structure of the actual
deployment model*.

1. Fine-tune DA3 on 1,200 Blender training frames
2. Generate 500 unseen eval frames on different geometry
3. Run inference on those frames
4. Compute per-pixel mean relative error `mean(|pred − gt| / gt)` across all 500
5. Resize to the RL depth resolution

During training, at every observation step:

```python
noise = randn(54, 96) * error_map * depth
depth = clip(depth + noise, 0, 1)
```

⚠️ **Known approximation:** the map was measured on the *old* square 85° camera and is resized
to 54×96, an anisotropic stretch (x×1.5, y×0.84). It is a smooth envelope, so this is a mild
distortion rather than a wrong signal — but it is an approximation, not a verified map at the
current geometry.

---

## Where the sim stops and hardware begins

The simulator states requirements on the rig; it does not describe it. The bring-up status,
measurements, and open mismatches live in [../hardware/README.md](../hardware/README.md), and
the authoritative list of what is still unresolved is
[CURRENT_PLAN.md](CURRENT_PLAN.md) §8.

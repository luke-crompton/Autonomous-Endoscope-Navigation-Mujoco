# Control-Loop Progress Report — Mujuco_V3 / v8_p1 — 2026-08-05

> **This is a dated snapshot, not a plan.** The single source of truth for where the project is
> and what happens next remains `docs/CURRENT_PLAN.md`. If this document and that one disagree,
> **CURRENT_PLAN.md wins** — this one only records the state of the control loop on the date in
> its title and does not decide anything.

**Snapshot date: 2026-08-05.** Written to be self-contained: it assumes the reader has **no
access to the repository**. Every number below was read directly out of the live code in
`navigation/v8_p1/` on this date, with file and line given so any claim can be traced.

**Scope of this document.** It describes *what the simulator does* — the control loop the
policy was trained inside, the exact tensors it consumes, the exact vector it emits, and the
arithmetic the environment performs between them. Where it says something about hardware, it
is stating **a requirement the sim imposes** ("the rig must supply X in form Y"), never an
assumption about how the physical rig currently behaves. Sections marked *"What the physical
implementation must provide"* are requirements derived from sim; they are not a description of
anything measured on hardware.

---

## 1. Status in five lines

1. The policy is a **trained recurrent PPO agent** (Sample Factory), checkpoint
   `best_000000940_3850240_reward_1.204.pth`, retrained from scratch 2026-08-04 against six
   simultaneous changes to physics, observation geometry, observation content and control rate.
2. Training reported **~97% mean progress** at 4.52M of a 5M step budget. That is a *training*
   statistic computed from **sampled** rollouts (see §7.2). **No evaluation run has been done on
   this checkpoint.**
3. The control loop now runs at **25 Hz** in sim (40 ms per decision), matching the intended
   deployment rate exactly. It used to be 100 Hz.
4. The observation is **one depth image + six scalars**. Everything in the six scalars is either
   the policy's own echo or a binary contact flag — there is no force, tension, current, or
   privileged pose information in it, by explicit design rule.
5. The action is **three numbers in [-1, 1]**: two cable-pair bend setpoints and one insertion
   feed rate.

---

## 2. Loop timing — what "one step" means

| Quantity | Value | Source |
|---|---|---|
| MuJoCo integrator timestep | 0.5 ms | scene XML |
| Physics substeps per decision | **80** | `scope_colon_env.py:138` `DEFAULT_PHYSICS_PER_STEP` |
| Wall-clock per decision | **40 ms → 25 Hz** | 80 × 0.5 ms |
| Fresh depth frames | **every step (25 Hz)** | `scope_colon_env.py:208` `DEPTH_DECIMATION = 1` |
| Episode cap | 2500 steps = **100 s** | `scope_colon_env.py:135` |
| Max commanded advance | 1.2 mm/step = **30 mm/s** | `scope_colon_env.py:136` |

The physics timestep itself was **not** changed when the control rate moved 100 Hz → 25 Hz;
only the number of substeps per decision. Physics cost per simulated second is identical
(2000 mj_steps/s either way) — the change bought decision-rate realism and a 4× shorter credit
assignment horizon, not speed.

`DEPTH_DECIMATION` exists because the loop previously ran at 100 Hz while the depth model could
only produce ~25 Hz; it held the previous frame for 3 of every 4 steps. Now that control is
itself 25 Hz, the mechanism is **inert but still in the code** (`= 1`), and can be raised again
if control and camera rates diverge.

---

## 3. Observation — exactly what the policy is fed

The observation space is a **dict of two entries** (`scope_colon_env.py:628-637`):

```
depth : Box(0, 1, shape=(1, 54, 96), float32)
state : Box(-inf, inf, shape=(6,), float32)
```

### 3.1 `depth` — (1, 54, 96), values in [0, 1]

Produced in `_observation()`, `scope_colon_env.py:1211-1236`:

1. Render the tip camera's depth buffer at **54 rows × 96 columns** (16:9).
   The camera is `fovy = 67.7°` vertical, which on 16:9 is **100.0° horizontal**
   (`build_collision_scene.py`, `TIP_CAM_FOVY`).
2. **Normalise by the per-frame maximum**, not by a fixed metric clip:
   `depth = clip(depth_raw / max(depth_raw), 0, 1)`.
   The farthest visible point in each frame is always exactly 1.0. This makes the obs
   **scale-relative, not metric** — the policy reads relative depth *structure* and has no
   access to absolute distance. (The env docstring mentions a `depth_clip_m`; the code does not
   use it. The code is what the policy was trained on.)
3. Add channel dim → `(1, H, W)`, PyTorch convention, not gym's `(H, W, C)`.
4. Inject **multiplicative, spatially structured noise**:
   `noise = randn(54, 96) * error_map * depth`, then re-clip to [0, 1].
   The error map is a measured envelope of depth-estimation error magnitude. Noise scales with
   depth value, so near walls get small absolute error and open lumen gets larger — matching how
   monocular depth error grows with distance.

⚠️ **Known approximation:** the error map was measured on the *old* square, 85° camera and is
resized to 54×96, which is now an anisotropic stretch (x×1.5, y×0.84). It is a smooth envelope,
so this is a mild distortion rather than a wrong signal, but it is an approximation, not a
verified map at the current geometry.

### 3.2 `state` — 6 floats

Built at `scope_colon_env.py:1252-1256`:

| idx | Name | Definition | What it is |
|---|---|---|---|
| 0 | `cmd_x_n` | `cmd_x_pair / MAX_PULL_X`, `MAX_PULL_X = 6.795 mm` | **Internal PD filter state**, normalised. Not a sensor. |
| 1 | `cmd_y_n` | `cmd_y_pair / MAX_PULL_Z`, `MAX_PULL_Z = 7.361 mm` | Same, other axis. |
| 2 | `last_action[0]` | previous step's **commanded** a[0], clipped to ±1 | Policy's own echo |
| 3 | `last_action[1]` | previous step's commanded a[1] | Policy's own echo |
| 4 | `last_action[2]` | previous step's commanded a[2] | Policy's own echo |
| 5 | `tip_contact` | `float(tip_contact_steps > 0)` — binary 0.0 / 1.0 | Only exogenous signal in the vector |

Two things are worth stating plainly, because they determine what the rig has to supply:

- **Five of the six are software, not sensors.** `cmd_x_n` / `cmd_y_n` are the state of a filter
  the environment runs internally (§5.2), and slots 2–4 are the previous action. None of them is
  read back from the simulated hardware. They must be **reproduced in software identically** on
  any deployment, not measured.
- **`tip_contact` is the only thing the policy learns about the world other than the depth
  image.** In sim it resolves as *any* contact between the last tip disk (the one carrying the
  camera, 9.0 × 1.68 mm) and the colon wall body — which includes the cylinder's side face,
  both flat end faces, and the joint ball. It is deliberately blind to tip-vs-shaft
  **self**-contact, so that tip curl-back does not read as a wall jam.

`tip_contact` is **not obs-only**: the same flag drives a per-step reward penalty and a
truncation condition (§6).

### 3.3 What was deliberately removed from the observation, and why

This is a design rule the project applies consistently, and it explains why the obs is as thin
as it is:

> **Only give the policy signals the simulator can model with genuine fidelity.** Otherwise the
> policy either overfits a fantasy signal or learns to ignore it as noise.

Removed under that rule:

- **`force_norm`, `net_fx`, `net_fz`** (removed 2026-07-10). Summed raw contact force and
  actuator-force readouts. In sim these are clean; any physical equivalent would be dominated by
  transmission friction the sim does not model.
- **`ten_x_n`, `ten_z_n`** — antagonistic cable-pair length differential, i.e. an encoder-like
  reading (removed 2026-08-04, taking the state vector 8-D → 6-D). Three reasons, documented at
  `scope_colon_env.py:248-289`:
  1. **Redundant in the normal case.** `MAX_PULL_X = 6.795 mm` against a pair range of
     `13.380 mm` — ratio 1.97, so the two normalisations land on the same scale and
     `ten_x_n ≈ cmd_x_n` whenever nothing is resisting.
  2. **The dead-zone case is inverted.** In sim a dead zone zeroes the actuator target, so the
     tendon never moves and the policy learns *"inside the dead zone, the encoder shows nothing
     happened."* If a dead zone on real hardware is cable slack, the motor turns and the encoder
     reads the full command while the tip does not move — the opposite reading. A reversed
     signal, not merely a weaker one.
  3. **Sim over-reports load.** Sim's position actuator visibly droops under resistance; a stiff
     closed-loop servo would hold position and draw current instead.

  Note this removed only the **observation**. Tendon length is still read every physics substep
  to drive the actuator control loop (§5.4).
- **True colon progress** is never observed. It is privileged sim information used only for
  reward and diagnostics.

---

## 4. The network — how the observation becomes an action

`sf_encoder.py` plus Sample Factory's actor-critic head.

```
depth (1, 54, 96) ──> Conv(1→32, k3, s2)  → ReLU     54→26  |  96→47
                      Conv(32→64, k3, s2) → ReLU     26→12  |  47→23
                      Conv(64→64, k3, s1) → ReLU     12→10  |  23→21
                      Flatten  10×21×64 = 13440
                      Linear(13440 → 512) → ReLU  ──────────────┐
                                                                ├─ concat → 640
state (6,) ─────────> Linear(6 → 64) → ReLU                     │
                      Linear(64 → 128) → ReLU ──────────────────┘
                                     │
                                     ▼
                          GRU, hidden size 512          (rnn_type=gru, recurrence=64)
                                     │
                        ┌────────────┴────────────┐
                        ▼                         ▼
              policy head → (mean, log_std)   value head → V(s)
                 3-D Gaussian
```

Both input widths are **derived at construction**, not hardcoded: the CNN sizes its flatten
layer from a `torch.zeros` probe, and the state MLP reads `obs_space["state"].shape[0]`. This is
why the 64×64 → 54×96 and 8-D → 6-D changes needed no encoder edit.

Useful forensic fact: a checkpoint's `encoder.depth_cnn.net.1.weight` shape tells you which obs
geometry it was trained on — **13440** = current 54×96, 10816 = the old 64×64.

**The GRU is load-bearing for the loop.** The hidden state must be carried from step to step and
zeroed **only** at episode start. Any deployment that re-instantiates the network per frame, or
forgets the hidden state, is not running the trained policy.

---

## 5. Action — what the agent emits, and the whole chain it drives

### 5.1 The action vector

`action_space = Box(-1, 1, shape=(3,), float32)` (`scope_colon_env.py:623`)

| idx | Meaning | Units after mapping |
|---|---|---|
| `a[0]` | X-axis cable-pair bend **setpoint** | ×6.795 mm = commanded pull, metres |
| `a[1]` | Z-axis cable-pair bend **setpoint** (called "y" internally) | ×7.361 mm = commanded pull, metres |
| `a[2]` | Insertion **feed rate**, signed | ×1.2 mm per step = ±30 mm/s |

`a[0]` and `a[1]` are **absolute position setpoints, not rates.** A rate-based action was
proposed and explicitly rejected: "command a pull amount" is the intended native interface.

`a[2]` is a **rate** — it accumulates into a monotonic insertion-depth target, see §5.3.

The action is clipped to ±1 on entry (`step()`, `scope_colon_env.py:895`).

### 5.2 Step 1 — the PD command shaper (this is a policy input, so it matters most)

`scope_colon_env.py:902-915`, run once per 25 Hz decision:

```python
target_x = a[0] * MAX_PULL_X          # 6.795 mm
target_y = a[1] * MAX_PULL_Z          # 7.361 mm

vel_x = cmd_x_pair - prev_cmd_x       # note: velocity of the FILTER, not of the tip
vel_y = cmd_y_pair - prev_cmd_y
prev_cmd_x, prev_cmd_y = cmd_x_pair, cmd_y_pair

cmd_x_pair = clip(cmd_x_pair + PD_KP*(target_x - cmd_x_pair) - PD_KD*vel_x, ±MAX_PULL_X)
cmd_y_pair = clip(cmd_y_pair + PD_KP*(target_y - cmd_y_pair) - PD_KD*vel_y, ±MAX_PULL_Z)

PD_KP = 0.1803        PD_KD = 0.0475        # scope_colon_env.py:153-154
```

Measured response of this filter at 25 Hz: **τ = 214 ms**, 95% settle **640 ms**, **0.000%
overshoot**, reversal slew 0.75×.

Three points that a reimplementation gets wrong easily:

- **`cmd_x_pair` after this update is what the policy observes** as `cmd_x_n`. The filter is
  inside the observation loop, not downstream of it.
- **The gains do not rescale arithmetically with rate.** Moving 100 Hz → 25 Hz by multiplying
  the old gains (0.06, 0.30) by 4 gives poles at |z| = 1.337 — **unstable, a diverging
  oscillation**. This was verified, not hypothesised. The current values were fitted against the
  wall-clock step response instead. Any future rate change must re-fit, not re-scale.
- **Order matters:** velocity is computed from the *previous* two filter states before the
  update, and the clip is applied inside the assignment.

### 5.3 Step 2 — insertion accumulator

`scope_colon_env.py:944-956`:

```python
ds_command = a[2] * 1.2 mm
target_insertion_depth = max(0.0, target_insertion_depth + ds_command)   # metres, monotonic floor at 0
roller_ctrl = -target_insertion_depth / 0.008          # radians; 8 mm roller radius
```

The rollers are position-controlled and grip the shaft by **friction**. They can slip. Progress
is therefore *never* read from this commanded value — it is measured from the true physical
position of the base link projected onto the colon centreline (`actual_base_s`,
`scope_colon_env.py:1004-1006`). The sim models advance **lagging** the command via slip, and
never **exceeding** it.

### 5.4 Step 3 — cable mapping and the inner control loop

Antagonistic mapping (`scope_colon_env.py:930-933`), four tendons in order px, pz, nx, nz:

```
ctrl[px] = +cmd_x        ctrl[nx] = -cmd_x
ctrl[pz] = +cmd_y        ctrl[nz] = -cmd_y
```

Then **80 physics substeps**, and inside *every* substep (`scope_colon_env.py:969-977`) each of
the four tendons runs a length-tracking integral loop:

```python
target_len     = REST_LENGTH - ctrl[pull_actuator]
length_error   = ten_length[tendon] - target_len
combined_force = actuator_force[P] + actuator_force[I]
saturated      = combined_force <= (-10.0 N + 0.25 N)     # ANTIWINDUP_FORCE_MARGIN
ctrl[integral_actuator] = 0.0 if saturated else length_error   # freeze accumulator = anti-windup
mj_step()
```

Actuator constants (`generate_videoscope_one_section.py`): `ACTUATOR_KP = 6000`,
`ACTUATOR_KI = 36000`, `forcerange = [-10, 0] N` — **pull only, cables cannot push**. Joint
`stiffness = 0.055 N·m/rad`.

That stiffness figure is the result of the most recent physics correction: it was `0.5`, which
required ~27 N of cable tension for a full bend. It is now `0.055`, measured at **2.92 N** for
100% of nominal bend on both axes, with zero drift and no folding over 40 simulated seconds.
The force rail came down 80 N → 10 N to match. Under the PD shaper — i.e. real training
conditions rather than an artificial step — peak tendon tension is **~7.8 N against the 10 N
rail**, so anti-windup never engages in normal operation.

### 5.5 Domain randomisation — training only, and what it does *not* cover

Drawn once per episode in `reset()` (`scope_colon_env.py:842-866`):

| Parameter | Range | Applied to |
|---|---|---|
| gain X / Y | 0.45 – 1.0 | actuator **force** output (both the P and I actuator) |
| dead zone X / Y | 0 – 3% of max pull | zeroes `ctrl` when the command is below threshold |
| command lag | 0 – 1 steps (0 – 40 ms) | ring buffer on the effective command |

Two honest limitations, both recorded in the code:

- **DR never randomises the command→deflection map.** The gain scales *force*, and with the
  integral actuator every episode still reaches 98–99% of commanded bend regardless. "Command
  0.5, get 0.5" has held in every episode ever trained.
- **The dead zone is stateless and zero-centred.** It models no slack take-up on direction
  reversal, and no hysteresis or backlash of any kind.

---

## 6. Reward and episode boundaries — what the loop was optimised for

Not part of a deployment loop, but it defines the behaviour:

```
reward = W_PROGRESS · new_territory        (W_PROGRESS = 4.0, per metre)
       − W_TIP_PENALTY  while tip in wall contact   (0.004/step = 0.1 /s)
       − W_STUCK_PENALTY on a no-progress ending    (0.05, terminal)
```

`new_territory` is **high-water-mark** advance in metres — going forward, back, and forward over
the same ground earns nothing the second time. Force-based reward discounts and lumen bonuses
exist in the code but are all zeroed.

| Ending | Type | Condition |
|---|---|---|
| `shaft_exhausted` | terminated | base of the shaft physically reached the feed rollers |
| `ejected` | terminated | scope backed out past its start by 50 mm |
| `stuck` | truncated | < 2 mm of new territory over a 75-step (3.0 s) window |
| `tip_stuck` | truncated | tip in wall contact for 38 consecutive steps (1.5 s) |
| `max_steps` | truncated | 2500 steps (100 s) |

**There is no action-smoothness term in the reward at all.** This was examined and an
action-rate penalty was deliberately **rejected**, on two grounds: the per-step progress reward
is only ~0.0048, so any weight large enough to matter would dominate it and train a passive
policy; and the deterministic action is already smooth (§7.2), so the failure such a penalty
prevents is not occurring.

---

## 7. The loop as it must actually run

### 7.1 Deployment loop, in order

```
                 ┌─────────────────────────────────────────────┐
                 │  once per 40 ms (25 Hz)                     │
                 └─────────────────────────────────────────────┘

  1.  DEPTH        camera frame → depth estimate → resize to 54×96
                   → divide by the frame's own maximum → clip [0,1] → shape (1,54,96)

  2.  STATE        assemble 6 floats:
                     [ cmd_x_pair/MAX_PULL_X,        ← filter state from the PREVIOUS iteration
                       cmd_y_pair/MAX_PULL_Z,
                       last_action[0..2],            ← the previous COMMANDED action
                       tip_contact (0.0 or 1.0) ]

  3.  INFERENCE    features = concat(CNN(depth) 512, MLP(state) 128)   → 640
                   h_t, out = GRU(features, h_{t-1})                    ← hidden state PERSISTS
                   a = policy_head(out).means                           ← the MEAN, not a sample
                   a = clip(a, -1, 1)

  4.  SHAPE        target = a[0:2] · [MAX_PULL_X, MAX_PULL_Z]
                   run the PD update of §5.2  → cmd_x_pair, cmd_y_pair
                   (these become the next iteration's obs slots 0 and 1)

  5.  DRIVE        cable X: pull +cmd_x_pair, release the opposing cable by the same amount
                   cable Z: pull +cmd_y_pair, release the opposing cable by the same amount
                   insertion: target_depth = max(0, target_depth + a[2]·1.2 mm)

  6.  ECHO         last_action = a       ← store the COMMANDED action, not anything measured
```

Step 6 is worth flagging: the sim stores the **commanded** action
(`scope_colon_env.py:1121`), so feeding back the command rather than a measured deflection is
faithful to what the policy was trained on.

### 7.2 Deterministic vs sampled — read this before judging any playback

The network emits Gaussian **parameters**, not an action. Sample Factory's `forward()` defaults
to `sample_actions=True`, so its `actions` output is a fresh draw `μ + σ·ε` — PPO's exploration
noise, redrawn every step. **Measured σ ≈ 0.627 on a ±1 action space**, so a sampled command is
dominated by noise.

| Measurement | Sampled | Deterministic (mean) |
|---|---|---|
| tip-command jitter, RMS mm/step | 0.985 | **0.093** (−91%) |
| mean \|a_xy\| (steering magnitude) | 0.468 | **0.034** |
| same obs ×5, action spread | [0.589, 0.376, 0.711] | **[0, 7e-9, 0]** |

Consequences:

- **Deployment must take the mean.** The viewer was changed to do this by default on 2026-08-04;
  before that it was showing training-mode noise and the policy looked violently unstable.
- The **~97% training figure was earned with σ ≈ 0.63 of command corruption injected**, because
  Sample Factory's training statistics come from sampled rollouts. The deterministic policy
  should be at least as good — the noise was not concealing a bad policy.
- **The evaluation script still samples.** Every archived eval number in this project, across all
  versions, was produced in sampling mode. An eval on this checkpoint has not been run at all.

**One unresolved question about the action itself.** Deterministic mean `|a_xy|` is only
**0.034** — about 3.4% of full bend. It is not yet established whether the policy is steering
efficiently with small precise corrections, or barely steering at all and letting shaft
compliance and colon geometry do the work. The second would transfer badly. The check is cheap
and specified but **not yet run**: log deterministic `|a_xy|` across full traverses and test
whether it *rises at bends*.

---

## 8. What a physical implementation must provide

Requirements the sim imposes. Nothing here describes hardware behaviour — it states what the
loop above needs in order to run at all.

| The policy consumes | Form it must arrive in | Nature |
|---|---|---|
| `depth` (1, 54, 96) | float32 in [0,1], **normalised by each frame's own maximum**, from a view equivalent to **100.0° horizontal × 67.7° vertical rectilinear**, 16:9 | perception pipeline |
| `cmd_x_n`, `cmd_y_n` | the PD filter of §5.2, reimplemented **exactly** — same gains, same update order, same clip, ticked once per 40 ms | **software, not a sensor** |
| `last_action[0..2]` | the previous **commanded** action | software |
| `tip_contact` | binary 0/1 | contact sensing |
| **The policy emits** | | |
| `a[0]`, `a[1]` | **absolute pull setpoints** in metres after scaling by MAX_PULL — a position command, not a rate | position/encoder control |
| `a[2]` | signed feed **rate**, full scale = 1.2 mm per 40 ms = 30 mm/s | insertion drive |

**Hard requirements before the loop can be trusted:**

1. **`MAX_PULL_X` and `MAX_PULL_Z` must be measured on the real scope.** The sim's 6.795 mm and
   7.361 mm are derived from sim geometry. They normalise `cmd_x_n` / `cmd_y_n` — *two of the six
   policy inputs* — and they scale the action into physical units. This is a correctness
   requirement, not a refinement. **Keep the two axes separate**: they differ because the X axis
   spans 12 joints and Z spans 13; do not collapse them to one number.
2. **The depth view must be rectilinear at 100° horizontal.** The sim camera and the depth
   model's training camera are the same camera by construction (both renderers are rectilinear
   16:9, and `fovy = 67.7°` on 16:9 gives exactly 100.03° horizontal). A wider real lens is
   necessarily barrel-distorted, and the correction is a perception-side remap to a
   100°-horizontal rectilinear virtual camera — a precomputed lookup table, one remap per frame.
   It changes only what the camera hands the depth model; it does not touch what sim renders.
3. **The GRU hidden state must persist across the loop** and reset only at the start of a run.
4. **Take the distribution mean, not a sample** (§7.2).

### 8.1 ⚠️ Camera frame — MUST BE MEASURED ON THE REAL SCOPE (added 2026-08-06)

**The scope camera already exists in hardware, so this is a measurement, not a design choice.**
Nothing below is a claim about how the physical camera is mounted — it states what sim does and
what must be tested against it.

**What sim does — certain, read straight off the model.** The tip camera is declared as a child
of the `disk_25` body (`build_collision_scene.py:862`):

```xml
<camera name="tip_cam" pos="0 0.0025 0" xyaxes="-1 0 0 0 0 -1" fovy="67.7"/>
```

No `mode` attribute, so MuJoCo defaults to `mode="fixed"` — rigidly welded to the tip body
frame, not tracking anything. **Nothing touches it at runtime.** The only two references in the
whole environment are one ID lookup at construction (`scope_colon_env.py:616`) and passing that
ID to `update_scene` (`:1214`). There is no write to `cam_pos`, `cam_quat` or `cam_xmat`
anywhere. **The sim camera is never rotated, and there is no roll stabilisation, horizon
levelling, or gravity-up reference anywhere in the loop.**

**Why that is the useful property.** The tendon holes live in the *same body frame* as the
camera (`hole_px_25` at body +X, `hole_nx_25` at −X, `hole_pz_25` at +Z, `hole_nz_25` at −Z).
Image and cables are therefore locked to one another, and **the relation between "where the
lumen is in the image" and "which cable steers toward it" is a constant of the system.** The
scope twists continuously as it rounds bends; the policy neither sees nor cares, because input
and output rotate together. This is why the 6-D state carries no orientation term and why
nothing breaks without one. Only a single **fixed roll offset** between the real camera and the
real cable axes can matter.

**The convention sim was trained under** (derived from `xyaxes` and MuJoCo's camera convention —
+X image right, +Y image up, looks down −Z; frame verified right-handed, so no mirroring):

| | Body frame | Cable pair | Joints / range / max pull |
|---|---|---|---|
| image **right** | body **−X** | `px`/`nx` | 12 joints, 120°, 6.795 mm |
| image **up** | body **−Z** | `pz`/`nz` | 13 joints, 130°, 7.361 mm |
| optical axis | body **+Y** (forward) | — | — |

Hence `a[0] > 0` → bends toward +X → **steers toward image LEFT**; `a[1] > 0` → bends toward +Z
→ **steers toward image DOWN**. Note also that the **wide** image axis (100° over 96 columns) is
the X pair and the **narrow** axis (67.7° over 54 rows) is the Z pair.

**THE TEST TO RUN. Do not measure the camera's mechanical roll and the cable positions
separately** — that yields two numbers with two error bars, when the quantity that matters is
the *relation* between them. Measure it directly:

> **Scope in free space. Pull each of the four cables in turn to a clearly visible bend, and
> record which way the image content translates.** Four pulls gives the complete image↔cable
> map — offset angle *and* all signs — with no calibration target, in exactly the frame the
> policy cares about.

**Acting on the result:**

- **Offset is a multiple of 90°** → fix in software with a **signed swap of `a[0]`/`a[1]`**.
  Relabelling the cables and rotating the command are the same operation here. No image
  rotation, no resampling, no aspect problem. **180° is exactly free.** 90°/270° costs the
  12-vs-13 joint asymmetry — `a[0]` would drive a 130° axis where sim trained it on 120°, about
  8% in pull and 10° in range.
- **Offset is NOT a multiple of 90°** → **it cannot be corrected in the action**, and the reason
  is the field of view rather than the network. The obs is **1.04°/column against 1.25°/row**, so
  a feature at 45° in the world does not land at 45° in the image: a rolled real frame is not a
  rotation of any frame sim can produce, and the distortion is already baked into the pixels the
  CNN reads. Secondary reasons pointing the same way: the CNN is not rotation-equivariant
  (convolutions are translation-equivariant, and the flatten→linear head discards even that);
  the bend map is itself anisotropic so a rotated command is not a rotated bend, with an
  elliptical reachable envelope that a rotated full-scale command can exceed; and the GRU's
  recurrent state goes out of distribution too. Fix the mount if possible, or rotate the frame
  **before** the downscale to 54×96 and absorb the crop there — this folds into the fisheye
  rectification remap §8 already requires for the 122°-vs-100° horizontal gap, so it is one LUT
  and no extra per-frame cost.

**⚠️ Trap, whichever route is taken.** Five of the six state slots are the action echo
(`last_action[0..2]`, plus the two PD filter slots derived from it). **They must be expressed in
the same frame as the image.** Rotating the outgoing command while feeding back the unrotated
one — or the reverse — hands the policy an image/echo pair that contradicts itself, a condition
it has never seen in training.

**Confidence.** That the sim camera never rotates is *structural and certain*. The specific sign
convention in the table above is **derived** from the XML and MuJoCo's camera convention, **not
measured** — the headless check (drive `a[0] = +1`, observe which way depth structure shifts) is
specified but **has not been run**. Run it before committing hardware wiring to the signs.

**Design rules already settled, which constrain any low-level implementation:**

- **Position/encoder control, not tension control.** This is decided.
- **Force sensing may drive the low-level loop and safety limits. It must never be a policy
  input.** Same rule that removed `force_norm` and the tension observations.
- If a tension-triggered safety governor is ever added underneath (e.g. slow the motor when line
  tension is high), the policy should be **robust to** it, not trained on it. That would,
  however, make an encoder observation deviate from command in a way sim already models, and
  would be a reason to revisit the removed `ten_*` inputs.

---

## 9. Known gaps in the sim's own control loop

Stated plainly, because they bound what the loop can currently claim.

1. **⚠️ The feed mechanism does not transport at slow insertion rates.** With the feed rate
   scaled to 10% (3 mm/s) and a fixed open-loop `[0, 0, 1]` action — **no policy involved** —
   the scope advances **9.45 mm and then stops permanently**, while the commanded insertion
   accumulator climbs to 84 mm. Roller actuator force stays at ~0.02–0.09 N and does not grow
   with the 74 mm position error. Same seed and action at full rate: 96 mm in 109 steps. So the
   friction-grip feeder transports at 30 mm/s and fails to transport at 3 mm/s.
   **Consequence: the sim cannot presently represent a slow, hand-paced insertion**, and the
   policy has only ever experienced 1.2 mm of visual progress per decision. Whether the GRU
   depends on that rate is **unknown and untestable until this is diagnosed.** Suspects: roller
   contact friction, solver contact parameters, actuator velocity gain. Note that fixing it may
   change behaviour at 30 mm/s and invalidate the current checkpoint.
2. **No transmission state.** The dead zone is stateless and zero-centred; there is no slack
   take-up on direction reversal, no hysteresis, no backlash, no pretension state. The sim has a
   pull-only force constraint, but no *slack* variable.
3. **The command→deflection map is never randomised.** See §5.5. The policy has never seen a
   tip that does not go where it is told.
4. **The depth error map predates the current camera geometry.** See §3.1.
5. **No evaluation on the current checkpoint**, and the one steering-quality question that would
   change how a failed trial is read (§7.2) is not yet answered.
6. **The camera roll offset is unmeasured (§8.1).** Sim's image↔cable relation is fixed and known
   by derivation, but the real scope's has never been tested against it. If it is off by a
   non-multiple of 90° it cannot be corrected in the action, only in the frame. The four-cable
   pull test settles it and has not been run.

---

## 10. Change log for the control loop, 2026-08-02 → 2026-08-04

Six changes shipped together in one retrain; all were verified by inspecting the trained
checkpoint's weight shapes, not assumed.

| | Change | Effect on the loop |
|---|---|---|
| **A** | Tendon force scale: joint stiffness 0.5 → 0.055, force rail 80 N → 10 N | Full bend now costs **2.92 N** instead of 26.57 N |
| **B** | Depth decimation mechanism added (25 Hz depth under 100 Hz control) | Superseded by F; now inert at `= 1` |
| **C** | Camera geometry: square 85° → **16:9, fovy 67.7°**; obs 64×64 → **54×96** | Sim camera and the depth model's training camera became the same camera |
| **D** | Shaft damping 0.3 → 0.05 (plus a hardcoded joint-damping bug fixed) | Traverse:relaxation time ratio 1.7 → 10.2 |
| **E** | Dropped `ten_x_n`/`ten_z_n`; state **8-D → 6-D** | Removed the one observation that was *inverted* against a real encoder |
| **F** | Control rate **100 Hz → 25 Hz** (80 substeps/step) | Matches the deployment rate; 11 step-denominated constants rescaled, PD gains **re-fitted, not scaled** |

Everything step-denominated was rescaled to hold its wall-clock meaning fixed — advance
0.3→1.2 mm/step (30 mm/s), episode cap 10000→2500 steps (100 s), stuck window 300→75 steps
(3.0 s), tip-stuck 150→38 steps (1.5 s), tip penalty 0.001→0.004/step (0.1/s). The
progress weight is per-metre and did **not** change. The ratio
`W_PROGRESS·ADVANCE / W_TIP_PENALTY` was verified unchanged at 1.200.

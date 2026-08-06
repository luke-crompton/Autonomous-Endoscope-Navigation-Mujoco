they # RL Endoscope Navigation — Progress Summary

---

## V5 — Baseline Autonomous Navigation

### What it does

V5 is a reinforcement learning agent that autonomously navigates a simulated flexible endoscope through a procedurally generated colon. The agent controls three outputs each timestep:

- **Bend X** — target deflection along the horizontal axis (via a pair of antagonistic tendons)
- **Bend Y** — target deflection along the vertical axis
- **Advance** — forward insertion rate

The agent operates at 100 mm/s equivalent physics time, with each policy step covering 10 ms of simulation.

---

### Observations

The policy receives two inputs at every step:

#### 1. Depth image — `(1, 48, 48)` float32
A single-channel depth image rendered from a camera mounted at the scope tip, normalised frame-relative (the farthest visible pixel is always 1.0). This allows the policy to read lumen geometry even when the scope is pressed against a wall, where a fixed-clip normalisation would produce a near-black image.

#### 2. State vector — `(10,)` float32

| Index | Value | Description | IRL valid? |
|-------|-------|-------------|------------|
| 0 | `cmd_x_n` | Current normalised X tendon command ∈ [−1, 1] | ✓ Motor encoder |
| 1 | `cmd_y_n` | Current normalised Y tendon command ∈ [−1, 1] | ✓ Motor encoder |
| 2 | `0.0` | **Progress placeholder — always zero.** True colon progress is privileged simulation information not available on real hardware; this slot is kept only for architecture compatibility. | — |
| 3 | `force_norm` | Total contact force / 30 N, clipped [0, 1] | ✓ Handle force sensor |
| 4 | `net_fx` | Net actuator force differential in X axis, normalised | ✓ Motor current |
| 5 | `net_fz` | Net actuator force differential in Z axis, normalised | ✓ Motor current |
| 6–8 | `last_action[0..2]` | The policy's own action from the previous timestep | ✓ Internal state |
| 9 | `tip_contact` | 0/1 flag: tip disk currently in contact with colon wall | ✓ Distal force sensor |

> The policy deliberately does not receive its position along the colon. It must navigate using visual information alone, matching deployment conditions where colon geometry is unknown.

---

### Architecture

```
depth (1, 48, 48) → Conv(1→32, s=2) → Conv(32→64, s=2) → Conv(64→64, s=1)
                  → Flatten → Linear(10816→512) → 512-D

state (10,)       → Linear(10→64) → ReLU → Linear(64→128) → 128-D

                                              [concat → 640-D] → GRU(512) → policy / value heads
```

The GRU recurrent memory allows the policy to integrate information over time — important for tracking lumen direction across frames where a single depth image may be ambiguous.

---

### Training setup

- **Framework:** Sample Factory APPO (asynchronous PPO)
- **Steps:** 5 million
- **Workers / envs:** 40 workers × 2 envs = 80 parallel environments
- **Colon geometry:** 1–3 bends, variable length up to 800 mm, randomly generated per seed

### Reward

```
r = W_PROGRESS × new_territory × (0.3 + 0.7 × (1 − force_norm))
  − W_TIP_PENALTY  (per step the tip disk contacts the wall)
```

Only *new-territory* advance earns reward — re-traversing ground already covered earns nothing. The force multiplier implicitly teaches the policy to centre in the lumen without a separate lumen-detection reward.

---

### Results

| Metric | Value |
|--------|-------|
| Mean progress along colon | 89.7% |
| Full completion rate | 44% |
| Max completion (best seed) | 100% |

The policy successfully navigates haustra rings and multi-bend colons. The remaining failures are predominantly tip-contact truncations on sharp bends where the scope deflection range (±90° per axis) is insufficient.

---
---

## V6 — Sim-to-Real Transfer

### Motivation

V5 was trained and evaluated entirely within simulation under ideal conditions: perfect depth observations, exact actuator response, and deflection angles that fall short of real hardware. V6 addresses these gaps to produce a policy that will transfer to a physical endoscope without requiring hardware calibration.

---

### Changes and reasoning

#### 1. Increased deflection range: 90° → 120° per axis
Real endoscopes deflect up to approximately 120° per bending axis. Training at 90° meant the V5 policy never learned manoeuvres that require the full hardware range — sharp in-vivo bends demand it. The hinge joint range in the MuJoCo model was updated to ±0.1745 rad (120°) per axis.

---

#### 2. Domain randomisation on actuator properties
On real hardware, the relationship between motor command and tip movement is not fixed. Friction increases at every shaft bend, tendons stretch, and command lag varies with insertion depth. Rather than attempting to measure and model these properties exactly, V6 randomises them per episode so the policy learns to adapt through visual feedback alone.

| Parameter | Range | Simulates |
|-----------|-------|-----------|
| Tendon gain X | 45–100% | Friction loss on X-axis tendons at shaft bends |
| Tendon gain Y | 45–100% | Friction loss on Y-axis tendons (independent) |
| Dead zone X/Y | 0–3% of max pull | Tendon slack before tip begins moving |
| Command lag | 0–2 steps (0–20 ms) | Mechanical inertia and tendon stretch delay |
| Advance gain | 65–100% | Shaft-body friction when pushing forward |

The policy never observes these parameters. It must infer the current episode's regime from visual feedback — for example, if the lumen does not respond to a commanded bend, the policy learns to push harder. This mirrors the only feedback available to an operator on real hardware.

---

#### 3. Depth image resolution: 48×48 → 64×64
The depth image was increased from 48×48 to 64×64 pixels. This provides more spatial detail for the convolutional encoder to extract lumen structure and reduces the per-pixel impact of observation noise: the same total noise energy is spread across 78% more pixels, improving signal-to-noise ratio.

---

#### 4. Observations (V6)

The depth image is now `(1, 64, 64)`. The **state vector is identical to V5** (10-D, same layout).

#### 5. Realistic depth observation noise from the fine-tuned DA3 network
A depth estimation network (DA3-small) was fine-tuned on endoscope imagery to a validation MAE of 1.60 mm (baseline: 8.30 mm). 500 unseen evaluation frames were rendered, predictions were compared to ground truth, and the spatial distribution of prediction errors was captured as a per-pixel relative-error map.

During V6 training, this map is used to inject noise into the depth observation:

```
depth_obs += N(0, error_map) × depth_obs
```

Noise is multiplicative so it scales with depth value — close walls receive small absolute noise, open lumen (large depth) receives proportionally larger noise. This is consistent with how monocular depth estimation error behaves in practice. The policy therefore learns to navigate under the actual uncertainty profile of the deployed depth network.

---

#### 6. Fixed colon length: 700 mm
In V5, colon length varied between seeds. This meant the maximum possible episode reward varied between seeds — a policy that advanced 600 mm in a 600 mm colon received the same reward as one that advanced 600 mm in a 750 mm colon, creating inconsistent training signals. V6 fixes all colons at 700 mm, normalising the reward scale across seeds.

Colon difficulty was also increased: 2–4 bends (was 1–3), with bend angles up to 150° (was 100°).

---

#### 7. Goal reward removed
V5 included a terminal bonus for reaching the colon end. IRL there is no well-defined endpoint at a set insertion depth — the clinician continues as anatomy permits. The bonus was also incentivising the policy to rush through haustra rather than navigate carefully. V6 removes it; finishing is still optimal because the high-water-mark progress reward accumulates continuously, but there is no spike bonus for arriving quickly.

A terminal stuck-penalty (`−0.05`) was added: if the episode ends due to no-progress or tip-contact truncation, a small penalty is applied to teach the policy that stopping anywhere in the colon is undesirable.

---

### Results

**V6 achieves 100% completion on every test seed**, compared to 44% for V5. The policy navigates all multi-bend 700 mm colons, including those with bends up to 150° — angles V5 could not handle. Crucially, this is achieved under domain randomisation, meaning the policy has learned to adapt to variable actuator characteristics through visual feedback rather than relying on a fixed sim-hardware correspondence.

---

### Summary comparison

| | V5 | V6 |
|---|---|---|
| Depth resolution | 48 × 48 | 64 × 64 |
| Hinge deflection | ±90° | ±120° |
| Domain randomisation | None | Gains, dead zones, lag, advance |
| Depth noise | None | DA3 spatial error map |
| Colon geometry | 1–3 bends, variable length | 2–4 bends, fixed 700 mm |
| Goal reward | Yes (W=1.0) | No — rushing incentive removed |
| Completion rate | 44% | 100% |
| Mean progress | 89.7% | — |

---
---

## Future Plans

### Simulation roadmap

The next phase of simulation improvements targets the largest remaining sim-to-real gap: the insertion shaft. Currently the scope body is represented as a single slide joint — the shaft does not exist in the model and advance force is transmitted perfectly. On a real endoscope a ~1.6 m flexible shaft connects the handle to the bending tip, and its mechanics dominate the insertion experience.

#### V7 — Flexible shaft model
Replace the single advance slide joint with a kinematic chain of short rigid links connected by soft angular joints, colliding with the colon wall. This enables:
- Geometry-dependent advance friction (the shaft bends around corners, increasing contact area and reducing force transmission to the tip)
- The domain randomisation advance gain becoming physically grounded rather than a blanket scalar
- Foundation for all downstream shaft mechanics

#### V8 — Looping detection and recovery
When the shaft pushes against a tight bend and tip resistance is high, the shaft buckles and forms a loop inside the colon rather than advancing the tip. This is the most clinically significant skill for trainee endoscopists. V8 would add:
- A force-feedback observation: the ratio of commanded advance force to actual tip advancement; a large discrepancy indicates a forming loop
- A looping penalty in the reward
- A jiggle recovery behaviour (brief withdrawal then re-advance to reduce loop)

#### V9 — Torsional transmission
Rotating the handle twists the shaft; torsional compliance means the twist propagates slowly to the tip, with lag scaling with shaft curvature. This phase adds handle rotation as a fourth action dimension and models the resulting torsional delay.

---

### Prerequisites for a real-world test

The following must be resolved before the trained policy can be deployed on physical hardware.

#### 1. Depth source

The policy expects a `(1, 64, 64)` float32 depth image normalised so the farthest visible pixel equals 1.0.

| Option | Status | Notes |
|--------|--------|-------|
| RGB endoscope + DA3-small (fine-tuned) | Ready in sim | DA3 runs ~100–200 ms/frame on CPU — **requires GPU or TensorRT export** for real-time use |
| Physical depth sensor | Hardware dependent | Simplest path if available; bypasses DA3 entirely |

#### 2. Obs vector hardware mapping

Each of the 10 state inputs must be sourced from real hardware:

| Obs slot | Sim source | Real hardware equivalent | Difficulty |
|----------|-----------|--------------------------|------------|
| `cmd_x_n`, `cmd_y_n` | Internal PD state | Motor encoder readback | Low |
| `force_norm` | Sum of MuJoCo contact forces / 30 N | Handle-mounted force/torque sensor | Medium — sensor hardware required |
| `net_fx`, `net_fz` | Actuator force differentials | Motor current differential (px − nx, pz − nz) | Low — available from motor drivers |
| `last_action[0..2]` | Previous policy output | Internal software state | Low |
| `tip_contact` | Tip disk geom contact flag | Threshold on distal motor current or simple pressure switch at tip | Low — binary signal only; no force magnitude required |

`tip_contact` is a binary 0/1 flag — it carries no force information. A simple threshold on motor current (current spike without corresponding tip movement) or a pressure switch at the distal end is sufficient.

#### 3. Control interface

The policy runs at ~100 Hz (10 ms per step) and outputs three continuous values in [−1, 1]. A hardware interface layer is needed to:
- Map `action[0]`, `action[1]` through the PD controller to motor position commands (same PD_KP=0.06, PD_KD=0.30 used in training)
- Map `action[2]` to a motor-driven insertion drive (if motorised) or display as an advance recommendation to the operator
- Apply the same normalisation as the simulation (max_pull = 24 mm for tendon commands)

#### 4. Domain randomisation calibration

V6 trains over randomised gain, dead zone, and lag ranges that are engineering estimates. Before deployment the actual ranges should be measured:
- **Gain calibration:** command a known tendon displacement in free air and at maximum insertion depth; the ratio of measured tip angle gives the real gain range
- **Dead zone calibration:** find the command threshold below which tip angle is zero at each insertion depth
- **Lag calibration:** step-response test — time from command to first tip movement

If the real hardware falls outside the trained DR ranges, the ranges can be updated and training continued from the V6 checkpoint.

#### 5. Safety layer

A hard-coded safety wrapper must sit between the policy and the hardware:
- Maximum insertion force cutoff (e.g. 30 N at handle) → emergency stop
- Maximum insertion depth limit
- Emergency stop button mapped to a reset
- Watchdog: if policy inference takes > 2× the expected step time, hold last action

#### 6. Shaft behaviour validation

The policy was trained without a flexible shaft model. Before a full colon test, a bench test should verify that the policy's visual adaptation (adjusting commands when lumen does not respond) is sufficient to compensate for real shaft friction and looping, or whether V7's shaft model is needed first.
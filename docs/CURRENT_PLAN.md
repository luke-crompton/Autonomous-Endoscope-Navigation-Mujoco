# Current Plan — Mujuco_V3

**Last updated: 2026-08-31.** This is the **single source of truth** for where the project is and
what happens next. If another document disagrees with this one, this one wins.

It replaces three documents, now in `docs/archive/superseded_plans/`:
`V8_PLAN.md`, `MODELLING_PROBLEMS.md`, and a standalone force/depth-rate plan. Everything from
them that survived scrutiny is carried forward below. See §10 for what was wrong with each.

---

## 1. Where the project is, in five lines

1. **The A–F retrain is DONE and the sim work is finished.** `runs_sf/v8_p1_v1` was retrained from
   scratch on the rig on 2026-08-04 against all six changes, and the user reports **~97% mean
   progress** on the final epoch. Run stopped at **4,521,984** of the 5M budget; best checkpoint is
   `best_000000940_3850240_reward_1.204.pth` (3.85M steps). ⚠️ That 97% is a **training** statistic,
   computed from *sampled* rollouts — see §6.3. **No eval has been run on this checkpoint yet.**
2. **The project is in real-hardware bring-up** (2026-08-02) and the next action is now a
   **physical rig trial** (§7). There is no further sim change queued before it.
3. **All six changes A–F shipped and are baked into the weights** — verified by inspection, not
   assumption: the checkpoint's `encoder.depth_cnn.net.1.weight` is (512, **13440**) = the 54×96
   16:9 obs, and `encoder.state_mlp.0.weight` is (64, **6**) = the 6-D state.
4. **The viewer now shows deployment behaviour, not training behaviour** (§6.3). This was the
   session's main finding and it changed how the policy looks entirely — see that section before
   drawing any conclusion from a viewer session.
5. **What comes after the rig trial is a further phase scoped from the physical-test results** —
   gravity and an anatomically-oriented colon are the likely content (§9), but nothing there is
   committed until the hardware shows which sim gap actually matters.

---

## 2. The hardware, and what the policy needs from it

The rig: videoscope tip, per-cable motors with encoders, tip contact sensing, cantilever force
bars on each cable, DA3 depth model running live at ~31 fps. Insertion is **hand-fed for now**.

| Policy needs | Source on the rig | Status |
|---|---|---|
| `depth` (1,64,64) | camera → DA3 → `to_obs_64()` | ⚠️ FOV/aspect uncalibrated (§8) |
| `cmd_x_n`, `cmd_y_n` | your own PD filter — **not a sensor** | software, must be reimplemented exactly |
| ~~`ten_x_n`, `ten_z_n`~~ | ~~cable encoder differential~~ | ❌ **REMOVED from the obs 2026-08-04** — §5 Change E |
| `last_action[0..2]` | previous action | software |
| `tip_contact` | Binary tip contact sensing | available from flex pcb |
| `a[0]`, `a[1]` → cable pull | position-controlled motors | ✅ native match |
| `a[2]` → insertion | **hand-fed** — becomes an operator cue | see §3 |

**`MAX_PULL_X/Z` must be re-measured on the real scope** (currently 6.795 / 7.361 mm from sim
geometry — do not reuse). Pull each cable to the mechanical bend limit and record the encoder delta.
X and Z genuinely differ (12 vs 13 joints) — do not collapse to one number. It normalises `cmd_x_n`
/ `cmd_y_n`, which are still policy inputs.

`TEN_X/Z_RANGE_M` (13.380 / 14.495 mm) **no longer needs measuring** — Change E removed the only
thing that used it. Take the pair differential anyway if you have the scope apart, in case the
encoder obs is ever reinstated (§5 Change E), but it is not on the critical path.

**`tip_contact` coverage — reviewed and accepted 2026-08-04, settled.** Sim resolves it as *any*
contact between `disk_25` (the last disk, 9.0 × 1.68 mm, the one carrying the camera) and the
`colon_wall` body — which includes the cylinder's **side face and both flat end faces**, plus the
joint ball. **The real sensor covers the cylindrical side face only.** Accepted on the grounds that
it is primarily a training signal, a trained policy should not be making tip contact at all, and
real-world contact would be along the side face anyway.

Two limitations recorded rather than fixed:
- A real **head-on** jam would be invisible to a side-only sensor — the precise case `tip_stuck`
  truncation exists to catch. Consider a front-face sensor later if hardware trials show head-on
  jamming.
- Sim deliberately excludes tip-vs-shaft **self**-contact (a 2026-07-10 fix, so tip curl-back does
  not read as a wall jam). A real contact film cannot tell the difference, so hardware will fire
  where sim reads 0.

Note this signal is not obs-only: it drives the `r_tip` penalty and the `tip_stuck` truncation.

---

## 3. Decisions made (2026-08-02) — these are settled

- **Position/encoder control, not tension control.** This resolves what the archived
  modelling-problems doc called "the single highest-leverage unresolved question." Consequence:
  the integral actuator stays, and sim's disturbance rejection is approximately right.
- **Force sensors are for the low-level loop and safety limits only. Never a policy input.**
  Tension at the motor is dominated by sheath friction, which sim does not model faithfully.
  This is the same rule that retired `force_norm` and `net_fx`/`net_fz` in Phase 1.
- **The action stays an absolute pull setpoint.** A rate-based action was proposed (archived doc
  §3.1) but is rejected — "command a pull amount" is the rig's native interface and the encoders
  make it exact.
- **Insertion is hand-fed.** `a[2]` is still emitted and still fed back as `last_action[2]` (the
  sim stores the *commanded* action, `scope_colon_env.py:917`, so this is faithful). Display it
  as an advance / hold / withdraw cue. Two rules: **cap feed at 30 mm/s** (full-scale `a[2]` is
  0.3 mm/step at 100 Hz), and **advance only when it commands advance** — the sim models progress
  *lagging* the command via roller slip, never *exceeding* it.
- **The deployment loop runs at 100 Hz with held depth frames**, and §5 Change B makes that the
  trained regime rather than an approximation.

---

## 4. What we learned recently

### 4.1 Tendon force is ~9× too high — the retrain driver

Bench measurement: **~3 N** of cable pull for full free-space bend, **~10 N** realistic maximum.
Sim needs ~27 N:

```
k = 0.5 N·m/rad, θ = 0.1745 rad, moment arm = 3.2 mm
T = (0.5 × 0.1745) / 0.0032 ≈ 27.3 N
```

The same tension acts at every joint the tendon drives, so ~27 N holds all 12 (X) or 13 (Z) at
their limit. Even a half-bend needs ~14 N, above the real ceiling. `ACTUATOR_PULL_FORCE_LIMIT`
is 80 N, 8× the real maximum.

**Independent confirmation from the code's own history:** 27.3 N against `ACTUATOR_KP = 6000`
implies 4.55 mm droop out of 6.795 mm max pull ≈ 33% of range. `generate_videoscope_one_section.py:105-107`
separately records that stiffness 0.5 "re-caps achievable range to ~35-45% of nominal." The
arithmetic and the 2026-07-12 empirical note agree.

### 4.2 Bowden tubes resolve the `ten_length` worry

The archived modelling-problems doc argued that sim's tendon spans the tip only (verified true —
`build_collision_scene.py:830` routes through `range(N_DISKS)` = 26 tip disks), so real proximal
reel-in would be contaminated by shaft path length change, at ~`r × θ` ≈ 3.2 mm × 3 rad ≈ **10 mm**
against a 6.8 mm max pull. It called this "structural" and "the killer."

**The cables run in Bowden tubes, which makes this wrong.** A Bowden cable is path-length
invariant: cable and sheath bend together, so cable length is set by the sheath's neutral-axis
length, which does not change under bending. The residual is governed by **liner clearance**,
not routing radius: ~0.2 mm × 3 rad ≈ **0.6 mm** — an order of magnitude smaller, and the
antagonistic differential cancels much of what remains.

**Consequence: sim's tip-only tendon routing is approximately the correct model of a
Bowden-tube scope**, and `ten_x_n`/`ten_z_n` are legitimate observations. No action needed.

**What survives from that document** (still real, unfixed gaps):
- **Sheath friction, hysteresis and backlash** — now the *dominant* transmission gap, and Bowden
  makes it worse: friction rises with sheath curvature, so it is worst in tight bends where
  steering matters most. Sim's 0–3% dead zone is stateless and zero-centred; it models no slack
  take-up on direction reversal, which is the dominant Bowden effect.
- **Slack and pretension** — cables must be pretensioned at zero bend or the tip flops. Sim has
  no slack *state*, only a pull-only force constraint.
- **DR does not randomise the command→deflection map.** The gain scales actuator *force*, so with
  the integral actuator every episode reaches 98–99% of commanded bend regardless. The policy has
  never seen a miscalibrated tip. Candidate future DR axis, not in this retrain.

### 4.3 Depth cannot run at 100 Hz

Deployment target is 25 Hz; the measured DA3 runtime is 31 fps / 32.18 ms. The env renders every
step at 100 Hz. Fixed by Change B below.

---

## 5. THE PLAN — next retrain, six changes

All invalidate the current checkpoint, so they share one run.

| | Change | Status |
|---|---|---|
| **A** | tendon force scale | ✅ **IMPLEMENTED + verified 2026-08-02** |
| **B** | depth 25 Hz / control 100 Hz | ✅ **IMPLEMENTED + verified 2026-08-02** |
| **C** | camera geometry (FOV/aspect) | ✅ **IMPLEMENTED + verified 2026-08-04** |
| **D** | shaft timescale (damping) | ✅ **IMPLEMENTED + verified 2026-08-02** |
| **E** | drop `ten_x_n`/`ten_z_n`, state 8-D → 6-D | ✅ **IMPLEMENTED + verified 2026-08-04** |
| **F** | control rate 100 Hz → 25 Hz | ✅ **IMPLEMENTED + verified 2026-08-04** |

**Advance rate is deliberately unchanged** (2026-08-02 decision): episodes stay ~10 s, and the
traverse/relaxation ratio is corrected via Change D instead. See Change D for why that is
equivalent and free.

### Change A — tendon force scale ✅ IMPLEMENTED

**Measured result: 26.57 N → 2.92 N** at 100% of nominal bend, both axes. Target was ~3 N.
Zero drift and no folding over 40 sim-seconds (5–6× a real episode), so the S-shape fix survives
the 9× stiffness reduction as predicted. Full numbers in §6.

**Do not simply delete the stiffness.** It is the 2026-07-12 S-shape fix: a tendon is one length
constraint on 12–13 DOF, so at `k=0` the chain's shape is underdetermined and drifts into a folded
S over many seconds. But the uniform-bend equilibrium (`T·r = k·θᵢ` ⇒ all `θᵢ` equal) is unique
for **any** `k > 0` — only the restoring gradient scales. Settling goes from ~0.07 s to ~0.64 s
(τ ≈ damping/stiffness), still well inside a 6–7.5 s episode. **That is an argument, not a
measurement — test A2 decides it.**

Target: `k = 3 × 0.0032 / 0.1745 = 0.055 N·m/rad` (9.1× reduction). Side effect: droop falls to
~0.5 mm (7%), so the range-cap problem `ACTUATOR_KI` exists to solve largely dissolves.

| File | Constant | From | To |
|---|---|---|---|
| `generate_videoscope_one_section.py` | joint `stiffness` (now `JOINT_STIFFNESS`) | `0.5` | **`0.055`** |
| `generate_videoscope_one_section.py` | `ACTUATOR_PULL_FORCE_LIMIT` | `80.0` | **`10.0`** |
| `generate_videoscope_one_section.py` | `ACTUATOR_KI` | `36000.0` | **unchanged** — re-verified, see below |
| `scope_colon_env.py` | `ANTIWINDUP_FORCE_MARGIN` | `2.0` | **`0.25`** |

**`ACTUATOR_KI` was kept at 36000.** The worry was that it had been sized against a droop about
to shrink 9× and would now overshoot. Test A3 says it does not: 0.000° overshoot, no oscillation,
98.4%/98.6% of nominal bend. Under the env's outer PD shaper — i.e. real training conditions, not
an artificial step — peak tendon tension is ~7.8 N against the 10 N rail, so **anti-windup never
trips at all**; it only engages on a step command (2.7% of substeps). The rail is not binding in
normal operation.

The stiffness is now a named constant `JOINT_STIFFNESS` rather than a literal in the XML f-string.

`ANTIWINDUP_FORCE_MARGIN` is a margin *below the force rail*: 2 N under 80 N is 2.5%, but would
become 20% of a 10 N rail. `ACTUATOR_KI` was sized against a droop about to shrink 9× — re-derive
empirically, do not scale by rule of thumb. The comment blocks at `:64-74` and `:97-112` document
reasoning this supersedes and need rewriting alongside the numbers.

**Fallback if drift returns at k=0.055:** do *not* raise `k` back. Add joint `frictionloss` —
dissipative, resists the slow creep, costs a constant tension offset rather than one proportional
to θ.

### Change B — depth at 25 Hz, control at 100 Hz ✅ IMPLEMENTED

Keep control at 100 Hz; render every 4th step and hold the previous frame. Everything in the env
is per-step (PD gains `0.06`/`0.30`, 0.3 mm advance, GRU hidden state), so slowing the whole loop
to 25 Hz would silently multiply per-step response by 4×. Staleness cost is small: at 0.3 mm/step,
four held steps smear ≤1.2 mm of tip travel.

Three details that matter more than the mechanism:

- **Cache the post-noise array.** A real held frame is the same pixels *including the same
  errors*. Re-sampling `error_map_64` noise each step lets the policy average it down and recover
  ~100 Hz of information — exactly what this change removes. Cache after `scope_colon_env.py:1020-1025`.
- **Randomise the phase per episode.** At fixed phase the GRU can learn "4 steps since the pixels
  changed ⇒ fresh data next" — a clock that does not exist when camera and control loop free-run.
  One RNG draw in `reset()`.
- **Interval left fixed at 4, not randomised.** 3–5 steps would cover 31 fps vs a 25 Hz target and
  is nearly free, but this retrain already changes the physics regime and a second novel DR axis
  makes failure ambiguous. Revisit after this trains.

```python
# near DEFAULT_DEPTH_RES (:116)
DEPTH_DECIMATION = 4        # 100 Hz control / 4 = 25 Hz depth, matching DA3 deployment

# reset()
self._last_depth = None                                   # forces a fresh render
self._depth_counter = int(_dr_rng.integers(0, DEPTH_DECIMATION))

# _observation()
if self._last_depth is None or self._depth_counter % DEPTH_DECIMATION == 0:
    ... render, per-frame-max normalise, inject error_map noise ...
    self._last_depth = depth
else:
    depth = self._last_depth
self._depth_counter += 1
```

`_lumen_score()` already reads `self._last_depth` and becomes stale-consistent. Inert either way
(`W_LUMEN_BOOST = 0.0`). `eval_sf.py` / `viewer_sf.py` inherit through the env.

Render calls drop 4× — **but this buys no throughput.** Measured A/B on the full scene (Windows,
`wgl`): 43.1 steps/s both with and without decimation, i.e. **1.00×**. Rendering is not the
bottleneck; contact resolution is (`ncon` mean 54, max 111). This is the same lesson as
"fewer bodies isn't faster" — cost here is contacts, not geometry or pixels. Do the change for
sim-to-real fidelity, which is the actual reason, and expect nothing for speed. (Unverified on
the Linux rig's EGL/GPU path with 40 workers, where the balance could differ.)

### Change C — camera geometry ✅ IMPLEMENTED

**The real lens was measured at 140° DIAGONAL** (2026-08-04), which unblocked this and resolved it
more cleanly than expected. Option **C2** was taken: sim renders 16:9, obs is now **96×54**.

| File | Constant | From | To |
|---|---|---|---|
| `build_collision_scene.py` | `TIP_CAM_FOVY` (new; was inline `fovy="85"`) | `85` | **`67.7`** |
| `scope_colon_env.py` | `DEFAULT_DEPTH_RES` | `64` (int, square) | **`(54, 96)`** (H, W) |

**Deriving 67.7 from 140° diagonal.** A 140° lens is necessarily barrel-distorted — a rectilinear
lens that wide does not exist at this scale — so the **equidistant** mapping `r = f·θ` applies,
not `r = f·tan θ`. On 16:9 that gives **H = 122.0°, V = 68.6°**. (The rectilinear reading would
give V = 106.8°; it is the wrong model and would have been a 38° error on the one number MuJoCo
takes.)

**Why 67.7 and not the measured 68.6.** MuJoCo is rectilinear, and so is Blender. Setting
`fovy = 67.7` on a 16:9 render gives back exactly **H = 100.0°** — which *is* the DA3 fine-tune's
Blender camera (`blender_depth_dataset.py`, `--fov_deg 100.0` on 320×180). So the sim camera and
the depth model's training camera become the same camera, and the vertical also matches the real
scope to within 0.9°. One number satisfies both constraints; that is not a coincidence, it falls
out of both renderers being rectilinear 16:9.

**Why 96×54 and not 64×36.** The obs shape had to change either way (square → 16:9). 64×36 would
have *coarsened* what the policy sees vertically — 1.88°/row against the old 64×64's 1.33°.
96×54 gives 1.25°/row and 1.04°/col, i.e. slightly finer than before on both axes. Affordable
because the C1 measurement showed rendering is not the bottleneck (1.00× from cutting renders 4×;
contacts dominate).

**No encoder change was needed** — `sf_encoder.py:80-81` sizes the flatten layer from a
`torch.zeros` probe. Verified: flatten goes 10816 → 13440, output stays 512-D.

**What this does NOT fix — and it is not an aspect problem any more.** Horizontally the real frame
covers 122° where sim and the DA3 training set cover 100°. That 22° gap *is* the lens's barrel
distortion. Correcting it is a **perception-side** job: fisheye-calibrate the camera
(`probe_camera.py --measure_fov checkerboard`, which writes `K` and `D`) and remap the live frame
to a 100°-horizontal rectilinear virtual camera before DA3. Until then, objects near the left and
right edges sit closer to centre than the policy expects; centre of frame, where the lumen usually
is, is unaffected. **This does not block the retrain** — see §8.

Note this inverts the earlier worry that a non-100° lens would force a Blender re-render plus
re-fine-tune (hours). Because Blender and MuJoCo are both rectilinear, the odd one out is the real
camera, and rectifying *it* is a `cv2.remap` LUT rather than a re-render.

### Change D — shaft timescale ✅ IMPLEMENTED

**The problem.** A full traverse is ~10.2 s of sim time (305 mm of task travel at the 30 mm/s
full-command advance). The shaft's own relaxation time was `tau = c/k = 0.3/0.05 = 6.0 s`.
Ratio **1.7** — the shaft never reached equilibrium during an episode; it was dragged through the
colon permanently lagging and stressed. A real colonoscope covers the same 305 mm at ~3-5 mm/s in
~61 s, a ratio of **~10**, i.e. near-quasi-static at every instant. The sim was in the wrong
dynamic regime by ~6×.

**Why the fix is damping, not a slower advance.** Every hinge here is enormously overdamped —
shaft `zeta ≈ 2592`, tip `zeta ≈ 24`. The dynamics are viscous-elastic, not inertial, so they
depend only on the **ratio** of driving rate to relaxation rate, never on absolute wall-clock
time. The ratio can therefore be corrected from either end:

```
SHAFT_DAMPING 0.3 → 0.05   ⇒   tau 6.0 s → 1.0 s   ⇒   ratio 1.7 → 10.2   ✓ matches real
```

Free. Episode length, step count and advance rate all unchanged. `SHAFT_STIFFNESS` is untouched,
so the shaft's **static** shape under load is identical — this is purely a timescale change.
Slowing the advance 3× instead would have cost 3× the compute *and* 3× fewer episodes inside a
fixed Sample Factory step budget, losing colon-seed diversity.

The shaft stays hugely overdamped afterwards (`zeta ≈ 432`), so there is no ringing risk.

**A latent bug found while verifying this.** The `shaft_tip_ball` joint — the single ball joint at
the shaft/bending-tip junction — had `damping="0.3"` **hardcoded** in its XML emission. It looked
like it tracked `SHAFT_DAMPING` but never did; the 0.3 was just a stale copy. Left alone it would
have sat at `tau = 0.3/0.125 = 2.4 s`, becoming the slowest element in the whole scope and
dominating the exact ratio this change fixes. Now a named `SHAFT_TIP_BALL_DAMPING = 0.125`
(`tau = 1.0 s`, matching the shaft) so it cannot drift out of lockstep again.

| File | Constant | From | To |
|---|---|---|---|
| `build_collision_scene.py` | `SHAFT_DAMPING` | `0.3` | **`0.05`** |
| `build_collision_scene.py` | `SHAFT_TIP_BALL_DAMPING` (new; was inline `0.3`) | `0.3` | **`0.125`** |

Also corrected a stale comment on `SHAFT_LINK_MASS`: it claimed "~150 g total shaft mass", but
44 links × 5 g = **~0.22 kg**. Comment only, no behaviour change.

### Change E — drop `ten_x_n` / `ten_z_n`, state 8-D → 6-D ✅ IMPLEMENTED

**2026-08-04.** State obs is now `[cmd_x_n, cmd_y_n, last_action[0..2], tip_contact]`. This closes
the §8 open question ("do `ten_x_n`/`ten_z_n` earn their place?") — not with the planned ablation
but with a fidelity argument that makes the ablation moot.

**1. Redundant in the normal case.** `MAX_PULL_X` = 6.795 mm, `TEN_X_RANGE_M` = 13.380 mm — a ratio
of **1.97**. The pair differential is ~2× the single-cable pull, so both normalisations land on the
same scale and `ten_x_n ≈ cmd_x_n` in free space. All the information is in the *difference*, whose
only sources are the DR dead zone, the DR lag, contact resistance at the force rail, and the DR gain.

**2. The dead-zone case is inverted against hardware — this is what decided it.** In sim the dead
zone zeroes `ctrl` (`step()`: `_eff_x = 0.0 if abs(cmd) < dead`), so the actuator target never
moves, `ten_length` never moves, and the policy learns *"inside the dead zone the encoder shows
nothing happened."* On the rig a dead zone is cable **slack**: the motor turns, the encoder reads
the **full commanded value**, and only the tip fails to move. The encoder says the opposite of what
sim taught. Not a weaker signal — a reversed one.

**3. Sim over-reports load.** The position actuator (`KP=6000`, 10 N rail) visibly droops under
resistance; a stiff closed-loop servo holds position and draws current until it stalls. Sim's
`ten_*` therefore carries *more* load information than the real encoder will.

Same rule that retired `force_norm` and `net_fx`/`net_fz` in Phase 1: only give the policy signals
sim can model with genuine fidelity, or it overfits a fantasy or learns to ignore it as noise.

**Not a claim that cable encoders are useless on hardware.** If a tension-triggered safety governor
is added to the low-level loop (slow the motor when line tension gets high — permitted under §3,
which allows force for the low-level loop and safety limits but never as a policy input), the
encoder *would* start deviating from command in a way sim already models, and this is worth
revisiting. The policy should be **robust to** such a governor, not trained on it.

`data.ten_length` is untouched and still drives the per-substep integral-actuator loop.
`TEN_X_RANGE_M`/`TEN_Z_RANGE_M` are kept but marked unused — they are measured physical constants,
and §2 still lists them among the quantities to re-measure if this is ever reinstated.

⚠️ **`sf_encoder.py` had `STATE_DIM = 8` hardcoded** with a "must match `scope_colon_env.py`"
comment — exactly the kind of pairing that drifts silently. It now reads
`obs_space["state"].shape[0]`, the same approach the depth CNN already used. That footgun is gone.

### Change F — control rate 100 Hz → 25 Hz ✅ IMPLEMENTED

**2026-08-04.** `DEFAULT_PHYSICS_PER_STEP` 20 → **80**, so an env step is 80 × 0.5 ms = **40 ms**.
The MuJoCo timestep is untouched, so the physics itself is unchanged.

**Why.** After Changes A and D the system's own time constants are tip settling ~0.64 s and shaft
relaxation τ = 1.0 s. At 100 Hz that was ~64 control decisions per settling time — roughly 4×
oversampled against the 10-20× rule of thumb. 25 Hz gives ~16, comfortably inside. The obs agrees:
depth was already 25 Hz, and of the 6 state dims, `cmd_x_n`, `cmd_y_n` and `last_action[0..2]` are
the policy's **own echo** — only `tip_contact` is exogenous and fast.

**Everything step-denominated rescaled by 4× in the same edit:**

| Constant | From | To | Wall-clock invariant |
|---|---|---|---|
| `DEFAULT_PHYSICS_PER_STEP` | 20 | **80** | 2000 mj_steps/s |
| `DEFAULT_ADVANCE_RATE_MAX` | 0.3 mm/step | **1.2 mm/step** | 30 mm/s |
| `DEFAULT_MAX_STEPS` | 10000 | **2500** | 100 s cap |
| `STUCK_WINDOW` | 300 | **75** | 3.0 s |
| `TIP_STUCK_WINDOW` | 150 | **38** | 1.5 s (rounded up, so in-contact travel doesn't shrink) |
| `W_TIP_PENALTY` | 0.001 | **0.004** | 0.1 /s |
| `FORCE_LIMIT_STEPS` | 10 | **3** | diagnostic only |
| `DEPTH_DECIMATION` | 4 | **1** | now inert — control rate == depth rate |
| `DR_LAG_RANGE` | (0, 2) = 0-20 ms | **(0, 1)** = 0-40 ms | same mean lag, coarser |
| `STUCK_THRESHOLD` | 0.002 m | **unchanged** | metres over a fixed 3 s window |
| `W_PROGRESS` | 4.0 | **unchanged** | per-metre, not per-step |

**`PD_KP` / `PD_KD` do NOT rescale by arithmetic — this is the one part that needed measuring.**
The filter `c[n+1] = (1−KP−KD)c[n] + KD·c[n−1] + KP·target` has characteristic
`z² − (1−KP−KD)z − KD = 0`, and `KD > 0` requires one **negative** root (`KD = −z₁z₂`). That second
mode is a discrete alternating artefact with no continuous-time counterpart, so there is nothing to
preserve by scaling it. **Scaling both by 4 gives (0.24, 1.20) → poles [−1.337, 0.897] → |z| = 1.337
→ unstable**, a diverging oscillation. Verified, not hypothetical.

Fitted instead against the wall-clock step response by **`diagnostics/pd_rate_rescale.py`** (test
F1): `PD_KP` 0.06 → **0.1803**, `PD_KD` 0.30 → **0.0475**.

**⚠️ This buys no throughput, and `TOTAL_STEPS` needs a decision.** Physics cost per sim-second is
identical by construction and confirmed by measurement (50 steps/s × 20 substeps before, 12.5 × 80
now — both 1000 mj_steps/s). So an env step is 4× more expensive in wall-clock. What it buys is a
**4× shorter GRU credit-assignment horizon** (~255 decisions per traverse instead of ~1020), a 4×
smaller step budget for the same experience, and an exact match to the deployment rate.
`v8_p1_v1` got 10M steps at 100 Hz ≈ 100,000 sim-seconds ≈ 9,800 episodes. The equivalent at 25 Hz
is **~2.5M steps**; leaving `TOTAL_STEPS` at 10M runs ~4× longer than the last run. Flagged in
`train_sf.py` and **left unchanged pending a decision** — SF checkpoints continuously, so headroom
is cheap and the run can be stopped on the high-water mark.

### Files changed

**A, B, D (2026-08-02):** `navigation/v8_p1/generate_videoscope_one_section.py`,
`navigation/v8_p1/scope_colon_env.py`, `navigation/v8_p1/build_collision_scene.py`.

**F (2026-08-04):** `scope_colon_env.py` (11 rate constants, table above) and `train_sf.py`
(`TOTAL_STEPS` comment only — value unchanged). New diagnostic
`diagnostics/pd_rate_rescale.py`.

**E (2026-08-04):** `scope_colon_env.py` (`STATE_OBS_DIM` 8 → 6, `_observation()`, three docstrings,
`TEN_*_RANGE_M` marked unused) and `sf_encoder.py` (hardcoded `STATE_DIM = 8` → derived from
`obs_space["state"]`).

**C (2026-08-04):** `build_collision_scene.py` (`TIP_CAM_FOVY`), `scope_colon_env.py`
(`DEFAULT_DEPTH_RES` → `(H, W)`, ~10 sites), plus the three scripts that carried their own
hardcoded `DEPTH_RES = 64` and now import the env's value so they cannot silently disagree:
`train_sf.py`, `viewer_sf.py`, `eval_sf.py` (the last also gained an `HxW` argparse type and
builds its own obs space at `:412`).
Perception side: `perception/realtime/da3_runtime.py` (`to_obs_64` → `to_obs`, old name kept as an
alias) and `preview_camera_depth.py` (`--obs_res` takes `HxW`; the preview panel no longer squares
up a 16:9 obs).

Not the copies in `simulation/`, `navigation/v6_dr/` (frozen) or `navigation/v7_shaft/`
(superseded) — `v8_p1/` is self-contained by design. Re-sync the whole folder to the Linux rig
**and** the WSL2 mirror afterwards; there is no auto-sync.

---

## 6. Tests

Short static diagnostics, seconds each — not training runs. The 2026-07-12 diagnostics lived in a
scratchpad and are gone; they need rewriting. On Windows, build the scope XML inside a 64 MB-stack
thread or the MuJoCo compiler blows the stack with a bare exit-127 and no traceback.

**All A, B and D tests were run on 2026-08-02 and passed.** The harness is checked in at
**`navigation/v8_p1/diagnostics/`** (see its README) — headless, portable across Windows / rig /
WSL2, and safe to script. Re-run it after any physics edit.

⚠️ `manual_test.py` is **not** a headless smoke test — it opens a viewer and waits for keypresses,
so it never exits. Use `diagnostics/` for automated checks and run `manual_test.py` by hand.

| # | Test | Pass condition | Result |
|---|---|---|---|
| **A0** | Tension on **pre-change** constants | ~27 N predicted | **26.57 N** @ 100% bend — arithmetic confirmed within 3% |
| **A1** | Same, post-change | ~3 N, ≥95% of nominal | **2.92 N** @ **100%** of nominal, X *and* Z ✅ |
| **A2** | Drift — 40 sim-seconds under sustained full pull | ~0 spread, no fold | **0.000° spread**, no fold, both axes ✅ |
| **A3** | 4-tendon antagonistic, env's real `ctrl` mapping | no fighting; anti-windup sane; re-derive KI | slack tendons **0.000 N**; 0.000° overshoot; no oscillation; **KI unchanged** ✅ |
| **B1** | Fresh-frame spacing and phase, 3 episodes | 1-in-4; phase varies; step 0 always fresh | steady-state gap **exactly 4**; offsets varied {3,1,1}; step 0 fresh ✅ |
| **B2** | Held-frame identity | bit-identical (noise held, not re-rolled) | longest identical run **3 = D−1** ✅ |
| **C1** | Full scene, decimated vs every-step | steps clean; record `ncon` and steps/s | 43.1 steps/s **both ways (1.00×)**; `ncon` mean 54, max 111; no crash ✅ |

| **D1** | `SHAFT_DAMPING` A/B on the full scene, identical seeds *and* identical action sequence | no NaN, no velocity blow-up, no contact-chatter explosion | **NaN 0**; peak `qvel_rms` **0.541 vs 0.656** (lower, not higher); `ncon` 0.93× mean / 1.13× max; throughput 0.96× ✅ |

Propagation was verified in the built scene XML: 25/25 tip joints at `stiffness="0.055"`, zero
left at `0.5`, all 8 tendon actuators at `forcerange="-10.000000 0"`, 44/44 shaft joints at
`damping="0.05"`, `shaft_tip_ball` at `damping="0.125"`, and **zero** stragglers at the old `0.3`.

**Change C tests, run 2026-08-04 — all passed.**

| # | Test | Pass condition | Result |
|---|---|---|---|
| **C2** | `fovy` propagates and implies the DA3 contract | built XML carries 67.7, no stale `85`; implied H = 100° | XML `fovy="67.7"`, zero `fovy="85"` left; implied **H = 100.03°**; aspect 1.778 ✅ |
| **C3** | Obs shape end to end | space, `reset()` and `step()` all (1,54,96); obs inside declared space | all three (1,54,96); `observation_space.contains()` true on reset *and* after 9 steps ✅ |
| **C4** | Non-square obs across a decimation cycle | shape holds on held *and* fresh frames | 9 steps, one shape only; **2 fresh frames** — consistent with `DEPTH_DECIMATION=4` ✅ |
| **C5** | Error map adapts | resized to (54, 96) | (54, 96) ✅ — but see the caveat below |
| **C6** | Encoder absorbs non-square input | flatten resizes, output stays 512-D | flatten **10816 → 13440**, out (2, 512) ✅ |

**Change F tests, run 2026-08-04 — all passed.**

| # | Test | Pass condition | Result |
|---|---|---|---|
| **F1** | PD gain re-fit (`diagnostics/pd_rate_rescale.py`) | stable poles; τ within 10%; no overshoot; steady state reaches target; reversal slew < 2× | τ **214.0 ms → 214.0 ms (−0.0%)**; 95% settle 640 ms unchanged; overshoot **0.000%**; trajectory RMS **0.0005 mm** over 2 s; reversal slew **0.75×** ✅ |
| **F2** | Naive ×4 rescale is unsafe | should be rejected | (0.24, 1.20) → \|z\| = **1.337, UNSTABLE** — confirms the fit was necessary ✅ |
| **F3** | Wall-clock invariants | advance, episode cap, stuck windows, tip penalty, mj_steps/s all match the 100 Hz values | 30.0 mm/s, 100 s, 3.0 s, 1.52 s, 0.1/s, 2000 mj_steps/s ✅ |
| **F4** | Reward balance preserved | `W_PROGRESS·ADVANCE / W_TIP_PENALTY` unchanged | **1.200**, same as at 100 Hz ✅ |
| **F5** | Live run, 2 seeds | no NaN, obs in space, depth fresh every step | 150 steps × 2 seeds clean; `ncon` mean 46-48; **150/150 fresh** (decimation inert) ✅ |

F5's fresh-frame count is the check that `DEPTH_DECIMATION = 1` really did make Change B inert —
B1/B2 in `depth_decimation_diag.py` only carry meaning when it is > 1.

**Change E tests, run 2026-08-04 — all passed.**

| # | Test | Pass condition | Result |
|---|---|---|---|
| **E1** | State width | `STATE_OBS_DIM`, obs space and live obs all 6 | all **(6,)**; `observation_space.contains()` true ✅ |
| **E2** | Slot identity after 6 driven steps | `[0]`=`cmd_x_pair/max_pull_x`, `[1]`=`cmd_y_pair/max_pull_y`, `[2:5]`=`last_action`, `[5]` binary | exact match to 1e-5; `last_action` = commanded `[0.9, −0.6, 1.0]`; `tip_contact` ∈ {0,1}; all finite ✅ |
| **E3** | `ten_length` still drives the actuator loop | tendons live, not inert | `ten_length[:4]` = [0.06185, 0.06300, 0.06314, 0.06199], varying ✅ |
| **E4** | Encoder sizes itself from the obs space | derives 6, no hardcoded 8 | `state_dim=6`, MLP out (2,128), feature 512+128 = **640** unchanged ✅ |

E3 matters because removing the *observation* must not touch the *control* path — `data.ten_length`
is still read every substep to drive each tendon's integral actuator.

⚠️ **C5 caveat:** `error_map_64.npy` was measured on the old *square, 85°* camera, so resizing it
to 54×96 is now an **anisotropic** stretch (x×1.5, y×0.84). The map is a smooth envelope of DA3
error magnitude, so this is a mild distortion rather than a wrong signal — but it is an
approximation, not a verified map. Regenerate via `perception/rd_v2/compute_error_map.py` at the
new geometry when convenient. Flagged in the code at the resize site.

C6 was verified by replicating `ScopeDepthCNN`'s exact conv stack standalone: Sample Factory
cannot be imported on Windows at all (`sample_factory/utils/utils.py` imports `pwd`, POSIX-only),
which is the same reason `eval_sf.py`/`viewer_sf.py` are WSL2-only.

Note on D1: velocities did *not* rise despite 6× less damping, because the shaft remains
overdamped by ~432:1 — consistent with the viscous-regime reasoning, and the reason there is no
ringing. The ~4% throughput cost is noise-level. (The "OLD" arm in the final run is a hybrid —
old `SHAFT_DAMPING`, new ball damping — so it is a stability control, not a clean historical
baseline. The NEW arm is the shipped configuration and it passes.)

Still to run: a **`manual_test.py` pass by hand** — the only check not yet done for C, and the one
that would catch a camera pointing the wrong way or a visibly wrong field, which none of the
automated shape/angle assertions can. C2–C6 live in the same headless harness style as the rest and
should be folded into `diagnostics/` rather than left as one-off scripts.

---

### 6.3 Post-retrain viewer work, 2026-08-04 — READ THIS BEFORE JUDGING A VIEWER SESSION

**The viewer was showing the policy in TRAINING mode, and it looked broken.** Sample Factory's
`actor_critic.forward()` defaults to `forward_tail(sample_actions=True)`, so `policy_out["actions"]`
is a fresh draw `μ + σ·ε` from the policy's Gaussian — PPO's exploration noise, redrawn **every
step**. The measured **σ ≈ 0.627 on a ±1 action space**, so the sampled command was dominated by
noise. Watching it, the tip thrashed violently.

`viewer_sf.py` now takes the **distribution mean by default** (`action_distribution().means`), with
`--stochastic` to opt back into training mode. Deterministic playback is visibly smooth — no
overshoot, no jitter, the tip holds position.

| Measurement | Sampled | Deterministic |
|---|---|---|
| tip-command jitter, RMS mm/step | 0.985 | **0.093** (−91%) |
| mean \|a_xy\| (steering magnitude) | 0.468 | **0.034** |
| same obs ×5, action spread | [0.589, 0.376, 0.711] | **[0, 7e-9, 0]** |

Three things follow, and they matter:

- **The 97% training number was earned WITH σ≈0.63 of command corruption injected**, because SF's
  training statistics come from sampled rollouts. The deterministic policy should be at least as
  good. The noise was not hiding a bad policy — the policy is robust to a remarkable amount of it.
- **`eval_sf.py` still samples** (`run_episode()`, same line). Every eval number in
  `docs/run_archive/`, across v6/v7/v8, was produced in sampling mode. See §8 for the open decision.
- **Deterministic mean `|a_xy|` is only 0.034** — ~3.4% of full bend. Unresolved whether the policy
  is steering efficiently or barely steering at all with shaft compliance and colon geometry doing
  the work. The second would transfer badly. Cheap to settle: log deterministic `|a_xy|` over full
  traverses and test whether it **rises at bends**. Not yet run.

**Why σ is so large — and why an action-rate penalty was considered and REJECTED.** PPO's entropy
bonus (`ENTROPY_COEFF = 0.003`) applies constant upward pressure on σ; the only opposition is
return actually lost to noise. σ settles large exactly when noise is *cheap*, which it is here:
the reward is almost entirely `W_PROGRESS`, and **there is no action-smoothness term at all**
(`prev_action` is stored at `scope_colon_env.py:1120` but never enters the reward).

An action-rate penalty on the *sampled* action would collapse σ analytically —
`E‖a_t − a_{t−1}‖² = ‖Δμ‖² + 2dσ²`, giving `σ* = √(c / 4wd)`. **Rejected**, on two grounds:
(1) the per-step progress reward is only `4.0 × 0.0012 = 0.0048`, so a weight large enough to matter
easily exceeds it and trains a *passive* policy — the classic failure; (2) the measurement above
shows μ is already smooth (`|a_xy| = 0.034`), i.e. the policy is **not** slamming the action around
and relying on the PD filter to clean up, which is the one failure a filter cannot fix. The
existing PD filter plus deterministic deployment achieves the same end with no competing objective
and no retrain. (Note for anyone revisiting: penalising `μ_t − μ_{t−1}` instead of the sampled
action does **not** move σ at all — the `σ²` term vanishes.)

**Viewer changes shipped this session:**

| Change | Detail |
|---|---|
| deterministic actions | now the default; `--stochastic` restores sampling. Mode printed at startup. |
| `max_steps` | was a hardcoded `20_000`; now tracks `DEFAULT_MAX_STEPS`. After Change F that literal had silently become an 800 s cap against the env's own 100 s. |
| `--feed_scale` | scales insertion feed, rescaling `max_steps` / `STUCK_WINDOW` / `TIP_STUCK_WINDOW` to hold their **distance** meaning fixed. See §8. |
| startup print | feed scale, action mode, DR state, depth-noise state — so a recorded run is never ambiguous. |

`eval_sf.py` default episodes-per-seed went **5 → 3** (an env step is 4× more wall-clock since
Change F; 3 × 80 seeds = 240 episodes still ranks seeds fine). Stale `v7_shaft` labels were
corrected across `train_sf.py`, `eval_sf.py`, `viewer_sf.py`, `manual_test.py`, and `sf_encoder.py`'s
CNN docstring was rewritten from the old 64×64 stack (`13×13×64 = 10816`) to the live 54×96 one
(`10×21×64 = 13440`).

---

## 7. Sequencing

1. ~~A0 (baseline, no edits)~~ ✅ done 2026-08-02
2. ~~Change A → A1, A2, A3~~ ✅ done 2026-08-02
3. ~~Change B → B1, B2, C1~~ ✅ done 2026-08-02
4. ~~Change D → D1~~ ✅ done 2026-08-02
5. ~~Measure the real camera FOV~~ ✅ done 2026-08-04 — **140° diagonal**
6. ~~Change C (C2 chosen) → C2–C6~~ ✅ done 2026-08-04
7. ~~Archive the old run, delete the run dir, re-sync~~ ✅ done 2026-08-04
8. ~~Retrain on the rig~~ ✅ done 2026-08-04 — 4.52M steps, ~97% mean progress (training stat)
9. ~~Re-sync the WSL2 mirror with the new code + checkpoint~~ ✅ done 2026-08-04
10. **PHYSICAL RIG TRIAL** ← *next action*. The first trials are **hand-fed** — the shaft is
    pushed in by hand (the automatic feed drive is not built) so the **steering response** can be
    judged on its own. Test the policy as-is and let the observed failure mode pick the next
    retrain, rather than fixing things blind. Pre-flight below.

### Pre-flight before the rig trial

Both are cheap, neither needs a retrain, and the first is not optional.

1. **Re-measure `MAX_PULL_X` / `MAX_PULL_Z` on the real scope.** Not an improvement — a
   correctness requirement. They normalise `cmd_x_n` / `cmd_y_n`, two of the six live policy
   inputs. The current 6.795 / 7.361 mm are derived from sim geometry. Pull each cable to its
   mechanical bend limit and record the encoder delta. **Keep X and Z separate** (12 vs 13 joints).
2. **Fisheye rectification** (§8) — the largest known obs mismatch, and perception-side only.
3. Optional but informative: a **deterministic eval** (§8) and the **steering check** (§6.3), both
   of which change how a failed trial should be read.

⚠️ **Sync state as of 2026-08-04 evening: the WSL2 mirror is CURRENT** (all 13 files byte-identical,
`diagnostics/` present, new checkpoints in place). **The Linux rig has NOT been re-synced** with
this session's viewer/eval edits — it does not need them to train, but they will drift.

The old run's checkpoints were verified byte-identical to
`docs/run_archive/v8_p1/v8_p1_v1_run2_post0713/checkpoint_p0/` by md5 *before* the mirror copy was
cleared, so nothing was lost this time (cf. the 2026-07-13 `--delete` incident).

---

## 8. Still open

- **Lens distortion (was "FOV/aspect calibration" — the aspect half is now fixed).** Change C made
  both sides 16:9 and put the sim camera on the DA3 training contract exactly. What remains is that
  the real lens is a **122° horizontal fisheye** feeding a pipeline built on **100° horizontal
  rectilinear**. Fix: run `probe_camera.py --measure_fov checkerboard` for `K`/`D`, then
  `cv2.fisheye.initUndistortRectifyMap` to a 100° rectilinear virtual camera, applied to the live
  frame before DA3. A precomputed LUT, so cost is one `cv2.remap` per frame — do it on the
  downscaled frame, not 1280×720, since the loop is already at 31 fps against a 25 Hz target.
  **Not a retrain blocker:** it changes only what the real camera hands to DA3, not what sim
  renders, so it can land after training starts. Worth doing before any hardware trial.
- ~~**Do `ten_x_n`/`ten_z_n` actually earn their place?**~~ **CLOSED 2026-08-04 — removed.** Not by
  the planned ablation: the ablation would have measured whether the signal helps *in sim*, and the
  problem is that the sim signal is **inverted** against a real encoder inside the dead zone, so a
  positive result would have been the bad news (the policy leaning on a fantasy) rather than a
  reason to keep it. See §5 Change E. Also note the ablation as specified was already compromised —
  it targeted the current checkpoint, which predates changes A–D and describes the old physics.
- **No eval has been run on the retrained checkpoint.** The ~97% is a training statistic from
  sampled rollouts. **Open decision: should `eval_sf.py` go deterministic?** It currently samples,
  like every archived eval CSV in `docs/run_archive/` across v6/v7/v8. Flipping the default makes
  eval deployment-honest but breaks comparability with all historical numbers; adding an opt-in
  `--deterministic` flag preserves comparison. Recommendation was **the flag** — the opposite call
  to the viewer, because eval numbers get compared across runs and viewer sessions do not.
- **Sim's roller feeder only transports at the trained speed.** It moves the shaft correctly at
  30 mm/s — the rate the policy trained and runs at. With `--feed_scale 0.1` (3 mm/s) it slips and
  stalls after ~9 mm (found 2026-08-04): a friction-contact artefact in the sim feeder, not
  diagnosed. This only matters for *slow-feed* training, which is not planned — the rig is hand-fed
  and the first trials judge steering at the policy's own pace (§7). Not on the critical path.
  If slow-feed training is ever wanted, suspects are roller contact friction, `solref`/`solimp`,
  actuator `kv`, and fixing it could shift behaviour at 30 mm/s.
- **Ranked list of what would meaningfully improve transfer** (2026-08-04), for after the rig trial:
  1. **DR over the command→deflection map.** The policy has never seen a miscalibrated tip — DR
     scales actuator *force*, and the integral actuator still reaches 98–99% of commanded bend every
     episode. "Command 0.5, get 0.5" has held in every episode ever trained. It will not on a real
     scope, especially one whose `MAX_PULL` you measured approximately.
  2. **Stateful backlash / slack on direction reversal.** Sim's dead zone is stateless and
     zero-centred; the dominant Bowden effect is winding in slack *before* the tip moves, and it
     worsens with sheath curvature — i.e. worst in tight bends. Same class as 1; same retrain.
  3. ~~Slow-feed training~~ — not planned; see the feeder note above.
  4. ~~Action-rate penalty~~ — **considered and rejected 2026-08-04**, see §6.3 for the reasoning.

---

## 9. Roadmap after this

**The next phase is defined by what the rig trial shows.** The physical tests decide which sim
gap is worth a retrain; the transfer-improvement list in §8 is the menu the rig picks from.
Nothing below is committed.

**The likely content is gravity + an anatomically-oriented colon** — the sim has neither. Real
work, unstarted, no plan document. The previous one (`V8_PHASE2_PLAN.md`, 2026-07-12) was deleted
2026-08-02: written against the pre-retrain shaft, its shaft mass was wrong by ~1.8× (~0.396 kg
vs the actual ~0.22 kg), and mass is the whole input to a gravity plan.

If it starts, **re-scope from live code.** Current true constants:

| Quantity | Value |
|---|---|
| Shaft | 44 links × 10 mm = **432.5 mm** |
| Shaft mass | 44 × 5 g = **~0.22 kg** |
| Total scope | **495 mm**, in a 500 mm colon (`colon_generator.py:38`) |
| Cantilevered outside the colon at reset | ~400 mm, ~0.20 kg ≈ **~2.0 N** — recompute, do not quote |

The two problems it would have to solve together: the colon currently starts heading world **+Z**
(anti-parallel to default gravity) with a random 3D bend axis per seed, so no single gravity
vector is anatomically consistent; and most of the shaft cantilevers outside the colon at reset
with no floor in the scene, which needs per-body `gravcomp` (unproven here). The anatomical
reference is `Mujuco_V2\Perception\Colon_mesh_xml.py` — read-only, never run from V2.

---

## 10. What was archived, and why

In `docs/archive/superseded_plans/`:

- **`V8_PLAN.md`** — fully superseded. Layer 1 shipped but diverged (8-D obs, not the 10-D this
  describes); Layers 2–3 replaced by the Phase 2 plan; every `navigation/v8/` path in it is
  fictional (the folder is `v8_p1/` and `v8/` never existed). Historical rationale only.
- **`MODELLING_PROBLEMS.md`** (2026-07-28) — its §1 inventory is accurate and verified, but its
  central thesis (§2.1, shaft path contamination) is **wrong given Bowden-tube routing**, and its
  proposed remedy (rate-based action, 8-D → 4-D obs) is rejected per §3. Its surviving content is
  carried into §4.2 above. ⚠️ It declares itself "a self-contained briefing document" that
  "assumes no access to the repository" — i.e. written to be handed to someone. **If it was sent
  to anyone, they need the correction in §4.2.**
- The standalone force/depth-rate plan from earlier today — fully absorbed into §5.

Deleted outright (2026-08-02, by the user): **`V8_PHASE2_PLAN.md`** — see §9 for the constants
worth keeping from it.

`README.md` and `CLAUDE.md` were both updated on 2026-08-02 to match this document.

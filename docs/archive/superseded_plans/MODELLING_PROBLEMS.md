# Modelling Problems — Tendon Actuation, Sim vs Hardware

**Written 2026-07-28.** Self-contained briefing document. It assumes no access to the
repository, so every constant, code path and design decision it relies on is quoted inline.

**Purpose.** We are building a physical colonoscope to deploy an RL policy trained in MuJoCo.
The policy is trained and stable in sim. A sim-to-real gap has been identified in the **tendon
actuation model** that is not a tuning problem but a *structural* modelling problem. This
document states what sim currently models, what the hardware actually does, and where the two
diverge — so the actuation architecture can be redesigned before the next training run.

**Attribution note:** the 25-hinge tendon-driven videoscope tip model described in §1 was
supervisor-provided. The RL environment, shaft, roller feeder, observation design and all
sim-to-real work around it are the author's.

---

## 0. TL;DR

Sim routes its tendons through **the articulating tip only**. Real Bowden cables run the entire
length of a **flexible, continuously-bending shaft**. Consequently:

- In sim, cable length is a **pure kinematic function of tip bend**. It is an exact tip-angle sensor.
- On hardware, proximal reel-in is **tip bend + shaft path length change + sheath compression +
  stretch**, and the shaft term moves independently of the tip.

Everything the policy currently observes about its own articulation (`cmd_x_n`, `cmd_y_n`,
`ten_x_n`, `ten_z_n`) is expressed as a fraction of a **calibrated maximum pull** that exists in
sim and does not exist on the rig. The RL action is likewise an **absolute position setpoint**
scaled by that same constant — not, as is easy to assume, a directional command.

The proposed direction of travel: **isolate the policy from the pull→deflection map entirely**,
by making the action directional/rate-based and closing the low-level loop on cable tension
(force sensors are present on each cable on the rig). This document is the input to that design
discussion, not a decision.

---

## 1. What the simulation currently models

### 1.1 Tip geometry

The articulating tip is a chain of **26 disks / 25 hinge joints**:

| Parameter | Value |
|---|---|
| Disk outer diameter | 9.0 mm |
| Disk spacing | 2.5 mm |
| Tendon hole radius (from centreline) | **3.2 mm** |
| Joints | 25, alternating bend axis |
| Per-joint range | ±10° (0.1745 rad) |
| X-axis joints | 12 → **120°** total |
| Z-axis joints | 13 → **130°** total |

The 12/13 split is an unavoidable consequence of an odd joint count, not a defect. It is the
origin of the per-axis asymmetry in every constant below.

Four tendons at **0°, 90°, 180°, 270°** around the hole circle, named `px`, `pz`, `nx`, `nz`.
They form two antagonistic pairs: `px`/`nx` bend the X axis, `pz`/`nz` bend Z.

### 1.2 Tendon routing — the critical detail

Each tendon is a MuJoCo `<spatial>` tendon threaded through one site per disk:

```python
for name, _angle, rgba in TENDONS:
    w(f'    <spatial name="tendon_{name}" width="...">')
    for disk_idx in range(scope_gen.N_DISKS):        # N_DISKS = 26
        w(f'      <site site="hole_{name}_{disk_idx}"/>')
    w('    </spatial>')
```

`N_DISKS = 26`. **The tendon path begins at tip disk 0 and ends at tip disk 25.** It does not
extend into the shaft, and there is no proximal actuator, sheath, or drive train modelled at all.

Therefore in sim, `data.ten_length` for a given tendon is a **closed-form function of the 25 tip
hinge angles and nothing else**. The shaft can be straight or tied in a knot; the tendon length
does not change.

### 1.3 The flexible shaft (for context)

Behind the tip is a **44-link, 432.5 mm** flexible shaft: pairs of X-hinge/Z-hinge universal
joints (no torsional DOF, matching a real colonoscope), with `shaft_link_0` on a free joint. It is
fed in by an entrance roller/capstan pair that can genuinely slip. Total scope length is 495 mm
inside a 500 mm colon.

**This shaft bends continuously and arbitrarily during an episode — and no tendon passes through
it.** That is the whole problem, stated geometrically.

### 1.4 Calibrated constants

Measured directly via `mj_forward` with all joints on one axis driven to their limit stops
(not analytically derived):

| Constant | Value | Meaning |
|---|---|---|
| `MAX_PULL_X` | **6.795 mm** | single-tendon pull to reach the 120° X limit |
| `MAX_PULL_Z` | **7.361 mm** | single-tendon pull to reach the 130° Z limit |
| `TEN_X_RANGE_M` | **13.380 mm** | `nx − px` length differential at the X limit |
| `TEN_Z_RANGE_M` | **14.495 mm** | `nz − pz` length differential at the Z limit |

Note the differentials are **not** exactly `2 × MAX_PULL` (13.590 / 14.722 would be). The pulled
tendon's shortening and the slack tendon's lengthening differ slightly, because the hole geometry
is not symmetric under bend. Sim already accounts for this small asymmetry by measuring rather
than deriving — worth knowing, because it is a *miniature* version of the hardware problem, and
the fact that sim bothered to measure it shows the effect is real even in the idealised case.

### 1.5 The actuator law

Each tendon has **two** actuators whose forces sum on the same tendon.

**(a) The P+D length servo** — a virtual spring-damper:

```xml
<general name="pull_px" tendon="tendon_px"
         gaintype="fixed" biastype="affine"
         ctrllimited="true"  ctrlrange="-0.006795 0.006795"
         forcelimited="true" forcerange="-80.0 0"
         gainprm="-6000.0 0 0"
         biasprm="<KP*REST_LENGTH> -6000.0 -0.08"/>
```

`ACTUATOR_KP = 6000`, `ACTUATOR_DAMPING = 0.08`. `forcerange="-80 0"` is the one piece of genuine
cable physics present: **force is sign-constrained, so a cable can pull but never push.** Control
input is a target length offset in metres, limited to ±`MAX_PULL`.

**(b) The integral actuator** (added 2026-07-12), a separate actuator on the same tendon:

```xml
<general name="pull_px_i" tendon="tendon_px"
         dyntype="integrator" gaintype="fixed" biastype="none"
         ctrllimited="false"
         forcelimited="true" forcerange="-80.0 0"
         gainprm="-36000.0 0 0"/>
```

`ACTUATOR_KI = 36000` (~6× KP), with anti-windup (freezes within 2 N of the 80 N limit). MuJoCo's
affine bias law has no accumulator, so a true PID cannot be expressed in a single actuator — hence
the second one.

**Why it exists, and why it matters here.** The P+D law alone settles *short* of target, because
zero position error means zero force, and the tip's own joint stiffness (0.5) pushes back. Pure-P
capped achievable bend at **35–45% of nominal** and allowed a slow drift into a folded S-shape.
The integral term drives the tip to **98–99% of commanded bend within 1–2 s**, comfortably inside
a 6–7.5 s episode.

**The consequence for sim-to-real: the integral actuator makes sim's command→deflection map exact
and load-independent.** Push the sim tip against a wall and it still reaches its commanded bend.
That is a deliberate choice — the reasoning at the time was "the rig will run its own encoder PID,
so only the *response characteristics* need to match, not the implementation." That reasoning is
sound **if the rig closes its loop on position**. It is wrong if the rig closes on tension. See §3.

### 1.6 The RL action — an absolute setpoint, not a direction

`action = Box(-1, 1, (3,))` = [X tendon pair, Z tendon pair, roller feed rate]. Per env step
(10 ms; 20 MuJoCo substeps × 0.5 ms, so the control rate is **100 Hz**):

```python
target_x = x * self.max_pull_x                       # absolute target, in metres
vel_x    = self.cmd_x_pair - self.prev_cmd_x
self.cmd_x_pair = clip(
    self.cmd_x_pair + PD_KP * (target_x - self.cmd_x_pair) - PD_KD * vel_x,
    -self.max_pull_x, self.max_pull_x,
)
```

`PD_KP = 0.06`, `PD_KD = 0.30` (heavily over-damped). The pair is driven antagonistically:
`ctrl[px] = +cmd_x_pair`, `ctrl[nx] = −cmd_x_pair`.

**`action[0] = 0.5` means "settle at 50% of maximum X pull".** The gains are slow enough that it
*behaves* rate-like over a handful of steps, which makes it easy to mistake for a directional
command — but its fixed point is an absolute position, scaled by `MAX_PULL_X`. Any claim that the
policy "only outputs a direction" is currently false.

### 1.7 The observation vector

8-D state, alongside a 64×64 depth image from the tip camera:

```
[cmd_x_n, cmd_y_n, ten_x_n, ten_z_n, last_action[0..2], tip_contact]
```

```python
cmd_x_n = self.cmd_x_pair / self.max_pull_x                              # controller setpoint
tl      = self.data.ten_length
ten_x_n = clip((tl[nx] - tl[px]) / TEN_X_RANGE_M, -1.0, 1.0)             # "encoder differential"
ten_z_n = clip((tl[nz] - tl[pz]) / TEN_Z_RANGE_M, -1.0, 1.0)
```

`ten_x_n`/`ten_z_n` were introduced deliberately in V8 Phase 1 to *replace* `force_norm` (summed
contact force) and `net_fx`/`net_fz` (actuator force), on the grounds that those are privileged
sim-only signals with no hardware equivalent, whereas **cable extension maps onto a real linear
encoder on each cable**. The governing design rule was, and remains:

> Only give the policy signals the simulation can model with genuine fidelity — kinematic,
> encoder, and contact-boolean signals. Never feed it force- or friction-derived signals. Sim
> friction is not real friction, so the policy either overfits a fantasy signal or learns to
> ignore it as noise.

**§2 argues that `ten_x_n`/`ten_z_n` fail that same test, for a reason that was missed at the
time.** The rule was applied correctly; the classification of this particular signal was wrong.

### 1.8 What domain randomisation currently covers

Sampled per episode, never observed by the policy:

| Parameter | Range | Applied to |
|---|---|---|
| Tendon gain X / Z | 0.45 – 1.00 | actuator **force** (`gainprm`), both P+D and integral |
| Dead zone X / Z | 0 – 3% of `MAX_PULL` (≈ 0.2 mm) | zero-centred command deadband |
| Command lag | 0 – 2 env steps (0–20 ms) | ring buffer on the effective command |

**Two limitations that matter enormously here, and are easy to miss:**

1. **DR does not randomise the command→deflection map.** The gain scales actuator *force*, and the
   code comment is explicit that this is intentional: *"the endpoint stays reachable at any gain,
   only the force/speed getting there varies."* Combined with the integral actuator, **every
   episode reaches 98–99% of commanded bend regardless of the sampled gain.** The mapping is
   fixed, exact and known in every single training episode. A policy trained here has never once
   encountered a miscalibrated tip.

2. **The dead zone is not backlash.** It is a stateless, symmetric threshold about *zero command*:
   `eff_x = 0 if abs(cmd_x) < dead_x else cmd_x`. It carries no state and models no slack take-up
   on **direction reversal** — which is the dominant Bowden effect. At ≤0.2 mm it is also roughly
   an order of magnitude smaller than realistic Bowden slack.

---

## 2. What the hardware actually does

Six divergences, roughly in decreasing order of severity. The first is structural; the rest are
degrees of "the map is not what sim thinks it is."

### 2.1 Shaft path length contamination — the structural one

Sim's tendon spans the tip only, so cable length is a **pure tip-bend readout**. The real cable
runs from a proximal motor, through ~1.6 m of Bowden sheath, down a shaft that is *itself bending
continuously and in multiple directions*, to the tip.

```
SIM:       [motor: none] ────────────────────────  [tip: 26 disks] → ten_length = f(tip angles)
HARDWARE:  [motor] ══ sheath through flexing 432 mm shaft ══ [tip] → reel-in = f(tip angles, SHAFT PATH, friction, stretch)
```

**Real reel-in = tip articulation + shaft path length change + sheath compression + cable stretch.**

The shaft term is the killer: as the scope advances through a colon, the shaft takes on multiple
bends in different directions. Cable path length changes **with no tip motion whatsoever**. The
policy would read a large `ten_x_n` swing and infer the tip had articulated, when in fact only the
shaft moved.

Note that a **compensating** effect exists in principle: on a real colonoscope the four cables are
routed symmetrically about the shaft's neutral axis, so a pure shaft bend lengthens one side and
shortens the other by *almost* equal amounts, and the antagonistic **differential** cancels much of
it. This is precisely why the differential form was chosen for the obs. But that cancellation
requires the symmetry that the off-centre routing on this rig **does not have** — and it degrades
further with sheath friction, which is direction-dependent and does not cancel.

**The claim this invalidates.** An earlier internal write-up asserted: *"wire is near-inextensible
→ proximal reel-in **is** the tendon length, not an approximation."* Near-inextensibility gives
length *conservation*; it does not give path *invariance*. For a Bowden cable in a bending sheath,
reel-in and tip articulation are simply different quantities. **This is the error at the root of
this document.**

### 2.2 Off-centre routing breaks the differential assumption

The rig's four cables are not at a clean symmetric radius. Consequences:

- The two cables of a pair have **different effective moment arms**, so pull and pay-out are not
  equal and opposite. Commanding `ctrl[px] = +c`, `ctrl[nx] = −c` (sim's exact antagonistic
  mapping) either **slackens** one cable or **over-tensions** it.
- `MAX_PULL` becomes **per-cable**, not per-axis — four different numbers, not two.
- The differential `(nx − px)` is no longer a clean proxy for bend, because the two terms are
  scaled differently by geometry.

Sim models a *0.2 mm* version of this asymmetry (§1.4) and treats it as a fixed measured constant.
The rig's version is larger and configuration-dependent.

### 2.3 Slack and pretension

At **zero commanded bend** the cables must already be tensioned, or they go slack and the tip
flops. Under **large bend**, one side must pay out more than the other takes up (§2.2), so a naive
antagonistic differential drive either creates slack on the outside or fights itself.

Sim never models slack. `forcerange="-80 0"` means a sim tendon carrying no load simply produces
zero force — the qualitative behaviour is right (a cable cannot push) but there is no slack
*state*, no take-up transient, and no pretension.

### 2.4 Sheath friction, hysteresis, backlash

Bowden transmission is **hysteretic**: pull to a position, reverse, and the tip does not retrace
the same path. Friction in the sheath is direction-dependent and load-dependent, and it *increases
with shaft curvature* — so it gets worse exactly when the scope is in a tight bend and steering
matters most.

Sim models **none** of this. The 0–3% dead zone is stateless and zero-centred (§1.8), which is not
backlash in any meaningful sense.

### 2.5 The map is non-linear and configuration-dependent

Sim: one fixed `MAX_PULL` per axis, measured once, exact forever. Hardware: the pull→deflection
relationship varies with shaft configuration, insertion depth, load, temperature and wear. It
cannot be calibrated once and trusted, and it **cannot be observed** — there is no sensor that
reports the shaft's shape, and the tip camera cannot see the shaft.

This is the user's original framing, and it is the correct one: *"I can't know the bend of the
shaft in real life as it can't see it, and it might make multiple bends in different directions."*

### 2.6 Load dependence

A subtler asymmetry, in the opposite direction to the others. Sim's integral actuator **rejects
disturbance**: press the tip against a wall and it still reaches commanded bend. Whether the
hardware does the same depends entirely on the low-level loop:

- **Position/encoder control** → stiff, disturbance-rejecting → sim is approximately right.
- **Tension control** → compliant, load-dependent → the tip yields on contact, and sim is wrong
  in a way that flatters the policy.

Since the tip is in wall contact for much of a real episode, this is not a corner case.

---

## 3. The proposed direction, and the open questions

### 3.1 The proposal

The rig has **force sensors on every cable**. The idea under consideration:

1. Hold all four cables at a controlled tension — eliminating slack, pretension and backlash
   *mechanically*, at the low level, where the sensors are.
2. Have the policy output only a **direction / rate**, never an absolute pull.
3. Let the policy close the loop **visually**: it sees the lumen off-centre in the depth image,
   commands "more +X", and watches it centre.

**Why this is the strongest version of the fix.** A visual servo is *invariant* to the
pull→deflection map. The policy never needs to know that 5 mm = 40°; it only needs the response to
be **monotone in the commanded direction**. Every problem in §2 degrades a *calibration*; none of
them break *monotonicity*. Editing the observation vector (removing `ten_*` or the `max_pull`
normalisation) only shaves the edges off the problem — it leaves an absolute-setpoint action still
scaled by a constant that does not exist on the rig.

### 3.2 What this would require changing in sim

- **Action semantics: absolute setpoint → rate.** `action[0]` becomes a *rate of change* of bend,
  integrated by the controller, rather than a target scaled by `MAX_PULL_X`. This is a real change
  (§1.6), not a relabelling.
- **Observation: remove the channels that encode the calibrated map.** `cmd_x_n`/`cmd_y_n` are
  `cmd / max_pull`; `ten_x_n`/`ten_z_n` are the contaminated encoder differential. Candidate
  reductions: 8-D → 6-D (drop `ten_*`) or 8-D → 4-D (drop both, leaving `last_action[0..2]` +
  `tip_contact`). **Any obs-shape change forces a from-scratch retrain** — acceptable, since the
  next planned phase requires one anyway.
- **Possibly the actuator law.** If the rig is tension-controlled, sim's integral actuator
  (§1.5) enforces an exactness the hardware will not have, and modelling a compliant,
  load-dependent tip may mean revisiting it. Note the tension: the integral term exists to fix a
  real instability (S-shape folding, 35–45% range cap). It cannot simply be deleted.
- **Widen DR to cover the map itself.** Currently the map is exact in every episode (§1.8). At
  minimum: randomise the command→deflection gain on the *target* rather than the force; make the
  dead zone a state-carrying backlash model with slack take-up on reversal; randomise per-cable
  rather than per-axis asymmetry.

### 3.3 Open questions

1. **Which low-level loop will the rig run?** Tension-differential (compliant tip, fully
   routing-agnostic, load-dependent bend) or position-rate with a tension floor (stiff tip, holds
   bend against wall load, closer to current sim)? This determines whether the integral actuator
   stays. It is the single highest-leverage unresolved question.

2. **Can a rate-command policy dead-reckon its articulation state?** With an absolute setpoint,
   `cmd_x_n` tells the policy where the tip is. With a rate command and `cmd_*` removed from the
   obs, the policy must integrate its own action history through a GRU (1 layer, 512 hidden,
   64-step BPTT window) across episodes of 600–2300 steps. That may be fine — vision provides the
   feedback — but it is unproven and is the main behavioural risk of the 4-D obs.

3. **How much do the existing obs channels actually contribute?** Cheaply answerable *without any
   retrain*: zero out `ten_x_n`/`ten_z_n` at eval time on the current stable checkpoint and measure
   the drop in mean progress across the standard 80 seeds × 5 episodes. If it barely moves, the
   channels were not load-bearing and dropping them is low-risk. **This experiment should precede
   the design decision, not follow it.**

4. **Should the sim's cables be routed down the shaft for real?** The high-fidelity fix is to
   thread the four tendon paths through sites on all 44 shaft links, so `ten_length` genuinely
   includes shaft path change. This would make the signal honest rather than removing it — but it
   changes the actuator transmission physics and would force a full re-tune of shaft and roller
   constants, all of which were swept under the current model.

5. **Should cable tension enter the observation, now that sensors exist?** **Recommendation: no.**
   Tension at the motor is dominated by sheath friction, which sim does not model faithfully.
   Owning a sensor does not make the signal sim-modellable — this is exactly the trap that retired
   `force_norm` and `net_fx`/`net_fz` in Phase 1. Use the force sensors for the low-level tension
   loop, which is a hardware control problem and the right job for them, and keep them out of the
   policy's input.

---

## 4. Summary table

| Aspect | Sim today | Hardware reality | Severity |
|---|---|---|---|
| Cable routing | Tip disks only (26 sites) | Full 1.6 m through a bending shaft | **Structural** |
| `ten_length` meaning | Exact tip-bend kinematics | Tip bend + shaft path + friction + stretch | **Structural** |
| Pull→deflection map | Fixed, exact, known, load-independent | Variable, non-linear, unobservable | **High** |
| Antagonist pair | Balanced, `ctrl[nx] = −ctrl[px]` | Unbalanced — off-centre routing | **High** |
| Slack / pretension | Not modelled (force-limited to pull-only) | Must be actively managed | **High** |
| Backlash / hysteresis | Not modelled (stateless 0.2 mm deadband) | Significant, curvature-dependent | **High** |
| RL action | Absolute setpoint × `MAX_PULL` | — (the constant does not exist) | **High** |
| Disturbance rejection | Total (integral actuator, 98–99%) | Depends on low-level loop choice | Medium |
| DR coverage of the map | None — exact in every episode | — | Medium |

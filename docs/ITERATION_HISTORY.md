# Iteration History — Autonomous Colonoscope Navigation (v1 → v7_shaft)

Consolidated design history of the Mujuco_V2 project (May–July 2026), written as the
single source for GitHub/portfolio write-ups. Every version's architecture, reward
design, modelling assumptions, results, and reason for supersession is captured here so
the old code and checkpoints can be archived.

**Attribution:** the 25-hinge tendon-driven videoscope tip simulation (`simulation/`) and
the photorealistic Blender colon-rendering environment were **supervisor-provided**.
The author's own work: the Stage-1 procedural MuJoCo colon (`Colon_mesh_xml.py`), the
perception CNNs, the closed-loop v1 controller, the DA3 fine-tune, and all collision and
RL work (collision meshing, environment, reward engineering, cheat diagnosis, domain
randomisation, the flexible shaft and V-groove roller design). One nuance worth stating
precisely in any write-up: `colon_generator.py`, used by every RL version from v2 onward,
is a pure-NumPy **port of the supervisor's Blender colon generator** — the procedural
distribution is the supervisor's; the port, the collision meshing, and all later
modifications are the author's.

**Naming convention warning:** there are two overlapping numbering tracks. *Codebase
generations* are folders (v1, v2_collision, v3_sample_factory, v4_hierarchical,
v5_smooth, v6_dr, v7_shaft). *Training runs* are numbered within a generation
(ppo_v1…ppo_v8 inside v2_collision; v9_sf…v17_sf inside v3_sample_factory;
v7_shaft_v1…v4 inside v7_shaft). Sections below are organised by codebase generation,
in chronological order of work (note v2_collision's run series ppo_v7/ppo_v8 predates
the v3_sample_factory folder).

---

## 0. Timeline

| Generation | Dates (2026) | Framework | Key idea | Result | Why superseded |
|---|---|---|---|---|---|
| Stages 1–3 | ≤05-25 | MuJoCo + PyTorch | Procedural colon, steering CNNs, scope tip | Working components | Foundations, not superseded |
| v1 | 05-25→26 | Proportional control | CNN target pixel → tendon commands, no collisions | Full traversal, no crash | No physical contact — controller never coped with constraint |
| DA3 depth | 05-27→28 | Depth Anything 3 | Full metric depth from RGB (head-only fine-tune) | 2.4–3.3 mm MAE, metric, 20 FPS | Not superseded — became the RL noise model |
| v2_collision | 05-28→06-24 | SB3 PPO | Physical scope–wall collisions + RL steering | 66%/85% runs both **cheated** | Sync-barrier throughput + honest reward needed APPO rework |
| v3_sample_factory | 06-24→25 | SF APPO | Async training, exploit whack-a-mole (v9–v17) | v17: mean 52%, max 100% | Jittery raw-joint control; exploit fixes matured |
| v4_hierarchical | 06-26 | SF APPO | RL picks image target, PID steers | **Abandoned** — provably ≡ v3 | Centroid dilemma (see §7) |
| v5_smooth | 06-26→29 | SF APPO | PD-smoothed tendon control, clean rebuild | 44% completion, 89.7% mean progress | Ideal-condition sim; no sim-to-real story |
| v6_dr | 06-28→29 | SF APPO | Domain randomisation + real CNN-error noise | **93.75% completion, 96.7% mean progress** (80 seeds) | Kinematic base advance was the last big sim shortcut |
| v7_shaft | 07-01→02 (+07-08→10) | SF APPO | Physical flexible shaft + roller feeder | **~92% mean progress, ~90% completion** (v2); v4 35.0%/72.3% pre-fix | Superseded by v8_p1 — v4 trained+evaluated 07-10 on a generator later found self-intersecting (fixed same day) |
| v8_p1 | 07-11→13 | SF APPO | V8 Layer 1: cable-extension obs (no force signal); tendon integral control; shaft/tip compliance rework | **63.2% mean progress, 25.7% completion** (v8_p1_v1, post-fix half) | **LIVE line.** `stuck` failure eliminated; `tip_stuck` then 73.3% of episodes — diagnosed 07-13 as a truncation-window artefact, fix untrained |

---

## 1. Foundations (Stages 1–3)

**Stage 1 — procedural colon (author's).** A seeded script generates a 500 mm
anatomical centreline (ascending → hepatic flexure → transverse → splenic flexure →
descending), ~15 mm internal radius, 35 haustral folds (single soft-arc folds with
random angular centres, 60° edge fade), a slight 3-lobed taenia-coli cross-section,
per-vertex normals for smooth shading, and endoscope-style tip lighting as the only
illumination. Outputs mesh + ground-truth centreline + MuJoCo scene. Key rendering
lessons: shadow acne from the tip light required `castshadow="false"`; depth-buffer
znear/zfar needed tightening; the passive viewer and offscreen renderer *share lighting
state* (toggling the viewer headlight corrupts what the CNN sees).

**Stage 2 — steering CNNs (author's).** ResNet-18 regressors predicting a "where to
drive next" pixel from RGB. Label strategies compared on identical frames:
- *deepest-depth-blob centroid* — shaky at corners (deepest region occluded behind bends);
- *deepest-visible-centreline point* — over-rotates at flexures (label sits on the wall
  corner where the centreline disappears);
- *pure-pursuit look-ahead* at 30/15/10 mm — 30 mm collapses to the wall corner at every
  bend (fallback fires), 15 mm traversed end-to-end but clipped walls at both flexures,
  **10 mm gave the gentlest corner commands and became the keeper**.

**Stage 3 — videoscope tip (supervisor-provided).** One articulating section: 26 disks
chained by 25 alternating-axis hinges, 4 antagonistic tendons, 9 mm OD. The author's
contribution is integration, not the scope model itself.

## 2. v1 — CNN closed-loop controller (no collisions)

First coupling of perception and actuation. Per step: render tip camera → CNN target
pixel → deadzone → rate-limited proportional update of the two antagonistic tendon
pairs → advance the base kinematically along the ground-truth centreline. With the
10 mm look-ahead CNN the scope traverses the full colon with no visible tip-cam crash.

**Modelling assumption that defined the next generation:** the colon mesh had
`contype=0` — *no scope–wall collisions*. The controller never had to cope with
physical constraint, and the base advance was a perfect kinematic rail.

**Follow-up (dropped):** swapping the steering CNN for the fine-tuned depth model with
a depth-centroid aim point + PD control, still collision-free, could *see* the lumen
reliably but could not traverse — it oscillated between good views and wall jams, and
every gain change traded one failure mode for another. Conclusion (2026-05-28): "target
pixel only, no collisions, basic controller logic is not cutting it." This motivated
both collisions and RL.

## 3. Depth perception — DA3 fine-tune (feeds every later version)

Depth-Anything-3-SMALL was evaluated as a full-depth-field alternative to the single
target pixel. Zero-shot it output only *relative inverse* depth (median GT/pred ratio
0.029, ~7 mm scale-aligned MAE) and failed structurally at flexures — placing the
"deep" region on a brightly lit wall instead of the off-centre lumen hole, the exact
frame type steering needs most. A **head-only fine-tune** (11% of parameters, 1469
Blender renders from the supervisor's photorealistic environment, mixed
scale-invariant-log + L1-metric loss) fixed all of it:

| Metric | Zero-shot | Fine-tuned |
|---|---|---|
| Held-out geometry MAE | 9.49 mm | **3.28 mm** |
| Unseen pilot set MAE | 7.10 mm | **2.40 mm** |
| Output type | relative (ratio 0.029) | **metric (ratio 1.014)** |
| Flexure failure | present | **fixed** |
| Speed | ~20 FPS | ~20 FPS |

A second fine-tune round (RD_V2: 320×180, 100° FOV, 1500 frames) reached **1.60 mm val
MAE** (8.30 mm baseline). Its per-pixel error statistics became v6's observation-noise
model (§9). Pipeline lives in `perception/rd_v2/`.

## 4. v2_collision — SB3 PPO with physical collisions (runs ppo_v1…ppo_v6)

**Collision modelling assumption:** MuJoCo's default mesh collision convex-hulls the
mesh — for a hollow tube that's a solid blob the scope ignores. Fix: **per-triangle
thin-prism decomposition** — every wall triangle becomes a 6-vertex convex mesh extruded
along its normal; adjacent prisms share edges so the wall behaves like the smooth mesh
(~5k prisms, ~1500 physics steps/s).

**RL formulation:** action `Box(-1,1,(3,))` = two tendon-pair commands + base-advance
rate; obs = 64×64 metric depth + small state vector; base attached to a kinematically
driven target via a **1-DOF slide joint** (not a free joint) so a jammed tip lags
instead of being forced through; reward = high-water-marked progress + goal − collision
penalty + waypoint credit.

**Run history — the two cheats are the key finding:**

| Run | Config change | Outcome |
|---|---|---|
| ppo_v1–v2 | collision penalty 2–1×10⁻⁴ | ~0% — any motion net-negative, agent froze |
| **ppo_v3** | penalty 2×10⁻⁵, ent 0.01 | **66% — CHEAT 1:** free-floating 6-DOF base let it drag itself backward and flip orientation |
| ppo_v4 | 1-DOF slide + dynamic slide floor + 0.5 mm walls (cheat closed) | flat 0.5% — reward became too sparse |
| ppo_v5 | + waypoints, ent 0.03 | flat 1% — entropy term ~100× the task signal |
| **ppo_v6** | ent 0.015 | **85% — CHEAT 2:** used the chain's ~770° of total bend to curl into a ball and punch *through* the 0.5 mm prism walls at bends |

Both high scores were illusory — the agent solved the *simulation*, not the *task*.
Supporting lesson: `ent_coef` must be calibrated to the task-reward magnitude
(~0.001/step here), not copied from defaults.

## 5. Lumen-gate rewards (runs ppo_v7/v7b/v7c) and THE REFRAME

Instead of stiffening the scope to stop cheat 2, three variants of a "forward-look
gated reward" were tried: progress credit gated by `aim = open_factor × fwd_factor`
(central depth openness × alignment with the centreline tangent). All three failed to
learn (v7: gate threshold too deep, reward trickle; v7b: entropy-dominated; v7c:
entropy fixed but exploration/credit-assignment still flat at ~0%).

**The reframe (user-driven, 2026-05-29) — the project's core design principle since:**
1. **Aim-shaping defeats the purpose of RL.** Rewarding "point at the lumen" bakes the
   perception→aim mapping into the reward — re-implementing the analytic aim point
   already available without RL. Steering must *emerge* from the depth observation.
2. **Cheat prevention belongs in physics/termination, not reward shape.** The slide
   architecture already stalls a mis-aimed tip; the only hole was wall *penetration*.
3. Reward stripped to pure outcome: high-water-marked progress (+ later, force terms).

## 6. Force-based termination session (run ppo_v8, SB3)

Closed cheat 2 with physics-side measurement instead of reward shaping. Three candidate
"insertion force" signals were calibrated under a scripted max-curl replay:
`qfrc_constraint` on the slide DOF (7× curl/gentle ratio — misses the cheat because the
curled chain moves *ahead* of the target so the slide floor never engages),
`cfrc_ext` on the anchor (zero — contacts land on disk bodies), and **sum of
`mj_contactForce` magnitudes (1242× ratio — chosen)**. `MAX_INSERTION_FORCE = 30 N`;
`force_norm` entered the observation vector and `force_norm ≥ 1` terminated the
episode. Max-curl now terminates in 8 steps; gentle advance reads ~0.006. SB3 run
ppo_v8 reached only mean 2.9–3.1% but **the curl cheat was dead** — and SB3's sync
barrier (30–60% core utilisation) had become the bottleneck, motivating the APPO port.

## 7. v3_sample_factory — APPO port and the exploit ladder (runs v9_sf…v17_sf)

Sample Factory APPO replaced SB3: env workers, policy worker, and learner run
asynchronously with no sync barrier (~1500–2000 fps vs ~450). The env stayed in
v2_collision; only the training harness changed.

**Exploit/fix ladder (each run's failure was diagnostic):**

| Run | Reward/termination | Result | Root cause |
|---|---|---|---|
| v9 | new-territory + 30 N hard termination | collapsed, force:100% | **sprint-and-die**: terminal at force limit lets a sprinting policy end episodes cheaply |
| v11 | W_FORCE=0.01 per-step | stagnant 2.6% | penalty 33× progress signal, advantage ≈ 0 |
| v12 | W_FORCE=0 | plateau 20% | sprint to first bend, terminate |
| v13 | terminal force penalty | KL=171970, NaN | 250× scale mismatch blew up log_std |
| v14 | per-step lumen reward | 3.4% | **lumen farming** — sit still and collect |
| v15 | lumen multiplier + hard force gate `×(1−f)` | 20%→9% | gate makes "do nothing" the optimum |
| v16 | gate removed | OOM crash ~15–18% | EGL mmap leak; force-limit exploit back at 99% |
| **v17** | see below | **mean 52.3%, max 100%** | — |

**v17's winning combination:** tip_contact obs bit (defeats the "UFO posture" — tip
pressed sideways into a haustra ring where the camera sees only wall);
high-water-mark stuck metric (`d_max_s`, not raw per-step delta — farming-proof);
stuck/tip_stuck as *truncation not termination* (bootstrapped value ≠ 0, so reaching an
obstacle isn't punished); ROLLOUT=64 so the BPTT window covers a full ring approach;
ENTROPY=0.003; obs slimmed 37→10-D (dropped 25 hinge angles — colon geometry the depth
image already covers); stiff wall contacts (`solref="0.002 1"`, 3 mm walls) to stop
tunnelling; renderer recycled every 50 episodes (EGL mmap leak); W_TIP_PENALTY=0.001 +
TIP_STUCK_WINDOW=50 added **only after** mean >30% (from step 1 they are a bootstrap
trap — random policies never see positive reward).

**Post-training eval bug (cost a day):** SF stores the GRU hidden state under
`"new_rnn_states"`, not `"rnn_states"` — eval/viewer code that `.get("rnn_states",…)`
silently zeroed the GRU every step, scoring 2–3% instead of 52%.

**Verdict:** navigation learned, but control was jittery — RL directly commanding raw
joint bend targets with exploration noise produced oscillating micro-corrections.

## 8. v4_hierarchical — the dead end (abandoned, no training kept)

Idea: RL picks a *point in the image* to aim at; a low-level PID drives the tendons to
centre it — hoping to smooth v17's jitter. Abandoned on analysis:

**The centroid dilemma.** For the hierarchy to differ from direct control, someone must
close the loop between the chosen image point and the tendon commands:
- put centroid feedback *in the controller* → the controller steers, RL learns nothing;
- put the centroid *in the observation* → not IRL-valid (needs an image-processing
  pipeline unavailable on the real device — a standing project constraint);
- neither → `target_x → cmd_x` is just a scaled linear map, **functionally identical to
  v3's direct tendon control**.

Also independently broken: v17's late-training anti-exploit penalties were applied from
step 1 (bootstrap trap), a PID integral wind-up bug, and a deadzone tested on the wrong
signal. Lessons kept; folder never trained to completion.

## 9. v5_smooth — clean rebuild with PD-smoothed control

Self-contained rebuild (the "version bundle" pattern every later generation follows):
each folder carries its own copies of all dependencies. Control smoothing moved into
the *env*, IRL-validly: a **PD position controller (KP=0.06, KD=0.30)** turns the
policy's tendon setpoints into rate-limited commands — the RL output is smoothed the
way a real motor controller would, not by post-hoc filtering. Advance slowed to
0.3 mm/step. Depth 48×48, frame-relative normalisation (`depth/depth.max()` — keeps
contrast when pressed against a wall, where a fixed metric clip goes near-black).
10-D state: cmd_x/y, progress placeholder (always 0 — true progress is privileged sim
information), force_norm, net_fx/fz (motor-current analogues), last_action, tip_contact.
Reward: `W_PROGRESS × new_territory × (0.3 + 0.7×(1−force_norm)) + W_GOAL − W_TIP_PENALTY`.
Colon generator rebuilt as piecewise circular arcs (1–3 bends, R=40–90 mm) with a
curvature-safety margin.

**Result (5M steps, 40 workers × 2 envs): 44% full completion, 89.7% mean progress,
max 100%.** Residual failures: tip-contact truncations at sharp bends where ±90°
deflection is insufficient.

## 10. v6_dr — domain randomisation MILESTONE

v5 assumed ideal conditions: perfect depth, exact actuator response, under-spec
deflection. v6 (folder `navigation/v6_dr/`, frozen) closed the sim-to-real gaps
*without hardware calibration* — an explicit design decision after rejecting an IK
model (tendon gain is not static: it changes with shaft bend friction, so free-air
calibration is stale the moment the scope curves).

1. **Deflection ±90°→±120°/axis** (real endoscope range).
2. **Per-episode domain randomisation, never observed by the policy:** tendon gains
   45–100% per axis, dead zones 0–3%, command lag 0–2 steps, advance gain 65–100%. The
   policy must infer the episode's regime from visual feedback — the same feedback a
   human operator has.
3. **Depth 48→64 px.**
4. **Real perception noise:** the RD_V2 DA3 model's per-pixel relative-error map
   (`error_map_64.npy`, from 500 held-out frames) injects multiplicative noise
   `depth += N(0, error_map) × depth` — the policy trains under the deployed depth
   network's actual uncertainty profile.
5. **Fixed 700 mm colon** (uniform reward scale across seeds), harder bends (2–4 bends,
   up to 150°).
6. **Goal bonus removed** (no IRL endpoint; it incentivised rushing) + small terminal
   stuck penalty (−0.05).

**Result (run v6_dr_700V3, best checkpoint at ~3M steps, eval 80 seeds × 5 episodes):
93.75% completion rate, 96.7% mean progress, 65/80 seeds at 100%.** Eval CSV:
`navigation/v6_dr/runs_sf/v6_dr_700V3/eval_best_…_eps5_seeds80_….csv`.

## 11. v7_shaft — physical flexible shaft + roller feeder (superseded by v8_p1)

v6's last major shortcut: the base advanced on an infinitely strong kinematic rail. In
reality a ~1.6 m flexible shaft transmits push force from outside the body, wall
contact at bends carries the force around corners, and slip/buckling/looping are the
clinically dominant failure modes. v7_shaft replaces the rail with real mechanics:

**Architecture.** 30× two-DOF universal joints (X-hinge + Z-hinge pairs, no torsional
DOF — matching a real colonoscope), even pairs fused into single 20 mm capsules to
halve contact count (+52% fps). `shaft_link_0` is on a **free joint** — the shaft is
held only by roller grip and wall contact. An **entrance roller/capstan feeder** (2
rings × 4 position-controlled rollers; two rings because one constrains lateral
position but not rotation) feeds the shaft in. Action a[2] drives roller target angle;
**the roller can genuinely slip**, so progress is measured from the shaft's true
physical position, never the commanded angle. Episode termination redesigned:
`shaft_exhausted` fires when the physical shaft end reaches the rollers — a natural
physical boundary replacing the arbitrary distance goal (a slip-heavy episode can
exhaust the shaft with low progress and earns accordingly).

**The roller-slip saga (the hard engineering of this generation):**
- Post-release creep: fixed by preload 0.4→0.6 mm (friction coefficient increases made
  it monotonically *worse* — high friction on a curved-on-curved contact produces
  stick-slip chatter, not grip).
- Slip under real driving load (v7_shaft_v1 plateaued at the first bend, 24.7% mean):
  root cause was geometric — a plain cylinder crossing the shaft at 90° is a *point*
  contact no matter the preload. **Fix: V-groove rollers** — two mirrored frustum mesh
  geoms per roller flaring *outward* from the waist to cradle the shaft. Must be two
  separate meshes: MuJoCo convex-hulls each mesh geom, and a single waist-and-shoulders
  mesh silently hulls back into a cylinder. Groove flare helps monotonically
  (contact-area lever); preload does *not* scale grip once contact is established
  (solref already near-rigid).
- Shaft stiffness walked 2.0→0.25 alongside (Euler-buckling check: short-span kinks
  stay force-limited at 0.25).

**The MuJoCo friction/priority lesson (transferable):** MuJoCo combines contact
friction by element-wise *max* — verified by reading `d.contact[i].friction` directly
(inferring it from slide distance gives wrong answers; the soft solver creeps below the
nominal static threshold). The shaft's low lubricated friction never took effect
against the wall's higher default. Using `priority` to force it through **broke
training**: priority hands over the *entire* contact parameter set, silently replacing
the wall's tunnelling-prevention `solref` with the soft default. Clean fix: lower the
scene's shared default friction instead; never use `priority` without also setting
explicit `solref/solimp` on the winning geom.

**Reward-scale pathology and fix:** v7_shaft_v1's reward *improved* while progress
*declined* — the policy learned to trigger `stuck` truncation early because the
unbounded per-step tip penalty could outgrow the distance-capped progress budget.
Diagnosed by comparing consecutive log windows (reward-up-while-progress-down is the
tell). Fix: `W_PROGRESS` 2.0→4.0, fresh network (**v7_shaft_v2**).

**Result (v7_shaft_v2, 8M steps, ~4.8 h on the Linux rig): ~92% mean progress, ~90%
completion (`shaft_exhausted`).** Best checkpoint at
`navigation/v7_shaft/runs_sf/v7_shaft_v2/checkpoint_p0/best_000001700_6963200_reward_0.581.pth`.

**Post-v2 hardening (July 8–10):** colon generator hardened (bend radii 32–48 mm near
the shaft's ~29 mm mechanical minimum; finer spline sampling; entrance-straight
invariant enforced so the pre-threaded shaft spawns inside the lumen); a
`<size memory="32M">` arena fix for high-contact scenes (generic MuJoCo bug, keep
everywhere); WSL2 investigated and **rejected** for training (GPU paravirtualization is
structurally bad for per-step small-render workloads; CUDA-IPC fixes
`ptrace_scope=0` + `expandable_segments` are known-good if ever needed); training
returned to the native Linux rig. Feasibility lesson: the generator rejection-samples
segment lengths against the total budget, so straight-length ranges must scale with task
length or generation goes infeasible (500 mm + 6–7 bends is a hard geometric wall —
0/100 seeds).

**v7_shaft_v3 (the WSL2 attempt, 07-08→10).** Resumed from v2's best checkpoint and run
under WSL2 to 8.33M steps. It never beat v2's `reward_0.581` — the run is the evidence
behind the WSL2 rejection above, not a result in its own right. Checkpoints discarded;
`config.json` + `sf_log.txt` archived at `docs/run_archive/v7_shaft/v7_shaft_v3/`.

**v7_shaft_v4 (trained 2026-07-10).** Same physics as v2, retargeted at a 500 mm / 4–5
bend colon (density-matched to the 700 mm / 5–7 task, 100/100 seeds validated), 50-link
shaft, `EXPERIMENT="v7_shaft_v4"`. Ran the full 10M-step budget on the native Linux rig
(`andrew-HP-Z640-Workstation`), finishing at 10,010,624 steps. Best policy
`best_000002330_9543680_reward_0.231.pth` — training reward was **still climbing at the
end** (0.209 → 0.231 over the final ~40 min), so the budget, not convergence, stopped
it. v4 is *not* comparable to v2's 0.581 (different colon, so different reward scale).

**v7_shaft_v4 eval (2026-07-10, WSL2/osmesa CPU, 24 workers, 80 seeds × 3 episodes =
240 episodes): 35.0% completion, 72.3% mean progress.** Distribution was sharply
bimodal, not a uniform shortfall: 12/80 seeds went 3-for-3 (100% every episode), 36/80
went 0-for-3, only 32/80 mixed. Failing episodes were almost universally `tip_stuck`
(camera pressed sideways into a haustral ring, blind — the "UFO posture" from §13) —
not force limits, not `max_steps`. This eval used CPU rendering (osmesa) rather than
the GPU EGL path used for v2's numbers; a same-seed, zero-action parity check found the
two rasterizers agree to ~0.1–0.3% on identical geometry (well inside the ~1.6 mm DA3
sensor noise the policy already trains under), so the renderer is not the source of the
result. **This eval also predates the self-intersection fix below — see that entry
before treating 35%/72.3% as a clean read of the policy.**

**Colon-generator self-intersection bug (found and fixed 2026-07-10).** Two of the 80
eval seeds (43, 34) produced near-zero-progress episodes for a reason that had nothing
to do with the policy: seed 43 ejected the shaft in 18 steps with mean contact count
~10× a healthy step; seed 34 showed the same signature, milder. Root cause: each colon
chains 4–5 bends of 80–150° about independently random 3D axes — 320–750° of total
turning, more than a full loop — and `generate_colon`'s only geometric guard
(`centreline_min_curvature_radius`) checks *local* curvature within a single bend. There
was no check for two *distant* sections of the same centreline swinging back through
each other in 3D. A scan of all 80 eval seeds found the walls actually interpenetrate
(closest-approach ratio to `2×tube_radius` below 1.0) on **6/80** — seeds 43 and 34 both
fold back within the first ~15% of arc length, plugging the lumen right past the
entrance and making the episode unsurvivable regardless of policy skill; the other 4
were milder and didn't always prevent completion. The remaining 74/80 already cleared a
comfortable margin (~1.2–1.5×), so this was a tail bug, not a systemic one — removing
just the 2 worst seeds only lifts the 35.0% figure to ~35.9%, so it is **not** the
primary explanation for v4's shortfall against v2 (still-climbing reward + a harder,
denser-bend task are the leading candidates for that).

Fix in `colon_generator.py`: `centreline_min_self_distance()` measures closest approach
between centreline points more than 6 cm apart in arc length (far enough to exclude
same-bend/local-thickness comparisons), folded into `generate_colon`'s existing
rejection-retry scoring alongside the curvature and length checks
(`SELF_INTERSECT_MARGIN = 1.02` — the literal "walls must not touch" boundary, plus
tiny fp slack; population already clears far higher margins naturally, so this rejects
only genuinely self-intersecting layouts). Also added a cheap directional bias in
`random_ctrl_points`: at each bend, of the two available arc directions, prefer the one
curving away from the path already travelled — doesn't guarantee avoidance (that's what
the rejection check is for) but reduces how often it needs to fire. Re-scanning all 80
seeds post-fix: **0/80 self-intersecting** (down from 6/80). v4's checkpoint and the
35.0%/72.3% eval above both predate this fix and reflect the old geometry distribution.

**The v4-retrain thread was never run** — it was overtaken by v8_p1 (§12), which forked from
v7_shaft carrying the corrected generator. v7_shaft is therefore superseded, and v4 vs v2 was
never settled. Its one durable finding is the bimodal seed split (12/80 seeds 3-for-3, 36/80
0-for-3, failures almost all `tip_stuck`), which §12 retro-explains: sharp bends were
*unsolvable* under the 50-step tip-contact truncation, at any skill level. That was an
environment bug, not a v4 policy deficiency.

## 12. v8_p1 — V8 Phase 1: hardware-matching observation (LIVE line)

Folder `navigation/v8_p1/`. Implements **Layer 1 only** of the original V8 plan — the
observation-vector swap. Layers 2 (anatomical colon) and 3 (gravity) are *not* in it; they
became "Phase 2", still unstarted. Note the folder is `v8_p1/`, not the `v8/` path the old
plan used throughout (that folder never existed). The original plan is archived at
[archive/superseded_plans/V8_PLAN.md](archive/superseded_plans/V8_PLAN.md) — historical only,
never take paths or numbers from it.

**The observation change (the point of the generation).** `force_norm` — the summed
contact-force magnitude — was dropped from the observation. It has no real-hardware sensor
equivalent: nothing on the physical device measures total scope-wall contact force. It was
replaced by **net per-axis cable extension**, computed from MuJoCo's free `data.ten_length`
output as `(ten_length[nx] - ten_length[px])` and `(ten_length[nz] - ten_length[pz])`,
normalised by `2 × max_pull`. That maps directly onto a linear encoder on each antagonistic
cable pair — a sensor the real robot actually has. `net_fx`/`net_fz` (motor-current
analogues) were dropped in the same pass, so the state vector went **9-D → 8-D**:
`[cmd_x, cmd_y, ten_x, ten_z, last_action×3, tip_contact]`. `force_norm` itself is untouched
in the reward path and the info dict — only its presence in the *observation* changed.

**The governing principle (worth stating in any write-up):** only give the policy signals the
simulation can model with genuine fidelity — kinematic, encoder, and contact-boolean signals.
Never feed it force- or friction-derived signals. Sim friction is not real friction, so the
policy either overfits a fantasy signal or learns to ignore it as noise. Cable *extension* is
a kinematic quantity; cable *tension* is not, and was deliberately not used.

**Reward rebalance (07-11).** Two changes, both undoing earlier over-correction:
- `W_FORCE_MULT` 0.7 → **0**. The force discount on progress reward assumed shaft-wall contact
  was pathological. It isn't — the shaft *physically cannot clear a colon bend* without
  pressing against the inside wall, and light grazing alone reads `force_norm ≈ 0.2`, taxing
  ~14% off every metre of legitimate progress. This was likely reproducing a milder version of
  the v11/v12 "policy refuses to move" collapse.
- `W_PROGRESS` 8.0 → **4.0**. The 8.0 doubling had been calibrated *while* the force discount
  was still taxing progress; zeroing the discount restores much of what the doubling existed to
  compensate for, and stacking both risked overshooting the other way.

**Colon-generator diversity fix (07-11).** The anti-loop directional bias added alongside the
07-10 self-intersection fix (§11) was removed — it was suppressing genuine layout diversity.
Re-verified: diversity up, self-intersection still 0/80. The rejection-sampling
`centreline_min_self_distance()` check remains and is what actually guarantees non-intersection;
the bias was only ever an optimisation to make it fire less often.

**The steady-state droop problem and the tendon integral fix (07-12).** The tip's P+D tendon
actuators (`ACTUATOR_KP = 6000`, `ACTUATOR_DAMPING = 0.08`) leave a steady-state droop against
the tip joints' own restoring stiffness — the commanded bend is never fully reached, and under
sustained command the tip settles into a persistent S-shape drift. The naive fix (cap or remove
the joint stiffness) was rejected as it re-opens the bend-range problem. **Real fix: a true
integral term.** MuJoCo's affine bias law has no accumulator, so a PID cannot be expressed in one
actuator — the integral was added as a **separate `dyntype="integrator"` actuator per tendon**,
whose force sums with the original P+D actuator at the same tendon (`ACTUATOR_KI = 36000`, ~6×
`ACTUATOR_KP`, chosen empirically to converge in ~1–2 s, well inside a ~6–7.5 s episode, without
overshoot). Anti-windup included. The integral actuator's `gainprm` gets the **same
domain-randomisation gain rescaling** as the P+D actuator — otherwise the policy could simply
lean on the un-randomised integral term and DR gain would stop meaning anything.

**The shaft-vs-tip compliance inversion (07-12).** Live-viewer inspection of the 37%-plateau
checkpoint showed the shaft not buckling *at all* at tight bends while the much more compliant
tip visibly over-deformed — backwards from a real endoscope, where the shaft is the part that
yields. Fixes, all applied together: `SHAFT_STIFFNESS` 0.25 → **0.05**; the shaft/tip junction
stub shrunk from a full 10 mm shaft unit to **2.5 mm** (matching the tip's own disk spacing — it
had been acting as a stiff moment arm dumping bend load straight onto the tip); tip joint
stiffness reinstated at **0.5**; and shaft joint density raised **50 → 76 links** (25 → 38
universal-joint pairs) to let the shaft form a smoother bend curve instead of concentrating load
at sparse joints.

**Training run v8_p1_v1 (2026-07-12 → 07-13, native Linux rig, 40 workers).** One experiment
directory, but **two distinct physics regimes** — the learner was stopped at 10.01M steps, the
fixes above were applied, and training **resumed from the 10M checkpoint** (weights carried over,
step counter continued) for a further 5M. The two halves are therefore not one run and must not
be read as a single curve:

| | steps 0 → 10.01M (pre-fix) | steps 10.01M → 15.02M (post-fix) |
|---|---|---|
| mean progress | **37.4%** (flat from ~3M) | **63.2%** |
| completion (`shaft_exhausted`) | ~0% | **25.7%** |
| `stuck` truncation | **82.2%** | **0.0%** |
| `tip_stuck` truncation | 17.8% | **73.3%** |
| best reward | 0.440 | **0.920** (at 13.9M) |

**Reading of that table — the fix worked, and it unmasked the next constraint.** The integral
control plus the compliance changes **completely eliminated the `stuck` failure** (82% → 0%): the
shaft now advances reliably. What was left standing was the tip jamming into the wall. In the
final 1M steps, `stuck`, `ejected` and `max_steps` truncations are all **exactly 0%** — every
single non-completing episode ends in `tip_stuck`. One failure mode, cleanly isolated. (Caveat
for anyone quoting the 60%: the post-fix policy is a *warm start* from 10M steps of a different
physics regime, and it only saw 5M steps of the current one. Reward peaked at 13.9M and drifted
slightly down to 15M — plateau-ish, but not conclusively converged.)

**`tip_stuck` diagnosed (07-13) — the bend was not hard, it was impossible.** `TIP_STUCK_WINDOW`
truncated the episode after 50 *consecutive* steps of tip-disk-to-wall contact, with **no check
on whether the scope was still advancing**. At `ADVANCE_RATE_MAX = 0.3 mm/step` that caps
in-contact travel at **15 mm** — far shorter than the in-contact arc of a 32–48 mm-radius bend. So
at a sharp bend the policy had *no surviving strategy*: push through and it truncates mid-bend,
hold still and it truncates anyway. The manoeuvre was simply absent from the MDP, at any skill
level. This also retro-explains the sharply bimodal seed distribution in the v7_shaft_v4 eval
(§11: 12/80 seeds 3-for-3, 36/80 seeds 0-for-3) — gentle bends clear inside 50 contact-steps,
sharp ones cannot, and no amount of training changes that. **`tip_stuck` was firing on a proxy
(tip touching wall) that is not equivalent to the thing it was meant to catch (tip not
progressing).** The genuine UFO posture is caught independently by `stuck` (300 steps / 2 mm of
new territory), and progress is measured at `disk_0` — the shaft/tip junction — so a jammed tip
*cannot* farm reward by buckling shaft in behind it. `tip_stuck` was never the load-bearing
anti-UFO guard it was assumed to be.

**A second, softer blocker found in the same pass (not yet fixed).** With `W_FORCE_MULT = 0` and
`W_SHAPING = 0`, per-step progress reward at full advance on new territory is
`W_PROGRESS × ADVANCE_RATE_MAX = 4.0 × 0.0003 = +0.0012`. `W_TIP_PENALTY` is `−0.001/step`
whenever the tip touches the wall. So advancing at *full speed* while in wall contact nets only
**+0.0002/step**, and any bend-clearing slower than **0.25 mm/step (83% of max)** is
**net-negative reward**. Since clearing a bend requires wall contact and is necessarily slower
than a straight sprint, the reward gradient actively discourages the one manoeuvre the policy
needs. This is the same argument that retired `W_FORCE_MULT`, applied to the tip. It was left in
place deliberately (user's call: the easier colons remain learnable under it, so it still teaches
the tip not to park on the wall) — but if the window increase alone doesn't lift progress, **the
fix is to gate the penalty on non-advancement** (`r_tip = −W_TIP_PENALTY if _tip_in_contact and
ds_actual <= 0`), *not* to re-raise `W_PROGRESS`, which has already failed twice.

**Changes applied 2026-07-13 (all three untrained as of writing; `v8_p1_v2` is the retrain).**

**(1) `TIP_STUCK_WINDOW` 50 → 150.** Buys ~45 mm of in-contact travel — enough for a typical bend
arc — while staying 2× tighter than `STUCK_WINDOW`. `W_TIP_PENALTY` deliberately unchanged (see
above: the easier colons remain learnable under it, so it still teaches the tip not to park on the
wall).

**(2) Shaft joint density reverted 76 → 50 links.** Two reasons pointing the same way: the 76-link
shaft **buckled hard at the entrance** (too flexible in practice), and the 1.5× joint count cost
~40% of rig throughput. Benchmarked on Windows, seed 0: **1.58× faster physics** (592 → 938
mj_step/s), mean contacts 60 → 52, `<body>` nesting depth 103 → 77.
This was a **literal** revert — `SHAFT_STIFFNESS`/`SHAFT_DAMPING` were *not* density-compensated
back, which is the point: effective beam stiffness and damping both scale as
`per-joint value × segment length`, so restoring the 20 mm fused-pair segment raises both by
**1.52×**, which is exactly the anti-buckling stiffening wanted. Two silent side effects of the
original density bump are also undone: shaft mass 396 g → 259 g (`SHAFT_LINK_MASS` had never been
density-compensated) and effective damping returns to its long-validated value.
**Consequence: joint density's individual contribution to the 37% → 60% jump was never isolated**
— it went in as one of five simultaneous changes. The revert is also the experiment that measures
it.

**(3) Shaft SHORTENED 50 → 44 links — the scope was longer than the colon.** Spotted in the
viewer: the tip was leaving the colon entirely. The arithmetic confirms it. A 50-link shaft builds
to **492.5 mm** (10 mm `link_0` + 24 fused 20 mm pairs + a 2.5 mm end stub); plus the 62.5 mm
flexible tip that is **555 mm of scope in a 500 mm colon**. `shaft_exhausted` fires when the
shaft's *base* reaches the roller ring (20 mm outside the entrance), so the episode ran until the
**tip was at ~535 mm — about 35 mm out the far end of the colon**, in free space with no wall
contact, for the last ~117 steps. And because `new_territory` reward is **not capped at `s_goal`**,
the policy was being *paid* to drive the tip out of the colon into open air, where no tip penalty
could reach it. 44 links gives a 432.5 mm shaft, **495 mm of scope in a 500 mm colon**, and the tip
ends at ~475 mm — **25 mm inside**, still in contact. (Measured off the compiled model, seed 0, not
derived.)

- `SHAFT_LINK_SPACING_M` was also changed from `0.5 / SHAFT_N_LINKS` to a literal **`0.010`**. The
  old expression pinned the shaft to a nominal 500 mm and *re-derived segment length from the link
  count* — so shortening the shaft would have silently thinned the segments to 11.4 mm and
  **softened the shaft**, undoing change (2). Segment length is a physics constant; shaft length is
  `SHAFT_N_LINKS`' job. They are now independent.
- **This is NOT a throughput change — do not repeat it expecting fps.** Measured: 50 → 44 gains
  only ~6% (1,027 → 1,088 mj_step/s), and **contact count is flat at 52.0**. The removed links are
  the free-floating tail *behind* the rollers, which touches nothing; physics cost here is dominated
  by the colon-wall contact solve, not shaft body count. (Change (2) was different — it thinned
  links *inside* the colon, cutting contacts 60 → 52, and that is where its 1.58× came from. Same
  intuition, opposite outcome, because of *where* the links were.)
- **Accepted cost: progress headroom falls 72.5 mm → 12.5 mm.** Headroom is how far `disk_0`
  overshoots `s_goal` (= colon length − `END_CLEARANCE` 100 mm) at termination, and it is the buffer
  that absorbs shaft **buckling** — a buckled shaft consumes more length than the centreline arc, so
  `disk_0`'s arc-`s` falls short of `ring + shaft_length`. At 12.5 mm, heavy buckling could cap
  completing episodes at ~95–99% progress rather than 100%. If that bites, the fix is
  `END_CLEARANCE` 0.10 → 0.13, **not** a further shortening: 42 links puts headroom at **−7.5 mm**,
  making 100% progress outright unreachable. The tension is structural (the scope is nearly as long
  as the colon), and any fix trades tip margin against headroom.
- ⚠️ **This REDEFINES the completion metric.** `shaft_exhausted` now fires ~60 mm of shaft feed
  (~200 steps at max advance) earlier, so completion % rises for reasons that have nothing to do
  with policy skill. **v8_p1_v1's 25.7% completion is not a valid baseline for anything measured
  after 2026-07-13.** Mean progress is also mildly affected (raw progress at termination drops from
  ~120% to ~103% of `s_goal`; both still clip to 100%, so the *clipping artefact* shrinks rather
  than the metric moving).

**Run disposition — note the name reuse.** The original `v8_p1_v1` checkpoints were **discarded** on
2026-07-13 (three physics/termination changes make its value head worthless as a warm start) and its
`runs_sf/` directory deleted on the rig, so Sample Factory starts a **fresh network**. `EXPERIMENT`
was left at `"v8_p1_v1"`, so **the retrain reuses that name** — there are now two distinct runs
called `v8_p1_v1`. The discarded first one is archived at
**`docs/run_archive/v8_p1/v8_p1_v1_run1_pre0713/`** (`config.json`, `sf_log.txt`, TensorBoard
summaries); every number quoted in this section is reproducible from it. Anything in `runs_sf/`
after 2026-07-13 is the *second* run, on the corrected geometry.

⚠️ **Deleting the run directory is what makes it "from scratch."** Sample Factory resumes from an
existing checkpoint in the experiment dir *by default* — reusing the name with the directory intact
silently continues the old policy (that is exactly how v8_p1_v1 ended up with two physics regimes in
one curve, §12 above).

**What to watch on the retrain.** The tell for success is **`avg_tip_stuck` collapsing without
`avg_stuck` ballooning to replace it**. If `stuck` simply takes over the failure share, the tip is
genuinely jamming rather than being cut off mid-bend, and the whole diagnosis above is wrong. Watch
`avg_shaft_exhausted` too, but remember it is now measured against a different boundary.

## 13. Hard-lessons catalogue

**RL cheats and exploits observed (in the wild, this project):**

| Exploit | Where | Countermeasure that worked |
|---|---|---|
| Drag-base-backward + orientation flip | ppo_v3 | 1-DOF slide joint (structurally forbids it) |
| Curl-into-a-ball, punch through walls | ppo_v6 | Contact-force measurement + termination (physics, not reward) |
| Sprint-and-die at force limit | v9_sf | Don't make force a cheap terminal; per-step observable signal |
| Lumen-reward farming (sit and collect) | v14_sf | No standalone per-step scene reward; progress must be high-water-marked |
| Oscillation farming of shaping reward | v17 iter | Stuck metric on high-water mark (`d_max_s`), not raw deltas |
| UFO posture (tip sideways into ring, camera blind) | v17 | tip_contact obs bit + penalty + truncation |
| Quit-early truncation to cap per-step penalties | v7_shaft_v1 | Rebalance W_PROGRESS; watch for reward↑/progress↓ across consecutive logs |
| *(not an exploit — the inverse)* Anti-exploit truncation so tight the correct manoeuvre is impossible | v8_p1_v1 | `tip_stuck` at 50 contact-steps capped in-contact travel at 15 mm, shorter than a bend's contact arc → sharp bends unsolvable at any skill. Window → 150 (§12) |

**Engineering gotchas (do not repeat):**

| Mistake | Lesson |
|---|---|
| Convex-hull mesh collision on hollow tubes | Per-triangle thin-prism decomposition (also: single mesh geoms silently convex-hull — the V-groove needed 2 geoms) |
| `priority` to win a friction contest | Overrides the whole contact set incl. solref — lower the shared default instead |
| Inferring friction combination from slide distance | Read `d.contact[i].friction` directly; the soft solver creeps |
| Hard force gate `×(1−force_norm)` | Zero gradient at high force → "do nothing" optimum |
| Terminal penalties at a different scale to per-step reward | 250× mismatch → KL explosion/NaN |
| `terminated=True` for stuck states | V=0 bootstrap punishes *reaching* the obstacle; use truncation |
| Anti-exploit penalties from step 1 | Bootstrap trap — add only after the policy earns real reward (mean >15–30%) |
| ROLLOUT shorter than the manoeuvre | BPTT window must cover a full ring/bend approach (32→64) |
| `.get("rnn_states")` on SF eval output | SF stores `"new_rnn_states"` — GRU silently zeroed, 52%→2% |
| ent_coef copied from defaults | Calibrate against actual per-step task reward magnitude |
| Centroid in obs or controller | Either breaks IRL validity or removes the learning problem (v4's dilemma) |
| IK/calibration for sim-to-real | Tendon gain varies with shaft bend under load — use domain randomisation |
| Sweeping preload for grip capacity | With stiff solref, preload only guarantees contact exists; contact *area* (flare) is the lever |
| Aim-shaping in the reward | Re-implements analytic control; steering must emerge from observation (the reframe, §5) |
| Local-only curvature check on a multi-bend random centreline | Chained random-axis bends have no memory of earlier segments — 320–750° of total turning routinely folds the path back through itself; check global self-distance, not just per-bend curvature (§11, v7_shaft_v4 seeds 43/34) |
| Truncating on a *proxy* for the failure instead of the failure | "Tip touching wall" ≠ "tip stuck". Contact is mandatory when clearing a bend. Truncate on the thing you actually care about (no new territory), which `stuck` already did (§12) |
| Per-step contact penalty vs. per-step progress reward, uncompared | Always divide them: `W_TIP_PENALTY` (0.001) vs `W_PROGRESS × ADVANCE_RATE_MAX` (0.0012) made bend-clearing net-negative below 83% of max advance. A penalty is only safe if you know its ratio to the reward it competes with (§12) |
| Changing joint density without rescaling stiffness/damping/mass | Effective beam stiffness and damping both scale as `per-joint value × segment length`. The 50→76-link bump compensated stiffness but silently left the shaft 1.5× more damped and 1.5× heavier (§12) |
| Deriving segment length from link count (`spacing = total / N`) | Couples shaft *length* to shaft *stiffness*: shortening the shaft silently thins the segments and softens it. State segment length as a literal; let N set length alone (§12) |
| Assuming fewer bodies = faster physics | Only if those bodies *make contacts*. Removing 6 shaft links behind the rollers (free-floating, zero contacts) gained 6%; removing 26 links from inside the colon cut contacts 60→52 and gained 58%. Check `ncon`, not body count (§12) |
| A scope longer than the colon it drives through | 555 mm of scope in a 500 mm colon put the tip 35 mm *outside* the far end at `shaft_exhausted` — and since `new_territory` reward isn't capped at `s_goal`, the policy was paid to drive into open air where no tip penalty could reach it. Check the terminal geometry, not just the initial (§12) |
| A physics fix bundled with four other changes | v8_p1's 37%→60% jump came from 5 simultaneous changes; joint density's own contribution is unmeasured, and reverting it is now the only way to find out (§12) |
| Resuming a run in the same experiment dir after changing the physics | The step counter continues and the TB curves concatenate, so the run *looks* like one continuous training curve. It isn't. v8_p1_v1's 0–10M and 10–15M halves are different environments (§12) |

## 14. Results summary

| | v5_smooth | v6_dr (700V3) | v7_shaft_v2 | v8_p1_v1 |
|---|---|---|---|---|
| Base advance | kinematic slide | kinematic slide | **physical shaft + rollers** | physical shaft + rollers |
| Depth obs | 48×48, clean | 64×64 + DA3 error-map noise | 64×64 + DA3 error-map noise | 64×64 + DA3 error-map noise |
| State obs | 10-D (incl. force) | 10-D (incl. force) | 9-D (incl. force) | **8-D, no force signal — cable extension** |
| Domain randomisation | none | gains/dead-zone/lag/advance | gains/dead-zone/lag (advance DR removed — real shaft friction replaces it) | same + DR applied to the integral actuator |
| Colon | 1–3 bends, variable length | 700 mm fixed, 2–4 bends ≤150° | 700 mm, then task variants | 500 mm, 4–5 bends |
| Completion | 44% | **93.75%** (80 seeds) | ~90% (`shaft_exhausted`) | 25.7% |
| Mean progress | 89.7% | **96.7%** | ~92% | 63.2% |

**Do not read that last column as a regression against v6/v7.** The task and the observation
are both strictly harder: no force signal at all (v6/v7 got `force_norm`, which no real device
can measure), and a shorter, denser-bend colon. v8_p1_v1's numbers are also from an unconverged
5M-step warm start whose dominant failure mode has since been diagnosed as an environment bug
(§12), not a policy limitation. The honest cross-generation comparison does not exist yet.

## 15. What comes next

The live line is **v8_p1**. v7_shaft is superseded — the v7_shaft_v4 retrain thread listed here
previously was overtaken by v8_p1, which carries the corrected generator forward anyway.

- **Retrain from scratch on the 07-13 changes — LAUNCHED 2026-07-13, RESULT NOT YET SEEN.** All
  three changes (`TIP_STUCK_WINDOW = 150`, 44-link shaft, decoupled segment length) are in the code
  and synced. The old checkpoints were discarded and the run directory deleted, so this is a genuine
  **fresh network** — but it **reuses the name `v8_p1_v1`** (see §12's run-disposition note; the
  discarded original is archived as `v8_p1_v1_run1_pre0713`). Success looks like
  `avg_tip_stuck` collapsing *without* `avg_stuck` rising to replace it; that would confirm the
  bend-clearing diagnosis and should take mean progress well past 63%. The 1.58× throughput
  recovery means a 15M-step budget now costs roughly what 10M did before, so a longer run is
  affordable — and v8_p1_v1's reward had not clearly converged at its step limit.
  **Do not compare its completion % to v8_p1_v1's 25.7%** — the `shaft_exhausted` boundary moved.
- **If progress still stalls: the tip-penalty coupling, not `W_PROGRESS`.** Gate `W_TIP_PENALTY`
  on non-advancement (§12). The code carries an explicit warning against re-doubling `W_PROGRESS`
  a third time.
- **Evaluate properly.** No v8_p1 checkpoint has ever been through `eval_sf.py`. All numbers in
  §12 are *training* statistics. Match v6_dr's protocol (80 seeds × 5 episodes) for a figure that
  is comparable to the table-14 baselines. `eval_sf.py`/`viewer_sf.py` cannot run on Windows —
  use the Linux rig or the WSL2 mirror at `~/mujoco_v3/`.
- **V8 Phase 2 — gravity + anatomical colon.** Layers 2 and 3 of the original V8 plan, still
  unstarted and with **no plan document** (the 2026-07-12 one was deleted on 2026-08-02 — its
  shaft mass and length figures predated the 2026-07-13 retrain and were wrong by ~1.8×).
  Re-scope from live code; current constants are in
  [CURRENT_PLAN.md](CURRENT_PLAN.md) §9. Expect a genuine physics re-tuning pass, not a flag
  flip: every shaft/roller constant in the project was swept under zero gravity.

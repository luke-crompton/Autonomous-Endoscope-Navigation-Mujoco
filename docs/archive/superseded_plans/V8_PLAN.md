# v8: gravity + anatomically-realistic colon + hardware-matching obs vector

> **⚠️ PARTIALLY SUPERSEDED (2026-07-12). Read this first.**
> - **Layer 1 (obs vector) is DONE and shipped.** The folder is **`navigation/v8_p1/`** — the
>   `navigation/v8/` path used throughout this document **does not exist**. Layer 1's actual
>   implementation also diverged from the plan below (8-D obs, not 10-D: `net_fx`/`net_fz` were
>   removed too, not just `force_norm`).
> - **Layers 2 and 3 are superseded by [`V8_PHASE2_PLAN.md`](V8_PHASE2_PLAN.md)** — the canonical,
>   implementable Phase 2 plan, with line numbers verified against the live `v8_p1/` code.
> - Everything below is retained for its **research findings and design rationale**, which are still
>   accurate and useful. Do not take file paths, line numbers, or step-by-step instructions from it.

**Status: PLANNED, not started.** This is the canonical v8 plan. Migrated into Mujuco_V3
on 2026-07-10 from `.claude/plans/flickering-herding-quasar.md` — **the plan itself is
unchanged; only file paths and sync/verification pointers were updated** for the new
project layout. All code line numbers below were re-verified against
`navigation/v7_shaft/` after the migration and are correct.

## Context

User has secured a contract to build the **physical** version of this cable-driven
colonoscope robot. For physical testing they'll 3D-print a maze (simpler than a real
colon shape — out of scope for this plan, mechanical/CAD work). But for a mini report
on the simulation side of colon testing specifically, they want the sim to be more
representative of reality in three ways, all folded into a single **v8** variant (their
explicit choice over splitting into a separate v9):

1. **Hardware-matching observation vector** — replace `force_norm` (no real-hardware
   sensor equivalent) with a cable-extension signal (maps to a real linear encoder).
2. **Gravity enabled** — currently `gravity="0 0 0"` (fully disabled) everywhere.
3. **Anatomically-realistic colon shape and orientation** — currently
   `colon_generator.py` picks a fully random 3D bend axis per segment, with no relation
   to real colon anatomy or patient positioning. User wants a "rough square" shape,
   reflecting that colonoscopy patients lie on one side (left lateral decubitus).

## Research findings

- **No file declares an up-axis explicitly.** MuJoCo's unmodified default is +Z-up
  (default gravity `0 0 -9.81`, currently overridden to `0 0 0`). No camera/visual code
  anywhere anchors an up direction.
- **Load-bearing problem**: `colon_generator.py:160` starts the colon's entrance
  direction along world **+Z** — the same axis MuJoCo treats as vertical. This is
  currently harmless (gravity is off), but naively enabling gravity as-is would have
  the scope entering straight upward, which is anatomically meaningless. The colon's
  starting direction and the bend-generation logic need to be reoriented together with
  enabling gravity, not as two independent changes.
- **All existing shaft/roller physics tuning assumes zero gravity.** `SHAFT_STIFFNESS`
  (2.0→1.0→0.5→0.25), `SHAFT_TIP_BALL_STIFFNESS`, `ROLLER_PRELOAD_M`, `GROOVE_FLARE_M`,
  `ROLLER_FRICTION`, roller actuator gains — all empirically swept under the current
  zero-g regime (see `ITERATION_HISTORY.md` §11). `shaft_link_0` is a free-joint body
  explicitly held in place "only by roller grip + colon wall contact," with no
  counteracting weight ever factored in. Enabling gravity introduces a real force these
  values were never calibrated against — expect a genuine re-tuning pass, not a flag
  flip.
- **`random_ctrl_points()` is fully unconstrained in 3D**: per bend, a random axis
  orthogonal to the current direction is drawn, arc angle/radius sampled independently,
  no anatomical labeling or gravity-relative bias anywhere.
- **A real anatomical generator already exists elsewhere**: `Colon_mesh_xml.py`
  (Stage 1, separate codepath) builds a **fixed**, deterministic centreline in one plane:
  ascending straight → hepatic flexure (90°, 40mm radius) → transverse straight →
  splenic flexure (90°) → descending straight. No gravity/positioning concept, but a
  useful structural reference for real segment naming/flexure geometry.
  **Pointer update:** this file was *not* migrated into Mujuco_V3. It still exists at
  `C:\Users\lukec\PycharmProjects\Mujuco_V2\Perception\Colon_mesh_xml.py` (V2 is a
  frozen code archive — read it there; copy it in only if it turns out to be worth
  reusing).
- **Zero existing patient-positioning code** anywhere in the repo (grepped for
  "decubitus" — no hits).
- Verified against actual code for the obs-vector swap specifically (line numbers
  re-confirmed in `navigation/v7_shaft/` post-migration):
  - `scope_colon_env.py:845-846` builds
    `state = np.array([cmd_x_n, cmd_y_n, self.force_norm, net_fx, net_fz, last_action[0..2], tip_contact])`
    — `force_norm` is `state[2]`, the field being replaced.
  - `force_norm`'s computation (contact-force sum, `MAX_INSERTION_FORCE=30.0`, line 162)
    and its use in reward shaping (`_force_mult`) and `info["force_norm"]` (line 747) are
    **untouched** — only its presence in the observation vector changes.
  - `data.ten_length` (MuJoCo's per-tendon length array) is **never referenced anywhere**
    in the current codebase — free physics-engine output, currently unused.
  - Tendon/actuator ordering confirmed identical and fixed:
    `generate_videoscope_one_section.py:54-59` `TENDONS = [("px",0),("pz",90),("nx",180),("nz",270)]`,
    iterated in this exact order for both `<tendon>` and `<actuator>` generation in
    `build_collision_scene.py` — so `data.ten_length[0:4]` indices line up exactly with
    the existing `data.actuator_force[0:4]` indices already used for `net_fx`/`net_fz`.
  - `sf_encoder.py:111` `STATE_DIM = 9` is the **only** other hardcoded shape
    dependency anywhere (feeds `nn.Linear(self.STATE_DIM, 64)`, line 124). `eval_sf.py`
    and `viewer_sf.py` both derive obs shape from the env/constant directly — no edit
    needed there for shape.

## Confirmed anatomical model (user approved)

Left lateral decubitus (patient lying on left side) maps the colon's path as:

**sigmoid (tortuous entry) → descending colon (roughly horizontal, low — left side is
down against the table) → splenic flexure (sharp turn) → transverse colon (roughly
VERTICAL riser — the body's left-right axis is now vertical) → hepatic flexure →
ascending colon (roughly horizontal, high — right side is up)**

This is the "rough square" shape the user described: two horizontal runners at
different heights (descending low, ascending high) connected by a vertical riser
(transverse), with a tortuous sigmoid lead-in matching the real entry anatomy.

## Approach

Build in three independently-verified layers, even though all three end up combined
in the final `v8` — this avoids compounding failures across three genuinely different
kinds of change (obs-vector shape, physics/gravity, geometry generation).

### Layer 1 — cable-extension obs vector (do this first, lowest risk, fully scoped)

1. **Copy `navigation/v7_shaft/` → `navigation/v8/`**, excluding `runs_sf`, `scenes`,
   and `__pycache__`. Excludes old checkpoints (incompatible once obs shape changes —
   would hard-crash on `nn.Linear` weight mismatch) and regenerable/cached scene XML.
   On Windows:
   ```powershell
   robocopy navigation\v7_shaft navigation\v8 /E /XD runs_sf scenes __pycache__ roller_feed_prototype
   ```
   (Decide whether `roller_feed_prototype/` comes along — Layer 3 reuses that harness
   for gravity re-tuning, so copying it is reasonable; it is excluded above only because
   it is dev tooling, not part of the trained env.)

2. **Edit `navigation/v8/scope_colon_env.py`**:
   - `STATE_OBS_DIM = 9` → `10` (line 117), with an added history comment documenting
     the swap and why (`force_norm` not hardware-measurable; tendon length is).
   - Update the two state-vector docstrings (module docstring ~line 15, `_observation()`
     docstring ~line 798-800) to reflect the new 10-field list.
   - In `_observation()` (~line 828, alongside `cmd_x_n`/`cmd_y_n`), add the net
     tendon-extension computation:
     ```python
     # Net per-axis cable extension: hardware-realistic analogue of a linear
     # encoder differential on each antagonistic cable pair. tendon order
     # (scope_gen.TENDONS): px(0), pz(1), nx(2), nz(3). Sign convention matches
     # cmd_x_n/cmd_y_n (positive = net pull toward +X/+Z): pulling +X shortens
     # px and lengthens nx, so (ten_length[nx] - ten_length[px]) grows positive.
     # Directly measurable on real hardware via motor/encoder position on each cable.
     tl = self.data.ten_length
     ten_x_n = float(np.clip((tl[2] - tl[0]) / (2.0 * self.max_pull), -1.0, 1.0))
     ten_z_n = float(np.clip((tl[3] - tl[1]) / (2.0 * self.max_pull), -1.0, 1.0))
     ```
   - Replace `self.force_norm` in the `state = np.array([...])` call (line 845) with
     `ten_x_n, ten_z_n`:
     ```python
     state = np.array([
         cmd_x_n, cmd_y_n,
         ten_x_n, ten_z_n,
         net_fx, net_fz,
         self.last_action[0], self.last_action[1], self.last_action[2],
         float(self._tip_contact_steps > 0),
     ], dtype=np.float32)
     ```
   - No other edits needed — `np.zeros(STATE_OBS_DIM, ...)` placeholders and
     `observation_space` construction already reference the constant.

3. **Edit `navigation/v8/sf_encoder.py`**: `STATE_DIM = 9` → `10` (line 111), update its
   inline comment, the `nn.Linear` comment (line 124), and the architecture docstring
   (lines ~15-22, which also has a pre-existing stale "576-D" concat number — appears at
   lines 19, 22 and 152 — worth correcting to the actual `640` while touching it).
   `CNN_OUT`/`MLP_OUT`/`get_out_size()` — no change (concat size independent of
   `STATE_DIM`).

4. **No shape-related edits needed** in `train_sf.py`, `eval_sf.py`, `viewer_sf.py`
   (all derive obs shape from the env/constant). Recommend a housekeeping rename of the
   hardcoded `EXPERIMENT` run-name in these three files to something like
   `"v8_cable_v1"` so training runs don't look like mislabeled v7 runs. (Note the v7
   copies currently read `"v7_shaft_v4"` / `"v7_shaft_v2"`.)

5. **`force_norm`'s reward usage and `info` diagnostics stay fully intact** — nothing
   in the reward function or `eval_sf.py`'s CSV/summary output needs to change.

**Layer 1 verification** (no training needed yet):
- Standalone env sanity check: instantiate `ScopeColonEnv`, `reset()`, confirm
  `obs["state"].shape == (10,)`. Step with a hard `+X` action for ~20 steps, confirm
  `state[2]` (`ten_x_n`) trends toward `+1`. Confirm `info["force_norm"]` still
  populates normally. **Runs on Windows** with `C:\Python314\python.exe` and
  `$env:MUJOCO_GL='wgl'` (no sample_factory import in this path).
- Encoder shape check: forward pass on a batch of reset observations, confirm output
  shape `(batch, 640)`, no `nn.Linear` size-mismatch.
- Wiring smoke test: run `navigation/v8/viewer_sf.py`, confirm obs→encoder→policy path
  loads and steps without a shape-mismatch crash (no trained checkpoint exists yet —
  this only validates plumbing). **Pointer update:** `viewer_sf.py`/`eval_sf.py` cannot
  currently run on Windows — Python 3.14 is missing `signal-slot-mp`, so the
  sample_factory import chain fails. Either `pip install signal-slot-mp` into
  `C:\Python314`, or do this step on the Linux rig. The first two checks above are
  unaffected.

### Layer 2 — anatomical colon shape

Rewrite `colon_generator.py`'s bend generation to produce the confirmed
sigmoid→descending→splenic-flexure→transverse→hepatic-flexure→ascending topology
instead of fully-random 3D bends. Keep some per-seed randomization (sigmoid
tortuosity, exact flexure angles within an anatomically-plausible range, segment
lengths) for training diversity, matching v7_shaft's existing style — the *topology
and gravity-relative orientation* becomes fixed/anatomical, not the exact geometry.
Reorient the colon's starting `direction` (currently world +Z) to match where a real
sigmoid/rectum entry would sit under this new orientation scheme, worked out together
with Layer 3's gravity vector, not independently.

**Watch out** (from the v7_shaft_v4 sizing work, `ITERATION_HISTORY.md` §11):
`random_ctrl_points()` rejection-samples segment lengths against the total
`TASK_LENGTH_M` budget and raises an uncaught `RuntimeError` if no combination fits
within 100 tries — this is *not* protected by `generate_colon()`'s outer retry loop.
Any change to segment-length distributions must be validated with a pass-rate sweep
across ~100 seeds, not a single seed.

**Layer 2 verification**: standalone script generating N seeds with the new anatomical
generator, visually inspect several in `view_colon_seed.py` (no gravity yet) to
confirm the sigmoid/flexure/runner topology looks anatomically reasonable before
touching physics. `view_colon_seed.py` runs on Windows.

### Layer 3 — gravity

Change `gravity="0 0 0"` → `gravity="0 0 -9.81"`, consistent with Layer 2's
reorientation. **Three locations** in `navigation/v8/` (the third was missed in the
original plan and matters because Layer 3's re-tuning harness lives there):
- `build_collision_scene.py:423`
- `generate_videoscope_one_section.py:150`
- `roller_feed_prototype/build_roller_scene.py:90` (+ its three checked-in
  `roller_scene*.xml` outputs, which are regenerated by that script)

Then re-validate shaft/roller physics empirically — same style of isolated-harness sweep
used for the original zero-g tuning (see `ITERATION_HISTORY.md` §11's V-groove roller
section for the methodology to reuse; the harness itself is
`roller_feed_prototype/`): check for new slip/sag/creep failure modes under the
shaft's own weight, re-tune stiffness/preload/friction as needed. **Flag to user up
front: this is likely the single most time-consuming part of the whole plan, not a
quick step.**

**Sub-problem identified 2026-07-10, with a concrete fix: gravity on the un-inserted
shaft tail.** At reset, only `SCOPE_TIP_AT_S` (95mm) of shaft is pre-threaded into the
colon — up to ~80% of the shaft's total length (at 500mm) starts out *behind* the
rollers, in open space, unsupported. Left naively under global gravity, this tail
would sag/fall into other scene geometry, generating exactly the kind of contact-count
spike that caused the `mj_stackAlloc` crash fixed by `<size memory="32M"/>` (see
`ITERATION_HISTORY.md` §11) — real risk, not just an unrealistic visual.

**Fix: per-body gravity compensation (`body/@gravcomp`), toggled dynamically per
shaft link.** MuJoCo applies a compensating force (`gravcomp × mass × gravity`) to
any body with `gravcomp > 0`; at `gravcomp=1.0` that body is effectively weightless
regardless of the global gravity vector. This lives in `model.body_gravcomp[body_id]`,
mutable live from Python every step (same pattern already verified safe for
`model.jnt_stiffness` elsewhere in this project — no XML recompile needed). Each
step, for every shaft link, reuse the *existing* projection check the
`shaft_exhausted` termination already computes (`shaft0_proj >= self._roller_ring1_proj`,
`scope_colon_env.py:628`) — just applied per-link instead of only to `shaft_link_0`:
links still behind the roller ring get `gravcomp=1.0` (weightless, matches current
zero-g behavior), links that have advanced past the rollers into the colon get
`gravcomp=0.0` (real gravity — physically correct since the colon wall is there to
support/constrain them, same as reality). No new geometry or contacts needed — this
is a pure per-body force-cancellation toggle, cheap to update (small array write per
step, negligible next to the physics solve itself).

**Layer 3 verification**: isolated physics harness (reuse the roller-pinch-harness
pattern from the original zero-g tuning) to re-sweep stiffness/preload/friction under
gravity before any full-scene test; then a full-scene `view_colon_seed.py` check with
gravity on to confirm the shaft doesn't visibly sag/fail at rest, AND specifically
confirm the un-inserted tail stays weightless/stable (no sag, no spurious contacts)
while the inserted portion behaves under real gravity as expected.

### Final combined check

Only after all three layers pass their own checks: a short smoke-test training run
(not the full 10M-step run) to confirm the combined v8 env doesn't crash and produces
sane early metrics, before committing to a full training run.

## Files to change

All under the new `navigation/v8/` copy:

- `navigation/v8/scope_colon_env.py`, `navigation/v8/sf_encoder.py` — Layer 1
- `navigation/v8/colon_generator.py` — Layer 2 (bend-generation rewrite)
- `navigation/v8/build_collision_scene.py`,
  `navigation/v8/generate_videoscope_one_section.py`,
  `navigation/v8/roller_feed_prototype/build_roller_scene.py` — Layer 3 (gravity +
  physics re-tuning)

**Sync pointer update:** the old `transfer/` mirror and the WSL copy are gone from the
workflow. `navigation/v8/` is a self-contained bundle — copy that single folder to the
native Linux training rig when it's time to train (minus `runs_sf/`, `scenes/`,
`__pycache__/`). See the README's multi-machine sync section. There is still no
auto-sync: verify on the far side after copying.

## Explicitly out of scope for this plan

- The physical 3D-printed maze (mechanical/CAD design).
- Writing the mini report itself (a separate deliverable once simulation results exist).
- Actually running the full v8 training (per this project's standing rule: training
  runs are deliberate, separate steps, never launched automatically).

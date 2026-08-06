# Mujuco_V3 — Autonomous Colonoscope Navigation (MuJoCo + RL)

Clean successor to `Mujuco_V2` (created 2026-07-10). V2 ballooned to ~8.8 GB across seven
superseded Stage4 versions and a duplicate `transfer\` tree; V3 carries only the
functional code, the milestone checkpoints, and a consolidated iteration history.
The full v1→v8_p1 story (architectures, rewards, results, cheats, lessons) is in
**[docs/ITERATION_HISTORY.md](docs/ITERATION_HISTORY.md)** — that file is the source
for any GitHub/portfolio write-up.

> 📍 **For where the project is *now* and what happens next, read
> [docs/CURRENT_PLAN.md](docs/CURRENT_PLAN.md).** It is the single source of truth; if
> anything here or elsewhere disagrees with it, it wins.

## Project map

| Folder | What it is |
|---|---|
| `navigation/v8_p1/` | **LIVE development line.** V8 Phase 1: hardware-matching observation (`force_norm` replaced by cable extension — 8-D state, no force signal), tendon integral control, reworked shaft/tip compliance. Self-contained bundle (own copies of the scope generator, `new8.stl`, `error_map_64.npy`) — sync this one folder to the Linux training rig as-is. Run `runs_sf/v8_p1_v1/` (from-scratch retrain, finished 2026-07-13, 10M steps) reached **93.5% mean progress**, with `tip_stuck` collapsing 73.3% → 12.3% and `stuck` not rising to replace it (2.4%) — training stats. ⚠️ A further retrain is planned that supersedes this checkpoint; see [docs/CURRENT_PLAN.md](docs/CURRENT_PLAN.md). |
| `navigation/v7_shaft/` | **Superseded by `v8_p1/`** (kept runnable). Physical flexible shaft + entrance roller/capstan feeder — the generation that introduced them. Milestone run `runs_sf/v7_shaft_v2/` (~92% mean progress / ~90% completion). `runs_sf/v7_shaft_v4/` trained to 10.01M steps and evaluated at 35.0% completion / 72.3% mean progress, but on a colon generator with a **self-intersection bug** fixed the same day — those numbers are not a clean read, and the planned retrain never happened (v8_p1 overtook it). |
| `navigation/v6_dr/` | **FROZEN milestone snapshot** — do not develop here. Domain-randomised kinematic-base version. Milestone run `runs_sf/v6_dr_700V3/` (93.75% completion / 96.7% mean progress, 80 seeds — eval CSV included). Kept runnable for eval/viewing. |
| `perception/rd_v2/` | DA3-SMALL depth fine-tune pipeline (Blender dataset gen → fine-tune → inference → per-pixel error map). `checkpoints/da3small_finetune_head.pth` is the fine-tuned model — the checkpoint reports **1.43 mm scale-aligned val MAE** vs 8.30 mm zero-shot. ⚠️ Output is **relative, not metric** (`metric_weight = 0`): 1.43 mm is scale-aligned against ground truth and is *not* an accuracy you can obtain at deploy. Never read millimetres off it. (Older docs saying "1.60 mm" are wrong.) `generate_error_map.py` regenerates `error_map_64.npy` end-to-end (needs Blender, hours). |
| `perception/realtime/` | **Deployment-side depth runtime.** `da3_runtime.py` is the single shared inference path — reproduces the fine-tune contract exactly (504×280, ImageNet norm) and provides `to_obs_64()` for the nav obs. `bench_da3.py` times it (**31 fps / 32.18 ms** measured). `probe_camera.py` / `preview_camera_depth.py` drive the live scope camera (**OpenCV index 1**; index 2 is a broken NVIDIA Broadcast device). |
| `simulation/` | Canonical supervisor-provided videoscope tip model (25-hinge tendon-driven section) + manual viewer. The navigation bundles carry their own copies by design — this is the reference. |
| `docs/CURRENT_PLAN.md` | **THE plan — single source of truth.** Where the project is, what's decided, the next retrain, what's still open. Consolidated 2026-08-02 from three conflicting plans. Keep it as the *only* plan document — update it in place rather than writing a new one. |
| `docs/archive/` | Mujuco_V2's root docs verbatim (PROJECT_CONTEXT, REPORT, USER_GUIDE, progress_summary, PROCESS_GUIDE), plus `superseded_plans/` (V3-era plans replaced by CURRENT_PLAN.md). Historical — **never act on anything in here**; paths and numbers inside are stale by definition. |
| `docs/run_archive/` | `config.json` + `sf_log.txt` of every historic training run (v9–v17, v5_smooth, v6_dr ladder, v7_shaft_v1, v7_shaft_v3, v8_p1_v1 — the last also keeps its TensorBoard summaries). **Archive a run here before deleting it.** |

**Naming note:** the live folder is `v8_p1/` ("V8 Phase 1"), *not* `v8/`. Phase 2 (gravity +
anatomical colon) is unstarted, has no folder, and **no longer has a standing plan document** —
the old one was deleted 2026-08-02 because its shaft mass and length figures predated the
2026-07-13 retrain and were wrong by ~1.8×. Re-scope it from live code if and when it starts.

## Interpreter matrix (Windows)

| Interpreter | Use for | Has |
|---|---|---|
| `C:\Python314\python.exe` | **`navigation/` scripts that don't import sample_factory** — `view_colon_seed.py`, `manual_test.py`, and the env itself | mujoco 3.8.1, CPU torch. **`eval_sf.py`/`viewer_sf.py` cannot run here** — Python 3.14 lacks `signal-slot-mp`, so the sample_factory import chain fails. Use the WSL2 mirror. |
| `C:\Users\lukec\AppData\Local\Programs\Python\Python311\python.exe` | **All `perception/` scripts** | CUDA torch (RTX 4060), transformers, editable depth-anything-3 from `C:\Users\lukec\PycharmProjects\Depth-Anything-3` (do not delete that repo) |
| Blender 5.1 (`C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`) | perception dataset regeneration only | headless Cycles GPU (OPTIX) |

Training itself runs on the remote native-Linux rig (40 workers, GPU rendering), not on
Windows. `train_sf.py` defaults `MUJOCO_GL=egl` (Linux); a Windows smoke test needs
`$env:MUJOCO_GL='wgl'` set first.

## Multi-machine sync

`navigation/v8_p1/` is the single sync unit for the Linux rig. **No auto-sync
exists** — copies drift silently the moment one is edited alone. After editing here,
re-copy the whole folder (minus `scenes/`, `runs_sf/`, `__pycache__/`) and verify on
the far side.

**WSL2 (Ubuntu-22.04)** holds a local mirror at `~/mujoco_v3/`, used *only* to eval and
view rig-trained checkpoints — never to train. It reproduces this repo's layout, so the
sync is a plain folder-to-folder copy:

```bash
# from Windows, into WSL
rsync -a --exclude 'scenes/' --exclude '__pycache__/' \
  /mnt/c/Users/lukec/PycharmProjects/Mujuco_V3/navigation/v8_p1 \
  ~/mujoco_v3/navigation/
```

⚠️ Think before adding `--delete`. The WSL mirror can be the *only* copy of a checkpoint
Sample Factory has already rotated out of the rig's `checkpoint_p0/` — `--delete` will
take it with it. (This has happened once, on 2026-07-13: it removed the last surviving
copies of v8_p1_v1's pre-fix `reward_0.440` best checkpoint.)

Its venv is `~/mujoco_v3/.venv` (Python 3.10, mujoco 3.10.0, torch cu130) — this is why
WSL exists at all: `eval_sf.py`/`viewer_sf.py` do not run on Windows. Include `runs_sf/`
in the copy when you want to view a new checkpoint. Run scripts as
`~/mujoco_v3/.venv/bin/python eval_sf.py …` with `MUJOCO_GL=egl` set.

## Quick commands

```powershell
# View a generated colon (regenerates scenes\ automatically)
C:\Python314\python.exe navigation\v8_p1\view_colon_seed.py --seed 0

# Manually drive the shaft/roller model — INTERACTIVE, opens a MuJoCo viewer window
# and waits for keypresses (arrows bend the tip, Shift/Ctrl feed and retract).
# It is NOT headless and never exits on its own — run it yourself, don't script it.
C:\Python314\python.exe navigation\v8_p1\manual_test.py --seed 0
```

**`eval_sf.py` and `viewer_sf.py` do not run on Windows** — Python 3.14 is missing
`signal-slot-mp`, so the sample_factory import chain fails. Run them in the WSL2 mirror:

```bash
cd ~/mujoco_v3/navigation/v8_p1
~/mujoco_v3/.venv/bin/python eval_sf.py --seed 0 --episodes 1 --workers 1
~/mujoco_v3/.venv/bin/python viewer_sf.py --seed 0
```

⚠️ **Do not prefix these with `MUJOCO_GL=egl`.** Each script already picks the right
backend, and an inherited `MUJOCO_GL` overrides that choice in both. `eval_sf.py` defaults
to **osmesa** (CPU): under WSL2's GPU paravirtualisation every depth render becomes a
high-latency round-trip, so egl is ~7x slower (~179 ms/env-step vs ~24 ms) and leaves CPU
*and* GPU idling at ~15%. Pass `--gl egl` only on the native rig, where it does win.
`viewer_sf.py` needs **glfw** to open a window at all — forcing egl on it gives you a
headless backend and no viewer.

`scenes/` folders are always regenerated on demand — safe to delete at any time. Scene XML is
**never cached across runs** (`build_scene()` always regenerates), so a physics/generator edit
cannot be silently masked by a stale scene.

## Current state (2026-08-04, evening)

**The A–F retrain is done. `runs_sf/v8_p1_v1` was retrained from scratch on the rig on 2026-08-04
against all six sim changes and reports ~97% mean progress on the final epoch** — a *training*
statistic from sampled rollouts; no eval has been run on this checkpoint. Run stopped at 4.52M of
5M steps; best checkpoint `best_000000940_3850240_reward_1.204.pth`. **The next action is a physical
rig trial**, not another sim change — see [docs/CURRENT_PLAN.md](docs/CURRENT_PLAN.md) §7 for the
two pre-flight measurements that are not optional.

⚠️ **`viewer_sf.py` now runs the policy deterministically by default** (the Gaussian's mean, i.e.
what you would deploy). It previously sampled, which is *training* mode — with the policy's learned
σ ≈ 0.627 on a ±1 action space, that made the tip look violently unstable when it is not. Pass
`--stochastic` to see the old behaviour. CURRENT_PLAN.md §6.3 has the numbers.

The rest of this section predates the retrain and is kept for the sim-gap background.

## Background — sim-to-real gaps (2026-08-02)

**Full detail — and everything about what happens next — is in
[docs/CURRENT_PLAN.md](docs/CURRENT_PLAN.md).** Summary only here.

**Sim is working.** The 2026-07-13 from-scratch retrain of `v8_p1_v1` passed its pre-registered
test: mean progress **63.2% → 93.5%**, `tip_stuck` **73.3% → 12.3%** without `stuck` rising to
replace it. That vindicated the diagnosis that the dominant failure was an environment bug
(`TIP_STUCK_WINDOW = 50` allowed only 15 mm of forward travel in wall contact — shorter than the
in-contact arc of a bend) rather than a policy limit. Fixes were `TIP_STUCK_WINDOW` 50 → 150,
shaft joint density 76 → 50 links (the 76-link shaft buckled; also +1.58× throughput), and shaft
shortened 50 → 44 links (432.5 mm shaft, 495 mm scope) so the tip stops ~25 mm inside the colon
rather than 35 mm outside it. Full reasoning:
[docs/ITERATION_HISTORY.md](docs/ITERATION_HISTORY.md) §12.

⚠️ **The completion metric moved on 2026-07-13** — `shaft_exhausted` fires ~60 mm of feed earlier,
so completion % is not comparable across that date. Mean progress is the honest axis.

**The project is now in real-hardware bring-up**, and that drives every sim change. Two
sim-to-real gaps are confirmed and quantified:

1. **Tendon force is ~9× too high.** Bench measurement is ~3 N for a full free-space bend; sim
   needs ~27 N (joint `stiffness = 0.5` over a 3.2 mm moment arm). `ACTUATOR_PULL_FORCE_LIMIT`
   is 80 N against a real ~10 N ceiling.
2. **Depth cannot run at 100 Hz.** Deployment is 25 Hz (DA3 measures 31 fps); the env currently
   renders every step.

Both fixes ride **one retrain**, because either alone invalidates the current checkpoint. Largest
remaining unknown: the depth FOV/aspect mismatch between sim (square 64×64 @ `fovy=85`) and the
real lens (16:9 @ ~100°), flagged in `perception/realtime/da3_runtime.py:245-251`.

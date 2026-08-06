# v8_p1_v1 — run 2 (post-2026-07-13 retrain)

**This is the run that worked: 93.5% mean progress**, 10M steps, finished 2026-07-13.
Archived 2026-08-04, pulled off the Linux training rig (`andrew-HP-Z640-Workstation`).

⚠️ **Read-only history. Never run the live line from here, never import from here, and never
take file paths from here** — same rule as `Mujuco_V2`. The live line is
`navigation/v8_p1/`.

## Why the code is archived alongside the weights

Unusually for this folder, this archive contains **`code/` as well as the checkpoint**. That is
deliberate and not optional.

`navigation/v8_p1/` was edited **in place** on 2026-08-02 and 2026-08-04 (Changes A–F, see
`docs/CURRENT_PLAN.md` §5). Those changes altered the observation shape, the state width and the
control rate, so **the current code physically cannot load or run this checkpoint** — the
encoder's first `Linear` shape-mismatches. This project has no version control, so without the
snapshot in `code/` the 93.5% result would have been permanently unreproducible.

`code/` is the exact bundle that trained these weights. Verified before archiving:

| | This snapshot | Live `navigation/v8_p1/` |
|---|---|---|
| tip camera | `fovy="85"` (square) | 67.7° / 16:9 |
| depth obs | 64×64 | 54×96 |
| `STATE_OBS_DIM` | 8 (incl. `ten_x_n`/`ten_z_n`) | 6 |
| control rate | 100 Hz (`PHYSICS_PER_STEP=20`) | 25 Hz (80) |
| `PD_KP` / `PD_KD` | 0.06 / 0.30 | 0.1803 / 0.0475 |
| joint stiffness | 0.5, force limit 80 N | 0.055, 10 N |
| `SHAFT_DAMPING` | 0.3 | 0.05 |
| `DEPTH_DECIMATION` | *did not exist* | 1 (inert) |

So this predates **all six** changes A–F, not just the 2026-08-04 ones. It also still contains the
latent `shaft_tip_ball damping="0.3"` hardcode that Change D found and fixed.

## Contents

```
code/            the 14-file bundle that trained this checkpoint (no scenes/, no __pycache__)
checkpoint_p0/   best_000002106_8626176_reward_1.195.pth   <- the 93.5% weights
                 checkpoint_000002432_9961472.pth          <- last two rotations
                 checkpoint_000002444_10010624.pth
.summary/        tensorboard events from the rig
config.json      Sample Factory run config
sf_log.txt       full training log
```

## To actually run it

Copy `code/` somewhere outside `navigation/` and run from there — it is self-contained (own scope
generator, `new8.stl`, `error_map_64.npy`). Do **not** drop it back into `navigation/v8_p1/`; that
would overwrite the live line.

`eval_sf.py` and `viewer_sf.py` still cannot run on Windows (Python 3.14 lacks `signal-slot-mp`) —
use the WSL2 mirror, and do not prefix with `MUJOCO_GL=egl` (the osmesa default is ~7× faster
there).

Sample Factory **resumes from any checkpoint it finds**, so if you point a training run at a
directory containing these, it will try to continue them and fail on the shape mismatch. Deleting
the run directory — not renaming `EXPERIMENT` — is what starts a run from scratch.

## Sibling

`../v8_p1_v1_run1_pre0713/` is the **earlier** run under the same experiment name, from before the
2026-07-13 physics change (shaft → 44 links, the "scope longer than colon" fix). Metadata only, no
weights. The two are not comparable: the completion metric moved on 2026-07-13, so pre-fix
completion numbers do not mean the same thing.

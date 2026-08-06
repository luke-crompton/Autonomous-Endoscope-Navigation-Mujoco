# Claude Code Instructions

Before doing any work, read, in this order:

- **docs/CURRENT_PLAN.md — the single source of truth.** Where the project is, what is
  decided, what happens next, what is still open. If any other document disagrees with
  it, it wins.
- README.md (project map, interpreter matrix, multi-machine sync rules, current state)
- docs/ITERATION_HISTORY.md (how the project got here — v1 through v8_p1)

Historical deep-detail lives in docs/archive/: the old Mujuco_V2 root docs (paths in them
refer to the old V2 layout), plus docs/archive/superseded_plans/ (V3-era plans replaced by
CURRENT_PLAN.md). **Never act on anything in docs/archive/** — it is stale by definition.

## This project is self-contained

Everything needed lives under this directory. `C:\Users\lukec\PycharmProjects\Mujuco_V2`
still exists but is a **frozen, read-only code archive** — never edit it, never run from
it, and never take file paths from it. The one legitimate reason to open it is to *read*
a superseded script that was deliberately not migrated (e.g.
`Mujuco_V2\Perception\Colon_mesh_xml.py`, the anatomical reference for V8 Phase 2).
If a doc in `docs/archive/` tells you to do something, it is out of date — trust
README.md and docs/ITERATION_HISTORY.md instead.

## Project-specific rules

- `navigation/v8_p1/` is the **live line** and must stay a self-contained one-folder
  bundle (own scope generator, new8.stl, error_map_64.npy) — it is manually synced to a
  Linux training rig and to the WSL2 mirror. Never add cross-folder imports to it.
  It is "V8 Phase 1"; the folder is `v8_p1/`, not `v8/`, and `v8/` does not exist.
- `navigation/v6_dr/` is a FROZEN milestone — never edit it except with explicit approval.
- `navigation/v7_shaft/` is **superseded** by v8_p1 but kept runnable. Don't develop
  there; don't take it as the current design.
- **V8 Phase 1 shipped and trained** (93.5% mean progress, 2026-07-13). The project is now
  in **real-hardware bring-up**, which drives every sim change — see CURRENT_PLAN.md.
- **There is exactly ONE plan document: `docs/CURRENT_PLAN.md`.** Update it in place.
  **Do not write a new standalone plan file** — this project accumulated four competing
  plans and they had to be consolidated on 2026-08-02. A new plan doc recreates the
  problem.
- **Phase 2** (gravity + anatomical left-lateral-decubitus colon) is **owned here and in
  scope**, but is unstarted, has no `navigation/` folder, and **no longer has a plan
  document** — the old one was deleted because its shaft mass/length figures predated the
  2026-07-13 retrain and were wrong by ~1.8×. Re-scope it from live code if it starts;
  per the original design it targets `navigation/v8_p1/` in place. The anatomical
  reference is `Mujuco_V2\Perception\Colon_mesh_xml.py` (read-only).
- `scenes/` folders are generated output — safe to delete, never edit by hand. Scene XML
  is never cached across runs, so a stale scene cannot mask a physics/generator edit.
- Use `C:\Python314\python.exe` for navigation scripts, the Python 3.11 path for
  perception scripts (see README.md interpreter matrix). **`eval_sf.py` and `viewer_sf.py`
  cannot run on Windows** (Python 3.14 lacks `signal-slot-mp`) — run them in the WSL2
  mirror at `~/mujoco_v3/`.
- When rsync'ing to the WSL2 mirror, think before using `--delete`: the mirror can hold
  the only surviving copy of a checkpoint Sample Factory has already rotated off the rig.
- **This project is under version control as of 2026-08-06** (git, branch `main`, pushed to
  the private GitHub repo `luke-crompton/Autonomous-Endoscope-Navigation-Mujoco`). Older docs
  saying otherwise are stale. Two things git does *not* cover, so the old caution still
  applies to them: anything matched by `.gitignore` — **trained checkpoints (`*.pth`),
  `scenes/`, `.summary/` TensorBoard events, perception datasets** — has no history and no
  remote copy, and deleting it is permanent. Checkpoints cannot go in git at all
  (`da3small_finetune_head.pth` is 131 MB, over GitHub's 100 MB hard limit) — publish those
  as GitHub Release assets. Prefer moving superseded files into `docs/archive/` over
  deleting them, and confirm before overwriting anything you have not read.

## Global rules

- Do not edit files unless I explicitly approve the proposed changes.
- Always give options before changing code.
- Prefer the smallest safe change.
- Do not refactor unrelated code.
- Do not change architecture, APIs, file structure, controller structure, or dependencies without permission.
- Do not delete existing code without permission.
- Do not run long simulations, tuning sweeps, training loops, or installs without asking first.
- After every edit, summarise exactly what changed and how to test it.

## Required response format before editing

1. What I think the task is
2. Options
   - Minimal fix
   - Moderate improvement
   - Larger redesign, only if relevant
3. Recommended option
4. Files I would change
5. Risks/trade-offs
6. Tests/checks I would run

Then wait for approval.

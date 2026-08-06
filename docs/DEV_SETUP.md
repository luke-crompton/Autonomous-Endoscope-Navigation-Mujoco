# Development setup

Everything needed to *run* this repo: interpreters, machines, sync rules, and the commands
that work. Moved out of the top-level README on 2026-08-06 so that README could serve a
first-time reader; this is the operational reference for whoever is actually working on it.

For **where the project is and what happens next**, see [CURRENT_PLAN.md](CURRENT_PLAN.md) —
the single source of truth.

---

## Interpreter matrix (Windows)

| Interpreter | Use for | Has |
|---|---|---|
| `C:\Python314\python.exe` | **`navigation/` scripts that don't import sample_factory** — `view_colon_seed.py`, `manual_test.py`, and the env itself | mujoco 3.8.1, CPU torch. **`eval_sf.py`/`viewer_sf.py` cannot run here** — Python 3.14 lacks `signal-slot-mp`, so the sample_factory import chain fails. Use the WSL2 mirror. |
| `C:\Users\lukec\AppData\Local\Programs\Python\Python311\python.exe` | **All `perception/` scripts** | CUDA torch (RTX 4060), transformers, editable depth-anything-3 from `C:\Users\lukec\PycharmProjects\Depth-Anything-3` (do not delete that repo) |
| Blender 5.1 (`C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`) | perception dataset regeneration only | headless Cycles GPU (OPTIX) |

Training itself runs on the remote native-Linux rig (40 workers, GPU rendering), not on
Windows. `train_sf.py` defaults `MUJOCO_GL=egl` (Linux); a Windows smoke test needs
`$env:MUJOCO_GL='wgl'` set first.

---

## Machines

| Machine | Role |
|---|---|
| Windows workstation | Development, scene generation, all perception work |
| Native-Linux training rig | **Where training actually happens.** Not reachable from the dev machine — training is never launched automatically. |
| WSL2 (Ubuntu-22.04) mirror at `~/mujoco_v3/` | Eval and viewing of rig-trained checkpoints **only** — never training. Exists because `eval_sf.py`/`viewer_sf.py` do not run on Windows. |

---

## Multi-machine sync

`navigation/v8_p1/` is the single sync unit for the Linux rig, and must stay a self-contained
one-folder bundle (its own scope generator, `new8.stl`, `error_map_64.npy`). **No auto-sync
exists** — copies drift silently the moment one is edited alone. After editing here, re-copy
the whole folder (minus `scenes/`, `runs_sf/`, `__pycache__/`) and verify on the far side.

The WSL2 mirror reproduces this repo's layout, so the sync is a plain folder-to-folder copy:

```bash
# from Windows, into WSL
rsync -a --exclude 'scenes/' --exclude '__pycache__/' \
  /mnt/c/Users/lukec/PycharmProjects/Mujuco_V3/navigation/v8_p1 \
  ~/mujoco_v3/navigation/
```

⚠️ **Think before adding `--delete`.** The WSL mirror can be the *only* copy of a checkpoint
Sample Factory has already rotated out of the rig's `checkpoint_p0/` — `--delete` will take it
with it. This has happened once, on 2026-07-13: it removed the last surviving copies of
v8_p1_v1's pre-fix `reward_0.440` best checkpoint.

The WSL venv is `~/mujoco_v3/.venv` (Python 3.10, mujoco 3.10.0, torch cu130). Include
`runs_sf/` in the copy when you want to view a new checkpoint.

---

## Commands

### Windows

```powershell
# View a generated colon (regenerates scenes\ automatically)
C:\Python314\python.exe navigation\v8_p1\view_colon_seed.py --seed 0

# Manually drive the shaft/roller model — INTERACTIVE, opens a MuJoCo viewer window
# and waits for keypresses (arrows bend the tip, Shift/Ctrl feed and retract).
# It is NOT headless and never exits on its own — run it yourself, don't script it.
C:\Python314\python.exe navigation\v8_p1\manual_test.py --seed 0
```

### WSL2 — eval and viewer

**`eval_sf.py` and `viewer_sf.py` do not run on Windows** (Python 3.14 is missing
`signal-slot-mp`, so the sample_factory import chain fails).

```bash
cd ~/mujoco_v3/navigation/v8_p1
~/mujoco_v3/.venv/bin/python eval_sf.py --seed 0 --episodes 1 --workers 1
~/mujoco_v3/.venv/bin/python viewer_sf.py --seed 0
```

⚠️ **Do not prefix these with `MUJOCO_GL=egl`.** Each script already picks the right backend,
and an inherited `MUJOCO_GL` overrides that choice in both:

- `eval_sf.py` defaults to **osmesa** (CPU). Under WSL2's GPU paravirtualisation every depth
  render becomes a high-latency round-trip, so egl is ~7× slower (~179 ms/env-step vs ~24 ms)
  and leaves CPU *and* GPU idling at ~15%. Pass `--gl egl` only on the native rig, where it
  does win.
- `viewer_sf.py` needs **glfw** to open a window at all — forcing egl gives you a headless
  backend and no viewer.

`viewer_sf.py` runs the policy **deterministically by default** (the Gaussian's mean, i.e. what
you would deploy). It previously sampled, which is *training* mode — with the policy's learned
σ ≈ 0.627 on a ±1 action space, that made the tip look violently unstable when it is not. Pass
`--stochastic` for the old behaviour.

WSLg's GUI can go unresponsive; `wsl --shutdown` from Windows fixes it.

---

## CAD — refreshing `hardware/cad/`

The SolidWorks working copy lives **outside the repo**, at
`C:\Users\lukec\Documents\Solidworks Projects\Endoscope`. `hardware/cad/` is a published copy and
goes stale when the design moves. To refresh it:

1. Open **`Full assem.SLDASM`** in the working folder.
2. `File → Pack and Go` — it resolves the reference tree, so live parts never have to be
   identified by hand. Tick **"Include suppressed components"** if any exist.
3. Click **`Add File...`** and add **`Layout Assembely.SLDASM`**.
4. Tick **"Flatten to single folder"**.
5. **Save to folder** → `hardware\cad`. Not "Save to Zip file". The original is copied, never
   moved.
6. Close everything, reopen from `hardware\cad`, and check the FeatureManager tree for
   unresolved-component flags. `File → Find References` confirms nothing still resolves back to
   `Documents\`.
7. Re-export `Full assem.STL` — `File → Save As → STL → Options`, with **"Save all components of
   an assembly in a single file"** ticked. A stale STL next to fresh sources is worse than none.

⚠️ **Do not run Pack and Go twice into the same folder.** Both assemblies share parts. A second
run collides with files the first wrote, leaving per-file overwrite decisions — and one wrong
call leaves the two assemblies pointing at *different copies of the same part*. One operation
with both files packages the union of their references once, consistently.

The 2026-08-06 port carried 10 of 16 source parts. The 6 left behind were superseded iterations:
`mg996r` and `Servo holding block` (the design moved to STS3032 serial servos), `Shaft ring`,
`Spool helic reverse`, `Spool_testing`, and the single `sts3032 Bracket` replaced by the
male/female pair. The source folder's `STL/` directory is print history and was deliberately not
copied.

---

## Generated output

`scenes/` folders are always regenerated on demand — safe to delete at any time. Scene XML is
**never cached across runs** (`build_scene()` always regenerates), so a physics or generator
edit cannot be silently masked by a stale scene.

---

## Version control

The project has been under git since 2026-08-06, on `main`, pushed to the **private** GitHub
repo `luke-crompton/Autonomous-Endoscope-Navigation-Mujoco`. `core.autocrlf` is set to `false`
locally so tracked files stay byte-identical to the Linux rig and WSL2 copies.

**git does not cover everything.** `.gitignore` excludes trained checkpoints (`*.pth`),
`scenes/`, `.summary/` TensorBoard events, perception datasets and `__pycache__/`. Those have
no history and no remote copy — deleting one is still permanent. Checkpoints cannot go into git
at all (`da3small_finetune_head.pth` is 131 MB, over GitHub's 100 MB hard limit); publish them
as GitHub Release assets instead.

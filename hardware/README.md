# hardware/ — the physical rig

The real scope, the mechanism that drives it, and the measurements that decide whether the
trained policy transfers. This is where the project is now: simulation work for this phase is
complete and the next milestone is a physical rig trial.

| Folder | What goes in it |
|---|---|
| [`cad/`](cad/) | SolidWorks sources and neutral exports (STEP, STL) for the scope, feeder and mounting |
| [`firmware/`](firmware/) | Arduino / microcontroller code driving the tendon motors and feed rollers |
| [`bringup/`](bringup/) | Test procedures, measurement scripts, and recorded results |

**Live camera and depth tooling is not here** — it lives in `perception/realtime/`
(`probe_camera.py`, `preview_camera_depth.py`, `bench_da3.py`). The dividing line: code that
processes images belongs to `perception/`, code that commands motors or reads encoders belongs
here. Procedures that exercise both go in `bringup/`.

---

## What the rig has to provide

These are requirements the simulator imposes, derived in
[`../docs/CONTROL_LOOP_REPORT_2026-08-05.md`](../docs/CONTROL_LOOP_REPORT_2026-08-05.md) §8.
Nothing here describes measured hardware behaviour.

| | Requirement |
|---|---|
| **Rate** | one decision per 40 ms (25 Hz) |
| **Camera** | 16:9 frame; the policy was trained on 100° horizontal rectilinear |
| **Tendons** | 4 cables in 2 antagonistic pairs, **pull-only**, position-commanded to an absolute pull in mm |
| **Feed** | signed insertion rate, ±30 mm/s at full command |
| **Contact** | a binary tip-contact flag — the only exogenous signal in the state vector besides the image |
| **Software state** | the PD command shaper and previous-action echo must be **reproduced in software**, not measured — five of the six state floats are internal, not sensors |

---

## Status

**Not yet trialled.** Bring-up is in progress; see [`bringup/`](bringup/) for the pre-flight
measurements that have to happen before the first policy-driven run, and
[`../docs/CURRENT_PLAN.md`](../docs/CURRENT_PLAN.md) §7 for the authoritative sequencing.

Three sim-to-real gaps are known and quantified:

1. **Lens distortion.** The real camera is a **122° horizontal fisheye**; the pipeline was built
   on 100° horizontal rectilinear. Fix is perception-side only (measure `K`/`D`, then a
   precomputed `cv2.fisheye` undistort LUT applied before DA3) and is not a retrain blocker.
2. **`MAX_PULL_X` / `MAX_PULL_Z` are sim-derived** (6.795 / 7.361 mm). They normalise two of the
   six live policy inputs, so measuring them on the real scope is a correctness requirement, not
   an improvement.
3. **Camera roll offset is unmeasured.** In sim the tip camera is body-fixed and never rotates,
   so the image↔cable relationship is a constant. On the real scope that constant is unknown.

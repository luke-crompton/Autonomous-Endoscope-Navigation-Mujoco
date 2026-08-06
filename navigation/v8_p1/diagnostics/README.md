# v8_p1/diagnostics/ — physics verification harness

Headless checks for the tendon/shaft/depth model. Written 2026-08-02 to verify Changes A, B and D
in [`docs/CURRENT_PLAN.md`](../../../docs/CURRENT_PLAN.md); kept because the equivalent 2026-07-12
diagnostics were written to a scratchpad and lost, and had to be rebuilt from scratch.

**These are headless and safe to script.** `manual_test.py` (one level up) is *not* — it opens a
MuJoCo viewer and waits for keypresses, so it never exits. Use these for automated verification.

Paths are resolved relative to this file, so they run unchanged on Windows, the Linux training
rig, and the WSL2 mirror. On Windows every script builds the model inside a **64 MB-stack thread**
— MuJoCo's recursive-descent XML compiler blows the default 1 MB stack on the 26-deep `<body>`
nesting and dies with a bare exit-127 and no traceback.

```powershell
# Windows
C:\Python314\python.exe navigation\v8_p1\diagnostics\tendon_force_diag.py --axis x --seconds 5
```
```bash
# WSL2 mirror / rig
~/mujoco_v3/.venv/bin/python navigation/v8_p1/diagnostics/tendon_force_diag.py --axis x
```

## The scripts

| Script | Tests | What it answers |
|---|---|---|
| `tendon_force_diag.py` | **A0 / A1 / A2** | How much cable tension holds full bend, what % of nominal is reached, and whether the joint chain drifts into an S-fold. Free-space tip only, no shaft or colon. `--drift` applies the A2 pass/fail. |
| `tendon_4way_diag.py` | **A3** | Both axes at once through the env's real antagonistic mapping: antagonist fighting, anti-windup behaviour, and `ACTUATOR_KI` overshoot/oscillation. `--pd` ramps via the env's outer PD shaper (what training actually does) rather than a step. |
| `depth_decimation_diag.py` | **B1 / B2 / C1** | That fresh depth frames land 1-in-`DEPTH_DECIMATION`, that the phase varies per episode, that held frames are bit-identical, and a throughput/`ncon` A/B of decimated vs every-step rendering. |
| `shaft_damping_ab.py` | **D1** | `SHAFT_DAMPING` A/B on the full scene with identical seeds *and* identical actions: NaN, velocity blow-up, contact chatter, throughput. |
| `pd_rate_rescale.py` | **F1** | Re-derives `PD_KP`/`PD_KD` for a new control rate by fitting the **wall-clock** step response, since these are the only rate-dependent constants that do not rescale by arithmetic. Reports poles, time constant, overshoot, and hard-reversal slew. No MuJoCo — runs in milliseconds. `--rate 50` for other rates. |

All read their constants live from the modules, so they stay valid after further edits.

## Results on the shipped configuration (2026-08-02)

| Test | Result |
|---|---|
| A0 (pre-change baseline) | 26.57 N to hold full bend — confirmed the 9×-too-stiff diagnosis |
| A1 | **2.92 N** @ 100% of nominal, both axes (bench target ~3 N) |
| A2 | 0.000° joint spread over 40 sim-seconds, no fold |
| A3 | slack tendons 0.000 N; 0.000° overshoot; anti-windup never trips under the PD shaper |
| B1 / B2 | steady-state gap exactly 4; phase varies; longest identical hold run 3 (= D−1) |
| C1 | 1.00× — **rendering is not the bottleneck**, contacts are (`ncon` mean ~54) |
| D1 | NaN 0; peak velocity *lower* than before; throughput 0.96× |
| F1 (2026-08-04) | `PD_KP` 0.06 → **0.1803**, `PD_KD` 0.30 → **0.0475** at 25 Hz. Time constant 214.0 ms preserved to −0.0%; zero overshoot; trajectory RMS 0.0005 mm. A naive ×4 of both gains gives \|z\| = 1.337 — **unstable** |

## Adding a check

Keep it headless, read constants from the modules rather than duplicating them, build inside the
64 MB-stack thread, and record the pass condition in the table above and in `CURRENT_PLAN.md` §6.
A diagnostic whose expected value lives only in a chat log is one that gets rewritten later.

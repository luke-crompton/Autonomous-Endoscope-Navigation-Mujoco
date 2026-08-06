"""
Test A3 — 4-tendon antagonistic diagnostic (docs/CURRENT_PLAN.md).

Drives BOTH axes simultaneously through the env's real antagonistic ctrl mapping
and the same per-substep integral drive + anti-windup loop. Checks:

  * antagonist fighting  — slack tendons must carry ~0 N (a cable cannot push)
  * anti-windup          — how often each integrator freezes against the rail
  * ACTUATOR_KI health   — overshoot and oscillation in the bend response

Optionally ramps the setpoint through the env's outer PD shaper (--pd) instead of
applying a step, which is what actually happens in training.
"""
import argparse
import os
import sys
import threading

import numpy as np

# This file lives in navigation/v8_p1/diagnostics/ and drives the bundle one level up.
# Resolved relatively so it works unchanged on Windows, the Linux training rig, and the
# WSL2 mirror -- v8_p1/ is synced as a unit, so an absolute path would break on the far side.
HERE = os.path.dirname(os.path.abspath(__file__))
V8P1 = os.path.dirname(HERE)
sys.path.insert(0, V8P1)
os.chdir(V8P1)                      # so new8.stl / error_map_64.npy resolve

import mujoco                                          # noqa: E402
import generate_videoscope_one_section as scope_gen     # noqa: E402
import scope_colon_env as env_mod                       # noqa: E402

NAMES = ("px", "pz", "nx", "nz")


def run(seconds: float, use_pd: bool) -> None:
    model = mujoco.MjModel.from_xml_string(scope_gen.generate_xml())
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    n_steps = int(seconds / dt)

    aid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"pull_{n}") for n in NAMES}
    aid_i = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"pull_{n}_i") for n in NAMES}
    tid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, f"tendon_{n}") for n in NAMES}

    jx, jz = [], []
    for d in range(1, scope_gen.N_DISKS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{d}")
        (jx if d % 2 == 0 else jz).append(model.jnt_qposadr[jid])

    mpx, mpz = scope_gen.MAX_PULL_X, scope_gen.MAX_PULL_Z
    rest = scope_gen.REST_LENGTH
    rail = -scope_gen.ACTUATOR_PULL_FORCE_LIMIT
    margin = env_mod.ANTIWINDUP_FORCE_MARGIN

    print(f"  stiffness {model.jnt_stiffness[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,'joint_2')]} | "
          f"rail {scope_gen.ACTUATOR_PULL_FORCE_LIMIT} N | KI {scope_gen.ACTUATOR_KI} | margin {margin} N")
    print(f"  driving BOTH axes to full pull, {'via env PD shaper' if use_pd else 'as a step'}\n")

    cmd_x = cmd_y = prev_x = prev_y = 0.0
    froze = {n: 0 for n in NAMES}
    hist_x, hist_z, ten_hist = [], [], []
    substeps = 0

    for step in range(n_steps):
        if use_pd and step % 20 == 0:           # env applies the shaper once per env step
            vx, vy = cmd_x - prev_x, cmd_y - prev_y
            prev_x, prev_y = cmd_x, cmd_y
            cmd_x = float(np.clip(cmd_x + env_mod.PD_KP * (mpx - cmd_x) - env_mod.PD_KD * vx, -mpx, mpx))
            cmd_y = float(np.clip(cmd_y + env_mod.PD_KP * (mpz - cmd_y) - env_mod.PD_KD * vy, -mpz, mpz))
        elif not use_pd:
            cmd_x, cmd_y = mpx, mpz

        data.ctrl[aid["px"]] = cmd_x
        data.ctrl[aid["nx"]] = -cmd_x
        data.ctrl[aid["pz"]] = cmd_y
        data.ctrl[aid["nz"]] = -cmd_y

        for n in NAMES:
            ap, ai, td = aid[n], aid_i[n], tid[n]
            err = data.ten_length[td] - (rest - data.ctrl[ap])
            combined = data.actuator_force[ap] + data.actuator_force[ai]
            sat = combined <= (rail + margin)
            if sat:
                froze[n] += 1
            data.ctrl[ai] = 0.0 if sat else err
        substeps += 1
        mujoco.mj_step(model, data)

        hist_x.append(np.degrees(np.abs([data.qpos[a] for a in jx]).sum()))
        hist_z.append(np.degrees(np.abs([data.qpos[a] for a in jz]).sum()))
        ten_hist.append([abs(data.actuator_force[aid[n]] + data.actuator_force[aid_i[n]]) for n in NAMES])

    hx, hz = np.array(hist_x), np.array(hist_z)
    th = np.array(ten_hist)

    print("  final tendon tensions (N):")
    for i, n in enumerate(NAMES):
        role = "PULLING" if n in ("px", "pz") else "slack  "
        print(f"    {n} [{role}] {th[-1, i]:7.3f}   peak {th[:, i].max():7.3f}   "
              f"anti-windup froze {100.0*froze[n]/substeps:5.1f}% of substeps")

    print(f"\n  X bend: final {hx[-1]:6.2f}° / 120°  ({100*hx[-1]/120:.1f}%)   peak {hx.max():6.2f}°  "
          f"overshoot {max(0.0, hx.max()-hx[-1]):.3f}°")
    print(f"  Z bend: final {hz[-1]:6.2f}° / 130°  ({100*hz[-1]/130:.1f}%)   peak {hz.max():6.2f}°  "
          f"overshoot {max(0.0, hz.max()-hz[-1]):.3f}°")

    # Oscillation: sign changes in d(bend)/dt over the settled second half.
    half = len(hx) // 2
    dx = np.diff(hx[half:])
    flips = int((np.sign(dx[:-1]) * np.sign(dx[1:]) < 0).sum())
    print(f"\n  settled-half direction reversals (X): {flips}   "
          f"{'OK — no oscillation' if flips < 50 else 'WARN — ringing'}")

    slack_peak = max(th[:, 2].max(), th[:, 3].max())
    print(f"  antagonist fighting: peak slack-side tension {slack_peak:.4f} N   "
          f"{'OK' if slack_peak < 0.5 else 'WARN'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--pd", action="store_true", help="ramp via the env's outer PD shaper")
    a = p.parse_args()

    box = {}
    def target():
        try:
            run(a.seconds, a.pd)
        except BaseException as e:      # noqa: BLE001
            box["e"] = e
            raise

    threading.stack_size(64 * 1024 * 1024)
    th = threading.Thread(target=target)
    th.start(); th.join()
    if "e" in box:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

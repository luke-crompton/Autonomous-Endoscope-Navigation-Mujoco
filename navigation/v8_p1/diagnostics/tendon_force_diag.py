"""
Free-space tendon diagnostic — tests A0 / A1 / A2 of docs/CURRENT_PLAN.md.

Tip only, no shaft, no colon, zero gravity. Holds one tendon pair at full pull
using the SAME antagonistic ctrl mapping and per-substep integral-actuator
anti-windup loop as scope_colon_env.py, then reports:

  A0/A1  cable tension required to hold full bend  (P force + I force)
         and the achieved bend as % of nominal
  A2     drift: angular spread across the same-axis joints
         (uniform bend => spread ~0; S-shape fold => spread grows)

Reads all constants live from the module, so it works unchanged before and
after the Change A edits.

Windows: MuJoCo's recursive-descent XML compiler blows the default 1 MB thread
stack on 26-deep <body> nesting (bare exit 127, no traceback), so the whole run
happens inside a 64 MB-stack thread.

Usage:
    C:\\Python314\\python.exe tendon_force_diag.py --axis x --seconds 5
    C:\\Python314\\python.exe tendon_force_diag.py --axis x --seconds 30 --drift
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

# Must match scope_colon_env.py
ANTIWINDUP_FORCE_MARGIN = 2.0
try:
    import scope_colon_env as _env
    ANTIWINDUP_FORCE_MARGIN = float(_env.ANTIWINDUP_FORCE_MARGIN)
except Exception:
    pass


def run(axis: str, seconds: float, drift: bool) -> None:
    xml = scope_gen.generate_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    dt = model.opt.timestep
    n_steps = int(seconds / dt)

    # px/nx bend X (12 even-idx joints); pz/nz bend Z (13 odd-idx joints).
    if axis == "x":
        pull_name, slack_name = "px", "nx"
        max_pull = scope_gen.MAX_PULL_X
        joint_parity, n_joints, limit_deg = 0, 12, 120.0
    else:
        pull_name, slack_name = "pz", "nz"
        max_pull = scope_gen.MAX_PULL_Z
        joint_parity, n_joints, limit_deg = 1, 13, 130.0

    aid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"pull_{n}")
           for n in ("px", "pz", "nx", "nz")}
    aid_i = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"pull_{n}_i")
             for n in ("px", "pz", "nx", "nz")}
    tid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, f"tendon_{n}")
           for n in ("px", "pz", "nx", "nz")}

    jids = []
    for disk_idx in range(1, scope_gen.N_DISKS):
        if disk_idx % 2 == joint_parity:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{disk_idx}")
            jids.append(model.jnt_qposadr[jid])
    assert len(jids) == n_joints, f"expected {n_joints} joints, found {len(jids)}"

    # Antagonistic mapping, exactly as scope_colon_env.py drives it.
    data.ctrl[aid[pull_name]] = max_pull
    data.ctrl[aid[slack_name]] = -max_pull

    rest = scope_gen.REST_LENGTH
    rail = -scope_gen.ACTUATOR_PULL_FORCE_LIMIT
    names = ("px", "pz", "nx", "nz")

    stiffness = float(model.jnt_stiffness[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_2")])

    print(f"  timestep {dt*1000:.2f} ms | joint stiffness {stiffness} N·m/rad | "
          f"force rail {scope_gen.ACTUATOR_PULL_FORCE_LIMIT} N | "
          f"KP {scope_gen.ACTUATOR_KP} KI {scope_gen.ACTUATOR_KI} | "
          f"anti-windup margin {ANTIWINDUP_FORCE_MARGIN} N")
    print(f"  driving {pull_name} at {max_pull*1000:.3f} mm "
          f"({n_joints} joints, {limit_deg}° nominal)\n")

    report_at = {int(f * n_steps) for f in (0.02, 0.1, 0.25, 0.5, 0.75, 1.0)}
    rows = []

    for step in range(1, n_steps + 1):
        # Per-substep integral drive with anti-windup (mirrors the env).
        for nm in names:
            ap, ai, td = aid[nm], aid_i[nm], tid[nm]
            target_len = rest - data.ctrl[ap]
            err = data.ten_length[td] - target_len
            combined = data.actuator_force[ap] + data.actuator_force[ai]
            data.ctrl[ai] = 0.0 if combined <= (rail + ANTIWINDUP_FORCE_MARGIN) else err
        mujoco.mj_step(model, data)

        if step in report_at:
            ang = np.array([data.qpos[a] for a in jids])
            total_deg = float(np.degrees(np.abs(ang).sum()))
            tension = abs(float(data.actuator_force[aid[pull_name]]
                                + data.actuator_force[aid_i[pull_name]]))
            spread_deg = float(np.degrees(ang.max() - ang.min()))
            signs = np.sign(ang[np.abs(ang) > 1e-4])
            folded = bool(len(signs) and not (signs == signs[0]).all())
            rows.append((step * dt, tension, total_deg,
                         100.0 * total_deg / limit_deg, spread_deg, folded))

    print(f"  {'t (s)':>7} {'tension (N)':>12} {'bend (deg)':>11} "
          f"{'% nominal':>10} {'spread (deg)':>13} {'folded':>7}")
    for t, ten, bend, pct, spread, folded in rows:
        print(f"  {t:7.2f} {ten:12.2f} {bend:11.1f} {pct:10.1f} "
              f"{spread:13.3f} {str(folded):>7}")

    t, ten, bend, pct, spread, folded = rows[-1]
    print(f"\n  RESULT @ {t:.1f}s: tension {ten:.2f} N, bend {pct:.1f}% of nominal, "
          f"joint spread {spread:.3f}°, folded={folded}")
    if drift:
        verdict = "PASS (no drift)" if (spread < 1.0 and not folded) else "FAIL (drifting/folded)"
        print(f"  A2 drift verdict: {verdict}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--axis", choices=["x", "z"], default="x")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--drift", action="store_true", help="apply the A2 pass/fail verdict")
    a = p.parse_args()

    box = {}
    def target():
        try:
            run(a.axis, a.seconds, a.drift)
        except BaseException as e:      # noqa: BLE001
            box["e"] = e
            raise

    threading.stack_size(64 * 1024 * 1024)
    th = threading.Thread(target=target)
    th.start()
    th.join()
    if "e" in box:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

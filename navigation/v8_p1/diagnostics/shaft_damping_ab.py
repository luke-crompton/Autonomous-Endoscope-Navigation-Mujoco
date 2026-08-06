"""
SHAFT_DAMPING A/B stability check (docs/CURRENT_PLAN.md Change D).

Reducing damping 6x is the cheap way to fix the traverse/relaxation ratio, but
damping also suppresses contact chatter and conditions the solver. This measures
what could actually go wrong:

  stability   NaN physics, exploding velocities
  chatter     ncon mean/max, and shaft joint angular-velocity RMS
  throughput  steps/s (less damping can mean a harder solve)
  behaviour   progress achieved under an identical open-loop forward command

Both arms run the SAME seeds and the SAME action sequence, so any difference is
the damping.
"""
import os
import sys
import threading
import time

import numpy as np

# This file lives in navigation/v8_p1/diagnostics/ and drives the bundle one level up.
# Resolved relatively so it works unchanged on Windows, the Linux training rig, and the
# WSL2 mirror -- v8_p1/ is synced as a unit, so an absolute path would break on the far side.
HERE = os.path.dirname(os.path.abspath(__file__))
V8P1 = os.path.dirname(HERE)
sys.path.insert(0, V8P1)
os.chdir(V8P1)                      # so new8.stl / error_map_64.npy resolve
# Windows needs wgl; on Linux leave it alone so the rig gets egl and WSL2 gets its
# osmesa default (forcing egl in WSL2 is ~7x slower -- see README).
if sys.platform == 'win32':
    os.environ.setdefault('MUJOCO_GL', 'wgl')

import build_collision_scene as bcs      # noqa: E402
import scope_colon_env as env_mod        # noqa: E402


def arm(damping, seeds, n_steps, actions):
    bcs.SHAFT_DAMPING = damping
    out = dict(ncon=[], qvel_rms=[], nan=0, progress=[], steps=0, term={})
    t0 = time.perf_counter()
    for seed in seeds:
        # Force a rebuild so the new damping reaches the XML.
        scene = os.path.join(V8P1, "scenes", f"collision_scene_seed{seed}.xml")
        if os.path.exists(scene):
            os.remove(scene)
        env = env_mod.ScopeColonEnv(seed=seed)
        env.reset()
        for i in range(n_steps):
            _, _, term, trunc, info = env.step(actions[i])
            out["ncon"].append(env.data.ncon)
            out["qvel_rms"].append(float(np.sqrt(np.mean(env.data.qvel ** 2))))
            if np.any(np.isnan(env.data.qpos)) or np.any(np.isnan(env.data.qvel)):
                out["nan"] += 1
            out["steps"] += 1
            if term or trunc:
                r = info.get("termination_reason", "trunc")
                out["term"][r] = out["term"].get(r, 0) + 1
                out["progress"].append(float(info.get("episode_extra_stats", {}).get("progress", 0.0)))
                env.reset()
        if not out["progress"]:
            out["progress"].append(float(env.actual_base_s))
        env.close()
    out["secs"] = time.perf_counter() - t0
    return out


def show(label, r):
    print(f"  {label:26s} {r['steps']/r['secs']:6.1f} steps/s | "
          f"ncon {np.mean(r['ncon']):5.1f}/{max(r['ncon']):4d} | "
          f"qvel_rms {np.mean(r['qvel_rms']):7.4f} (max {max(r['qvel_rms']):7.3f}) | "
          f"NaN {r['nan']} | term {r['term']}")


def run():
    seeds = [0, 1]
    n = 400
    rng = np.random.default_rng(42)
    # Identical open-loop command for both arms: steady advance + slow steering sweep.
    actions = []
    for i in range(n):
        actions.append(np.array([
            0.6 * np.sin(i / 60.0),
            0.6 * np.cos(i / 85.0),
            1.0,
        ], dtype=np.float32))

    print(f"  {len(seeds)} seeds x {n} steps, identical action sequence in both arms\n")
    old = arm(0.3, seeds, n, actions)
    new = arm(0.05, seeds, n, actions)

    print("  arm                        throughput | contacts     | shaft motion            | stability")
    show("OLD  SHAFT_DAMPING=0.3", old)
    show("NEW  SHAFT_DAMPING=0.05", new)

    print()
    print(f"  throughput change : {(new['steps']/new['secs'])/(old['steps']/old['secs']):.2f}x")
    print(f"  ncon change       : {np.mean(new['ncon'])/np.mean(old['ncon']):.2f}x mean, "
          f"{max(new['ncon'])/max(old['ncon']):.2f}x max")
    print(f"  qvel_rms change   : {np.mean(new['qvel_rms'])/np.mean(old['qvel_rms']):.2f}x "
          f"(expect >1 — less damping means faster motion; watch for a blow-up, not a rise)")
    verdict = "PASS" if (new["nan"] == 0 and max(new["qvel_rms"]) < 100) else "FAIL"
    print(f"\n  STABILITY VERDICT : {verdict}  (NaN={new['nan']}, peak qvel_rms={max(new['qvel_rms']):.3f})")

    # Leave the scenes dir consistent with the committed constant.
    bcs.SHAFT_DAMPING = 0.05


def main():
    box = {}
    def target():
        try:
            run()
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

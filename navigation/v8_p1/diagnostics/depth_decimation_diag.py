"""
Tests B1 / B2 / C1 — depth decimation + full-scene sanity (docs/CURRENT_PLAN.md).

B1  fresh renders land exactly 1-in-DEPTH_DECIMATION; the phase differs between
    episodes; step 0 of every episode is always fresh.
B2  within a hold window the obs is BIT-IDENTICAL (noise held, not re-rolled).
C1  the full scene steps cleanly; reports ncon and steps/sec.

Detects a "fresh" frame by comparing the obs array to the previous one, which is
what the policy actually experiences — not by instrumenting the render call.
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

import scope_colon_env as env_mod            # noqa: E402


def episode_pattern(env, n_steps, rng):
    obs, _ = env.reset()
    prev = obs["depth"].copy()
    fresh_flags = [True]          # step 0 is the reset frame, always fresh
    identical_in_hold = True
    for _ in range(n_steps):
        a = rng.uniform(-1, 1, size=3).astype(np.float32)
        obs, _, term, trunc, _ = env.step(a)
        d = obs["depth"]
        changed = not np.array_equal(d, prev)
        fresh_flags.append(changed)
        if not changed and not np.array_equal(d, prev):
            identical_in_hold = False       # belt and braces
        prev = d.copy()
        if term or trunc:
            break
    return fresh_flags, identical_in_hold


def run():
    D = env_mod.DEPTH_DECIMATION
    print(f"  DEPTH_DECIMATION = {D}  ->  expect 1 fresh frame per {D} steps "
          f"({100.0/D:.0f} Hz depth on a 100 Hz loop)\n")

    rng = np.random.default_rng(0)
    patterns = {}
    for seed in (0, 1, 2):
        env = env_mod.ScopeColonEnv(seed=seed)
        flags, ok = episode_pattern(env, 40, rng)
        idx = [i for i, f in enumerate(flags) if f]
        gaps = np.diff(idx)
        patterns[seed] = idx
        frac = sum(flags) / len(flags)
        print(f"  seed {seed}: fresh at steps {idx[:12]}{'...' if len(idx) > 12 else ''}")
        print(f"          fresh fraction {frac:.3f} (expect ~{1.0/D:.3f}); "
              f"gaps unique={sorted(set(gaps.tolist()))}; step0 fresh={flags[0]}")
        env.close()

    print()
    # The FIRST gap is the per-episode phase offset (reset always forces a fresh
    # frame regardless of phase), so it is 1..D by design. Every gap after that
    # must be exactly D.
    steady_ok = all(
        set(np.diff(idx)[1:].tolist()) <= {D} for idx in patterns.values()
    )
    first_gaps = {s: int(np.diff(idx)[0]) for s, idx in patterns.items()}
    print(f"  B1 spacing : {'PASS' if steady_ok else 'FAIL'} — every steady-state gap == {D} "
          f"(first gap is the phase offset, excluded)")
    phase_varies = len(set(first_gaps.values())) > 1
    print(f"  B1 phase   : offsets per seed {first_gaps} — "
          f"{'PASS (varies across episodes)' if phase_varies else 'FAIL (fixed phase)'}")

    # B2: longest run of consecutive bit-identical frames — should reach D-1 holds.
    env = env_mod.ScopeColonEnv(seed=7)
    obs, _ = env.reset()
    prev = obs["depth"].copy()
    run_len, best = 0, 0
    for _ in range(40):
        obs, _, term, trunc, _ = env.step(np.zeros(3, dtype=np.float32))
        if np.array_equal(obs["depth"], prev):
            run_len += 1
            best = max(best, run_len)
        else:
            run_len = 0
        prev = obs["depth"].copy()
        if term or trunc:
            break
    print(f"  B2 identity: {'PASS' if best == D - 1 else 'FAIL'} — longest bit-identical "
          f"hold run = {best} (expect {D - 1})")
    env.close()

    # C1: throughput + contacts on the full scene, decimated vs every-step.
    print()
    results = {}
    for label, dec in (("every-step (old, D=1)", 1), (f"decimated (new, D={D})", D)):
        env_mod.DEPTH_DECIMATION = dec
        env = env_mod.ScopeColonEnv(seed=0)
        env.reset()
        ncon, n = [], 300
        env.step(np.zeros(3, dtype=np.float32))          # warm up the renderer
        t0 = time.perf_counter()
        for _ in range(n):
            _, _, term, trunc, _ = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            ncon.append(env.data.ncon)
            if term or trunc:
                env.reset()
        dt = time.perf_counter() - t0
        results[label] = (n / dt, float(np.mean(ncon)), max(ncon))
        env.close()
    env_mod.DEPTH_DECIMATION = D

    for label, (sps, m, mx) in results.items():
        print(f"  C1 {label:24s}: {sps:6.1f} steps/s | ncon mean {m:5.1f}, max {mx}")
    old = results["every-step (old, D=1)"][0]
    new = results[f"decimated (new, D={D})"][0]
    print(f"  C1 speedup : {new/old:.2f}x  (rendering is "
          f"{'a meaningful share of' if new/old > 1.15 else 'NOT the bottleneck in'} step cost)")


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

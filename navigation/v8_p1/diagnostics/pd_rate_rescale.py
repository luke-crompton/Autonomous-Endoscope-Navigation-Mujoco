"""
Test F1 — re-derive PD_KP / PD_KD for a new control rate (docs/CURRENT_PLAN.md).

The env's outer PD command shaper is a discrete filter whose gains are
PER STEP, so changing the control rate silently changes how fast the tip
tracks in WALL-CLOCK time. Most of the other rate-dependent constants rescale
by exact arithmetic (advance/step x4, step windows /4); these two do not, which
is why they get their own diagnostic.

The filter, verbatim from scope_colon_env.step():

    vel   = c[n] - c[n-1]
    c[n+1] = c[n] + KP*(target - c[n]) - KD*vel

i.e.  c[n+1] = (1 - KP - KD)*c[n] + KD*c[n-1] + KP*target

    characteristic:  z^2 - (1 - KP - KD)*z - KD = 0
    steady state:    c -> target for any KP > 0

Note KD > 0 REQUIRES one negative root (KD = -z1*z2), so the "damping" here is
an alternating-mode term, not a continuous-time derivative. That is exactly why
these gains cannot be rescaled analytically by 4x -- the second mode has no
continuous-time counterpart to preserve. So this fits the WALL-CLOCK step
response numerically instead, which is the thing that actually has to be
invariant.

No MuJoCo, no scene build: the shaper is pure software upstream of ctrl. Runs in
milliseconds.

    python diagnostics/pd_rate_rescale.py                 # 100 Hz -> 25 Hz
    python diagnostics/pd_rate_rescale.py --rate 50
    python diagnostics/pd_rate_rescale.py --seconds 3 --coarse 90
"""
import argparse
import os
import sys

import numpy as np

# This file lives in navigation/v8_p1/diagnostics/ and drives the bundle one level up.
# Resolved relatively so it works unchanged on Windows, the Linux training rig, and the
# WSL2 mirror -- v8_p1/ is synced as a unit, so an absolute path would break on the far side.
HERE = os.path.dirname(os.path.abspath(__file__))
V8P1 = os.path.dirname(HERE)
sys.path.insert(0, V8P1)

import scope_colon_env as env_mod            # noqa: E402
import generate_videoscope_one_section as scope_gen   # noqa: E402

# ---------------------------------------------------------------------------
# The REFERENCE baseline is pinned here, NOT read from scope_colon_env.
# ---------------------------------------------------------------------------
# This is the one diagnostic in this folder that must not read its reference
# live from the module. The other scripts check a property of whatever is
# currently configured; this one checks that the CURRENT gains still reproduce
# a HISTORICAL response. Once Change F landed, reading PD_KP/PD_KD from the
# module made the script fit 25 Hz against itself and pass tautologically --
# caught immediately after the edit, which is exactly the failure mode a
# self-referential check produces.
#
# These are the v8_p1_v1 values -- the configuration that trained to 93.5% mean
# progress at 100 Hz. They are the behaviour being preserved across rate
# changes. DO NOT update them when PD_KP/PD_KD change; that would erase the
# anchor.
REF_RATE = 100.0
REF_KP   = 0.06
REF_KD   = 0.30
MJ_DT    = 0.0005          # MuJoCo timestep -- fixed, never rescaled


# --------------------------------------------------------------------------
# The filter
# --------------------------------------------------------------------------

def pd_run(kp, kd, target_of_t, n_steps, limit):
    """Run the shaper for n_steps. target_of_t(k) gives the setpoint at step k.

    Mirrors step() exactly, including the clip to +-max_pull that bounds the
    command (NOT the tip angle).
    """
    c = 0.0
    prev = 0.0
    out = np.empty(n_steps + 1, dtype=np.float64)
    out[0] = 0.0
    for k in range(n_steps):
        vel = c - prev
        prev = c
        c = float(np.clip(c + kp * (target_of_t(k) - c) - kd * vel, -limit, limit))
        out[k + 1] = c
    return out


def poles(kp, kd):
    """Roots of z^2 - (1-KP-KD)z - KD = 0."""
    return np.roots([1.0, -(1.0 - kp - kd), -kd])


def describe(kp, kd, dt):
    z = poles(kp, kd)
    dom = z[np.argmax(np.abs(z))]
    mag = abs(dom)
    # Dominant time constant in SECONDS -- the rate-invariant quantity.
    tau = float('inf') if mag >= 1.0 or mag <= 0 else -dt / np.log(mag)
    return z, mag, tau


def step_metrics(traj, target, dt):
    """Overshoot %, 95% settling time (s), and first-step size."""
    peak = float(np.max(np.abs(traj)))
    overshoot = 100.0 * (peak - abs(target)) / abs(target) if abs(target) > 0 else 0.0
    reached = np.where(np.abs(traj) >= 0.95 * abs(target))[0]
    settle = float(reached[0] * dt) if len(reached) else float('nan')
    return max(0.0, overshoot), settle, float(traj[1])


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------

def fit_gains(ref_traj, ref_dt, new_dt, seconds, limit, target,
              coarse=120, refines=4):
    """Find (KP, KD) at new_dt whose WALL-CLOCK trajectory best matches ref_traj.

    Compares only at the coarse grid's sample times -- the new rate cannot
    represent anything faster than its own step, and pretending otherwise would
    bias the fit toward mimicking sampling artefacts instead of the response.
    """
    ratio = int(round(new_dt / ref_dt))
    n_new = int(round(seconds / new_dt))
    ref_at_new = ref_traj[: n_new * ratio + 1 : ratio]     # resample reference

    def err(kp, kd):
        if kp <= 0:
            return np.inf
        z = np.abs(poles(kp, kd))
        if np.max(z) >= 1.0:                # unstable -- reject outright
            return np.inf
        t = pd_run(kp, kd, lambda _k: target, n_new, limit)
        n = min(len(t), len(ref_at_new))
        return float(np.sqrt(np.mean((t[:n] - ref_at_new[:n]) ** 2)))

    lo_p, hi_p, lo_d, hi_d = 1e-4, 0.95, 0.0, 0.95
    best = (np.inf, None, None)
    for _ in range(refines):
        for kp in np.linspace(lo_p, hi_p, coarse):
            for kd in np.linspace(lo_d, hi_d, coarse):
                e = err(kp, kd)
                if e < best[0]:
                    best = (e, kp, kd)
        _, bp, bd = best
        sp, sd = (hi_p - lo_p) / coarse, (hi_d - lo_d) / coarse
        lo_p, hi_p = max(1e-4, bp - 2 * sp), bp + 2 * sp
        lo_d, hi_d = max(0.0, bd - 2 * sd), bd + 2 * sd
    return best[1], best[2], best[0]


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rate', type=float, default=None,
                    help="target control rate in Hz. Default: whatever scope_colon_env is "
                         "CURRENTLY configured for, so the run doubles as a check that the "
                         "shipped gains match the fit.")
    ap.add_argument('--seconds', type=float, default=2.0,
                    help='wall-clock window to match over')
    ap.add_argument('--coarse', type=int, default=120, help='grid points per axis')
    ap.add_argument('--axis', choices=['x', 'z'], default='x')
    args = ap.parse_args()

    # Reference is the pinned historical baseline, never the live module.
    kp0, kd0, dt0 = REF_KP, REF_KD, 1.0 / REF_RATE
    # The live module supplies what we are CHECKING, not what we compare against.
    cur_rate = 1.0 / (env_mod.DEFAULT_PHYSICS_PER_STEP * MJ_DT)
    rate = cur_rate if args.rate is None else args.rate
    dt1 = 1.0 / rate
    limit = (scope_gen.MAX_PULL_X if args.axis == 'x' else scope_gen.MAX_PULL_Z)
    target = limit                      # full-scale step, the worst case

    print(f"Reference : {REF_RATE:.0f} Hz  KP={kp0}  KD={kd0}   "
          f"<- PINNED v8_p1_v1 baseline, not read from the module")
    print(f"Target    : {rate:.0f} Hz  -> PHYSICS_PER_STEP={int(round(dt1/MJ_DT))}")
    print(f"Live env  : {cur_rate:.0f} Hz  KP={env_mod.PD_KP}  KD={env_mod.PD_KD}")
    print(f"Axis {args.axis}: full-scale step to {limit*1e3:.3f} mm\n")

    n0 = int(round(args.seconds / dt0))
    ref = pd_run(kp0, kd0, lambda _k: target, n0, limit)
    z0, m0, tau0 = describe(kp0, kd0, dt0)
    o0, s0, f0 = step_metrics(ref, target, dt0)

    print("--- reference response ---")
    print(f"  poles           {np.round(z0, 4)}   |dominant| = {m0:.4f}")
    print(f"  time constant   {tau0*1e3:7.1f} ms   <- the rate-invariant quantity")
    print(f"  95% settle      {s0*1e3:7.1f} ms")
    print(f"  overshoot       {o0:7.3f} %")
    print(f"  first step      {f0*1e3:7.4f} mm\n")

    kp1, kd1, rms = fit_gains(ref, dt0, dt1, args.seconds, limit, target,
                              coarse=args.coarse)
    n1 = int(round(args.seconds / dt1))
    new = pd_run(kp1, kd1, lambda _k: target, n1, limit)
    z1, m1, tau1 = describe(kp1, kd1, dt1)
    o1, s1, f1 = step_metrics(new, target, dt1)

    print(f"--- fitted at {rate:.0f} Hz ---")
    print(f"  PD_KP = {kp1:.4f}   (was {kp0})")
    print(f"  PD_KD = {kd1:.4f}   (was {kd0})")
    print(f"  poles           {np.round(z1, 4)}   |dominant| = {m1:.4f}")
    print(f"  time constant   {tau1*1e3:7.1f} ms   ({100*(tau1-tau0)/tau0:+.1f}% vs reference)")
    print(f"  95% settle      {s1*1e3:7.1f} ms   ({(s1-s0)*1e3:+.1f} ms)")
    print(f"  overshoot       {o1:7.3f} %")
    print(f"  first step      {f1*1e3:7.4f} mm   ({f1/f0:.2f}x reference)")
    print(f"  trajectory RMS  {rms*1e3:7.4f} mm over {args.seconds:.1f} s\n")

    # ---- the property KD exists for: resist a hard command reversal ----
    # Comment at PD_KD: "damps cmd velocity - resists rapid reversals that
    # rate-limiting only clipped rather than actively opposing." A gain fit that
    # matched the step but lost this would be a regression, so test it directly.
    def rev(dt):
        return lambda k: (target if k * dt < args.seconds / 2 else -target)

    r0 = pd_run(kp0, kd0, rev(dt0), n0, limit)
    r1 = pd_run(kp1, kd1, rev(dt1), n1, limit)
    # Peak per-second slew during the reversal -- rate-invariant, unlike per-step.
    slew0 = float(np.max(np.abs(np.diff(r0)))) / dt0
    slew1 = float(np.max(np.abs(np.diff(r1)))) / dt1
    print("--- hard reversal (+full -> -full) ---")
    print(f"  peak slew   reference {slew0*1e3:8.1f} mm/s")
    print(f"              fitted    {slew1*1e3:8.1f} mm/s   ({slew1/slew0:.2f}x)")

    # ---- pass conditions ----
    checks = [
        ("stable (all |z| < 1)",           float(np.max(np.abs(z1))) < 1.0),
        ("time constant within 10%",       abs(tau1 - tau0) / tau0 < 0.10),
        ("no overshoot (<1%)",             o1 < 1.0),
        ("steady state reaches target",    abs(new[-1] - target) / target < 0.02),
        ("reversal slew within 2x",        slew1 / slew0 < 2.0),
    ]
    # When fitting at the rate the env is actually configured for, also assert the
    # SHIPPED gains match the fit. This is what stops the script degenerating into
    # a tautology once a rate change has landed -- without it, re-running after the
    # edit proves nothing.
    if abs(rate - cur_rate) < 1e-9:
        agree = (abs(env_mod.PD_KP - kp1) < 0.02 and abs(env_mod.PD_KD - kd1) < 0.02)
        checks.append(
            (f"shipped gains match fit (env {env_mod.PD_KP}/{env_mod.PD_KD} "
             f"vs fit {kp1:.4f}/{kd1:.4f})", agree))
    else:
        print(f"\n  NOTE: fitting at {rate:.0f} Hz but the env runs at {cur_rate:.0f} Hz "
              f"-- shipped-gain check skipped.")
    print("\n--- F1 pass conditions ---")
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    print(f"\n{'F1 PASSED' if ok else 'F1 FAILED'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

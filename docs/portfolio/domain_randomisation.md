> ⚠️ **Harvested 2026-08-06 from the v6_dr portfolio snapshot (`colonoscope-rl`, last touched
> 2026-06-30). NOT yet refreshed for v8_p1.** It describes the kinematic-base v6 line — no
> physical flexible shaft, no roller/capstan feeder, and the pre-v7 observation design. Anything
> here is superseded by [`../CURRENT_PLAN.md`](../CURRENT_PLAN.md) and
> [`../ITERATION_HISTORY.md`](../ITERATION_HISTORY.md), which win in any disagreement.

# Domain Randomisation

## Motivation

A real flexible videoscope's tendon-to-tip mapping is not static. Bowden cable friction increases at each bend in the shaft, reducing tip deflection for the same motor command. Tendon slack varies with insertion depth. Command signals experience mechanical delay from inertia and tendon stretch. Forward advance resistance depends on how much shaft is inserted and how many bends it traverses.

A policy trained on a fixed simulation would fail immediately on hardware because the actuator dynamics it learned simply do not transfer. Rather than modelling these effects explicitly — which would require continuously measuring unmeasurable shaft state — domain randomisation trains the policy to adapt implicitly. Each training episode samples a fresh set of actuator parameters from the ranges below. The RL visual feedback loop (depth + proprioception) then provides the implicit gain estimation; DR trains that mechanism across the full expected hardware range.

The result is a policy that requires **no hardware calibration step** at deployment. It adapts online to whatever actuator dynamics the current insertion state presents.

---

## Parameters

All ranges are sampled uniformly per episode, independently per axis where applicable.

| Parameter | Code name | Range | Physical source | Calibration method |
|-----------|-----------|-------|-----------------|-------------------|
| Steering gain X | `DR_GAIN_X` | [0.45, 1.0] | Tendon friction at shaft bends. 1.0 = free-air; 0.45 ≈ half efficiency at a tight S-bend. X and Y axes are independent because shaft curvature is asymmetric during insertion. | Command a known tendon displacement in free air vs at maximum insertion depth; ratio of measured tip angle gives the gain range. |
| Steering gain Y | `DR_GAIN_Y` | [0.45, 1.0] | Same as above, independent axis. | Same as above. |
| Dead zone X | `DR_DEAD_X` | [0.0, 0.03] × max_pull | Tendon slack that must be taken up before the tip moves. Slack varies with insertion depth and shaft geometry. 0.03 → ~0.7 mm of slack on a 24 mm max-pull actuator. | Measure the command threshold below which tip angle is zero, at each insertion depth. |
| Dead zone Y | `DR_DEAD_Y` | [0.0, 0.03] × max_pull | Same as above, independent axis. | Same as above. |
| Command lag | `DR_LAG` | [0, 2] steps (0–20 ms) | Mechanical inertia + tendon stretch delay between issuing a command and tip response. Increases with shaft length and bend tightness. Each env step = 10 ms. | Step-response test: measure time from command issuance to first measurable tip movement. |
| Advance gain | `DR_ADV_GAIN` | [0.65, 1.0] | Friction on the shaft body when pushing forward. Wider range than steering gains because the whole rigid shaft is being pushed, not just a Bowden wire. Floor of 0.65 keeps insertion speed usable (≥19.5 mm/s at default rate). | Measure actual insertion speed vs commanded advance rate at different insertion depths and colon curvatures. |

---

## DA3 depth noise injection

Alongside actuator DR, the depth observation is corrupted by spatially-resolved multiplicative noise:

```
depth_obs += N(0, error_map) × depth_obs
```

`error_map_64.npy` is a 64×64 spatial map of mean relative depth errors computed from 500 held-out Blender evaluation frames. It captures the systematic structure of DA3-SMALL's residual errors (e.g. near lumen edges and highlight regions). Injecting this noise during RL training bridges the gap between MuJoCo's clean depth rendering and the real DA3 output that will be used at deployment.

---

## Effect on training

Without domain randomisation, the v5 policy achieved 44% completion on 80 seeds. Adding actuator DR + DA3 noise injection in v6 brought this to **93.75%** — a 2.1× improvement in completion rate. The policy is not simply more robust; it learned a qualitatively different strategy: it actively corrects for slow or lagged actuator response by steering more aggressively when progress stalls, rather than treating its own commands as perfectly executed.

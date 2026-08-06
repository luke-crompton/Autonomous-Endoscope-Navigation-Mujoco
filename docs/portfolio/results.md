> ⚠️ **Harvested 2026-08-06 from the v6_dr portfolio snapshot (`colonoscope-rl`, last touched
> 2026-06-30). NOT yet refreshed for v8_p1.** It describes the kinematic-base v6 line — no
> physical flexible shaft, no roller/capstan feeder, and the pre-v7 observation design. Anything
> here is superseded by [`../CURRENT_PLAN.md`](../CURRENT_PLAN.md) and
> [`../ITERATION_HISTORY.md`](../ITERATION_HISTORY.md), which win in any disagreement.

# Evaluation Results

## Protocol

**Checkpoint:** `v6_dr_700V3/checkpoint_p0/best_000000728_2981888_reward_1.238.pth` (~3M training steps)

**Setup:** 80 procedurally-generated colon seeds × 5 episodes per seed = 400 episodes total. All colons are exactly 700 mm long, so episode progress ∈ [0, 1] is an absolute fraction of the task — no per-seed normalisation is applied and all seeds have identical maximum reward. Deterministic policy rollout (no exploration noise). 64×64 depth observation, 32 parallel workers.

**Progress definition:** high-water-mark insertion depth as a fraction of 700 mm. An episode counts as "completed" only when the scope reaches 100% (≥ 700 mm inserted).

---

## Aggregate results

| Metric | Value |
|--------|-------|
| Seeds at 100% completion | 65 / 80 |
| Seeds at ≥ 80% completion rate | 74 / 80 |
| Mean progress across all seeds | 96.7% |
| Mean progress across failing seeds | 82.0% |
| Worst-seed mean progress | 59.2% (seed 61) |
| Closest near-miss (consistently) | seed 31 — 92% mean progress, only 1/5 episodes completed |
| Mean episode length | ~2,680 steps (~26.8 s at 100 Hz) |
| Primary failure mode | `stuck` — no new territory for 300 steps |

---

## Per-seed breakdown (15 worst seeds)

Seeds ranked 1–65 all achieved 100% completion on all 5 episodes. The table below shows the 15 seeds where at least one episode failed.

| Rank | Seed | Completion | Mean progress | Goal gap (mm) | Primary failure |
|------|------|-----------|--------------|--------------|----------------|
| 66 | 8 | 80% | 97.2% | 18.1 | tip_stuck |
| 67 | 66 | 80% | 96.4% | 23.0 | tip_stuck |
| 68 | 33 | 80% | 94.7% | 34.2 | tip_stuck |
| 69 | 34 | 80% | 94.0% | 38.7 | stuck |
| 70 | 6 | 80% | 91.6% | 53.9 | stuck |
| 71 | 3 | 80% | 88.7% | 72.7 | stuck |
| 72 | 40 | 80% | 85.0% | 96.3 | stuck |
| 73 | 65 | 80% | 84.1% | 102.3 | stuck |
| 74 | 7 | 80% | 83.2% | 107.7 | stuck |
| 75 | 70 | 60% | 78.3% | 139.3 | stuck + tip_stuck |
| 76 | 12 | 60% | 63.7% | 233.0 | stuck |
| 77 | 32 | 60% | 63.5% | 234.8 | stuck |
| 78 | 23 | 40% | 63.1% | 237.0 | stuck + tip_stuck |
| 79 | 61 | 40% | 59.2% | 262.0 | stuck |
| 80 | 31 | 20% | 92.0% | 51.2 | tip_stuck (×3 of 4 failures) |

*Goal gap = mean distance remaining at episode end, in mm. Completion = fraction of 5 episodes where 100% progress was reached.*

---

## Failure mode analysis

**`stuck` truncation** (no new territory for 300 consecutive steps) is the dominant failure. It typically occurs at sharp-bend sections where the scope needs to slow, steer toward the lumen centre, then re-advance — the policy succeeds at this most of the time but occasionally anchors at a haustral ring.

**`tip_stuck` truncation** (tip disk in wall contact for 50 consecutive steps) appears mostly on seeds with tight, asymmetric bends. The tip-contact obs bit gives the policy a signal to escape, but in a few episodes it cannot disengage before the truncation window expires.

**Seed 31** is the most interesting failure: mean progress 92% across 5 episodes, but only 1/5 episodes reached completion. The policy consistently navigates to the final 50 mm then gets caught in a tip-stuck cycle at what is likely a tight final flexure on that particular geometry.

---

## Limitations

- **Simulation only.** The policy has not been tested on physical hardware. Real insertion dynamics (tissue deformation, fluid, peristalsis) are not modelled.
- **Fixed colon length.** All evaluation colons are exactly 700 mm. Real colonoscopy targets vary, and the policy has no representation of remaining distance.
- **Simplified geometry.** Haustra and random bends are modelled, but the colon does not move, deform, or change lumen diameter dynamically during an episode.
- **Perception gap.** The `error_map_64.npy` noise model captures the spatial structure of DA3-SMALL's mean errors on Blender-rendered imagery. Real tissue appearance may produce different error distributions beyond what this injection covers.

---

## Training configuration

Final run: `v6_dr_700V3`, trained to 10M steps on a Linux workstation (RTX GPU). Best checkpoint saved at ~3M steps.

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Algorithm | APPO | Asynchronous PPO via SampleFactory |
| Rollout workers | 40 × 2 envs = 80 parallel envs | |
| Rollout length | 64 steps | Also sets BPTT window for GRU |
| Batch size | 4096 | |
| Minibatches per epoch | 2 | |
| Epochs per update | 1 | |
| Learning rate | 1e-4 | Constant schedule |
| Entropy coefficient | 0.003 | Tuned to match task reward scale (~0.001/step) |
| Gamma | 0.99 | |
| GAE lambda | 0.95 | |
| PPO clip ratio | 0.2 | |
| Max grad norm | 0.5 | |
| Optimizer | Adam | β1=0.9, β2=0.999, ε=1e-6 |
| GRU hidden size | 512 | 1 layer |
| Encoder | NatureCNN → 512D MLP | Custom depth CNN (`sf_encoder.py`) |
| Normalise returns | True | |
| Normalise observations | False | Depth normalised per-frame in env |

---

## Full data

`docs/eval_results.csv` contains the complete per-seed table with all columns: completion metrics, progress mean/min/max, goal gap, reward, steps, contact force, weld lag, and per-reason termination counts.

View training curves:
```bash
tensorboard --logdir docs/training_curves/
```

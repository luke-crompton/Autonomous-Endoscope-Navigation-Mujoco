> ⚠️ **Harvested 2026-08-06 from the v6_dr portfolio snapshot (`colonoscope-rl`, last touched
> 2026-06-30). NOT yet refreshed for v8_p1.** It describes the kinematic-base v6 line — no
> physical flexible shaft, no roller/capstan feeder, and the pre-v7 observation design. Anything
> here is superseded by [`../CURRENT_PLAN.md`](../CURRENT_PLAN.md) and
> [`../ITERATION_HISTORY.md`](../ITERATION_HISTORY.md), which win in any disagreement.

# Design Iterations: Reward Shaping and Observation Vector

## Framework: SB3 PPO → SampleFactory APPO

Early training runs (PPO v1–v6) used Stable-Baselines3 synchronous PPO. After the curl-cheat fix reached 85% mean progress, the switch to SampleFactory APPO was made to accelerate further iteration — faster rollout throughput meant reward and observation changes could be evaluated in hours rather than days. All subsequent runs (APPO v7 onward) use SF.

Beyond raw speed, the architectural differences were meaningful for this environment:

**Async rollout/train overlap.** In synchronous PPO the GPU sits idle while workers step environments, then all workers sit idle while the GPU runs a gradient update. SF decouples the two: CPUs are continuously stepping environments and the GPU is continuously running inference and training. Neither idles waiting for the other.

**80 truly parallel environments without GIL contention.** Each of the 40 rollout workers is a separate process with CPU core affinity pinned, giving 80 concurrent environment instances (40 workers × 2 envs per worker). SB3 vectorised environments share a single Python process and therefore a single GIL. MuJoCo physics stepping and depth rendering are CPU-bound, so eliminating that contention directly increases throughput.

**Policy lag tolerance.** Workers can run a small number of updates behind the latest policy weights (`max_policy_lag = 1000`). There is no hard synchronisation barrier, so a worker handling a geometrically complex colon seed that takes longer to step does not stall the rest of the pipeline.

**Native GRU support with correct BPTT.** SF treats the rollout window as the backpropagation-through-time window (`rollout = recurrence = 64`). This is important here because the task is partially observable — the GRU must integrate depth frames across haustral ring approaches to build a sense of lumen direction. SB3's recurrent policy support is less tightly integrated and harder to configure correctly for this use case.

---

## Final reward function

```
r = W_PROGRESS × new_territory × (0.3 + 0.7 × (1 − force_norm))
  − W_TIP_PENALTY  [if tip disk in wall contact]
  − W_STUCK_PENALTY  [terminal, if episode ends with no progress]
```

| Constant | Value | Role |
|----------|-------|------|
| `W_PROGRESS` | 2.0 | Scales new-territory advance reward |
| `W_TIP_PENALTY` | 0.001 / step | Dense penalty while tip disk presses wall (UFO posture) |
| `W_STUCK_PENALTY` | 0.05 | Terminal penalty for no-progress truncations |
| `MAX_INSERTION_FORCE` | 30 N | Calibrated contact-force ceiling (see below) |

### Key terms

**`new_territory`** — `max(0, actual_base_s − max_actual_base_s)`. Only the first time the scope reaches a given depth earns reward. Re-advancing over already-covered ground earns nothing. This prevents oscillation farming (advance → retreat → advance) which naive `Δs` rewards allow.

**`force_norm`** — `clip(Σ ‖contactForce_i‖ / 30 N, 0, 1)`. The 30 N ceiling was calibrated empirically: a straight gentle advance peaks at ~0.17 N (`force_norm ≈ 0.006`); the curl cheat peaks at ~215 N (`force_norm >> 1`, instant termination). Values between 0 and 1 scale the progress reward continuously — this matters (see iteration 4 below).

**Force multiplier `(0.3 + 0.7 × (1 − f))`** — a soft ramp, not a hard gate. A clean advance earns ~99% credit; a high-contact-but-not-terminal advance earns ≥30% credit (gradient floor). The floor is load-bearing: without it the policy loses gradient signal entirely at wall contact and cannot learn to steer away.

**`tip_contact` obs bit** — a 0/1 flag appended to the state vector indicating whether the tip disk is in contact with the colon wall. This was added after training revealed a sideways-press exploit (see iteration 6) and gives the policy a direct signal to act on.

**Episode termination** — `force_norm ≥ 1.0` (extreme curl) or `actual_base_s < base_s_init` (ejected). **Truncation** — no new territory in `STUCK_WINDOW` steps, or tip in continuous contact for `TIP_STUCK_WINDOW = 50` steps. Truncations use bootstrap value (not zero-value terminal); treating them as terminations collapses training by penalising reaching any obstacle.

---

## Iteration history

Reward choices and observation design are coupled — a reward signal is only useful if the policy can observe the quantities needed to act on it. The table below documents both dimensions for each major revision.

**Obs vector timeline (final 10D):**
`[cmd_x_n, cmd_y_n, progress_placeholder, force_norm, net_fx, net_fz, last_action[0..2], tip_contact]`

Early versions included `hinge_angles[25]` (joint angles, ~67% of the vector) and an additional `prev_action` lag buffer. These were removed in the v17 restructure; see iteration 6.

| Revision | Reward change | Obs change | Outcome |
|----------|--------------|-----------|---------|
| PPO v1–v2 | `W_COLLISION = 0.0002–0.0001` per contact step | None — early obs: depth + cmd + hinge_angles[25] + progress | Any motion was net-negative; policy collapsed to do-nothing within 50k steps. Fix: reduced to `0.00001`; added sparse distance-milestone rewards to provide gradient before the goal was reachable |
| PPO v5 / APPO v7b | `ent_coef = 0.03` | None | Entropy term (~0.064/step) was 100× task reward (~0.001/step); policy maximised entropy and ignored navigation. Fix: `ent_coef = 0.003` |
| PPO v6 — curl cheat | Added force termination: `Σ contactForce / 30 N ≥ 1.0` ends episode | Added `force_norm` to state obs so policy can perceive its own insertion force | Policy exploited 25 hinges × 0.55 rad (~770° total bend) by curling into a tight ball and pushing through 0.5 mm prism walls. Force termination closed the exploit; `force_norm` in obs gives the policy the signal it needs to avoid approaching the limit |
| APPO v11 | Added `−W_FORCE × force_norm` per-step penalty (`W_FORCE = 0.01`) | None — `force_norm` already in obs | Per-step accumulation reached ~0.01/step vs task reward of ~0.001/step; policy refused to move entirely; value function collapsed. Removed per-step force penalty — force is now encoded only in the progress multiplier |
| APPO v14–v17 | `r_progress × (1 − force_norm)` hard gate → soft ramp `(0.3 + 0.7 × (1 − f))` | None | Hard gate: zero gradient at `force_norm = 1`; chicken-and-egg with no reward to learn steering and no steering to earn reward. Policy stalled at haustral ring approach. Soft ramp's 0.3 floor keeps gradient alive even at maximum wall contact |
| APPO v17 — UFO posture | Added `W_TIP_PENALTY = 0.001/step` while tip disk contacts wall; added `TIP_STUCK_WINDOW = 50` truncation | Removed `hinge_angles[25]` (geometry already encoded in depth; removing freed 71% of state vector) and `prev_action` lag buffer (redundant with GRU hidden state). Added `tip_contact` bit (0/1). State: ~34D → 10D | Policy pressed tip disk sideways at 90° into a haustral ring ("UFO posture") — tip held firm, contact force moderate, kinematic base still advanced and earned progress. `tip_contact` obs bit gives the policy a direct signal to detect and escape the posture; `TIP_STUCK_WINDOW` caps how long it can persist |
| v5 — PD controller | Removed `W_GOAL = 1.0` terminal bonus (set to 0.0). High-water-mark progress already makes completing the colon the optimal strategy | Action semantics changed: actions become PD setpoints (`Kp=0.06, Kd=0.30`) rather than raw tendon forces. Added `net_fx`, `net_fz` (tendon force differentials — directly observable from motor currents on real hardware) | PD + heavy damping eliminated jitter from direct force control. Terminal bonus had incentivised rushing past haustral rings; removing it let the policy slow and steer correctly. **Result: 44% completion rate, 89.7% mean progress (80 seeds)** |
| v6 — domain randomisation | Per-episode actuator DR: gain [0.45, 1.0], dead-zone [0, 0.03], lag [0, 2] steps, advance gain [0.65, 1.0] | (1) Depth normalised by max value in frame, not fixed 0.5 m clip — prevents near-wall images going black when scope is pressed against a haustral ring. (2) DA3 multiplicative noise injected: `depth += N(0, error_map) × depth` using `error_map_64.npy`. (3) Progress slot zeroed (`OBS_PROGRESS_PLACEHOLDER = 0.0`) — true colon progress is privileged sim info not available at deployment | Completion 44% → **93.75%**. Max-frame normalisation was critical: near-wall images under fixed clipping were near-black (no gradient signal); max-normalised images still show lumen geometry. Zeroing progress removed a privileged obs that would not transfer to hardware |

---

See [domain_randomisation.md](domain_randomisation.md) for the full DR parameter table, physical motivation, and calibration methodology.

---

## Key ablation results

The full iteration narrative is in the table above. These two endpoints summarise the overall trajectory:

| Condition | Completion rate | Mean progress |
|-----------|----------------|---------------|
| v5 — PD controller, no domain randomisation | 44% | 89.7% |
| v6 — + actuator DR + DA3 depth noise injection | **93.75%** | **96.7%** |

The 2.1× gain in completion rate is attributable to domain randomisation (actuator gain, dead-zone, lag, advance gain) combined with per-frame depth normalisation and DA3 noise injection — together these prevented the two most common transfer failures: near-wall images going black under fixed depth clipping, and the policy over-fitting to a single set of actuator dynamics.

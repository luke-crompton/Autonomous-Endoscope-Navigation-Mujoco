# Progress Report — 14 Jul 2026

**Headline:** V8 Phase 1 passed. Mean progress **63.2% → 93.5%**; `tip_stuck` failures **73.3% → 12.3%**
without `stuck` rising to replace them (2.4%). The remaining failure was an *environment bug*
(a truncation window that made sharp bends unsolvable at any skill level), not a policy limit.

Awaiting the eval run to confirm — it also settles whether there is a real ~95% progress ceiling.

---

## 1. Actuation — two nested loops

**Outer (per RL step): PD command shaper.** Policy outputs a bend setpoint in [-1, 1] → scaled to the
measured physical pull limits (6.80 mm X / 7.36 mm Z, read off the joint stops, not guessed).
PD (Kp 0.06, Kd 0.30, over-damped) ramps the *command*. Cables driven antagonistically.

**Inner (per physics substep): P+I tendon force law.**
- Original actuator is **pure proportional** (Kp 6000) — a virtual spring, limited to −80…0 N
  (a cable pulls, never pushes).
- Problem: against the tip's joint springs, P-only settles **short of target** (zero error = zero
  force). Capped bend at **35–45% of nominal**.
- Fix: a **separate integral actuator per tendon** (Ki 36000, MuJoCo `dyntype="integrator"`), fed the
  live length error each substep, with **anti-windup** (freezes within 2 N of the 80 N cable limit).
  Forces sum on the shared tendon; the original actuator's stroke limit stays physically honest.
- Result: **~98–99% of commanded bend in 1–2 s**, stable, S-shaped folding drift gone.

**Third channel:** entrance feed rollers — friction grip that can *slip*, so progress is always
measured from the tip's true position, never the commanded roller angle.

## 2. What the obs vector needs in real life

8-D state + 64×64 depth image.

| Field | Hardware requirement |
|---|---|
| `cmd_x`, `cmd_y` | Nothing — controller's own setpoint |
| `ten_x`, `ten_z` | Motor/encoder position per cable. Wire is near-inextensible → proximal reel-in **is** the tendon length, not an approximation |
| `last_action[0..2]` | Nothing — policy's own last output |
| `tip_contact` | Binary "tip touching wall". **The one sensor not yet built** |
| `depth` | Monocular depth estimator on the tip camera — the fine-tuned DA3 model (**1.60 mm** MAE vs 8.30 mm zero-shot) |

**Design rule:** only signals sim can model with *genuine fidelity* — kinematic / encoder / contact.
Force signals were **removed** in Phase 1: in sim they're a clean readout, on hardware they'd be motor
current dominated by unmodelled cable-sheath friction. Either the policy trusts a fantasy signal, or
it learns to ignore it. (They still drive reward and diagnostics — just not what the policy sees.)
True colon progress is privileged sim info and is likewise excluded.

## 3. How sim is matched to reality

1. **Obs parity** — nothing in the policy's input is unavailable at deployment.
2. **Measured depth noise** — the *actual per-pixel error map* of the deployed depth model is injected,
   scaling with distance. Not white noise.
3. **Domain randomisation** each episode — cable efficiency 0.45–1.0, dead zone 0–3%, command lag 0–2
   steps. Applied to actuator *force*, not the target (endpoint stays reachable; effort varies) — and
   applied to the integral actuators too, so the policy can't bypass DR by leaning on them.
4. **Calibrated limits** — pull ranges measured against joint stops; 80 N cable limit; cannot push.
5. **Low-level loops deliberately independent** — sim runs P+I+anti-windup, the rig will run its own
   encoder PID. The policy never sees the actuator mechanism, so only *response characteristics* need
   to match, not implementations.

**Known gaps:** zero gravity; colon not anatomically oriented; tip contact sensor unbuilt; cable stretch
assumed negligible.

## 4. Phase 2 — in brief

**Gravity + an anatomical (left-lateral-decubitus) colon.** They must land *together*: today each seed's
bends use a random 3D axis, so there is no canonical "up" — gravity alone would be meaningless, and
reorienting the colon alone changes nothing. Goal: a policy that actively steers **against** gravity.

Two hard blockers:
- The colon currently runs straight *up* the gravity axis — naive gravity would drag the scope back out
  on almost every episode.
- Most of the shaft hangs outside the colon at reset with no floor in the scene → needs per-body gravity
  compensation (unproven here; has an explicit verification gate before anything is built on it).

Obs/action shapes are **unchanged** (no network surgery), but the physics change enough to need a
**from-scratch retrain**. The real cost is expected to be re-tuning the shaft/roller constants — every
one of them was tuned at zero-g with no shaft weight on the rollers.

**Open question for the meeting:** "if it looks good Phase 2 can start" sets a *gate* — but who owns
writing the Phase 2 plan?

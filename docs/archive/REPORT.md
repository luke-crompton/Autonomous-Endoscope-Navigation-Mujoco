---
title: "Simulated Videoscope Navigation in a Procedural Colon — Progress Report"
date: "June 2026"
---

> ## ⚠️ HISTORICAL SNAPSHOT — June 2026, DO NOT FOLLOW ITS PATHS
> This supervisor progress report reflects the project **as of June 2026** (the
> `Stage4/v2_collision` era) and the **old Mujuco_V2 layout**. Its "current work" and
> "next steps" sections are long superseded — v5, v6_dr, and v7_shaft all came after it.
> - Current results and full history: `../ITERATION_HISTORY.md`.
> - Current layout: root `README.md`.
>
> Kept verbatim as a dated record. Its authorship/attribution notes remain accurate and
> binding. Do not edit.

# Simulated Videoscope Navigation in a Procedural Colon

**Progress report — June 2026**

## Executive summary

The project builds a complete simulation testbed for designing and evaluating an
automatic videoscope (colonoscope-style) controller, entirely in MuJoCo. My supervisor
provided two simulation assets to build on: a physically simulated 25-hinge
tendon-driven scope tip, and a photorealistic Blender colon-rendering environment.
Using these, I developed and validated: (1) my own MuJoCo procedural colon model with
camera and depth rendering; (2) a perception stack that estimates a steering target and
a full metric depth map from a single RGB frame; (3) a closed-loop controller that
drives the supervisor's scope through my colon; and (4) a collision-enabled
reinforcement-learning (RL) pipeline for the steering controller. A vision-only
proportional controller traverses the full colon in the original (collision-free)
scene. I then added physical
scope–wall collisions and moved to reinforcement learning (RL) for steering. The RL
agent reached high numerical success but exploited two physics loopholes ("cheats"),
which I diagnosed and which now drive a cleaner go-forward design: a pure-outcome
reward with cheat-prevention enforced in the physics rather than the reward function.
The simulation and perception components are working; the steering controller under
realistic collisions is the current open problem.

## 1. Aim and approach

The goal is a controller-design-and-evaluation loop: given what an onboard camera
sees, decide how to bend the scope tip so it advances safely along the colon lumen.
Rather than work on hardware, the entire perception–control loop runs in simulation so
controllers can be iterated and stress-tested cheaply. The work is organised in stages:
build the environment, build perception, build the actuator, then close the loop.

## 2. Simulation environment (done)

**My procedural colon model.** I built a 500 mm anatomical centreline (ascending →
hepatic flexure → transverse → splenic flexure → descending) with a ~15 mm internal
radius, 35 haustral folds, soft three-lobed cross-section, and per-vertex normals for
smooth shading. The mesh, ground-truth centreline, and a MuJoCo scene are generated
from a single seeded script, so geometry is reproducible and can be re-randomised for
training variety. Illumination is an endoscope-style tip light only, matching real
conditions. (My supervisor separately provided a photorealistic Blender colon-rendering
environment, which I use later for the depth-estimation work in §3.)

**Rendering.** The scene renders both RGB and pixel-accurate depth from a tip-mounted
camera, with the light following the camera as on a real scope. This provides the
training data and the live observations for the controller.

## 3. Perception (done)

Two complementary perception approaches were developed.

**Steering CNN (target-point regressor).** I trained a ResNet-18 on my colon model to
predict a single "where to drive next" pixel from RGB. Several labelling strategies were
compared — deepest depth blob, deepest visible centreline point, and pure-pursuit
look-ahead at 30/15/10 mm. The **10 mm look-ahead** gave the gentlest, most stable
corner commands and is the current best; it drives the scope through the full colon
end-to-end with no tip-camera crash.

**Metric depth from RGB (Depth Anything 3, fine-tuned).** To obtain a full depth field
rather than a single point, I evaluated Depth-Anything-3-SMALL. Zero-shot it produced
only relative inverse depth (≈7 mm scale-aligned error) and, critically, failed at
flexures — confusing a brightly lit wall for the deepest region, which is the worst
possible failure for steering. Using my supervisor's photorealistic Blender colon
renderer, I generated a training set and ran a **head-only fine-tune** (training only
the 11% of parameters in the depth head, on 1469 renders), which fixed this:

| Metric | Zero-shot | Fine-tuned |
|---|---|---|
| Held-out geometry, scale-aligned MAE | 9.49 mm | **3.28 mm** |
| Unseen pilot set, scale-aligned MAE | 7.10 mm | **2.40 mm** |
| Output type (GT/pred ratio) | relative (0.029) | **metric (1.014)** |
| Flexure failure mode | present | **fixed** |
| Inference speed | ~20 FPS | ~20 FPS |

The fine-tuned model is metric, real-time, and resolves the flexure case visually.

## 4. The scope and closing the loop

**Scope tip (supervisor-provided).** My supervisor provided a MuJoCo model of one
articulating scope section — 26 disks joined by 25 alternating-axis hinge joints, driven
by 4 antagonistic tendons, so the tip can bend in any direction. My contribution is
integrating this scope into the perception–control loop, not modelling the scope itself.

**Closed loop, collision-free (working).** Combining perception, the supervisor's scope,
and my colon, a proportional controller maps the predicted target pixel to antagonistic
tendon commands
while advancing the base along the centreline. With the 10 mm steering CNN this
traverses the full colon. This confirmed the loop works but with an important caveat:
there were no scope–wall collisions, so the controller never had to cope with physical
constraint.

**Vision-only depth + PD, collision-free (dropped).** Swapping in the fine-tuned depth
model with a depth-centroid aim point and a PD controller, the scope could *see* the
lumen reliably but could not traverse — it oscillated between good views and wall-jams.
Every controller-gain fix traded one failure for another. Conclusion: a bare target
pixel plus simple PD, with no collisions, is not sufficient on this curvature.

## 5. Adding physics: collisions and RL (current work)

The diagnosis above motivated adding **physical scope–wall collisions** so the colon
geometry constrains the scope shape, and learning the controller with RL. (The colon
geometry used here follows my supervisor's procedural colon distribution; the collision
modelling, RL environment, training, and analysis below are my own.)

**Collision modelling.** A hollow tube cannot use MuJoCo's default convex-hull mesh
collision (the hull is a solid blob the scope ignores). Instead each wall triangle is
turned into a thin extruded prism; adjacent prisms share edges so the wall behaves like
the smooth mesh while fully capturing the inner surface. This runs fast enough for RL
(~1500 physics steps/s).

**RL formulation.** A Gymnasium environment exposes a 3-D action (two tendon-pair
commands + a base-advance rate) and an observation of the 64×64 metric depth image plus
a small state vector. The scope base is attached to a kinematically driven target via a
1-DOF slide joint, so a jammed tip causes the chain to lag rather than being forced
through. Reward = forward progress (high-water-marked so oscillation cannot farm it) +
goal bonus − collision penalty + intermediate waypoint credit. Training uses PPO with a
CNN-on-depth policy.

**Results and the two "cheats" (the key finding).** This is the most instructive part
of the work:

| Run | Outcome | What happened |
|---|---|---|
| v1–v2 | ~0% | Collision penalty made any motion net-negative; agent froze |
| **v3** | 66% | **Cheat 1:** with a free-floating base it dragged itself backward and flipped orientation |
| v4–v5 | ~1% | Closing cheat 1 made the reward too sparse / entropy-dominated to learn |
| **v6** | 85% | **Cheat 2:** curled the over-flexible chain into a ball and punched *through* the thin walls |
| v7 (×3) | ~0% | A reward that explicitly rewarded "aim at the lumen" failed to learn and, by design, defeated the point of using RL |

Both high scores were illusory — the agent solved the *simulation*, not the *task*. The
v7 experiments confirmed that hand-engineering the aim into the reward is both
ineffective and self-defeating (it re-implements the analytic aim point we already had).

## 6. The reframe and next steps

The current plan (agreed, not yet implemented) follows directly from the cheat analysis:

1. **Strip the reward to pure outcome** — progress + goal − collision only. Let correct
   aiming *emerge* from the depth observation instead of being scripted into the reward.
2. **Prevent cheating in the physics, not the reward.** The slide-joint architecture
   already stalls a mis-aimed tip automatically; the only remaining loophole is wall
   *penetration* (cheat 2). The next concrete step is to verify, with a scripted
   max-curl test, whether the scope can still punch through walls — and if so, harden
   the walls or make the base advance force-limited (so a jammed chain physically cannot
   be shoved forward). Notably, this fixes the exploit *without* crippling the scope's
   articulation.
3. Replace the 1dof sliding base joint with an actual tube this should require the RL agent to aim the tip
    rather than trying to curl up in a ball and force its way through, also reverify collisions work as intended with this new model
   and find a proper way to penalise wall collisions.

The simulation and perception are in good shape; the open research question is a
steering controller that traverses reliably under realistic physical contact.

## Appendix — environment and key files

- **Simulation:** MuJoCo. My procedural colon model (seeded script). The articulating
  scope tip (`Videoscopesimulations/`) and the photorealistic Blender colon renderer
  (`colon_realistic_geometry_handoff_...`) were provided by my supervisor.
- **Perception:** ResNet-18 steering CNN (PyTorch); fine-tuned Depth-Anything-3-SMALL
  (head-only, metric, ~20 FPS on an RTX 4060 Laptop).
- **RL:** Gymnasium + Stable-Baselines3 PPO, multi-input (CNN depth + MLP state),
  per-triangle prism collisions, 1-DOF slide-joint base.
- **Hardware:** RTX 4060 Laptop (8 GB), 16 GB RAM; ~75 min per 1M-step PPO run.
- **Key directories:** `Perception/` (colon + CNNs + depth model),
  `Videoscopesimulations/` (scope, supervisor-provided), `Stage4/v1/` (my working
  collision-free baseline),
  `Stage4/v2_collision/` (collision RL pipeline, including all PPO run checkpoints
  retained for progress review).
- **Trained artifacts retained:** fine-tuned depth model
  (`da3small_finetune_head.pth`), best steering CNN (10 mm look-ahead), all PPO run
  checkpoints, closed-loop run videos.

*Note: large training datasets and inference outputs are regenerable from the included
scripts and may be excluded from the delivered archive to keep it compact.*

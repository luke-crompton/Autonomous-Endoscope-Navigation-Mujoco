# hardware/bringup/

Test procedures, measurement scripts, and recorded results for taking the rig from "assembled"
to "running the policy". **No results recorded yet as of 2026-08-06.**

[`../../docs/CURRENT_PLAN.md`](../../docs/CURRENT_PLAN.md) §7 is authoritative for sequencing.
This page is the practical checklist and the home for what comes back.

## Suggested layout

```
bringup/
├── README.md          this file — what to measure and why
├── <script>.py        measurement tooling
└── results/           dated results, one file per measurement
```

Record results **with their date and the rig state**, and keep them even when superseded — a
measurement that later turns out wrong is still the evidence for why something changed.

---

## Pre-flight, before any policy-driven run

### 1. Measure `MAX_PULL_X` and `MAX_PULL_Z` — not optional

These normalise `cmd_x_n` and `cmd_y_n`, **two of the six live policy inputs**. The current
values (6.795 / 7.361 mm) are derived from sim geometry, not from the real scope. A wrong value
does not scale the output — it hands the network an observation it was never trained on.

Pull each cable to its mechanical bend limit and record the encoder delta. **Keep X and Z
separate** — they genuinely differ, spanning 12 vs 13 joints.

### 2. Camera roll offset

In sim the tip camera is welded to the tip body and the tendon holes are in that same body
frame, so "where the lumen is in the image" and "which cable steers toward it" are locked
together — the scope can twist freely down a bend and the policy neither sees nor cares. Only a
single **fixed roll offset** between the real camera and the real cable axes can matter, and it
is unmeasured.

**The convention sim was trained under** (derived from the camera XML, *not* yet verified by the
headless `a[0] = +1` check):

| Image direction | Body axis | Cable pair | Span |
|---|---|---|---|
| right | −X | `px` / `nx` | 12 joints, 120°, `MAX_PULL_X` |
| up | −Z | `pz` / `nz` | 13 joints, 130°, `MAX_PULL_Z` |

So `a[0] > 0` steers toward image **left**, `a[1] > 0` toward image **down**. The wide image
axis (100° over 96 columns) is the X pair; the narrow axis (67.7° over 54 rows) is the Z pair.

**The measurement.** Scope in free space, pull each of the four cables in turn to a clearly
visible bend, and record which way the image content translates. Four pulls give the full
image↔cable map — offset angle and all signs — with no calibration target.

**Acting on the result:**

- **Offset is a multiple of 90°** → fix in software with a signed swap of `a[0]` / `a[1]`
  (relabelling the cables and rotating the command are the same operation here). **180° is
  free.** 90° / 270° costs the 12-vs-13-joint asymmetry — `a[0]` would then drive the 130° axis
  where it trained on 120°, ~8% in pull and ~10° in range.
- **Offset is not a multiple of 90°** → **it cannot be corrected in the action.** The obs is
  anisotropic (1.04°/column vs 1.25°/row), so a rolled real frame is not a rotation of any frame
  sim can render; the CNN is not rotation-equivariant; the bend map itself is anisotropic (a
  rotated full-scale command can exceed the elliptical reachable envelope); and the GRU state
  goes out of distribution. Fix the mount, or rotate the frame **before** the 54×96 downscale —
  which folds into the fisheye rectification LUT (§3), so it is one remap and no extra cost.

⚠️ **Trap, either route:** five of the six state slots are the action echo
(`last_action[0..2]` plus the two PD-filter slots derived from it). They must be expressed in
the **same frame as the image**. Rotating the outgoing command while feeding back the unrotated
one — or vice versa — hands the policy an image/echo pair that contradicts itself.

### 3. Fisheye rectification

The largest known observation mismatch, and perception-side only. The real lens is **122°
horizontal fisheye**; the policy was trained on **100° horizontal rectilinear**.

Run `probe_camera.py --measure_fov checkerboard` for `K`/`D`, then build a
`cv2.fisheye.initUndistortRectifyMap` LUT to a 100° rectilinear virtual camera and apply it to
the live frame before DA3. Cost is one `cv2.remap` per frame — do it on the **downscaled** frame,
not 1280×720, since the depth loop is already at 31 fps against a 25 Hz target.

### 4. Optional but informative

A **deterministic eval** and the **steering check** both change how a failed trial should be
read. The steering check matters most: deterministic mean `|a_xy|` is only **0.034**, about 3.4%
of full bend, and it is not yet established whether the policy is steering efficiently with
small precise corrections or barely steering at all and letting shaft compliance and colon
geometry do the work. The second would transfer badly. The test is to log deterministic `|a_xy|`
across full traverses and check whether it **rises at bends**.

---

## Known open problem: the feeder stalls at slow insertion

Found 2026-08-04 in **simulation**, still unexplained, and it constrains what a rig trial can
tell you.

With `--feed_scale 0.1` (3 mm/s, real-colonoscopy pace) and a fixed open-loop `[0,0,1]` action —
no policy involved — the scope advances **9.45 mm and then stops permanently**, while the
commanded insertion climbs to 84 mm. Roller actuator force stays at ~0.02–0.09 N and does not
grow with the 74 mm position error. Same seed and action at `--feed_scale 1.0`: 96 mm in 109
steps.

So the friction-grip feeder transports at 30 mm/s and fails to transport at 3 mm/s.

**Consequence:** the sim cannot currently represent a hand-fed insertion, and the policy has only
ever experienced 30 mm/s — 1.2 mm of visual progress per decision, where hand-feeding gives
0.12 mm. Whether the GRU depends on that rate is unknown and untestable until this is diagnosed.
Since insertion is hand-fed by design, this is a first-class gap, not a curiosity.

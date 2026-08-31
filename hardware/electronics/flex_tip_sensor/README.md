# hardware/electronics/flex_tip_sensor/

The flexible printed circuit that supplies the policy's one non-visual input: **`tip_contact`**,
a binary "the tip is touching the wall" flag. It is the only exogenous scalar in the 6-D state
vector — the other five are software state reproduced on the policy side, not sensors (see
[`../../README.md`](../../README.md) and [`../../../docs/CURRENT_PLAN.md`](../../../docs/CURRENT_PLAN.md) §2).

**Status: work in progress.** The board has been fabricated. It is **not yet calibrated,
characterised, or wired to the rig** — no resistance threshold set, no bench data, no integration.
Treat the geometry and construction below as settled and everything about its electrical
behaviour as open. `docs/PROGRESS_REPORT_2026-07-14.md` still lists this sensor as unbuilt; that
is the only thing now out of date there.

<img src="tip_build_photo.jpg" alt="the flex sensor alongside the articulating tip">

*The orange flex sensor — comb head at the left end, pigtail and green hook-up wires trailing off
— next to a bare articulating tip stack (top) and a tip section with its tendon cables and camera
wiring in place (right). This is the build stage before the sensor is bonded and the spring and
braided housing go on.*

<img src="kicad_layout.png" alt="KiCad layout of the flex sensor" width="170">

*KiCad layout, to scale: the comb head (top) and the 1.5 mm pigtail running down to the two
solder pads.*

## What it is

A single-layer polyimide flex circuit in two parts:

| Part | Size | Function |
|---|---|---|
| **Head** | ~31 × 4.5 mm | An interdigitated copper comb — two interleaved electrodes, ~0.6 mm pitch, ~29 mm wide — ENIG-finished and left bare. Wraps **once around the 9 mm-OD tip disk**, the disk that also carries the camera (`disk_25` in the sim tip model). |
| **Pigtail** | 1.5 mm × ~84 mm | Polyimide ribbon carrying the two electrode traces back through the endoscope working channel to two exposed wire-solder pads (`TP1`, `TP2`) with a local stiffener. Kept as slim as possible — no stiffener along its length. |

**How it senses contact:** a piezoresistive film is bonded over the comb. Pressing the film
across the interleaved fingers lowers the resistance between the two electrodes; a threshold on
that resistance becomes the binary flag. **The copper forms no connection on its own** — which is
why there is no meaningful schematic, and why DRC "unconnected" warnings on the comb fingers are
expected (and excluded in the design file). `flex tip sensor.kicad_sch` is a stub holding only
the two solder-pad test points.

## Construction

- Polyimide substrate, 18 µm (½ oz) copper, polyimide coverlay both faces — **except** the comb
  (bare ENIG) and the solder pads (bare).
- Local flexible polyimide stiffener bonded under the solder-pad area only, to protect the wire
  joint from handling stress.
- ~0.1 mm total thickness. Minimum bend radius in service **4.5 mm, static** — the head is
  wrapped once around the tip disk, not repeatedly flexed.
- The KiCad file carries two copper layers only because KiCad requires an even count; `B.Cu` is
  empty and excluded from the Gerbers. **The manufactured part is single-layer.**
- Full fabrication notes are on the board's `Cmts.User` text layer — open it in KiCad, or read
  `Gerbers/flex tip sensor-User_Comments.gbr`.

## Files

| | |
|---|---|
| `flex tip sensor.kicad_pcb` | the layout — open this |
| `flex tip sensor.kicad_pro` | KiCad 10 project |
| `flex tip sensor.kicad_sch` | stub (two test points); kept only so the project opens cleanly |
| `Gerbers/` | fabrication output — `F_Cu`, `F_Mask`, `Edge_Cuts`, `User_Comments`, job file |

Designed in KiCad 10.0.4, 2026-07-21.

## Sim-to-real note

The simulator resolves `tip_contact` as *any* contact between the camera disk and the colon wall
— the cylindrical side face and both flat end faces. **This sensor covers the side face only**,
so a head-on jam is invisible to it. That gap is reviewed and accepted, with reasons, in
[`../../../docs/CURRENT_PLAN.md`](../../../docs/CURRENT_PLAN.md) §2. Setting the film threshold and
checking coverage against that assumption is a [`../../bringup/`](../../bringup/) task once the
sensor is live.

# Videoscope Simulations

One-section MuJoCo videoscope model with manual tendon pulling.

## Model

- 25 hinge joints
- 26 disks
- 9.0 mm outer diameter
- 4 tendon holes at 0, 90, 180, and 270 degrees
- Tendon holes are 3.2 mm from the central origin
- Ball/socket rotation diameter is 2.0 mm
- Adjacent disk centers are 2.5 mm apart
- The distance from each joint center to the top of its disk is 2.5 mm
- `new8.stl` is rotated by 90 degrees on alternating disks about the long axis
- The next joint is placed at the top of the previous disk to keep the joint
  interface visually closed during bending
- No joystick and no automated trajectory

## Files

- `generate_videoscope_one_section.py` builds the XML.
- `videoscope_one_section.xml` is the MuJoCo model.
- `manual_tendon_viewer.py` launches the viewer and leaves tendon sliders manual.
- `new8.stl` is overlaid as the visual mesh for each joint disk.

## Requirements

- Python 3.10 or newer
- MuJoCo Python package

Install MuJoCo with:

```bash
python -m pip install mujoco
```

On some systems the command may be `python3` instead of `python`.

## Run The Simulation

Open a terminal in this folder, then run:

```bash
python generate_videoscope_one_section.py
python manual_tendon_viewer.py
```

If your system uses `python3`, run:

```bash
python3 generate_videoscope_one_section.py
python3 manual_tendon_viewer.py
```

The first command regenerates `videoscope_one_section.xml`. The second command
opens the MuJoCo viewer.

## Manual Tendon Controls

Use the MuJoCo viewer Controls panel. The viewer script enforces antagonistic
tendon pairs:

- X pair: `pull_px` and `pull_nx`
- Z pair: `pull_pz` and `pull_nz`

Pulling one side releases the opposite side by the same amount. Slider units
are metres. For example, `0.003` is 3 mm of tendon pull and `-0.003` is 3 mm
of tendon release. The maximum pair command is `+/-0.024`, which is 24 mm.

## Notes

- There is no joystick control.
- There is no automated movement sequence.
- The visual geometry uses `new8.stl`; keep that STL in the same folder as the
  XML and Python scripts.
- The hidden simple cylinder geometry is used only to keep the simulation
  stable; the STL remains the visible disk shape.

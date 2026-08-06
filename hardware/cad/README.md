# hardware/cad/

The physical rig: tendon drive, spool, shaft clamp and cantilever force mount. Roughly
180 × 130 × 110 mm.

**[▶ View the assembly in 3D](Full%20assem.STL)** — GitHub renders it interactively; rotate and
zoom in the browser, no software needed.

> 🖼️ **TODO — a couple of renders go here** (the assembled rig, and a close-up of the drive end).
> `Ctrl+Shift+S` in SolidWorks saves the viewport as a PNG; drop them in this folder and link
> them above.

## Bought-in components

Two parts in this folder are **off-the-shelf, not my design.** They are included because the
surrounding parts were designed around them — their mounting patterns, envelopes and load paths
drive the geometry of everything they touch:

| Part | What it is | What was designed around it |
|---|---|---|
| `sts3032.SLDPRT` | STS3032 serial bus servo | `sts3032 Bracket male` / `Bracket Female` — the mounting pair that carries it |
| `Load cell block.SLDPRT` | cantilever force bar (load cell) | `Cantilever bracket` — the mount that fixes it and sets the measuring axis |

Everything else here is my own: the spool, pulley, bearing wheel, shaft clamp and clamp top.

## Files

| File | |
|---|---|
| `Full assem.SLDASM` | the complete assembly |
| `Layout Assembely.SLDASM` | layout assembly, shares parts with the above |
| `Full assem.STL` | merged mesh of the whole rig — for viewing, not for printing |
| 10 × `.SLDPRT` | the individual parts |

Flat folder by design: SolidWorks resolves references between files in the same directory with
no path rewriting, so the assembly opens correctly straight from here.

Print-ready geometry is not kept here — open the assembly and export what you need.

---

*Working copy lives outside the repo; see [DEV_SETUP.md](../../docs/DEV_SETUP.md) for how this
folder is refreshed after a design change.*

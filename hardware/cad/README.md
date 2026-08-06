# hardware/cad/

CAD for the physical rig. **Empty as of 2026-08-06** — the working model lives at
`C:\Users\lukec\Documents\Solidworks Projects\Endoscope` (36 files, 13.1 MB: 16 parts,
2 assemblies, plus STEP/STL/3MF exports).

`Full assem.SLDASM` is the current, complete assembly and is the thing to port. The source
folder also holds **superseded iterations** that should not come across — three spool variants
(`Spool`, `Spool_testing`, `Spool helic reverse`), three `sts3032` bracket variants, and an
`STL/` folder that is print history (`SpoolV1 → V2 → V3`, `Pulley_print1`,
`maleV2 cable holes`, dated 2026-07-24 → 2026-08-05). Don't sort these by hand — Pack and Go
resolves which parts are live, and STLs should be re-exported fresh rather than copied.

## Layout

```
cad/
├── native/     SolidWorks sources — .SLDPRT, .SLDASM
├── step/       STEP exports — the archival, CAD-neutral format
├── stl/        STL exports — GitHub renders these in an interactive 3D viewer
└── print/      3MF / print-ready files
```

Keep **STEP and STL committed even when the native files are**. They are the copies that stay
readable without a SolidWorks licence, and STL is what makes the geometry visible directly in
the browser on GitHub.

## Moving the SolidWorks files without breaking the assemblies

The concern is real but narrower than it looks. SolidWorks resolves external references by
**relative path first**, so an assembly and all of its parts moved *together as one folder* will
re-resolve normally. What actually breaks references is:

- moving or renaming files **individually**, or copying only some of them
- references to files **outside** the folder — Toolbox/library hardware in particular, which
  lives in a central location and is easy to forget about
- a copy made while the assembly is open in SolidWorks

**Use `File → Pack and Go` rather than Explorer copy-paste.** It walks the assembly's full
reference tree, collects every referenced file including ones from outside the folder, writes
them to a destination you choose, and rewrites the references to match. That is the method with
a guarantee attached; a folder copy merely usually works.

### Procedure

1. Open **`Full assem.SLDASM`** in SolidWorks, from its current location.
2. `File → Pack and Go`. Tick **"Include suppressed components"** if any exist; leave drawings
   and simulation results off unless you want them.
3. **Read the file list it shows you.** This is the authoritative answer to which parts are
   live — anything absent from it is a superseded iteration. Worth a screenshot before you
   proceed.
4. Choose **"Save to folder"** → `…\Mujuco_V3\hardware\cad\native`. Do **not** use "Save to
   Zip file" unless you want it archived rather than versioned.
5. **The original is untouched** — Pack and Go copies. Leave
   `Documents\Solidworks Projects\Endoscope` exactly as it is.
6. Close everything, then open `hardware\cad\native\Full assem.SLDASM` **from the new
   location**. Check the FeatureManager tree for unresolved-component flags, and run
   `File → Find References` to confirm nothing still resolves back to `Documents\`.
7. With the copy open, re-export fresh: **STEP → `../step/`**, **STL → `../stl/`**. Do not copy
   the old STLs across; they are earlier iterations and would misrepresent the current design.

`Layout Assembely.SLDASM` is a second, separate assembly. If it is still current, Pack and Go it
separately in the same way. If it is an early layout sketch, leave it behind.

## Which copy is the working copy?

Worth deciding deliberately, because two live copies drift:

- **Keep working in `Documents\`, export here** — no risk to the working model, no SolidWorks
  lock files or backups in git, and the repo holds a published snapshot. Cost: you must remember
  to re-export after design changes.
- **Work directly in this folder** — one source of truth, every change versioned. Cost: git
  churn on binaries (each save stores a full copy, though at 13 MB total that is comfortable),
  and this folder must then never be moved again casually.

The first is the safer default while the design is still changing. Switch to the second once it
settles.

## Size limits

GitHub rejects any single file over **100 MB** and warns above 50 MB. Nothing in the current
model comes close — the largest file is 0.73 MB. If a future export does, keep it out of git and
attach it to a Release instead.

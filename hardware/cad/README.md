# hardware/cad/

CAD for the physical rig. **Empty as of 2026-08-06** — the working model lives at
`C:\Users\lukec\Documents\Solidworks Projects\Endoscope` (36 files, 13.1 MB: 16 parts,
2 assemblies, plus STEP/STL/3MF exports).

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

Then, in order:

1. **Copy, do not move.** Leave the original in `Documents\Solidworks Projects\Endoscope`
   untouched.
2. Open **both** assemblies from the new location — `Full assem.SLDASM` and
   `Layout Assembely.SLDASM`.
3. Check for dangling references: the FeatureManager tree flags unresolved components, and
   `File → Find References` lists what each assembly actually points at. Confirm nothing still
   resolves back to the old folder.
4. Only once both open clean, decide whether the original is redundant.

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

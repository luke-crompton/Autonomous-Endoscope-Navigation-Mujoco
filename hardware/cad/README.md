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

## Exporting STL, and the GitHub 3D viewer

GitHub renders committed `.stl` files in an **interactive 3D viewer** — rotate, zoom, wireframe,
and a revision-comparison mode. Nothing needs to be configured or installed: commit the file and
click it in the repo file browser. Two limits:

- **It does not render inline in a README.** The viewer only appears on the file's own page. A
  README needs a rendered PNG for anything that shows on the page itself.
- **Very large STLs may not render.** Export at a sensible resolution rather than maximum.

From SolidWorks: `File → Save As → STL (*.stl) → Options`. The decision that matters is
**"Save all components of an assembly in a single file"**:

| Setting | Produces | Use for |
|---|---|---|
| ticked | one STL of the whole assembly | the GitHub viewer — one file, one click, the assembled rig |
| unticked | one STL per part, each in its own frame | printing |

Do both. Resolution **Fine** is normally enough; drop to a coarser custom deviation if the
combined file is heavy.

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
3. Click **`Add File...`** and add **`Layout Assembely.SLDASM`**, so both top-level assemblies
   go into one package. See "Two assemblies, shared parts" below for why this must be one
   operation and not two.
4. Tick **"Flatten to single folder"** — the source has `Prints/`, `Step/` and `STL/`
   subfolders, and there is no reason to recreate that tree inside `native/`.
5. **Read the file list it shows you.** This is the authoritative answer to which parts are
   live — anything absent from it is a superseded iteration. Worth a screenshot before you
   proceed.
6. Choose **"Save to folder"** → `…\Mujuco_V3\hardware\cad\native`. Do **not** use "Save to
   Zip file" unless you want it archived rather than versioned.
5. **The original is untouched** — Pack and Go copies. Leave
   `Documents\Solidworks Projects\Endoscope` exactly as it is.
6. Close everything, then open `hardware\cad\native\Full assem.SLDASM` **from the new
   location**. Check the FeatureManager tree for unresolved-component flags, and run
   `File → Find References` to confirm nothing still resolves back to `Documents\`.
7. With the copy open, re-export fresh: **STEP → `../step/`**, **STL → `../stl/`**. Do not copy
   the old STLs across; they are earlier iterations and would misrepresent the current design.

### Two assemblies, shared parts

`Layout Assembely.SLDASM` and `Full assem.SLDASM` share parts, and it was not certain whether
the former is a component of the latter or an independent top-level assembly. **The procedure
above is correct either way** — if it is already a sub-assembly, adding it via `Add File...`
changes nothing (Pack and Go deduplicates); if it is independent, it gets captured.

To settle the question in five seconds: open `Full assem.SLDASM` → `File → Find References`. If
`Layout Assembely.SLDASM` is in that list, it is a referenced component.

⚠️ **Do not run Pack and Go twice into the same folder.** The second run collides with files the
first already wrote, leaving you making overwrite decisions file by file — and one wrong call
leaves the two assemblies pointing at *different copies of the same part*, which is exactly the
failure this whole procedure exists to avoid. One operation with both files packages the
**union** of their references once, with every path rewritten consistently.

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

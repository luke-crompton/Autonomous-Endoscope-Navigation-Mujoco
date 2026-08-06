# hardware/cad/

CAD for the physical rig — SolidWorks sources plus a viewable STL of the assembled machine, all
in this one folder. Flat by design: no subfolders to keep in sync, and SolidWorks resolves
references between files in the same directory without any path rewriting.

**[`Full assem.STL`](Full%20assem.STL) is the whole rig — click it on GitHub for an interactive
3D view** (rotate, zoom, wireframe). No plugin or setup needed; GitHub renders `.stl` natively.

| File | What it is |
|---|---|
| `Full assem.SLDASM` | the current, complete assembly |
| `Layout Assembely.SLDASM` | second top-level assembly, shares parts with the above |
| `Full assem.STL` | merged mesh of the whole assembly, for viewing — not for printing or re-import |
| 10 × `.SLDPRT` | every part the two assemblies actually reference |

Anyone wanting print-ready geometry can open the assembly and export it themselves; per-part
STLs are not kept here.

---

## Provenance

Ported 2026-08-06 from `C:\Users\lukec\Documents\Solidworks Projects\Endoscope` via SolidWorks
**Pack and Go**, which resolves the reference tree rather than requiring the live parts to be
identified by hand. **The original folder is untouched** and remains the working copy.

Pack and Go carried 10 of 16 source parts. The 6 it left behind are superseded iterations, and
that list is a useful record of the design history:

| Left behind | Why |
|---|---|
| `mg996r`, `Servo holding block` | the design moved off MG996R hobby servos to STS3032 serial servos |
| `Spool helic reverse`, `Spool_testing` | only `Spool.SLDPRT` is live |
| `sts3032 Bracket` | superseded by the `male` / `Female` pair |
| `Shaft ring` | no longer used |

The source folder also holds an `STL/` folder of print history (`SpoolV1 → V2 → V3`,
`Pulley_print1`, `maleV2 cable holes`, 2026-07-24 → 2026-08-05). It was deliberately not copied
— those are earlier iterations and would misrepresent the current design.

---

## Re-porting after design changes

The working copy is still `Documents\Solidworks Projects\Endoscope`, so this folder goes stale
when the design moves. To refresh it:

1. Open **`Full assem.SLDASM`** in the working folder.
2. `File → Pack and Go`. Tick **"Include suppressed components"** if any exist.
3. Click **`Add File...`** and add **`Layout Assembely.SLDASM`** — see below for why this must
   be one operation and not two.
4. Tick **"Flatten to single folder"**.
5. **Read the file list it shows you.** It is the authoritative answer to which parts are live.
6. **Save to folder** → this directory. Not "Save to Zip file".
7. Close everything, reopen from **this** folder, and check the FeatureManager tree for
   unresolved-component flags. `File → Find References` confirms nothing still resolves back to
   `Documents\`.
8. Re-export `Full assem.STL` (below). A stale STL next to fresh sources is worse than none.

### Two assemblies, shared parts

`Layout Assembely.SLDASM` and `Full assem.SLDASM` share parts. Whether the former is a component
of the latter or an independent top-level assembly was never established — **and the procedure
above is correct either way**: if it is already a sub-assembly, adding it via `Add File...`
changes nothing (Pack and Go deduplicates); if it is independent, it gets captured.

⚠️ **Do not run Pack and Go twice into the same folder.** The second run collides with files the
first wrote, leaving you making overwrite decisions file by file — and one wrong call leaves the
two assemblies pointing at *different copies of the same part*, exactly the failure this
procedure exists to prevent. One operation with both files packages the **union** of their
references once, with every path rewritten consistently.

---

## Exporting the STL

`File → Save As → STL (*.stl) → Options`, with **"Save all components of an assembly in a single
file" ticked** — that gives one mesh of the whole rig in its assembled positions, which is what
the GitHub viewer wants. Resolution **Fine** is normally enough; drop to a coarser custom
deviation if the file gets heavy.

Two limits on the viewer:

- **It does not render inline in a README.** The viewer only appears on the file's own page. To
  show the rig *on* a page you need a rendered PNG, optionally linked through to the STL.
- **Very large STLs may fail to render.** The current export is 6.6 MB and works.

---

## Size limits

GitHub rejects any single file over **100 MB** and warns above 50 MB. The largest file here is
the 6.6 MB STL. If a future export exceeds the limit, keep it out of git and attach it to a
Release instead.

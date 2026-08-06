# Documentation index

Start at the [top-level README](../README.md). This page is the map of everything below it.

Documents here fall into three kinds, and the difference matters:

- 📘 **Reference** — describes how the system works. Written to be read.
- 🧾 **Record** — a dated snapshot or an archive. True *on its date*, never updated afterwards.
- 🔧 **Working note** — written for whoever is building this, not for a reader.

---

## Read these

| | Document | What it gives you |
|---|---|---|
| 📘 | [CONTROL_LOOP_REPORT_2026-08-05.md](CONTROL_LOOP_REPORT_2026-08-05.md) | **The best single technical description of the system.** The full 25 Hz loop — observation → network → action → cables → physics — with every constant traced to a file and line. Self-contained: written to be readable with no repo access. |
| 📘 | [ITERATION_HISTORY.md](ITERATION_HISTORY.md) | How the project got from v1 to v8_p1: what each version changed, what it fixed, and what it broke. The dead ends are included on purpose — they are most of the story. |
| 📘 | [DEV_SETUP.md](DEV_SETUP.md) | How to actually run things. Interpreter matrix, the three machines, multi-machine sync rules, and the commands that work (plus the ones that silently don't). |
| 🔧 | [CURRENT_PLAN.md](CURRENT_PLAN.md) | **The single source of truth for project state.** Where things stand, what is decided, what is next, what is still open. If any other document disagrees with it, this one wins. Kept as the *only* plan document — deliberately, after four competing plans had to be consolidated on 2026-08-02. |

---

## Records

| | Document | Note |
|---|---|---|
| 🧾 | [PROGRESS_REPORT_2026-07-14.md](PROGRESS_REPORT_2026-07-14.md) | Status as of 2026-07-14. Superseded by CURRENT_PLAN.md; kept as a dated record. |
| 🧾 | [run_archive/](run_archive/) | `config.json` + `sf_log.txt` for every historic training run (v9–v17, v5_smooth, the v6_dr ladder, v7_shaft, v8_p1). `v8_p1_v1_run2_post0713/code/` additionally snapshots the exact source that produced that run — **this is the reference for what the code looked like before the 2026-08-04 changes**, since there is no git history from before 2026-08-06. Archive a run here before deleting it. |
| 🧾 | [media/](media/) | Recorded runs. Currently v1-era only. |
| 🧾 | [portfolio/](portfolio/) | Four hand-written docs (architecture, results, domain randomisation, design iterations) harvested from the earlier public-facing v6 snapshot, plus its publishing checklist. **v6_dr-era and not yet refreshed** — each carries a banner saying so. Being folded into refreshed reference docs; this folder disappears when that is done. |
| 🧾 | [archive/](archive/) | Mujuco_V2's root docs verbatim, plus V3-era plans superseded by CURRENT_PLAN.md. **Never act on anything in here** — paths and numbers inside are stale by definition. |

---

## Planned

The reference layer is mid-rewrite. Target is four documents, one per domain, each answering
"how does this part work":

| Planned | Seeded from |
|---|---|
| `architecture.md` | CONTROL_LOOP_REPORT + `portfolio/architecture.md` |
| `simulation.md` | colon generator, scope model, contact physics + `portfolio/domain_randomisation.md` |
| `perception.md` | DA3 fine-tune, error map, realtime runtime |
| `hardware.md` | rig, CAD, firmware, bring-up measurements |

Until they exist, the documents in **Read these** above are the current reference.

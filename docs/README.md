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
| 📘 | [architecture.md](architecture.md) | **Start here.** How the whole system fits together — the three pipelines, the 25 Hz control loop, the network, the simulated hardware, and the error map that bridges sim and real. Describes the live `v8_p1` line. |
| 📘 | [CONTROL_LOOP_REPORT_2026-08-05.md](CONTROL_LOOP_REPORT_2026-08-05.md) | The same loop at line-by-line depth, with every constant traced to a file and line, plus the measured evidence behind each claim. Self-contained: written to be readable with no repo access. Read `architecture.md` first, this when you need the exact number. |
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
| 🧾 | [portfolio/](portfolio/) | Hand-written docs harvested from the earlier public-facing v6 snapshot, plus its publishing checklist. **v6_dr-era and not yet refreshed** — each carries a banner saying so. Being folded into the reference docs one at a time; the folder disappears when the last one goes. Remaining: results, domain randomisation, design iterations. |
| 🧾 | [archive/](archive/) | Mujuco_V2's root docs verbatim, V3-era plans superseded by CURRENT_PLAN.md, and `portfolio_v6/` — v6 docs already folded into current reference docs. **Never act on anything in here** — paths and numbers inside are stale by definition. |

---

## Planned

The reference layer is mid-rewrite. Target is one document per domain, each answering "how does
this part work":

| Planned | Seeded from | Status |
|---|---|---|
| `architecture.md` | CONTROL_LOOP_REPORT + `portfolio/architecture.md` | ✅ **done** 2026-08-06 |
| `simulation.md` | colon generator, scope model, contact physics + `portfolio/domain_randomisation.md` | pending |
| `perception.md` | DA3 fine-tune, error map, realtime runtime | pending |
| hardware | — | ✅ lives in [`../hardware/README.md`](../hardware/README.md), next to the thing it documents |

`results.md` is deliberately last: the current checkpoint has no evaluation run, so there is
nothing honest to write beyond what the top-level README already labels.

Until the pending two exist, the documents in **Read these** above are the current reference.

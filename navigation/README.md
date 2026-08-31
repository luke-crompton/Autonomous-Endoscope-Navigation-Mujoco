# navigation/ — RL navigation lines

| Folder | Status | Result |
|---|---|---|
| `v8_p1/` | **LIVE** — all new work happens here | `v8_p1_v1`, A–F retrain (2026-08-04, from scratch, stopped at 4.52M steps): **~97% mean progress** — a *training* statistic from sampled rollouts, **no eval run yet**. Supersedes the post-0713 run (93.5% mean progress, `tip_stuck` 73.3% → 12.3%). See [../docs/CURRENT_PLAN.md](../docs/CURRENT_PLAN.md). |
| `v7_shaft/` | **SUPERSEDED** by `v8_p1/` — kept runnable, don't develop here | v7_shaft_v2: ~92% mean progress / ~90% completion. v7_shaft_v4 evaluated at 35.0% completion / 72.3% mean progress, but on a colon generator with a self-intersection bug fixed the same day — not a clean read. |
| `v6_dr/` | **FROZEN milestone** — eval/view only, do not edit | v6_dr_700V3: 93.75% completion / 96.7% mean progress over 80 seeds × 5 episodes (eval CSV in `runs_sf/v6_dr_700V3/`). Last version with the kinematic mocap base advance. |

v7_shaft replaced v6's kinematic base with a real physical flexible shaft (two-DOF universal
joints) fed by an entrance roller/capstan mechanism — wall contact transmits push force around
bends, and slip/buckling are genuine failure modes, as in real colonoscopy.

**v8_p1** ("V8 Phase 1" — the folder is `v8_p1/`, *not* `v8/`, which never existed) then replaced
the privileged force observations with a hardware-matching **cable-extension** signal, added tendon
integral control, and reworked shaft/tip compliance. The 2026-08-04 A–F retrain went further:
the state is now **6-D with no force or cable channel at all** (`force_norm`, `net_fx`/`net_fz`,
and finally `ten_x_n`/`ten_z_n` all lack a faithful real-hardware equivalent), tendon force was cut
~9× to a bench-measured ~3 N, the tip camera moved to a 16:9 100°-horizontal frame, and the control
loop dropped to 25 Hz to match the deployment rate. Full lineage and design rationale:
[../docs/ITERATION_HISTORY.md](../docs/ITERATION_HISTORY.md).

All three folders are deliberately self-contained bundles (each carries its own
`generate_videoscope_one_section.py`, `new8.stl`, `error_map_64.npy`): the frozen v6 snapshot
stays runnable with zero edits, and the live `v8_p1/` folder syncs to the Linux training rig and
the WSL2 mirror as a single unit. **Never add cross-folder imports.**

**Phase 2** (gravity + anatomical left-lateral-decubitus colon) is unstarted and has no folder.
Per the original design it would target `v8_p1/` in place rather than forking a new line.

> ⚠️ **v6-era checklist, harvested 2026-08-06.** Written for publishing the `colonoscope-rl`
> v6_dr snapshot. The items are still the right shape, but the checkpoint names, result figures
> and file paths all refer to v6 and need re-scoping for v8_p1 before use.

# Publishing Checklist

Items remaining before this repo is ready to make public. Delete this file in the final commit.

---

## Visual assets  ← biggest gap

- [ ] Record a GIF or short MP4 of the policy navigating a colon end-to-end
      (`viewer_sf.py` + screen recorder; add to `docs/figures/navigation_demo.gif`)
- [ ] Export training curve as a PNG and add to `docs/figures/training_curve.png`
      (`tensorboard --logdir docs/training_curves/` → screenshot, or parse `.tfevents` with `tensorboard.backend.event_processing`)
- [ ] Create architecture diagram (perception → RL → deployment) and add to `docs/figures/architecture.png`
- [ ] Reference all three figures from README (demo GIF near the top, others below the results table)

---

## GitHub Release assets

- [ ] Upload `best_000000728_2981888_reward_1.238.pth` as a GitHub Release asset (v6_dr_700V3)
- [ ] Upload `da3small_finetune_head.pth` as a GitHub Release asset
- [ ] Confirm the download links in README → Checkpoints table resolve correctly after upload
- [ ] Test the full quickstart cold: fresh venv → `pip install -r requirements.txt` → download checkpoint → `python navigation/eval_sf.py --seeds 5 --episodes 1`

---

## Documentation gaps

- [ ] Fill in actual GPU model and approximate training wall-clock time in `docs/results.md`
      (currently says "RTX GPU" with no timing; check workstation specs)
- [ ] Add a CITATION section to README if this becomes part of a report or dissertation
      (BibTeX entry for the repo, or a pointer to the paper/report once available)

---

## Final checks

- [ ] Run `python navigation/build_collision_scene.py --seed 0` on a clean install to confirm scene generation works
- [ ] Confirm `python navigation/eval_sf.py --seeds 80 --episodes 5` reproduces the 93.75% result with the released checkpoint
- [ ] Delete this file and commit as "Ready to publish"

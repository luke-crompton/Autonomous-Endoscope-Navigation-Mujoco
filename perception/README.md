# perception/ — monocular depth

The policy's only view of the world is a depth image. This folder produces the model that
generates it, and the runtime that runs it on the real scope.

| Folder | Role | Runs |
|---|---|---|
| `rd_v2/` | **Offline pipeline.** Fine-tunes DA3-SMALL on rendered colon imagery and measures its error. Run once; produces two artefacts the rest of the project consumes. | Blender + CUDA, hours |
| `realtime/` | **Deployment runtime.** The live inference path, camera tooling, and latency benchmark. | live camera, 31 fps |

Both require **Python 3.11**, not the 3.14 used for `navigation/` — the CUDA torch,
`depth_anything_3` and OpenCV stack exists only there. `realtime/_py311.py` relaunches a script
under the right interpreter automatically; otherwise invoke as `py -3.11`.

---

## What it produces

| Artefact | Consumed by |
|---|---|
| `da3small_finetune_head.pth` | the deployment runtime — **131 MB, not in git**, distributed as a Release asset |
| `error_map_64.npy` | the RL environment, copied into each `navigation/` bundle — the sim-to-real bridge |

The error map is the project's main sim-to-real idea: rather than injecting generic noise into
simulated depth, it injects *the measured spatial error structure of the actual deployment
model*. See [../docs/architecture.md](../docs/architecture.md).

## ⚠️ The depth is relative, not metric

`metric_weight = 0`, so the model's output is **scale-relative**. The headline **1.43 mm val
MAE** (against 8.30 mm zero-shot) is *scale-aligned against ground truth* — it is not a distance
accuracy obtainable at deploy time, and millimetres must never be read off the output. Older
documents quoting **1.60 mm** are wrong; the checkpoint log is authoritative.

This is consistent with how the RL side uses it: the environment normalises each depth frame by
its own maximum, so the policy reads relative structure and never has access to absolute
distance.

---

## `rd_v2/` — the offline pipeline

```
blender_depth_dataset.py       →  training frames (Blender/Cycles, OPTIX)
filter_void_frames.py          →  drops frames with too many pure-black pixels
finetune_da3.py                →  da3small_finetune_head.pth   (head-only; backbone frozen)

blender_depth_dataset_eval.py  →  unseen eval frames, different geometry
run_da3_inference.py           →  predictions
compute_error_map.py           →  error_map_64.npy
```

`generate_error_map.py` runs that whole chain end to end — it needs Blender and takes hours.
`test_navigation.py` closes the loop offline, driving the MuJoCo scope with the fine-tuned model
as its perception. `error_map_48.npy` and `error_map_full.npy` are alternate resolutions;
**`error_map_64.npy` is the one in use**.

## `realtime/` — the deployment runtime

| Script | |
|---|---|
| `da3_runtime.py` | **the single shared inference path.** Reproduces the fine-tune contract exactly (504×280, ImageNet normalisation) and provides `to_obs_64()` for the navigation observation |
| `bench_da3.py` | latency benchmark — **31 fps / 32.18 ms measured**, which is why the control loop is 25 Hz |
| `probe_camera.py` | reports what the camera *actually* delivers, as opposed to what it claims |
| `preview_camera_depth.py` | live camera → depth preview, for eyeballing output and frame rate |

The scope camera is **OpenCV index 1**. Index 2 is a broken NVIDIA Broadcast virtual device —
opening it wastes time.

---

## Known gaps

- **Lens distortion.** The real camera is a **122° horizontal fisheye**; the pipeline was built
  on **100° horizontal rectilinear**. The fix is perception-side only — measure `K`/`D`, then a
  precomputed `cv2.fisheye` undistort LUT applied before DA3 — and it does not require a
  retrain. This is the largest known observation mismatch.
- **The error map's geometry is approximate.** It was measured on the older square 85° camera
  and is resized to the current 54×96 frame, an anisotropic stretch. It is a smooth envelope, so
  this is a mild distortion rather than a wrong signal — but it is not a verified map at the
  current geometry.

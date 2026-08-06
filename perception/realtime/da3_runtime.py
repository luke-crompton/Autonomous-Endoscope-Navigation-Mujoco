"""
Deployment-side runtime for the fine-tuned DA3-SMALL depth model.

This is the single shared inference path. `bench_da3.py` times it and
`realtime_loop.py` (later) runs it on live camera frames, so the number the
benchmark reports is the number the loop actually pays. Do not fork this
logic into either caller.

Why not reuse run_da3_inference.py: that script calls `model.inference([path])`,
which does DA3's own file-loading and preprocessing. The fine-tuned head was
NOT trained through that path -- finetune_da3.py builds the tensor itself
(504x280 PIL-bilinear + ImageNet norm) and calls the inner `model.model(...)`.
This runtime reproduces the training path, which is the one the checkpoint's
head is valid for, and which also lets us time preprocessing separately.

⚠️ OUTPUT IS RELATIVE, NOT METRIC. The checkpoint's stored args show
`metric_weight: 0.0` -- it was fine-tuned on scale-invariant loss alone, so it
has no absolute anchor. Larger value = farther, but the global scale is
arbitrary and varies frame to frame. The "1.43 mm" figure the checkpoint
reports is a SCALE-ALIGNED error, measured by fitting the median pred/GT ratio
per frame; it is not an accuracy you can obtain without ground truth.

This costs nothing downstream: the nav policy consumes depth normalised by the
per-frame maximum (scope_colon_env.py:1009-1013), so absolute scale is
discarded before the policy sees it. The whole chain is scale-free. Just never
read a distance in millimetres off this output.

Usage:
    from da3_runtime import Da3Depth
    engine = Da3Depth(checkpoint='.../da3small_finetune_head.pth')
    depth = engine(frame_rgb)              # (280, 504) float32, relative
    obs = engine.to_obs(depth)             # (1, 54, 96) float32 in [0,1]
"""

import os
import time

import numpy as np
import torch
from PIL import Image


# ------------------------- Training contract ------------------------------
# MUST MATCH perception/rd_v2/finetune_da3.py. The fine-tuned head is only
# valid for inputs built exactly this way -- change nothing here without
# re-checking that file (TARGET_W/TARGET_H at :63-64, norm at :69-70).

TARGET_W = 504          # DA3 ViT patch size is 14; 504 = 36 patches
TARGET_H = 280          # 280 = 20 patches
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# The render spec the model was fine-tuned on (blender_depth_dataset.py:48-54).
# Training frames were rendered at 320x180 and then UPSCALED to 504x280, so
# they carry a 320x180 blur budget. A native 1280x720 camera frame resized
# straight to 504x280 is sharper than anything the model has seen -- see
# `match_train_blur` below.
SRC_W = 320
SRC_H = 180
SRC_FOV_DEG = 100.0     # Blender lens_unit='FOV' => HORIZONTAL on 16:9
CLIP_END_M = 1.0        # depth >= 0.99 m was written as 0 (invalid) in training

# 504/280 = 1.8000 but 320/180 = 1.7778: the training resize is NOT
# aspect-preserving (finetune_da3.py's comment claims it is). It is a ~1.3%
# horizontal stretch baked into the fine-tune. Reproduce it, don't fix it.

DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'rd_v2', 'checkpoints', 'da3small_finetune_head.pth',
)

_PRECISIONS = {
    'fp32': None,
    'bf16': torch.bfloat16,     # what finetune_da3.py trained/evaluated under
    'fp16': torch.float16,
}


def resolve_device(arg='auto'):
    if arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return arg


def sync(device):
    """Block until queued CUDA work is done. Required before any timing read --
    without it every GPU stage appears to take ~0 ms and the cost lands on
    whichever line next touches the result."""
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


class Da3Depth:
    """Fine-tuned DA3-SMALL, held resident and callable per frame.

    Stages are exposed separately (`preprocess` / `forward` / `postprocess`)
    so the benchmark can attribute latency. `__call__` chains all three.
    """

    def __init__(self,
                 checkpoint=DEFAULT_CHECKPOINT,
                 model_name='depth-anything/DA3-SMALL',
                 device='auto',
                 precision='bf16',
                 channels_last=False,
                 compile_model=False,
                 match_train_blur=True):
        if precision not in _PRECISIONS:
            raise ValueError(f'precision must be one of {list(_PRECISIONS)}')
        self.device = torch.device(resolve_device(device))
        self.precision = precision
        self.autocast_dtype = _PRECISIONS[precision]
        self.match_train_blur = match_train_blur

        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as e:
            raise SystemExit(
                'depth_anything_3 is not installed in this interpreter.\n'
                '  Use the Python 3.11 env (see README interpreter matrix):\n'
                '  C:\\Users\\lukec\\AppData\\Local\\Programs\\Python\\Python311\\python.exe\n'
                f'Original error: {e}'
            )

        t0 = time.time()
        self.model = DepthAnything3.from_pretrained(model_name)
        self.model = self.model.to(device=self.device)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=self.device)
            state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            self.model.load_state_dict(state)
            self.ckpt_val_mae_mm = (
                ckpt.get('val_scale_aligned_mae_m', float('nan')) * 1000
                if isinstance(ckpt, dict) else float('nan')
            )
        else:
            self.ckpt_val_mae_mm = float('nan')
        self.model.eval()
        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.channels_last = channels_last

        self.compiled = False
        if compile_model:
            # Note: first call after this pays a long one-off graph compile.
            # The benchmark's warm-up must cover it or the median is garbage.
            self.model.model = torch.compile(self.model.model)
            self.compiled = True

        self.load_seconds = time.time() - t0
        self.checkpoint = checkpoint
        self.model_name = model_name

    # ---------------------------------------------------------- stages

    def to_model_pil(self, img, bgr=False):
        """Frame -> the exact 504x280 PIL image the network is fed.

        Accepts a path, a PIL.Image, or an (H, W, 3) uint8 array. OpenCV hands
        back BGR -- pass bgr=True rather than slicing outside, so the benchmark
        and the live loop pay the identical cost.

        Split out from `preprocess` so a preview can DISPLAY the model's actual
        input rather than a lookalike built by duplicated resize code.

        PIL bilinear is used, not cv2.INTER_LINEAR: on downscale PIL applies a
        support-scaled (antialiased) filter and cv2 does not, so they are not
        interchangeable. Training used PIL.
        """
        if isinstance(img, (str, os.PathLike)):
            pil = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            arr = img[:, :, ::-1] if bgr else img
            pil = Image.fromarray(np.ascontiguousarray(arr)).convert('RGB')
        else:
            pil = img.convert('RGB')

        if self.match_train_blur and pil.size != (SRC_W, SRC_H):
            # Reproduce the 320x180 bottleneck the training renders passed
            # through, so a high-res camera frame carries the same sharpness
            # budget as the fine-tune data.
            pil = pil.resize((SRC_W, SRC_H), Image.BILINEAR)

        return pil.resize((TARGET_W, TARGET_H), Image.BILINEAR)

    def preprocess(self, img, bgr=False):
        """Frame -> normalised (1, 1, 3, H, W) tensor on device."""
        return self.preprocess_pil(self.to_model_pil(img, bgr=bgr))

    def preprocess_pil(self, pil):
        """Already-504x280 PIL -> normalised (1, 1, 3, H, W) tensor on device.

        For callers that already built the model input with `to_model_pil`
        (e.g. to display it). Calling `preprocess` on that PIL instead would
        run the whole resize chain a SECOND time -- and with match_train_blur
        on it would re-apply the 320x180 bottleneck to an already-bottlenecked
        image, double-blurring it.
        """
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(arr).permute(2, 0, 1)          # (3, H, W)
        t = t.unsqueeze(0).to(self.device, non_blocking=True)
        if self.channels_last:
            t = t.contiguous(memory_format=torch.channels_last)
        return t.unsqueeze(1)                                # (1, 1, 3, H, W)

    @torch.no_grad()
    def forward(self, image_5d):
        """Normalised tensor -> (H, W) depth tensor on device, still on GPU.

        Call signature mirrors finetune_da3.py's train/eval path exactly.
        """
        if self.autocast_dtype is not None:
            ctx = torch.autocast(device_type=self.device.type,
                                 dtype=self.autocast_dtype)
        else:
            ctx = torch.autocast(device_type=self.device.type, enabled=False)
        with ctx:
            out = self.model.model(
                image_5d, extrinsics=None, intrinsics=None,
                export_feat_layers=[], infer_gs=False,
                use_ray_pose=False, ref_view_strategy='saddle_balanced',
            )
            depth = out['depth'].squeeze(1).float()          # (1, H, W)
        return depth[0]

    def postprocess(self, depth_t):
        """Device tensor -> (H, W) float32 numpy. Relative depth, not metres."""
        return depth_t.detach().to('cpu', copy=False).numpy()

    def __call__(self, img, bgr=False):
        return self.postprocess(self.forward(self.preprocess(img, bgr=bgr)))

    # ---------------------------------------------------------- nav obs

    @staticmethod
    def to_obs(depth, res=(54, 96)):
        """Predicted depth -> the nav policy's depth obs, (1, H, W) in [0, 1].

        `res` is (HEIGHT, WIDTH), matching numpy's .shape and the env's
        (DEFAULT_DEPTH_RES_H, DEFAULT_DEPTH_RES_W). An int means square.

        Normalisation matches scope_colon_env.py -- divide by the per-frame
        MAXIMUM, not by a fixed metric clip. (The env's own docstring says
        `depth_clip_m`; the code does not. The code is what the policy was
        trained on.)

        ASPECT: resolved 2026-08-04. Both sides are now 16:9 -- the sim camera
        moved to fovy=67.7 deg vertical on a 96x54 render (see TIP_CAM_FOVY in
        navigation/v8_p1/build_collision_scene.py), which is 100.0 deg
        horizontal and therefore an exact reproduction of the camera the DA3
        fine-tune was rendered with in Blender. This resize no longer squashes
        the aspect: a 16:9 frame goes to a 16:9 obs.

        ⚠️ STILL OPEN -- lens distortion, NOT aspect. The real scope is 140 deg
        DIAGONAL, which on 16:9 under the equidistant mapping a lens that wide
        must use is 122 deg horizontal / 68.6 deg vertical. The vertical matches
        sim to within a degree, but horizontally the real frame packs 122 deg
        into the width where sim (and the DA3 training set) has 100 deg. That
        difference IS the lens's barrel distortion, and it is not corrected
        here. Fixing it means fisheye-calibrating the camera
        (probe_camera.py --measure_fov checkerboard, which writes K and D) and
        remapping the frame to a 100-deg-horizontal rectilinear virtual camera
        before inference. Until then, objects near the left/right edges sit
        closer to centre than the policy expects. Centre-of-frame -- where the
        lumen usually is -- is unaffected.
        """
        h, w = (res, res) if isinstance(res, int) else (int(res[0]), int(res[1]))
        d = np.asarray(depth, dtype=np.float32)
        pil = Image.fromarray(d).resize((w, h), Image.BILINEAR)  # PIL takes (W, H)
        d = np.asarray(pil, dtype=np.float32)
        max_d = max(float(d.max()), 1e-6)
        return np.clip(d / max_d, 0.0, 1.0)[None, :, :]

    # Old name, kept so bench_da3.py / preview_camera_depth.py keep working.
    # The "_64" is a lie since 2026-08-04 (the obs is 54x96) -- prefer to_obs().
    to_obs_64 = to_obs

    # ---------------------------------------------------------- info

    def describe(self):
        gpu = (torch.cuda.get_device_name(self.device)
               if self.device.type == 'cuda' else 'cpu')
        return {
            'model': self.model_name,
            'checkpoint': self.checkpoint,
            'ckpt_val_mae_mm': self.ckpt_val_mae_mm,
            'device': str(self.device),
            'gpu': gpu,
            'precision': self.precision,
            'channels_last': self.channels_last,
            'compiled': self.compiled,
            'match_train_blur': self.match_train_blur,
            'input_hw': [TARGET_H, TARGET_W],
            'torch': torch.__version__,
            'load_seconds': round(self.load_seconds, 2),
        }

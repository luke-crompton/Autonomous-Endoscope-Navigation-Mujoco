"""
Self-contained Blender dataset generator for RD_V2.

Generates RGB + metric-depth pairs from procedural colon interiors.

Run:
    blender --background --python blender_depth_dataset.py

Or with overrides:
    blender --background --python blender_depth_dataset.py -- \\
        --out_dir /path/to/dataset_finetune --n_geoms 5 --frames_per_geom 300

Default: 5 geoms * 300 frames = 1500 total frames (0-1499).
finetune_da3.py holds out frames 1200-1499 (last geom) as the val set.

Outputs:
    <out_dir>/rgb/frame_NNNNN.png       Cycles-rendered RGB
    <out_dir>/depth/depth_NNNNN.npy     metric depth in metres (0 = invalid)

Depth method: a second 1-sample emission render (strength = Z/clip_end)
is done per frame. Background pixels (no hit) remain 0 = invalid.
"""

import argparse
import glob
import math
import os
import sys
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector

# ---- arg parsing --------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
_p = argparse.ArgumentParser()
_p.add_argument('--out_dir', default=os.path.join(_HERE, 'dataset_finetune'))
_p.add_argument('--n_geoms', type=int, default=5)
_p.add_argument('--frames_per_geom', type=int, default=300)
_p.add_argument('--samples', type=int, default=32)
_p.add_argument('--seed', type=int, default=0)
_p.add_argument('--width', type=int, default=320)
_p.add_argument('--height', type=int, default=180)
_p.add_argument('--fov_deg', type=float, default=100.0)
_args = _p.parse_args(_argv)

W, H = _args.width, _args.height
CLIP_END = 1.0  # metres


# ---- GPU ----------------------------------------------------------------

def _enable_gpu():
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for device_type in ('OPTIX', 'CUDA'):
        try:
            prefs.compute_device_type = device_type
            prefs.get_devices()
        except Exception:
            continue
        gpus = [d for d in prefs.devices if d.type == device_type]
        if gpus:
            for d in prefs.devices:
                d.use = (d.type == device_type)
            return device_type
    return None

_GPU = _enable_gpu()
print(f'Render device: {_GPU or "CPU"}', flush=True)


# ---- colon geometry (identical to blender_render_worker.py) -------------

def cubic_interp(t_c, vals, t_o):
    n = len(t_c); r = np.zeros(len(t_o))
    for i in range(n - 1):
        m = (t_o >= t_c[i]) & (t_o <= t_c[min(i + 1, n - 1)])
        if i == n - 2:
            m = t_o >= t_c[i]
        if m.any():
            f = (t_o[m] - t_c[i]) / (t_c[i + 1] - t_c[i] + 1e-10)
            p0, p1 = vals[max(i - 1, 0)], vals[i]
            p2, p3 = vals[min(i + 1, n - 1)], vals[min(i + 2, n - 1)]
            r[m] = 0.5 * ((2 * p1) + (-p0 + p2) * f
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)
    return r


def random_centerline(seed):
    np.random.seed(seed)
    n = np.random.randint(5, 10)
    L = np.random.uniform(0.20, 0.35)
    ctrl = np.zeros((n, 3))
    ctrl[:, 2] = np.linspace(0, L, n)
    amp = np.random.uniform(0.01, 0.06)
    for i in range(1, n):
        ctrl[i, 0] = amp * np.sin(np.random.uniform(0, 2 * np.pi))
        ctrl[i, 1] = amp * 0.6 * np.sin(np.random.uniform(0, 2 * np.pi))
    t_c = np.linspace(0, 1, n); t_f = np.linspace(0, 1, 500)
    return np.column_stack([cubic_interp(t_c, ctrl[:, k], t_f) for k in range(3)])


def compute_frames(cl):
    T = np.gradient(cl, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-10
    N = np.zeros_like(T)
    N[0] = np.cross(T[0], [1, 0, 0])
    N[0] /= np.linalg.norm(N[0]) + 1e-10
    for i in range(1, len(cl)):
        proj = N[i - 1] - np.dot(N[i - 1], T[i]) * T[i]
        pn = np.linalg.norm(proj)
        N[i] = proj / pn if pn > 1e-8 else N[i - 1]
    B = np.cross(T, N)
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-10
    return T, N, B


def build_tube(cl, radius, n_haustra, T, N, B):
    n_rad = 36
    theta = np.linspace(0, 2 * np.pi, n_rad, endpoint=False)
    t_p = np.linspace(0, 1, len(cl))
    hf = n_haustra * 2 * np.pi
    h = 1.0 + 0.22 * np.cos(hf * t_p) + 0.07 * np.cos(hf * 3 * t_p + 1.0)
    h -= 0.05 * np.clip(-np.cos(hf * t_p), 0, 1) ** 2

    md = bpy.data.meshes.new('C'); bm_obj = bmesh.new(); vg = []
    for i in range(len(cl)):
        ring = []; c, n_v, b_v = cl[i], N[i], B[i]; r = radius * h[i]
        for j in range(n_rad):
            te = 0.025 * np.cos(3 * theta[j]) ** 6
            rj = r - te * radius
            pos = c + rj * (np.cos(theta[j]) * n_v + np.sin(theta[j]) * b_v)
            ring.append(bm_obj.verts.new(Vector(pos.tolist())))
        vg.append(ring)
    bm_obj.verts.ensure_lookup_table()
    for i in range(len(cl) - 1):
        for j in range(n_rad):
            jn = (j + 1) % n_rad
            bm_obj.faces.new([vg[i][j], vg[i][jn], vg[i + 1][jn], vg[i + 1][j]])
    bm_obj.to_mesh(md); bm_obj.free(); md.update()
    obj = bpy.data.objects.new('C', md)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()
    mod = obj.modifiers.new('S', 'SUBSURF'); mod.levels = 0; mod.render_levels = 1
    return obj


def create_material(seed):
    np.random.seed(seed)
    mat = bpy.data.materials.new('T'); mat.use_nodes = True
    n = mat.node_tree.nodes; l = mat.node_tree.links; n.clear()
    out = n.new('ShaderNodeOutputMaterial')
    bsdf = n.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Subsurface Weight'].default_value = np.random.uniform(0.03, 0.18)
    bsdf.inputs['Specular IOR Level'].default_value = np.random.uniform(0.4, 0.9)
    bsdf.inputs['Roughness'].default_value = np.random.uniform(0.12, 0.35)
    bsdf.inputs['Coat Weight'].default_value = np.random.uniform(0.2, 0.6)
    bsdf.inputs['Coat Roughness'].default_value = np.random.uniform(0.03, 0.15)
    n1 = n.new('ShaderNodeTexNoise')
    n1.inputs['Scale'].default_value = np.random.uniform(25, 80)
    n1.inputs['Detail'].default_value = np.random.uniform(6, 14)
    n2 = n.new('ShaderNodeTexNoise')
    n2.inputs['Scale'].default_value = np.random.uniform(100, 250)
    n2.inputs['Detail'].default_value = np.random.uniform(8, 14)
    mu = n.new('ShaderNodeMath'); mu.operation = 'MULTIPLY'
    l.new(n1.outputs['Fac'], mu.inputs[0]); l.new(n2.outputs['Fac'], mu.inputs[1])
    ramp = n.new('ShaderNodeValToRGB')
    br = 0.72 + np.random.uniform(-0.12, 0.12)
    bg = 0.30 + np.random.uniform(-0.08, 0.08)
    bb = 0.24 + np.random.uniform(-0.06, 0.06)
    ramp.color_ramp.elements[0].position = 0.10
    ramp.color_ramp.elements[0].color = (br - 0.20, bg - 0.14, bb - 0.12, 1)
    ramp.color_ramp.elements[1].position = 0.45
    ramp.color_ramp.elements[1].color = (br, bg, bb, 1)
    el = ramp.color_ramp.elements.new(0.80)
    el.color = (br + 0.12, bg + 0.18, bb + 0.14, 1)
    l.new(mu.outputs['Value'], ramp.inputs['Fac'])
    l.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    bn = n.new('ShaderNodeTexNoise')
    bn.inputs['Scale'].default_value = np.random.uniform(150, 400)
    bn.inputs['Detail'].default_value = 8.0
    bp = n.new('ShaderNodeBump')
    bp.inputs['Strength'].default_value = np.random.uniform(0.1, 0.3)
    bp.inputs['Distance'].default_value = np.random.uniform(0.0002, 0.0006)
    l.new(bn.outputs['Fac'], bp.inputs['Height'])
    l.new(bp.outputs['Normal'], bsdf.inputs['Normal'])
    l.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


# ---- scene helpers ------------------------------------------------------

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for m in list(bpy.data.meshes):
        bpy.data.meshes.remove(m)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)
    for ld in list(bpy.data.lights):
        bpy.data.lights.remove(ld)


def setup_render(light_energy, exposure):
    """Camera + spotlight + Cycles config. No compositor."""
    bpy.ops.object.camera_add()
    cam = bpy.context.object; cam.name = 'E'
    cam.data.lens_unit = 'FOV'
    cam.data.angle = math.radians(_args.fov_deg)
    cam.data.clip_start = 0.0005
    cam.data.clip_end = CLIP_END
    bpy.context.scene.camera = cam

    ld = bpy.data.lights.new('L', type='SPOT')
    ld.energy = light_energy
    ld.color = (1.0, np.random.uniform(0.90, 0.98), np.random.uniform(0.80, 0.92))
    ld.spot_size = math.radians(140); ld.spot_blend = 0.3
    ld.shadow_soft_size = 0.004; ld.use_shadow = True
    lo = bpy.data.objects.new('L', ld)
    bpy.context.collection.objects.link(lo)
    lo.parent = cam; lo.location = (0, 0, 0)

    s = bpy.context.scene
    s.render.engine = 'CYCLES'
    s.cycles.device = 'GPU' if _GPU is not None else 'CPU'
    s.render.resolution_x = W; s.render.resolution_y = H
    s.render.image_settings.file_format = 'PNG'
    s.cycles.samples = _args.samples
    s.cycles.use_denoising = True
    s.cycles.max_bounces = 2
    s.cycles.diffuse_bounces = 1
    s.cycles.glossy_bounces = 2
    s.view_settings.view_transform = 'Filmic'
    s.view_settings.look = 'Very High Contrast'
    s.view_settings.exposure = exposure
    if s.world and hasattr(s.world, 'node_tree') and s.world.node_tree:
        bg = s.world.node_tree.nodes.get('Background')
        if bg:
            bg.inputs['Color'].default_value = (0, 0, 0, 1)
            bg.inputs['Strength'].default_value = 0

    return cam


# ---- depth via emission material ----------------------------------------

def _depth_emission_mat():
    """Emission shader: strength = camera Z-depth / CLIP_END.
    With 1 sample and no bounces this gives exact per-pixel depth in EXR.
    """
    mat = bpy.data.materials.new('__depth__')
    mat.use_nodes = True
    tree = mat.node_tree; tree.nodes.clear()
    out = tree.nodes.new('ShaderNodeOutputMaterial')
    emit = tree.nodes.new('ShaderNodeEmission')
    cam_data = tree.nodes.new('ShaderNodeCameraData')
    divide = tree.nodes.new('ShaderNodeMath')
    divide.operation = 'DIVIDE'
    divide.inputs[1].default_value = CLIP_END
    tree.links.new(cam_data.outputs['View Z Depth'], divide.inputs[0])
    tree.links.new(divide.outputs['Value'], emit.inputs['Strength'])
    tree.links.new(emit.outputs['Emission'], out.inputs['Surface'])
    return mat


def render_depth(scene, depth_tmp, frame_idx):
    """Render depth pass. Returns (H, W) float32 in metres, or None on failure."""
    dm = _depth_emission_mat()
    vl = scene.view_layers.get('ViewLayer') or scene.view_layers[0]
    vl.material_override = dm

    # Save render state
    old_samples = scene.cycles.samples
    old_bounces = scene.cycles.max_bounces
    old_denoise = scene.cycles.use_denoising
    old_format = scene.render.image_settings.file_format
    old_filepath = scene.render.filepath

    scene.cycles.samples = 1
    scene.cycles.max_bounces = 0
    scene.cycles.use_denoising = False
    scene.render.image_settings.file_format = 'OPEN_EXR'
    try:
        scene.render.image_settings.color_depth = '32'
    except Exception:
        pass
    tmp_stem = os.path.join(depth_tmp, f'z_{frame_idx:05d}')
    scene.render.filepath = tmp_stem
    bpy.ops.render.render(write_still=True)

    # Restore render state
    vl.material_override = None
    scene.cycles.samples = old_samples
    scene.cycles.max_bounces = old_bounces
    scene.cycles.use_denoising = old_denoise
    scene.render.image_settings.file_format = old_format
    scene.render.filepath = old_filepath
    bpy.data.materials.remove(dm)

    # Find the written EXR (Blender may append extension)
    candidates = glob.glob(tmp_stem + '*.exr') + glob.glob(tmp_stem + '*.EXR')
    if not candidates:
        return None
    exr_path = candidates[0]

    img = bpy.data.images.load(os.path.abspath(exr_path))
    w_img, h_img = img.size
    ch = img.channels
    px = np.array(img.pixels[:], dtype=np.float32)
    bpy.data.images.remove(img)
    os.remove(exr_path)

    arr = px.reshape(h_img, w_img, ch)[:, :, 0]
    arr = np.flipud(arr)                          # Blender stores bottom-to-top
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    arr *= CLIP_END                               # un-normalise → metres
    arr[arr >= CLIP_END * 0.99] = 0.0            # background / no-hit → invalid
    return arr.astype(np.float32)


# ---- camera pose sampling -----------------------------------------------

def sample_pose(cl, T, N, B, radius, rng):
    lo = int(0.15 * len(cl)); hi = int(0.85 * len(cl))
    idx = rng.randint(lo, hi)
    n_v, b_v = N[idx], B[idx]
    fwd = T[idx].copy()

    lat_frac = rng.uniform(0.0, 0.20)
    lat_ang = rng.uniform(0, 2 * np.pi)
    offset = radius * lat_frac * (np.cos(lat_ang) * n_v + np.sin(lat_ang) * b_v)
    pos = cl[idx] + offset

    tilt_angle = rng.uniform(0.0, math.radians(12))
    tilt_ang = rng.uniform(0, 2 * np.pi)
    tilt_axis = np.cos(tilt_ang) * n_v + np.sin(tilt_ang) * b_v
    fwd = (fwd * np.cos(tilt_angle)
           + np.cross(tilt_axis, fwd) * np.sin(tilt_angle)
           + tilt_axis * np.dot(tilt_axis, fwd) * (1 - np.cos(tilt_angle)))
    fwd /= np.linalg.norm(fwd) + 1e-10

    right = np.cross(fwd, np.array([0, 0, 1]))
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(fwd, np.array([1, 0, 0]))
    right /= np.linalg.norm(right) + 1e-10
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up) + 1e-10
    return pos, right, up, fwd


def set_camera_pose(cam, pos, right, up, fwd):
    cam.location = Vector(pos.tolist())
    rot = Matrix([
        list(right) + [0],
        [-up[0], -up[1], -up[2], 0],
        [-fwd[0], -fwd[1], -fwd[2], 0],
        [0, 0, 0, 1],
    ]).transposed()
    cam.rotation_euler = rot.to_euler()
    bpy.context.view_layer.update()


# ---- main ---------------------------------------------------------------

def main():
    total = _args.n_geoms * _args.frames_per_geom
    rgb_dir = os.path.join(_args.out_dir, 'rgb')
    depth_dir = os.path.join(_args.out_dir, 'depth')
    depth_tmp = os.path.join(_args.out_dir, '_depth_tmp')
    for d in (rgb_dir, depth_dir, depth_tmp):
        os.makedirs(d, exist_ok=True)

    frame_idx = 0
    for gi in range(_args.n_geoms):
        geom_seed = _args.seed * 100 + gi
        clear_scene()

        cl = random_centerline(geom_seed)
        radius = float(np.random.uniform(0.017, 0.032))
        n_haustra = int(np.random.randint(8, 24))
        T, N, B = compute_frames(cl)

        tube = build_tube(cl, radius, n_haustra, T, N, B)
        mat = create_material(geom_seed)
        tube.data.materials.append(mat)

        light_e = float(np.random.uniform(4.0, 12.0))
        exposure = float(np.random.uniform(-7.0, -5.0))
        cam = setup_render(light_e, exposure)
        scene = bpy.context.scene

        pose_rng = np.random.RandomState(geom_seed + 50000)

        for fi in range(_args.frames_per_geom):
            pos, right, up, fwd = sample_pose(cl, T, N, B, radius, pose_rng)
            set_camera_pose(cam, pos, right, up, fwd)

            rgb_path = os.path.join(rgb_dir, f'frame_{frame_idx:05d}.png')
            scene.render.filepath = rgb_path
            bpy.ops.render.render(write_still=True)

            depth = render_depth(scene, depth_tmp, frame_idx)
            if depth is not None:
                np.save(os.path.join(depth_dir, f'depth_{frame_idx:05d}.npy'), depth)

            frame_idx += 1
            print(f'{frame_idx}/{total}', flush=True)

    print(f'Done -> {_args.out_dir}', flush=True)


main()

"""
Persistent Blender worker for the closed-loop nav test.

Builds a procedural colon mesh once at startup (same algorithm as
`blender_depth_dataset.py`), then enters a stdin loop. Each loop iteration:
  - reads one JSON pose: {"pos": [x,y,z], "right": [x,y,z], "up": [x,y,z], "fwd": [x,y,z]}
  - sets the camera, renders, writes to a fixed output path
  - prints "DONE" on stdout and flushes

Send "EXIT" or close stdin to terminate.

Invoke from the main controller subprocess:
    blender.exe --background --python blender_render_worker.py -- \
        --seed 42 --render_path /path/to/temp/render.png \
        --width 320 --height 256 --samples 32

Stdout protocol:
    "READY\n"               -- after scene + GPU + camera are set up
    "DONE\n"                -- after each successful render
    "ERR: <message>\n"      -- on parse/render failure (recoverable)
"""

import argparse
import bpy
import bmesh
import json
import math
import sys

import numpy as np
from mathutils import Vector, Matrix


# ----- arg parsing on the post-"--" tail --------------------------------

_argv = sys.argv
if '--' in _argv:
    _argv = _argv[_argv.index('--') + 1:]
else:
    _argv = []

_parser = argparse.ArgumentParser()
_parser.add_argument('--seed', type=int, default=42)
_parser.add_argument('--width', type=int, default=320)
_parser.add_argument('--height', type=int, default=180)
_parser.add_argument('--samples', type=int, default=32)
_parser.add_argument('--render_path', required=True)
_args = _parser.parse_args(_argv)

W, H = _args.width, _args.height


# ----- GPU enable (OPTIX/CUDA fallback) ---------------------------------

def _enable_gpu():
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for device_type in ('OPTIX', 'CUDA'):
        try:
            prefs.compute_device_type = device_type
            prefs.get_devices()
        except Exception as exc:
            print(f"[worker] GPU {device_type} probe failed ({exc})", file=sys.stderr)
            continue
        gpus = [d for d in prefs.devices if d.type == device_type]
        if gpus:
            for d in prefs.devices:
                d.use = (d.type == device_type)
            print(f"[worker] GPU enabled: {device_type} on {[d.name for d in gpus]}", file=sys.stderr)
            return device_type
    print("[worker] No GPU; falling back to CPU", file=sys.stderr)
    return None


_GPU_DEVICE = _enable_gpu()


# ----- copies from blender_depth_dataset.py -----------------------------

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
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f**2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * f**3)
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
    h -= 0.05 * np.clip(-np.cos(hf * t_p), 0, 1)**2

    md = bpy.data.meshes.new('C'); bm = bmesh.new(); vg = []
    for i in range(len(cl)):
        ring = []; c, n_v, b_v = cl[i], N[i], B[i]; r = radius * h[i]
        for j in range(n_rad):
            te = 0.025 * np.cos(3 * theta[j])**6
            rj = r - te * radius
            pos = c + rj * (np.cos(theta[j]) * n_v + np.sin(theta[j]) * b_v)
            ring.append(bm.verts.new(Vector(pos.tolist())))
        vg.append(ring)
    bm.verts.ensure_lookup_table()
    for i in range(len(cl) - 1):
        for j in range(n_rad):
            jn = (j + 1) % n_rad
            bm.faces.new([vg[i][j], vg[i][jn], vg[i + 1][jn], vg[i + 1][j]])
    bm.to_mesh(md); bm.free(); md.update()
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
    n1 = n.new('ShaderNodeTexNoise'); n1.inputs['Scale'].default_value = np.random.uniform(25, 80)
    n1.inputs['Detail'].default_value = np.random.uniform(6, 14)
    n2 = n.new('ShaderNodeTexNoise'); n2.inputs['Scale'].default_value = np.random.uniform(100, 250)
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
    bn = n.new('ShaderNodeTexNoise'); bn.inputs['Scale'].default_value = np.random.uniform(150, 400)
    bn.inputs['Detail'].default_value = 8.0
    bp = n.new('ShaderNodeBump')
    bp.inputs['Strength'].default_value = np.random.uniform(0.1, 0.3)
    bp.inputs['Distance'].default_value = np.random.uniform(0.0002, 0.0006)
    l.new(bn.outputs['Fac'], bp.inputs['Height'])
    l.new(bp.outputs['Normal'], bsdf.inputs['Normal'])
    l.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


def setup_render(light_energy, exposure):
    bpy.ops.object.camera_add()
    cam = bpy.context.object; cam.name = 'E'
    cam.data.lens_unit = 'FOV'; cam.data.angle = math.radians(100)
    cam.data.clip_start = 0.0005; cam.data.clip_end = 1.0
    bpy.context.scene.camera = cam

    ld = bpy.data.lights.new('L', type='SPOT')
    ld.energy = light_energy
    ld.color = (1.0, np.random.uniform(0.90, 0.98), np.random.uniform(0.80, 0.92))
    ld.spot_size = math.radians(140); ld.spot_blend = 0.3
    ld.shadow_soft_size = 0.004; ld.use_shadow = True
    lo = bpy.data.objects.new('L', ld)
    bpy.context.collection.objects.link(lo)
    lo.parent = cam; lo.location = (0, 0, 0)

    s = bpy.context.scene; s.render.engine = 'CYCLES'
    s.cycles.device = 'GPU' if _GPU_DEVICE is not None else 'CPU'
    s.render.resolution_x = W; s.render.resolution_y = H
    s.cycles.samples = _args.samples
    s.cycles.use_denoising = True
    s.cycles.max_bounces = 2
    s.cycles.diffuse_bounces = 1
    s.cycles.glossy_bounces = 2
    s.view_settings.view_transform = 'Filmic'
    s.view_settings.look = 'Very High Contrast'
    s.view_settings.exposure = exposure
    s.world.use_nodes = True
    bg = s.world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs['Color'].default_value = (0, 0, 0, 1)
        bg.inputs['Strength'].default_value = 0
    return cam


# ----- scene build (once at startup) ------------------------------------

# Replicate the seeded random-call order from blender_depth_dataset.py so a
# given seed produces a deterministic colon. The dataset script does:
#   np.random.seed(42) at module top, then per geom:
#     cl = random_centerline(gi * 100)   # re-seeds internally
#     radius = uniform(0.017, 0.032)
#     n_haustra = randint(8, 24)
#     ...
#     mat = create_material(gi * 100 + si)  # re-seeds internally
#     light_e = uniform(4.0, 12.0)
#     exposure = uniform(-7.0, -5.0)
# For our standalone scene we just use args.seed wherever gi*100 appears.

np.random.seed(_args.seed)
cl = random_centerline(_args.seed)
radius = float(np.random.uniform(0.017, 0.032))
n_haustra = int(np.random.randint(8, 24))
T, N, B = compute_frames(cl)

# Clear the default cube/light/camera that come with a fresh Blender scene.
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
for m in bpy.data.meshes:
    bpy.data.meshes.remove(m)
for m in bpy.data.materials:
    bpy.data.materials.remove(m)
for ld in bpy.data.lights:
    bpy.data.lights.remove(ld)

tube = build_tube(cl, radius, n_haustra, T, N, B)
mat = create_material(_args.seed)
tube.data.materials.append(mat)

light_e = float(np.random.uniform(4.0, 12.0))
exposure = float(np.random.uniform(-7.0, -5.0))
cam = setup_render(light_e, exposure)
scene = bpy.context.scene

print(f"[worker] seed={_args.seed} radius={radius*1000:.1f}mm haustra={n_haustra} "
      f"length={np.linalg.norm(cl[-1]-cl[0]):.3f}m", file=sys.stderr)
print("READY", flush=True)


# ----- render loop ------------------------------------------------------

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    if line == "EXIT":
        break
    try:
        pose = json.loads(line)
        pos = pose['pos']
        right = pose['right']
        up = pose['up']
        fwd = pose['fwd']
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"ERR: parse failed: {exc}", flush=True)
        continue

    cam.location = Vector(pos)
    # Same camera-matrix construction as blender_depth_dataset.py so DA3
    # sees images in the same orientation it was trained on.
    rot = Matrix([list(right) + [0], [-up[0], -up[1], -up[2], 0],
                  [-fwd[0], -fwd[1], -fwd[2], 0], [0, 0, 0, 1]]).transposed()
    cam.rotation_euler = rot.to_euler()
    bpy.context.view_layer.update()

    scene.render.filepath = _args.render_path
    try:
        bpy.ops.render.render(write_still=True)
    except Exception as exc:
        print(f"ERR: render failed: {exc}", flush=True)
        continue
    print("DONE", flush=True)

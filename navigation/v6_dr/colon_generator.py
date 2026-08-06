"""
Procedural colon mesh generator (pure NumPy port of blender_depth_dataset.py,
with harder bend sampling for Stage 4 training).

Uses the same mesh construction as the DA3 Blender data path, but samples
larger/more frequent bends so navigation is harder. One seed produces:

  - High-res visual mesh   (n_axial=500 + straight distal tail, n_radial=36)
    -- what the tip cam renders against.
  - Low-res collision mesh (n_axial=125, n_radial=20,  ~5k tris) -- matches B2's
    per-triangle-prism performance envelope.
  - Centreline (500 points) for scope placement + base advancement.
  - Tube radius and haustra count.

Both meshes are sampled from the same parametric tube surface, so the
collision wall is just polygonally coarser by sub-mm vs the visual wall.

CLI:
    python colon_generator.py --seed 0 [--out_dir scenes]

Writes two STLs (visual + collision) for inspection.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


VISUAL_N_AXIAL = 500
VISUAL_N_RADIAL = 36
COLLISION_N_AXIAL = 125
COLLISION_N_RADIAL = 20
DISTAL_STRAIGHT_EXTENSION_M = 0.20
DISTAL_VISUAL_SPACING_M = 0.005

TASK_LENGTH_M = 0.70
CTRL_SPACING_M = 0.050
BEND_COUNT_MIN = 2
BEND_COUNT_MAX_EXCLUSIVE = 5      # 2-4 bends
STRAIGHT_LEN_RANGE_M = (0.08, 0.18)
BEND_ANGLE_RANGE_RAD = (np.radians(80), np.radians(150))
BEND_RADIUS_RANGE_M = (0.045, 0.080)


# ---------------------------------------------------------------------------
# Random centreline (same distribution as blender_depth_dataset.py)
# ---------------------------------------------------------------------------

def _cubic_interp(t_c: np.ndarray, vals: np.ndarray, t_o: np.ndarray) -> np.ndarray:
    """Catmull-Rom on 1-D values (verbatim port of
    blender_depth_dataset.cubic_interp)."""
    n = len(t_c)
    r = np.zeros(len(t_o))
    for i in range(n - 1):
        m = (t_o >= t_c[i]) & (t_o <= t_c[min(i + 1, n - 1)])
        if i == n - 2:
            m = t_o >= t_c[i]
        if m.any():
            f = (t_o[m] - t_c[i]) / (t_c[i + 1] - t_c[i] + 1e-10)
            p0 = vals[max(i - 1, 0)]
            p1 = vals[i]
            p2 = vals[min(i + 1, n - 1)]
            p3 = vals[min(i + 2, n - 1)]
            r[m] = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * f
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f ** 2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3
            )
    return r


def _sample_bounded_lengths(
    rng: np.random.RandomState,
    n_sections: int,
    total_length: float,
    bounds: tuple[float, float],
) -> np.ndarray | None:
    """Randomly split total_length into bounded section lengths."""
    low, high = bounds
    if total_length < n_sections * low or total_length > n_sections * high:
        return None

    lengths = np.empty(n_sections, dtype=float)
    remaining = float(total_length)
    for i in range(n_sections):
        n_left = n_sections - i - 1
        lo_i = max(low, remaining - n_left * high)
        hi_i = min(high, remaining - n_left * low)
        if i == n_sections - 1:
            lengths[i] = remaining
        else:
            lengths[i] = rng.uniform(lo_i, hi_i)
        remaining -= lengths[i]
    return lengths


def _polyline_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _scale_centreline_to_length(
    centreline: np.ndarray,
    target_length: float,
) -> np.ndarray:
    length = _polyline_length(centreline)
    if length <= 1e-12:
        return centreline.copy()
    return centreline * (target_length / length)


def random_ctrl_points(seed: int) -> tuple[np.ndarray, float, int]:
    """Piecewise colon anatomy: straight sections joined by circular-arc bends.

    Structure: [straight] → [arc 80–150°, R=4.5–8cm] → [straight] → ...
    with 2–4 major bends (3–5 segments). Each arc is discretised as points
    every ~5cm so arc-length resampling preserves the sampled bend geometry.
    Straight sections are randomly allocated from the remaining distance so
    the physical task centreline is always exactly TASK_LENGTH_M.

    Variance: n_bends (2–4), straight allocation, bend_angle (80–150°),
    R_bend (4.5–8cm), and 3D bend axis are random.
    """
    rng = np.random.RandomState(seed)

    for _ in range(100):
        n_bends = rng.randint(BEND_COUNT_MIN, BEND_COUNT_MAX_EXCLUSIVE)
        bend_angles = rng.uniform(*BEND_ANGLE_RANGE_RAD, size=n_bends)
        bend_radii = rng.uniform(*BEND_RADIUS_RANGE_M, size=n_bends)
        bend_poly_lengths = []
        for bend_angle, R_bend in zip(bend_angles, bend_radii):
            arc_len = float(R_bend * bend_angle)
            n_arc = max(2, int(round(arc_len / CTRL_SPACING_M)))
            chord = 2.0 * R_bend * np.sin(bend_angle / (2.0 * n_arc))
            bend_poly_lengths.append(float(n_arc * chord))

        straight_lengths = _sample_bounded_lengths(
            rng,
            n_sections=n_bends + 1,
            total_length=TASK_LENGTH_M - float(sum(bend_poly_lengths)),
            bounds=STRAIGHT_LEN_RANGE_M,
        )
        if straight_lengths is not None:
            break
    else:
        raise RuntimeError("Could not sample feasible fixed-length colon layout")

    pos = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])   # start along +Z
    pts = [pos.copy()]

    for seg in range(n_bends + 1):
        # Straight section: sample every ~5cm
        seg_len = float(straight_lengths[seg])
        n_pts = max(2, int(round(seg_len / CTRL_SPACING_M)))
        step = seg_len / n_pts
        for _ in range(n_pts):
            pos = pos + direction * step
            pts.append(pos.copy())

        if seg < n_bends:
            bend_angle = float(bend_angles[seg])
            R_bend = float(bend_radii[seg])

            # Random rotation axis perpendicular to current direction.
            v = rng.randn(3)
            v -= v.dot(direction) * direction
            v /= np.linalg.norm(v) + 1e-10

            # In-plane vector: u2 = cross(v, direction).
            # Arc sweeps from `direction` toward `u2`.
            u2 = np.cross(v, direction)
            u2 /= np.linalg.norm(u2) + 1e-10

            # Arc center offset from pos by R_bend in the u2 direction.
            # arc(alpha) = center + R_bend*(-u2*cos(a) + direction*sin(a))
            # At a=0 → pos; tangent=direction. At a=bend_angle → new direction.
            center = pos + R_bend * u2
            arc_len = R_bend * bend_angle
            n_arc = max(2, int(round(arc_len / CTRL_SPACING_M)))
            for k in range(1, n_arc + 1):
                alpha = bend_angle * k / n_arc
                pts.append(
                    center + R_bend * (-u2 * np.cos(alpha) + direction * np.sin(alpha))
                )

            # Update position and direction to end of arc
            pos = center + R_bend * (
                -u2 * np.cos(bend_angle) + direction * np.sin(bend_angle)
            )
            direction = direction * np.cos(bend_angle) + u2 * np.sin(bend_angle)
            direction /= np.linalg.norm(direction)

    ctrl = np.array(pts)
    radius = rng.uniform(0.018, 0.030)
    n_haustra = int(rng.randint(8, 24))
    return ctrl, float(radius), n_haustra


def centreline_from_ctrl(ctrl: np.ndarray, n_samples: int) -> np.ndarray:
    """Smooth dense path points into an exact-length centreline."""
    t_c = np.linspace(0, 1, len(ctrl))
    t_f = np.linspace(0, 1, n_samples)
    centreline = np.column_stack(
        [_cubic_interp(t_c, ctrl[:, k], t_f) for k in range(3)]
    )
    return _scale_centreline_to_length(centreline, TASK_LENGTH_M)


def append_straight_distal_lumen(
    centreline: np.ndarray,
    extension_m: float = DISTAL_STRAIGHT_EXTENSION_M,
) -> np.ndarray:
    """Append a straight distal lumen for rendering only.

    The returned points preserve the original centreline exactly and add a
    straight tail after it. Use this for the visual mesh only, so the tip camera
    does not see an artificial open void near completion while collision/contact
    geometry stays on the original task colon.
    """
    if extension_m <= 0.0:
        return centreline.copy()

    tangent = centreline[-1] - centreline[-2]
    tangent /= np.linalg.norm(tangent) + 1e-12

    step_count = max(2, int(np.ceil(extension_m / DISTAL_VISUAL_SPACING_M)))
    tail_s = np.linspace(0.0, extension_m, step_count + 1)[1:]
    tail = centreline[-1][None, :] + tail_s[:, None] * tangent[None, :]
    return np.vstack([centreline, tail])


# ---------------------------------------------------------------------------
# Tube geometry
# ---------------------------------------------------------------------------

def parallel_transport_frames(
    centreline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-point (tangent, normal, binormal). Same algorithm as
    blender_depth_dataset.compute_frames, with a fallback if the first
    tangent happens to be parallel to world +X."""
    T = np.gradient(centreline, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-10
    N = np.zeros_like(T)
    seed_vec = np.array([1.0, 0.0, 0.0])
    N[0] = np.cross(T[0], seed_vec)
    n0 = np.linalg.norm(N[0])
    if n0 < 1e-6:
        seed_vec = np.array([0.0, 1.0, 0.0])
        N[0] = np.cross(T[0], seed_vec)
        n0 = np.linalg.norm(N[0])
    N[0] /= n0 + 1e-10
    for i in range(1, len(centreline)):
        proj = N[i - 1] - np.dot(N[i - 1], T[i]) * T[i]
        pn = np.linalg.norm(proj)
        N[i] = proj / pn if pn > 1e-8 else N[i - 1]
    B = np.cross(T, N)
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-10
    return T, N, B


def build_tube(
    centreline: np.ndarray,
    radius: float,
    n_haustra: int,
    n_radial: int = 36,
    add_end_caps: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Tube mesh around the centreline. Same parametric form as
    blender_depth_dataset.build_tube (haustra + 3-lobed tenia coli).

    Returns (vertices, triangles) where triangles is (M, 3) int indices.
    Quads are split into 2 tris each with inward normals.
    """
    n_axial = len(centreline)
    _, N, B = parallel_transport_frames(centreline)
    theta = np.linspace(0, 2 * np.pi, n_radial, endpoint=False)
    t_p = np.linspace(0, 1, n_axial)
    hf = n_haustra * 2 * np.pi
    h = 1.0 + 0.22 * np.cos(hf * t_p) + 0.07 * np.cos(hf * 3 * t_p + 1.0)
    h -= 0.05 * np.clip(-np.cos(hf * t_p), 0, 1) ** 2

    tenia = 0.025 * np.cos(3 * theta) ** 6
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    verts = np.empty((n_axial * n_radial, 3))
    for i in range(n_axial):
        c, n_, b_ = centreline[i], N[i], B[i]
        r = radius * h[i]
        rj = r - tenia * radius
        # ring positions: (n_radial, 3)
        ring = (
            c[None, :]
            + rj[:, None] * (cos_th[:, None] * n_[None, :] + sin_th[:, None] * b_[None, :])
        )
        verts[i * n_radial:(i + 1) * n_radial] = ring

    tris: list[list[int]] = []
    for i in range(n_axial - 1):
        for j in range(n_radial):
            jn = (j + 1) % n_radial
            v00 = i * n_radial + j
            v01 = i * n_radial + jn
            v10 = (i + 1) * n_radial + j
            v11 = (i + 1) * n_radial + jn
            # Inward-facing normals: tri winding goes around the quad in the
            # opposite sense from a CCW-from-outside loop.
            tris.append([v00, v11, v01])
            tris.append([v00, v10, v11])

    if add_end_caps:
        # Cap fan at start: normal should point into the tube (+tangent at s=0).
        start_center = len(verts)
        verts = np.vstack([verts, centreline[0][None, :]])
        for j in range(n_radial):
            jn = (j + 1) % n_radial
            v0 = j
            v1 = jn
            tris.append([start_center, v0, v1])
        # Cap fan at end: normal should point into the tube (-tangent at s=L).
        end_center = len(verts)
        verts = np.vstack([verts, centreline[-1][None, :]])
        for j in range(n_radial):
            jn = (j + 1) % n_radial
            v0 = (n_axial - 1) * n_radial + j
            v1 = (n_axial - 1) * n_radial + jn
            tris.append([end_center, v1, v0])

    return verts, np.array(tris, dtype=np.int32)


# ---------------------------------------------------------------------------
# Validity check (reject seeds that self-intersect)
# ---------------------------------------------------------------------------

def centreline_min_curvature_radius(centreline: np.ndarray) -> float:
    """Minimum radius of curvature along the centreline (in metres). Smaller
    = sharper bend. If smaller than the tube radius, the tube self-intersects."""
    T = np.gradient(centreline, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-10
    dT = np.gradient(T, axis=0)
    ds = np.linalg.norm(np.gradient(centreline, axis=0), axis=1)
    kappa = np.linalg.norm(dT, axis=1) / (ds + 1e-12)
    return float(1.0 / (kappa.max() + 1e-12))


def is_valid(centreline: np.ndarray, tube_radius: float, margin: float = 1.0) -> bool:
    """True if min curvature radius >= margin * tube_radius. Margin=1.0 is
    the strict self-intersection boundary; the original Blender dataset
    script doesn't check at all, so even 1.0 is more conservative."""
    return centreline_min_curvature_radius(centreline) >= margin * tube_radius


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------

MIN_CURVATURE_MARGIN = 1.5
# Safety cap; generated physical centrelines should already equal TASK_LENGTH_M.
MAX_COLON_ARC_M = TASK_LENGTH_M + 1e-6
MAX_GENERATION_ATTEMPTS = 100


def generate_colon(seed: int) -> dict:
    """Generate visual + collision meshes + centreline for the given seed.

    Retries with derived seeds (deterministic) until the generated centreline
    satisfies is_valid(margin=MIN_CURVATURE_MARGIN) AND its arc length is
    within MAX_COLON_ARC_M, or for up to MAX_GENERATION_ATTEMPTS attempts
    (after which the closest valid-looking candidate is used).

    Returns a dict with: seed, ctrl, tube_radius, n_haustra, centreline,
    visual_centreline, visual_vertices, visual_triangles, collision_vertices,
    collision_triangles. The returned centreline is the physical/task
    centreline, not the visual-only distal tail."""
    retry_rng = np.random.RandomState(seed ^ 0xDEAD)
    internal_seed = seed
    ctrl = radius = n_haustra = centreline = None
    best_candidate = None
    best_score = float("inf")
    for _ in range(MAX_GENERATION_ATTEMPTS):
        ctrl, radius, n_haustra = random_ctrl_points(internal_seed)
        task_centreline = centreline_from_ctrl(ctrl, VISUAL_N_AXIAL)
        centreline_arc = float(np.sum(np.linalg.norm(np.diff(task_centreline, axis=0), axis=1)))
        curvature_ratio = centreline_min_curvature_radius(task_centreline) / max(radius, 1e-12)
        curve_deficit = max(0.0, MIN_CURVATURE_MARGIN - curvature_ratio) / MIN_CURVATURE_MARGIN
        length_excess = max(0.0, centreline_arc - MAX_COLON_ARC_M) / MAX_COLON_ARC_M
        score = 10.0 * curve_deficit + length_excess
        if score < best_score:
            best_score = score
            best_candidate = (ctrl, radius, n_haustra, task_centreline)
        if curve_deficit == 0.0 and length_excess == 0.0:
            centreline = task_centreline
            break
        internal_seed = int(retry_rng.randint(1, 2**31 - 1))
    else:
        ctrl, radius, n_haustra, centreline = best_candidate

    visual_centreline = append_straight_distal_lumen(
        centreline,
        extension_m=DISTAL_STRAIGHT_EXTENSION_M,
    )

    visual_v, visual_t = build_tube(
        visual_centreline, radius, n_haustra,
        n_radial=VISUAL_N_RADIAL, add_end_caps=True,
    )

    # Sub-sample the centreline axially for the collision mesh.
    # No end caps on collision -- otherwise the scope body extending back
    # along -tangent from the entrance hits the cap and the scope can't enter.
    coll_idx = np.linspace(0, len(centreline) - 1, COLLISION_N_AXIAL).astype(int)
    coll_centreline = centreline[coll_idx]
    coll_v, coll_t = build_tube(
        coll_centreline, radius, n_haustra,
        n_radial=COLLISION_N_RADIAL, add_end_caps=False,
    )

    return {
        "seed": int(seed),
        "ctrl": ctrl,
        "tube_radius": float(radius),
        "n_haustra": int(n_haustra),
        "centreline": centreline,
        "visual_centreline": visual_centreline,
        "visual_vertices": visual_v,
        "visual_triangles": visual_t,
        "collision_vertices": coll_v,
        "collision_triangles": coll_t,
    }


# ---------------------------------------------------------------------------
# Binary STL writer (no external deps)
# ---------------------------------------------------------------------------

def write_binary_stl(vertices: np.ndarray, triangles: np.ndarray, path: Path) -> None:
    """Write triangles as binary STL: 80B header + uint32 count + 50B/tri."""
    n_tri = len(triangles)
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", n_tri))
        for tri_idx in triangles:
            v0 = vertices[tri_idx[0]].astype(np.float32)
            v1 = vertices[tri_idx[1]].astype(np.float32)
            v2 = vertices[tri_idx[2]].astype(np.float32)
            normal = np.cross(v1 - v0, v2 - v0)
            n_len = float(np.linalg.norm(normal))
            if n_len > 1e-12:
                normal = (normal / n_len).astype(np.float32)
            else:
                normal = np.zeros(3, dtype=np.float32)
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate a procedural random colon and write visual + collision STLs."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_dir", type=Path,
        default=Path(__file__).parent / "scenes",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    colon = generate_colon(args.seed)
    visual_stl = args.out_dir / f"colon_visual_seed{colon['seed']}.stl"
    coll_stl = args.out_dir / f"colon_collision_seed{colon['seed']}.stl"
    write_binary_stl(colon["visual_vertices"], colon["visual_triangles"], visual_stl)
    write_binary_stl(colon["collision_vertices"], colon["collision_triangles"], coll_stl)

    print(f"Seed:           {colon['seed']}")
    print(f"Tube radius:    {colon['tube_radius'] * 1000:.1f} mm")
    print(f"n_haustra:      {colon['n_haustra']}")
    print(
        f"Centreline:     {len(colon['centreline'])} pts, "
        f"z {colon['centreline'][:, 2].min() * 1000:.0f}..{colon['centreline'][:, 2].max() * 1000:.0f} mm"
    )
    task_arc = float(np.sum(np.linalg.norm(np.diff(colon["centreline"], axis=0), axis=1)))
    visual_arc = float(np.sum(np.linalg.norm(np.diff(colon["visual_centreline"], axis=0), axis=1)))
    print(
        f"Rendered length: {visual_arc * 1000:.0f} mm "
        f"(task/collision {task_arc * 1000:.0f} mm + visual tail "
        f"{(visual_arc - task_arc) * 1000:.0f} mm)"
    )
    print(
        f"Visual mesh:    {len(colon['visual_triangles'])} tris "
        f"({len(colon['visual_vertices'])} verts) -> {visual_stl.name}"
    )
    print(
        f"Collision mesh: {len(colon['collision_triangles'])} tris "
        f"({len(colon['collision_vertices'])} verts) -> {coll_stl.name}"
    )
    min_curv_r = centreline_min_curvature_radius(colon["centreline"]) * 1000
    print(
        f"Min curvature R: {min_curv_r:.1f} mm  "
        f"(margin vs tube radius: {min_curv_r / (colon['tube_radius'] * 1000):.2f}x)"
    )


if __name__ == "__main__":
    main()

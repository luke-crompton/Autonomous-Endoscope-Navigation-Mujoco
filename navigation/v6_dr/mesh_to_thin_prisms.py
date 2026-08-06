"""
STL -> MuJoCo thin-prism collision geoms.

For each triangle in a binary STL, emit a <geom type="box"> whose plane
contains the triangle and whose thickness extends along the triangle's normal.
Adjacent triangles' boxes overlap slightly where they share an edge -- this
is intentional and harmless for collision (just double-thickness in the
overlap, no gaps) and avoids the convex-hull problem that <geom type="mesh">
has on hollow geometry.

CLI:
    python mesh_to_thin_prisms.py path/to/colon_mujoco.stl \
        --thickness 0.001 --out colon.geoms.xml
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def read_binary_stl(stl_path: Path) -> np.ndarray:
    """Return (N, 3, 3) float32 array of triangle vertices (metres).

    Binary STL layout: 80B header, uint32 n_tri, then n_tri * 50B records
    (12B normal + 3*12B vertices + 2B attribute).
    """
    data = Path(stl_path).read_bytes()
    n_tri = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + n_tri * 50
    if len(data) < expected_size:
        raise ValueError(
            f"STL truncated: expected >= {expected_size} bytes for {n_tri} "
            f"triangles, got {len(data)}"
        )
    tris = np.empty((n_tri, 3, 3), dtype=np.float32)
    for i in range(n_tri):
        verts = struct.unpack_from("<9f", data, 84 + i * 50 + 12)
        tris[i] = np.array(verts, dtype=np.float32).reshape(3, 3)
    return tris


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> MuJoCo quat [w, x, y, z] via Shepperd's method."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def triangle_to_box_params(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, thickness_m: float
):
    """Return (pos, quat[w,x,y,z], halfsize[hx,hy,hz]) for a thin OBB that
    covers the triangle, or None for degenerate triangles.

    Local frame: +Z = triangle normal, +X = e1 direction (v0->v1 normalised
    and orthogonalised), +Y = Z x X. Box halfsizes (hx, hy) are the tightest
    extents of the triangle's vertices about the centroid along (X, Y).
    """
    v0 = np.asarray(v0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    centroid = (v0 + v1 + v2) / 3.0
    e1 = v1 - v0
    e2 = v2 - v0
    normal = np.cross(e1, e2)
    n_len = np.linalg.norm(normal)
    if n_len < 1e-12:
        return None
    z_axis = normal / n_len
    e1_len = np.linalg.norm(e1)
    if e1_len < 1e-12:
        return None
    x_axis = e1 / e1_len
    # Re-orthogonalise X against Z (cheap insurance against floating-point drift).
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_len = np.linalg.norm(x_axis)
    if x_len < 1e-12:
        return None
    x_axis /= x_len
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    verts_local = (np.stack([v0, v1, v2]) - centroid) @ R
    hx = float(np.max(np.abs(verts_local[:, 0])))
    hy = float(np.max(np.abs(verts_local[:, 1])))
    hz = float(thickness_m) / 2.0
    # Guard against zero-sized boxes (MuJoCo rejects them).
    hx = max(hx, 1e-6)
    hy = max(hy, 1e-6)
    quat = rotation_matrix_to_quat(R)
    return centroid, quat, np.array([hx, hy, hz], dtype=np.float64)


def emit_triangle_prism_meshes(
    stl_path: Path,
    thickness_m: float = 0.0001,
    *,
    contype: int = 1,
    conaffinity: int = 1,
    rgba: str = "0.92 0.58 0.55 1",
    asset_indent: str = "    ",
    geom_indent: str = "      ",
    name_prefix: str = "wall_tri",
) -> tuple[str, str, int]:
    """Emit one MuJoCo <mesh> asset + one <geom type='mesh'> per triangle.

    Each triangle is extruded along its normal by ``thickness_m`` (half each
    side of the triangle plane), giving 6 vertices that MuJoCo treats as a
    convex triangular prism. Adjacent triangles' prisms share edges and side
    faces exactly, so the wall renders as the original mesh (no overhang
    shards) while still providing full per-triangle collision coverage.

    Returns (asset_xml_lines, geom_xml_lines, n_emitted). Both strings already
    have newlines between entries. Degenerate triangles are skipped.
    """
    tris = read_binary_stl(stl_path)
    asset_lines: list[str] = []
    geom_lines: list[str] = []
    n_emitted = 0
    half_t = float(thickness_m) / 2.0
    for i, tri in enumerate(tris):
        v0 = np.asarray(tri[0], dtype=np.float64)
        v1 = np.asarray(tri[1], dtype=np.float64)
        v2 = np.asarray(tri[2], dtype=np.float64)
        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2)
        n_len = np.linalg.norm(normal)
        if n_len < 1e-12:
            continue
        normal /= n_len
        offset = half_t * normal
        verts = np.stack([v0 - offset, v1 - offset, v2 - offset,
                          v0 + offset, v1 + offset, v2 + offset])
        verts_str = "  ".join(
            f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in verts
        )
        name = f"{name_prefix}_{i}"
        asset_lines.append(
            f'{asset_indent}<mesh name="{name}" vertex="{verts_str}"/>'
        )
        geom_lines.append(
            f'{geom_indent}<geom type="mesh" mesh="{name}" '
            f'contype="{contype}" conaffinity="{conaffinity}" '
            f'rgba="{rgba}" mass="0"/>'
        )
        n_emitted += 1
    return "\n".join(asset_lines), "\n".join(geom_lines), n_emitted


def emit_thin_prisms(
    stl_path: Path,
    thickness_m: float = 0.001,
    *,
    contype: int = 1,
    conaffinity: int = 1,
    rgba: str = "0.92 0.58 0.55 1",
    indent: str = "      ",
    name_prefix: str = "wall_tri",
) -> tuple[str, int]:
    """Read STL, return (geoms_xml_string, n_emitted).

    OBB approximation: each triangle becomes an oriented bounding box centred
    on its plane. Boxes overhang triangle edges -- visually shard-like but
    collision-wise continuous. See ``emit_triangle_prism_meshes`` for a
    mesh-accurate alternative.

    Degenerate triangles (zero-area or zero-length edge) are skipped.
    """
    tris = read_binary_stl(stl_path)
    lines: list[str] = []
    n_emitted = 0
    for i, tri in enumerate(tris):
        result = triangle_to_box_params(tri[0], tri[1], tri[2], thickness_m)
        if result is None:
            continue
        pos, quat, size = result
        lines.append(
            f'{indent}<geom name="{name_prefix}_{i}" type="box" '
            f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
            f'quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}" '
            f'size="{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}" '
            f'contype="{contype}" conaffinity="{conaffinity}" '
            f'rgba="{rgba}" mass="0"/>'
        )
        n_emitted += 1
    return "\n".join(lines), n_emitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("stl", type=Path, help="Path to binary STL file")
    parser.add_argument(
        "--thickness", type=float, default=0.001,
        help="Wall thickness in metres (default 0.001 = 1 mm)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output XML fragment path (default: <stl>.geoms.xml)",
    )
    parser.add_argument(
        "--rgba", default="0.92 0.58 0.55 1",
        help='Geom rgba (default "0.92 0.58 0.55 1")',
    )
    args = parser.parse_args()

    xml, n = emit_thin_prisms(args.stl, args.thickness, rgba=args.rgba)
    target = args.out if args.out else args.stl.with_suffix(".geoms.xml")
    target.write_text(xml + "\n", encoding="utf-8")
    print(f"Read STL:    {args.stl}")
    print(f"Wrote {n} thin-prism <geom> elements to: {target}")


if __name__ == "__main__":
    main()

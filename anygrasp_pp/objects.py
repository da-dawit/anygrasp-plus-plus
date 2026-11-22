"""Hidden-weight objects from scanned meshes: voxelize the convex collision pieces, then fill one end cap with a
dense core so the CoM moves by s along the long axis. The cap is picked once per object (for the largest shift),
so the object looks the same at every shift and only the hidden densities change.
"""
import glob
import os

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree

VOXEL = 0.002            # m
WALL = 0.002             # skin of shell material around the core [m]
RHO_MAX = 19300.0        # tungsten [kg/m^3], the densest practical filler
MASS = 1.0               # [kg]
SHIFTS = (0.0, 0.02, 0.04, 0.06, 0.08)
MIN_SHELL = 0.10         # a real housing weighs something [kg]


def read_obj_vertices(path):
    v = [line.split()[1:4] for line in open(path) if line.startswith("v ")]
    return np.asarray(v, dtype=float)


def collision_dir(model_dir, collision="vhacd"):
    """Folder with the convex pieces: the scan's own V-HACD pieces, or a regenerated set (e.g. "coacd")."""
    return model_dir if collision == "vhacd" else os.path.join(model_dir, collision)


def convex_pieces(model_dir, collision="vhacd"):
    d = collision_dir(model_dir, collision)
    paths = sorted(glob.glob(os.path.join(d, "model_collision_*.obj")),
                   key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    return [read_obj_vertices(p) for p in paths]


def occupancy(pieces, voxel=VOXEL):
    """Boolean grid of voxel centres inside the union of convex pieces. Returns (occ, origin): centre of voxel
    (i, j, k) is origin + voxel * (i, j, k)."""
    allv = np.vstack(pieces)
    lo, hi = allv.min(0) - voxel, allv.max(0) + voxel
    shape = np.ceil((hi - lo) / voxel).astype(int) + 1
    occ = np.zeros(shape, bool)
    for v in pieces:
        if len(v) < 4:
            continue
        try:
            hull = ConvexHull(v)
        except Exception:  # degenerate (flat) piece has no interior
            continue
        a, b = hull.equations[:, :3], hull.equations[:, 3]
        i0 = np.floor((v.min(0) - lo) / voxel).astype(int)
        i1 = np.ceil((v.max(0) - lo) / voxel).astype(int) + 1
        grids = np.meshgrid(*[np.arange(i0[k], i1[k]) for k in range(3)], indexing="ij")
        idx = np.stack([g.ravel() for g in grids], 1)
        pts = lo + voxel * idx
        inside = (pts @ a.T + b <= 1e-9).all(1)
        occ[tuple(idx[inside].T)] = True
    return occ, lo


def principal_frame(points):
    """Centroid and principal axes (rows, long axis first) of a uniform solid given by its voxel centres."""
    c = points.mean(0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    if np.linalg.det(vt) < 0:
        vt[2] *= -1                       # right-handed frame
    return c, vt


def core_masses(s, VF, VS, xF, M=MASS):
    """Core mass for CoM shift s along the long axis: M (s VS / xF + VF) / V, xF measured from the solid's
    centroid. Shell mass is M minus it."""
    return M * (s * VS / xF + VF) / (VF + VS)


def choose_core(x, eroded, voxel=VOXEL, s_max=max(SHIFTS), M=MASS, rho_max=RHO_MAX, min_shell=MIN_SHELL):
    """Outermost end cap on either side that can make an s_max shift. x: long-axis coordinate of every solid voxel
    (from the solid centroid); eroded: which solid voxels are at least WALL inside. Returns a dict or None."""
    v = voxel ** 3
    V = len(x) * v
    best = None
    for sign in (1, -1):
        xs = sign * x
        cand = np.sort(np.unique(np.round(xs[eroded] / voxel)))[::-1] * voxel   # cut positions, outermost first
        for cut in cand:
            if cut <= 0:
                break
            F = eroded & (xs >= cut - 1e-12)
            if not F.any():                   # rounding put the cut past the last voxel
                continue
            VF = F.sum() * v
            xF = xs[F].mean()
            mf = core_masses(s_max, VF, V - VF, xF, M)
            if mf / VF <= rho_max and M - mf >= min_shell:
                if best is None or cut > best["cut"]:
                    best = dict(sign=sign, cut=cut, F=F, VF=VF, VS=V - VF, xF=xF)
                break
    return best


def mass_properties(points, core, s, voxel=VOXEL, M=MASS):
    """Mass, CoM and inertia about the CoM (3x3) of the object at shift s, from voxels (cube self-inertia
    included). points: solid voxel centres in the object frame."""
    v = voxel ** 3
    mf = core_masses(s, core["VF"], core["VS"], core["xF"], M)
    rho = np.where(core["F"], mf / core["VF"], (M - mf) / core["VS"])
    m = rho * v
    com = (m[:, None] * points).sum(0) / m.sum()
    r = points - com
    inertia = (m * (r ** 2).sum(1)).sum() * np.eye(3) - (m[:, None, None] * r[:, :, None] * r[:, None, :]).sum(0)
    inertia += m.sum() * voxel ** 2 / 6 * np.eye(3)
    return dict(mass=m.sum(), com=com, inertia=inertia, core_mass=mf, core_density=mf / core["VF"],
                shell_density=(M - mf) / core["VS"])


def fidelity(occ, origin, visual_vertices, voxel=VOXEL, n=60000, seed=0, mesh=None):
    """How well the collision solid matches the visual surface the camera sees.
    bulge: distance from collision-solid boundary voxels to the nearest visual surface point (phantom material)
    gap:   distance from visual surface points to the nearest collision-solid voxel (surface the fingers pass through)
    Both as (95th percentile, max) in metres."""
    if mesh is not None:
        import trimesh
        surf = trimesh.sample.sample_surface(mesh, n, seed=seed)[0]
    else:
        surf = visual_vertices
    boundary = np.argwhere(occ & ~ndimage.binary_erosion(occ))
    bulge = cKDTree(surf).query(origin + voxel * boundary)[0]
    gap = cKDTree(origin + voxel * np.argwhere(occ)).query(surf)[0]
    return dict(bulge95=np.percentile(bulge, 95), bulge_max=bulge.max(),
                gap95=np.percentile(gap, 95), gap_max=gap.max())


def parts(occ, bridge=1, min_frac=0.02):
    """Number of separate pieces making up at least min_frac of the solid each, after closing gaps of up to
    2*bridge voxels (convex pieces of one object can leave hairline gaps). A set of loose items (a furniture set,
    game pieces) has several; a single rigid object has one."""
    lab, n = ndimage.label(ndimage.binary_dilation(occ, iterations=bridge), structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab[occ])[1:]
    return int((sizes >= min_frac * occ.sum()).sum())


def build(model_dir, voxel=VOXEL, wall=WALL, collision="vhacd", mass=MASS):
    """Everything about one object: solid voxels, principal frame, heavy-end core, properties per shift."""
    occ, origin = occupancy(convex_pieces(model_dir, collision), voxel)
    idx = np.argwhere(occ)
    points = origin + voxel * idx
    c, axes = principal_frame(points)
    x = (points - c) @ axes[0]
    k = max(1, int(round(wall / voxel)))
    eroded = ndimage.binary_erosion(occ, iterations=k)[tuple(idx.T)]
    core = choose_core(x, eroded, voxel, M=mass)
    out = dict(occ=occ, origin=origin, points=points, centroid=c, axes=axes, core=core,
               length=np.ptp(x), volume=len(points) * voxel ** 3, parts=parts(occ))
    if core is not None:
        e = core["sign"] * axes[0]                   # unit vector toward the heavy end
        out["heavy_axis"] = e
        out["props"] = {s: mass_properties(points, core, s, voxel, M=mass) for s in SHIFTS}
        out["shift_check"] = {s: float((out["props"][s]["com"] - c) @ e) for s in SHIFTS}
    return out

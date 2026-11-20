"""M2: score a grasp before touching the object (proposal M2; decisions D9, D22; derivations C.6).

Two predictions from what the robot can observe (the D435 cloud, the grasp, the robot model):

1. the contact patch, by a "virtual squeeze" of the rubber-layer pad (B.6) against the observed surface:
   per fingertip, the cloud points in the pad's 17 x 17 mm footprint are expressed in the hand frame, the surface
   distance along the closing axis is fitted by least squares (the cloud is noisier than the pad's sink), the pad
   stops at the outermost surface point, every grid cell with observed material engages by its gap (Winkler), and
   the patch's twist capacity (A.12) gives this grasp's slip distance d*_pred = (T_left + T_right) / W (A.5);
2. the centre of mass, from the visible shape: the middle of the object's footprint, with the uncertainty of a
   uniform prior over the footprint (C.5) -- before touching it, nothing tells the robot where a hidden weight is.

Score = detector score x P(hold), P(hold) = P(|d| < d*_pred) under the CoM prior (C.5, D9).
"""
import numpy as np

from . import physics as P
from .contact_rig import GRID_N, PAD_HALF, grid_points

K_CELL = 3.0e6 * (2 * PAD_HALF / GRID_N) ** 2 / 0.002      # B.6: E A_cell / h
FN = 70.0
MIN_PTS = 12              # fewer points than this in a pad's footprint: that side is treated as unseen
CELL_SUPPORT = 0.003      # a grid cell has material if an observed point lies within this of it (pad plane) [m]
TABLE_CLEAR = 0.004       # cloud points this close to the table belong to the table, not the object [m]


def winkler_forces(gaps, F=FN, k=K_CELL):
    """Forces of a Winkler bed pressed with total F against gaps (inf = no material). B.6."""
    gaps = np.asarray(gaps, float)
    ok = np.isfinite(gaps)
    if not ok.any():
        return np.zeros_like(gaps)
    g = gaps[ok]
    lo, hi = g.min(), g.min() + F / k + 1e-3
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (k * np.maximum(0, mid - g)).sum() < F else (lo, mid)
    f = np.zeros_like(gaps)
    f[ok] = k * np.maximum(0, (lo + hi) / 2 - g)
    return f


def _fit_surface(xz, s):
    """Least-squares surface s(x, z) over the footprint: quadratic if there are enough points, else a plane."""
    x, z = xz[:, 0], xz[:, 1]
    if len(s) >= 20:
        A = np.stack([np.ones_like(x), x, z, x * x, z * z, x * z], 1)
    else:
        A = np.stack([np.ones_like(x), x, z], 1)
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    def f(q):
        q = np.atleast_2d(q)
        X = [np.ones(len(q)), q[:, 0], q[:, 1]] + ([q[:, 0] ** 2, q[:, 1] ** 2, q[:, 0] * q[:, 1]] if len(coef) == 6 else [])
        return np.stack(X, 1) @ coef
    return f


def pad_patch(pts_hand, side, w_open, cell_world_z=None, table_z=0.0):
    """Predicted per-cell forces on one fingertip. pts_hand: cloud in the hand frame (x across, y closing, z
    approach); side: +1 or -1 (which finger); the pad centre is at (0, side*y, 0). Returns (forces (16,), n_pts)."""
    grid = grid_points()                                  # (x, z_up) on the pad face
    x, y, z = pts_hand[:, 0], pts_hand[:, 1], pts_hand[:, 2]
    zu = -z                                               # pad "up" is against the approach
    s = side * y                                          # distance from the centre line toward this pad
    sel = (np.abs(x) <= PAD_HALF + CELL_SUPPORT) & (np.abs(zu) <= PAD_HALF + CELL_SUPPORT) & (s > 0) & (s <= w_open / 2)
    if sel.sum() < MIN_PTS:
        return None, int(sel.sum())
    xz, sv = np.stack([x[sel], zu[sel]], 1), s[sel]
    f = _fit_surface(xz, sv)
    # a cell has material if it is within the observed footprint across the pad, not above the object's observed
    # top edge, and above the table (side faces are seen at grazing angles, so the cloud there is sparse)
    m_ = CELL_SUPPORT
    has = ((grid[:, 0] >= xz[:, 0].min() - m_) & (grid[:, 0] <= xz[:, 0].max() + m_)
           & (grid[:, 1] <= xz[:, 1].max() + m_))
    if cell_world_z is not None:
        has &= cell_world_z > table_z + 0.002
    surf = f(grid)
    if not has.any():
        return None, int(sel.sum())
    gaps = np.where(has, surf[has].max() - surf, np.inf)  # the pad stops at the outermost surface point
    return winkler_forces(gaps), int(sel.sum())


def predicted_dstar(pts_world, p_tcp, R_tcp, w_open, mass, mu=P.MU, accel=0.0):
    """This grasp's predicted slip distance [m] from the cloud (C.6). Unseen sides mirror the seen one; if neither
    pad sees enough of the surface the grasp gets the one-point (worst) value 0."""
    q = (pts_world - p_tcp) @ R_tcp
    W = mass * (P.G + accel)
    g = grid_points()
    cell_z = p_tcp[2] + g[:, 0] * R_tcp[2, 0] - g[:, 1] * R_tcp[2, 2]   # world height of each pad cell
    fl, nl = pad_patch(q, +1, w_open, cell_z)
    fr, nr = pad_patch(q, -1, w_open, cell_z)
    if fl is None and fr is None:
        return 0.0, dict(n_left=nl, n_right=nr, mirrored=False)
    mirrored = fl is None or fr is None
    fl = fr if fl is None else fl
    fr = fl if fr is None else fr
    grid = grid_points()
    T = 0.0
    for fpad in (fl, fr):
        on = fpad > 1e-6
        if on.sum() == 0:
            return 0.0, dict(n_left=nl, n_right=nr, mirrored=mirrored)
        T += P.patch_torque_capacity(grid[on], W / 2, mu, fpad.sum(), weights=fpad[on])
    return T / W, dict(n_left=nl, n_right=nr, mirrored=mirrored, pts_left=int((fl > 1e-6).sum()),
                       pts_right=int((fr > 1e-6).sum()))


def object_points(pts_world, table_z=0.0):
    """The object's points: everything clearly above the table in the workspace crop."""
    return pts_world[pts_world[:, 2] > table_z + TABLE_CLEAR]


def com_prior(obj_pts):
    """Uniform CoM prior over the visible footprint (C.5), in the form physics.p_hold_uniform takes: centre c0
    (middle of the footprint along its principal axes, at half the visible height), V (3, 2) the footprint's
    horizontal principal axes as columns, and the half extents (2,) along them."""
    xy = obj_pts[:, :2]
    c = xy.mean(0)
    _, _, vt = np.linalg.svd(xy - c, full_matrices=False)
    loc = (xy - c) @ vt.T
    lo, hi = loc.min(0), loc.max(0)
    mid = c + ((lo + hi) / 2) @ vt
    half = (hi - lo) / 2
    zmax = obj_pts[:, 2].max()
    c0 = np.array([mid[0], mid[1], zmax / 2])
    V = np.zeros((3, 2))
    V[:2, 0], V[:2, 1] = vt[0], vt[1]
    return c0, V, half

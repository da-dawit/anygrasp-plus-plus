"""Two-finger grasp physics (docs/derivations.tex, sections A and C). General: any grasp pose, object, mass, friction.

Model assumptions (all of them):
  * rigid object, two opposing pads with equal normal force fn, quasi-static lift;
  * gravity plus a vertical lift acceleration is the only load;
  * closing axis close to horizontal (decisions.md D3: approach within 30 deg of vertical), so the weight is
    carried by friction, not by a pad's normal force;
  * soft-finger pad: uniform pressure over a disc of radius a, combined sliding + twisting limited by the
    Howe-Cutkosky ellipse (A.7), an outer approximation that is at most ~0.2% optimistic at nominal values;
  * patch pad (patch_dstar): equal load on a set of contact points, exact rigid-sliding limit surface (A.12).
Every function is vectorized over N grasps.

Shapes: p (N,3) grasp centres; axis (N,3) or (3,) closing axes (any non-zero length; normalized here);
        c (3,) CoM estimate; cov (3,3) its covariance. mu, fn, a, m, accel: scalars or arrays of shape (N,).
"""
import numpy as np

from scipy.stats import norm

G = 9.81
EZ = np.array([0.0, 0.0, 1.0])

# Nominal values for THIS study (docs/derivations.tex A.1). Defaults only; pass your own for anything else.
MU = 0.5          # pad-object friction
FN = 70.0         # normal force per pad [N] (Franka Hand continuous grasp force)
A_PAD = 0.006     # soft-finger contact patch radius [m] (point-pad model only)
MASS = 0.6        # object mass used in the hidden-weight experiment [kg]
MAX_TILT_DEG = 30.0   # grasps allowed: approach within this angle of straight down (decisions.md D3)


# ---------------------------------------------------------------------------------------------- helpers
def _unit_axes(axis):
    axis = np.atleast_2d(np.asarray(axis, float))
    n = np.linalg.norm(axis, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("closing axis of zero length")
    return axis / n


def _per_grasp(x, n, name):
    """Scalar or shape-(n,) array; anything else (e.g. (n,1)) would broadcast wrongly, so refuse it."""
    x = np.asarray(x, float)
    if x.ndim == 0 or x.shape == (n,):
        return x
    raise ValueError(f"{name} must be a scalar or shape ({n},), got {x.shape}")


def approach_ok(approach, max_deg=MAX_TILT_DEG):
    """True where the approach direction is within max_deg of straight down (decisions.md D3).
    Then the closing axis (perpendicular to the approach) is within max_deg of horizontal."""
    ap = _unit_axes(approach)
    return ap @ -EZ >= np.cos(np.radians(max_deg)) - 1e-12


# ---------------------------------------------------------------------------------------------- A: statics
def torsional_mu(mu=MU, a=A_PAD):
    """MuJoCo torsional friction (a length) for a uniform-pressure disc pad: mu3 = 2/3 mu a  (A.4, B.1)."""
    return 2.0 / 3.0 * np.asarray(mu) * a


def hinge_vector(axis):
    """h = e_z x a_hat. Horizontal, perpendicular to the closing axis; |h| = cos(tilt of the axis) (A.2, A.3)."""
    return np.cross(EZ, _unit_axes(axis))


def lever_arm(p, axis, c):
    """Signed lever arm d = (c - p) . h of gravity about the closing axis (A.2). |d| twists the object."""
    return np.einsum("ij,ij->i", np.atleast_2d(c) - np.atleast_2d(p), hinge_vector(axis))


def dstar(mu=MU, fn=FN, a=A_PAD, m=MASS, accel=0.0, g=G, exact=True):
    """Largest |d| a soft-finger pad pair can hold (A.8). accel = upward lift acceleration (A.6). Broadcasts.

    Returns 0 where the object would slide straight down (m (g + accel) >= 2 mu fn), in both forms."""
    mu, fn, a, m, accel = np.broadcast_arrays(*map(np.asarray, (mu, fn, a, m, accel)))
    w = m * (g + accel)                                    # effective weight
    x = w / (2.0 * mu * fn)
    base = 4.0 * mu * fn * a / (3.0 * w)
    out = base * np.sqrt(np.clip(1.0 - x**2, 0.0, None)) if exact else base
    return np.where(x < 1.0, out, 0.0)


def required_grip(d, m, mu=MU, a=A_PAD, accel=0.0, g=G, margin=1.0):
    """Smallest per-pad normal force that holds lever arm d with a soft-finger pad (A.10), times a margin.

    fn_min = (m g_eff / mu) * sqrt(1/4 + (3 d / (4 a))^2)."""
    w = np.asarray(m) * (g + np.asarray(accel))
    return margin * (w / mu) * np.sqrt(0.25 + (3.0 * np.abs(d) / (4.0 * a)) ** 2)


def patch_torque_capacity(points, f_up, mu=MU, fn=FN, weights=None):
    """Largest twisting moment one pad can resist while also pushing up with f_up (A.12).

    points: (K,2) contact points (x along the long axis, z up) relative to the pad centre; each carries normal
    load n_k = fn * weights_k (equal by default) and Coulomb friction |f_k| <= mu n_k. Exact, by convex duality:
        T_max = min over c in R^2 of  sum_k mu n_k |p_k - c|  +  c_x f_up,
    where the minimizing c is the centre the object would turn about when it starts to slip. Works for any
    point set (full face, a line of points, off-centre patches)."""
    pts = np.asarray(points, float)
    w = np.full(len(pts), 1.0 / len(pts)) if weights is None else np.asarray(weights, float) / np.sum(weights)
    if f_up >= mu * fn * (1 - 1e-12):
        return 0.0
    from scipy.optimize import minimize
    obj = lambda c: mu * fn * (w * np.linalg.norm(pts - c, axis=1)).sum() + c[0] * f_up
    # Weiszfeld-type fixed point: the gradient sum_k w_k (c - p_k)/|c - p_k| + (f_up/(mu fn), 0) = 0 gives
    # c = (sum_k w_k p_k/|c - p_k| - (f_up/(mu fn), 0)) / sum_k w_k/|c - p_k|. Convex objective -> global minimum.
    g = np.array([f_up / (mu * fn), 0.0])
    span = np.abs(pts).max() + 1e-4
    c = np.array([-span * f_up / (mu * fn), 0.0])
    for _ in range(5000):
        dist = np.maximum(np.linalg.norm(pts - c, axis=1), 1e-12)
        q = w / dist
        c_new = (q @ pts - g) / q.sum()
        if np.linalg.norm(c_new - c) < 1e-14:
            c = c_new
            break
        c = c_new
    # The minimum can sit exactly on a contact point (e.g. a line of points), where Weiszfeld slows down: also try
    # every contact point as a candidate, then polish the best one.
    cands = [c] + (list(pts) if len(pts) <= 2000 else [])
    c = min(cands, key=obj)
    r = minimize(obj, c, method="Nelder-Mead",
                 options=dict(xatol=1e-13, fatol=1e-15, maxiter=2000, initial_simplex=[c, c + [1e-4, 0], c + [0, 1e-4]]))
    return max(min(obj(c), r.fun), 0.0)


def patch_dstar(points, mu=MU, fn=FN, m=MASS, accel=0.0, g=G, weights=None):
    """Largest |d| a pair of identical patch pads can hold (A.12). Each pad pushes up with W/2 and resists
    W d / 2 of twist (equal sharing is optimal, A.5), so d* = 2 T_max(W/2) / W. Returns 0 if W/2 >= mu fn."""
    w = m * (g + accel)
    if w / 2 >= mu * fn:
        return 0.0
    return 2.0 * patch_torque_capacity(points, w / 2, mu, fn, weights) / w


# ---------------------------------------------------------------------------------------------- C: scoring
def p_hold(p, axis, c, cov, m, mu=MU, fn=FN, a=A_PAD, accel=0.0, ds=None):
    """Probability each grasp holds, CoM belief Gaussian N(c, cov) (C.2, C.3). Used after the test lift.

    d_i ~ N(mu_d, sigma_d^2), mu_d = (c - p_i).h_i, sigma_d^2 = h_i^T cov h_i;  P = Pr(|d_i| <= d*).
    ds: optional per-grasp d* (e.g. from patch_dstar); default is the soft-finger dstar."""
    h = hinge_vector(axis)
    n = len(np.atleast_2d(p))
    mu_d = np.einsum("ij,ij->i", np.atleast_2d(c) - np.atleast_2d(p), h)
    var = np.einsum("ij,jk,ik->i", h, np.asarray(cov, float), h)
    sig_d = np.sqrt(np.clip(var, 0.0, None))
    if ds is None:
        ds = dstar(*(_per_grasp(v, n, k) for v, k in ((mu, "mu"), (fn, "fn"), (a, "a"), (m, "m"), (accel, "accel"))))
    sig_d = np.maximum(sig_d, 1e-12)                     # point belief -> step function
    return norm.cdf((ds - mu_d) / sig_d) - norm.cdf((-ds - mu_d) / sig_d)


def _uniform_sum_cdf(t, alpha, beta):
    """CDF of X + Y, X ~ U[-alpha, alpha], Y ~ U[-beta, beta] independent (C.5: trapezoid)."""
    t, alpha, beta = np.broadcast_arrays(*map(np.asarray, (t, alpha, beta)))
    hi, lo = np.maximum(alpha, beta), np.minimum(alpha, beta)
    r2 = lambda u: np.maximum(u, 0.0) ** 2
    area = 0.5 * (r2(t + hi + lo) - r2(t - hi + lo) - r2(t + hi - lo) + r2(t - hi - lo))
    with np.errstate(divide="ignore", invalid="ignore"):
        trap = area / (4 * hi * lo)
        unif = np.clip((t + hi) / (2 * hi), 0, 1)              # lo -> 0: uniform on [-hi, hi]
    step = (t >= 0).astype(float)                              # both -> 0: point mass at 0
    out = np.where(lo > 1e-9 * np.maximum(hi, 1e-30), trap, np.where(hi > 0, unif, step))
    return np.clip(out, 0.0, 1.0)


def p_hold_uniform(p, axis, c0, V, half_extents, m, mu=MU, fn=FN, a=A_PAD, accel=0.0, ds=None):
    """Exact P_hold for the pre-touch prior: CoM uniform on a rectangle (C.1, C.5), no Gaussian approximation.

    c0 (3,) rectangle centre; V (3,2) its two horizontal unit axes as columns; half_extents (L/2, W/2)."""
    h = hinge_vector(axis)
    n = len(np.atleast_2d(p))
    mu_d = np.einsum("ij,ij->i", np.atleast_2d(c0) - np.atleast_2d(p), h)
    V = np.asarray(V, float)
    alpha = np.abs(h @ V[:, 0]) * half_extents[0]
    beta = np.abs(h @ V[:, 1]) * half_extents[1]
    if ds is None:
        ds = dstar(*(_per_grasp(v, n, k) for v, k in ((mu, "mu"), (fn, "fn"), (a, "a"), (m, "m"), (accel, "accel"))))
    return _uniform_sum_cdf(ds - mu_d, alpha, beta) - _uniform_sum_cdf(-ds - mu_d, alpha, beta)


def margin(p, axis, c, m, mu=MU, fn=FN, a=A_PAD, accel=0.0):
    """Deterministic margin rho = 1 - |d| / d* (C.4, F.2). Negative = predicted to twist out; -inf if d* = 0."""
    ds = dstar(mu, fn, a, m, accel)
    d = np.abs(lever_arm(p, axis, c))
    with np.errstate(divide="ignore"):
        return np.where(ds > 0, 1.0 - d / np.where(ds > 0, ds, 1.0), -np.inf)


if __name__ == "__main__":
    from anygrasp_pp.contact_rig import grid_points
    print(f"mu3 = {torsional_mu()*1000:.2f} mm")
    print(f"soft-finger d* = {float(dstar())*100:.2f} cm exact, {float(dstar(exact=False))*100:.2f} cm simple")
    gp = grid_points()
    inner = gp[np.abs(gp[:, 1]) < 0.004]                 # the two middle rows: line-like contact on a handle
    print(f"grid pad (4x4 on 17 mm face): mean radius {np.linalg.norm(gp,axis=1).mean()*1000:.2f} mm, "
          f"d* = {patch_dstar(gp)*100:.2f} cm; two middle rows d* = {patch_dstar(inner)*100:.2f} cm; "
          f"one off-centre row (2-D search) d* = {patch_dstar(gp[np.isclose(gp[:,1], gp[1,1])])*100:.2f} cm")

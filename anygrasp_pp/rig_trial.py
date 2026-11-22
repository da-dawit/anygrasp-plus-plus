"""One grasp-lift-hold trial on the contact rig, with full contact monitoring, and the twist-onset search."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .contact_rig import build_xml, grip_target, OPEN

T_SETTLE, T_CLOSE, T_SQUEEZE = 0.3, 0.5, 0.5     # [s]
T_GRAVITY = 0.5                                   # mid-air grasps: gravity ramps in over this time [s]
LIFT_H, LIFT_T, HOLD_T = 0.10, 1.2, 1.5           # [m], [s], [s]
TWIST_DEG = 10.0                                  # tilt in the hand that counts as twisted out
MONITOR_EVERY = 5                                 # contact monitoring stride [steps]


def min_jerk(tau):
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def a_peak(lift_h=LIFT_H, lift_t=LIFT_T):
    """Peak acceleration of the min-jerk lift [m/s^2]."""
    return 10 * np.sqrt(3) / 3 * lift_h / lift_t**2


class _Geoms:
    """Classifies every hand geom that can touch the object."""

    def __init__(self, m):
        self.obj = m.body("obj").id
        self.fingers = {m.body("left_finger").id: "L", m.body("right_finger").id: "R"}
        meshes = {m.geom("left_finger_mesh").id, m.geom("right_finger_mesh").id}
        self.kind = {}
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if b in self.fingers and m.geom_contype[g] and g not in meshes:
                self.kind[g] = "pad" + self.fingers[b]
        for g in meshes:
            self.kind[g] = "finger_body"
        self.kind[m.geom("palm").id] = "palm"

    def scan(self, m, d):
        """Normal force / count / depth of pad contacts per finger, and force on finger bodies or palm."""
        f6 = np.zeros(6)
        out = {"padL": [0.0, 0, 0.0], "padR": [0.0, 0, 0.0], "other": 0.0}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if self.obj not in (b1, b2):
                continue
            g = c.geom2 if b1 == self.obj else c.geom1
            k = self.kind.get(g)
            if k is None:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            if k.startswith("pad"):
                out[k][0] += f6[0]
                out[k][1] += 1
                out[k][2] = max(out[k][2], -1000 * c.dist)
            else:
                out["other"] += f6[0]
        return out


def run_trial(shift, lift_h=LIFT_H, lift_t=LIFT_T, hold_t=HOLD_T, twist_deg=TWIST_DEG, **rig):
    """Grasp at the object center, (mid-air: ramp gravity in), lift, hold. rig: options for build_xml."""
    grip = rig.get("grip", "force")
    midair = rig.get("midair", False)
    m = mujoco.MjModel.from_xml_string(build_xml(shift, **rig))
    d = mujoco.MjData(m)
    geoms = _Geoms(m)
    hand, obj = m.body("hand").id, m.body("obj").id
    for j in ("left_finger_joint", "right_finger_joint"):
        d.qpos[m.joint(j).qposadr[0]] = OPEN
    i_lift, i_grip = m.actuator("lift").id, m.actuator("grip").id
    g_world = m.opt.gravity.copy()
    t_load = T_SETTLE + T_CLOSE + T_SQUEEZE                  # weight starts acting on the grasp
    t_lift = t_load + (T_GRAVITY if midair else 0.0)
    t_end = t_lift + lift_t + hold_t
    if midair:
        m.opt.gravity[:] = 0.0

    def rel_pose():
        Rh, Rb = d.xmat[hand].reshape(3, 3), d.xmat[obj].reshape(3, 3)
        return Rh.T @ Rb, Rh.T @ (d.xpos[obj] - d.xpos[hand])

    def rot_in_hand(R0):
        """Object rotation in the hand since R0, as a rotvec in the hand frame (x along the object, y closing,
        z approach). Returns (tilt, squaring, total) in deg. tilt = away from the approach axis (the twist that
        matters), squaring = about the approach axis (object lining up with the fingers, harmless)."""
        rv = np.degrees(Rot.from_matrix(rel_pose()[0] @ R0.T).as_rotvec())
        return np.hypot(rv[0], rv[1]), abs(rv[2]), np.linalg.norm(rv)

    def angle(R0):
        return rot_in_hand(R0)[0]

    out, pre, ref, z_ref = {}, None, None, None
    fn_min = {"L": np.inf, "R": np.inf}
    fn_max = {"L": 0.0, "R": 0.0}
    pts = set()
    depth = other = peak = 0.0
    step = 0
    while d.time < t_end:
        t = d.time
        d.ctrl[i_grip] = ((255.0 if t < T_SETTLE else 0.0) if grip == "stock" else grip_target(t, T_SETTLE))
        d.ctrl[i_lift] = lift_h * min_jerk((t - t_lift) / lift_t)
        if midair and t >= t_load:
            m.opt.gravity[:] = g_world * min(1.0, (t - t_load) / T_GRAVITY)
        mujoco.mj_step(m, d)
        step += 1
        if pre is None and d.time >= T_SETTLE:
            pre = rel_pose()
        if ref is None and d.time >= t_load:
            ref, z_ref = rel_pose(), d.xpos[obj][2]
            s = geoms.scan(m, d)
            out["fn_left"], out["pts_left"] = s["padL"][0], s["padL"][1]
            out["fn_right"] = s["padR"][0]
            out["deg_close"] = np.degrees(Rot.from_matrix(ref[0] @ pre[0].T).magnitude())
        if ref is not None and step % MONITOR_EVERY == 0:
            s = geoms.scan(m, d)
            for k in "LR":
                fn_min[k] = min(fn_min[k], s["pad" + k][0])
                fn_max[k] = max(fn_max[k], s["pad" + k][0])
                pts.add(s["pad" + k][1])
                depth = max(depth, s["pad" + k][2])
            other = max(other, s["other"])
            peak = max(peak, angle(ref[0]))
        if "deg_lift" not in out and d.time >= t_lift + lift_t:
            out["deg_lift"] = angle(ref[0])
    out["deg_end"], out["deg_square_end"], out["deg_total_end"] = rot_in_hand(ref[0])
    out["deg_peak"] = max(peak, out["deg_end"])
    out["slip_mm"] = 1000 * np.linalg.norm(rel_pose()[1] - ref[1])
    out["fn_min"], out["fn_max"] = min(fn_min.values()), max(fn_max.values())
    out["pts_min"], out["pts_max"] = min(pts), max(pts)
    out["depth_max_mm"], out["hand_contact_N"] = depth, other
    end = geoms.scan(m, d)
    dropped = (d.xpos[obj][2] < z_ref + 0.5 * lift_h) or (end["padL"][1] + end["padR"][1] == 0)
    out["outcome"] = "dropped" if dropped else "twisted" if out["deg_end"] > twist_deg else "held"
    out["hand_caught"] = bool(out["outcome"] != "dropped" and end["other"] > 0.5)
    return out


def failed(r):
    return r["outcome"] != "held"


def bisect_onset(hi, iters=8, pre=9, **kw):
    """Twist onset on d in [0, hi]: coarse pre-sweep (must be monotone: held ... held, failed ... failed),
    then bisection inside the bracket. Returns (onset, flag); flag is "ok", "never", "always" or "non-monotone"."""
    grid = np.linspace(0.0, hi, pre)
    f = [failed(run_trial(s, **kw)) for s in grid]
    if not any(f):
        return np.inf, "never"
    if f[0]:
        return 0.0, "always"
    k = f.index(True)
    if not all(f[k:]):
        return np.nan, "non-monotone"
    lo, hi = grid[k - 1], grid[k]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if not failed(run_trial(mid, **kw)) else (lo, mid)
    return 0.5 * (lo + hi), "ok"

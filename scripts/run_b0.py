"""B0 baseline: run the best-scoring detected grasp that IK can reach. One trial = one object, one hidden CoM shift,
one seed (rest yaw, xy jitter, sensor noise).

    MUJOCO_GL=egl python scripts/run_b0.py scenes --out results/b0      # settle, render overhead D435, save cloud
    python gpu/detect_grasps.py --detector gsnet --scenes results/b0/scenes --out results/b0/cache
    python scripts/run_b0.py run --out results/b0

success = ends in the bin, not dropped, and never tilted more than 10 deg in the hand (rotation about the approach
axis doesn't count) from lift to end of carry. Bin contact while placing doesn't count against the grasp.
"""
import argparse
import csv
import json
import os
import sys
import zlib
from multiprocessing import Pool

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anygrasp_pp import objects as O  # noqa: E402
from anygrasp_pp import physics as P  # noqa: E402
from anygrasp_pp import robot as RB  # noqa: E402
from anygrasp_pp import scene as S  # noqa: E402

MODELS = {"gso": os.environ.get("GSO_MODELS", "/root/src/mujoco_scanned_objects/models"),
          "ycb": os.environ.get("YCB_MODELS", "/root/src/ycb/models")}
SEEDS = 6                # development seeds 0-5; the final comparison uses fresh seeds (--seeds)
JITTER = 0.03            # m, uniform xy jitter of the resting position
COLLISION = "coacd"
MASS_KG = float(os.environ.get("B0_MASS", O.MASS))   # object mass [kg]
MAX_TILT = 10.0          # deg
MAX_WIDTH = 0.08         # max reported grasp width [m]
D3_DEG = 30.0            # max approach angle from straight down [deg]
PRE = 0.10               # pre-grasp stand-off along the approach [m]
LIFT_H, LIFT_T, HOLD_T = 0.10, 1.2, 2.0
T_SETTLE, T_REACH, T_DESCEND, T_CONVERGE, T_CLOSE, T_SQUEEZE = 1.0, 2.5, 1.5, 1.0, 0.5, 0.5
T_UP, T_ACROSS, T_CARRY_JOINT, T_LOWER, T_RELEASE, T_RETRACT, T_AFTER = 2.0, 3.5, 4.0, 1.5, 0.5, 0.8, 0.5
RETRACT_H = 0.10         # hand moves up this far after opening, before the bin check [m]
BIN_ABOVE = 0.40         # TCP height above the bin center for the carry [m]
BIN_YAWS = (0, 45, -45, 90, -90, 135, -135, 180)   # wrist turns tried for the carry [deg]
RELEASE_CLEAR = 0.02     # lowest object point this far above the bin floor at release [m]
DEPTH_ADD = float(os.environ.get("DEPTH_ADD", 0.0))   # probe: push each grasp this much deeper [m] (0 = as detected)
TIP_CLEAR = 0.003        # the deepened fingertips stay this far above the table [m]
FRAME_HOOK = None        # set by scripts/render_rollouts.py to record a trial; it only reads the state


def deepened(R, t, depth):
    """Push the grasp DEPTH_ADD deeper, but keep the pad corners (17 mm along the height axis) TIP_CLEAR above the
    table."""
    if DEPTH_ADD <= 0:
        return depth
    a, h = R[:, 0], R[:, 2]
    if a[2] >= 0:
        return depth
    deepest = (t[2] - RB.PAD_HALF * abs(h[2]) - S.TABLE_Z - TIP_CLEAR) / -a[2]
    return float(min(depth + DEPTH_ADD, max(depth, deepest)))


def _phase(name):
    if FRAME_HOOK is not None:
        FRAME_HOOK.phase = name


def feasible(r, mass=MASS_KG):
    """Can this object hide the largest shift at this mass (core no denser than tungsten, shell >= 100 g)?"""
    return O.build(os.path.join(MODELS[r["source"]], r["name"]), collision=COLLISION, mass=mass)["core"] is not None


def trials(final_set, seeds=range(SEEDS)):
    rows = [r for r in csv.DictReader(open(final_set)) if feasible(r)]
    return [dict(source=r["source"], name=r["name"], shift=s, seed=k,
                 tag=f"{r['name']}__s{int(round(s * 100))}__k{k}")
            for r in rows for s in O.SHIFTS for k in seeds]


def build(tr, cache={}):
    """Model + data of one trial before settling; the object is body 'obj'."""
    import mujoco
    key = (tr["source"], tr["name"])
    if key not in cache:
        cache.clear()                     # one object per worker, 128 workers caching every object ran out of memory
        model_dir = os.path.join(MODELS[tr["source"]], tr["name"])
        ob = O.build(model_dir, collision=COLLISION, mass=MASS_KG)
        cache[key] = (model_dir, ob, S.rest_poses(ob)[0])
    model_dir, ob, pose = cache[key]
    rng = np.random.default_rng(zlib.crc32(f"{tr['name']}|{tr['seed']}".encode()))   # same pose at every shift
    yaw = rng.uniform(0, 360)
    jit = rng.uniform(-JITTER, JITTER, 2)
    R, z = pose[0], pose[1]
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    Rf = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ R
    pos = np.array([S.PICK[0] + jit[0], S.PICK[1] + jit[1], S.TABLE_Z + z + 0.0005])
    pos[:2] -= (Rf @ ob["centroid"])[:2]
    xml = S.scene_xml([S.object_xml("obj", model_dir, ob, tr["shift"], pos, Rf, collision=COLLISION)],
                      robot=RB.panda_parts(), with_bin=True)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    arm = RB.Panda(m)
    arm.set_q(d, RB.Q_CAPTURE)
    for j in ("finger_joint1", "finger_joint2"):
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = 0.04
    mujoco.mj_forward(m, d)
    return m, d, arm, ob


def settle(m, d, arm):
    import mujoco
    while d.time < T_SETTLE:
        arm.control(d, RB.Q_CAPTURE, np.zeros(7))
        d.ctrl[arm.grip] = 0.04
        mujoco.mj_step(m, d)


def scene_one(args):
    tr, out = args
    import mujoco
    m, d, arm, _ = build(tr)
    settle(m, d, arm)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 0, 1, 0, 0, 0]
    with mujoco.Renderer(m, S.CAM_H, S.CAM_W) as r:
        r.enable_depth_rendering()
        r.update_scene(d, camera="overhead", scene_option=opt)
        dl = r.render().copy()
        r.update_scene(d, camera="overhead_right", scene_option=opt)
        dr = r.render().copy()
    K = S.camera_K()
    depth = S.realsense_depth(dl, dr, K, np.random.default_rng(zlib.crc32(("noise|" + tr["tag"]).encode())))
    cid = m.camera("overhead").id
    xm = d.cam_xmat[cid].reshape(3, 3)
    R_wc, p_wc = S.R_world_cv(xm[:, 0], xm[:, 1]), d.cam_xpos[cid].copy()
    pts = S.backproject(depth, K)
    wide, tight = S.workspace_masks(pts @ R_wc.T + p_wc)
    np.savez_compressed(os.path.join(out, "scenes", tr["tag"] + ".npz"), points=pts[tight].astype(np.float32),
                        scene_points=pts[wide].astype(np.float32), K=K, R_wc=R_wc, p_wc=p_wc)
    return tr["tag"], int(tight.sum())


def choose_grasp(m, arm, scene, cache, skip=0):
    """B0's grasp: in detector order (collision-free, NMS, best score first), the first one within D3_DEG of
    straight down, width <= MAX_WIDTH, above the table, with IK for pre-grasp, grasp, lift and above-bin. Tries
    both closing directions (the hand is symmetric). skip=k gives the (k+1)-th such grasp."""
    import mujoco
    g = cache["grasps"]
    Rw, tw = S.grasps_to_world(g, scene["R_wc"], scene["p_wc"])
    ok = S.approach_ok_world(Rw, D3_DEG) & (g[:, 1] <= MAX_WIDTH) & (tw[:, 2] > S.TABLE_Z + 0.005)
    dk = mujoco.MjData(m)
    tried = 0
    for rank, i in enumerate(cache["order"]):
        if not ok[i]:
            continue
        tried += 1
        skip_before = skip
        for flip in (False, True):
            if skip != skip_before:       # this grasp was reachable and skipped: move to the next grasp
                break
            R = Rw[i].copy()
            if flip:                       # 180 deg about the approach axis
                R[:, 1:] *= -1
            dep = deepened(R, tw[i], float(g[i, 3]))
            p_g, R_g = RB.grasp_to_tcp(R, tw[i], dep)
            a = R_g[:, 2]
            q_pre, ok1, _, _ = arm.ik(dk, p_g - PRE * a, R_g, RB.Q_READY)
            q_g, ok2, _, _ = arm.ik(dk, p_g, R_g, q_pre)
            q_l, ok3, _, _ = arm.ik(dk, p_g + [0, 0, LIFT_H], R_g, q_g)
            if not (ok1 and ok2 and ok3):
                continue
            for yaw in BIN_YAWS:          # the wrist may turn about the vertical on the way to the bin
                c_, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
                R_b = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1.0]]) @ R_g
                q_b, ok4, _, _ = arm.ik(dk, S.BIN_CENTER + [0, 0, BIN_ABOVE], R_b, q_l)
                if ok4 and skip > 0:      # a reachable grasp, but the caller wants a later one
                    skip -= 1
                    break
                if ok4:
                    return dict(index=int(i), rank=rank, tried=tried, flip=flip, score=float(g[i, 0]),
                                width=float(g[i, 1]), depth=dep, p_g=p_g, R_g=R_g, R_b=R_b,
                                q_pre=q_pre, q_g=q_g, q_l=q_l, q_b=q_b, bin_yaw=yaw,
                                approach_deg=float(np.degrees(np.arccos(-a[2]))))
    return dict(index=-1, tried=tried)


def line_qs(arm, dk, p0, p1, R, q0, n=11):
    """IK along a straight TCP line, warm-started point to point."""
    qs, q = [], q0
    for s in np.linspace(0, 1, n):
        q, _, _, _ = arm.ik(dk, p0 + s * (p1 - p0), R, q)
        qs.append(q)
    return np.array(qs)


def path_qs(arm, dk, ps, Rs, q0):
    """Same as line_qs for arbitrary TCP poses. Returns (qs, all converged)."""
    qs, q, ok_all = [], q0, True
    for p, R in zip(ps, Rs):
        q, ok, _, _ = arm.ik(dk, p, R, q)
        ok_all &= ok
        qs.append(q)
    return np.array(qs), ok_all


def carry_path(arm, dk, p_l, R_g, R_b, q_l, n=21):
    """Straight up to travel height, then straight across to above the bin, wrist slerping from R_g to R_b.
    Returns [(qs, T), ...], or None if IK fails anywhere."""
    from scipy.spatial.transform import Rotation as Rot, Slerp
    p_up = np.array([p_l[0], p_l[1], S.TABLE_Z + BIN_ABOVE])
    p_b = S.BIN_CENTER + [0, 0, BIN_ABOVE]
    s_ = np.linspace(0, 1, n)
    up, ok1 = path_qs(arm, dk, [p_l + u * (p_up - p_l) for u in s_], [R_g] * n, q_l)
    slerp = Slerp([0, 1], Rot.from_matrix([R_g, R_b]))
    across, ok2 = path_qs(arm, dk, [p_up + u * (p_b - p_up) for u in s_], slerp(s_).as_matrix(), up[-1])
    return [(up, T_UP), (across, T_ACROSS)] if ok1 and ok2 else None


def lowest_point(m, d, geoms):
    """Lowest z of the object's collision meshes, exact from the vertices."""
    z = np.inf
    for g in geoms:
        if m.geom_type[g] != 7 or not (m.geom_contype[g] or m.geom_conaffinity[g]):    # 7 = mesh
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        z = min(z, float((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()))
    return z


def along(qs, T, t):
    """Joint reference (q, qd, qdd) along the IK samples qs, piecewise linear, min-jerk timing over T."""
    f, fd, fdd = RB.min_jerk_scale(T, t)
    x = f * (len(qs) - 1)
    k = min(int(x), len(qs) - 2)
    dq = (qs[k + 1] - qs[k]) * (len(qs) - 1)
    return qs[k] + (x - k) * (qs[k + 1] - qs[k]), dq * fd, dq * fdd


def contact_dstar(m, d, arm, is_obj, is_pad, ob, mass=O.MASS):
    """Slip distance d* [cm] from the pad contacts the grasp actually made. Per pad, contact points in the pad plane
    (x across, z up, from the TCP) weighted by normal force; the pads share the weight, so
    d* = (T_left(W/2) + T_right(W/2)) / W."""
    import mujoco
    from anygrasp_pp.rig_trial import a_peak
    tcp, R = arm.tcp_pose(d)
    pts = {"left": [], "right": []}
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if not (is_obj[c.geom1] ^ is_obj[c.geom2]):
            continue
        other = c.geom2 if is_obj[c.geom1] else c.geom1
        if not is_pad[other]:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        r = c.pos - tcp
        side = "left" if m.geom(other).name.startswith("left") else "right"
        pts[side].append((r @ R[:, 0], -(r @ R[:, 2]), f6[0]))
    out = dict(n_pts_left=len(pts["left"]), n_pts_right=len(pts["right"]))
    if not pts["left"] or not pts["right"]:
        return dict(out, dstar_lift_cm=0.0, dstar_static_cm=0.0)
    for name, acc in (("dstar_lift_cm", a_peak()), ("dstar_static_cm", 0.0)):
        W = mass * (P.G + acc)
        T = 0.0
        for side in ("left", "right"):
            a_ = np.array(pts[side])
            fn = a_[:, 2].sum()
            out[f"fn_{side}"] = float(fn)
            T += P.patch_torque_capacity(a_[:, :2], W / 2, P.MU, fn, weights=np.maximum(a_[:, 2], 1e-9))
        out[name] = float(T / W * 100)
    return out


def run_one(args):
    tr, out = args
    import mujoco
    rec = dict(tr, mass_kg=MASS_KG)
    try:
        m, d, arm, ob = build(tr)
        settle(m, d, arm)
        scene = np.load(os.path.join(out, "scenes", tr["tag"] + ".npz"))
        cache = np.load(os.path.join(out, "cache", "gsnet", tr["tag"] + ".npz"))
        gr = choose_grasp(m, arm, scene, cache, skip=tr.get("cand", 0))
        rec.update(grasp_index=gr["index"], grasp_tried=gr["tried"])
        if gr["index"] < 0:
            rec.update(outcome="no reachable grasp", success=False)
            return rec
        rec.update({k: gr[k] for k in ("rank", "flip", "score", "width", "depth", "approach_deg", "bin_yaw")})
        dk = mujoco.MjData(m)
        obj = m.body("obj").id
        obj_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == obj}
        is_obj = np.zeros(m.ngeom, bool)
        is_obj[list(obj_geoms)] = True
        is_pad = np.zeros(m.ngeom, bool)
        is_pad[list(arm.pad_geoms)] = True
        is_finger = np.zeros(m.ngeom, bool)
        is_finger[list(arm.finger_mesh)] = True
        a = gr["R_g"][:, 2]
        descend = line_qs(arm, dk, gr["p_g"] - PRE * a, gr["p_g"], gr["R_g"], gr["q_pre"])
        lift = line_qs(arm, dk, gr["p_g"], gr["p_g"] + [0, 0, LIFT_H], gr["R_g"], gr["q_g"])
        w_open = min(gr["width"] + 0.01, RB.MAX_OPEN)
        log = dict(tilt_max=0.0, square_max=0.0, palm=False, finger_body=False, pad_lost_t=0.0, dropped=False,
                   tau_sat=0, t_tilt=None, t_drop=None)
        state = {}

        def step(q_ref, qd_ref, grip, qdd_ref=None):
            arm.control(d, q_ref, qd_ref, qdd_ref)
            d.ctrl[arm.grip] = grip
            mujoco.mj_step(m, d)
            if np.any(np.abs(d.ctrl[arm.act]) >= RB.TAU_MAX - 1e-9):
                log["tau_sat"] += 1
            if FRAME_HOOK is not None:
                FRAME_HOOK(m, d)

        def rel():
            Rh = d.xmat[arm.hand].reshape(3, 3)
            return Rh.T @ d.xmat[obj].reshape(3, 3), Rh.T @ (d.xipos[obj] - d.xpos[arm.hand])

        def track():
            R_rel, p_rel = rel()
            rv = Rot.from_matrix(R_rel @ state["R0"].T).as_rotvec()
            # hand z = approach; turning about it (squaring up) is not a failure
            log["tilt_max"] = max(log["tilt_max"], float(np.degrees(np.hypot(rv[0], rv[1]))))
            if log["t_tilt"] is None and log["tilt_max"] > MAX_TILT:
                log["t_tilt"] = round(d.time, 3)
            log["square_max"] = max(log["square_max"], float(np.degrees(abs(rv[2]))))
            if np.linalg.norm(p_rel - state["p0"]) > 0.05 and not log["dropped"]:
                log["dropped"], log["t_drop"] = True, round(d.time, 3)
            n = d.ncon
            g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
            touch = is_obj[g1] ^ is_obj[g2]
            other = np.where(is_obj[g1], g2, g1)[touch]
            log["palm"] |= bool((other == arm.palm).any())
            log["finger_body"] |= bool(is_finger[other].any())
            return bool(is_pad[other].any())

        _phase("reach")
        # gripper opens on the way
        t0 = d.time
        while d.time - t0 < T_REACH:
            q, qd, qdd = RB.min_jerk(RB.Q_CAPTURE, gr["q_pre"], T_REACH, d.time - t0)
            step(q, qd, RB.grip_ctrl_open(w_open), qdd)
        _phase("descend")
        t0 = d.time
        while d.time - t0 < T_DESCEND:
            q, qd, qdd = along(descend, T_DESCEND, d.time - t0)
            step(q, qd, RB.grip_ctrl_open(w_open), qdd)
        # close only once the arm is there, like a real robot finishing its motion before the grasp command
        t0 = d.time
        while d.time - t0 < T_CONVERGE:
            step(gr["q_g"], np.zeros(7), RB.grip_ctrl_open(w_open))
            if (np.linalg.norm(arm.tcp_pose(d)[0] - gr["p_g"]) < 1e-3
                    and np.abs(d.qvel[arm.dadr]).max() < 1e-2):
                break
        rec["tcp_error_mm"] = float(np.linalg.norm(arm.tcp_pose(d)[0] - gr["p_g"]) * 1e3)
        _phase("close")
        # Franka-style grasp: target past the object, the force clamp sets 70 N per pad
        t0 = d.time
        while d.time - t0 < T_CLOSE + T_SQUEEZE:
            step(gr["q_g"], np.zeros(7), RB.grip_ctrl_closing(w_open, d.time - t0))
        state["R0"], state["p0"] = rel()
        n = d.ncon
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        touch = is_obj[g1] ^ is_obj[g2]
        rec["pads_on_object"] = int(is_pad[np.where(is_obj[g1], g2, g1)[touch]].sum())
        rec.update(contact_dstar(m, d, arm, is_obj, is_pad, ob, mass=MASS_KG))
        tcp, Rt = arm.tcp_pose(d)
        rec["lever_arm_cm"] = float(abs(P.lever_arm(tcp, Rt[:, 1], d.xipos[obj])[0]) * 100)
        rec["com_shift_seen_cm"] = float(tr["shift"] * 100)
        grip = RB.grip_ctrl_closing(w_open, 1e9)
        _phase("lift and hold")
        t0 = d.time
        while d.time - t0 < LIFT_T + HOLD_T:
            q, qd, qdd = along(lift, LIFT_T, d.time - t0)
            step(q, qd, grip, qdd)
            if not track():
                log["pad_lost_t"] += m.opt.timestep
        rec["tilt_after_hold"] = log["tilt_max"]
        _phase("carry")
        # up, then across; joint-space fallback if IK fails on the path
        segs = carry_path(arm, dk, gr["p_g"] + [0, 0, LIFT_H], gr["R_g"], gr["R_b"], gr["q_l"])
        rec["carry"] = "cartesian" if segs else "joint"
        for qs_, T_ in (segs or []):
            t0 = d.time
            while d.time - t0 < T_:
                q, qd, qdd = along(qs_, T_, d.time - t0)
                step(q, qd, grip, qdd)
                if not track():
                    log["pad_lost_t"] += m.opt.timestep
        if not segs:
            t0 = d.time
            while d.time - t0 < T_CARRY_JOINT:
                q, qd, qdd = RB.min_jerk(gr["q_l"], gr["q_b"], T_CARRY_JOINT, d.time - t0)
                step(q, qd, grip, qdd)
                if not track():
                    log["pad_lost_t"] += m.opt.timestep
        q_top = segs[-1][0][-1] if segs else gr["q_b"]
        _phase("place")
        # lower until the object's lowest point is RELEASE_CLEAR above the floor
        drop = max(0.0, lowest_point(m, d, obj_geoms) - (S.TABLE_Z + RELEASE_CLEAR))
        p_b = S.BIN_CENTER + [0, 0, BIN_ABOVE]
        lower = line_qs(arm, dk, p_b, p_b - [0, 0, min(drop, BIN_ABOVE - 0.05)], gr["R_b"], q_top)
        t0 = d.time
        while d.time - t0 < T_LOWER:        # placing: bin contact isn't a grasp failure, so no tracking
            q, qd, qdd = along(lower, T_LOWER, d.time - t0)
            step(q, qd, grip, qdd)
        t0 = d.time
        while d.time - t0 < T_RELEASE:
            step(lower[-1], np.zeros(7), RB.MAX_OPEN / 2)
        _phase("release")
        # retract straight up with the hand open, then let the object settle
        p_low = p_b - [0, 0, min(drop, BIN_ABOVE - 0.05)]
        retract = line_qs(arm, dk, p_low, p_low + [0, 0, RETRACT_H], gr["R_b"], lower[-1])
        t0 = d.time
        while d.time - t0 < T_RETRACT:
            q, qd, qdd = along(retract, T_RETRACT, d.time - t0)
            step(q, qd, RB.MAX_OPEN / 2, qdd)
        t0 = d.time
        while d.time - t0 < T_AFTER:
            step(retract[-1], np.zeros(7), RB.MAX_OPEN / 2)
        rec["final_com"] = [round(float(v), 4) for v in d.xipos[obj]]
        n_ = d.ncon
        g1_, g2_ = d.contact.geom1[:n_], d.contact.geom2[:n_]
        tch = is_obj[g1_] ^ is_obj[g2_]
        oth = np.where(is_obj[g1_], g2_, g1_)[tch]
        rec["touching_hand_at_end"] = bool(np.isin(oth, list(arm.pad_geoms | arm.finger_mesh | {arm.palm})).any())
        rec["touching_bin_at_end"] = bool(any(m.geom(int(o)).name.startswith("bin") for o in oth))
        in_bin = S.in_bin(d.xipos[obj], free_of_hand=not rec["touching_hand_at_end"])
        tilt_fail = log["tilt_max"] > MAX_TILT
        success = in_bin and not log["dropped"] and not tilt_fail
        if success:
            outcome = "success"
        elif rec["pads_on_object"] == 0:
            outcome = "no grip (blocked on approach)" if rec["tcp_error_mm"] > 5 else "no grip (closed on air)"
        elif tilt_fail and (not log["dropped"] or log["t_tilt"] <= log["t_drop"]):
            outcome = "turned in hand"
        elif log["dropped"]:
            outcome = "dropped"
        elif np.linalg.norm(d.xipos[obj][:2] - S.BIN_CENTER[:2]) > 1.0:
            outcome = "ejected at release (simulation)"   # a rare contact blow-up, flagged rather than hidden
        else:
            outcome = "missed the bin"
        rec.update(outcome=outcome, success=success, in_bin=in_bin, **log)
        return rec
    except Exception as e:  # a crash is logged, never silently dropped
        rec.update(outcome="error", success=False, error=repr(e)[:200])
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scenes", "run", "candidates"],
                    help="candidates: execute each scene's first K reachable grasps (re-ranking headroom)")
    ap.add_argument("--k", type=int, default=10, help="grasps per scene for 'candidates'")
    ap.add_argument("--out", default="results/b0")
    ap.add_argument("--final-set", default="results/objects/final_set.csv")
    ap.add_argument("--jobs", type=int, default=96)
    ap.add_argument("--only", default=None, help="comma-separated tags (debugging)")
    ap.add_argument("--seeds", default=f"0:{SEEDS}", help="seed range start:stop (fresh seeds for the final run)")
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.seeds.split(":"))
    trs = trials(a.final_set, range(lo, hi))
    if a.only:
        trs = [t for t in trs if t["tag"] in a.only.split(",")]
    if a.cmd == "candidates":
        trs = [dict(t, cand=c) for t in trs for c in range(a.k)]
    os.makedirs(os.path.join(a.out, "scenes"), exist_ok=True)
    if a.cmd == "scenes":
        with Pool(min(a.jobs, len(trs)), maxtasksperchild=20) as pool:
            res = pool.map(scene_one, [(t, a.out) for t in trs], chunksize=1)
        print(f"{len(res)} scenes; detector input points: min {min(n for _, n in res)}, "
              f"median {int(np.median([n for _, n in res]))}")
        return
    # append each trial as soon as it finishes, so a killed run loses nothing and a rerun resumes
    parts = os.path.join(a.out, f"{a.cmd}_parts.jsonl")
    key = lambda r: f"{r['tag']}#{r.get('cand', 0)}"
    done = {}
    if os.path.exists(parts):
        for line in open(parts):
            r = json.loads(line)
            done[key(r)] = r
    todo = [t for t in trs if key(t) not in done]
    print(f"{len(trs)} trials, {len(done)} already done, {len(todo)} to run", flush=True)
    with Pool(min(a.jobs, max(len(todo), 1)), maxtasksperchild=20) as pool, open(parts, "a") as f:
        for r in pool.imap_unordered(run_one, [(t, a.out) for t in todo], chunksize=4):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            done[key(r)] = r
    rows = [done[key(t)] for t in trs if key(t) in done]
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out, "candidates.csv" if a.cmd == "candidates" else "trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(Counter(r["outcome"] for r in rows))
    for s in O.SHIFTS:
        rs = [r for r in rows if r["shift"] == s]
        print(f"  shift {s * 100:.0f} cm: success {sum(r['success'] for r in rs)}/{len(rs)}")


if __name__ == "__main__":
    main()

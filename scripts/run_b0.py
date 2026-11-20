"""B0: the usual way of using the detector -- execute the best-scoring grasp that IK can reach (proposal; D3,
D14, D18). One trial = one object at one hidden shift and one seed (resting pose yaw + position jitter + sensor
noise).

    scenes  settle each trial's scene (arm parked out of view), render the overhead D435 (D17), save the cloud
    run     re-create the same settled scene, pick B0's grasp from the cached detections, execute it with the
            torque-controlled Panda, log the outcome

    MUJOCO_GL=egl python scripts/run_b0.py scenes --out results/b0
    python gpu/detect_grasps.py --detector gsnet --scenes results/b0/scenes --out results/b0/cache
    python scripts/run_b0.py run --out results/b0

Outcome (D2): success = the object ends in the bin, was not dropped before release, and never turned more than
10 deg in the hand (tilt, rotation about the approach axis excluded) between the lift and the release.
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

MODELS = {"gso": "/root/src/mujoco_scanned_objects/models", "ycb": "/root/src/ycb/models"}
SEEDS = 6
JITTER = 0.03            # m, uniform xy jitter of the resting position
COLLISION = "coacd"
MAX_TILT = 10.0          # deg (D2)
MAX_WIDTH = 0.08         # reported width (D14)
D3_DEG = 30.0
PRE = 0.10               # pre-grasp stand-off along the approach [m]
LIFT_H, LIFT_T, HOLD_T = 0.10, 1.2, 2.0
T_SETTLE, T_REACH, T_DESCEND, T_CONVERGE, T_CLOSE, T_SQUEEZE = 1.0, 2.5, 1.5, 1.0, 0.5, 0.5
T_CARRY, T_LOWER, T_RELEASE, T_AFTER = 2.5, 1.2, 0.5, 1.0
BIN_ABOVE = 0.40         # TCP height above the bin centre for the carry [m]
BIN_YAWS = (0, 45, -45, 90, -90, 135, -135, 180)   # wrist turns tried for the carry [deg]
RELEASE_CLEAR = 0.04     # lowest object point this far above the bin floor at release [m]


def trials(final_set):
    rows = list(csv.DictReader(open(final_set)))
    return [dict(source=r["source"], name=r["name"], shift=s, seed=k,
                 tag=f"{r['name']}__s{int(round(s * 100))}__k{k}")
            for r in rows for s in O.SHIFTS for k in range(SEEDS)]


def build(tr, cache={}):
    """Model + data of one trial at its initial (unsettled) state; the object is the body 'obj'."""
    import mujoco
    key = (tr["source"], tr["name"])
    if key not in cache:
        model_dir = os.path.join(MODELS[tr["source"]], tr["name"])
        ob = O.build(model_dir, collision=COLLISION)
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


# ----------------------------------------------------------------------------- scenes
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


# ----------------------------------------------------------------------------- B0 grasp choice
def choose_grasp(m, arm, scene, cache):
    """B0: in the detector's order (collision-free, NMS, best score first), the first grasp that passes D3 and
    D14, sits on the object, and has IK solutions for pre-grasp, grasp, lift and above-bin. Both closing
    directions of a grasp are tried (the hand is symmetric)."""
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
        for flip in (False, True):
            R = Rw[i].copy()
            if flip:                       # 180 deg about the approach axis
                R[:, 1:] *= -1
            p_g, R_g = RB.grasp_to_tcp(R, tw[i], g[i, 3])
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
                if ok4:
                    return dict(index=int(i), rank=rank, tried=tried, flip=flip, score=float(g[i, 0]),
                                width=float(g[i, 1]), depth=float(g[i, 3]), p_g=p_g, R_g=R_g, R_b=R_b,
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


def along(qs, T, t):
    """Joint reference (q, qd, qdd) following the IK samples qs (piecewise linear in the path parameter) with a
    minimum-jerk timing over T."""
    f, fd, fdd = RB.min_jerk_scale(T, t)
    x = f * (len(qs) - 1)
    k = min(int(x), len(qs) - 2)
    dq = (qs[k + 1] - qs[k]) * (len(qs) - 1)
    return qs[k] + (x - k) * (qs[k + 1] - qs[k]), dq * fd, dq * fdd


# ----------------------------------------------------------------------------- execution
def run_one(args):
    tr, out = args
    import mujoco
    rec = dict(tr)
    try:
        m, d, arm, ob = build(tr)
        settle(m, d, arm)
        scene = np.load(os.path.join(out, "scenes", tr["tag"] + ".npz"))
        cache = np.load(os.path.join(out, "cache", "gsnet", tr["tag"] + ".npz"))
        gr = choose_grasp(m, arm, scene, cache)
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

        def rel():
            Rh = d.xmat[arm.hand].reshape(3, 3)
            return Rh.T @ d.xmat[obj].reshape(3, 3), Rh.T @ (d.xipos[obj] - d.xpos[arm.hand])

        def track():
            R_rel, p_rel = rel()
            rv = Rot.from_matrix(R_rel @ state["R0"].T).as_rotvec()
            # hand axes: z = approach; rotation about it (squaring up) is not a failure (D2)
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

        # 1. reach the pre-grasp (gripper opens on the way)
        t0 = d.time
        while d.time - t0 < T_REACH:
            q, qd, qdd = RB.min_jerk(RB.Q_CAPTURE, gr["q_pre"], T_REACH, d.time - t0)
            step(q, qd, RB.grip_ctrl_open(w_open), qdd)
        # 2. descend along the approach, dwell
        t0 = d.time
        while d.time - t0 < T_DESCEND:
            q, qd, qdd = along(descend, T_DESCEND, d.time - t0)
            step(q, qd, RB.grip_ctrl_open(w_open), qdd)
        # close only once the arm is there, as a real robot's motion finishes before the grasp command
        t0 = d.time
        while d.time - t0 < T_CONVERGE:
            step(gr["q_g"], np.zeros(7), RB.grip_ctrl_open(w_open))
            if (np.linalg.norm(arm.tcp_pose(d)[0] - gr["p_g"]) < 1e-3
                    and np.abs(d.qvel[arm.dadr]).max() < 1e-2):
                break
        rec["tcp_error_mm"] = float(np.linalg.norm(arm.tcp_pose(d)[0] - gr["p_g"]) * 1e3)
        # 3. close (Franka grasp: run the target past the object; the force clamp sets 70 N per pad) and squeeze
        t0 = d.time
        while d.time - t0 < T_CLOSE + T_SQUEEZE:
            step(gr["q_g"], np.zeros(7), RB.grip_ctrl_closing(w_open, d.time - t0))
        state["R0"], state["p0"] = rel()
        n = d.ncon
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        touch = is_obj[g1] ^ is_obj[g2]
        rec["pads_on_object"] = int(is_pad[np.where(is_obj[g1], g2, g1)[touch]].sum())
        tcp, Rt = arm.tcp_pose(d)
        rec["lever_arm_cm"] = float(abs(P.lever_arm(tcp, Rt[:, 1], d.xipos[obj])[0]) * 100)
        rec["com_shift_seen_cm"] = float(tr["shift"] * 100)
        grip = RB.grip_ctrl_closing(w_open, 1e9)
        # 4. lift, hold
        t0 = d.time
        while d.time - t0 < LIFT_T + HOLD_T:
            q, qd, qdd = along(lift, LIFT_T, d.time - t0)
            step(q, qd, grip, qdd)
            if not track():
                log["pad_lost_t"] += m.opt.timestep
        rec["tilt_after_hold"] = log["tilt_max"]
        # 5. carry to above the bin
        t0 = d.time
        while d.time - t0 < T_CARRY:
            q, qd, qdd = RB.min_jerk(gr["q_l"], gr["q_b"], T_CARRY, d.time - t0)
            step(q, qd, grip, qdd)
            if not track():
                log["pad_lost_t"] += m.opt.timestep
        # 6. lower until the object's lowest point is RELEASE_CLEAR above the bin floor, release
        low = min(d.geom_xpos[g][2] - m.geom_rbound[g] for g in obj_geoms if m.geom_contype[g] or m.geom_conaffinity[g])
        drop = max(0.0, low - (S.TABLE_Z + RELEASE_CLEAR))
        p_b = S.BIN_CENTER + [0, 0, BIN_ABOVE]
        lower = line_qs(arm, dk, p_b, p_b - [0, 0, min(drop, BIN_ABOVE - 0.12)], gr["R_b"], gr["q_b"])
        t0 = d.time
        while d.time - t0 < T_LOWER:
            q, qd, qdd = along(lower, T_LOWER, d.time - t0)
            step(q, qd, grip, qdd)
            if not track():
                log["pad_lost_t"] += m.opt.timestep
        t0 = d.time
        while d.time - t0 < T_RELEASE + T_AFTER:
            step(lower[-1], np.zeros(7), RB.MAX_OPEN / 2)
        in_bin = S.in_bin(d.xipos[obj])
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
        else:
            outcome = "missed the bin"
        rec.update(outcome=outcome, success=success, in_bin=in_bin, **log)
        return rec
    except Exception as e:  # a crash is logged, never silently dropped
        rec.update(outcome="error", success=False, error=repr(e)[:200])
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scenes", "run"])
    ap.add_argument("--out", default="results/b0")
    ap.add_argument("--final-set", default="results/objects/final_set.csv")
    ap.add_argument("--jobs", type=int, default=128)
    ap.add_argument("--only", default=None, help="comma-separated tags (debugging)")
    a = ap.parse_args()
    trs = trials(a.final_set)
    if a.only:
        trs = [t for t in trs if t["tag"] in a.only.split(",")]
    os.makedirs(os.path.join(a.out, "scenes"), exist_ok=True)
    with Pool(min(a.jobs, len(trs))) as pool:
        if a.cmd == "scenes":
            res = pool.map(scene_one, [(t, a.out) for t in trs], chunksize=1)
            print(f"{len(res)} scenes; detector input points: min {min(n for _, n in res)}, "
                  f"median {int(np.median([n for _, n in res]))}")
            return
        rows = pool.map(run_one, [(t, a.out) for t in trs], chunksize=1)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out, "trials.csv"), "w", newline="") as f:
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

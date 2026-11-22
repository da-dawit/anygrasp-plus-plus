"""M3: grasp, 2 cm test lift (1 s up, 2 s hold) read by an ideal wrist F/T sensor, regrasp if bad. At most N_TRIES
grasps per scene. The sensor gives mass and horizontal CoM, so the grasp's real lever arm.

  agpp   M2 picks; keep if full weight carried and lever arm steady, else set down, re-image, pick by known-CoM model
  retry  B0 picks; same trigger plus a sim oracle (tilt > 10 deg or drop in the test lift); regrasp = next grasp

Outcome judged as in run_b0, tilt counted from the squeeze, so turning in the test lift counts against a kept grasp.
Env: DETQ = file queue to gpu/detect_service.py for re-detection (default /root/detq); DETECTOR=reuse reuses the
original detections (debug only, the object may have moved); M3_TAU > 0 also regrasps if the known-CoM model is
below it.

    B0_MASS=0.45 MUJOCO_GL=egl python scripts/run_m3.py --run results/b0_m045_v2 --policy agpp --out results/m3_dev
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import run_b0 as B  # noqa: E402
from m2_learn import design, fit_logistic  # noqa: E402
from anygrasp_pp import m2 as M2  # noqa: E402
from anygrasp_pp import physics as P  # noqa: E402
from anygrasp_pp import robot as RB  # noqa: E402
from anygrasp_pp import scene as S  # noqa: E402
from anygrasp_pp.rig_trial import a_peak  # noqa: E402

N_TRIES = 2              # one regrasp, as in the proposal
K_CANDS = 10             # reachable grasps considered per image
TEST_H, TEST_T, TEST_HOLD = 0.02, 1.0, 2.0    # test lift height [m], time [s], hold [s]
SETTLE_SKIP = 0.3        # start of the hold not used for the reading [s]
DRIFT_MM = 3.0           # lever-arm drift over the hold that counts as turning [mm]
LIFTED_FRAC = 0.8        # hand must carry at least this much of the weight (less: one end is still on the table,
                         # so it already turned)
TAU = float(os.environ.get("M3_TAU", 0.0))   # also regrasp if the known-CoM model gives less than this (0 = off)
DUP_DIST, DUP_DEG = 0.015, 30.0     # a regrasp this close to a tried grasp (center, closing axis) is the same grasp
T_DOWN, T_OPEN, T_BACK, T_PARK, T_REST = 1.0, 0.5, 0.8, 2.5, 1.0
DETQ = os.environ.get("DETQ", "/root/detq")
DETECTOR = os.environ.get("DETECTOR", "service")
FRAME_HOOK = None        # set by scripts/render_plain.py to film a trial; it only reads the state
_MODELS = {}


def known_design(rows):
    """M2 features with the CoM known: lever arm about the known CoM instead of the shape-based one."""
    X = []
    for f in rows:
        ds, d = 100 * float(f["dstar_pred"]), float(f["d_known_cm"])
        X.append([ds, d, d / max(ds, 0.1), float(f["score"]), float(f["approach_deg"]), 100 * float(f["width"]),
                  100 * float(f["depth"])])
    return np.array(X)


def models(dev_run, hold_out):
    """(M2, known-CoM) logistic models fit on the dev candidates minus object `hold_out`, so a model never ranks an
    object it was trained on."""
    key = (dev_run, hold_out)
    if key not in _MODELS:
        F = [f for f in csv.DictReader(open(os.path.join(dev_run, "candidates_features.csv"))) if f["name"] != hold_out]
        y = np.array([f["success"] == "True" for f in F], float)
        for f in F:
            f["d_known_cm"] = f["lever_arm_cm"] or 30
        _MODELS[key] = (fit_logistic(design(F), y)[0], fit_logistic(known_design(F), y)[0])
    return _MODELS[key]


class Trial:
    def __init__(self, tr, run, out, hook=None):
        import mujoco
        self.mj = mujoco
        self.tr, self.run, self.out = tr, run, out
        self.hook = hook if hook is not None else FRAME_HOOK
        self.m, self.d, self.arm, self.ob = B.build(tr)
        B.settle(self.m, self.d, self.arm)
        m = self.m
        self.dk = mujoco.MjData(m)
        self.obj = m.body("obj").id
        self.obj_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.obj}
        self.is_obj = np.zeros(m.ngeom, bool)
        self.is_obj[list(self.obj_geoms)] = True
        self.is_pad = np.zeros(m.ngeom, bool)
        self.is_pad[list(self.arm.pad_geoms)] = True
        self.is_finger = np.zeros(m.ngeom, bool)
        self.is_finger[list(self.arm.finger_mesh)] = True
        hand_bodies = {m.body(n).id for n in ("hand", "left_finger", "right_finger")}
        self.is_hand = np.array([m.geom_bodyid[g] in hand_bodies for g in range(m.ngeom)])
        self.f6 = np.zeros(6)
        self.tau_sat = 0

    # step/rel/track as in run_b0.run_one
    def step(self, q, qd, grip, qdd=None):
        self.arm.control(self.d, q, qd, qdd)
        self.d.ctrl[self.arm.grip] = grip
        self.mj.mj_step(self.m, self.d)
        if np.any(np.abs(self.d.ctrl[self.arm.act]) >= RB.TAU_MAX - 1e-9):
            self.tau_sat += 1
        if self.hook is not None:
            self.hook(self.m, self.d)

    def phase(self, name):
        if self.hook is not None:
            self.hook.phase = name

    def rel(self):
        d, arm = self.d, self.arm
        Rh = d.xmat[arm.hand].reshape(3, 3)
        return Rh.T @ d.xmat[self.obj].reshape(3, 3), Rh.T @ (d.xipos[self.obj] - d.xpos[arm.hand])

    def track(self, log):
        R_rel, p_rel = self.rel()
        rv = Rot.from_matrix(R_rel @ log["R0"].T).as_rotvec()
        log["tilt_max"] = max(log["tilt_max"], float(np.degrees(np.hypot(rv[0], rv[1]))))
        if log["t_tilt"] is None and log["tilt_max"] > B.MAX_TILT:
            log["t_tilt"] = round(self.d.time, 3)
        if np.linalg.norm(p_rel - log["p0"]) > 0.05 and not log["dropped"]:
            log["dropped"], log["t_drop"] = True, round(self.d.time, 3)
        n = self.d.ncon
        g1, g2 = self.d.contact.geom1[:n], self.d.contact.geom2[:n]
        touch = self.is_obj[g1] ^ self.is_obj[g2]
        other = np.where(self.is_obj[g1], g2, g1)[touch]
        log["palm"] |= bool((other == self.arm.palm).any())
        log["finger_body"] |= bool(self.is_finger[other].any())

    def hand_wrench(self, p):
        """Ideal wrist F/T sensor: total force of the hand on the object and its moment about p, world frame."""
        F, T = np.zeros(3), np.zeros(3)
        d = self.d
        for i in range(d.ncon):
            c = d.contact[i]
            if not (self.is_obj[c.geom1] ^ self.is_obj[c.geom2]):
                continue
            other = c.geom2 if self.is_obj[c.geom1] else c.geom1
            if not self.is_hand[other]:
                continue
            self.mj.mj_contactForce(self.m, d, i, self.f6)
            fw = c.frame.reshape(3, 3).T @ self.f6[:3]          # force on geom2, world frame
            f = fw if self.is_obj[c.geom2] else -fw
            F += f
            T += np.cross(c.pos - p, f)
        return F, T

    def capture(self, name):
        """New overhead D435 image with the arm parked, saved as a scene file like run_b0.scene_one."""
        mj = self.mj
        opt = mj.MjvOption()
        opt.geomgroup[:] = [1, 0, 1, 0, 0, 0]
        with mj.Renderer(self.m, S.CAM_H, S.CAM_W) as r:
            r.enable_depth_rendering()
            r.update_scene(self.d, camera="overhead", scene_option=opt)
            dl = r.render().copy()
            r.update_scene(self.d, camera="overhead_right", scene_option=opt)
            dr = r.render().copy()
        K = S.camera_K()
        depth = S.realsense_depth(dl, dr, K, np.random.default_rng(zlib.crc32(("noise|" + name).encode())))
        cid = self.m.camera("overhead").id
        xm = self.d.cam_xmat[cid].reshape(3, 3)
        R_wc, p_wc = S.R_world_cv(xm[:, 0], xm[:, 1]), self.d.cam_xpos[cid].copy()
        pts = S.backproject(depth, K)
        wide, tight = S.workspace_masks(pts @ R_wc.T + p_wc)
        path = os.path.join(self.out, "scenes", name + ".npz")
        np.savez_compressed(path, points=pts[tight].astype(np.float32), scene_points=pts[wide].astype(np.float32),
                            K=K, R_wc=R_wc, p_wc=p_wc)
        return path

    def detect(self, scene_path, name):
        if DETECTOR == "reuse":
            return dict(np.load(os.path.join(self.run, "cache", "gsnet", self.tr["tag"] + ".npz")))
        os.makedirs(os.path.join(DETQ, "in"), exist_ok=True)
        tmp = os.path.join(DETQ, "in", name + ".part")
        shutil.copy(scene_path, tmp)
        os.replace(tmp, os.path.join(DETQ, "in", name + ".npz"))
        res = os.path.join(DETQ, "out", name + ".npz")
        t0 = time.time()
        while not os.path.exists(res):
            if time.time() - t0 > 1800:
                raise TimeoutError("no detection for " + name)
            time.sleep(0.3)
        out = dict(np.load(res))
        shutil.copy(res, os.path.join(self.out, "cache", name + ".npz"))
        return out

    # features as in eval_rerank.features, but on the planned grasp
    def candidates(self, scene, cache, mass, c_known=None, exclude=()):
        pts = M2.object_points(scene["points"] @ scene["R_wc"].T + scene["p_wc"])
        if len(pts) < 10:
            return []
        c0, V, half = M2.com_prior(pts)
        out = []
        for k in range(K_CANDS):
            gr = B.choose_grasp(self.m, self.arm, scene, cache, skip=k)
            if gr["index"] < 0:
                break
            if any(np.linalg.norm(gr["p_g"] - p) < DUP_DIST and abs(gr["R_g"][:, 1] @ a) > np.cos(np.radians(DUP_DEG))
                   for p, a in exclude):
                continue
            w_open = min(gr["width"] + 0.01, RB.MAX_OPEN)
            ds, _ = M2.predicted_dstar(pts, gr["p_g"], gr["R_g"], w_open, mass, accel=a_peak())
            axis = gr["R_g"][:, 1]
            ph = float(np.ravel(P.p_hold_uniform(gr["p_g"], axis, c0, V, half, mass, ds=ds))[0]) if ds > 0 else 0.0
            f = dict(cand=k, dstar_pred=ds, p_hold=ph, d_vis=float(abs(P.lever_arm(gr["p_g"], axis, c0)[0])),
                     score=gr["score"], approach_deg=gr["approach_deg"], width=gr["width"], depth=gr["depth"], gr=gr)
            if c_known is not None:
                f["d_known_cm"] = float(abs(P.lever_arm(gr["p_g"], axis, c_known)[0]) * 100)
            out.append(f)
        return out

    # approach, close, test lift; returns the sensor reading
    def grasp_and_test(self, gr, q_start):
        d, arm = self.d, self.arm
        a = gr["R_g"][:, 2]
        descend = B.line_qs(arm, self.dk, gr["p_g"] - B.PRE * a, gr["p_g"], gr["R_g"], gr["q_pre"])
        w_open = min(gr["width"] + 0.01, RB.MAX_OPEN)
        log = dict(tilt_max=0.0, palm=False, finger_body=False, dropped=False, t_tilt=None, t_drop=None)
        self.phase("reach")
        t0 = d.time
        while d.time - t0 < B.T_REACH:
            q, qd, qdd = RB.min_jerk(q_start, gr["q_pre"], B.T_REACH, d.time - t0)
            self.step(q, qd, RB.grip_ctrl_open(w_open), qdd)
        self.phase("descend")
        t0 = d.time
        while d.time - t0 < B.T_DESCEND:
            q, qd, qdd = B.along(descend, B.T_DESCEND, d.time - t0)
            self.step(q, qd, RB.grip_ctrl_open(w_open), qdd)
        t0 = d.time
        while d.time - t0 < B.T_CONVERGE:
            self.step(gr["q_g"], np.zeros(7), RB.grip_ctrl_open(w_open))
            if np.linalg.norm(arm.tcp_pose(d)[0] - gr["p_g"]) < 1e-3 and np.abs(d.qvel[arm.dadr]).max() < 1e-2:
                break
        log["tcp_error_mm"] = float(np.linalg.norm(arm.tcp_pose(d)[0] - gr["p_g"]) * 1e3)
        self.phase("close")
        t0 = d.time
        while d.time - t0 < B.T_CLOSE + B.T_SQUEEZE:
            self.step(gr["q_g"], np.zeros(7), RB.grip_ctrl_closing(w_open, d.time - t0))
        log["R0"], log["p0"] = self.rel()
        n = d.ncon
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        touch = self.is_obj[g1] ^ self.is_obj[g2]
        log["pads_on_object"] = int(self.is_pad[np.where(self.is_obj[g1], g2, g1)[touch]].sum())
        tcp, Rt = arm.tcp_pose(d)
        log["lever_arm_cm"] = float(abs(P.lever_arm(tcp, Rt[:, 1], d.xipos[self.obj])[0]) * 100)
        self.phase("test lift")
        grip = RB.grip_ctrl_closing(w_open, 1e9)
        up = B.line_qs(arm, self.dk, gr["p_g"], gr["p_g"] + [0, 0, TEST_H], gr["R_g"], gr["q_g"])
        reads = []
        t0 = d.time
        while d.time - t0 < TEST_T + TEST_HOLD:
            q, qd, qdd = B.along(up, TEST_T, d.time - t0)
            self.step(q, qd, grip, qdd)
            self.track(log)
            if d.time - t0 > TEST_T + SETTLE_SKIP:
                p, R = arm.tcp_pose(d)
                F, T = self.hand_wrench(p)
                if F[2] > 1e-3:
                    dx, dy = -T[1] / F[2], T[0] / F[2]            # (c - p) x (0, 0, -W) = -T, horizontally
                    lev = float(np.array([dx, dy, 0.0]) @ P.hinge_vector(R[:, 1])[0])
                    reads.append((F[2], p[0] + dx, p[1] + dy, lev))
        reads = np.array(reads) if reads else np.zeros((0, 4))
        W = float(np.median(reads[:, 0])) if len(reads) else 0.0
        k = max(1, len(reads) // 6)
        out = dict(log=log, gr=gr, q_top=up[-1], w_open=w_open, mass_meas=W / P.G, lifted=W > LIFTED_FRAC * B.MASS_KG * P.G)
        if len(reads):
            out["com_meas"] = np.array([np.median(reads[:, 1]), np.median(reads[:, 2]), 0.0])
            out["d_meas_cm"] = float(abs(np.median(reads[:, 3])) * 100)
            out["drift_mm"] = float(abs(np.median(reads[-k:, 3]) - np.median(reads[:k, 3])) * 1000)
        return out

    def set_down_and_park(self, res):
        d, arm, gr = self.d, self.arm, res["gr"]
        self.phase("set down")
        down = B.line_qs(arm, self.dk, gr["p_g"] + [0, 0, TEST_H], gr["p_g"], gr["R_g"], res["q_top"])
        grip = RB.grip_ctrl_closing(res["w_open"], 1e9)
        t0 = d.time
        while d.time - t0 < T_DOWN:
            q, qd, qdd = B.along(down, T_DOWN, d.time - t0)
            self.step(q, qd, grip, qdd)
        t0 = d.time
        while d.time - t0 < T_OPEN:
            self.step(down[-1], np.zeros(7), RB.MAX_OPEN / 2)
        a = gr["R_g"][:, 2]
        back = B.line_qs(arm, self.dk, gr["p_g"], gr["p_g"] - B.PRE * a, gr["R_g"], down[-1])
        t0 = d.time
        while d.time - t0 < T_BACK:
            q, qd, qdd = B.along(back, T_BACK, d.time - t0)
            self.step(q, qd, RB.MAX_OPEN / 2, qdd)
        self.phase("look again")
        t0 = d.time
        while d.time - t0 < T_PARK:
            q, qd, qdd = RB.min_jerk(back[-1], RB.Q_CAPTURE, T_PARK, d.time - t0)
            self.step(q, qd, 0.04, qdd)
        t0 = d.time
        while d.time - t0 < T_REST:
            self.step(RB.Q_CAPTURE, np.zeros(7), 0.04)

    def transport(self, res):
        """Lift the rest of the way, hold, carry, place and judge, as in run_b0.run_one."""
        d, arm, gr, log = self.d, self.arm, res["gr"], res["log"]
        grip = RB.grip_ctrl_closing(res["w_open"], 1e9)
        self.phase("lift and hold")
        t_lift = B.LIFT_T * np.sqrt((B.LIFT_H - TEST_H) / B.LIFT_H)       # same peak acceleration as B0's lift
        lift = B.line_qs(arm, self.dk, gr["p_g"] + [0, 0, TEST_H], gr["p_g"] + [0, 0, B.LIFT_H], gr["R_g"], res["q_top"])
        t0 = d.time
        while d.time - t0 < t_lift + B.HOLD_T:
            q, qd, qdd = B.along(lift, t_lift, d.time - t0)
            self.step(q, qd, grip, qdd)
            self.track(log)
        log["tilt_after_hold"] = log["tilt_max"]
        self.phase("carry")
        segs = B.carry_path(arm, self.dk, gr["p_g"] + [0, 0, B.LIFT_H], gr["R_g"], gr["R_b"], lift[-1])
        for qs_, T_ in (segs or []):
            t0 = d.time
            while d.time - t0 < T_:
                q, qd, qdd = B.along(qs_, T_, d.time - t0)
                self.step(q, qd, grip, qdd)
                self.track(log)
        if not segs:
            q_l = lift[-1]
            t0 = d.time
            while d.time - t0 < B.T_CARRY_JOINT:
                q, qd, qdd = RB.min_jerk(q_l, gr["q_b"], B.T_CARRY_JOINT, d.time - t0)
                self.step(q, qd, grip, qdd)
                self.track(log)
        q_top = segs[-1][0][-1] if segs else gr["q_b"]
        self.phase("place")
        drop = max(0.0, B.lowest_point(self.m, d, self.obj_geoms) - (S.TABLE_Z + B.RELEASE_CLEAR))
        p_b = S.BIN_CENTER + [0, 0, B.BIN_ABOVE]
        low = min(drop, B.BIN_ABOVE - 0.05)
        lower = B.line_qs(arm, self.dk, p_b, p_b - [0, 0, low], gr["R_b"], q_top)
        t0 = d.time
        while d.time - t0 < B.T_LOWER:
            q, qd, qdd = B.along(lower, B.T_LOWER, d.time - t0)
            self.step(q, qd, grip, qdd)
        t0 = d.time
        while d.time - t0 < B.T_RELEASE:
            self.step(lower[-1], np.zeros(7), RB.MAX_OPEN / 2)
        self.phase("release")
        retract = B.line_qs(arm, self.dk, p_b - [0, 0, low], p_b - [0, 0, low - B.RETRACT_H], gr["R_b"], lower[-1])
        t0 = d.time
        while d.time - t0 < B.T_RETRACT:
            q, qd, qdd = B.along(retract, B.T_RETRACT, d.time - t0)
            self.step(q, qd, RB.MAX_OPEN / 2, qdd)
        t0 = d.time
        while d.time - t0 < B.T_AFTER:
            self.step(retract[-1], np.zeros(7), RB.MAX_OPEN / 2)
        n_ = d.ncon
        g1_, g2_ = d.contact.geom1[:n_], d.contact.geom2[:n_]
        tch = self.is_obj[g1_] ^ self.is_obj[g2_]
        oth = np.where(self.is_obj[g1_], g2_, g1_)[tch]
        touching_hand = bool(np.isin(oth, list(arm.pad_geoms | arm.finger_mesh | {arm.palm})).any())
        in_bin = S.in_bin(d.xipos[self.obj], free_of_hand=not touching_hand)
        tilt_fail = log["tilt_max"] > B.MAX_TILT
        success = in_bin and not log["dropped"] and not tilt_fail
        if success:
            outcome = "success"
        elif log["pads_on_object"] == 0:
            outcome = "no grip (blocked on approach)" if log["tcp_error_mm"] > 5 else "no grip (closed on air)"
        elif tilt_fail and (not log["dropped"] or log["t_tilt"] <= log["t_drop"]):
            outcome = "turned in hand"
        elif log["dropped"]:
            outcome = "dropped"
        elif np.linalg.norm(d.xipos[self.obj][:2] - S.BIN_CENTER[:2]) > 1.0:
            outcome = "ejected at release (simulation)"
        else:
            outcome = "missed the bin"
        return dict(outcome=outcome, success=bool(success), in_bin=bool(in_bin), tilt_max=log["tilt_max"],
                    dropped=log["dropped"], final_com=[round(float(v), 4) for v in d.xipos[self.obj]])


def run_trial(args):
    tr, run, dev, out, policy, loo = args
    rec = dict(tr, policy=policy, mass_kg=B.MASS_KG, tries=[])
    try:
        T = Trial(tr, run, out)
        m2_model, known_model = models(dev, tr["name"] if loo else None)
        scene = dict(np.load(os.path.join(run, "scenes", tr["tag"] + ".npz")))
        cache = dict(np.load(os.path.join(run, "cache", "gsnet", tr["tag"] + ".npz")))
        mass_belief, com_belief, tried = B.MASS_KG, None, []
        q_start = RB.Q_CAPTURE
        for k in range(N_TRIES):
            cands = T.candidates(scene, cache, mass_belief, com_belief, exclude=tried)
            if not cands:
                rec.update(outcome="no reachable grasp" if k == 0 else "no regrasp available", success=False)
                return rec
            if policy == "retry":
                pick = cands[0]
            elif com_belief is None:
                pick = max(cands, key=lambda f: (m2_model(design([f]))[0], -f["cand"]))
            else:
                pick = max(cands, key=lambda f: (known_model(known_design([f]))[0], -f["cand"]))
            gr = pick["gr"]
            res = T.grasp_and_test(gr, q_start)
            log = res["log"]
            info = dict(cand=pick["cand"], grasp_index=gr["index"], flip=gr["flip"], score=gr["score"],
                        lever_arm_cm=log["lever_arm_cm"], pads_on_object=log["pads_on_object"],
                        tilt_test=log["tilt_max"], dropped_test=log["dropped"], lifted=res["lifted"],
                        mass_meas=res["mass_meas"], d_meas_cm=res.get("d_meas_cm"), drift_mm=res.get("drift_mm"))
            if "com_meas" in res:
                info["com_err_cm"] = float(np.linalg.norm((res["com_meas"] - T.d.xipos[T.obj])[:2]) * 100)
            sensed_bad = (not res["lifted"]) or res.get("drift_mm", 1e9) > DRIFT_MM
            if policy == "retry":
                bad = sensed_bad or log["tilt_max"] > B.MAX_TILT or log["dropped"]
            else:
                p_known = None
                if res["lifted"] and "d_meas_cm" in res:
                    f = dict(pick, d_known_cm=res["d_meas_cm"])
                    p_known = float(known_model(known_design([f]))[0])
                info["p_known"] = p_known
                bad = sensed_bad or (TAU > 0 and p_known is not None and p_known < TAU)
            info["regrasp"] = bool(bad and k + 1 < N_TRIES)
            rec["tries"].append(info)
            tried.append((gr["p_g"], gr["R_g"][:, 1]))
            if not bad or k + 1 == N_TRIES:
                rec.update(T.transport(res), n_tries=k + 1, tau_sat=T.tau_sat)
                return rec
            if res["lifted"] and "com_meas" in res and policy != "retry":
                com_belief, mass_belief = res["com_meas"], res["mass_meas"]
            T.set_down_and_park(res)
            name = f"{tr['tag']}__{policy}__{os.path.basename(os.path.normpath(out))}__t{k + 1}"
            path = T.capture(name)
            scene = dict(np.load(path))
            cache = T.detect(path, name)
            q_start = RB.Q_CAPTURE
        return rec
    except Exception as e:  # logged, never dropped
        import traceback
        rec.update(outcome="error", success=False, error=repr(e)[:300], trace=traceback.format_exc()[-800:])
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/b0_m045_v2", help="scenes + detections (and, for dev, the M2 data)")
    ap.add_argument("--dev", default="results/b0_m045_v2", help="development candidates the models are fitted on")
    ap.add_argument("--policy", choices=["agpp", "retry"], default="agpp")
    ap.add_argument("--out", default="results/m3_dev")
    ap.add_argument("--seeds", default="0:6")
    ap.add_argument("--only", default=None)
    ap.add_argument("--jobs", type=int, default=64)
    a = ap.parse_args()
    for sub in ("scenes", "cache"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    lo, hi = (int(v) for v in a.seeds.split(":"))
    trs = B.trials("results/objects/final_set.csv", range(lo, hi))
    if a.only:
        trs = [t for t in trs if t["tag"] in a.only.split(",")]
    loo = os.path.abspath(a.run) == os.path.abspath(a.dev)      # development scenes: leave the object out
    parts = os.path.join(a.out, f"{a.policy}_parts.jsonl")
    done = {}
    if os.path.exists(parts):
        for line in open(parts):
            r = json.loads(line)
            done[r["tag"]] = r
    todo = [t for t in trs if t["tag"] not in done]
    print(f"{a.policy}: {len(trs)} scenes, {len(done)} done, {len(todo)} to run (models {'leave-one-object-out' if loo else 'frozen'})",
          flush=True)
    with Pool(min(a.jobs, max(len(todo), 1)), maxtasksperchild=10) as pool, open(parts, "a") as f:
        for r in pool.imap_unordered(run_trial, [(t, a.run, a.dev, a.out, a.policy, loo) for t in todo], chunksize=1):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            done[r["tag"]] = r
    rows = [done[t["tag"]] for t in trs if t["tag"] in done]
    n_ok = sum(bool(r.get("success")) for r in rows)
    print(f"{a.policy}: success {n_ok}/{len(rows)} = {100 * n_ok / max(len(rows), 1):.1f}%; "
          f"regrasped {sum(len(r.get('tries', [])) > 1 for r in rows)}")
    from collections import Counter
    print(Counter(r.get("outcome") for r in rows))


if __name__ == "__main__":
    main()

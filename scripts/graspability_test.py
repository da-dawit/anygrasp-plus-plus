"""Final object-selection test (D16): does the frozen detector give usable grasps on each candidate?

    render    each candidate in its most probable resting pose that is stable at every shift, at several yaws;
              checks in MuJoCo that it stays put for 1 s at shift 0 and 8 cm; renders the overhead depth camera
              and saves the cloud (OpenCV frame) for detect_grasps.py
    evaluate  converts the cached grasps to the world and keeps those that are collision-free, approach within
              30 deg of vertical (D3), report width <= 8 cm (D14) and sit on the object (not the table)

    MUJOCO_GL=egl python scripts/graspability_test.py render --names names.txt --out results/objects/gtest
    python gpu/detect_grasps.py --detector gsnet --scenes results/objects/gtest/scenes --out results/objects/gtest/cache
    python scripts/graspability_test.py evaluate --out results/objects/gtest
"""
import argparse
import csv
import json
import os
import sys
import zlib
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anygrasp_pp import objects as O  # noqa: E402
from anygrasp_pp import scene as S  # noqa: E402

YAWS = (0, 90, 180, 270)
SETTLE_T = 1.0
MAX_WIDTH = 0.08
ON_OBJECT_Z = 0.005       # grasp centre at least this far above the table
MAX_SETTLE_MM = 5.0       # a resting pose that moves more than this in 1 s (at shift 0 or 8 cm) is not a rest


def place(ob, pose, yaw_deg):
    R, z, _, _ = pose
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    Rf = Rz @ R
    pos = np.array([S.PICK[0], S.PICK[1], S.TABLE_Z + z + 0.0005])
    pos[:2] -= (Rf @ ob["centroid"])[:2]
    return pos, Rf


def settle(model_dir, ob, pos, Rf, shift, collision):
    import mujoco
    xml = S.scene_xml([S.object_xml("obj", model_dir, ob, shift, pos, Rf, collision=collision)])
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    p0, q0 = d.xpos[m.body("obj").id].copy(), d.xquat[m.body("obj").id].copy()
    while d.time < SETTLE_T:
        mujoco.mj_step(m, d)
    p1, q1 = d.xpos[m.body("obj").id], d.xquat[m.body("obj").id]
    dq = np.empty(3)
    mujoco.mju_subQuat(dq, q1, q0)
    return float(np.linalg.norm(p1 - p0)), float(np.degrees(np.linalg.norm(dq)))


def render_one(args):
    name, out, tilt, models, collision, noise = args
    cam = S.camera_position(tilt)
    import mujoco
    model_dir = os.path.join(models, name)
    ob = O.build(model_dir, collision=collision)
    rec = dict(name=name, parts=ob["parts"], feasible=ob["core"] is not None)
    if ob["core"] is None:
        return [dict(rec, status="no core")]
    poses = S.rest_poses(ob)
    if not poses:
        return [dict(rec, status="no pose stable at every shift")]
    pose = poses[0]
    rows = []
    for yaw in YAWS:
        pos, Rf = place(ob, pose, yaw)
        moves = {s: settle(model_dir, ob, pos, Rf, s, collision) for s in (0.0, max(O.SHIFTS))}
        xml = S.scene_xml([S.object_xml("obj", model_dir, ob, 0.0, pos, Rf, collision=collision)], cam_pos=cam)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = [1, 0, 1, 0, 0, 0]
        with mujoco.Renderer(m, S.CAM_H, S.CAM_W) as r:
            r.update_scene(d, camera="overhead", scene_option=opt)
            rgb = r.render().copy()
            r.enable_depth_rendering()
            r.update_scene(d, camera="overhead", scene_option=opt)
            depth = r.render().copy()
            if noise:
                r.update_scene(d, camera="overhead_right", scene_option=opt)
                depth_r = r.render().copy()
        cid = m.camera("overhead").id
        x_cam, y_cam = d.cam_xmat[cid].reshape(3, 3)[:, 0], d.cam_xmat[cid].reshape(3, 3)[:, 1]
        R_wc, p_wc = S.R_world_cv(x_cam, y_cam), d.cam_xpos[cid].copy()
        K = S.camera_K()
        if noise:   # seeded per scene, so a rerun gives the same noisy image
            rng = np.random.default_rng(zlib.crc32(f"{name}|{yaw}".encode()))
            depth = S.realsense_depth(depth, depth_r, K, rng)
        pts = S.backproject(depth, K)
        wide, tight = S.workspace_masks(pts @ R_wc.T + p_wc)
        tag = f"{name}__yaw{yaw:03d}"
        np.savez_compressed(os.path.join(out, "scenes", tag + ".npz"), points=pts[tight].astype(np.float32),
                            scene_points=pts[wide].astype(np.float32), K=K, R_wc=R_wc, p_wc=p_wc)
        np.save(os.path.join(out, "rgb", tag + ".npy"), rgb[::2, ::2])
        rows.append(dict(rec, status="ok", tag=tag, yaw=yaw, pose_prob=pose[2], pose_margin=pose[3],
                         move0_mm=moves[0.0][0] * 1e3, rot0_deg=moves[0.0][1],
                         move8_mm=moves[max(O.SHIFTS)][0] * 1e3, rot8_deg=moves[max(O.SHIFTS)][1],
                         n_points=int(tight.sum())))
    return rows


def render(a):
    names = [n.strip() for n in open(a.names) if n.strip()]
    for sub in ("scenes", "rgb"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    with Pool(min(64, len(names))) as pool:
        rows = [r for rs in pool.map(render_one, [(n, a.out, a.cam_tilt, a.models, a.collision, not a.no_noise) for n in names]) for r in rs]
    json.dump(rows, open(os.path.join(a.out, "render.json"), "w"), indent=1)
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['name'][:50]:50s} {r['status']}")
    ok = [r for r in rows if r["status"] == "ok"]
    worst = max(ok, key=lambda r: max(r["move8_mm"], r["move0_mm"]))
    print(f"{len(ok)} scenes rendered; largest settle motion {max(worst['move0_mm'], worst['move8_mm']):.2f} mm "
          f"({worst['name']}, yaw {worst['yaw']})")


def evaluate(a):
    rows = json.load(open(os.path.join(a.out, "render.json")))
    per = {}
    for r in rows:
        if r["status"] != "ok":
            per.setdefault(r["name"], []).append(None)
            continue
        s = np.load(os.path.join(a.out, "scenes", r["tag"] + ".npz"))
        c = np.load(os.path.join(a.out, "cache", a.detector, r["tag"] + ".npz"))
        g = c["grasps"]
        Rw, tw = S.grasps_to_world(g, s["R_wc"], s["p_wc"])
        ok = (~c["collision"]) & S.approach_ok_world(Rw, a.max_deg) & (g[:, 1] <= MAX_WIDTH) & (tw[:, 2] > S.TABLE_Z + ON_OBJECT_Z)
        order = [i for i in c["order"] if ok[i]]
        r.update(n_raw=len(g), n_free=int((~c["collision"]).sum()), n_ok=int(ok.sum()), n_ok_nms=len(order),
                 top_score=float(g[order[0], 0]) if order else 0.0)
        per.setdefault(r["name"], []).append(r)
    out_rows = []
    for name, rs in sorted(per.items()):
        good = [r for r in rs if r]
        n_ok = [r["n_ok_nms"] for r in good]
        out_rows.append(dict(name=name, yaws=len(good), min_ok=min(n_ok) if n_ok else 0,
                             median_ok=float(np.median(n_ok)) if n_ok else 0.0,
                             max_settle_mm=max((max(r["move0_mm"], r["move8_mm"]) for r in good), default=np.nan),
                             passes=bool(n_ok) and min(n_ok) >= a.min_grasps and all(
                                 max(r["move0_mm"], r["move8_mm"]) <= MAX_SETTLE_MM for r in good)))
    with open(os.path.join(a.out, f"graspability_{a.detector}_{a.max_deg:.0f}deg.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    for r in sorted(out_rows, key=lambda r: -r["median_ok"]):
        print(f"  {'PASS' if r['passes'] else 'fail'} {r['name'][:52]:52s} usable grasps per yaw: min {r['min_ok']:3d} "
              f"median {r['median_ok']:5.1f}   settle <= {r['max_settle_mm']:.2f} mm")
    print(f"{sum(r['passes'] for r in out_rows)}/{len(out_rows)} objects pass (>= {a.min_grasps} usable grasps at every "
          f"yaw, settles within {MAX_SETTLE_MM:.0f} mm)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "evaluate"])
    ap.add_argument("--names")
    ap.add_argument("--models", default="/root/src/mujoco_scanned_objects/models")
    ap.add_argument("--collision", default="vhacd", help="convex pieces: vhacd (as scanned) or coacd")
    ap.add_argument("--no-noise", action="store_true", help="ideal depth instead of the D435 model (I.4)")
    ap.add_argument("--out", default="results/objects/gtest")
    ap.add_argument("--min-grasps", type=int, default=3)
    ap.add_argument("--detector", default="gsnet", choices=["gsnet", "baseline"])
    ap.add_argument("--max-deg", type=float, default=30.0, help="D3 limit on the approach angle")
    ap.add_argument("--cam-tilt", type=float, default=float(np.degrees(np.arctan2(0.25, 0.60))),
                    help="camera angle off vertical [deg] (render)")
    a = ap.parse_args()
    render(a) if a.cmd == "render" else evaluate(a)

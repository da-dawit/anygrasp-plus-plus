"""Offline evaluation of re-ranking policies on executed candidates (D22).

`run_b0.py candidates` executes every scene's first K reachable grasps (detector order), so each policy that picks
one of those K grasps can be scored exactly with the recorded outcome of the grasp it picks. Policies:

    B0              the first reachable grasp (the detector's choice)
    centre          the grasp closest to the visible centre of the object (lever arm about the closing axis)
    M2              detector score x P(hold), CoM uniform over the visible footprint (C.5), d* from the virtual
                    squeeze of the cloud (C.6)
    M2, known CoM   P(hold) with the true CoM (what M2 could do after an ideal M3 test lift)
    oracle          success if any of the K grasps succeeded (the ceiling for re-ranking these K)

This is the development set (seeds 0-5). The final comparison uses fresh seeds.

    python scripts/eval_rerank.py --run results/b0_m045
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anygrasp_pp import m2 as M2  # noqa: E402
from anygrasp_pp import physics as P  # noqa: E402
from anygrasp_pp import robot as RB  # noqa: E402
from anygrasp_pp import scene as S  # noqa: E402
from anygrasp_pp.rig_trial import a_peak  # noqa: E402


def features(args):
    r, run, mass = args
    sc = np.load(os.path.join(run, "scenes", r["tag"] + ".npz"))
    ca = np.load(os.path.join(run, "cache", "gsnet", r["tag"] + ".npz"))
    i = int(r["grasp_index"])
    Rw, tw = S.grasps_to_world(ca["grasps"][i:i + 1], sc["R_wc"], sc["p_wc"])
    R = Rw[0].copy()
    if r["flip"] == "True":
        R[:, 1:] *= -1
    p_g, R_g = RB.grasp_to_tcp(R, tw[0], float(r["depth"]))
    w_open = min(float(r["width"]) + 0.01, RB.MAX_OPEN)
    pts = M2.object_points(sc["points"] @ sc["R_wc"].T + sc["p_wc"])
    ds, _ = M2.predicted_dstar(pts, p_g, R_g, w_open, mass, accel=a_peak())   # _: cloud point counts
    c0, V, half = M2.com_prior(pts)
    axis = R_g[:, 1]
    ph = float(np.ravel(P.p_hold_uniform(p_g, axis, c0, V, half, mass, ds=ds))[0]) if ds > 0 else 0.0
    d_vis = float(abs(P.lever_arm(p_g, axis, c0)[0]))
    return dict(r, score=float(ca["grasps"][i, 0]), dstar_pred=ds, p_hold=ph, d_vis=d_vis,
                n_pts_cloud_left=_.get("n_left"), n_pts_cloud_right=_.get("n_right"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/b0_m045")
    ap.add_argument("--mass", type=float, default=0.45)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(os.path.join(a.run, "candidates.csv")))
            if r["grasp_index"] not in ("", "-1") and r.get("depth")]
    with Pool(96) as pool:
        feats = pool.map(features, [(r, a.run, a.mass) for r in rows])
    by = defaultdict(list)
    for f in feats:
        by[f["tag"]].append(f)
    scenes = sorted({r["tag"] for r in csv.DictReader(open(os.path.join(a.run, "candidates.csv")))})
    succ = lambda f: f["success"] == "True"
    dstar_true = lambda f: float(f["dstar_lift_cm"] or 0) / 100

    def known_com(f):    # ideal M3: true |d|, predicted d*
        return 1.0 if float(f["lever_arm_cm"] or 99) / 100 < f["dstar_pred"] else 0.0

    policies = {
        "B0": lambda c: min(c, key=lambda f: int(f["cand"])),
        "centre": lambda c: min(c, key=lambda f: f["d_vis"]),
        "M2": lambda c: max(c, key=lambda f: (f["score"] * f["p_hold"], -int(f["cand"]))),
        "M2, known CoM": lambda c: max(c, key=lambda f: (known_com(f), f["dstar_pred"] - float(f["lever_arm_cm"] or 99) / 100, -int(f["cand"]))),
    }
    res = {k: defaultdict(lambda: [0, 0]) for k in list(policies) + ["oracle"]}
    for tag in scenes:
        c = by.get(tag, [])
        shift = tag.split("__s")[1].split("__")[0]
        for k, pol in policies.items():
            res[k][shift][1] += 1
            if c and succ(pol(c)):
                res[k][shift][0] += 1
        res["oracle"][shift][1] += 1
        if any(succ(f) for f in c):
            res["oracle"][shift][0] += 1
    shifts = sorted({t.split("__s")[1].split("__")[0] for t in scenes}, key=int)
    print(f"{len(scenes)} scenes, {len(feats)} executed candidates (up to {max(len(v) for v in by.values())} per scene)")
    print(f"{'policy':16s} " + " ".join(f"{s + ' cm':>9s}" for s in shifts) + "     all")
    for k in list(policies) + ["oracle"]:
        tot = [sum(res[k][s][0] for s in shifts), sum(res[k][s][1] for s in shifts)]
        print(f"{k:16s} " + " ".join(f"{res[k][s][0]:4d}/{res[k][s][1]:<4d}" for s in shifts)
              + f"  {tot[0]:4d}/{tot[1]} = {100 * tot[0] / tot[1]:.0f}%")
    with open(os.path.join(a.run, "candidates_features.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(feats[0]))
        w.writeheader()
        w.writerows(feats)


if __name__ == "__main__":
    main()

"""Compare the slip distance M2 predicts from the D435 cloud alone with the one measured from the sim contacts
(run_b0.contact_dstar), and how well each predicts whether the grasp held. Writes <run>/m2_validation.csv.

    python scripts/validate_m2.py --run results/b0_m045 --trials trials.csv
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anygrasp_pp import m2 as M2  # noqa: E402
from anygrasp_pp import robot as RB  # noqa: E402
from anygrasp_pp import scene as S  # noqa: E402
from anygrasp_pp.rig_trial import a_peak  # noqa: E402


def one(args):
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
    ds, info = M2.predicted_dstar(pts, p_g, R_g, w_open, mass, accel=a_peak())
    return dict(tag=r["tag"], cand=r.get("cand", "0"), dstar_pred_cm=ds * 100, **info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/b0_m045")
    ap.add_argument("--trials", default="trials.csv")
    ap.add_argument("--mass", type=float, default=0.45)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(os.path.join(a.run, a.trials)))
            if r.get("dstar_lift_cm") not in ("", None) and r["grasp_index"] not in ("", "-1")]
    with Pool(64) as pool:
        pred = pool.map(one, [(r, a.run, a.mass) for r in rows])
    dm = np.array([float(r["dstar_lift_cm"]) for r in rows])
    dp = np.array([p["dstar_pred_cm"] for p in pred])
    d = np.array([float(r["lever_arm_cm"]) for r in rows])
    held = np.array([float(r["tilt_after_hold"]) <= 10 for r in rows])
    mir = np.array([p["mirrored"] for p in pred])
    print(f"{len(rows)} executed grasps; predicted from the cloud vs measured from the contacts:")
    print(f"  d* median: predicted {np.median(dp):.2f} cm, measured {np.median(dm):.2f} cm; "
          f"Spearman rank correlation {np.corrcoef(np.argsort(np.argsort(dp)), np.argsort(np.argsort(dm)))[0, 1]:.2f}; "
          f"one side mirrored in {100 * mir.mean():.0f}%")
    for name, ds in (("measured d*", dm), ("predicted d*", dp)):
        ok = d < ds
        print(f"  |d| < {name:13s}: agrees with held in {100 * np.mean(ok == held):.0f}%  "
              f"(predicted hold: {held[ok].mean() * 100 if ok.any() else float('nan'):.0f}% held, n {ok.sum()}; "
              f"predicted fail: {100 - held[~ok].mean() * 100:.0f}% failed, n {(~ok).sum()})")
    with open(os.path.join(a.run, "m2_validation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(pred[0]))
        w.writeheader()
        w.writerows(pred)


if __name__ == "__main__":
    main()

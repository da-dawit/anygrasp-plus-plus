"""Calibrated M2 (D23): a small logistic model over physics quantities the robot can observe before touching the
object, evaluated leave-one-object-out on the executed candidates, so the model that ranks an object's grasps has
never seen that object. Re-ranking is then scored exactly as in eval_rerank.py (the picked grasp's recorded outcome).

Features (all observable): predicted d* (cloud virtual squeeze, C.6), lever arm about the visible centre, their
ratio, the uniform-prior P(hold) (C.5), the detector score, the approach angle, the reported width and depth.

    python scripts/m2_learn.py --run results/b0_m045_v2
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

FEATS = ["dstar_pred_cm", "d_vis_cm", "ratio", "p_hold", "score", "approach_deg", "width_cm", "depth_cm"]


def design(rows):
    X = []
    for f in rows:
        ds, dv = 100 * float(f["dstar_pred"]), 100 * float(f["d_vis"])
        X.append([ds, dv, dv / max(ds, 0.1), float(f["p_hold"]), float(f["score"]), float(f["approach_deg"]),
                  100 * float(f["width"]), 100 * float(f["depth"])])
    return np.array(X)


def fit_logistic(X, y, l2=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    def nll(w):
        z = Z @ w
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * l2 * np.sum(w[1:] ** 2)
    def grad(w):
        p = 1 / (1 + np.exp(-(Z @ w)))
        g = Z.T @ (p - y)
        g[1:] += l2 * w[1:]
        return g
    w = minimize(nll, np.zeros(Z.shape[1]), jac=grad, method="L-BFGS-B").x
    return lambda Xn: 1 / (1 + np.exp(-(np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ w))), w


def auc(p, y):
    order = np.argsort(p)
    r = np.empty(len(p))
    r[order] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/b0_m045_v2")
    a = ap.parse_args()
    F = [f for f in csv.DictReader(open(os.path.join(a.run, "candidates_features.csv")))]
    X, y = design(F), np.array([f["success"] == "True" for f in F], float)
    objs = np.array([f["name"] for f in F])
    p_cv = np.zeros(len(F))
    for o in sorted(set(objs)):
        tr, te = objs != o, objs == o
        model, _ = fit_logistic(X[tr], y[tr])
        p_cv[te] = model(X[te])
    _, w_all = fit_logistic(X, y)
    print(f"{len(F)} executed candidates, {int(y.sum())} successes; leave-one-object-out AUC {auc(p_cv, y):.3f}")
    print("  standardized weights (all data): " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(FEATS, w_all[1:])))
    for i, n in enumerate(FEATS):
        print(f"  AUC of {n:14s} alone: {max(auc(X[:, i], y), 1 - auc(X[:, i], y)):.3f}")
    by = defaultdict(list)
    for f, p in zip(F, p_cv):
        by[f["tag"]].append((p, f))
    scenes = sorted({r["tag"] for r in csv.DictReader(open(os.path.join(a.run, "candidates.csv")))})
    n = len(scenes)
    pick = lambda key: sum(1 for t in scenes if by.get(t) and max(by[t], key=key)[1]["success"] == "True")
    b0 = pick(lambda pf: -int(pf[1]["cand"]))
    learned = pick(lambda pf: (pf[0], -int(pf[1]["cand"])))
    oracle = sum(1 for t in scenes if any(f["success"] == "True" for _, f in by.get(t, [])))
    print(f"re-ranking the executed top-10 (held-out objects): B0 {b0}/{n} = {100 * b0 / n:.0f}%, "
          f"calibrated M2 {learned}/{n} = {100 * learned / n:.0f}%, oracle {oracle}/{n} = {100 * oracle / n:.0f}%")


if __name__ == "__main__":
    main()

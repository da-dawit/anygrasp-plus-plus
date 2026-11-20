"""Midpoint analysis of the B0 trials (proposal: Minimum tier; derivations G.3, G.4).

    success by shift and by object, outcome counts
    B0 success with no shift (proposal criterion: >= 80%)
    failure point: success vs the true lever arm d at the grasp (|d|, cm), logistic P = 1/(1+exp(k(d-d50))) with
    k bounded (simulated outcomes are nearly separable, G.4), d50 with a 95% bootstrap interval resampled by
    object (G.3); compared with the slip distance d* predicted before any trial (A.12, D15)

Trials where the hand never gripped the object (blocked on approach, closed on air, no reachable grasp) have no
meaningful lever arm; they count as B0 failures in the success rates but are left out of the d50 fit.

    python scripts/analyze_b0.py --trials results/b0/trials.csv --out results/b0
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anygrasp_pp import physics as P  # noqa: E402
from anygrasp_pp.contact_rig import grid_points  # noqa: E402
from anygrasp_pp.rig_trial import a_peak  # noqa: E402

K_MAX = 50.0      # 1/cm: logistic slope bound (a 10-90% width of 0.09 cm, finer than the shift steps)
MASS = 1.0        # D15


def fit_d50(d, y):
    """Bounded maximum-likelihood logistic fit; returns (d50, k)."""
    def nll(x):
        d50, k = x
        z = k * (d - d50)
        return np.sum(np.logaddexp(0, z) * y + np.logaddexp(0, -z) * (1 - y))
    best = None
    for d0 in np.linspace(max(d.min(), 0.5), d.max(), 12):
        r = minimize(nll, [d0, 2.0], bounds=[(0.0, 30.0), (0.05, K_MAX)], method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="results/b0/trials.csv")
    ap.add_argument("--out", default="results/b0")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.trials)))
    for r in rows:
        r["success"] = r["success"] == "True"
        r["shift"] = float(r["shift"])
    n = len(rows)
    out = {"n_trials": n, "outcomes": Counter(r["outcome"] for r in rows)}
    print(f"{n} trials; outcomes: {dict(out['outcomes'])}")
    by_shift = {}
    for s in sorted({r["shift"] for r in rows}):
        rs = [r for r in rows if r["shift"] == s]
        by_shift[f"{s * 100:.0f}"] = (sum(r["success"] for r in rs), len(rs))
    out["success_by_shift_cm"] = by_shift
    print("success by shift:", {k: f"{v[0]}/{v[1]}" for k, v in by_shift.items()})
    by_obj = defaultdict(lambda: [0, 0])
    for r in rows:
        by_obj[r["name"]][0] += r["success"]
        by_obj[r["name"]][1] += 1
    out["success_by_object"] = dict(by_obj)
    s0 = by_shift["0"]
    out["no_shift_success"] = s0[0] / s0[1]
    print(f"no-shift success {s0[0]}/{s0[1]} = {100 * s0[0] / s0[1]:.0f}% (proposal criterion >= 80%)")

    gripped = [r for r in rows if r.get("lever_arm_cm") not in ("", None) and not r["outcome"].startswith("no grip")]
    d = np.array([float(r["lever_arm_cm"]) for r in gripped])
    y = np.array([float(r["success"]) for r in gripped])
    objs = np.array([r["name"] for r in gripped])
    d50, k = fit_d50(d, y)
    rng = np.random.default_rng(0)
    names = sorted(set(objs))
    boots = []
    for _ in range(a.boot):
        pick = rng.choice(names, len(names), replace=True)
        idx = np.concatenate([np.flatnonzero(objs == o) for o in pick])
        if 0 < y[idx].sum() < len(idx):
            boots.append(fit_d50(d[idx], y[idx])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    pts = grid_points()
    dstar_static = float(P.patch_dstar(pts, mu=P.MU, fn=P.FN, m=MASS)) * 100
    dstar_lift = float(P.patch_dstar(pts, mu=P.MU, fn=P.FN, m=MASS, accel=a_peak())) * 100
    hold, fail = d[y == 1], d[y == 0]
    out.update(gripped_trials=len(gripped), d50_cm=d50, k_per_cm=k, d50_ci95_cm=[lo, hi],
               dstar_cm=[dstar_lift, dstar_static], largest_hold_cm=float(hold.max()) if len(hold) else None,
               smallest_fail_cm=float(fail.min()) if len(fail) else None)
    print(f"failure point over {len(gripped)} gripped trials: d50 = {d50:.2f} cm (95% CI {lo:.2f}-{hi:.2f}, "
          f"object bootstrap), slope k = {k:.2f}/cm; predicted d* = {dstar_lift:.2f}-{dstar_static:.2f} cm; "
          f"proposal criterion: within 1.5 cm")
    if len(hold) and len(fail):
        print(f"largest lever arm that held: {hold.max():.2f} cm; smallest that failed: {fail.min():.2f} cm")
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    jit = rng.uniform(-0.03, 0.03, len(y))
    ax.scatter(d, y + jit, s=12, alpha=0.6, c=np.where(y == 1, "tab:green", "tab:red"))
    xs = np.linspace(0, max(d.max(), 10), 400)
    ax.plot(xs, 1 / (1 + np.exp(k * (xs - d50))), "k-", lw=1.5, label=f"fit: d50 = {d50:.2f} cm")
    ax.axvspan(dstar_lift, dstar_static, color="tab:blue", alpha=0.25, label=f"predicted d* {dstar_lift:.2f}-{dstar_static:.2f} cm")
    ax.set_xlabel("lever arm |d| at the grasp [cm]")
    ax.set_ylabel("B0 success")
    ax.set_title(f"B0, {len(gripped)} gripped trials, 10 objects")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "b0_success_vs_lever_arm.png"), dpi=120)


if __name__ == "__main__":
    main()

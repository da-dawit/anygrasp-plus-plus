"""Figures for the midpoint memo and the project page, from committed results.

    fig_methods.png   success of each method, development scenes (held-out objects) and fresh scenes, 95% CI
    fig_shift.png     success vs the hidden CoM shift
    fig_physics.png   held through lift and hold vs |d| / d* of each grasp's own contact patch
    fig_failure.png   contact points and per-grasp d* with the step-1 pad and the rubber-layer pad (D19, D20)

    python scripts/make_figures.py --out results/midpoint
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_b0 import fit_d50  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False})
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def rows(path):
    return list(csv.DictReader(open(path)))


def boot_rate(ok, objs, n=4000, seed=0):
    ok, objs = np.asarray(ok, float), np.asarray(objs)
    names = sorted(set(objs))
    idx = {o: np.flatnonzero(objs == o) for o in names}
    rng = np.random.default_rng(seed)
    b = [ok[np.concatenate([idx[o] for o in rng.choice(names, len(names))])].mean() for _ in range(n)]
    return ok.mean(), np.percentile(b, [2.5, 97.5])


def m3_rows(path):
    if not os.path.exists(path):
        return None
    return [json.loads(line) for line in open(path)]


def methods(out, m3_dirs):
    """Bars: B0, B0 + blind retry, M2, AnyGrasp++ (M2 + M3), best of the executed grasps; dev and fresh scenes."""
    groups = []
    for label, pol_file, m3 in (("development scenes\n(objects held out)", "policies.json", m3_dirs.get("dev")),
                                ("fresh scenes", "policies_fresh.json", m3_dirs.get("fresh"))):
        pol = json.load(open(os.path.join(out, pol_file)))["groups"]["all shifts"]
        k = list(pol)
        bars = [("B0", pol[k[0]]["rate"], pol[k[0]]["ci"])]
        if m3 and m3.get("retry"):
            r = m3["retry"]
            bars.append(("B0 + blind retry", *boot_rate([x.get("success", False) for x in r], [x["name"] for x in r])))
        bars.append(("M2", pol[k[4]]["rate"], pol[k[4]]["ci"]))
        if m3 and m3.get("agpp"):
            r = m3["agpp"]
            bars.append(("AnyGrasp++ (M2 + M3)", *boot_rate([x.get("success", False) for x in r], [x["name"] for x in r])))
        bars.append(("best of the executed grasps", pol[k[6]]["rate"], pol[k[6]]["ci"]))
        groups.append((label, bars))
    col = {"B0": C[7], "B0 + blind retry": C[1], "M2": C[0], "AnyGrasp++ (M2 + M3)": C[2],
           "best of the executed grasps": "0.8"}
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    seen = set()
    for gi, (label, bars) in enumerate(groups):
        w = 0.8 / len(bars)
        for bi, (n, r, ci) in enumerate(bars):
            x = gi + (bi - (len(bars) - 1) / 2) * w
            ax.bar(x, 100 * r, w * 0.92, color=col[n], label=None if n in seen else n)
            seen.add(n)
            ax.errorbar(x, 100 * r, yerr=[[100 * (r - ci[0])], [100 * (ci[1] - r)]], color="k", lw=0.8, capsize=2)
            ax.text(x, 100 * ci[1] + 1.5, f"{100 * r:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylabel("success (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(os.path.join(out, "fig_methods.png"))
    plt.close(fig)


def shift_curve(explorer, out, m3_dev):
    shifts = [0, 2, 4, 6, 8]

    def rate(path):
        R = rows(path)
        return [100 * np.mean([r["success"] == "True" for r in R if round(float(r["shift"]) * 100) == s]) for s in shifts]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(shifts, rate("results/b0_m045_v2/trials.csv"), "o-", color=C[7], label="B0")
    ax.plot(shifts, rate("results/b0_v2/trials.csv"), "s--", color=C[7], alpha=0.6, label="B0, 1 kg objects")
    m2 = [100 * np.mean([bool(e["cands"]) and next(f for f in e["cands"] if f["cand"] == e["m2"])["ok"]
                         for e in explorer if e["shift"] == s]) for s in shifts]
    ax.plot(shifts, m2, "o-", color=C[0], label="M2")
    if m3_dev:
        for pol, c, lab in (("retry", C[1], "B0 + blind retry"), ("agpp", C[2], "AnyGrasp++")):
            if m3_dev.get(pol):
                r = m3_dev[pol]
                ax.plot(shifts, [100 * np.mean([x.get("success", False) for x in r if round(float(x["shift"]) * 100) == s])
                                 for s in shifts], "o-", color=c, label=lab)
    ceil = [100 * np.mean([any(f["ok"] for f in e["cands"]) for e in explorer if e["shift"] == s]) for s in shifts]
    ax.plot(shifts, ceil, ":", color="0.5", label="best of the executed grasps")
    ax.set_xticks(shifts)
    ax.set_xlabel("hidden CoM shift (cm)")
    ax.set_ylabel("success (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(out, "fig_shift.png"))
    plt.close(fig)


def per_grasp(path):
    has = [r for r in rows(path) if r.get("lever_arm_cm") not in ("", None) and not r["outcome"].startswith("no grip")
           and r.get("dstar_lift_cm") not in ("", None) and float(r["dstar_lift_cm"]) > 0]
    ratio = np.array([float(r["lever_arm_cm"]) / float(r["dstar_lift_cm"]) for r in has])
    held = np.array([float(float(r["tilt_after_hold"]) <= 10.0) for r in has])
    return ratio, held


def physics(out):
    ratio, held = per_grasp("results/b0_m045_v2/trials.csv")
    r50, k = fit_d50(ratio, held)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    rng = np.random.default_rng(1)
    ax.scatter(np.minimum(ratio, 7.9), held + rng.uniform(-0.03, 0.03, len(held)), s=8, alpha=0.4, color=C[0], lw=0)
    edges = [0, 0.5, 0.9, 1.1, 1.5, 2, 3, 8, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ratio >= lo) & (ratio < hi)
        if sel.sum() >= 5:
            p = held[sel].mean()
            ax.errorbar(np.median(np.minimum(ratio[sel], 7.9)), p, yerr=1.96 * np.sqrt(p * (1 - p) / sel.sum()),
                        fmt="o", color="k", mfc="white", capsize=2)
    x = np.linspace(0, 8, 400)
    ax.plot(x, 1 / (1 + np.exp(k * (x - r50))), "k-", lw=1, label=f"logistic fit (50% at {r50:.2f})")
    ax.axvline(1, color=C[3], lw=1, ls="--", label="model: holds if |d| < d*")
    ax.set_xlim(0, 8)
    ax.set_xlabel("|d| / d* (lever arm over this grasp's slip distance)")
    ax.set_ylabel("held through lift and hold")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.savefig(os.path.join(out, "fig_physics.png"))
    plt.close(fig)


def all_contacts(path):
    """D19's counting: every trial with a measured lever arm, including grasps where one pad never touched."""
    g = [r for r in rows(path) if r.get("lever_arm_cm") not in ("", None)]
    ds = np.array([float(r["dstar_lift_cm"] or 0) for r in g])
    ratio = np.array([float(r["lever_arm_cm"]) for r in g]) / np.maximum(ds, 1e-9)
    held = np.array([float(float(r["tilt_after_hold"] or 99) <= 10.0) for r in g])
    pts = np.array([min(int(r["n_pts_left"] or 0), int(r["n_pts_right"] or 0)) for r in g])
    return ratio, held, pts, ds


def failure(out):
    before = all_contacts("results/b0/trials_pad_v1.csv")
    after = all_contacts("results/b0/trials.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for data, c, lab in ((before, C[3], "step-1 pad"), (after, C[0], "rubber-layer pad")):
        axs[0].hist(np.minimum(data[2], 16), bins=np.arange(18) - 0.5, color=c, alpha=0.5, label=lab)
        axs[1].hist(np.minimum(data[3], 8), bins=np.linspace(0, 8, 33), color=c, alpha=0.5, label=lab)
    axs[0].set_xlabel("contact points on the weaker pad (of 16)")
    axs[0].set_ylabel("grasps")
    axs[1].set_xlabel("slip distance d* of the grasp (cm)")
    axs[0].legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(out, "fig_failure.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/midpoint")
    ap.add_argument("--m3-dev", default="results/m3_dev")
    ap.add_argument("--m3-fresh", default="results/m3_fresh")
    a = ap.parse_args()
    explorer = json.load(open(os.path.join(a.out, "explorer.json")))
    m3 = {k: {p: m3_rows(os.path.join(d, f"{p}_parts.jsonl")) for p in ("agpp", "retry")}
          for k, d in (("dev", a.m3_dev), ("fresh", a.m3_fresh))}
    methods(a.out, m3)
    shift_curve(explorer, a.out, m3["dev"])
    physics(a.out)
    failure(a.out)
    for f in ("fig_policies.png", "fig_policies_fresh.png"):     # replaced by fig_methods.png
        if os.path.exists(os.path.join(a.out, f)):
            os.remove(os.path.join(a.out, f))
    print("figures written to", a.out)


if __name__ == "__main__":
    main()

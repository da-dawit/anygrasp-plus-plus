"""Figures for the midpoint memo and the project page, all from committed results:

    fig_policies.png   success of each grasp-choice policy on the development scenes, 95% CI (object bootstrap)
    fig_shift.png      success vs the hidden CoM shift: B0 at 0.45 and 1 kg, calibrated M2, best of the executed grasps
    fig_physics.png    per-grasp check of the friction model: held through lift and hold vs |d| / d*_grasp
    fig_failure.png    the first failure mode, before and after: contact points and per-grasp d* with the step-1 pad
                       and with the rubber-layer pad (D19, D20)

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

INK, MUTED, GRID = "#1f2430", "#5c6475", "#e6e8ee"
B0C, M2C, RETRY, CEIL, HEAVY = "#8a93a6", "#1b8a7a", "#d08a2c", "#c9d3e6", "#4f5d7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
                     "axes.titlesize": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


def rows(path):
    return list(csv.DictReader(open(path)))


def policies(pol, out, name="fig_policies.png", note="0.45 kg, development seeds 0-5."):
    names = list(pol["groups"]["all shifts"])
    short = {"B0 (detector's first reachable grasp)": "B0: detector's first grasp",
             "B0 + blind retry (upper bound)": "B0 + one blind retry*",
             "Closest to the visible centre": "Closest to the visible centre",
             "M2 v1 (physics only)": "M2 v1, physics only",
             "M2 calibrated (held-out objects)": "M2 calibrated (held-out objects)",
             "M2 calibrated (frozen, fresh scenes)": "M2 calibrated (frozen)",
             "M2 calibrated + blind retry (upper bound)": "M2 calibrated + one retry*",
             "Best of the executed grasps (ceiling)": "Best of the executed grasps"}
    col = {n: B0C for n in names}
    col.update({names[1]: RETRY, names[5]: RETRY, names[4]: M2C, names[3]: "#7fb8ae", names[2]: "#a7b0c2",
                names[6]: CEIL})
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, g in zip(axs, ("all shifts", "6-8 cm shift")):
        G = pol["groups"][g]
        y = np.arange(len(names))[::-1]
        r = np.array([G[n]["rate"] for n in names]) * 100
        lo = np.array([G[n]["ci"][0] for n in names]) * 100
        hi = np.array([G[n]["ci"][1] for n in names]) * 100
        ax.barh(y, r, color=[col[n] for n in names], height=0.66,
                hatch=["//" if "retry" in n else "" for n in names], edgecolor="white", linewidth=0.5)
        ax.errorbar(r, y, xerr=[r - lo, hi - r], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5, alpha=0.7)
        for yi, ri, n in zip(y, r, names):
            ax.text(ri + 1.2, yi + 0.18, f"{ri:.0f}%", va="center", ha="left", fontsize=9.5, color=INK,
                    fontweight="bold" if n == names[4] else "normal")
        ax.axvline(G[names[0]]["rate"] * 100, color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(0, 100)
        ax.set_title(f"{g} ({G[names[0]]['n']} scenes)", loc="left")
        ax.set_xlabel("pick-and-place success [%]")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axs[0].set_yticks(np.arange(len(names))[::-1], [short[n] for n in names])
    fig.text(0.01, -0.07, "* upper bound: any failure triggers the retry, and the second grasp is scored on the "
             "undisturbed scene.\nBars: 95% CI, bootstrap over objects. " + note,
             fontsize=8.5, color=MUTED)
    fig.savefig(os.path.join(out, name))
    plt.close(fig)


def shift_curve(explorer, out):
    shifts = [0, 2, 4, 6, 8]

    def rate(path):
        R = rows(path)
        return [100 * np.mean([r["success"] == "True" for r in R if round(float(r["shift"]) * 100) == s]) for s in shifts]
    b0_045, b0_1 = rate("results/b0_m045_v2/trials.csv"), rate("results/b0_v2/trials.csv")
    m2 = [100 * np.mean([bool(e["cands"]) and next(f for f in e["cands"] if f["cand"] == e["m2"])["ok"]
                         for e in explorer if e["shift"] == s]) for s in shifts]
    ceil = [100 * np.mean([any(f["ok"] for f in e["cands"]) for e in explorer if e["shift"] == s]) for s in shifts]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.fill_between(shifts, b0_045, ceil, color=CEIL, alpha=0.45, lw=0, label="room for re-ranking")
    ax.plot(shifts, ceil, "-", color="#9aa7c4", lw=1.5, marker="o", ms=4, label="best of the executed grasps")
    ax.plot(shifts, m2, "-", color=M2C, lw=2.4, marker="o", ms=5, label="M2 calibrated (held-out objects)")
    ax.plot(shifts, b0_045, "-", color=B0C, lw=2.4, marker="s", ms=5, label="B0, 0.45 kg (9 objects)")
    ax.plot(shifts, b0_1, "--", color=HEAVY, lw=1.6, marker="s", ms=4, label="B0, 1 kg (10 objects)")
    ax.set_xticks(shifts, [f"{s} cm" for s in shifts])
    ax.set_xlabel("hidden centre-of-mass shift")
    ax.set_ylabel("success [%]")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.8, loc="upper right", ncol=1)
    ax.set_title("Success vs the hidden shift (54 scenes per point)", loc="left")
    fig.savefig(os.path.join(out, "fig_shift.png"))
    plt.close(fig)


def per_grasp(path):
    has = [r for r in rows(path) if r.get("lever_arm_cm") not in ("", None) and not r["outcome"].startswith("no grip")
           and r.get("dstar_lift_cm") not in ("", None) and float(r["dstar_lift_cm"]) > 0]
    ratio = np.array([float(r["lever_arm_cm"]) / float(r["dstar_lift_cm"]) for r in has])
    held = np.array([float(float(r["tilt_after_hold"]) <= 10.0) for r in has])
    pts = np.array([min(int(r["n_pts_left"]), int(r["n_pts_right"])) for r in has])
    ds = np.array([float(r["dstar_lift_cm"]) for r in has])
    return ratio, held, pts, ds


def physics(out, summary):
    ratio, held, _, _ = per_grasp("results/b0_m045_v2/trials.csv")
    r50, k = fit_d50(ratio, held)
    edges = [0, 0.5, 0.9, 1.1, 1.5, 2, 3, 8, np.inf]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    rng = np.random.default_rng(1)
    x = np.minimum(ratio, 7.9)
    ax.scatter(x, held + rng.uniform(-0.035, 0.035, len(held)), s=11, alpha=0.45,
               c=np.where(held == 1, M2C, "#c4553f"), lw=0)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ratio >= lo) & (ratio < hi)
        if sel.sum() >= 5:
            p, n = held[sel].mean(), sel.sum()
            se = np.sqrt(p * (1 - p) / n + 1e-9)
            xc = np.median(np.minimum(ratio[sel], 7.9))
            ax.errorbar(xc, p, yerr=1.96 * se, fmt="o", color=INK, ms=6, mfc="white", mew=1.6, capsize=2.5)
    xr = np.linspace(0, 8, 400)
    ax.plot(xr, 1 / (1 + np.exp(k * (xr - r50))), color=INK, lw=1.6, label=f"logistic fit: 50% at {r50:.2f}")
    ax.axvline(1, color=M2C, lw=2, alpha=0.35, label="friction model: holds while |d| < d*")
    lo, hi = summary["per_grasp"]["ratio50_ci95"]
    ax.axvspan(lo, hi, color=M2C, alpha=0.08, lw=0, label=f"95% CI of the 50% point ({lo:.2f}-{hi:.2f})")
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel("lever arm / this grasp's own slip distance   |d| / d*")
    ax.set_ylabel("held through lift and 2 s hold")
    ax.legend(frameon=False, fontsize=8.6, loc="center right")
    ax.set_title(f"The friction model predicts each grasp ({len(held)} B0 grasps)", loc="left")
    ax.text(7.9, 0.22, f"agrees with the outcome\nin {100 * summary['per_grasp']['agreement']:.0f}% of grasps",
            ha="right", va="center", fontsize=8.6, color=MUTED)
    fig.savefig(os.path.join(out, "fig_physics.png"))
    plt.close(fig)


def all_contacts(path):
    """D19's definition: every trial with a measured lever arm, including grasps where one pad never touched
    (d* = 0, so the model predicts a failure)."""
    g = [r for r in rows(path) if r.get("lever_arm_cm") not in ("", None)]
    ds = np.array([float(r["dstar_lift_cm"] or 0) for r in g])
    ratio = np.array([float(r["lever_arm_cm"]) for r in g]) / np.maximum(ds, 1e-9)
    held = np.array([float(float(r["tilt_after_hold"] or 99) <= 10.0) for r in g])
    pts = np.array([min(int(r["n_pts_left"] or 0), int(r["n_pts_right"] or 0)) for r in g])
    return ratio, held, pts, ds


def failure(out):
    before = all_contacts("results/b0/trials_pad_v1.csv")
    after = all_contacts("results/b0/trials.csv")
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    b = np.arange(0, 17)
    for data, c, lab in ((before, "#c4553f", "step-1 pad (too stiff)"), (after, M2C, "rubber-layer pad")):
        axs[0].hist(np.minimum(data[2], 16), bins=b - 0.5, color=c, alpha=0.55, label=lab)
        axs[1].hist(np.minimum(data[3], 8), bins=np.linspace(0, 8, 33), color=c, alpha=0.55, label=lab)
    axs[0].set_xlabel("contact points on the weaker pad (of 16)")
    axs[0].set_ylabel("grasps")
    axs[0].set_title("Contact patch", loc="left")
    axs[1].set_xlabel("this grasp's slip distance d* [cm]")
    axs[1].set_title("Slip distance", loc="left")
    for ax, k in ((axs[0], 2), (axs[1], 3)):
        for data, c in ((before, "#c4553f"), (after, M2C)):
            ax.axvline(np.median(data[k]), color=c, lw=1.8, ls="--")
        ax.legend(frameon=False, fontsize=8.5)
    agr = [100 * np.mean((d[0] < 1) == (d[1] == 1)) for d in (before, after)]
    axs[2].bar([0, 1], agr, color=["#c4553f", M2C], alpha=0.8, width=0.6)
    for i, v in enumerate(agr):
        axs[2].text(i, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold", color=INK)
    axs[2].set_xticks([0, 1], ["step-1 pad", "rubber layer"])
    axs[2].set_ylim(0, 100)
    axs[2].set_ylabel("|d| < d* agrees with outcome [%]")
    axs[2].set_title("Physics agreement", loc="left")
    for ax in axs:
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    fig.text(0.01, -0.05, f"B0 at 1 kg, every trial with a measured lever arm ({len(before[0])} and {len(after[0])}). Medians: {np.median(before[2]):.0f} -> "
             f"{np.median(after[2]):.0f} contact points, d* {np.median(before[3]):.1f} -> {np.median(after[3]):.1f} cm.",
             fontsize=8.5, color=MUTED)
    fig.savefig(os.path.join(out, "fig_failure.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/midpoint")
    a = ap.parse_args()
    pol = json.load(open(os.path.join(a.out, "policies.json")))
    explorer = json.load(open(os.path.join(a.out, "explorer.json")))
    summary = json.load(open("results/b0_m045_v2/summary.json"))
    policies(pol, a.out)
    fresh = os.path.join(a.out, "policies_fresh.json")
    if os.path.exists(fresh):
        policies(json.load(open(fresh)), a.out, "fig_policies_fresh.png",
                 "0.45 kg, fresh seeds 6-11; M2 frozen after fitting on the development seeds.")
    shift_curve(explorer, a.out)
    physics(a.out, summary)
    failure(a.out)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()

"""Step 1: contact calibration. Grasp a box at its centre while a hidden weight shifts the CoM by d, lift, hold,
and measure how far it turns in the hand. Compares the twist onset with the prediction for each pad setup,
then across (mu, fn, a, m).

    python scripts/calibrate_contact.py
Outputs: results/calib/sweep.csv, general.csv, tables.tex, calib.png
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anygrasp_pp.contact_rig import X_HIDDEN, CONTACT, RIGID_CONTACT, DEFAULT_CONTACT, grid_points  # noqa: E402
from anygrasp_pp.physics import dstar, patch_dstar, FN, MU, MASS, A_PAD  # noqa: E402
from anygrasp_pp.rig_trial import run_trial, bisect_onset, a_peak, TWIST_DEG  # noqa: E402

SLIP_DEG = 1.0   # slip has started, tests the static prediction. TWIST_DEG (10) is what counts as failure

A_PEAK = a_peak()
DMAX = 0.95 * X_HIDDEN

# (label, short name, rig options)
CONFIGS = [
    ("Menagerie as shipped", "menagerie", dict(grip="stock", pads="stock", menagerie_friction=True, contact=DEFAULT_CONTACT)),
    ("1-point pad, MuJoCo default contacts", "point_default", dict(pads="point", contact=DEFAULT_CONTACT)),
    ("1-point pad (simulator check)", "point", dict(pads="point", contact=RIGID_CONTACT)),
    ("grid pad, rigid (hard plastic)", "grid_rigid", dict(pads="grid", contact=RIGID_CONTACT)),
    ("grid pad, rubber (study)", "grid", dict(pads="grid", contact=CONTACT)),
]


def predict(pads, mu=MU, fn=FN, a=A_PAD, m=MASS, accel=0.0):
    if pads == "point":
        return float(dstar(mu, fn, a, m, accel))
    return patch_dstar(grid_points(), mu, fn, m, accel)


def bracket(rows):
    """(last held d before the first failure, first failed d)."""
    fail = [r["d"] for r in rows if r["outcome"] != "held"]
    first = min(fail, default=np.nan)
    held = [r["d"] for r in rows if r["outcome"] == "held" and (np.isnan(first) or r["d"] < first)]
    return max(held, default=np.nan), first


def tex(x, nd=2):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def main():
    out = ROOT / "results" / "calib"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Lift peak acceleration {A_PEAK:.3f} m/s^2; twist threshold {TWIST_DEG:.0f} deg")
    print(f"Predicted onset, point pad: {predict('point')*100:.2f} cm static, {predict('point', accel=A_PEAK)*100:.2f} at peak")
    print(f"Predicted onset, grid pad:  {predict('grid')*100:.2f} cm static, {predict('grid', accel=A_PEAK)*100:.2f} at peak\n")

    # part 1: sweep the CoM shift for every setup
    shifts = np.round(np.arange(0.0, DMAX + 1e-9, 0.0025), 4)
    sweep, summary = [], []
    for label, short, rig in CONFIGS:
        rows = [{"config": short, "d": float(s), **run_trial(s, **rig)} for s in shifts]
        sweep += rows
        lo, hi = bracket(rows)
        upto = [r for r in rows if np.isnan(hi) or r["d"] < hi] or rows   # holding regime (before failure)
        s = dict(label=label, short=short, pts=f"{min(r['pts_min'] for r in upto)}-{max(r['pts_max'] for r in upto)}",
                 fn=f"{min(r['fn_min'] for r in upto):.1f}-{max(r['fn_max'] for r in upto):.1f}",
                 depth=max(r["depth_max_mm"] for r in upto), creep=max((r["deg_end"] for r in upto), default=np.nan),
                 caught=sum(r["hand_caught"] for r in rows), lo=lo, hi=hi, d0=rows[0]["outcome"])
        summary.append(s)
        print(f"{label:38s} pts/pad {s['pts']:>6s}  fn {s['fn']:>10s} N  depth {s['depth']:.3f} mm  d=0 {s['d0']:8s} "
              f"max rot. before failure {s['creep']:5.2f} deg  holds to {tex(lo*100)} cm, fails from {tex(hi*100)} cm"
              f"  (palm/finger catches: {s['caught']})")
    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader()
        w.writerows(sweep)

    # part 2: does the measured onset still follow the prediction when the parameters change?
    grids = {
        "point": [(0.5, 70, 0.006, 0.6), (0.8, 70, 0.006, 0.6), (0.3, 70, 0.006, 0.6), (0.5, 50, 0.006, 0.6),
                  (0.5, 30, 0.006, 0.6), (0.5, 70, 0.004, 0.6), (0.5, 70, 0.008, 0.6), (0.5, 70, 0.006, 0.8)],
        "grid": [(0.5, 70, None, 0.6), (0.3, 70, None, 0.6), (0.4, 70, None, 0.6), (0.5, 40, None, 0.6),
                 (0.5, 55, None, 0.6), (0.5, 70, None, 0.8), (0.6, 70, None, 0.9)],
    }
    grids["grid_rigid"] = grids["grid"]
    pad_contact = {"point": ("point", RIGID_CONTACT), "grid": ("grid", CONTACT), "grid_rigid": ("grid", RIGID_CONTACT)}
    print(f"\nSlip onset ({SLIP_DEG:.0f} deg) vs predicted band [onset at peak lift acceleration, static onset]")
    general = []
    for label, combos in grids.items():
        pads, ct = pad_contact[label]
        for mu, fn, a, m in combos:
            aa = A_PAD if a is None else a
            p0, p1 = predict(pads, mu, fn, aa, m), predict(pads, mu, fn, aa, m, A_PEAK)
            meas, flag = bisect_onset(DMAX, pads=pads, mu=mu, fn=fn, a=aa, mass=m, contact=ct, twist_deg=SLIP_DEG)
            fail, flag10 = bisect_onset(DMAX, pads=pads, mu=mu, fn=fn, a=aa, mass=m, contact=ct, twist_deg=TWIST_DEG)
            tol = DMAX / 8 / 2**8                                # bisection resolution
            ok = flag == "ok" and p1 - tol <= meas <= p0 + tol
            ok10 = flag10 == "ok" and p1 - tol <= fail <= p0 + tol
            general.append(dict(pads=label, mu=mu, fn=fn, a=a if a else "", m=m, pred_peak=p1, pred_static=p0,
                                measured=meas, flag=flag, ratio=meas / p0, in_band=ok, fail10=fail, flag10=flag10,
                                ratio10=fail / p0, in_band10=ok10))
            print(f"  {label:10s} mu={mu:.1f} fn={fn:3.0f} a={'--' if a is None else f'{a*1000:.0f} mm':>5s} m={m:.1f}  "
                  f"band {p1*100:5.2f}-{p0*100:5.2f} cm  1-deg {meas*100:5.2f} ({(meas-p0)/p0*100:+5.1f}%) "
                  f"{'in' if ok else 'out'}   10-deg {fail*100:5.2f} ({(fail-p0)/p0*100:+5.1f}%) {'in' if ok10 else 'out'}", flush=True)
    with open(out / "general.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(general[0].keys()))
        w.writeheader()
        w.writerows(general)

    # LaTeX tables
    with open(out / "tables.tex", "w") as f:
        f.write("% generated by scripts/calibrate_contact.py -- do not edit\n")
        f.write("\\newcommand{\\calibsetups}{%\n")
        for s in summary:
            f.write(f"{s['label']} & {s['pts']} & {s['fn']} & {s['depth']:.3f} & {s['d0']} & {tex(s['creep'],1)} & "
                    f"{tex(s['lo']*100)} & {tex(s['hi']*100)} & {s['caught']} \\\\\n")
        f.write("}\n\\newcommand{\\calibgeneral}{%\n")
        for g in general:
            a = "--" if g["a"] == "" else f"{g['a']*1000:.0f}"
            pad = {"point": "point (rigid)", "grid": "grid, rubber", "grid_rigid": "grid, rigid"}[g["pads"]]
            v = lambda x, ok: f"{tex(x*100)} ({(x-g['pred_static'])/g['pred_static']*100:+.1f}\\%){'' if ok else '$^\\dagger$'}"
            f.write(f"{pad} & {g['mu']:.1f} & {g['fn']:.0f} & {a} & {g['m']:.1f} & {g['pred_peak']*100:.2f}--"
                    f"{g['pred_static']*100:.2f} & {v(g['measured'], g['in_band'])} & {v(g['fail10'], g['in_band10'])} \\\\\n")
        f.write("}\n")
        f.write(f"\\newcommand{{\\calibapeak}}{{{A_PEAK:.3f}}}\n")
        f.write(f"\\newcommand{{\\calibpointpred}}{{{predict('point')*100:.2f}}}\n")
        f.write(f"\\newcommand{{\\calibgridpred}}{{{predict('grid')*100:.2f}}}\n")
        f.write(f"\\newcommand{{\\calibgridpredpeak}}{{{predict('grid', accel=A_PEAK)*100:.2f}}}\n")
        for lab in ("point", "grid", "grid_rigid"):
            gg = [g for g in general if g["pads"] == lab]
            nm = {"point": "Point", "grid": "Rubber", "grid_rigid": "Rigid"}[lab]
            f.write(f"\\newcommand{{\\calibinbandone{nm}}}{{{sum(g['in_band'] for g in gg)}/{len(gg)}}}\n")
            f.write(f"\\newcommand{{\\calibinbandten{nm}}}{{{sum(g['in_band10'] for g in gg)}/{len(gg)}}}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [1.6, 1]})
    for (label, short, _), st in zip(CONFIGS, [":", (0, (1, 3)), "--", "-.", "-"]):
        rr = [r for r in sweep if r["config"] == short]
        ax1.plot([r["d"] * 100 for r in rr], [95 if r["outcome"] == "dropped" else r["deg_end"] for r in rr],
                 ls=st, marker="o", ms=2.5, lw=1.4, label=label)
        caught = [r for r in rr if r["hand_caught"]]
        if caught:
            ax1.plot([r["d"] * 100 for r in caught], [r["deg_end"] for r in caught], "kx", ms=5)
    for pads, col in (("point", "C2"), ("grid", "C4")):
        ax1.axvspan(predict(pads, accel=A_PEAK) * 100, predict(pads) * 100, color=col, alpha=0.25)
    ax1.axhline(TWIST_DEG, color="0.5", lw=0.8, ls="--")
    ax1.text(0.2, TWIST_DEG + 1.5, f"{TWIST_DEG:.0f}$^\\circ$ = twisted out", fontsize=7, color="0.4")
    ax1.set(xlabel="hidden CoM shift d [cm] (grasp at geometric centre)",
            ylabel="rotation in hand after hold [deg]\n(95 = dropped; x = resting on palm/finger)", ylim=(-3, 100))
    ax1.legend(fontsize=7, loc="upper left")
    for lab, mk, txt in (("point", "o", "point pad, 1-deg onset"), ("grid_rigid", "s", "grid rigid, 1-deg onset"),
                         ("grid", "D", "grid rubber, 10-deg failure")):
        gg = [g for g in general if g["pads"] == lab and g["flag"] == "ok"]
        x = [g["pred_static"] * 100 for g in gg]
        ax2.plot(x, [g["ratio10" if lab == "grid" else "ratio"] for g in gg], mk, ms=5, label=txt)
        ax2.plot(x, [g["pred_peak"] / g["pred_static"] for g in gg], "_", color="k", ms=10)
    ax2.axhline(1.0, color="k", lw=0.8)
    ax2.set(xlabel="static prediction [cm]", ylabel="measured onset / static prediction", ylim=(0.9, 1.03))
    ax2.text(0.02, 0.03, "bars: lower edge of predicted band (peak lift acceleration)", transform=ax2.transAxes, fontsize=7)
    ax2.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "calib.png", dpi=150)
    print(f"\nsaved {out}/sweep.csv, general.csv, tables.tex, calib.png")


if __name__ == "__main__":
    main()

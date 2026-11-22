"""Midpoint numbers: each re-ranking policy on the executed candidates, 95% CI from an object bootstrap, plus the
B0 + blind retry control. Retry rows are upper bounds (oracle trigger, 2nd grasp scored on the undisturbed scene).
Also writes explorer.json for the project page and video_picks.json for the rollout videos.

    python scripts/midpoint_results.py --run results/b0_m045_v2 --out results/midpoint
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_learn import design, fit_logistic  # noqa: E402

N_BOOT = 4000


def loo_scores(F, X, y):
    """Calibrated M2 probability per candidate, from a model that never saw that object."""
    objs = np.array([f["name"] for f in F])
    p = np.zeros(len(F))
    for o in sorted(set(objs)):
        model, _ = fit_logistic(X[objs != o], y[objs != o])
        p[objs == o] = model(X[objs == o])
    return p


def score(F, p_cal, run, label):
    """Every policy on one set of executed candidates, with object-bootstrap intervals."""
    by = defaultdict(list)
    for f, p in zip(F, p_cal):
        by[f["tag"]].append(dict(f, p_cal=float(p), cand=int(f["cand"]), ok=f["success"] == "True"))
    scenes = sorted({r["tag"] for r in csv.DictReader(open(os.path.join(run, "candidates.csv")))})
    info = {t: dict(name=t.split("__s")[0], shift=int(t.split("__s")[1].split("__")[0]),
                    seed=int(t.split("__k")[1])) for t in scenes}

    def first(c, key):
        return max(c, key=key) if c else None

    def ok(f):
        return bool(f and f["ok"])

    def two(c, key):                     # first choice, then one blind retry with the next in the same order
        o = sorted(c, key=key, reverse=True)
        return ok(o[0] if o else None) or ok(o[1] if len(o) > 1 else None)

    det = lambda f: -f["cand"]                                   # detector order
    cal = lambda f: (f["p_cal"], -f["cand"])
    policies = {
        "B0 (detector's first reachable grasp)": lambda c: ok(first(c, det)),
        "B0 + blind retry (upper bound)": lambda c: two(c, det),
        "Closest to the visible centre": lambda c: ok(first(c, lambda f: (-float(f["d_vis"]), -f["cand"]))),
        "M2 v1 (physics only)": lambda c: ok(first(c, lambda f: (float(f["score"]) * float(f["p_hold"]), -f["cand"]))),
        f"M2 calibrated ({label})": lambda c: ok(first(c, cal)),
        "M2 calibrated + blind retry (upper bound)": lambda c: two(c, cal),
        "Best of the executed grasps (ceiling)": lambda c: any(f["ok"] for f in c),
    }
    S = {k: np.array([pol(by.get(t, [])) for t in scenes], float) for k, pol in policies.items()}
    objs = sorted({v["name"] for v in info.values()})
    obj_idx = {o: np.array([i for i, t in enumerate(scenes) if info[t]["name"] == o]) for o in objs}
    shift = np.array([info[t]["shift"] for t in scenes])
    groups = {"all shifts": shift >= 0, "no shift": shift == 0, "6-8 cm shift": shift >= 6}
    rng = np.random.default_rng(0)
    boots = [rng.choice(objs, len(objs)) for _ in range(N_BOOT)]
    base = "B0 (detector's first reachable grasp)"
    out = dict(run=run, model=label, n_scenes=len(scenes), n_candidates=len(F), objects=objs, groups={})
    print(f"{len(scenes)} scenes, {len(F)} executed candidates, {len(objs)} objects; 95% CI: object-clustered bootstrap")
    for g, mask in groups.items():
        out["groups"][g] = {}
        print(f"\n{g} ({int(mask.sum())} scenes)")
        for k, s in S.items():
            rate = s[mask].mean()
            bs, bd = [], []
            for b in boots:
                idx = np.concatenate([obj_idx[o] for o in b])
                idx = idx[mask[idx]]
                bs.append(s[idx].mean())
                bd.append(s[idx].mean() - S[base][idx].mean())
            lo, hi = np.percentile(bs, [2.5, 97.5])
            dlo, dhi = np.percentile(bd, [2.5, 97.5])
            diff = rate - S[base][mask].mean()
            out["groups"][g][k] = dict(success=int(s[mask].sum()), n=int(mask.sum()), rate=float(rate),
                                       ci=[float(lo), float(hi)], diff_vs_b0=float(diff), diff_ci=[float(dlo), float(dhi)])
            print(f"  {k:44s} {int(s[mask].sum()):4d}/{int(mask.sum())} = {100 * rate:5.1f}%  "
                  f"[{100 * lo:4.1f}, {100 * hi:4.1f}]   vs B0 {100 * diff:+5.1f} [{100 * dlo:+5.1f}, {100 * dhi:+5.1f}]")
    return out, by, scenes, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/b0_m045_v2")
    ap.add_argument("--out", default="results/midpoint")
    ap.add_argument("--test", default=None, help="fresh-seed candidates run: score the model frozen on --run")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    F = list(csv.DictReader(open(os.path.join(a.run, "candidates_features.csv"))))
    X, y = design(F), np.array([f["success"] == "True" for f in F], float)
    p_cv = loo_scores(F, X, y)
    out, by, scenes, info = score(F, p_cv, a.run, "held-out objects")
    json.dump(out, open(os.path.join(a.out, "policies.json"), "w"), indent=1)
    if a.test:   # held-out scenes: the model is fitted once on every development candidate, then frozen
        model, _ = fit_logistic(X, y)
        T = list(csv.DictReader(open(os.path.join(a.test, "candidates_features.csv"))))
        out_t, _, _, _ = score(T, model(design(T)), a.test, "frozen, fresh scenes")
        json.dump(out_t, open(os.path.join(a.out, "policies_fresh.json"), "w"), indent=1)
        return

    # per-scene data for the page's trial explorer
    cal = lambda f: (f["p_cal"], -f["cand"])
    keep = ("cand", "ok", "outcome", "lever_arm_cm", "d_vis", "dstar_lift_cm", "score", "p_cal", "approach_deg",
            "tilt_max", "depth", "width")
    explorer = []
    for t in scenes:
        c = sorted(by.get(t, []), key=lambda f: f["cand"])
        rows = [{k: (round(float(f[k]), 4) if k not in ("ok", "outcome", "cand") and f[k] not in ("", None) else f[k])
                 for k in keep} for f in c]
        explorer.append(dict(tag=t, **info[t], cands=rows,
                             b0=c[0]["cand"] if c else None,
                             m2=max(c, key=cal)["cand"] if c else None))
    json.dump(explorer, open(os.path.join(a.out, "explorer.json"), "w"), separators=(",", ":"))

    # rollout videos, one scene per object: B0 turned in hand at 6-8 cm and M2 held, else B0 failed and M2 held,
    # else both held at the largest shift. plus two where both failed so the page doesn't only show wins
    L, R = "B0: the detector's top grasp", "M2 (prototype): re-ranked grasp"
    picks, fails = {}, []
    def rank(e, b0, m2):
        if b0["outcome"] == "turned in hand" and m2["ok"] and e["shift"] >= 6:
            return (3, e["shift"])
        if not b0["ok"] and m2["ok"]:
            return (2, e["shift"])
        if b0["ok"] and m2["ok"]:
            return (1, e["shift"])
        return (0, 0)
    for e in explorer:
        if not e["cands"] or e["m2"] == e["b0"]:
            continue
        b0 = e["cands"][0]
        m2 = next(f for f in e["cands"] if f["cand"] == e["m2"])
        k = rank(e, b0, m2)
        if k[0] and k > picks.get(e["name"], ((0, 0), None))[0]:
            picks[e["name"]] = (k, e["tag"], b0["cand"], m2["cand"])
        if b0["outcome"] == "turned in hand" and not m2["ok"] and e["shift"] == 8:
            fails.append((e["name"], e["tag"], b0["cand"], m2["cand"]))
    kinds = {3: "b0 turned, m2 held", 2: "b0 failed, m2 held", 1: "both held"}
    out_p = [dict(tag=t, left=l, right=r, left_title=L, right_title=R, kind=kinds[k[0]])
             for k, t, l, r in sorted(picks.values(), key=lambda v: -v[0][0])]
    seen = set()
    for name, t, l, r in fails:
        if name not in seen and len(seen) < 2:
            seen.add(name)
            out_p.append(dict(tag=t, left=l, right=r, left_title=L, right_title=R, kind="both failed"))
    json.dump(out_p, open(os.path.join(a.out, "video_picks.json"), "w"), indent=1)
    print(f"\nvideo picks ({len(out_p)}):")
    for p in out_p:
        print(f"  {p['kind']:20s} {p['tag']}  B0 cand {p['left']} vs M2 cand {p['right']}")

if __name__ == "__main__":
    main()

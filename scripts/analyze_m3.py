"""M3 results: AnyGrasp++ vs B0 and B0 + blind retry on the same scenes, object-bootstrap CIs.
Also checks the test lift: measured CoM vs true, and whether the grasps it gave up on would really have failed
(every first grasp was also run without a test lift in the candidates run).

    python scripts/analyze_m3.py --m3 results/m3_dev --cands results/b0_m045_v2 --out results/midpoint/m3_dev.json
"""
import argparse
import csv
import json
import os

import numpy as np


def load(path):
    return {r["tag"]: r for r in (json.loads(line) for line in open(path))} if os.path.exists(path) else {}


def boot(ok, objs, n=4000, seed=0):
    ok, objs = np.asarray(ok, float), np.asarray(objs)
    names = sorted(set(objs))
    idx = {o: np.flatnonzero(objs == o) for o in names}
    rng = np.random.default_rng(seed)
    picks = [np.concatenate([idx[o] for o in rng.choice(names, len(names))]) for _ in range(n)]
    return [float(np.percentile([ok[p].mean() for p in picks], q)) for q in (2.5, 97.5)], picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3", default="results/m3_dev")
    ap.add_argument("--cands", default="results/b0_m045_v2")
    ap.add_argument("--out", default="results/midpoint/m3_dev.json")
    a = ap.parse_args()
    A, Rr = load(os.path.join(a.m3, "agpp_parts.jsonl")), load(os.path.join(a.m3, "retry_parts.jsonl"))
    C = {(r["tag"], r["cand"]): r for r in csv.DictReader(open(os.path.join(a.cands, "candidates.csv")))}
    tags = sorted(set(A) & set(Rr))
    objs = [A[t]["name"] for t in tags]
    out = dict(scenes=len(tags), policies={})
    b0 = [C.get((t, "0"), {}).get("success") == "True" for t in tags]
    for name, ok in (("B0", b0), ("B0 + blind retry", [bool(Rr[t].get("success")) for t in tags]),
                     ("AnyGrasp++", [bool(A[t].get("success")) for t in tags])):
        ci, picks = boot(ok, objs)
        out["policies"][name] = dict(success=int(sum(ok)), n=len(tags), rate=float(np.mean(ok)), ci=ci)
    ok_a = np.array([bool(A[t].get("success")) for t in tags], float)
    ok_r = np.array([bool(Rr[t].get("success")) for t in tags], float)
    _, picks = boot(ok_a, objs)
    diffs = [ok_a[p].mean() - ok_r[p].mean() for p in picks]
    out["agpp_minus_retry"] = dict(diff=float(ok_a.mean() - ok_r.mean()), ci=[float(np.percentile(diffs, 2.5)),
                                                                              float(np.percentile(diffs, 97.5))],
                                   only_agpp=int(((ok_a == 1) & (ok_r == 0)).sum()),
                                   only_retry=int(((ok_a == 0) & (ok_r == 1)).sum()))
    shifts = sorted({int(A[t]["shift"] * 100 + 0.5) for t in tags})
    out["by_shift"] = {s: {k: float(np.mean([v for t, v in zip(tags, ok) if int(A[t]["shift"] * 100 + 0.5) == s]))
                           for k, ok in (("B0", b0), ("retry", ok_r), ("agpp", ok_a))} for s in shifts}
    for name, P in (("AnyGrasp++", A), ("B0 + blind retry", Rr)):
        first = [P[t]["tries"][0] for t in tags if P[t].get("tries")]
        errs = [f["com_err_cm"] for f in first if f.get("lifted") and f.get("com_err_cm") is not None]
        regr = [P[t] for t in tags if len(P[t].get("tries", [])) > 1]
        kept = [P[t] for t in tags if len(P[t].get("tries", [])) == 1]
        would = lambda r: C.get((r["tag"], str(r["tries"][0]["cand"])), {}).get("success") == "True"
        out[name] = dict(regrasped=len(regr), kept=len(kept),
                         gave_up_on_a_grasp_that_would_have_held=int(sum(would(r) for r in regr)),
                         kept_a_grasp_that_would_have_failed=int(sum(not would(r) for r in kept)),
                         regrasp_success=float(np.mean([bool(r.get("success")) for r in regr])) if regr else None,
                         com_error_cm_median=float(np.median(errs)) if errs else None,
                         com_error_cm_p90=float(np.percentile(errs, 90)) if errs else None, com_readings=len(errs))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

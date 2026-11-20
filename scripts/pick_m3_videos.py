"""Scenes for the B0 vs AnyGrasp++ videos: per object, one where B0's grasp failed and AnyGrasp++ succeeded after
noticing a slip in its test lift and regrasping (else one where it succeeded with its first grasp), preferring larger
shifts; plus one scene where AnyGrasp++ failed as well.

    python scripts/pick_m3_videos.py --m3 results/m3_dev --cands results/b0_m045_v2 --out results/midpoint/m3_video_picks.json
"""
import argparse
import csv
import json

ap = argparse.ArgumentParser()
ap.add_argument("--m3", default="results/m3_dev")
ap.add_argument("--cands", default="results/b0_m045_v2")
ap.add_argument("--out", default="results/midpoint/m3_video_picks.json")
a = ap.parse_args()
A = {r["tag"]: r for r in (json.loads(x) for x in open(a.m3 + "/agpp_parts.jsonl"))}
b0 = {r["tag"]: r for r in csv.DictReader(open(a.cands + "/candidates.csv")) if r["cand"] == "0"}
best, fail = {}, None
for t, r in A.items():
    if t not in b0 or b0[t]["success"] == "True" or b0[t]["grasp_index"] in ("", "-1"):
        continue
    regr = len(r.get("tries", [])) > 1
    if r.get("success"):
        key = (2 if regr else 1, r["shift"])
        if key > best.get(r["name"], ((0, 0), None))[0]:
            best[r["name"]] = (key, t)
    elif fail is None and r["shift"] >= 0.06 and regr:
        fail = t
picks = [dict(tag=t, left="b0", right="agpp") for _, t in sorted(best.values(), key=lambda v: (-v[0][0], -v[0][1]))]
if fail:
    picks.append(dict(tag=fail, left="b0", right="agpp"))
json.dump(picks, open(a.out, "w"), indent=1)
print(len(picks), "pairs:", [p["tag"] for p in picks])

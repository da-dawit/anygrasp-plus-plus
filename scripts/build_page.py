"""Builds page/index.html from page/template.html with the results embedded, so it opens from disk, no server.

    python scripts/build_page.py
"""
import csv
import json
import os

R = "results/midpoint"


def load(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default


def m3(path):
    if not os.path.exists(path):
        return None
    rows = [json.loads(line) for line in open(path)]
    return sum(bool(r.get("success")) for r in rows), len(rows), sum(len(r.get("tries", [])) > 1 for r in rows)


def summary():
    dev, fresh = load(os.path.join(R, "policies.json")), load(os.path.join(R, "policies_fresh.json"))
    g, f = dev["groups"]["all shifts"], fresh["groups"]["all shifts"]
    k, kf = list(g), list(f)
    pct = lambda s, n: f"{100 * s / n:.0f}%"
    out = [f"On the 270 development scenes (9 objects at 0.45&nbsp;kg, 5 shifts, 6 seeds) B0 succeeds in "
           f"{pct(g[k[0]]['success'], g[k[0]]['n'])}. M2, which only changes which grasp is tried first, gets "
           f"{pct(g[k[4]]['success'], g[k[4]]['n'])} when each object is left out of its training, and on 270 new scenes "
           f"it gets {pct(f[kf[4]]['success'], f[kf[4]]['n'])} against B0's {pct(f[kf[0]]['success'], f[kf[0]]['n'])}."]
    a, r = m3("results/m3_dev/agpp_parts.jsonl"), m3("results/m3_dev/retry_parts.jsonl")
    if a and r:
        out.append(f"With the test lift and one regrasp, AnyGrasp++ gets {pct(a[0], a[1])} on the development scenes. "
                   f"The control asked for in the proposal feedback, B0 with the same test lift and one blind retry "
                   f"(the detector's next grasp, with the simulator telling it when to retry), gets {pct(r[0], r[1])}.")
    af, rf = m3("results/m3_fresh/agpp_parts.jsonl"), m3("results/m3_fresh/retry_parts.jsonl")
    if af and rf:
        out.append(f"On the new scenes the two get {pct(af[0], af[1])} and {pct(rf[0], rf[1])}.")
    out.append("If the arm always took the best of the first ten grasps it can reach, it would succeed in about "
               f"{pct(g[k[6]]['success'], g[k[6]]['n'])} of scenes, so there is still a lot left to gain.")
    return out


def main():
    explorer = load(os.path.join(R, "explorer.json"))
    keep = ("cand", "ok", "outcome", "lever_arm_cm", "dstar_lift_cm", "score", "p_cal")
    for e in explorer:
        e["cands"] = [{k: f[k] for k in keep} for f in e["cands"]]
    vdir = "videos_m3" if os.path.exists(os.path.join(R, "videos_m3", "manifest.json")) else "videos"
    data = dict(videos=load(os.path.join(R, vdir, "manifest.json"), []), vdir=f"../results/midpoint/{vdir}/",
                explorer=explorer, summary=summary(), repro=load(os.path.join(R, "reproduction.json")),
                gallery=[dict(name=r["name"], source=r["source"])
                         for r in csv.DictReader(open("results/objects/final_set.csv"))])
    html = open("page/template.html").read()
    blob = json.dumps(data, separators=(",", ":"), default=float).replace("</", "<\\/")
    open("page/index.html", "w").write(html.replace("/*DATA*/", blob))
    print(f"page/index.html: {len(html) + len(blob)} bytes, {len(data['videos'])} videos from {vdir}, "
          f"{len(explorer)} scenes")


if __name__ == "__main__":
    main()

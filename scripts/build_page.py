"""Builds page/index.html from page/template.html by embedding the results it shows (the page then opens straight
from the file system, no server needed).

    python scripts/build_page.py
"""
import csv
import json
import os

R = "results/midpoint"


def load(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default


def main():
    explorer = load(os.path.join(R, "explorer.json"))
    keep = ("cand", "ok", "outcome", "lever_arm_cm", "dstar_lift_cm", "score", "p_cal")
    for e in explorer:
        e["cands"] = [{k: f[k] for k in keep} for f in e["cands"]]
    data = dict(videos=load(os.path.join(R, "videos", "manifest.json"), []),
                explorer=explorer,
                pol=load(os.path.join(R, "policies.json")),
                fresh=load(os.path.join(R, "policies_fresh.json")),
                repro=load(os.path.join(R, "reproduction.json")),
                gallery=[dict(name=r["name"], source=r["source"])
                         for r in csv.DictReader(open("results/objects/final_set.csv"))])
    html = open("page/template.html").read()
    blob = json.dumps(data, separators=(",", ":"), default=float).replace("</", "<\\/")
    open("page/index.html", "w").write(html.replace("/*DATA*/", blob))
    print(f"page/index.html: {len(html) + len(blob)} bytes, {len(data['videos'])} rollouts, "
          f"{len(explorer)} scenes, fresh-scene check {'included' if data['fresh'] else 'not yet run'}")


if __name__ == "__main__":
    main()

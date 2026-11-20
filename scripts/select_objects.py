"""Object selection for the hidden-weight study, from all 1030 Google Scanned Objects (MuJoCo models by
K. Zakka). Decisions D14-D16; the hidden core is derivations section H.

Stage 1 (visual meshes, all objects)  -> all_objects.csv, candidates.txt
    principal axes from area-weighted surface samples give extents L >= M >= S; 1 cm slices along the long axis
    give the width across (M) and the height (S), assuming the object lies with S vertical.
    candidate: length 18-45 cm and >= 4 cm of slices with across <= 8/1.2 cm and height >= 1.5 cm.
Stage 2 (collision solids, candidates) -> solid_survey.csv, stage2_shortlist.txt
    per object: separate pieces, hidden-core feasibility (anygrasp_pp.objects.build), graspable length on the
    solid, collision-vs-visual fidelity. Shortlist: one piece, core feasible, >= 5 cm graspable, not footwear
    (name pattern below; shoes are deformable) and not in exclusions.csv (reason given per object).
The final test is scripts/graspability_test.py: resting pose stable at every shift, overhead render, detector.

Criteria come from earlier decisions:
    width rule (D14): reported width = 1.2 x thickness <= 8 cm, so across <= 6.7 cm
    fingers close above the table: height >= 1.5 cm (17 mm pads)
    an 8 cm shift at 1 kg needs ~9-10 cm from the centroid to the core (D11, D15, H.2): length >= 18 cm;
    45 cm keeps the object inside the camera's view and the Panda's reach

Usage:  python scripts/select_objects.py --stage 1 --models <mujoco_scanned_objects>/models
        python scripts/select_objects.py --stage 2 --models <mujoco_scanned_objects>/models
"""
import argparse
import csv
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

MAX_ACROSS = 0.08 / 1.2
MIN_HEIGHT = 0.015
LEN_RANGE = (0.18, 0.45)
MIN_GRASP_LEN_1 = 0.04        # stage 1 (surface, loose)
MIN_GRASP_LEN = 0.05          # stage 2 (solid)
SLICE = 0.01
FOOTWEAR = re.compile(
    r"shoe|boot|flat|sneaker|trainer|reebok|asics|tieks|tory_burch|ugg|timberland|teva|sperry|chukka|bahama|"
    r"angelfish|bluefish|hikerfish|_moc_|court|zig|dmx|d_rose|crazy|samoa|fyw_|adizero|enfr|top_ten|trx|"
    r"real_deal|miracle|trochilus|amberlight|climacool|pheehan|tzx|colton|martin_wedge|rayna|kamikaze|pump_omni|"
    r"punch_drop|reef_|jogger|retro_rush|allylynn|cushrun|breakpoint|easytone|legacy|fueltrain|smoothflex|dibello|"
    r"ultimatic|cooperstown|gel|hyper|deckhand|girls_|_w$|lace_boot|bootie|slip_on|ballet|loafer|sandal|mens_|"
    r"womens_|hilary|dragon_w|wings_20|11pro|chelsea", re.I)


def surface_frame(mesh, n=20000, seed=0):
    pts = trimesh.sample.sample_surface(mesh, n, seed=seed)[0]
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return (pts - c) @ vt.T                      # columns: long, middle, short axis


def slices(local):
    """Per 1 cm slice along the long axis: (x0, x1, across, height) in the principal frame."""
    x = local[:, 0]
    edges = np.arange(x.min(), x.max() + SLICE, SLICE)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = local[(x >= a) & (x < b)]
        rows.append((a, b, np.ptp(s[:, 1]), np.ptp(s[:, 2])) if len(s) >= 5 else (a, b, 0.0, 0.0))
    return np.array(rows)


def graspable_length(local):
    sl = slices(local)
    return ((sl[:, 2] <= MAX_ACROSS) & (sl[:, 3] >= MIN_HEIGHT) & (sl[:, 2] > 0)).sum() * SLICE


def stage1_one(path):
    name = os.path.basename(os.path.dirname(path))
    try:
        local = surface_frame(trimesh.load(path, force="mesh", process=True))
        L, M, S = np.ptp(local, axis=0)
        return dict(name=name, L=L, M=M, S=S, grasp_len=graspable_length(local), error="")
    except Exception as e:  # one bad mesh must not stop the survey
        return dict(name=name, error=repr(e)[:120])


def stage2_one(model_dir):
    from anygrasp_pp import objects as O
    name = os.path.basename(model_dir)
    try:
        ob = O.build(model_dir)
        local = (ob["points"] - ob["centroid"]) @ ob["axes"].T
        mesh = trimesh.load(os.path.join(model_dir, "model.obj"), force="mesh", process=True)
        r = dict(name=name, L=ob["length"], grasp_len=graspable_length(local), volume_cm3=ob["volume"] * 1e6,
                 parts=ob["parts"], feasible=ob["core"] is not None, error="",
                 **O.fidelity(ob["occ"], ob["origin"], None, mesh=mesh))
        if ob["core"] is not None:
            c, p8 = ob["core"], ob["props"][max(O.SHIFTS)]
            r.update(core_len=float(np.ptp(local[c["F"], 0])), core_density_8=p8["core_density"],
                     shell_mass_8=O.MASS - p8["core_mass"],
                     shift_err=max(abs(v - s) for s, v in ob["shift_check"].items()))
        return r
    except Exception as e:
        return dict(name=name, error=repr(e)[:160])


def write_csv(path, rows, keys):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.5g}" if isinstance(r.get(k), (float, np.floating)) else r.get(k, ""))
                        for k in keys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", default="results/objects")
    ap.add_argument("--jobs", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.stage == 1:
        paths = sorted(os.path.join(a.models, n, "model.obj") for n in os.listdir(a.models)
                       if os.path.exists(os.path.join(a.models, n, "model.obj")))
        with Pool(a.jobs) as pool:
            rows = pool.map(stage1_one, paths, chunksize=4)
        write_csv(os.path.join(a.out, "all_objects.csv"), rows, ["name", "L", "M", "S", "grasp_len", "error"])
        cand = [r["name"] for r in rows if not r["error"] and LEN_RANGE[0] <= r["L"] <= LEN_RANGE[1]
                and r["grasp_len"] >= MIN_GRASP_LEN_1]
        open(os.path.join(a.out, "candidates.txt"), "w").write("\n".join(cand) + "\n")
        print(f"stage 1: {len(rows)} objects, {sum(1 for r in rows if r['error'])} failed to load, "
              f"{len(cand)} candidates")
        return
    names = [n.strip() for n in open(os.path.join(a.out, "candidates.txt")) if n.strip()]
    with Pool(a.jobs) as pool:
        rows = pool.map(stage2_one, [os.path.join(a.models, n) for n in names], chunksize=1)
    write_csv(os.path.join(a.out, "solid_survey.csv"), rows,
              ["name", "L", "grasp_len", "volume_cm3", "parts", "feasible", "core_len", "core_density_8",
               "shell_mass_8", "shift_err", "bulge95", "bulge_max", "gap95", "gap_max", "error"])
    excl = {r["name"]: r["reason"] for r in csv.DictReader(open(os.path.join(a.out, "exclusions.csv")))}
    ok = [r for r in rows if not r["error"]]
    steps = [("core feasible", lambda r: r["feasible"]),
             (f">= {MIN_GRASP_LEN * 100:.0f} cm graspable", lambda r: r["grasp_len"] >= MIN_GRASP_LEN),
             ("one piece", lambda r: r["parts"] == 1),
             ("not footwear", lambda r: not FOOTWEAR.search(r["name"])),
             ("not excluded (exclusions.csv)", lambda r: r["name"] not in excl)]
    print(f"stage 2: {len(rows)} candidates, {len(rows) - len(ok)} errors")
    for label, keep in steps:
        ok = [r for r in ok if keep(r)]
        print(f"  {label:32s} {len(ok):4d}")
    open(os.path.join(a.out, "stage2_shortlist.txt"), "w").write("\n".join(sorted(r["name"] for r in ok)) + "\n")


if __name__ == "__main__":
    main()

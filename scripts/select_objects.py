"""Object selection for the hidden-weight study from the 1030 Google Scanned Objects (MuJoCo models by K. Zakka).
stage 1 (visual meshes) -> all_objects.csv, candidates.txt; stage 2 (collision solids, hidden core, one piece,
not footwear, not in exclusions.csv) -> solid_survey.csv, stage2_shortlist.txt. Last filter is graspability_test.py.

    python scripts/select_objects.py --stage 1 --models <mujoco_scanned_objects>/models
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

MAX_ACROSS = 0.08 / 1.2       # reported width ~1.2 x thickness, has to be <= 8 cm
MIN_HEIGHT = 0.015            # fingers close above the table (17 mm pads)
LEN_RANGE = (0.18, 0.45)      # 8 cm shift at 1 kg needs ~9-10 cm centroid to core; 45 cm stays in view and reach
MIN_GRASP_LEN_1 = 0.04        # stage 1 (surface, loose)
MIN_GRASP_LEN = 0.05          # stage 2 (solid)
SLICE = 0.01
FOOTWEAR = re.compile(        # shoes are deformable
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
    """(x0, x1, across, height) per 1 cm slice along the long axis; assumes the short axis ends up vertical."""
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


def stage2_one(args):
    from anygrasp_pp import objects as O
    model_dir, collision = args
    name = os.path.basename(model_dir)
    try:
        ob = O.build(model_dir, collision=collision)
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
    ap.add_argument("--collision", default="vhacd", help="convex pieces to use: vhacd (as scanned) or coacd")
    ap.add_argument("--candidates", default="candidates.txt", help="stage 2 input list (in --out)")
    ap.add_argument("--suffix", default="", help="stage 2 output name suffix, e.g. _coacd")
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
    names = [n.strip() for n in open(os.path.join(a.out, a.candidates)) if n.strip()]
    with Pool(a.jobs) as pool:
        rows = pool.map(stage2_one, [(os.path.join(a.models, n), a.collision) for n in names], chunksize=1)
    write_csv(os.path.join(a.out, f"solid_survey{a.suffix}.csv"), rows,
              ["name", "L", "grasp_len", "volume_cm3", "parts", "feasible", "core_len", "core_density_8",
               "shell_mass_8", "shift_err", "bulge95", "bulge_max", "gap95", "gap_max", "error"])
    ex_path = os.path.join(a.out, "exclusions.csv")
    excl = {r["name"]: r["reason"] for r in csv.DictReader(open(ex_path))} if os.path.exists(ex_path) else {}
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
    open(os.path.join(a.out, f"stage2_shortlist{a.suffix}.txt"), "w").write("\n".join(sorted(r["name"] for r in ok)) + "\n")


if __name__ == "__main__":
    main()

"""Collision pieces by CoACD (Wei et al., SIGGRAPH 2022), the same settings for every object, written to
<model_dir>/coacd/model_collision_<k>.obj with the settings in coacd/params.json.

Why: the scanned library's V-HACD pieces bulge a median 6.7 mm beyond the visible surface (D16), enough to put
the fingers on material the camera cannot see. CoACD follows concave shapes much more closely.

Settings: concavity threshold 0.04 (smaller = more faithful, more pieces), no cap on the number of pieces,
manifold preprocessing when needed, seed 0. Pieces under 1 mm^3 are dropped (MuJoCo rejects degenerate meshes).

    python scripts/make_collision.py --models <dir> --names list.txt [--jobs 64]
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

PARAMS = dict(threshold=0.04, max_convex_hull=-1, preprocess_mode="auto", preprocess_resolution=50,
              resolution=2000, mcts_nodes=20, mcts_iterations=150, mcts_max_depth=3, pca=False, merge=True,
              seed=0)
MIN_VOLUME = 1e-9   # m^3


def run_one(model_dir):
    os.environ["OMP_NUM_THREADS"] = "2"
    import coacd
    import numpy as np
    import trimesh
    coacd.set_log_level("error")
    t0 = time.time()
    out = os.path.join(model_dir, "coacd")
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.startswith("model_collision_"):
            os.remove(os.path.join(out, f))
    mesh = trimesh.load(os.path.join(model_dir, "model.obj"), force="mesh", process=True)
    parts = coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)), **PARAMS)
    k = 0
    for v, f in parts:
        hull = trimesh.Trimesh(v, f, process=True).convex_hull
        if len(hull.vertices) < 4 or hull.volume < MIN_VOLUME:
            continue
        hull.export(os.path.join(out, f"model_collision_{k}.obj"))
        k += 1
    json.dump(dict(PARAMS, pieces=k, seconds=round(time.time() - t0, 1)), open(os.path.join(out, "params.json"), "w"))
    return os.path.basename(model_dir), k, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--jobs", type=int, default=64)
    a = ap.parse_args()
    names = [n.strip() for n in open(a.names) if n.strip()]
    with Pool(min(a.jobs, len(names))) as pool:
        for name, k, dt in pool.imap_unordered(run_one, [os.path.join(a.models, n) for n in names]):
            print(f"  {name[:56]:56s} {k:3d} pieces  {dt:6.1f} s", flush=True)


if __name__ == "__main__":
    main()

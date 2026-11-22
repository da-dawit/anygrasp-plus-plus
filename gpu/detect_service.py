"""Detector service for trials that need a new detection mid-trial (the M3 regrasp).

Watches <queue>/in for scene .npz files (the format run_b0.py writes), runs the detector the same way as
detect_grasps.py, and writes the cache file to <queue>/out/<name>.npz. Several copies can share one queue, each
claims a file by renaming it into <queue>/work first.

    python gpu/detect_service.py --detector gsnet --queue /root/detq      # run 4-6 of these in the background
"""
import argparse
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_grasps import Detector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=["gsnet", "baseline"], default="gsnet")
    ap.add_argument("--queue", default="/root/detq")
    ap.add_argument("--weights", default="/root/weights")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    inq, work, outq = (os.path.join(a.queue, s) for s in ("in", "work", "out"))
    for p in (inq, work, outq):
        os.makedirs(p, exist_ok=True)
    det = Detector(a.detector, a.weights)
    print(f"detector service ready (pid {os.getpid()})", flush=True)
    while True:
        names = sorted(f for f in os.listdir(inq) if f.endswith(".npz"))
        if not names:
            time.sleep(0.2)
            continue
        for f in names:
            mine = os.path.join(work, f"{os.getpid()}__{f}")
            try:
                os.rename(os.path.join(inq, f), mine)
            except OSError:          # another copy took it
                continue
            n = f[:-4]
            t0 = time.perf_counter()
            s = np.load(mine)
            pts = s["points"].astype(np.float32)
            scene_pts = s["scene_points"].astype(np.float32) if "scene_points" in s else pts
            seed = (zlib.crc32(n.encode()) + a.seed) % 2**32
            if len(pts) == 0:
                grasps, collision, order, t_net = np.zeros((0, 17), np.float32), np.zeros(0, bool), np.zeros(0, int), 0.0
            else:
                grasps, collision, order, t_net = det.run(pts, scene_pts, seed)
            tmp = os.path.join(outq, n + ".tmp.npz")
            np.savez_compressed(tmp, grasps=grasps, collision=collision, order=order, seed=seed, n_points_in=len(pts),
                                seconds_net=t_net, seconds_total=time.perf_counter() - t0)
            os.replace(tmp, os.path.join(outq, n + ".npz"))
            os.remove(mine)
            print(f"  {n}: {len(order)} after NMS, {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()

"""Run a frozen grasp detector (GSNet or graspnet-baseline) once per scene and cache the result.

    python gpu/detect_grasps.py --detector gsnet --scenes DIR --out DIR [--shard i/n]

Input: a directory of <name>.npz scenes, each with
    points        (N,3) float32, metres, camera frame, OpenCV convention (x right, y down, z forward).
                  What the detector sees; crop it to the workspace first like the authors do.
    scene_points  optional (M,3) float32, same frame, only for the collision check (default: points).
                  Pass the full cloud with table and bin walls so grasps that hit them get flagged.
    K             optional (3,3) intrinsics, only for plotting.

Output: <out_dir>/<detector>/<name>.npz with
    grasps     (G,17) float32, graspnetAPI layout: score, width, height, depth, R (9, row-major), t (3), object id (-1).
               columns of R are the grasp axes: x = approach, y = closing direction, z = finger height.
    collision  (G,) bool, True = rejected by the model-free collision check.
    order      (K,) int64 indices into grasps: collision-free, after NMS, best score first.
    seed, n_points_in, seconds_net (forward + decode), seconds_total.

Cached scenes are skipped. Point sampling is seeded per scene (CRC32 of the name + --seed) so a rerun feeds the
network the same input. Settings follow the authors' graspness_unofficial/infer_vis_grasp.py and
graspnet-baseline/demo.py. Only one detector per process, since both repos have a top-level models/graspnet.py.
"""
import argparse
import os
import sys
import time
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "16")  # MinkowskiEngine warns and picks this itself when unset

import numpy as np
import torch

SRC =os.environ.get("DETECTOR_SRC", "/root/src")
ROOTS = {"gsnet": os.path.join(SRC, "graspness_unofficial"), "baseline": os.path.join(SRC, "graspnet-baseline")}
WEIGHTS = {"gsnet": "minkuresunet_realsense.tar", "baseline": "checkpoint-rs.tar"}
NUM_POINT = {"gsnet": 15000, "baseline": 20000}
VOXEL_GSNET = 0.005


def sample_indices(n, k, rng):
    """The authors' sampling: k distinct points if available, otherwise all points plus random repeats."""
    if n >= k:
        return rng.choice(n, k, replace=False)
    return np.concatenate([np.arange(n), rng.choice(n, k - n, replace=True)])


class Detector:
    def __init__(self, name, weights_dir, device="cuda:0"):
        self.name, self.device = name, torch.device(device)
        root = ROOTS[name]
        if name == "gsnet":
            # pointnet2/pointnet2_utils.py does a bare `import pytorch_utils`, so its folder must be on the path
            sys.path += [root, os.path.join(root, "utils"), os.path.join(root, "pointnet2")]
            from models.graspnet import GraspNet, pred_decode
            from dataset.graspnet_dataset import minkowski_collate_fn
            self.collate = minkowski_collate_fn
            net = GraspNet(seed_feat_dim=512, is_training=False)
        else:
            sys.path += [os.path.join(root, "models"), os.path.join(root, "utils")]
            from graspnet import GraspNet, pred_decode
            net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.05,
                           hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
        from collision_detector import ModelFreeCollisionDetector
        from graspnetAPI import GraspGroup
        self.pred_decode, self.MFCD, self.GraspGroup = pred_decode, ModelFreeCollisionDetector, GraspGroup
        ckpt = torch.load(os.path.join(weights_dir, WEIGHTS[name]), map_location=self.device)
        net.load_state_dict(ckpt["model_state_dict"])  # strict: every key must match
        self.epoch = ckpt.get("epoch")
        self.net = net.to(self.device).eval()

    @torch.no_grad()
    def forward(self, cloud):
        """cloud: (num_point,3) float32 sampled input. Returns (G,17) float32 raw predictions."""
        if self.name == "gsnet":
            batch = self.collate([{"point_clouds": cloud, "coors": cloud / VOXEL_GSNET,
                                   "feats": np.ones_like(cloud)}])
            for k in batch:
                batch[k] = batch[k].to(self.device)
        else:
            batch = {"point_clouds": torch.from_numpy(cloud[None]).to(self.device)}
        preds = self.pred_decode(self.net(batch))
        return preds[0].float().cpu().numpy()

    def run(self, points, scene_points, seed):
        rng = np.random.default_rng(seed)
        cloud = points[sample_indices(len(points), NUM_POINT[self.name], rng)].astype(np.float32)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        grasps = self.forward(cloud)
        torch.cuda.synchronize()
        t_net = time.perf_counter() - t0
        gg = self.GraspGroup(grasps.copy())
        collision = self.MFCD(scene_points, voxel_size=0.01).detect(gg, approach_dist=0.05, collision_thresh=0.01)
        order = nms_order(self.GraspGroup, grasps, ~collision)
        return grasps, np.asarray(collision, bool), order, t_net


def nms_order(GraspGroup, grasps, keep):
    """Indices of the grasps that survive graspnetAPI NMS among `keep`, best score first.
    The row index rides through NMS in the object-id column, which NMS does not compare."""
    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        return idx.astype(np.int64)
    tagged = grasps[idx].copy()
    tagged[:, 16] = idx
    out = GraspGroup(tagged).nms().sort_by_score()
    return out.grasp_group_array[:, 16].round().astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--detector", choices=["gsnet", "baseline"], required=True)
    ap.add_argument("--scenes", required=True, help="directory of <name>.npz scene clouds")
    ap.add_argument("--out", required=True, help="cache directory; results go to <out>/<detector>/")
    ap.add_argument("--weights", default="/root/weights")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="recompute even if the cached result exists")
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th scene from i (parallel runs)")
    a = ap.parse_args()

    names = sorted(f[:-4] for f in os.listdir(a.scenes) if f.endswith(".npz"))
    out_dir = os.path.join(a.out, a.detector)
    os.makedirs(out_dir, exist_ok=True)
    i, n_sh = (int(v) for v in a.shard.split("/"))
    todo = [n for n in names[i::n_sh] if a.force or not os.path.exists(os.path.join(out_dir, n + ".npz"))]
    print(f"{a.detector}: {len(names)} scenes, {len(names) - len(todo)} cached, {len(todo)} to run")
    if not todo:
        return
    det = Detector(a.detector, a.weights)
    print(f"loaded {WEIGHTS[a.detector]} (epoch {det.epoch}) on {torch.cuda.get_device_name(0)}")
    for n in todo:
        t0 = time.perf_counter()
        s = np.load(os.path.join(a.scenes, n + ".npz"))
        pts = s["points"].astype(np.float32)
        scene_pts = s["scene_points"].astype(np.float32) if "scene_points" in s else pts
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0 or not np.isfinite(pts).all():
            raise ValueError(f"{n}: points must be a non-empty finite (N,3) array, got {pts.shape}")
        seed = (zlib.crc32(n.encode()) + a.seed) % 2**32
        grasps, collision, order, t_net = det.run(pts, scene_pts, seed)
        tmp = os.path.join(out_dir, n + ".tmp.npz")
        np.savez_compressed(tmp, grasps=grasps, collision=collision, order=order, seed=seed,
                            n_points_in=len(pts), seconds_net=t_net, seconds_total=time.perf_counter() - t0)
        os.replace(tmp, os.path.join(out_dir, n + ".npz"))  # atomic: a killed run never leaves a half file
        top = grasps[order[0], 0] if len(order) else float("nan")
        print(f"  {n}: {len(grasps)} raw, {int((~collision).sum())} collision-free, {len(order)} after NMS, "
              f"top score {top:.3f}, net {t_net * 1e3:.0f} ms")


if __name__ == "__main__":
    main()

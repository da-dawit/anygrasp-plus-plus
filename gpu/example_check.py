"""End-to-end check of the detector cache on the authors' example scene (graspnet-baseline/doc/example_data,
a RealSense D435 capture at 1280x720).

    make   -> builds <scenes>/example.npz from the depth image, exactly as graspnet-baseline/demo.py does
    check  -> sanity statistics per detector, a rerun-determinism comparison, and an overlay PNG

Statistics reported for the best K collision-free, NMS-filtered grasps of each detector:
    surface distance  distance from the grasp centre t to the nearest scene point; the detectors place t on the
                      object surface, so this should be a few millimetres
    approach . ray    cosine between the approach axis R[:,0] and the camera ray to t; > 0 means the gripper
                      comes from the camera side, which is the only side the camera can see
    width             predicted opening; the Panda hand opens to 0.08 m
"""
import argparse
import os
import sys

import numpy as np

SRC = os.environ.get("DETECTOR_SRC", "/root/src")
EXAMPLE = os.path.join(SRC, "graspnet-baseline", "doc", "example_data")
PANDA_MAX_WIDTH = 0.08


def make(a):
    from PIL import Image
    import scipy.io as scio
    sys.path.append(os.path.join(SRC, "graspnet-baseline", "utils"))
    from data_utils import CameraInfo, create_point_cloud_from_depth_image
    depth = np.array(Image.open(os.path.join(EXAMPLE, "depth.png")))
    ws = np.array(Image.open(os.path.join(EXAMPLE, "workspace_mask.png")))
    meta = scio.loadmat(os.path.join(EXAMPLE, "meta.mat"))
    K, factor = meta["intrinsic_matrix"], meta["factor_depth"]
    cam = CameraInfo(1280.0, 720.0, K[0][0], K[1][1], K[0][2], K[1][2], factor)
    cloud = create_point_cloud_from_depth_image(depth, cam, organized=True)
    valid = depth > 0
    os.makedirs(a.scenes, exist_ok=True)
    np.savez_compressed(os.path.join(a.scenes, "example.npz"), points=cloud[ws.astype(bool) & valid].astype(np.float32),
                        scene_points=cloud[valid].astype(np.float32), K=K.astype(np.float64))
    print(f"example.npz: {int((ws.astype(bool) & valid).sum())} workspace points, {int(valid.sum())} scene points")


def gripper_lines(g):
    """Line segments of the graspnetAPI gripper drawing (plot_gripper_pro_max) for one 17-column grasp row."""
    w, d, R, t = g[1], g[3], g[4:13].reshape(3, 3), g[13:16]
    base, tail = 0.02, 0.04
    loc = [((-base, -w / 2), (d, -w / 2)), ((-base, w / 2), (d, w / 2)),
           ((-base, -w / 2), (-base, w / 2)), ((-base, 0), (-base - tail, 0))]
    return [(t + R @ np.array([p[0], p[1], 0.0]), t + R @ np.array([q[0], q[1], 0.0])) for p, q in loc]


def stats(name, r, scene_pts, k):
    from scipy.spatial import cKDTree
    g, col, order = r["grasps"], r["collision"], r["order"]
    top = g[order[:k]]
    t, R = top[:, 13:16], top[:, 4:13].reshape(-1, 3, 3)
    dist = cKDTree(scene_pts).query(t)[0]
    ray = t / np.linalg.norm(t, axis=1, keepdims=True)
    cos = np.einsum("ij,ij->i", R[:, :, 0], ray)
    ortho = np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max()
    print(f"\n[{name}] {len(g)} raw -> {int((~col).sum())} collision-free -> {len(order)} after NMS "
          f"(net {float(r['seconds_net']) * 1e3:.0f} ms, seed {int(r['seed'])})")
    print(f"  top-{len(top)} score       {top[0, 0]:.3f} (best) ... {top[-1, 0]:.3f}")
    print(f"  surface distance    median {np.median(dist) * 1e3:.1f} mm, max {dist.max() * 1e3:.1f} mm")
    print(f"  approach . ray      min {cos.min():.2f}, median {np.median(cos):.2f} (> 0: from the camera side)")
    print(f"  width               {top[:, 1].min() * 100:.1f}-{top[:, 1].max() * 100:.1f} cm, "
          f"{int((top[:, 1] <= PANDA_MAX_WIDTH).sum())}/{len(top)} fit the Panda (<= 8 cm)")
    print(f"  rotation |R R^T-I|  {ortho:.1e}")
    return dict(dist_med=np.median(dist), cos_med=np.median(cos), ortho=ortho)


def check(a):
    s = np.load(os.path.join(a.scenes, "example.npz"))
    scene_pts, K = s["scene_points"], s["K"]
    dets = [d for d in ("baseline", "gsnet") if os.path.exists(os.path.join(a.cache, d, "example.npz"))]
    ok = True
    for d in dets:
        r = np.load(os.path.join(a.cache, d, "example.npz"))
        st = stats(d, r, scene_pts, a.k)
        ok &= bool(st["dist_med"] < 0.01 and st["cos_med"] > 0 and st["ortho"] < 1e-4)
        if a.rerun:
            r2 = np.load(os.path.join(a.rerun, d, "example.npz"))
            same = all(np.array_equal(r[x], r2[x]) for x in ("grasps", "collision", "order"))
            dmax = np.abs(r["grasps"] - r2["grasps"]).max() if r["grasps"].shape == r2["grasps"].shape else np.inf
            print(f"  rerun               identical: {same} (max |diff| {dmax:.1e})")
    print("\nSANITY", "PASS" if ok else "FAIL")
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        img = np.array(Image.open(os.path.join(EXAMPLE, "color.png")))
        fig, axes = plt.subplots(1, len(dets), figsize=(8 * len(dets), 5), squeeze=False)
        for ax, d in zip(axes[0], dets):
            r = np.load(os.path.join(a.cache, d, "example.npz"))
            ax.imshow(img)
            top = r["grasps"][r["order"][:a.k]]
            cmap = plt.get_cmap("viridis")
            for i, g in enumerate(top[::-1]):  # best drawn last, on top
                c = cmap(1 - (len(top) - 1 - i) / max(len(top) - 1, 1))
                for p, q in gripper_lines(g):
                    uv = [(K[0, 0] * x[0] / x[2] + K[0, 2], K[1, 1] * x[1] / x[2] + K[1, 2]) for x in (p, q)]
                    ax.plot([uv[0][0], uv[1][0]], [uv[0][1], uv[1][1]], color=c, lw=1.6)
            ax.set_title(f"{d}: best {len(top)} grasps (yellow = highest score)")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(a.png, dpi=110, bbox_inches="tight")
        print("wrote", a.png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "check"])
    ap.add_argument("--scenes", default="/root/run/scenes")
    ap.add_argument("--cache", default="/root/run/cache")
    ap.add_argument("--rerun", default=None, help="second cache directory to compare against")
    ap.add_argument("--png", default=None)
    ap.add_argument("-k", type=int, default=20)
    a = ap.parse_args()
    make(a) if a.cmd == "make" else check(a)

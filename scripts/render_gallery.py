"""Stills for the project page:

    gallery/<name>.png      each study object in its resting pose, see-through, with its centre of mass at the five
                            hidden shifts (0, 2, 4, 6, 8 cm; white to red). The outside looks the same at every shift.
    depth_ideal.png         the overhead camera's ideal rendered depth of one development scene
    depth_d435.png          the same view through the RealSense D435 model the detector sees (D17)

    MUJOCO_GL=egl B0_MASS=0.45 python scripts/render_gallery.py --out results/midpoint
"""
import argparse
import csv
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import run_b0 as B  # noqa: E402
from anygrasp_pp import objects as O  # noqa: E402
from anygrasp_pp import scene as S  # noqa: E402

W, H = 900, 700


def gallery(out):
    import mujoco
    os.makedirs(os.path.join(out, "gallery"), exist_ok=True)
    for r in csv.DictReader(open("results/objects/final_set.csv")):
        model_dir = os.path.join(B.MODELS[r["source"]], r["name"])
        mass = B.MASS_KG if B.feasible(r) else 1.0
        ob = O.build(model_dir, collision=B.COLLISION, mass=mass)
        R, z = S.rest_poses(ob)[0][:2]
        pos = np.array([S.PICK[0], S.PICK[1], S.TABLE_Z + z + 0.0005])
        pos[:2] -= (R @ ob["centroid"])[:2]
        xml = S.scene_xml([S.object_xml("obj", model_dir, ob, max(O.SHIFTS), pos, R, collision=B.COLLISION)])
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == m.body("obj").id and m.geom_matid[g] >= 0:
                m.mat_rgba[m.geom_matid[g], 3] = 0.35
        m.vis.global_.offwidth, m.vis.global_.offheight = W, H
        cam = mujoco.MjvCamera()
        cam.lookat[:] = pos + R @ ob["centroid"]
        cam.distance, cam.azimuth, cam.elevation = 1.15 * ob["length"] + 0.12, 125.0, -32.0
        opt = mujoco.MjvOption()
        with mujoco.Renderer(m, H, W) as rr:
            rr.update_scene(d, camera=cam, scene_option=opt)
            scn = rr.scene
            for k, s in enumerate(O.SHIFTS):
                c = pos + R @ ob["props"][s]["com"]
                t = k / (len(O.SHIFTS) - 1)
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.0075 + 0.003 * t, 0, 0]), c,
                                    np.eye(3).flatten(), np.array([1.0, 1 - 0.85 * t, 1 - 0.85 * t, 1], np.float32))
                g.emission = 0.6
                scn.ngeom += 1
            img = rr.render()
        from PIL import Image
        Image.fromarray(img).save(os.path.join(out, "gallery", r["name"] + ".png"))
        print("  gallery", r["name"], f"{mass} kg")


def depth(out, tag="048_hammer__s8__k0"):
    import mujoco
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = next(t for t in B.trials("results/objects/final_set.csv") if t["tag"] == tag)
    m, d, arm, _ = B.build(tr)
    B.settle(m, d, arm)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 0, 1, 0, 0, 0]
    with mujoco.Renderer(m, S.CAM_H, S.CAM_W) as r:
        r.enable_depth_rendering()
        r.update_scene(d, camera="overhead", scene_option=opt)
        dl = r.render().copy()
        r.update_scene(d, camera="overhead_right", scene_option=opt)
        dr = r.render().copy()
    real = S.realsense_depth(dl, dr, S.camera_K(), np.random.default_rng(zlib.crc32(("noise|" + tag).encode())))
    ys, xs = slice(150, 600), slice(380, 980)
    lo, hi = np.percentile(dl[ys, xs], [1, 99])
    for name, img in (("depth_ideal", dl), ("depth_d435", real)):
        a = img[ys, xs].astype(float)
        a[a <= 0] = np.nan
        cm = matplotlib.colormaps["turbo_r"].copy()
        cm.set_bad("black")
        plt.imsave(os.path.join(out, name + ".png"), np.ma.masked_invalid(a), cmap=cm, vmin=lo, vmax=hi)
    print("  depth images of", tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/midpoint")
    a = ap.parse_args()
    gallery(a.out)
    depth(a.out)


if __name__ == "__main__":
    main()

"""Contact sheet of candidate objects (textured visual mesh), rendered headless with MuJoCo.
Red dot = centroid of the heavy core picked by anygrasp_pp.objects.

Usage (Linux with EGL):  MUJOCO_GL=egl python scripts/render_candidates.py --models <dir> --names list.txt --out sheet.png
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def render_one(args):
    model_dir, size = args
    import mujoco
    from anygrasp_pp import objects as O
    spec = mujoco.MjSpec.from_file(os.path.join(model_dir, "model.xml"))
    ob = O.build(model_dir)
    spec.worldbody.add_light(pos=[0, 0, 2], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    spec.worldbody.add_light(pos=[1, 1, 1], dir=[-1, -1, -1], diffuse=[0.4, 0.4, 0.4])
    if ob["core"] is not None:
        F = ob["points"][ob["core"]["F"]]
        spec.worldbody.add_site(pos=F.mean(0), size=[0.006, 0, 0], rgba=[1, 0, 0, 1], group=2)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = size, size
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = ob["centroid"]
    cam.distance = 1.6 * ob["length"]
    cam.azimuth, cam.elevation = 45, -55
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [0, 0, 1, 0, 0, 0]
    opt.sitegroup[:] = [0, 0, 1, 0, 0, 0]
    with mujoco.Renderer(m, size, size) as r:
        r.update_scene(d, camera=cam, scene_option=opt)
        img = r.render()
    return os.path.basename(model_dir), img, ob["length"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--names", required=True, help="text file, one model name per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--cols", type=int, default=8)
    a = ap.parse_args()
    names = [n.strip() for n in open(a.names) if n.strip()]
    with Pool(min(32, len(names))) as pool:
        res = pool.map(render_one, [(os.path.join(a.models, n), a.size) for n in names])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = -(-len(res) // a.cols)
    fig, axes = plt.subplots(rows, a.cols, figsize=(2.4 * a.cols, 2.6 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for i, (name, img, L) in enumerate(res):
        ax = np.ravel(axes)[i]
        ax.imshow(img)
        ax.set_title(f"{i + 1}. {name[:26]}\n{L * 100:.0f} cm", fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=80)
    print("wrote", a.out, len(res), "objects")


if __name__ == "__main__":
    main()

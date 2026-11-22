"""Plain rollout videos for the project page: same trials as the logs, one fixed camera, plain floor and sky, a
small label. Two policies on one scene side by side; left/right are b0 (run_b0.run_one) or agpp/retry (run_m3).

    MUJOCO_GL=egl B0_MASS=0.45 python scripts/render_plain.py --run results/b0_m045_v2 \
        --pairs results/midpoint/m3_video_picks.json --out results/midpoint/videos_m3
"""
import argparse
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import run_b0 as B  # noqa: E402
import run_m3 as M3  # noqa: E402
from anygrasp_pp import scene as S  # noqa: E402

FPS, W, H = 30, 960, 720
CAM = dict(lookat=(0.50, 0.24, 0.20), distance=1.45, azimuth=150.0, elevation=-17.0)
LABEL = {"b0": "B0", "agpp": "AnyGrasp++", "retry": "B0 + blind retry"}

_scene_xml = S.scene_xml


def _plain_scene_xml(*args, **kw):
    xml = _scene_xml(*args, **kw)
    xml = xml.replace('builtin="checker" rgb1="0.82 0.74 0.6" rgb2="0.78 0.7 0.56"',
                      'builtin="flat" rgb1="0.80 0.78 0.75" rgb2="0.80 0.78 0.75"')
    return xml.replace("<asset>", '<asset>\n    <texture type="skybox" builtin="gradient" rgb1="0.97 0.97 0.97" '
                                  'rgb2="0.86 0.87 0.89" width="256" height="1536"/>', 1)


S.scene_xml = _plain_scene_xml        # visual only: textures don't enter the physics or the depth image


def ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


class Film:
    def __init__(self, path, label):
        self.path, self.label, self.phase, self.next_t, self.r = path, label, "", 0.0, None

    def __call__(self, m, d):
        if self.r is None:
            import mujoco
            from PIL import ImageFont
            import matplotlib
            m.vis.global_.offwidth, m.vis.global_.offheight = max(W, m.vis.global_.offwidth), max(H, m.vis.global_.offheight)
            self.r = mujoco.Renderer(m, H, W)
            self.cam = mujoco.MjvCamera()
            self.cam.lookat[:] = CAM["lookat"]
            self.cam.distance, self.cam.azimuth, self.cam.elevation = CAM["distance"], CAM["azimuth"], CAM["elevation"]
            self.font = ImageFont.truetype(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf",
                                                        "DejaVuSans.ttf"), 30)
            self.p = subprocess.Popen([ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                       f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                       "-crf", "23", "-preset", "medium", "-movflags", "+faststart", self.path],
                                      stdin=subprocess.PIPE)
        if d.time + 1e-9 < self.next_t:
            return
        self.next_t += 1.0 / FPS
        self.r.update_scene(d, camera=self.cam)
        self.write(self.r.render())

    def write(self, img):
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        ImageDraw.Draw(im).text((22, 16), self.label, font=self.font, fill=(40, 40, 40))
        self.p.stdin.write(np.asarray(im).tobytes())

    def close(self, hold_s=1.0):
        if self.r is None:
            return
        img = self.r.render()
        for _ in range(int(hold_s * FPS)):
            self.write(img)
        self.p.stdin.close()
        self.p.wait()
        self.r.close()


def film(args):
    run, dev, tr, kind, path, loo, as_run = args
    f = Film(path, LABEL[kind])
    if kind == "b0":
        B.FRAME_HOOK = f
        r = B.run_one((dict(tr, cand=0), run))
    else:
        M3.FRAME_HOOK = f
        # the work folder's name enters the seeds of the new image after a set-down, so it must match the logged run
        r = M3.run_trial((tr, run, dev, os.path.join(os.path.dirname(path), as_run), kind, loo))
    f.close()
    return dict(tag=tr["tag"], kind=kind, outcome=r.get("outcome"), success=bool(r.get("success")),
                tries=len(r.get("tries", [])) or 1, video=os.path.basename(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/b0_m045_v2")
    ap.add_argument("--dev", default="results/b0_m045_v2")
    ap.add_argument("--pairs", required=True, help='json list of {"tag", "left", "right"}')
    ap.add_argument("--out", default="results/midpoint/videos_m3")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--as-run", default="m3_dev", help="name of the logged M3 run folder these trials come from")
    ap.add_argument("--logged", default="results/m3_dev", help="the logged run, to check each video reproduces it")
    a = ap.parse_args()
    for sub in (f"{a.as_run}/scenes", f"{a.as_run}/cache"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    pairs = json.load(open(a.pairs))
    seeds = sorted({int(p["tag"].split("__k")[1]) for p in pairs})
    trs = {t["tag"]: t for t in B.trials("results/objects/final_set.csv", range(min(seeds), max(seeds) + 1))}
    loo = os.path.abspath(a.run) == os.path.abspath(a.dev)
    jobs = [(a.run, a.dev, trs[p["tag"]], k, os.path.join(a.out, f"{p['tag']}__{k}.mp4"), loo, a.as_run)
            for p in pairs for k in (p["left"], p["right"])]
    with Pool(min(a.jobs, len(jobs)), maxtasksperchild=1) as pool:
        res = pool.map(film, jobs, chunksize=1)
    logged = {}
    for k in ("agpp", "retry"):
        f = os.path.join(a.logged, f"{k}_parts.jsonl")
        if os.path.exists(f):
            logged[k] = {r["tag"]: r for r in (json.loads(x) for x in open(f))}
    manifest = []
    for p in pairs:
        l, r = (next(x for x in res if x["tag"] == p["tag"] and x["kind"] == k) for k in (p["left"], p["right"]))
        both = os.path.join(a.out, p["tag"] + ".mp4")
        subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", os.path.join(a.out, l["video"]), "-i",
                        os.path.join(a.out, r["video"]), "-filter_complex",
                        "[0:v]pad=iw+8:ih:0:0:white[a];[a][1:v]hstack=inputs=2:shortest=0,scale=1440:-2[v]",
                        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart",
                        both], check=True)
        subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-ss", "2", "-i", both, "-frames:v", "1", "-q:v", "3",
                        both.replace(".mp4", ".jpg")], check=True)
        manifest.append(dict(p, video=os.path.basename(both), left_outcome=l["outcome"], right_outcome=r["outcome"],
                             left_tries=l["tries"], right_tries=r["tries"],
                             right_logged=logged.get(p["right"], {}).get(p["tag"], {}).get("outcome")))
        print(p["tag"], LABEL[p["left"]], l["outcome"], "|", LABEL[p["right"]], r["outcome"], f"({r['tries']} tries)",
              "logged:", manifest[-1]["right_logged"], flush=True)
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

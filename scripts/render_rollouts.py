"""Rollout videos: re-run trials exactly as run_b0.py does (same scene, grasp, controller and outcome rule) while a
camera records them, then put two grasps on the same scene side by side.

The recorder only reads the simulation state, so the physics is untouched. It makes the object see-through and
draws two markers the robot never sees: red at the true centre of mass, blue at the visible centre the camera's
point cloud suggests. Each panel shows the phase and the object's tilt in the hand; the verdict comes from the
trial itself, and the manifest records whether each re-run reproduced the logged outcome.

    MUJOCO_GL=egl python scripts/render_rollouts.py --run results/b0_m045_v2 \
        --picks results/midpoint/video_picks.json --out results/midpoint/videos
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import run_b0 as B  # noqa: E402
from anygrasp_pp import m2 as M2  # noqa: E402
from anygrasp_pp import scene as S  # noqa: E402

FPS = 30
W, H = 960, 720
CAM = dict(lookat=(0.56, 0.22, 0.20), distance=1.45, azimuth=150.0, elevation=-22.0)
FONT = None


def font(size):
    from PIL import ImageFont
    import matplotlib
    path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
    return ImageFont.truetype(path, size)


def ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


class Recorder:
    """FRAME_HOOK for run_b0.run_one: renders at FPS, overlays the phase and the tilt in the hand, streams to ffmpeg."""

    def __init__(self, path, title, c_vis):
        self.path, self.title, self.c_vis = path, title, c_vis
        self.phase, self.next_t, self.r, self.proc = "", 0.0, None, None
        self.R0, self.tilt, self.tilt_max, self.c_body, self.n = None, 0.0, 0.0, None, 0

    def _start(self, m, d):
        import mujoco
        self.mj = mujoco
        self.obj = m.body("obj").id
        self.hand = m.body("hand").id
        for g in range(m.ngeom):              # see-through object (rendering only)
            if m.geom_bodyid[g] == self.obj and m.geom_matid[g] >= 0:
                m.mat_rgba[m.geom_matid[g], 3] = 0.32
        m.vis.global_.offwidth, m.vis.global_.offheight = max(W, m.vis.global_.offwidth), max(H, m.vis.global_.offheight)
        self.r = mujoco.Renderer(m, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = CAM["lookat"]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = CAM["distance"], CAM["azimuth"], CAM["elevation"]
        Ro, po = d.xmat[self.obj].reshape(3, 3), d.xpos[self.obj]
        self.c_body = Ro.T @ (self.c_vis - po)          # the visible centre, fixed to the object from the capture
        self.proc = subprocess.Popen([ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                      "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                                      "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "medium",
                                      "-movflags", "+faststart", self.path], stdin=subprocess.PIPE)
        self.fbig, self.fsmall = font(30), font(24)

    def _sphere(self, pos, rgba, size=0.017):
        scn = self.r.scene
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        self.mj.mjv_initGeom(g, self.mj.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]),
                             np.asarray(pos, float), np.eye(3).flatten(), np.array(rgba, np.float32))
        g.emission = 0.7
        scn.ngeom += 1

    def __call__(self, m, d):
        if self.r is None:
            self._start(m, d)
        Rh, Ro = d.xmat[self.hand].reshape(3, 3), d.xmat[self.obj].reshape(3, 3)
        if self.phase in ("lift and hold", "carry"):
            if self.R0 is None:
                self.R0 = Rh.T @ Ro
            rv = Rot.from_matrix((Rh.T @ Ro) @ self.R0.T).as_rotvec()
            self.tilt = float(np.degrees(np.hypot(rv[0], rv[1])))
            self.tilt_max = max(self.tilt_max, self.tilt)
        if d.time + 1e-9 < self.next_t:
            return
        self.next_t += 1.0 / FPS
        self.r.update_scene(d, camera=self.cam)
        self._sphere(d.xipos[self.obj], (0.95, 0.15, 0.15, 1.0))
        self._sphere(Ro @ self.c_body + d.xpos[self.obj], (0.15, 0.45, 0.95, 1.0), 0.014)
        self.write(self.r.render())

    def write(self, img, verdict=None):
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im, "RGBA")
        dr.rectangle([0, 0, W, 54], fill=(15, 18, 28, 190))
        dr.text((18, 10), self.title, font=self.fbig, fill=(255, 255, 255))
        dr.rectangle([0, H - 46, W, H], fill=(15, 18, 28, 160))
        dr.text((18, H - 38), self.phase, font=self.fsmall, fill=(230, 230, 230))
        x = 250
        for txt, c in (("true centre of mass", (255, 80, 80)), ("visible centre", (90, 150, 255))):
            dr.ellipse([x, H - 31, x + 14, H - 17], fill=c)
            dr.text((x + 20, H - 36), txt, font=self.fsmall, fill=(220, 220, 220))
            x += 30 + dr.textlength(txt, font=self.fsmall)
        if self.R0 is not None:
            col = (255, 110, 100) if self.tilt_max > B.MAX_TILT else (140, 230, 150)
            dr.text((W - 262, 15), f"tilt in hand {self.tilt:5.1f}°", font=self.fsmall, fill=col)
        if verdict:
            txt, ok = verdict
            col = (40, 160, 80, 230) if ok else (200, 50, 50, 230)
            tw = dr.textlength(txt, font=self.fbig)
            dr.rounded_rectangle([W / 2 - tw / 2 - 24, H / 2 - 34, W / 2 + tw / 2 + 24, H / 2 + 34], 14, fill=col)
            dr.text((W / 2 - tw / 2, H / 2 - 20), txt, font=self.fbig, fill=(255, 255, 255))
        self.proc.stdin.write(np.asarray(im).tobytes())
        self.n += 1

    def close(self, verdict, hold_s=2.0):
        img = self.r.render()
        for _ in range(int(hold_s * FPS)):
            self.write(img, verdict)
        self.proc.stdin.close()
        self.proc.wait()
        self.r.close()


VERDICT = {"success": "Held. In the bin", "turned in hand": "Turned in the hand", "dropped": "Dropped",
           "missed the bin": "Missed the bin"}


def visible_centre(run, tag):
    sc = np.load(os.path.join(run, "scenes", tag + ".npz"))
    pts = M2.object_points(sc["points"] @ sc["R_wc"].T + sc["p_wc"])
    return M2.com_prior(pts)[0]


def one(args):
    run, tr, cand, title, path = args
    rec = Recorder(path, title, visible_centre(run, tr["tag"]))
    B.FRAME_HOOK = rec
    r = B.run_one((dict(tr, cand=cand), run))
    B.FRAME_HOOK = None
    if rec.r is None:
        return dict(r, video=None)
    out = r.get("outcome", "error")
    rec.phase = "done"
    rec.close((VERDICT.get(out, out[:1].upper() + out[1:]), bool(r.get("success"))))
    return dict(r, video=os.path.basename(path), frames=rec.n)


def side_by_side(left, right, out):
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", left, "-i", right, "-filter_complex",
                    "[0:v][1:v]hstack=inputs=2:shortest=0,scale=1440:-2[v]", "-map", "[v]", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "25", "-preset", "medium", "-movflags", "+faststart", out],
                   check=True)
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-sseof", "-1", "-i", out, "-frames:v", "1",
                    "-q:v", "3", out.replace(".mp4", ".jpg")], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/b0_m045_v2")
    ap.add_argument("--picks", default="results/midpoint/video_picks.json")
    ap.add_argument("--out", default="results/midpoint/videos")
    ap.add_argument("--jobs", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    picks = json.load(open(a.picks))
    logged = {(r["tag"], int(r["cand"])): r for r in csv.DictReader(open(os.path.join(a.run, "candidates.csv")))}
    trs = {t["tag"]: t for t in B.trials("results/objects/final_set.csv")}
    jobs = []
    for p in picks:
        tag, left, right = p["tag"], p["left"], p["right"]
        for cand, title, side in ((left, p["left_title"], "L"), (right, p["right_title"], "R")):
            jobs.append((a.run, trs[tag], cand, title, os.path.join(a.out, f"{tag}__{side}{cand}.mp4")))
    with Pool(min(a.jobs, len(jobs))) as pool:
        res = pool.map(one, jobs, chunksize=1)
    manifest = []
    for p in picks:
        tag = p["tag"]
        rl, rr = (next(r for r in res if r["tag"] == tag and r["cand"] == c) for c in (p["left"], p["right"]))
        both = os.path.join(a.out, f"{tag}.mp4")
        if rl.get("video") and rr.get("video"):
            side_by_side(os.path.join(a.out, rl["video"]), os.path.join(a.out, rr["video"]), both)
        entry = dict(p, video=os.path.basename(both))
        for side, r in (("left", rl), ("right", rr)):
            lg = logged.get((tag, int(r["cand"])), {})
            entry[side + "_outcome"] = r.get("outcome")
            entry[side + "_logged"] = lg.get("outcome")
            entry[side + "_reproduced"] = lg.get("outcome") == r.get("outcome")
            entry[side + "_lever_arm_cm"] = r.get("lever_arm_cm")
            entry[side + "_dstar_cm"] = r.get("dstar_lift_cm")
        manifest.append(entry)
        print(f"{tag:60s} L{p['left']}: {rl.get('outcome')} (logged {entry['left_logged']})   "
              f"R{p['right']}: {rr.get('outcome')} (logged {entry['right_logged']})", flush=True)
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

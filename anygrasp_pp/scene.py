"""Scene pieces: scanned objects with hidden cores, table, bin, the overhead D435 camera and its depth noise,
and the camera <-> world conversions the detector needs.

Frames
    world:  robot base at the origin, table top at z = TABLE_Z, x away from the robot
    MuJoCo camera: looks along -z_cam, x_cam right, y_cam up (OpenGL convention)
    OpenCV camera: x right, y down, z forward; what GraspNet-1Billion and the detectors use
"""
import os

import mujoco
import numpy as np

from . import objects as O
from .contact_rig import CONTACT, object_mu3
from .physics import MU, A_PAD

TABLE_Z = 0.0
PICK = np.array([0.50, 0.0, TABLE_Z])     # center of the pick area [m]
CAM_W, CAM_H = 1280, 720                  # RealSense D435 depth stream used by GraspNet-1Billion
CAM_FY = 927.37                           # GraspNet-1Billion RealSense intrinsics, fy [px]
CAM_FOVY = float(np.degrees(2 * np.arctan(CAM_H / 2 / CAM_FY)))
CAM_DIST = 0.65                           # camera to pick-area center [m]


def camera_position(tilt_deg, dist=CAM_DIST, pick=PICK):
    """Camera on the far side of the pick area from the robot, tilt_deg off vertical, looking at the pick center."""
    t = np.radians(tilt_deg)
    return pick + dist * np.array([np.sin(t), 0.0, np.cos(t)])


CAM_POS = camera_position(np.degrees(np.arctan2(0.25, 0.60)))   # 22.6 deg off vertical: (0.25, 0, 0.60) from PICK
CROP = 0.30                               # half side of the square workspace crop around PICK [m]


def look_at(pos, target, up=(0.0, 0.0, 1.0)):
    """MuJoCo camera axes (x_cam, y_cam) as rows, for a camera at pos looking at target."""
    f = np.asarray(target, float) - pos
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    if np.linalg.norm(x) < 1e-9:          # looking straight down: let the image's up point toward the robot (-x)
        x = np.cross(f, (-1.0, 0.0, 0.0))
    x /= np.linalg.norm(x)
    y = np.cross(-f, x)                   # y_cam = z_cam x x_cam with z_cam = -f
    return x, y


def camera_K(w=CAM_W, h=CAM_H, fovy=CAM_FOVY):
    """Pinhole K of a MuJoCo camera in pixel-index coords: square pixels, centered principal point, pixel (u, v)
    centered at continuous (u + 1/2, v + 1/2)."""
    f = (h / 2) / np.tan(np.radians(fovy) / 2)
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])


def R_world_cv(x_cam, y_cam):
    """OpenCV camera frame -> world rotation. Columns x_cv = x_cam, y_cv = -y_cam, z_cv = -z_cam (forward)."""
    z_cam = np.cross(x_cam, y_cam)
    return np.stack([x_cam, -y_cam, -z_cam], axis=1)


def backproject(depth, K, max_depth=2.0):
    """OpenCV-frame points of every valid pixel of a planar depth image."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    ok = (depth > 0) & (depth < max_depth)
    z = depth[ok]
    return np.stack([(u[ok] - K[0, 2]) * z / K[0, 0], (v[ok] - K[1, 2]) * z / K[1, 1], z], 1)


STEREO_BASELINE = 0.050   # m, D435 left-right imager baseline
SUBPIXEL_RMS = 0.08       # px, Intel's typical D435 disparity (subpixel) RMS
DEPTH_UNIT = 0.001        # m, z16 depth units (GraspNet-1Billion factor_depth = 1000)
EDGE_JUMP = 0.015         # m, depth jump inside a 3x3 window that marks an edge pixel
EDGE_DROP = 0.5           # probability an edge pixel is invalid


def stereo_shadow(depth_left, depth_right, K, baseline=STEREO_BASELINE, tol=0.003):
    """Left-image pixels whose surface point the right imager can't see. Rectified pair, right camera = left
    moved by `baseline` along x_cam, so left (u, v) at depth z lands at (u - f b / z, v); occluded if the right
    image sees something nearer there."""
    h, w = depth_left.shape
    f = K[0, 0]
    v, u = np.mgrid[0:h, 0:w]
    z = depth_left
    ok = z > 0
    ur = np.round(u - f * baseline / np.where(ok, z, np.inf)).astype(int)
    inside = ok & (ur >= 0) & (ur < w)
    zr = np.where(inside, depth_right[v, np.clip(ur, 0, w - 1)], np.inf)
    return ok & (~inside | (zr < z - tol))


def realsense_depth(depth_left, depth_right, K, rng, baseline=STEREO_BASELINE, sigma_d=SUBPIXEL_RMS,
                    unit=DEPTH_UNIT, edge_jump=EDGE_JUMP, edge_drop=EDGE_DROP):
    """Ideal planar depth -> what a D435 would report: stereo shadows become holes, disparity d = f b / z gets
    N(0, sigma_d^2) noise, edge pixels drop out with probability edge_drop, depth quantized to `unit`.
    Invalid pixels are 0, as on the sensor."""
    z = depth_left.astype(np.float64).copy()
    ok = z > 0
    fb = K[0, 0] * baseline
    hole = stereo_shadow(depth_left, depth_right, K, baseline)
    d = np.where(ok, fb / np.where(ok, z, 1.0), 0.0) + rng.normal(0.0, sigma_d, z.shape)
    z = np.where(ok & (d > 0), fb / np.where(d > 0, d, 1.0), 0.0)
    pad = np.pad(depth_left, 1, mode="edge")
    win = np.stack([pad[i:i + z.shape[0], j:j + z.shape[1]] for i in range(3) for j in range(3)])
    edge = (win.max(0) - win.min(0)) > edge_jump
    drop = edge & (rng.random(z.shape) < edge_drop)
    z = np.round(z / unit) * unit
    z[hole | drop | ~ok] = 0.0
    return z.astype(np.float32)


def workspace_masks(world_pts, above=0.01, margin=0.02, crop=CROP, pick=PICK, table_z=TABLE_Z):
    """(wide, tight) masks over world points, like GraspNet-1Billion feeds its detectors (no oracle).
    wide:  the square pick area, table included (for the collision check)
    tight: footprint of everything more than `above` over the known table height, grown by `margin` (their
           workspace mask uses object extent + 2 cm), table points inside it kept."""
    w = world_pts
    wide = (np.abs(w[:, 0] - pick[0]) <= crop) & (np.abs(w[:, 1] - pick[1]) <= crop) & (w[:, 2] >= table_z - 0.01)
    up = wide & (w[:, 2] > table_z + above)
    if not up.any():
        return wide, np.zeros_like(wide)
    lo, hi = w[up, :2].min(0) - margin, w[up, :2].max(0) + margin
    tight = wide & (w[:, 0] >= lo[0]) & (w[:, 0] <= hi[0]) & (w[:, 1] >= lo[1]) & (w[:, 1] <= hi[1])
    return wide, tight


def grasps_to_world(grasps, R_wc, p_wc):
    """graspnetAPI rows (17 columns, camera frame) -> world rotation (N,3,3) and translation (N,3)."""
    R = grasps[:, 4:13].reshape(-1, 3, 3)
    return R_wc @ R, grasps[:, 13:16] @ R_wc.T + p_wc


def approach_ok_world(R_world, max_deg=30.0):
    """Approach axis (first column) within max_deg of straight down."""
    return R_world[:, 2, 0] <= -np.cos(np.radians(max_deg))


def _quat_wxyz(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def object_xml(tag, model_dir, ob, shift, pos, R, collision="vhacd", mu=MU, pads="grid", contact=CONTACT):
    """(assets, body) MJCF snippets for one scanned object with its hidden core at this shift.
    pos, R: pose of the mesh frame in the world. The inertial is explicit, so geoms carry no mass."""
    p = ob["props"][shift]
    I = p["inertia"]
    coll_dir = O.collision_dir(model_dir, collision)
    pieces = sorted((f for f in os.listdir(coll_dir) if f.startswith("model_collision_") and f.endswith(".obj")),
                    key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))
    assets = [f'<texture type="2d" name="{tag}_tex" file="{os.path.join(model_dir, "texture.png")}"/>',
              f'<material name="{tag}_mat" texture="{tag}_tex" specular="0.3" shininess="0.3"/>',
              f'<mesh name="{tag}_vis" file="{os.path.join(model_dir, "model.obj")}"/>']
    geoms = [f'<geom name="{tag}_vis" type="mesh" mesh="{tag}_vis" material="{tag}_mat" contype="0" conaffinity="0" '
             f'group="2" mass="0"/>']
    fr = f'{mu} {object_mu3(pads, mu, A_PAD)} 0.0001'
    extra = (f' solref="{contact["solref"]}"' if contact.get("solref") else "") + \
            (f' solimp="{contact["solimp"]}"' if contact.get("solimp") else "")
    for k, f in enumerate(pieces):
        assets.append(f'<mesh name="{tag}_c{k}" file="{os.path.join(coll_dir, f)}"/>')
        geoms.append(f'<geom name="{tag}_c{k}" type="mesh" mesh="{tag}_c{k}" group="3" priority="1" condim="4" '
                     f'friction="{fr}"{extra} mass="0" rgba="0.35 0.55 0.85 1"/>')
    q = _quat_wxyz(R)
    body = (f'<body name="{tag}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
            f'quat="{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}">\n  <freejoint name="{tag}"/>\n'
            f'  <inertial mass="{p["mass"]:.9f}" pos="{p["com"][0]:.9f} {p["com"][1]:.9f} {p["com"][2]:.9f}" '
            f'fullinertia="{I[0, 0]:.6e} {I[1, 1]:.6e} {I[2, 2]:.6e} {I[0, 1]:.6e} {I[0, 2]:.6e} {I[1, 2]:.6e}"/>\n  '
            + "\n  ".join(geoms) + "\n</body>")
    return "\n".join(assets), body


BIN_CENTER = PICK + np.array([0.0, 0.52, 0.0])   # outside the camera crop; a 40 cm object fits inside
BIN_INNER, BIN_WALL_H, BIN_WALL_T = 0.40, 0.10, 0.01


def bin_geoms():
    c, h, t, w = BIN_CENTER, BIN_WALL_H, BIN_WALL_T, BIN_INNER / 2
    walls = [((c[0], c[1] - w - t / 2), (w + t, t / 2)), ((c[0], c[1] + w + t / 2), (w + t, t / 2)),
             ((c[0] - w - t / 2, c[1]), (t / 2, w)), ((c[0] + w + t / 2, c[1]), (t / 2, w))]
    return "\n".join(f'<geom name="bin{k}" type="box" pos="{p[0]:.4f} {p[1]:.4f} {TABLE_Z + h / 2:.4f}" '
                     f'size="{s[0]:.4f} {s[1]:.4f} {h / 2:.4f}" rgba="0.3 0.45 0.6 1"/>'
                     for k, (p, s) in enumerate(walls))


BIN_REST_Z = 0.25        # an object resting in the bin has its CoM below this (rim 0.10 m; a 29 cm figure standing: 0.16)


def in_bin(p, free_of_hand=True):
    """CoM inside the bin footprint and low enough to be resting there (standing, lying or leaning on a wall),
    and the hand has let go."""
    return bool(abs(p[0] - BIN_CENTER[0]) < BIN_INNER / 2 and abs(p[1] - BIN_CENTER[1]) < BIN_INNER / 2
                and p[2] < TABLE_Z + BIN_REST_Z and free_of_hand)


def scene_xml(object_snippets, timestep=0.002, impratio=10, contact=CONTACT, cam_pos=CAM_POS, cam_target=PICK,
              robot=None, with_bin=False):
    x, y = look_at(cam_pos, cam_target)
    cr = np.asarray(cam_pos) + STEREO_BASELINE * x          # the D435's right imager
    assets = "\n".join(a for a, _ in object_snippets)
    bodies = "\n".join(b for _, b in object_snippets)
    return f"""
<mujoco model="anygrasp_pp_scene">
  <compiler angle="radian" autolimits="true"{f' meshdir="{robot["meshdir"]}"' if robot else ""}/>
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic" impratio="{impratio}"
          noslip_iterations="{contact.get('noslip', 0)}"/>
  <visual><global offwidth="{CAM_W}" offheight="{CAM_H}"/><map znear="0.05" zfar="4"/></visual>
  {"<default>" + robot["default"] + "</default>" if robot else ""}
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.74 0.6" rgb2="0.78 0.7 0.56" width="64" height="64"/>
    <material name="table" texture="grid" texrepeat="8 8"/>
{assets}
{robot["asset"] if robot else ""}
  </asset>
  <worldbody>
    <light pos="0.5 -0.5 1.5" dir="0 0.3 -1" directional="true" diffuse="0.7 0.7 0.7"/>
    <light pos="0.5 0.5 1.5" dir="0 -0.3 -1" directional="true" diffuse="0.3 0.3 0.3"/>
    <geom name="table" type="plane" size="1.2 1.2 0.01" pos="0 0 {TABLE_Z}" material="table"/>
    <camera name="overhead" pos="{cam_pos[0]:.6f} {cam_pos[1]:.6f} {cam_pos[2]:.6f}"
            xyaxes="{x[0]:.9f} {x[1]:.9f} {x[2]:.9f} {y[0]:.9f} {y[1]:.9f} {y[2]:.9f}" fovy="{CAM_FOVY:.9f}"/>
    <camera name="overhead_right" pos="{cr[0]:.6f} {cr[1]:.6f} {cr[2]:.6f}"
            xyaxes="{x[0]:.9f} {x[1]:.9f} {x[2]:.9f} {y[0]:.9f} {y[1]:.9f} {y[2]:.9f}" fovy="{CAM_FOVY:.9f}"/>
{bodies}
{robot["body"] if robot else ""}
{bin_geoms() if with_bin else ""}
  </worldbody>
{"".join(robot[k] for k in ("tendon", "equality", "contact", "actuator")) if robot else ""}
</mujoco>"""


def _support_margin(hull_pts_world, com_world, tol=1e-3):
    """Signed distance [m] of the CoM's ground projection inside the support polygon (negative = outside)."""
    from scipy.spatial import ConvexHull
    low = hull_pts_world[hull_pts_world[:, 2] < hull_pts_world[:, 2].min() + tol][:, :2]
    if len(low) < 3:
        return -np.inf
    try:
        hull = ConvexHull(low)
    except Exception:
        return -np.inf
    a, b = hull.equations[:, :2], hull.equations[:, 2]      # a.x + b <= 0 inside, |a| = 1
    return float(-(a @ com_world[:2] + b).max())


def rest_poses(ob, min_margin=0.005):
    """Resting orientations of the object on a flat table, most probable first, keeping only those whose
    support polygon contains the CoM with min_margin at every shift. Returns a list of (R, z, prob, margin): R
    rotates the mesh frame, z lifts it so the lowest point touches the table."""
    import trimesh
    pts = ob["points"]
    hull = trimesh.convex.convex_hull(pts)
    com0 = ob["props"][0.0]["com"]
    T, prob = trimesh.poses.compute_stable_poses(hull, center_mass=com0)
    out = []
    for Tk, pk in zip(T, prob):
        R = Tk[:3, :3]
        v = hull.vertices @ R.T
        z = -v[:, 2].min()
        v[:, 2] += z
        margins = [_support_margin(v, R @ ob["props"][s]["com"] + [0, 0, z]) for s in O.SHIFTS]
        if min(margins) >= min_margin:
            out.append((R, z, float(pk), min(margins)))
    return out

"""Franka Panda for the study: Menagerie arm and hand (pinned commit) with the 4x4 rubber grid pads, the 70 N
force-clamped gripper and 7 torque motors at the Panda's limits (87 N m joints 1-4, 12 N m joints 5-7).
No gravcomp on the links, so the joint torques carry the payload's weight (M3 reads it from them).
"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .contact_rig import FRANKA, PAD_Z, PAD_HALF, CLOSED, CLOSE_SPEED, _pad_geoms, _grip_actuator
from .physics import FN

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
TAU_MAX = np.array([87, 87, 87, 87, 12, 12, 12], float)
KP = np.array([600, 600, 600, 600, 250, 150, 50], float)       # libfranka joint_impedance example
KD = np.array([50, 50, 50, 20, 20, 20, 10], float)
Q_READY = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4])
Q_CAPTURE = Q_READY + np.array([-np.pi / 2, 0, 0, 0, 0, 0, 0])   # arm turned to -y, out of the camera's crop
MAX_OPEN = 0.08            # m, total opening


def panda_parts(pads="grid", fn=FN):
    """MJCF pieces of the robot (strings): default, asset, body, tendon, equality, actuator, contact."""
    root = ET.parse(FRANKA / "panda.xml").getroot()
    panda_def = root.find("default/default[@class='panda']")
    pad = ET.SubElement(panda_def, "default", {"class": "pad"})
    ET.SubElement(pad, "geom", {"condim": "4", "group": "3", "rgba": "0.9 0.2 0.2 1"})
    for name in ("left_finger", "right_finger"):
        body = root.find(f".//body[@name='{name}']")
        for g in list(body.findall("geom")):
            if g.get("class", "").startswith("fingertip_pad_collision"):
                body.remove(g)
            elif g.get("class") == "collision":
                g.set("name", f"{name}_mesh")
        for k, g in enumerate(ET.fromstring(f"<x>{_pad_geoms(pads)}</x>")):
            g.set("name", f"{name}_pad{k}")
            body.append(g)
    hand = root.find(".//body[@name='hand']")
    for g in hand.findall("geom"):
        if g.get("class") == "collision":
            g.set("name", "palm")
    ET.SubElement(hand, "site", {"name": "tcp", "pos": f"0 0 {PAD_Z}", "size": "0.004", "group": "4"})
    s = lambda e: ET.tostring(e, encoding="unicode")
    motors = "\n".join(f'<motor name="tau{i + 1}" joint="{j}" ctrlrange="{-TAU_MAX[i]} {TAU_MAX[i]}"/>'
                       for i, j in enumerate(ARM_JOINTS))
    return dict(default=s(panda_def), asset="\n".join(s(a) for a in root.find("asset")),
                body=s(root.find("worldbody/body")), tendon=s(root.find("tendon")),
                equality=s(root.find("equality")), contact=s(root.find("contact")),
                actuator=f"<actuator>\n{motors}\n{_grip_actuator('force', fn)}\n</actuator>",
                meshdir=str(FRANKA / "assets"))


class Panda:
    """Indices into a compiled model that contains panda_parts()."""

    def __init__(self, m):
        self.m = m
        self.qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        self.dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
        self.act = np.array([m.actuator(f"tau{i}").id for i in range(1, 8)])
        self.grip = m.actuator("grip").id
        self.tcp = m.site("tcp").id
        self.hand = m.body("hand").id
        self.lo, self.hi = m.jnt_range[[m.joint(j).id for j in ARM_JOINTS]].T
        self.pad_geoms = {g for g in range(m.ngeom) if "_pad" in (m.geom(g).name or "")}
        self.finger_mesh = {m.geom("left_finger_mesh").id, m.geom("right_finger_mesh").id}
        self.palm = m.geom("palm").id
        self.robot_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(("link", "hand", "left_f", "right_f"))}

    def set_q(self, d, q):
        d.qpos[self.qadr] = q
        d.qvel[self.dadr] = 0.0

    def control(self, d, q_ref, qd_ref, qdd_ref=None):
        """tau = M(q) qdd_ref + c(q, qd) + g(q) + KP (q_ref - q) + KD (qd_ref - qd), clipped to the motor limits.
        M, c, g come from the arm-and-hand model only, so like the real robot it doesn't know the payload."""
        q, qd = d.qpos[self.qadr], d.qvel[self.dadr]
        tau = KP * (q_ref - q) + KD * (qd_ref - qd) + d.qfrc_bias[self.dadr]
        if qdd_ref is not None:
            v = np.zeros(self.m.nv)
            v[self.dadr] = qdd_ref
            Mv = np.zeros(self.m.nv)
            mujoco.mj_mulM(self.m, d, Mv, v)
            tau += Mv[self.dadr]
        d.ctrl[self.act] = np.clip(tau, -TAU_MAX, TAU_MAX)

    def tcp_pose(self, d):
        return d.site_xpos[self.tcp].copy(), d.site_xmat[self.tcp].reshape(3, 3).copy()

    def ik(self, dk, p_t, R_t, q0, iters=300, tol_p=1e-4, tol_r=np.radians(0.2)):
        """Damped least squares IK on the TCP site, joint limits enforced. dk: a scratch MjData.
        Returns (q, ok, position error, orientation error)."""
        m = self.m
        dk.qpos[:] = 0
        dk.qpos[self.qadr] = q0
        Jp, Jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, dk)
            mujoco.mj_comPos(m, dk)
            p, R = dk.site_xpos[self.tcp], dk.site_xmat[self.tcp].reshape(3, 3)
            er = 0.5 * sum(np.cross(R[:, i], R_t[:, i]) for i in range(3))
            e = np.concatenate([p_t - p, er])
            if np.linalg.norm(e[:3]) < tol_p and np.linalg.norm(er) < tol_r * 0.5:
                break
            mujoco.mj_jacSite(m, dk, Jp, Jr, self.tcp)
            J = np.vstack([Jp[:, self.dadr], Jr[:, self.dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
            dk.qpos[self.qadr] = np.clip(dk.qpos[self.qadr] + dq, self.lo, self.hi)
        mujoco.mj_kinematics(m, dk)
        p, R = dk.site_xpos[self.tcp], dk.site_xmat[self.tcp].reshape(3, 3)
        ep = float(np.linalg.norm(p_t - p))
        eang = float(np.arccos(np.clip((np.trace(R.T @ R_t) - 1) / 2, -1, 1)))
        q = dk.qpos[self.qadr].copy()
        return q, ep < 1e-3 and eang < np.radians(1.0), ep, eang


def min_jerk_scale(T, t):
    """Minimum-jerk time scaling: s(t), s'(t), s''(t) for 0 <= t <= T, held at the ends."""
    u = np.clip(t / T, 0.0, 1.0)
    inside = 0.0 < t < T
    f = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
    fd = (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / T if inside else 0.0
    fdd = (60 * u - 180 * u ** 2 + 120 * u ** 3) / T ** 2 if inside else 0.0
    return f, fd, fdd


def min_jerk(q0, q1, T, t):
    """Position, velocity, acceleration on the min-jerk path from q0 to q1 in time T."""
    f, fd, fdd = min_jerk_scale(T, t)
    q0, q1 = np.asarray(q0), np.asarray(q1)
    return q0 + (q1 - q0) * f, (q1 - q0) * fd, (q1 - q0) * fdd


def grasp_to_tcp(R_w, t_w, depth):
    """GraspNet grasp -> Panda TCP target (position, rotation). R_w columns = approach, closing, height; t_w on
    the object surface; depth = fingertip depth past t_w. Pad centre = t + approach * (depth - PAD_HALF), hand
    z = approach, hand y = closing."""
    a, c = R_w[:, 0], R_w[:, 1]
    R = np.stack([np.cross(c, a), c, a], axis=1)
    return t_w + a * (depth - PAD_HALF), R


def grip_ctrl_open(width):
    """Grip target for an opening `width` (total) before closing: each finger at width / 2."""
    return min(width, MAX_OPEN) / 2


def grip_ctrl_closing(w0, t):
    """Closing target t s after the close command, from opening w0. Same as contact_rig.grip_target: moves at
    CLOSE_SPEED and runs past contact to CLOSED so the force clamp sets 70 N per pad."""
    return max(w0 / 2 - CLOSE_SPEED * t, CLOSED)

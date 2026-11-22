"""Floating Franka hand + an object with a hidden weight, for checking MuJoCo contacts against the slip model.

No arm: the hand hangs from a vertical slide. Object long axis = world x, fingers close along world y, so a CoM
shift d along x twists the object about the closing axis.
Object geoms have priority 1 so MuJoCo takes all contact params from the object (equal priority mixes them).
Pads: "grid" = 4x4 points over the 17 x 17 mm fingertip (the study), "point" = one sphere + torsional friction
(textbook soft finger, for checking the simulator), "stock" = Menagerie's 5 boxes.
"""
from pathlib import Path

import numpy as np

from .physics import MU, FN, MASS, A_PAD, torsional_mu

ROOT = Path(__file__).resolve().parents[1]
FRANKA = ROOT / "assets" / "menagerie" / "franka_emika_panda"

BOX_HALF = (0.10, 0.02, 0.02)   # default object half sizes [m]
X_HIDDEN = 0.09                 # where the hidden weight sits along the long axis [m]
PAD_CENTRE_Z = 0.0445           # pad centre along the finger's z axis (Menagerie fingertip_pad_collision_1)
PAD_Z = 0.0584 + PAD_CENTRE_Z   # hand origin -> pad centre along the hand's z axis [m]
PAD_HALF = 0.0085               # half side of the fingertip face [m] (Menagerie: 17 x 17 mm)
PAD_FACE_Y = 0.0                # fingertip face in the finger frame (Menagerie's protruding pads end at y = 0)
GRID_N = 4                      # grid pad: GRID_N x GRID_N points at the cell centres of the face
GRID_R = 0.0015                 # radius of each grid point sphere [m]

OPEN, CLOSED = 0.04, -0.01      # grip targets (tendon length = finger position); CLOSED is past contact
CLOSE_SPEED = 0.1               # [m/s], Franka Hand max finger speed

# friction is made hard by the noslip solver. noslip_iterations is only a cap (it stops at tolerance): 20 left
# 0.8 deg of false creep on the grid pad, 2000/5000/20000 give identical results even misaligned (500 does not).
# nominal fingertip: 2 mm rubber layer, E = 3 MPa, as a Winkler bed. each grid point is a linear spring
# k_c = E A_cell / h = 27.1 kN/m, so a load on a few points sinks more there, like the real layer
# (scripts/calibrate_pad_layer.py). solimp "d d" keeps the spring linear; solref "-K -b" is set so a free 1 kg
# object sees 27.1 kN/m per point (flat sink 0.161 mm at 70 N).
CONTACT = dict(solref="-5537.89 -148.834", solimp="0.95 0.95 0.001", noslip=2000)
# step-1 nominal, kept for comparison: 60 kN/m per point and stiffer with depth, so misaligned or curved
# contacts only loaded the edge rows
CONTACT_V1 = dict(solref="0.01 1", solimp="0.95 0.99 0.001", noslip=2000)
# hard plastic pads (0.003-0.007 mm sink), sensitivity case since misaligned grasps depend a lot on pad softness
RIGID_CONTACT = dict(solref="0.004 1", solimp="0.99 0.999 0.001", noslip=2000)
DEFAULT_CONTACT = dict(solref=None, solimp=None, noslip=0)       # MuJoCo defaults, for comparison only


def grid_points(n=GRID_N, half=PAD_HALF):
    """(x, z) of the n x n cell centres on the pad face, relative to the pad centre."""
    c = (np.arange(n) + 0.5) / n * 2 * half - half
    return np.array([(x, z) for x in c for z in c])


def grip_target(t, t_close):
    """Grip ctrl: open, then a target closing at CLOSE_SPEED (like Franka's grasp(width, speed, force))."""
    return OPEN if t < t_close else max(OPEN - CLOSE_SPEED * (t - t_close), CLOSED)


def _pad_geoms(pads):
    if pads == "stock":  # copied from Menagerie panda.xml fingertip_pad_collision_1..5
        boxes = [("0.0085 0.004 0.0085", "0 0.0055 0.0445"), ("0.003 0.002 0.003", "0.0055 0.002 0.05"),
                 ("0.003 0.002 0.003", "-0.0055 0.002 0.05"), ("0.003 0.002 0.0035", "0.0055 0.002 0.0395"),
                 ("0.003 0.002 0.0035", "-0.0055 0.002 0.0395")]
        return "".join(f'<geom class="pad" type="box" size="{s}" pos="{p}"/>' for s, p in boxes)
    if pads == "point":
        r = 0.008
        return f'<geom class="pad" type="sphere" size="{r}" pos="0 {PAD_FACE_Y + r} {PAD_CENTRE_Z}"/>'
    if pads == "grid":
        return "".join(f'<geom class="pad" type="sphere" size="{GRID_R}" '
                       f'pos="{x:.6f} {PAD_FACE_Y + GRID_R} {PAD_CENTRE_Z + z:.6f}"/>' for x, z in grid_points())
    raise ValueError(pads)


def _grip_actuator(grip, fn):
    if grip == "stock":  # Menagerie actuator8, unchanged: ctrl 255 = open, 0 = closed
        return ('<general name="grip" tendon="split" forcerange="-100 100" ctrlrange="0 255" '
                'gainprm="0.01568627451 0 0" biasprm="0 -100 -10" dyntype="none" biastype="affine"/>')
    if grip == "force":
        # stiff position servo on the finger tendon, force clamped at 2 fn. the target runs past the object, so
        # once the fingers are blocked the clamp is active and each pad gets exactly fn
        kp = 2 * fn / 0.005                      # saturates 5 mm past contact
        kv = 2 * (kp * 0.23) ** 0.5              # critical damping; 0.23 kg = tendon-space finger inertia
        return (f'<general name="grip" tendon="split" ctrlrange="{CLOSED} {OPEN}" forcerange="{-2*fn} {2*fn}" '
                f'gainprm="{kp} 0 0" biasprm="0 {-kp} {-kv}" dyntype="none" biastype="affine"/>')
    raise ValueError(grip)


def object_mu3(pads, mu, a):
    """Torsional friction per contact. The point pad needs the whole patch's twist resistance in its one
    contact; on the grid pad it comes from the spread of the points, so each point gets ~0."""
    return torsional_mu(mu, a) if pads == "point" else 1e-6


MIDAIR_Z = 0.30                 # object centre height for mid-air grasps (gravity off until the lift)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2)


def build_xml(d, grip="force", pads="grid", mu=MU, a=A_PAD, fn=FN, mass=MASS, contact=CONTACT,
              shape="box", size=BOX_HALF, x_hidden=X_HIDDEN, yaw_deg=0.0, roll_deg=0.0, offset=(0.0, 0.0, 0.0),
              midair=False, timestep=0.002, impratio=10, menagerie_friction=False):
    """MJCF string. d = CoM shift along the object's long axis [m].

    shape "box": size = half sizes (x, y, z); "cylinder": size = (half length, radius), axis along x.
    roll_deg rolls object and hand about the long axis, so the closing axis tilts up by roll_deg (needs midair).
    offset: hand position from the object centre (x long axis, y closing axis, z up) [m].
    midair: object starts at MIDAIR_Z, gravity only comes on at the lift.
    menagerie_friction: MuJoCo's default object friction ("Menagerie as shipped")."""
    if roll_deg and not midair:
        raise ValueError("a rolled grasp needs midair=True (a rolled box cannot rest on the table)")
    m_hidden = mass * d / x_hidden
    m_shell = mass - m_hidden
    if not 0 <= m_hidden < mass:
        raise ValueError("shift too large for this hidden-weight position")
    hidden = (f'<body name="hidden" pos="{x_hidden} 0 0"><inertial mass="{m_hidden}" pos="0 0 0" '
              f'diaginertia="1e-6 1e-6 1e-6"/></body>') if m_hidden > 0 else ""
    if menagerie_friction:
        obj_params = 'condim="3"'
    else:
        obj_params = f'condim="4" friction="{mu} {object_mu3(pads, mu, a)} 0.0001"'
    if contact.get("solref"):
        obj_params += f' solref="{contact["solref"]}"'
    if contact.get("solimp"):
        obj_params += f' solimp="{contact["solimp"]}"'
    if shape == "box":
        hx, hy, hz = size
        geom = f'<geom name="obj" class="object" type="box" size="{hx} {hy} {hz}" mass="{m_shell}"/>'
        z_centre = hz
    elif shape == "cylinder":
        hl, rad = size
        geom = (f'<geom name="obj" class="object" type="cylinder" size="{rad} {hl}" quat="0.7071068 0 0.7071068 0" '
                f'mass="{m_shell}"/>')
        z_centre = rad
    else:
        raise ValueError(shape)
    if midair:
        z_centre = MIDAIR_Z
    yaw, roll = np.radians(yaw_deg), np.radians(roll_deg)
    q_roll = (np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0)                 # about world x (the long axis)
    q_obj = " ".join(f"{v:.9f}" for v in _quat_mul(q_roll, (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))))
    q_wrist = " ".join(f"{v:.9f}" for v in q_roll)
    # pad centre sits at the object centre (+ offset): the wrist frame is rolled, the pad is PAD_Z "down" in it
    c, s = np.cos(roll), np.sin(roll)
    pad_to_wrist = np.array([0.0, -s * PAD_Z, c * PAD_Z])                  # R_x(roll) @ (0, 0, PAD_Z)
    hp = np.array([0.0, 0.0, z_centre]) + pad_to_wrist + np.asarray(offset, float)
    hand_pos = f"{hp[0]:.9f} {hp[1]:.9f} {hp[2]:.9f}"
    finger = lambda name, quat: f"""
        <body name="{name}" pos="0 0 0.0584" {quat} gravcomp="1">
          <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
          <joint name="{name}_joint" class="finger"/>
          <geom mesh="finger_0" material="off_white" class="visual"/>
          <geom mesh="finger_1" material="black" class="visual"/>
          <geom name="{name}_mesh" mesh="finger_0" class="collision"/>
          {_pad_geoms(pads)}
        </body>"""
    return f"""
<mujoco model="contact_rig">
  <compiler angle="radian" meshdir="{FRANKA / 'assets'}" autolimits="true"/>
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic" impratio="{impratio}"
          noslip_iterations="{contact.get('noslip', 0)}"/>
  <default>
    <default class="panda">
      <joint armature="0.1" damping="1"/>
      <default class="finger"><joint axis="0 1 0" type="slide" range="0 0.04"/></default>
      <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="2"/></default>
      <default class="collision"><geom type="mesh" group="3"/></default>
      <default class="pad"><geom condim="4" group="3" rgba="0.9 0.2 0.2 1"/></default>
    </default>
    <default class="object"><geom priority="1" {obj_params} rgba="0.35 0.55 0.85 1"/></default>
  </default>
  <asset>
    <material name="white" rgba="1 1 1 1"/>
    <material name="off_white" rgba="0.901961 0.921569 0.929412 1"/>
    <material name="black" rgba="0.25 0.25 0.25 1"/>
    <mesh name="hand_c" file="hand.stl"/>
    <mesh file="hand_0.obj"/><mesh file="hand_1.obj"/><mesh file="hand_2.obj"/>
    <mesh file="hand_3.obj"/><mesh file="hand_4.obj"/>
    <mesh file="finger_0.obj"/><mesh file="finger_1.obj"/>
  </asset>
  <worldbody>
    <light pos="0 -0.5 1.5" dir="0 0.3 -1" directional="true"/>
    <geom name="table" type="plane" size="1 1 0.01" rgba="0.8 0.7 0.55 1"/>
    <body name="mount" pos="{hand_pos}" gravcomp="1">
      <joint name="lift" type="slide" axis="0 0 1"/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="1e-3 1e-3 1e-3"/>
      <body name="wrist" quat="{q_wrist}" gravcomp="1">
      <body name="hand" childclass="panda" quat="0 1 0 0" gravcomp="1">
        <inertial mass="0.73" pos="-0.01 0 0.03" diaginertia="0.001 0.0025 0.0017"/>
        <geom mesh="hand_0" material="off_white" class="visual"/>
        <geom mesh="hand_1" material="black" class="visual"/>
        <geom mesh="hand_2" material="black" class="visual"/>
        <geom mesh="hand_3" material="white" class="visual"/>
        <geom mesh="hand_4" material="off_white" class="visual"/>
        <geom name="palm" mesh="hand_c" class="collision"/>
        {finger("left_finger", "")}
        {finger("right_finger", 'quat="0 0 0 1"')}
      </body>
      </body>
    </body>
    <body name="obj" pos="0 0 {z_centre}" quat="{q_obj}">
      <freejoint name="obj"/>
      {geom}
      {hidden}
    </body>
  </worldbody>
  <contact>
    <exclude body1="hand" body2="left_finger"/>
    <exclude body1="hand" body2="right_finger"/>
  </contact>
  <tendon>
    <fixed name="split"><joint joint="left_finger_joint" coef="0.5"/><joint joint="right_finger_joint" coef="0.5"/></fixed>
  </tendon>
  <equality>
    <joint joint1="left_finger_joint" joint2="right_finger_joint" solimp="0.95 0.99 0.001" solref="0.005 1"/>
  </equality>
  <actuator>
    <position name="lift" joint="lift" kp="20000" kv="500"/>
    {_grip_actuator(grip, fn)}
  </actuator>
</mujoco>"""

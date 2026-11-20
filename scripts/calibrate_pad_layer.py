"""The fingertip as a thin rubber layer (decisions D20; derivations B.6).

Winkler model of a thin layer (thickness h, Young's modulus E) on a rigid finger: each cell of the 4x4 pad grid,
area A_c = (2 PAD_HALF / 4)^2, is an independent linear spring k_c = E A_c / h. Against a surface whose gap to
the pad face is s_i at cell i, the cells sink delta_i = max(0, delta0 - s_i), and delta0 is set by the squeeze
force F = sum_i k_c delta_i.

MuJoCo side: with a constant impedance d (solimp d d) and a direct stiffness K (solref -K -b), each contact point
behaves in equilibrium as a linear spring of stiffness K / ((1-d) A_ii) (B.5), A_ii being the contact's inverse
inertia, dominated here by the finger (its armature). This script
  1. measures A_ii-weighted stiffness and sets k so a flat face sinks like the layer (0.161 mm at 70 N);
  2. compares, with nothing refitted, the per-point forces against the Winkler solution for a flat face, faces
     misaligned by 1, 2 and 4 deg, and a 15 mm radius cylinder, and the resulting twist capacity d* (A.12).

    python scripts/calibrate_pad_layer.py
"""
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anygrasp_pp import physics as P  # noqa: E402
from anygrasp_pp.contact_rig import (build_xml, grid_points, grip_target, GRID_N, PAD_HALF, PAD_CENTRE_Z,  # noqa: E402
                                     CONTACT)

E_RUBBER = 3.0e6          # Pa
H_LAYER = 0.002           # m
A_CELL = (2 * PAD_HALF / GRID_N) ** 2
K_CELL = E_RUBBER * A_CELL / H_LAYER       # N/m per grid point
D_IMP = 0.95              # constant impedance
FN = 70.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "pad")


def winkler(gaps, F=FN, k=K_CELL):
    """Per-cell forces of a Winkler layer pressed against gaps s_i with total force F (bisection on delta0)."""
    gaps = np.asarray(gaps, float)
    lo, hi = gaps.min(), gaps.min() + F / k + 1e-3
    for _ in range(200):
        mid = (lo + hi) / 2
        if (k * np.maximum(0, mid - gaps)).sum() < F:
            lo = mid
        else:
            hi = mid
    return k * np.maximum(0, (lo + hi) / 2 - gaps)


def contact_setting(k_acc, d=D_IMP):
    b = 2 * np.sqrt(k_acc)                      # critical damping of the constraint dynamics
    return dict(solref=f"{-k_acc:.6g} {-b:.6g}", solimp=f"{d} {d} 0.001", noslip=2000)


def squeeze(contact, shape="box", size=(0.10, 0.02, 0.02), yaw_deg=0.0, fixed=True, t_end=1.6):
    """Close the force gripper (70 N per pad) on the object; returns the left pad's contacts as
    (x, z, penetration, normal force) in the pad frame, sorted by x then z."""
    xml = build_xml(0.0, contact=contact, shape=shape, size=size, yaw_deg=yaw_deg, midair=True, mass=1.0)
    if fixed:
        xml = xml.replace('<freejoint name="obj"/>', "")
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    m.opt.gravity[:] = 0.0
    for j in ("left_finger_joint", "right_finger_joint"):
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = 0.04
    gid = m.actuator("grip").id
    while d.time < t_end:
        d.ctrl[gid] = grip_target(d.time, 0.3)
        if "lift" in [m.actuator(i).name for i in range(m.nu)]:
            d.ctrl[m.actuator("lift").id] = d.qpos[m.jnt_qposadr[m.joint("lift").id]]
        mujoco.mj_step(m, d)
    left = m.body("left_finger").id
    obj_b = m.body("obj").id
    f6 = np.zeros(6)
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if {b1, b2} != {left, obj_b}:
            continue
        g = c.geom1 if b1 == left else c.geom2
        mujoco.mj_contactForce(m, d, i, f6)
        r = m.geom_pos[g]                                 # pad point position in the finger frame
        out.append((r[0], r[2] - PAD_CENTRE_Z, -c.dist, f6[0]))
    return np.array(sorted(out))


def grid_xz():
    return np.array([(x, z) for x, z in grid_points()])


def compare(label, meas, gaps, rows, k=K_CELL):
    """meas: (x, z, pen, f) from MuJoCo; gaps: Winkler gaps on the full grid."""
    pts = grid_xz()
    fm = np.zeros(len(pts))
    for x, z, _, f in meas:
        j = int(np.argmin(np.hypot(pts[:, 0] - x, pts[:, 1] - z)))
        fm[j] += f
    # which pad edge a misalignment loads depends on its sign: compare with the better of the two mirror images
    mirror = np.array([int(np.argmin(np.hypot(pts[:, 0] + x, pts[:, 1] - z))) for x, z in pts])
    fw = min((winkler(gaps, k=k), winkler(gaps[mirror], k=k)), key=lambda f: np.sum((fm - f) ** 2))
    W = 1.0 * (P.G + 0.0)
    T = lambda f: P.patch_torque_capacity(pts[f > 1e-6], W / 2, P.MU, f.sum(), weights=f[f > 1e-6]) if (f > 1e-6).any() else 0.0
    dw, dm = 2 * T(fw) / W * 100, 2 * T(fm) / W * 100
    row = dict(case=label, engaged_winkler=int((fw > 0.05).sum()), engaged_mujoco=int((fm > 0.05).sum()),
               force_rms_error_N=float(np.sqrt(np.mean((fm - fw) ** 2))), total_N=float(fm.sum()),
               dstar_winkler_cm=dw, dstar_mujoco_cm=dm,
               sink_max_mm=float(meas[:, 2].max() * 1e3) if len(meas) else 0.0)
    rows.append(row)
    print(f"  {label:22s} engaged {row['engaged_mujoco']:2d} (Winkler {row['engaged_winkler']:2d})  "
          f"force rms err {row['force_rms_error_N']:.2f} N of {row['total_N']:.1f}  d* {dm:.2f} cm (Winkler {dw:.2f})  "
          f"max sink {row['sink_max_mm']:.3f} mm")


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"layer: E {E_RUBBER / 1e6:.1f} MPa, h {H_LAYER * 1e3:.1f} mm, cell {A_CELL * 1e6:.1f} mm^2 -> "
          f"k_c = {K_CELL:.0f} N/m; flat sink at 70 N = {FN / 16 / K_CELL * 1e3:.3f} mm")
    print("current pad (D6):")
    m0 = squeeze(CONTACT)
    print(f"  flat face: {len(m0)} contacts, mean sink {m0[:, 2].mean() * 1e3:.3f} mm, per-point stiffness "
          f"{(m0[:, 3] / np.maximum(m0[:, 2], 1e-12)).mean():.0f} N/m")
    # calibrate: per-point stiffness is linear in k_acc at constant impedance
    # calibrate in the study's condition (a free 1 kg object); per-point stiffness is linear in k_acc
    k_acc = 2.0e4
    for _ in range(5):
        meas = squeeze(contact_setting(k_acc), fixed=False)
        k_meas = np.median(meas[:, 3] / np.maximum(meas[:, 2], 1e-12))
        k_acc *= K_CELL / k_meas
    meas = squeeze(contact_setting(k_acc), fixed=False)
    k_meas = np.median(meas[:, 3] / np.maximum(meas[:, 2], 1e-12))
    print(f"layer pad: solref -{k_acc:.0f} -{2 * np.sqrt(k_acc):.0f}, solimp {D_IMP} {D_IMP}: per-point stiffness "
          f"{k_meas:.0f} N/m on a free 1 kg box (target {K_CELL:.0f}); flat sink {meas[:, 2].mean() * 1e3:.3f} mm")
    meas = squeeze(contact_setting(k_acc))
    k_fixed = float(np.median(meas[:, 3] / np.maximum(meas[:, 2], 1e-12)))
    print(f"  against a fixed object the same setting gives {k_fixed:.0f} N/m (sink {meas[:, 2].mean() * 1e3:.3f} mm); "
          f"the shape tests below use the fixed object and this stiffness")
    rows = []
    pts = grid_xz()
    for name, setting in (("current pad", CONTACT), ("layer pad", contact_setting(k_acc))):
        flat = squeeze(setting)
        k_set = float(np.median(flat[:, 3] / np.maximum(flat[:, 2], 1e-12)))
        print(f"{name} (per-point stiffness here {k_set:.0f} N/m):")
        compare("flat", flat, np.zeros(16), rows, k_set)
        for th in (1.0, 2.0, 4.0):
            compare(f"misaligned {th:.0f} deg", squeeze(setting, yaw_deg=th),
                    np.tan(np.radians(th)) * (pts[:, 0] - pts[:, 0].min()), rows, k_set)
        R = 0.015
        compare("cylinder R 15 mm", squeeze(setting, shape="cylinder", size=(0.10, R)), pts[:, 1] ** 2 / (2 * R),
                rows, k_set)
        for r in rows[-5:]:
            r["pad"] = name
    json.dump(dict(k_cell=K_CELL, k_acc=k_acc, d=D_IMP, setting=contact_setting(k_acc), rows=rows,
                   k_fixed_object=k_fixed), open(os.path.join(OUT, "layer.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

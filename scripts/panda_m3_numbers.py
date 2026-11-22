"""Numbers for the M3 test lift on the Menagerie Panda: CoM accuracy from joint torques at realistic lift poses,
joint motion of a 2 cm lift, and the CoM bias the tare leaves when the link masses are off.

    python scripts/panda_m3_numbers.py
Outputs: results/m3/numbers.json, numbers.tex (LaTeX macros)
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anygrasp_pp.contact_rig import FRANKA, PAD_Z  # noqa: E402

G, M_OBJ = 9.81, 0.6
HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
rng = np.random.default_rng(0)


def load():
    m = mujoco.MjModel.from_xml_path(str(FRANKA / "panda.xml"))
    return m, mujoco.MjData(m)


def grasp_point(m, d):
    h = m.body("hand").id
    return d.xpos[h] + d.xmat[h].reshape(3, 3)[:, 2] * PAD_Z, h


def jac(m, d):
    p, h = grasp_point(m, d)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jp, jr, p, h)
    return np.vstack([jp, jr])[:, :7]


def ik(m, d, p_target, yaw, q0=HOME, iters=500):
    """Damped least squares IK: grasp point at p_target, hand pointing down, rotated by yaw."""
    R_t = Rot.from_euler("z", yaw).as_matrix() @ np.diag([1.0, -1.0, -1.0])
    d.qpos[:7] = q0
    lo, hi = m.jnt_range[:7, 0], m.jnt_range[:7, 1]
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p, h = grasp_point(m, d)
        R = d.xmat[h].reshape(3, 3)
        e = np.concatenate([p_target - p, 0.5 * (np.cross(R[:, 0], R_t[:, 0]) + np.cross(R[:, 1], R_t[:, 1])
                                                  + np.cross(R[:, 2], R_t[:, 2]))])
        if np.linalg.norm(e) < 1e-9:
            break
        J = jac(m, d)
        d.qpos[:7] = np.clip(d.qpos[:7] + J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e), lo, hi)
    mujoco.mj_forward(m, d)
    return d.qpos[:7].copy(), np.linalg.norm(e)


def gravity_torque(m, d, q):
    d.qpos[:7] = q
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    return d.qfrc_bias[:7].copy()


def main():
    out = ROOT / "results" / "m3"
    out.mkdir(parents=True, exist_ok=True)
    m, d = load()
    m_err = load()[0]
    poses = []
    for x in (0.40, 0.50, 0.60):
        for y in (-0.15, 0.0, 0.15):
            for yaw in (0.0, np.pi / 4, np.pi / 2):
                p = np.array([x, y, 0.05])
                q0, e0 = ik(m, d, p, yaw)
                q1, e1 = ik(m, d, p + [0, 0, 0.02], yaw, q0)
                if max(e0, e1) < 1e-6:
                    poses.append((q0, q1))
    print(f"{len(poses)} lift poses solved (grasp point at x 0.4-0.6 m, y +-0.15 m, z 5 cm, yaw 0/45/90 deg)")

    # CoM accuracy: horizontal entries of the angular block of (J J^T)^-1 at the lift pose
    lam, lam_nocouple, covs = [], [], []
    for q0, q1 in poses:
        d.qpos[:7] = q1
        mujoco.mj_forward(m, d)
        J = jac(m, d)
        A = np.linalg.inv(J @ J.T)
        covs.append(A[3:5, 3:5])
        lam.append(np.sqrt(np.trace(A[3:5, 3:5]) / 2))
        lam_nocouple.append(np.sqrt(np.trace(np.linalg.inv(J[3:] @ J[3:].T)[:2, :2]) / 2))
    res = {"n_poses": len(poses), "rms_sqrt_lambda_median": float(np.median(lam)),
           "rms_sqrt_lambda_range": [float(min(lam)), float(max(lam))],
           "rms_sqrt_lambda_nocoupling_median": float(np.median(lam_nocouple)),
           "isotropic_guess": float(np.sqrt(3 / 7))}

    def com_error(sig_tau, N, draws=4000):
        """Median 2-D CoM error [cm] over poses: n ~ N(0, 2 sig^2/N * A_hor), r_perp = (n x e_z)/(m g)."""
        errs = []
        for C in covs:
            n = rng.multivariate_normal(np.zeros(2), 2 * sig_tau**2 / N * C, draws)
            errs.append(np.linalg.norm(n, axis=1) / (M_OBJ * G) * 100)
        return float(np.median(np.concatenate(errs)))

    table = {}
    for sig in (0.1, 0.5, 1.0):
        for hold, N in ((0.3, 150), (1.5, 750)):
            table[f"{sig}_{hold}"] = com_error(sig, N)
    res["median_com_error_cm"] = table
    need = None
    for N in range(150, 5001, 10):
        if com_error(0.5, N, 1500) < 1.0:
            need = N
            break
    res["samples_for_1cm_at_0.5"] = need

    # joint motion of the 2 cm lift, and the tare residual with link masses off by up to 5%
    dq = [np.abs(q1 - q0).max() for q0, q1 in poses]
    res["lift_dq_max_rad_median"] = float(np.median(dq))
    res["lift_dq_max_rad_range"] = [float(min(dq)), float(max(dq))]
    body_mass0 = m.body_mass.copy()
    no_tare, with_tare, com_bias = [], [], []
    for trial in range(200):
        scale = 1.05 if trial == 0 else 1 + rng.uniform(-0.05, 0.05, m.nbody)
        m_err.body_mass[:] = body_mass0 * scale
        for q0, q1 in poses:
            dg0 = gravity_torque(m_err, d, q0) - gravity_torque(m, d, q0)
            dg1 = gravity_torque(m_err, d, q1) - gravity_torque(m, d, q1)
            no_tare.append(np.linalg.norm(dg1))
            bias = dg1 - dg0
            with_tare.append(np.linalg.norm(bias))
            d.qpos[:7] = q1
            mujoco.mj_forward(m, d)
            J = jac(m, d)
            w = np.linalg.solve(J @ J.T, J @ bias)
            com_bias.append(np.linalg.norm(np.cross(w[3:], [0, 0, 1])[:2]) / (M_OBJ * G) * 100)
    res["gravity_error_no_tare_Nm_median"] = float(np.median(no_tare))
    res["gravity_error_tare_Nm_median"] = float(np.median(with_tare))
    res["tare_com_bias_cm_median"] = float(np.median(com_bias))
    res["tare_com_bias_cm_p90"] = float(np.percentile(com_bias, 90))
    json.dump(res, open(out / "numbers.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

    f = lambda x, n=2: f"{x:.{n}f}"
    macros = {
        "mthreeposes": str(len(poses)),
        "mthreelam": f(res["rms_sqrt_lambda_median"]),
        "mthreelamlo": f(res["rms_sqrt_lambda_range"][0]), "mthreelamhi": f(res["rms_sqrt_lambda_range"][1]),
        "mthreelamnc": f(res["rms_sqrt_lambda_nocoupling_median"]),
        "mthreeerrshort": f(table["0.5_0.3"]), "mthreeerrlong": f(table["0.5_1.5"]),
        "mthreeerrlowshort": f(table["0.1_0.3"]), "mthreeerrhighlong": f(table["1.0_1.5"]),
        "mthreeneed": str(need), "mthreeneeds": f(need / 500, 1) if need else "--",
        "mthreedq": f(res["lift_dq_max_rad_median"], 3),
        "mthreedqlo": f(res["lift_dq_max_rad_range"][0], 3), "mthreedqhi": f(res["lift_dq_max_rad_range"][1], 3),
        "mthreenotare": f(res["gravity_error_no_tare_Nm_median"]), "mthreetare": f(res["gravity_error_tare_Nm_median"], 3),
        "mthreebias": f(res["tare_com_bias_cm_median"]), "mthreebiasninety": f(res["tare_com_bias_cm_p90"]),
    }
    with open(out / "numbers.tex", "w") as fh:
        fh.write("% generated by scripts/panda_m3_numbers.py -- do not edit\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"saved {out}/numbers.json, numbers.tex")


if __name__ == "__main__":
    main()

"""Independent checks for docs/derivations.tex. Each check is labelled with its section.

Kinds of check (printed in brackets):
  [sym]  the algebra is redone from scratch in sympy;
  [num]  random inputs, Monte Carlo, or an independent algorithm (e.g. linear programming);
  [mj]   read from the MuJoCo simulator itself;
  [arith] arithmetic only: confirms a number quoted in the text, does not test the derivation.

    python scripts/check_derivations.py
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import sympy as sp
from scipy.optimize import linprog, brentq
from scipy.spatial.transform import Rotation as Rot
from scipy.stats import norm, binom, chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anygrasp_pp import physics as P  # noqa: E402
from anygrasp_pp.contact_rig import (build_xml, grid_points, grip_target, CONTACT, DEFAULT_CONTACT,  # noqa: E402
                                     BOX_HALF, X_HIDDEN, OPEN, FRANKA, PAD_Z)

rng = np.random.default_rng(0)
EZ = np.array([0.0, 0.0, 1.0])
results = []
MU, FN, A, M, G = 0.5, 70.0, 0.006, 0.6, 9.81
W = M * G


def check(label, kind, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {label:5s} [{kind}] {detail}", flush=True)


def close(a, b, rtol=1e-9, atol=1e-12):
    return np.allclose(a, b, rtol=rtol, atol=atol)


def grasp_rig(xml, t_end=1.3):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for j in ("left_finger_joint", "right_finger_joint"):
        d.qpos[m.joint(j).qposadr[0]] = OPEN
    while d.time < t_end:
        if m.actuator_ctrlrange[m.actuator("grip").id][1] > 100:      # stock gripper: ctrl 0 = closed
            d.ctrl[m.actuator("grip").id] = 255.0 if d.time < 0.3 else 0.0
        else:
            d.ctrl[m.actuator("grip").id] = grip_target(d.time, 0.3)
        mujoco.mj_step(m, d)
    return m, d


def pad_contacts(m, d, finger="left_finger"):
    b, o = m.body(finger).id, m.body("obj").id
    return [i for i in range(d.ncon) if {m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]} == {b, o}]


# ============================================================================ A. grasp statics
r, a = rng.normal(size=(1000, 3)), rng.normal(size=(1000, 3))
a /= np.linalg.norm(a, axis=1, keepdims=True)
lhs = np.einsum("ij,ij->i", a, np.cross(r, -W * EZ))
check("A.2", "num", close(lhs, -W * np.einsum("ij,ij->i", r, np.cross(EZ, a))), "a.(r x W) = -W r.(e_z x a), 1000 random cases")
p0 = rng.normal(size=(1000, 3))
check("A.2", "num", close(P.lever_arm(p0, a, p0 + r), np.einsum("ij,ij->i", r, np.cross(EZ, a))), "lever_arm matches")

phi = rng.uniform(-1.5, 1.5, 500)
psi = rng.uniform(0, 2 * np.pi, 500)
ax = np.stack([np.cos(phi) * np.cos(psi), np.cos(phi) * np.sin(psi), np.sin(phi)], 1)
check("A.3", "num", close(np.linalg.norm(P.hinge_vector(ax), axis=1), np.cos(phi)), "|e_z x a_hat| = cos(tilt)")
apr = np.array([[0, 0, -1.0], [np.sin(np.radians(29)), 0, -np.cos(np.radians(29))],
                [np.sin(np.radians(31)), 0, -np.cos(np.radians(31))], [1.0, 0, 0]])
check("A.3", "num", list(P.approach_ok(apr)) == [True, True, False, False], "approach filter: 0/29 deg pass, 31/90 deg fail")

rho, As, mu, fn, th = sp.symbols("rho a mu f_n theta", positive=True)
tau_u = sp.integrate(mu * fn / (sp.pi * As**2) * rho * 2 * sp.pi * rho, (rho, 0, As))
check("A.4", "sym", sp.simplify(tau_u - sp.Rational(2, 3) * mu * fn * As) == 0, f"uniform disc: {tau_u}")
ph = sp.symbols("p_h", positive=True)
ph_sol = sp.solve(sp.Eq(sp.integrate(ph * sp.sqrt(1 - rho**2 / As**2) * 2 * sp.pi * rho, (rho, 0, As)), fn), ph)[0]
import mpmath as mpm
mpm.mp.dps = 30
a_test = mpm.mpf("0.006")
I_num = mpm.quad(lambda rr: rr**2 * mpm.sqrt(1 - rr**2 / a_test**2), [0, a_test])
check("A.4", "sym", sp.simplify(ph_sol - 3 * fn / (2 * sp.pi * As**2)) == 0, f"Hertz normalization: p_h = {ph_sol}")
check("A.4", "num", abs(I_num / (mpm.pi * a_test**3 / 16) - 1) < 1e-20,
      "Hertz torque integral, quadrature in rho (no substitution): int rho^2 sqrt(1 - rho^2/a^2) = pi a^3 / 16 to 1e-20")

tmax = lambda f: 2 / 3 * MU * FN * A * np.sqrt(np.clip(1 - (f / (MU * FN))**2, 0, None))
s_grid = np.linspace(0, 1, 2001)
best = s_grid[np.argmax([tmax(s * W) + tmax((1 - s) * W) for s in s_grid])]
check("A.5", "num", abs(best - 0.5) < 1e-9 and abs((tmax(W / 2) * 2) / W - float(P.dstar())) < 1e-12,
      "total twist capacity over all force splits is largest at the equal split, and equals W d*")
vals = [(2 * tmax(W / 2) if al == 0 else tmax(W / 2 + W * al / 0.04) + tmax(W / 2 - W * al / 0.04)) / W * 100
        for al in (0, 0.005, 0.01, 0.02)]
check("A.5", "arith", np.allclose(vals, [4.7402, 4.7391, 4.7360, 4.7232], atol=5e-5),
      f"d* with CoM offset alpha = 0/0.5/1/2 cm along the axis (w = 4 cm): {np.round(vals, 4)} cm")

check("A.6", "arith", close(P.dstar(exact=False, accel=0.4) / P.dstar(exact=False), G / (G + 0.4)),
      "simple d* scales by g/(g + a); the physical test is R.3 (fast and slow lifts)")

nr, nt = 400, 720
rr_, tt_ = np.meshgrid((np.arange(nr) + 0.5) / nr * A, (np.arange(nt) + 0.5) / nt * 2 * np.pi)
disc = np.c_[(rr_ * np.cos(tt_)).ravel(), (rr_ * np.sin(tt_)).ravel()]
dw = rr_.ravel()
d_exact, d_ell = P.patch_dstar(disc, weights=dw), float(P.dstar())
check("A.7", "num", 0.001 < d_ell / d_exact - 1 < 0.003 and
      abs(P.patch_torque_capacity(disc, 0.0, weights=dw) / (MU * FN * A) - 2 / 3) < 1e-4,
      f"exact disc limit surface d* = {d_exact*100:.4f} cm < ellipse {d_ell*100:.4f} cm (ellipse optimistic by "
      f"{(d_ell/d_exact-1)*100:.2f}%); pure torsion = 2/3 mu fn a")

Wsym, d = sp.symbols("W d", positive=True)
ellipse = (Wsym / 2 / (mu * fn))**2 + ((Wsym * d / 2) / (sp.Rational(2, 3) * mu * fn * As))**2 - 1
target = 4 * mu * fn * As / (3 * Wsym) * sp.sqrt(1 - (Wsym / (2 * mu * fn))**2)
vals_ = {mu: .5, fn: 70, As: .006, Wsym: 5.886}
check("A.8", "sym", any(abs(float((s_ - target).subs(vals_))) < 1e-12 for s_ in sp.solve(ellipse, d)),
      "sympy solution of the ellipse boundary == d* formula")
tab = {(0.3, 0.3): 5.69, (0.3, 0.5): 9.51, (0.3, 0.8): 15.22, (0.6, 0.3): 2.83, (0.6, 0.5): 4.74,
       (0.6, 0.8): 7.60, (1.0, 0.3): 1.67, (1.0, 0.5): 2.83, (1.0, 0.8): 4.55}
check("A.8", "arith", max(abs(float(P.dstar(mu=k[1], m=k[0])) * 100 - v) for k, v in tab.items()) < 0.006,
      "table of d* values")

x = W / (2 * MU * FN)
el = (np.log(float(P.dstar(mu=MU * 1.0001))) - np.log(float(P.dstar(mu=MU * 0.9999)))) / (np.log(1.0001) - np.log(0.9999))
check("A.9", "num", abs(el - 1 / (1 - x**2)) < 1e-6 and close(P.dstar(mu=0.25, fn=140), P.dstar(mu=0.5, fn=70)),
      f"elasticity in mu*fn = {el:.6f} = 1/(1-x^2) = {1/(1-x**2):.6f}; depends on mu and fn only through mu*fn")

f_ = sp.symbols("f", positive=True)
ell_f = (Wsym / 2 / (mu * f_))**2 + ((Wsym * d / 2) / (sp.Rational(2, 3) * mu * f_ * As))**2 - 1
f_target = Wsym / mu * sp.sqrt(sp.Rational(1, 4) + (3 * d / (4 * As))**2)
vv = {mu: .5, As: .006, Wsym: 5.886, d: .04}
check("A.10", "sym", any(abs(float((s_ - f_target).subs(vv))) < 1e-9 for s_ in sp.solve(ell_f, f_)),
      f"fn_min = W/mu sqrt(1/4 + (3d/4a)^2) = {float(f_target.subs(vv)):.2f} N at d = 4 cm")
dd = rng.uniform(0, 0.08, 200)
check("A.10", "num", close(P.dstar(fn=P.required_grip(dd, M), m=M), dd), "round trip d*(fn_min(d)) = d")

ah = np.c_[np.cos(psi[:50]), np.sin(psi[:50]), np.zeros(50)]
hh = np.cross(EZ, ah)
al_, be_ = rng.normal(size=50), rng.normal(size=50)
rh = al_[:, None] * ah + be_[:, None] * hh
check("A.11", "num", close(np.cross(rh, -W * EZ), W * (al_[:, None] * hh - be_[:, None] * ah)),
      "moment of the weight = W (alpha h - beta a_hat) for horizontal axes")
check("A.11", "arith", abs((MU * FN - W / 2) * 0.04 - 1.282) < 1e-3, "tilt capacity (mu fn - W/2) w = 1.282 N m")

# A.12 patch pads
gp = grid_points()
patches = {"full face": gp, "two middle rows": gp[np.abs(gp[:, 1]) < 0.004], "one row": gp[np.isclose(gp[:, 1], gp[1, 1])]}


def lp_capacity(pts, f_up, K=256):
    n = len(pts)
    c = np.zeros(2 * n)
    c[0::2], c[1::2] = pts[:, 1], -pts[:, 0]
    ang = 2 * np.pi * np.arange(K) / K
    A_ub = np.zeros((n * K, 2 * n))
    for k in range(n):
        A_ub[k * K:(k + 1) * K, 2 * k] = np.cos(ang)
        A_ub[k * K:(k + 1) * K, 2 * k + 1] = np.sin(ang)
    A_eq = np.zeros((2, 2 * n))
    A_eq[0, 0::2], A_eq[1, 1::2] = 1, 1
    res = linprog(c, A_ub=A_ub, b_ub=np.full(n * K, MU * FN / n * np.cos(np.pi / K)), A_eq=A_eq, b_eq=[0, f_up],
                  bounds=[(None, None)] * (2 * n), method="highs")
    return -res.fun, 1 - np.cos(np.pi / K)


for name, pts in patches.items():
    K = 256
    Tl, _ = lp_capacity(pts, W / 2, K)
    T_hi = P.patch_torque_capacity(pts, W / 2)
    T_lo = P.patch_torque_capacity(pts, W / 2, mu=MU * np.cos(np.pi / K))
    check("A.12", "num", T_lo - 1e-9 <= Tl <= T_hi + 1e-9,
          f"{name}: formula with mu cos(pi/K) {T_lo:.6f} <= linear program {Tl:.6f} <= formula {T_hi:.6f} N m")


def ic_capacity(pts, f_up):
    """Independent: centre of rotation on z = 0 found by root finding (symmetric patches only)."""
    def forces(cx):
        rel = pts - [cx, 0.0]
        v = np.stack([-rel[:, 1], rel[:, 0]], 1)
        f = -MU * FN / len(pts) * v / np.linalg.norm(v, axis=1, keepdims=True)
        return f[:, 1].sum(), abs((pts[:, 0] * f[:, 1] - pts[:, 1] * f[:, 0]).sum())
    cx = brentq(lambda c: forces(c)[0] - f_up, 1e-9, 100.0, xtol=1e-14)
    return forces(cx)[1]


check("A.12", "num", abs(ic_capacity(gp, W / 2) / P.patch_torque_capacity(gp, W / 2) - 1) < 1e-7,
      "full face: duality formula == centre-of-rotation root finding")
check("A.12", "num", abs(P.patch_torque_capacity(gp, 0.0) - MU * FN * np.linalg.norm(gp, axis=1).mean()) < 1e-9,
      f"pure twist = mu fn x mean distance to the centre ({np.linalg.norm(gp, axis=1).mean()*1000:.2f} mm)")
sq = np.c_[((np.arange(300) + 0.5) / 300 * 2 - 1).repeat(300), np.tile((np.arange(300) + 0.5) / 300 * 2 - 1, 300)] * 0.0085
check("A.12", "num", abs(np.linalg.norm(sq, axis=1).mean() * 1000 - 6.51) < 0.01,
      f"continuous 17 mm square: mean radius {np.linalg.norm(sq, axis=1).mean()*1000:.3f} mm (grid: 6.37 mm, 2% lower)")
m_, d_ = grasp_rig(build_xml(0.0, pads="grid"))
fpts = []
for i in pad_contacts(m_, d_):
    f6 = np.zeros(6)
    mujoco.mj_contactForce(m_, d_, i, f6)
    fpts.append(f6[0])
fpts = np.array(fpts)
check("A.12", "mj", len(fpts) == 16 and fpts.max() / fpts.min() < 1.01,
      f"MuJoCo shares the 70 N equally over the 16 points: {fpts.min():.3f}-{fpts.max():.3f} N")

# ============================================================================ B. simulator model
sx = """<mujoco><option cone="elliptic" impratio="1"/><worldbody>
<geom name="floor" type="plane" size="1 1 .1" friction="0.3 0.001 0.0001" solref="0.02 1" condim="4" {pf}/>
<body pos="0 0 0.05"><freejoint/><geom name="box" type="box" size=".05 .05 .05" mass="1"
 friction="0.8 0.004 0.0001" solref="0.004 1" condim="4" {pb}/></body></worldbody></mujoco>"""
out = {}
for case, pf, pb in (("equal", "", ""), ("box wins", "", 'priority="1"'), ("floor wins", 'priority="1"', "")):
    mm = mujoco.MjModel.from_xml_string(sx.format(pf=pf, pb=pb))
    dm = mujoco.MjData(mm)
    for _ in range(50):
        mujoco.mj_step(mm, dm)
    out[case] = (dm.contact[0].friction[0], dm.contact[0].friction[2], dm.contact[0].solref[0])
check("B.1", "mj", np.allclose(out["equal"], (0.8, 0.004, 0.012)) and np.allclose(out["box wins"], (0.8, 0.004, 0.004))
      and np.allclose(out["floor wins"], (0.3, 0.001, 0.02)),
      f"equal priority: max friction, averaged solref {out['equal']}; priority: one geom supplies all")
mm = mujoco.MjModel.from_xml_string(sx.format(pf="", pb=""))
dm = mujoco.MjData(mm)
for _ in range(200):
    mujoco.mj_step(mm, dm)
dm.qvel[0] = 0.3
ratios = []
for _ in range(40):
    mujoco.mj_step(mm, dm)
    ft = fnn = 0.0
    for i in range(dm.ncon):
        f6 = np.zeros(6)
        mujoco.mj_contactForce(mm, dm, i, f6)
        fnn += f6[0]
        ft += np.hypot(f6[1], f6[2])
    if fnn > 1.0 and abs(dm.qvel[0]) > 0.01:          # still in contact and still sliding
        ratios.append(ft / fnn)
check("B.1", "mj", len(ratios) > 5 and abs(np.median(ratios) - 0.8) < 0.01,
      f"a sliding box sits on the cone boundary: |f_t|/f_n = {np.median(ratios):.4f} = mu (0.8)")
check("B.1", "mj", np.isclose(d_.contact[pad_contacts(m_, d_)[0]].friction[0], 0.5) and
      np.isclose(d_.contact[pad_contacts(m_, d_)[0]].solref[0], float(CONTACT['solref'].split()[0])), "rig: pad contacts use the object's mu and solref")

for dshift in (0.0, 0.02, 0.047, 0.08):
    mm = mujoco.MjModel.from_xml_string(build_xml(dshift))
    dm = mujoco.MjData(mm)
    mujoco.mj_forward(mm, dm)
    b = mm.body("obj").id
    assert abs((dm.subtree_com[b] - dm.xpos[b])[0] - dshift) < 1e-9 and abs(mm.body_subtreemass[b] - 0.6) < 1e-9
check("B.2", "mj", True, "subtree CoM offset = d and mass = 0.6 kg for d = 0, 2, 4.7, 8 cm")

M_ = np.zeros((m_.nv, m_.nv))
mujoco.mj_fullM(m_, d_, M_)
Jt = np.zeros(m_.nv)
for j in ("left_finger_joint", "right_finger_joint"):
    Jt[m_.joint(j).dofadr[0]] = 0.5
check("B.3", "mj", abs(1 / (Jt @ np.linalg.solve(M_, Jt)) - 0.23) < 1e-9, "tendon-space finger inertia = 0.23 kg")
ms, ds = grasp_rig(build_xml(0.0, grip="stock", pads="stock", menagerie_friction=True, contact=DEFAULT_CONTACT))
stock = -ds.actuator_force[ms.actuator("grip").id] / 2
check("B.3", "mj", abs(stock - 1.0) < 0.05, f"stock Menagerie grip, commanded closed on a 4 cm box: {stock:.3f} N per finger")
fpad = sum(mujoco.mj_contactForce(m_, d_, i, f6 := np.zeros(6)) or f6[0] for i in pad_contacts(m_, d_))
check("B.3", "mj", abs(fpad - 70.0) < 0.05, f"force grip: {fpad:.2f} N per pad")


import anygrasp_pp.contact_rig as CR
OLD_GRIP = ('<general name="grip" tendon="split" ctrlrange="-140 140" forcerange="-140 140" '
            'gainprm="1 0 0" biasprm="0 0 -1400" dyntype="none" biastype="affine"/>')


def chatter(old):
    """Real rig, object far away, fingers closing on air for 0.1 s: count sign flips of the grip force."""
    orig = CR._grip_actuator
    if old:
        CR._grip_actuator = lambda grip, fn: OLD_GRIP
    try:
        mm = mujoco.MjModel.from_xml_string(CR.build_xml(0.0, offset=(0.6, 0, 0)))
    finally:
        CR._grip_actuator = orig
    dm = mujoco.MjData(mm)
    for j in ("left_finger_joint", "right_finger_joint"):
        dm.qpos[mm.joint(j).qposadr[0]] = OPEN
    fs = []
    while dm.time < 0.4:
        t = dm.time
        dm.ctrl[mm.actuator("grip").id] = (20.0 if t < 0.3 else -140.0) if old else grip_target(t, 0.3)
        mujoco.mj_step(mm, dm)
        if t >= 0.3:
            fs.append(dm.actuator_force[mm.actuator("grip").id])
    return int(np.sum(np.diff(np.sign(fs)) != 0)), len(fs)


(old, n_old), (new, n_new) = chatter(True), chatter(False)
check("B.4", "mj", old > 0.8 * n_old and new == 0,
      f"rig, implicitfast, fingers closing on air: old controller flips force sign {old}x in {n_old} steps; new one {new}x")
check("B.4", "arith", 1400 * 0.002 / 0.23 >= 2 > 160.5 * 0.002 / 0.23,
      f"clamp-to-clamp cycle condition kv dt/m >= 2: old {1400*0.002/0.23:.2f}, new {160.5*0.002/0.23:.2f}")


def imp(r, solimp):
    d0, dw, width, mid, pw = solimp[:5]
    xx = min(abs(r) / width, 1.0)
    y = (xx**pw / mid**(pw - 1)) if xx <= mid else (1 - (1 - xx)**pw / (1 - mid)**(pw - 1))
    return d0 + y * (dw - d0)


for name, contact in (("stiff (CONTACT)", CONTACT), ("MuJoCo default", DEFAULT_CONTACT)):
    mm, dm = grasp_rig(build_xml(0.0, pads="point", contact=contact))
    i = pad_contacts(mm, dm)[0]
    c = dm.contact[i]
    row = c.efc_address
    f, Ahat, depth = dm.efc_force[row], dm.efc_diagA[row], -dm.efc_pos[row]
    tc, zeta = c.solref
    dmax = c.solimp[1]
    pred = brentq(lambda rr: rr - (1 - imp(rr, c.solimp)) * Ahat * f * dmax**2 * tc**2 * zeta**2 / imp(rr, c.solimp)**2,
                  0.0, 0.05)
    check("B.5", "mj", abs(pred / depth - 1) < 0.02,
          f"{name}: depth formula {pred*1000:.4f} mm vs MuJoCo {depth*1000:.4f} mm (f = {f:.1f} N, A_hat = {Ahat:.3f})")

# ============================================================================ C. scoring
L, xs = sp.symbols("L", positive=True), sp.symbols("x", real=True)
check("C.1", "sym", sp.simplify(sp.integrate(xs**2 / L, (xs, -L / 2, L / 2)) - L**2 / 12) == 0, "Var(U[-L/2, L/2]) = L^2/12")
Q = Rot.random(random_state=1).as_matrix()
cov = Q @ np.diag([0.03, 0.01, 0.02])**2 @ Q.T
c = np.array([0.03, -0.01, 0.1])
pg, axg = rng.normal(scale=0.02, size=(5, 3)), rng.normal(size=(5, 3))
samples = rng.multivariate_normal(c, cov, 400_000)
h = P.hinge_vector(axg)
d_mc = np.einsum("sij,ij->si", (samples[:, None, :] - pg[None]), h)
check("C.2", "num", close(d_mc.var(0), np.einsum("ij,jk,ik->i", h, cov, h), rtol=0.01), "Var(d) = h^T Sigma h")
p_mc = (np.abs(d_mc) <= float(P.dstar())).mean(0)
check("C.3", "num", np.abs(p_mc - P.p_hold(pg, axg, c, cov, m=M)).max() < 0.003, "Gaussian P_hold vs Monte Carlo")
ds0 = P.dstar(m=7.5)
check("C.4", "num", P.margin(np.zeros((1, 3)), [0, 1, 0], [0, 0, 0], m=7.5)[0] == -np.inf and float(ds0) == 0,
      "margin is -inf when nothing can hold (m g >= 2 mu fn)")
al, be = 0.03, 0.012
ts = np.linspace(-0.05, 0.05, 41)
mc = rng.uniform(-al, al, 500_000) + rng.uniform(-be, be, 500_000)
check("C.5", "num", np.abs(P._uniform_sum_cdf(ts, al, be) - (mc[:, None] <= ts).mean(0)).max() < 0.003,
      "trapezoid CDF of a sum of two uniforms vs Monte Carlo")
V = np.c_[[1.0, 0, 0], [0, 1.0, 0]]
for Lh, off in ((0.10, 0.0), (0.05, 0.0), (0.10, 0.08)):
    cs = np.c_[rng.uniform(-Lh, Lh, 400_000) + off, rng.uniform(-0.02, 0.02, 400_000), np.zeros(400_000)]
    pmc = (np.abs(cs @ np.cross(EZ, [0, 1.0, 0])) <= float(P.dstar())).mean()
    pu = P.p_hold_uniform(np.zeros((1, 3)), [0, 1, 0], [off, 0, 0], V, (Lh, 0.02), m=M)[0]
    pgau = P.p_hold(np.zeros((1, 3)), [0, 1, 0], [off, 0, 0], np.diag([(2 * Lh)**2 / 12, 0.04**2 / 12, 0]), m=M)[0]
    check("C.5", "num", abs(pu - pmc) < 0.003,
          f"footprint {2*Lh*100:.0f} cm, offset {off*100:.0f} cm: exact {pu:.3f} = MC {pmc:.3f}; Gaussian would give {pgau:.3f}")

# ============================================================================ D. motion
t = sp.symbols("tau")
cs_ = sp.symbols("c0:6")
s = sum(cs_[i] * t**i for i in range(6))
sol = sp.solve([s.subs(t, 0), sp.diff(s, t).subs(t, 0), sp.diff(s, t, 2).subs(t, 0),
                s.subs(t, 1) - 1, sp.diff(s, t).subs(t, 1), sp.diff(s, t, 2).subs(t, 1)], cs_)
check("D.1", "sym", [sol[ci] for ci in cs_] == [0, 0, 0, 10, -15, 6], "boundary conditions give 10, -15, 6")
s_mj = s.subs(sol)
tau_star = [r_ for r_ in sp.solve(sp.diff(s_mj, t, 3), t) if 0 < r_ < sp.Rational(1, 2)][0]
peak = sp.simplify(sp.diff(s_mj, t, 2).subs(t, tau_star))
check("D.1", "sym", sp.simplify(peak - 10 * sp.sqrt(3) / 3) == 0, f"peak s'' = {peak} at tau = {tau_star}")
tt = np.linspace(0, 1, 4001)
jerk0 = np.trapezoid(np.gradient(np.gradient(np.gradient(10*tt**3 - 15*tt**4 + 6*tt**5, tt), tt), tt)**2, tt)
worse = 0
for _ in range(50):
    bump = tt**3 * (1 - tt)**3 * np.polyval(rng.normal(size=3), tt) * rng.uniform(1, 30)
    sv = 10*tt**3 - 15*tt**4 + 6*tt**5 + bump
    worse += np.trapezoid(np.gradient(np.gradient(np.gradient(sv, tt), tt), tt)**2, tt) > jerk0
check("D.1", "num", worse == 50, "50 random admissible perturbations all have more squared jerk (optimality)")
J = rng.normal(size=(6, 7))
e = rng.normal(size=6)
lam = 0.05
dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), e)
dq_ls = np.linalg.lstsq(np.vstack([J, lam * np.eye(7)]), np.concatenate([e, np.zeros(7)]), rcond=None)[0]
check("D.2", "num", close(dq, dq_ls, rtol=1e-8), "DLS step = argmin |J dq - e|^2 + lam^2 |dq|^2")
check("D.2", "num", close(np.sqrt(np.linalg.det(J @ J.T)), np.prod(np.linalg.svd(J, compute_uv=False))), "manipulability")
Rs = Rot.random(200, random_state=2)
check("D.3", "num", close(np.arccos(np.clip((np.trace(Rs.as_matrix(), axis1=1, axis2=2) - 1) / 2, -1, 1)), Rs.magnitude(), rtol=1e-7),
      "rotation angle = arccos((tr R - 1)/2)")

# ============================================================================ E. test lift
pm = mujoco.MjModel.from_xml_path(str(FRANKA / "panda.xml"))
pdd = mujoco.MjData(pm)
pdd.qpos[:7] = [0.3, -0.4, 0.2, -2.2, 0.1, 1.9, 0.6]
mujoco.mj_forward(pm, pdd)
hid = pm.body("hand").id
pt = pdd.xpos[hid] + pdd.xmat[hid].reshape(3, 3)[:, 2] * PAD_Z
jp, jr = np.zeros((3, pm.nv)), np.zeros((3, pm.nv))
mujoco.mj_jac(pm, pdd, jp, jr, pt, hid)
Jp = np.vstack([jp, jr])[:, :7]
fw = np.array([0.3, -0.2, -5.886])
nw = np.cross(np.array([0.04, 0.01, 0.0]), fw)
qf = np.zeros(pm.nv)
mujoco.mj_applyFT(pm, pdd, fw, nw, pt, hid, qf)
check("E.1", "mj", close(qf[:7], Jp.T @ np.r_[fw, nw], rtol=1e-9, atol=1e-12),
      "MuJoCo's wrench-to-joint mapping (mj_applyFT) of force f at the grasp point plus moment n equals J^T [f; n]")
nums = json.load(open(ROOT / "results" / "m3" / "numbers.json"))
check("E.2", "num", nums["gravity_error_tare_Nm_median"] < 0.1 * nums["gravity_error_no_tare_Nm_median"],
      f"Panda, +-5% link masses: model error {nums['gravity_error_no_tare_Nm_median']:.3f} N m without tare, "
      f"{nums['gravity_error_tare_Nm_median']:.4f} with; 2 cm lift moves joints {nums['lift_dq_max_rad_median']:.3f} rad (median)")
J = rng.normal(size=(6, 7))
w_true = rng.normal(size=6)
sig_t, Nn = 0.5, 150
R0 = rng.normal(scale=sig_t, size=(20000, Nn, 7)).mean(1)
R1 = rng.normal(scale=sig_t, size=(20000, Nn, 7)).mean(1)
Y = (J.T @ w_true)[None] + R1 - R0
W_hat = np.linalg.solve(J @ J.T, J @ Y.T).T
C_pred = 2 * sig_t**2 / Nn * np.linalg.inv(J @ J.T)
relerr = np.linalg.norm(np.cov(W_hat.T) - C_pred) / np.linalg.norm(C_pred)
check("E.3", "num", np.abs(W_hat.mean(0) - w_true).max() < 0.01 and relerr < 0.03,
      f"per-sample noise, two holds averaged and differenced: unbiased, covariance off by {relerr*100:.2f}%")
rr, ff = rng.normal(size=(500, 3)), rng.normal(size=(500, 3))
nn = np.cross(rr, ff)
r_perp = rr - ff * (np.einsum("ij,ij->i", ff, rr) / np.einsum("ij,ij->i", ff, ff))[:, None]
check("E.4", "num", close(np.cross(ff, nn) / np.einsum("ij,ij->i", ff, ff)[:, None], r_perp), "(f x n)/|f|^2 = r_perp")
check("E.4", "num", close(np.cross(rr + 0.37 * ff, ff), nn) and close(np.einsum("ij,ij->i", nn, ff), 0),
      "height unobservable; n . f = 0")
r_h = np.c_[rng.normal(size=(500, 2)), np.zeros(500)]
n_v = np.cross(r_h, -W * EZ)
check("E.4", "num", close(np.cross(n_v, EZ) / W, r_h) and not close(np.cross(EZ, n_v) / W, r_h),
      "vertical gravity: r_perp = (n x e_z)/(m g); the old (e_z x n)/(m g) has the wrong sign")
A6 = np.linalg.inv(Jp @ Jp.T)[3:, 3:]
A_nc = np.linalg.inv(Jp[3:] @ Jp[3:].T)
check("E.5", "num", np.linalg.eigvalsh(A6 - A_nc).min() > -1e-10,
      "Panda: the coupled angular block of (J J^T)^-1 >= the uncoupled one (Schur complement)")
check("E.5", "num", nums["rms_sqrt_lambda_median"] > 2 * nums["isotropic_guess"],
      f"Panda lift poses: sqrt(lambda) median {nums['rms_sqrt_lambda_median']:.2f} vs isotropic guess "
      f"{nums['isotropic_guess']:.2f}; median CoM error at 0.5 N m, 0.3 s: {nums['median_com_error_cm']['0.5_0.3']:.2f} cm")
dl, tsy, Kp = sp.symbols("delta tau_s K_p", positive=True)
v_q = sp.integrate(xs**2 / dl, (xs, -dl / 2, dl / 2))
v_s = sp.integrate(xs**2 / (2 * tsy), (xs, -tsy, tsy))
check("E.6", "sym", sp.simplify(v_q - dl**2 / 12) == 0 and sp.solve(sp.Eq(Kp**2 * v_q, v_s), Kp) == [2 * tsy / dl],
      f"variances {v_q}, {v_s}; crossover K_p = 2 tau_s / delta")

# ============================================================================ F. belief update
B = rng.normal(size=(2, 2))
S0 = B @ B.T + 0.1 * np.eye(2)
B = rng.normal(size=(2, 2))
Rn = 0.1 * (B @ B.T) + 0.01 * np.eye(2)
c0, zz = rng.normal(size=2), rng.normal(size=2)
K = S0 @ np.linalg.inv(S0 + Rn)
S_i = np.linalg.inv(np.linalg.inv(S0) + np.linalg.inv(Rn))
check("F.1", "num", close(c0 + K @ (zz - c0), S_i @ (np.linalg.solve(S0, c0) + np.linalg.solve(Rn, zz))) and
      close((np.eye(2) - K) @ S0, S_i), "gain form == information form")
ds_, z = float(P.dstar()), norm.ppf(0.9)
agree = {}
for ratio in (1.5, 3, 5, 10, 24):
    sig = ds_ / ratio
    mus = np.linspace(0, ds_, 4001)
    phold = norm.cdf((ds_ - mus) / sig) - norm.cdf((-ds_ - mus) / sig)
    agree[ratio] = np.mean((phold >= 0.9) == (1 - mus / ds_ >= z * sig / ds_))
check("F.2", "num", agree[24] > 0.999 and agree[5] > 0.999 and agree[1.5] < 0.99,
      "P_hold >= 0.9 <=> rho >= z sigma/d*: agreement " + ", ".join(f"{k}: {v*100:.1f}%" for k, v in agree.items())
      + " (d*/sigma)")

# ============================================================================ G. statistics
check("G.1", "arith", abs(1.96 * np.sqrt(0.5 / 120) - 0.1265) < 1e-4 and
      abs(1.96 * np.sqrt(0.5 * (1 + 5 * 0.2) / 120) - 0.179) < 1e-3 and abs(1.96 * np.sqrt(0.5 * (1 + 5 * 0.5) / 120) - 0.237) < 1e-3,
      "worst-case half-width n = 120: 12.7 pts independent; 17.9 / 23.7 pts with 6 trials/object at rho = 0.2 / 0.5")
nobj, k, rho_icc, reps = 20, 6, 0.3, 400
cover_naive = cover_boot = 0
for _ in range(reps):
    u = rng.normal(size=nobj) * np.sqrt(rho_icc)
    y = (u[:, None] + rng.normal(size=(nobj, k)) * np.sqrt(1 - rho_icc) > 0).astype(float)
    pbar = y.mean()
    hw = 1.96 * np.sqrt(pbar * (1 - pbar) / (nobj * k))
    cover_naive += abs(pbar - 0.5) <= hw
    idx = rng.integers(0, nobj, size=(1000, nobj))
    boots = y.mean(1)[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    cover_boot += lo <= 0.5 <= hi
check("G.1", "num", cover_naive / reps < 0.90 and cover_boot / reps > 0.90,
      f"clustered data (20 objects x 6, ICC 0.3): naive 95% interval covers {cover_naive/reps*100:.0f}%, "
      f"cluster bootstrap {cover_boot/reps*100:.0f}%")
pb, pc = 0.06, 0.04
check("G.2", "num", close(pb + pc - (pb - pc)**2, (np.array([1, 1, 0]) * [pb, pc, 1 - pb - pc]).sum() - (pb - pc)**2),
      "Var(D) = pi_b + pi_c - (pi_b - pi_c)^2")
bs, ns_ = [], []
for _ in range(20000):
    cat = rng.choice(3, size=200, p=[0.05, 0.05, 0.9])
    b_, c_ = (cat == 0).sum(), (cat == 1).sum()
    if b_ + c_ == 20:
        bs.append(b_)
bs = np.array(bs)
check("G.2", "num", abs(bs.mean() - 10) < 0.15 and abs(bs.var() - 5) < 0.5,
      f"under H0, b given b + c = 20 has mean {bs.mean():.2f} (10) and variance {bs.var():.2f} (5): Bin(20, 1/2)")
check("G.2", "num", abs(min(1, 2 * binom.cdf(5, 20, 0.5)) - chi2.sf((abs(15 - 5) - 1)**2 / 20, 1)) < 0.01,
      "exact and continuity-corrected chi^2 McNemar agree (b = 15, c = 5)")
k_, icc, ncl = 15, 0.3, 20
Ys = [(rng.normal(size=(ncl, 1)) * np.sqrt(icc) + rng.normal(size=(ncl, k_)) * np.sqrt(1 - icc)).mean() for _ in range(20000)]
check("G.3", "num", abs(np.var(Ys) / ((1 + (k_ - 1) * icc) / (ncl * k_)) - 1) < 0.04, "design effect 1 + (k-1) rho")
dl_ = rng.uniform(0, 0.08, 300)
yl = (rng.uniform(size=300) < 1 / (1 + np.exp(150 * (dl_ - 0.047)))).astype(float)
ll = lambda kk, d50, y=yl, x=dl_: np.sum(y * -np.logaddexp(0, kk * (x - d50)) + (1 - y) * -np.logaddexp(0, -kk * (x - d50)))
k0, d0 = 120.0, 0.05
pp = 1 / (1 + np.exp(k0 * (dl_ - d0)))
g_an = np.array([np.sum((yl - pp) * (d0 - dl_)), k0 * np.sum(yl - pp)])
g_fd = np.array([(ll(k0 + 1e-5, d0) - ll(k0 - 1e-5, d0)) / 2e-5, (ll(k0, d0 + 1e-8) - ll(k0, d0 - 1e-8)) / 2e-8])
check("G.4", "num", close(g_an, g_fd, rtol=1e-4), "log-likelihood gradient (analytic vs finite difference)")
ysep = (dl_ < 0.047).astype(float)
lls = [ll(kk, 0.047, ysep) for kk in (1e2, 1e3, 1e4, 1e5)]
check("G.4", "num", all(np.diff(lls) > 0), "perfectly separated data: likelihood keeps rising with k, so no finite MLE")

# ---------------------------------------------------------------- H. hidden-weight objects
from anygrasp_pp import objects as O  # noqa: E402

hx, hy, hz = 0.12, 0.025, 0.02                       # half extents of a test box [m], long axis x
box = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
yaw = Rot.from_euler("xyz", [0.3, -0.2, 0.7]).as_matrix()           # tilted, so the axis must be found
off = np.array([0.05, -0.02, 0.1])
occ, org = O.occupancy([box @ yaw.T + off])
pts = org + O.VOXEL * np.argwhere(occ)
c_, ax_ = O.principal_frame(pts)
check("H.1", "num", np.linalg.norm(c_ - off) < 1e-3 and abs(abs(ax_[0] @ yaw[:, 0]) - 1) < 1e-4,
      f"box solid: centroid off by {np.linalg.norm(c_ - off) * 1e3:.2f} mm, long axis within "
      f"{np.degrees(np.arccos(min(1, abs(ax_[0] @ yaw[:, 0])))):.3f} deg")

Ms, ss, VFs, VSs, xFs, xSs, mFs = sp.symbols("M s V_F V_S xbar_F xbar_S m_F", real=True)
sol = sp.solve([sp.Eq(VFs * xFs + VSs * xSs, 0), sp.Eq(mFs * xFs + (Ms - mFs) * xSs, Ms * ss)], [mFs, xSs], dict=True)[0]
check("H.2", "sym", sp.simplify(sol[mFs] - Ms * (ss * VSs / xFs + VFs) / (VFs + VSs)) == 0,
      "m_F = M (s V_S / xbar_F + V_F) / V from the CoM and centroid conditions")

ob = {"points": pts, "centroid": c_, "axes": ax_}
x_ = (pts - c_) @ ax_[0]
idx_ = np.argwhere(occ)
from scipy import ndimage as _nd  # noqa: E402
er_ = _nd.binary_erosion(occ, iterations=1)[tuple(idx_.T)]
core_ = O.choose_core(x_, er_)
errs, dens_ok = [], True
for s_ in O.SHIFTS:
    p_ = O.mass_properties(pts, core_, s_)
    errs.append(abs((p_["com"] - c_) @ (core_["sign"] * ax_[0]) - s_))
    dens_ok &= p_["core_density"] <= O.RHO_MAX + 1e-6 and O.MASS - p_["core_mass"] >= O.MIN_SHELL - 1e-9
p0 = O.mass_properties(pts, core_, 0.0)
check("H.2", "num", core_ is not None and max(errs) < 1e-12 and dens_ok
      and abs(p0["core_density"] - p0["shell_density"]) < 1e-9 * p0["shell_density"],
      f"24 cm box: CoM at s for every shift (max error {max(errs):.1e} m), core <= tungsten, shell >= "
      f"{O.MIN_SHELL * 1e3:.0f} g, uniform at s = 0 (core cap {core_['VF'] * 1e6:.1f} cm^3)")

h_ = O.VOXEL
nx, ny, nz = 60, 12, 10                              # a box of whole cubes, 12 x 2.4 x 2 cm
g_ = np.stack(np.meshgrid(*[(np.arange(n) + 0.5) * h_ for n in (nx, ny, nz)], indexing="ij"), -1).reshape(-1, 3)
fake = {"F": np.zeros(len(g_), bool), "VF": 1e-30, "VS": len(g_) * h_ ** 3, "xF": 1.0}   # no core: uniform box
pI = O.mass_properties(g_, fake, 0.0)
a_, b_, c3 = nx * h_, ny * h_, nz * h_
I_an = O.MASS / 12 * np.diag([b_ ** 2 + c3 ** 2, a_ ** 2 + c3 ** 2, a_ ** 2 + b_ ** 2])
check("H.3", "num", close(pI["inertia"], I_an, rtol=1e-9, atol=1e-15),
      "inertia of a box made of whole cubes equals M(b^2+c^2)/12 etc. exactly (cube self-term h^2/6 included)")
pB = O.mass_properties(pts, core_, 0.0)
I_box = O.MASS / 3 * np.diag([hy ** 2 + hz ** 2, hx ** 2 + hz ** 2, hx ** 2 + hy ** 2])
I_box = yaw @ I_box @ yaw.T
check("H.3", "num", np.abs(pB["inertia"] - I_box).max() < 0.02 * np.abs(I_box).max(),
      f"tilted box through the full voxel pipeline: inertia within "
      f"{np.abs(pB['inertia'] - I_box).max() / np.abs(I_box).max() * 100:.2f}% of the closed form")

# ---------------------------------------------------------------- I. camera and detector frames
# depth tolerance 0.3 mm: far below RealSense noise (~1 mm) and the detector's 5 mm voxels; the GPU (EGL)
# depth buffer reads ~0.1 mm differently from macOS
from anygrasp_pp import scene as SC  # noqa: E402

bx, by, bz, cx_, cy_ = 0.06, 0.04, 0.05, 0.52, -0.03          # box half sizes and centre (table at z = 0)
xml_cam = SC.scene_xml([]).replace(
    "</worldbody>", f'<geom name="probe" type="box" size="{bx} {by} {bz}" pos="{cx_} {cy_} {bz}" rgba="1 0 0 1"/>'
                    "</worldbody>")
mc = mujoco.MjModel.from_xml_string(xml_cam)
dc = mujoco.MjData(mc)
mujoco.mj_forward(mc, dc)
optc = mujoco.MjvOption()
with mujoco.Renderer(mc, SC.CAM_H, SC.CAM_W) as rr:
    rr.enable_depth_rendering()
    rr.update_scene(dc, camera="overhead", scene_option=optc)
    dep = rr.render().copy()
cid = mc.camera("overhead").id
xm = dc.cam_xmat[cid].reshape(3, 3)
Rwc = SC.R_world_cv(xm[:, 0], xm[:, 1])
pw = SC.backproject(dep, SC.camera_K()) @ Rwc.T + dc.cam_xpos[cid]
top = pw[(np.abs(pw[:, 0] - cx_) < bx - 0.005) & (np.abs(pw[:, 1] - cy_) < by - 0.005) & (pw[:, 2] > bz / 2)]
onbox = pw[pw[:, 2] > 2 * bz - 0.001]
ext = np.array([onbox[:, 0].min(), onbox[:, 0].max(), onbox[:, 1].min(), onbox[:, 1].max()])
want = np.array([cx_ - bx, cx_ + bx, cy_ - by, cy_ + by])
px = 0.7 / SC.camera_K()[0, 0]                                 # one pixel at 0.7 m, in metres
check("I.1", "mj", len(top) > 1000 and np.abs(top[:, 2] - 2 * bz).max() < 3e-4 and np.abs(ext - want).max() < 1.5 * px,
      f"rendered box: top face at {top[:, 2].mean() * 100:.3f} cm (want {2 * bz * 100:.1f}), edges within "
      f"{np.abs(ext - want).max() * 1e3:.2f} mm (one pixel = {px * 1e3:.2f} mm)")
check("I.1", "num", close(Rwc.T @ Rwc, np.eye(3), atol=1e-9) and abs(np.linalg.det(Rwc) - 1) < 1e-9,
      "R_wc is a proper rotation")

Rg = Rot.random(200, random_state=1).as_matrix()
tg = rng.normal(size=(200, 3))
gc = np.zeros((200, 17))
gc[:, 4:13] = (Rwc.T @ Rg).reshape(200, 9)
gc[:, 13:16] = (tg - dc.cam_xpos[cid]) @ Rwc
Rb, tb = SC.grasps_to_world(gc, Rwc, dc.cam_xpos[cid])
ang = np.degrees(np.arccos(np.clip(-Rg[:, 2, 0], -1, 1)))
check("I.2", "num", close(Rb, Rg, atol=1e-12) and close(tb, tg, atol=1e-12)
      and np.array_equal(SC.approach_ok_world(Rb, 30.0), ang <= 30.0),
      "world -> camera -> world round trip; approach test equals the angle to straight down <= 30 deg")

sq = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0], [0.05, 0.05, 0.08]])
m_in, m_edge, m_out = (SC._support_margin(sq, np.array(q)) for q in ([0.03, 0.05, 0.2], [0.1, 0.05, 0.2],
                                                                    [0.13, 0.05, 0.2]))
check("I.3", "num", abs(m_in - 0.03) < 1e-12 and abs(m_edge) < 1e-12 and abs(m_out + 0.03) < 1e-12,
      "support margin of a 10 cm square: 3 cm inside, 0 on the edge, -3 cm outside")

Kc = SC.camera_K()
fb_ = Kc[0, 0] * SC.STEREO_BASELINE
sig_ok, sig_txt = True, []
for zz in (0.40, 0.65, 1.00):
    plane = np.full((200, 400), zz)
    zn = SC.realsense_depth(plane, plane, Kc, np.random.default_rng(3), unit=1e-9, edge_drop=0.0)
    vals = zn[zn > 0].astype(np.float64)
    pred = zz ** 2 * SC.SUBPIXEL_RMS / fb_
    sig_ok &= abs(vals.std() / pred - 1) < 0.03
    sig_txt.append(f"{zz:.2f} m: {vals.std() * 1e3:.3f}/{pred * 1e3:.3f} mm")
check("I.4", "num", sig_ok, "depth noise sigma_z = z^2 sigma_d/(f b), Monte Carlo/formula: " + "; ".join(sig_txt))
plane = np.full((50, 400), 0.65)
band = (SC.stereo_shadow(plane, plane, Kc) .all(0)).sum()
check("I.4", "num", abs(band - fb_ / 0.65) <= 1, f"invalid band at the left edge: {band} columns (f b / z = {fb_ / 0.65:.1f})")

ztop_h = 0.10
xml_st = SC.scene_xml([], cam_pos=SC.camera_position(0.0)).replace(
    "</worldbody>", f'<geom type="box" size="0.05 0.08 {ztop_h / 2}" pos="{SC.PICK[0]} 0 {ztop_h / 2}"/></worldbody>')
ms = mujoco.MjModel.from_xml_string(xml_st)
ds = mujoco.MjData(ms)
mujoco.mj_forward(ms, ds)
with mujoco.Renderer(ms, SC.CAM_H, SC.CAM_W) as rr:
    rr.enable_depth_rendering()
    rr.update_scene(ds, camera="overhead", scene_option=optc)
    dl = rr.render().copy()
    rr.update_scene(ds, camera="overhead_right", scene_option=optc)
    dr = rr.render().copy()
sh = SC.stereo_shadow(dl, dr, Kc)
rows = np.arange(SC.CAM_H // 2 - 40, SC.CAM_H // 2 + 40)                 # rows through the box
band0 = int(np.ceil(fb_ / dl.max())) + 2
width = sh[rows][:, band0:].sum(1)
z1, z2 = SC.CAM_DIST - ztop_h, SC.CAM_DIST
want_w = fb_ * (1 / z1 - 1 / z2)
box_rows = np.abs(np.arange(SC.CAM_H) - Kc[1, 2]) <= Kc[1, 1] * 0.05 / z1 + 2      # image rows the box covers
stray = int(sh[~box_rows][:, band0:].sum())
check("I.4", "mj", np.abs(width - want_w).max() <= 1.5 and stray == 0,
      f"shadow strip behind a 10 cm box: {width.min()}-{width.max()} px wide (f b (1/z1 - 1/z2) = {want_w:.1f} px); "
      f"{stray} false holes on the open table")

# ---------------------------------------------------------------- J. executing a grasp with the arm
from anygrasp_pp import robot as RB  # noqa: E402

mr = mujoco.MjModel.from_xml_string(SC.scene_xml([], robot=RB.panda_parts()))
dr_ = mujoco.MjData(mr)
arm_ = RB.Panda(mr)
arm_.set_q(dr_, RB.Q_READY)
mujoco.mj_forward(mr, dr_)
while dr_.time < 1.0:
    arm_.control(dr_, RB.Q_READY, np.zeros(7))
    mujoco.mj_step(mr, dr_)
e_hold = np.abs(dr_.qpos[arm_.qadr] - RB.Q_READY).max()
F = np.array([0.0, 0.0, -10.0])                                      # a known 10 N load on the hand
dr_.xfrc_applied[arm_.hand, :3] = F
while dr_.time < 4.0:
    arm_.control(dr_, RB.Q_READY, np.zeros(7))
    mujoco.mj_step(mr, dr_)
Jp_ = np.zeros((3, mr.nv))
mujoco.mj_jac(mr, dr_, Jp_, None, dr_.xipos[arm_.hand], arm_.hand)
e_pred = np.linalg.solve(np.diag(RB.KP), Jp_[:, arm_.dadr].T @ F)  # q - q_d = Kp^-1 J^T F
e_meas = dr_.qpos[arm_.qadr] - RB.Q_READY
dr_.xfrc_applied[:] = 0
check("J.1", "mj", e_hold < 1e-9 and np.abs(e_meas - e_pred).max() < 0.03 * np.abs(e_pred).max(),
      f"holds a pose to {e_hold:.1e} rad; 10 N load sags {np.abs(e_meas).max() * 1e3:.3f} mrad vs Kp^-1 J^T F "
      f"{np.abs(e_pred).max() * 1e3:.3f} mrad")
arm_.set_q(dr_, RB.Q_READY)
dr_.time = 0.0
mujoco.mj_forward(mr, dr_)
err_ff = 0.0
while dr_.time < 2.0:
    qd_, vd_, ad_ = RB.min_jerk(RB.Q_READY, RB.Q_CAPTURE, 2.0, dr_.time)
    arm_.control(dr_, qd_, vd_, ad_)
    mujoco.mj_step(mr, dr_)
    err_ff = max(err_ff, np.abs(qd_ - dr_.qpos[arm_.qadr]).max())
check("J.1", "mj", err_ff < 2e-3, f"tracking error during a 90 deg base turn in 2 s: {err_ff * 1e3:.3f} mrad")

ok_j2 = True
for _ in range(200):
    Rg = Rot.random(random_state=rng.integers(1 << 31)).as_matrix()
    tg_, dep = rng.normal(size=3), rng.uniform(0.01, 0.04)
    p_, Rh = RB.grasp_to_tcp(Rg, tg_, dep)
    Rf = Rg.copy()
    Rf[:, 1:] *= -1
    p2_, Rh2 = RB.grasp_to_tcp(Rf, tg_, dep)
    w_ = 0.05
    pads1 = {tuple(np.round(p_ + s_ * w_ / 2 * Rh[:, 1], 12)) for s_ in (-1, 1)}
    pads2 = {tuple(np.round(p2_ + s_ * w_ / 2 * Rh2[:, 1], 12)) for s_ in (-1, 1)}
    ok_j2 &= (close(Rh.T @ Rh, np.eye(3), atol=1e-12) and abs(np.linalg.det(Rh) - 1) < 1e-12
              and close(Rh[:, 2], Rg[:, 0]) and close(Rh[:, 1], Rg[:, 1])
              and close(p_, tg_ + (dep - 0.0085) * Rg[:, 0]) and pads1 == pads2)
check("J.2", "num", ok_j2, "hand frame orthonormal and right-handed, z = approach, y = closing, pad centre at "
      "t + (depth - h) a; the half-turn grasp puts the pads in the same two places")

T_, ts_ = 1.3, np.linspace(0.05, 1.25, 25)
h_ = 1e-6
fd_ok = all(abs((RB.min_jerk_scale(T_, t + h_)[0] - RB.min_jerk_scale(T_, t - h_)[0]) / (2 * h_)
                - RB.min_jerk_scale(T_, t)[1]) < 1e-6
            and abs((RB.min_jerk_scale(T_, t + h_)[1] - RB.min_jerk_scale(T_, t - h_)[1]) / (2 * h_)
                    - RB.min_jerk_scale(T_, t)[2]) < 1e-5 for t in ts_)
check("J.3", "num", fd_ok, "minimum-jerk velocity and acceleration match finite differences")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)

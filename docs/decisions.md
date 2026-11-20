# AnyGrasp++: decision log

Every change from the proposal (`proposal/proposal_final.tex`), with the reason and the evidence. All of them were made **before any study results existed**; the midpoint memo can cite this file. Equations referenced as A.x, B.x, … are in `docs/derivations.pdf`.

| # | Decision | Date |
|---|---|---|
| D1 | Detector: GSNet, not the AnyGrasp SDK | 2026-11-17 |
| D2 | Failure = rotated more than 10° in the hand (was 30°) | 2026-11-17 |
| D3 | Only grasps whose approach is within 30° of straight down, for every method | 2026-11-17 |
| D4 | Realistic fingertip: 4×4 contact grid over the real 17×17 mm pad; rubber, with rigid as a sensitivity case | 2026-11-17 |
| D5 | Objects own their contact parameters (MuJoCo priority) | 2026-11-17 |
| D6 | Contact settings: stiff, hard friction, converged | 2026-11-17 |
| D7 | Gripper: force-controlled at 70 N per pad | 2026-11-17 |
| D8 | M3 test lift: hold 2 s, at full grip, slip detected across the lift | 2026-11-17 |
| D9 | M2 score: detector score × P(hold), with an exact uniform prior | 2026-11-17 |
| D10 | Statistics: everything clustered by object | 2026-11-17 |
| D11 | Wording fixes to the proposal | 2026-11-17 |
| D12 | Full-arm scene rules for step 2 | 2026-11-17 |
| D13 | New control "B0 + blind retry" (professor's required change) | 2026-11-20 |
| D14 | Detector runs: pinned GPU environment, cached once per scene; Panda width rule | 2026-11-20 |
| D15 | Hidden-weight objects weigh 1.0 kg (was 0.6 kg) | 2026-11-20 |
| D16 | Hidden weight is a dense core in the object's heavy end; object selection from all 1030 scans | 2026-11-20 |

---

## D1. Detector: GSNet, not the AnyGrasp SDK

**Decision.** Use the open GSNet implementation `github.com/graspnet/graspness_unofficial` with its RealSense weights; `graspnet-baseline` is the fallback. AnyGrasp stays a Stretch item, as in the proposal, and is skipped for now.

**Why not AnyGrasp.**
- **Closed and machine-locked.** Its detector ships only as a compiled library: "Due to the IP issue, currently we can only release the SDK library file of AnyGrasp in a licensed manner" (SDK README). The licence is tied to the feature ID of one machine.
- **Linux x86_64 with CUDA only.** It cannot run on a Mac.
- **Approval is slow and not guaranteed.** You apply with a form (name, affiliation, advisor, signature, feature ID). The authors "usually reply in 5 workdays".
- **The weights come only with the licence.**
- **Rented GPUs break the licence.** A stopped Vast.ai instance may not get the same machine back. Keeping one alive for the week until approval costs money even if the licence never comes.

**Why GSNet is enough for this question.**
- AnyGrasp's detector is GSNet: "Our geometry processing module is based on GSNet, with a minor modification to incorporate the stable score into the network" (AnyGrasp paper, arXiv 2212.08333).
- It adds a learned centre-of-mass "stable score", 168 extra training scenes, an explicit collision check on the top 100 grasps, and tracking. The paper gives no head-to-head accuracy numbers against GSNet.
- Our method treats the detector as a black box and re-ranks its grasps. The hidden weight is equally invisible to both detectors.
- Baseline B2 re-ranks by the *visible-shape* centre of mass, which plays the role of AnyGrasp's stable score.

**Licence and weights.**
- GSNet and graspnet-baseline are "academic or non-profit organization noncommercial research use only", which covers a course project.
- **Do not redistribute the weights.** Keep them out of public repositories and out of the submitted package.
- Weights are in `assets/weights/`:
  - `minkuresunet_realsense.tar`: 15.4 M parameters, SHA-256 `a0b9d85b…`
  - `checkpoint-rs.tar`: 1.0 M parameters, SHA-256 `60680087…`
- Both were checked: they load, and every layer name matches its model code.

**Revisit if** a fixed Linux PC with an NVIDIA GPU becomes available (for example a school lab machine). Then AnyGrasp can be added as an extra baseline: "does AnyGrasp's learned centre-of-mass score already handle hidden weights?"

## D2. Failure = rotated more than 10° in the hand (was 30°)

**Why.** In MuJoCo, an object pushed past its slip limit swings 20–40° and then either sticks again or comes to rest on the palm. With a 30° rule, many real twist-outs would count as successes, depending on object height. Example: a 6 cm-tall box at d = 8 cm ended at 21°, which would score as a success.

**Evidence.**
- Single-point pad: the failure point barely moves for any threshold from 0.5° to 15°.
- Realistic grid pad: slipping starts inside the predicted band, and the 10° failure comes 4–10% later because the object re-sticks as its lever arm shrinks (appendix R).

**Refinement (same day).** The 10° counts rotation *away from the approach axis*: the twist about the closing axis plus any tilt in the finger plane, measured in the hand's frame. Rotation *about* the approach axis is the object squaring up to the fingers. It is logged separately and does not count as a failure. A 10°-yawed box on a rigid pad squares up by 9.5° with only 0.25° of twist, which the total-angle metric had wrongly scored as a near-failure.

## D3. Only grasps whose approach is within 30° of straight down, for every method (B0 included)

**Why.**
- The slip model needs the closing axis close to horizontal, so that friction, not a pad's normal force, carries the weight.
- AnyGrasp's authors use exactly this filter in their real-robot example code ("within a 30-degree angle with the vertical pose"), so B0 is still "the usual way".
- Tilts of 15° and 30° were checked in simulation (appendix R.3).

**In code:** `physics.approach_ok()`.

## D4. Realistic fingertip: 4×4 contact grid over the real 17×17 mm pad

**Decision.** Each fingertip is 16 contact points spread over the real Franka fingertip face. The size comes from the Menagerie model's pad box, 17 × 17 mm.

**Why.**
- The size of the contact patch decides when an object twists out.
- A real fingertip touches across its whole face on a flat side, along a line on a handle, and along one edge when the object is slightly turned or the grasp is off-centre. The grid reproduces all of these.
- A single point with a fixed patch radius (the proposal's a = 6 mm) cannot.
- With the grid, twisting resistance *emerges* from the contact geometry (derivation A.12). Side question (a) therefore becomes a real test, instead of something the simulator satisfies by construction.

**Consequences.**
- A flat face holds up to d\* = 7.53 cm at 0.6 kg, μ = 0.5, 70 N. The proposal's point-pad value was 4.74 cm.
- A handle-like line contact holds 5.74 cm. Edge contacts hold much less.
- **Open item for step 2:** at 0.6 kg with 0–8 cm shifts, box-like objects barely reach their failure point. The object masses and shift range need revisiting when the 20 objects are chosen. Curved tools fail earlier, so the set may still work.

**Pad softness: rubber by default, rigid as a sensitivity case.**
- Softness barely matters when the fingertip sits flat on the object, but it decides what happens when it doesn't. A box yawed 3° starts slipping at 1.4 cm with a rigid pad, 2.8 cm with rubber (0.11 mm sink) and 5.7 cm with a soft pad (0.6 mm). It fails (10°) at 1.6 cm with the rigid pad and 5.2 cm with rubber. GSNet's grasp angles come in 15° steps, so misalignments of a few degrees are normal.
- The default is a physical estimate, not a tuned value. Treat the fingertip as a 2 mm rubber layer with stiffness E ≈ 3 MPa pressed by 70 N over 17×17 mm: that is 0.24 MPa, which compresses it about 0.16 mm. The rubber setting sinks 0.11 mm.
- Hard-plastic (rigid) pads are run as a sensitivity case in the evaluation, like μ = 0.8.

**Consequence for M2.** The patch model assumes the fingertip is parallel to the object face. Detecting misaligned grasps from the point cloud (which pad points will touch) is the main open modelling question for step 4.

**The single-point pad is kept** only to verify the simulator. MuJoCo implements the textbook soft-finger model exactly, and 8 of 8 parameter combinations fall in the predicted band.

## D5. Objects own their contact parameters (MuJoCo priority)

**Decision.** Every object geom has `priority="1"` and carries its own friction, condim, solref and solimp.

**Why.** With equal priority, MuJoCo takes the *larger* friction of the two geoms and a 50/50 *average* of solref/solimp. Before this fix:
- the "calibrated" stiff pad settings never reached the contact (it ran at solref 0.012, not 0.004);
- an object could never be less slippery than the pad, which breaks the glossy-handle trick object and M3's "lower the friction guess where it slipped" logic.

This also removes the earlier "visible shell ≥ 10% of the mass" rule, which was only a side effect of the averaged, softer contact.

## D6. Contact settings: stiff, hard friction, converged

**Settings.**
- Rubber pad (nominal): solref 0.01 1, solimp 0.95 0.99 0.001.
- Rigid pad (sensitivity case): solref 0.004 1, solimp 0.99 0.999 0.001.
- Both: elliptic friction cone, impratio 10, `noslip_iterations` = 2000, time step 2 ms.

**Why.**
- MuJoCo's defaults let a 70 N contact sink millimetres and let friction creep under load. With them, both pad types fail near 4.6 cm whatever the pad, so the pad physics is hidden.
- The noslip solver stops at its tolerance, so the setting is a cap. 20 iterations left 0.8° of false creep on the grid pad. 500 was enough for aligned grasps but not for misaligned ones. 2000, 5000 and 20,000 give identical results.
- solref 0.004 is the smallest time constant MuJoCo allows at a 2 ms step (2 × dt). A larger step would silently clamp it.

**Cost.** About 1–6 s of computer time per grasp trial with the grid pad; misaligned grasps are the slow ones. Re-measure with the full arm in step 2.

## D7. Gripper: force-controlled at 70 N per pad

**Decision.** A stiff position servo on the finger tendon with its force clamped at 2 × 70 N, closing at 0.1 m/s, like Franka's `grasp(width, speed, force)`.

**Why.** Menagerie's gripper is a weak position spring. It squeezed a 4 cm object with 1.0 N per finger and dropped it even with no weight shift (derivation B.3).

**The first version chattered.** Its damping was too large for the 2 ms step once the force hit its clamp. The documented MuJoCo behaviour is that a clamped force gets no implicit damping (B.4).

**In the full scene,** Menagerie's ±100 N actuator range (50 N per pad) must be replaced.

## D8. M3 test lift: hold 2 s, at full grip, slip detected across the lift

**Hold time.** The CoM accuracy estimate in the proposal notes was 3.3× too optimistic. It assumed random joint axes and ignored the coupling inside the arm Jacobian (E.5).

Computed from the real Menagerie Panda at 27 lift poses across the workspace (`scripts/panda_m3_numbers.py`), the median CoM error is:

| Torque noise | Hold 0.3 s (150 samples) | Hold 1.5 s |
|---|---|---|
| 0.1 N·m | 0.5 cm | — |
| 0.5 N·m | 2.5 cm | 1.1 cm |
| 1.0 N·m | — | 2.2 cm |

Reaching < 1 cm at 0.5 N·m needs about 940 samples, a 1.9 s hold. **Decision: hold 2 s.** Also prefer lift poses whose Jacobian measures the CoM well (step 4). The < 1 cm target is unchanged.

**Grip force.** Do the test lift at the full 70 N. At the gentle-grip low end (30 N) the slip limit is only 2 cm, so objects would twist *during* the measurement. Lower the grip afterwards, for transport, using A.10 (not the proposal's formula, which drops the ¼ term).

**Slip detection.** In simulation, twisting happens during the lift, not during the hold. So detect slip from the change in pose or moment between the pre-lift and post-lift readings, not from drift during the hold.

## D9. M2 score: detector score × P(hold), with an exact uniform prior

**Decision.** Score each grasp by detector score × P(hold), instead of the margin ρ the proposal mentions.
- **Before the lift,** the CoM is uniform over the footprint, and P(hold) is computed exactly (C.5). Replacing the uniform with a Gaussian of the same spread was off by up to 0.11 in probability.
- **After the lift,** the belief is Gaussian (C.3).

The margin ρ is the large-certainty limit of the same rule (F.2).

## D10. Statistics: everything clustered by object

**Why.** Trials on the same object are correlated.
- The proposal's "±13 points at worst" assumed independent trials. With 6 trials per object and a within-object correlation of 0.2–0.5, the true worst case is 18–24 points.
- McNemar's test also assumes independent pairs. It tests "any difference", not "at least 15 points".

**Rules.**
- 95% intervals come from the cluster bootstrap over objects, as the proposal already planned.
- The paired test uses that bootstrap, or a clustered McNemar test.
- The B0 failure-point fit uses a bounded or penalized (Firth) logistic regression, because simulated outcomes are nearly deterministic in d and an unbounded fit would not converge.

**Threshold rule:** a threshold like "≥ 15 points" is met when the point estimate is ≥ 15 points *and* the cluster-bootstrap 95% interval excludes 0.

## D11. Wording fixes to the proposal (for the midpoint memo)

- d is the horizontal distance from the CoM to the vertical plane containing the closing axis, not to "the line between the fingers".
- ρ = 1 − |μ_d|/d\* (with an absolute value).
- d\* = 4.74 cm for the exact soft-finger formula; the simple form gives 4.76 cm. Neither applies to the realistic pad (D4).
- Gentle grip: use A.10, f_n,min = (W/μ)·√(¼ + (3d/4a)²).
- An 8 cm shift needs at least 8.9 cm from the centroid to the hidden weight, so objects at least about 18 cm long.
- `proposal/derivations.tex` Sec. 4.4 has a sign error: r⊥ = (n × e_z)/(mg), not (e_z × n)/(mg).

## D12. Full-arm scene rules for step 2

- **No `gravcomp` on the arm links.** The joint torques must contain the hand's and object's weight, and the 5% link-mass error must reach the readings.
- **Copy the scene options:** cone, impratio, noslip, the object class with priority, the grid pads and the force gripper.
- **Measure the real hand acceleration during the lift.** The arm's servos will not follow the min-jerk profile exactly, and the prediction band depends on it.
- **Log every non-pad contact** (finger body, palm) as its own outcome.

## D13. New control "B0 + blind retry" (professor's required change)

**Professor's feedback (proposal decision):** "AnyGrasp++ gets a second attempt that B0 doesn't, so part of any gain could come from simply retrying. Add a 'B0 + blind retry' control that regrasps with the next-best grasp without re-ranking." To be fixed before the midpoint.

**Proposed definition (to confirm before the main runs):**
- B0's first grasp, then the same 2 cm test lift and hold as AnyGrasp++.
- If the object has slipped (tilt > 10° or dropped), set it down and regrasp once with the detector's **next-best reachable grasp**, in the detector's own order, with no physics re-ranking.
- Slip is read from the simulator. That is an oracle trigger, which gives blind retrying its best possible chance, so the control is deliberately strong. If AnyGrasp++ still beats it, its gain does not come from simply retrying.

**Effect on the plan:** one more method on every object set (B0, B0 + retry, B1, B2, AnyGrasp++, AnyGrasp++ ideal). Detector outputs are cached per scene, so the retry needs a new detection only after the set-down. The extra cost is simulation time on the Mac.

## D14. Detector runs: pinned GPU environment, cached once per scene

**Environment** (`gpu/setup_gpu.sh`): Python 3.10, PyTorch 2.1.2 + CUDA 11.8, gcc 11, numpy 1.23.5, MinkowskiEngine from the GSNet authors' fork (`chenxi-wang/MinkowskiEngine`, master), all four source repos pinned to a commit. This is the stack the authors' code was written against. The rented image's own PyTorch 2.14 / CUDA 13 is newer than MinkowskiEngine has been tested with, so it is left untouched. The compiled extensions are kept as wheels for sm_80/86/89, so a later session on any RTX 30xx/40xx or A-series card installs them in minutes instead of recompiling.

**One extension build serves both detectors.** GSNet's copy of `pointnet2` differs from graspnet-baseline's only in line endings and comments. GSNet's copy of `knn` still uses PyTorch's removed THC API, and GSNet uses `knn` only to build training labels, so the baseline's updated copy is the one built.

**Pipelines follow the authors' inference scripts exactly** (`gpu/detect_grasps.py`): GSNet with 15000 points and 5 mm voxels; the baseline with 20000 points, 300 views, 12 angles and 4 depths. Then the authors' model-free collision check (1 cm voxels, 5 cm approach, 1 cm threshold) and graspnetAPI NMS (3 cm, 30°). The cache stores the raw network output, the collision mask and the NMS order, so filters such as D3 are applied later without rerunning the network. Point sampling is seeded per scene, and a rerun reproduces the output bit for bit (checked on the authors' example scene).

**Panda width rule (decided 2026-11-20).** Both detectors report width = 1.2 × predicted object thickness, clamped at 10 cm (they were trained for a 10 cm gripper). The Panda opens to 8 cm. On the authors' example scene only 6 of each detector's top 20 grasps report ≤ 8 cm. Rule for every method: keep a grasp only if its reported width ≤ 8 cm. This keeps the authors' 20% clearance, so objects up to 6.7 cm thick are graspable.

**Effect on the plan:** the retry in B0 + blind retry (D13) and AnyGrasp++'s regrasp both happen after a set-down that can move the object, so they need a second detection on the new scene. Detection therefore runs in two batched passes on the GPU: initial scenes, then post-set-down scenes. The setup above is built to be reused for that.

## D15. Hidden-weight objects weigh 1.0 kg (was 0.6 kg)

**Why:** the proposal chose 0.6 kg so that the slip distance d\* (4.7 cm under the point-contact model) sat in the middle of the 0–8 cm shifts. B0 should then hold on one side of d\* and fail on the other, which is what the failure-point fit and the main comparison need. The realistic fingertip of D4 is stronger in torsion: at 0.6 kg its d\* is 7.5 cm at rest and 7.2 cm at the lift's peak acceleration, close to the top of the range, so B0 would rarely fail.

**Numbers** (`physics.patch_dstar`, 4×4 grid pad, 70 N per pad, at rest / at the lift's peak acceleration 0.40 m/s²):

| mass | d\* at μ = 0.5 | d\* at μ = 0.8 |
|---|---|---|
| 0.6 kg | 7.53 / 7.23 cm | 12.08 / 11.61 cm |
| 0.8 kg | 5.62 / 5.39 cm | 9.05 / 8.69 cm |
| **1.0 kg** | **4.47 / 4.29 cm** | **7.22 / 6.93 cm** |
| 1.2 kg | 3.70 / 3.55 cm | 6.00 / 5.76 cm |

At 1.0 kg, d\* returns to the proposal's design point in the middle of the shift range at μ = 0.5. The μ = 0.8 repeat still has d\* inside the range. 1 kg is an ordinary household weight (hammer, drill, full bottle) and well under the Panda's 3 kg payload. The shifts (0, 2, 4, 6, 8 cm) and everything else stay as proposed.

**Chosen 2026-11-20** to keep the test hard and realistic.

## D16. Hidden weight is a dense core in the heavy end; object selection from all 1030 scans

**The hidden weight must physically fit.** The proposal placed an invisible point mass (B.2). At 1.0 kg an 8 cm shift needs about 0.8 kg of hidden mass, and even tungsten (19.3 g/cm³, the densest practical filler) needs about 41 cm³ for that, a slug about 4 cm across. The proposal's own example, a weight inside a spatula blade, could not exist. So the hidden weight is now a **core**: the end cap of the object beyond a cut plane, inside a 2 mm skin, filled with denser material. Think of a hammer head, the batteries at one end of a flashlight, or sand at the bottom of a bottle. The rest of the object is a uniform shell. The cut is chosen once per object, as the outermost one that reaches the 8 cm shift with the core at or below tungsten density and at least 100 g left in the shell. The same core is used at every shift and only the two densities change, so the object looks identical at every shift (derivations H.2; the voxel CoM equals the target shift to 1e-12 m).

**Selection pipeline** (`scripts/select_objects.py`, `anygrasp_pp/objects.py`), run on all 1030 MuJoCo scanned objects:

1. Surface survey: length 18–45 cm and some part thin enough for the width rule (D14). 289 of 1030 pass.
2. Solid survey of the collision solid: the core is feasible (244) and at least 5 cm of the length is graspable (181).
3. A single rigid piece (153). Sets of loose items (furniture sets, game pieces, tea sets) are found automatically by counting separate pieces after closing gaps under 2 mm.
4. Not footwear (31; shoes are deformable), and not in `results/objects/exclusions.csv` (17). That file gives the reason for each by-eye exclusion: soft items, cords and strings the scan made rigid, and stacks of loose pieces scanned while touching.

Relaxing the length to 15 cm or the shell to 50 g adds no objects; the 8 cm shift at 1 kg is the binding limit.

**Final test** (`scripts/graspability_test.py`; results in `results/objects/gtest/`). Each of the 17 is placed in its most probable resting pose whose support polygon holds the CoM with 5 mm to spare at every shift (derivations I.3). It is simulated for 1 s at shift 0 and 8 cm and must move less than 5 mm. It is rendered by the overhead camera at four yaws and run through both detectors, using the authors' workspace crop (I.1). It passes if every yaw gives at least 3 grasps that are collision-free, report a width ≤ 8 cm (D14), sit on the object and approach within the angle limit.

- Physics removes three: the teapot has no pose stable at every shift, one track piece rolls 395 mm, and one action figure tips over at 8 cm.
- The angle limit decides the rest. Objects passing (of 17):

| approach limit | 30° (D3) | 45° | 60° | none |
|---|---|---|---|---|
| GSNet | 5 | 6 | 8 | 12 |
| graspnet-baseline | 5 | 8 | 12 | 12 |

- At 30° only an elephant figure, an excavator toy, a shark toy, a bookend and a wooden track piece pass. The mallet, flashlight, dustpan and boxes get grasps, but GSNet's grasps on them are mostly steep side grasps (median 73° from vertical) or collide; on the flashlight the body is 7.5 cm, too wide for the hand.

**Open (decided with the user next):** the realistic GSO set cannot supply 20 hidden-weight objects that the detector grasps reliably from above. The options are to relax D3, to add objects from another scanned set, or to run fewer objects with more seeds. Collision geometry (below) and the camera noise model (proposal L7) may also change GSNet's grasps, since it was trained on real, noisy depth.

**Collision fidelity:** the scans' V-HACD collision pieces bulge a median 6.7 mm (95th percentile) beyond the visible surface, enough to put the fingers on material the camera cannot see. The chosen objects get new collision pieces from CoACD (named in the proposal), checked with the same bulge/gap measure (`objects.fidelity`).

# Decision log

This is where I record every change from the proposal (`proposal/proposal_final.tex`), why I made it, and what the evidence was. All of these were decided before any study results existed, so the midpoint memo can point here. Equation numbers like A.12 refer to `docs/derivations.pdf`.

| # | Decision | Date |
|---|---|---|
| D1 | GSNet as the detector instead of the AnyGrasp SDK | 2026-11-17 |
| D2 | A grasp fails if the object turns more than 10° in the hand (was 30°) | 2026-11-17 |
| D3 | Only grasps within 30° of straight down, for every method | 2026-11-17 |
| D4 | A realistic fingertip: 4×4 contact grid over the real 17×17 mm pad | 2026-11-17 |
| D5 | Objects own their contact parameters | 2026-11-17 |
| D6 | Stiff, converged contact settings | 2026-11-17 |
| D7 | Force-controlled gripper at 70 N per pad | 2026-11-17 |
| D8 | M3 test lift: 2 s hold at full grip | 2026-11-17 |
| D9 | M2 score = detector score × P(hold) | 2026-11-17 |
| D10 | All statistics clustered by object | 2026-11-17 |
| D11 | Wording fixes to the proposal | 2026-11-17 |
| D12 | Rules for the full-arm scene | 2026-11-17 |
| D13 | New control: B0 + blind retry | 2026-11-20 |
| D14 | Detector environment, caching, and the Panda width rule | 2026-11-20 |
| D15 | Objects weigh 1.0 kg (was 0.6 kg) | 2026-11-20 |
| D16 | The hidden weight is a dense core; 10 objects out of 1108 scans | 2026-11-20 |
| D17 | Simulated depth behaves like a RealSense D435 | 2026-11-20 |
| D18 | How a B0 trial runs | 2026-11-20 |
| D19 | First B0 run: the fingertip was too stiff | 2026-11-20 |
| D20 | The fingertip as a thin rubber layer | 2026-11-20 |
| D21 | Object mass re-chosen with the measured slip distance: 0.45 kg | 2026-11-20 |

---

## D1. GSNet as the detector instead of the AnyGrasp SDK

I use the open GSNet code (`github.com/graspnet/graspness_unofficial`) with its RealSense weights, and keep graspnet-baseline as a fallback. AnyGrasp stays a stretch goal, as in the proposal.

AnyGrasp turned out to be impractical here. The detector only ships as a compiled library ("Due to the IP issue, currently we can only release the SDK library file of AnyGrasp in a licensed manner", SDK README), and the licence is locked to one machine's feature ID. It runs only on Linux x86_64 with CUDA, so not on my Mac. Getting a licence means a form with an advisor's signature and about five working days of waiting, and the weights only come with the licence. On top of that I rent GPUs on Vast.ai, and a stopped instance may not come back as the same machine, which would break a machine-locked licence.

GSNet is enough for this question. AnyGrasp's own paper says its geometry module "is based on GSNet, with a minor modification to incorporate the stable score into the network" (arXiv 2212.08333). AnyGrasp adds a learned centre-of-mass "stable score", 168 more training scenes, a collision check on the top 100 grasps, and tracking, but the paper doesn't compare it head to head with GSNet. My method treats the detector as a black box anyway, and a hidden weight is just as invisible to both. Baseline B2, which re-ranks by the centre of the visible shape, stands in for AnyGrasp's stable score.

GSNet and graspnet-baseline are licensed for academic, noncommercial research, which covers a course project. The weights must not be redistributed, so they stay out of the repository. They live in `assets/weights/`: `minkuresunet_realsense.tar` (15.4 M parameters, SHA-256 `a0b9d85b…`) and `checkpoint-rs.tar` (1.0 M, `60680087…`). Both load, and every layer name matches the model code.

If I get access to a fixed Linux machine with an NVIDIA GPU (a lab PC, say), AnyGrasp could come back as an extra baseline to ask whether its learned centre-of-mass score already handles hidden weights.

## D2. A grasp fails if the object turns more than 10° in the hand (was 30°)

In MuJoCo, an object pushed past its slip limit swings 20–40° and then either sticks again or comes to rest against the palm. With the proposal's 30° rule, many real twist-outs would count as successes depending on the object's height. A 6 cm-tall box at d = 8 cm, for example, ended at 21°.

With a single-point pad, the failure point barely moves for any threshold between 0.5° and 15°. With the realistic grid pad, slipping starts inside the predicted band, and the 10° failure comes 4–10% later, because the object re-sticks as its lever arm shrinks (appendix R).

The 10° counts rotation away from the approach axis: the twist about the closing axis plus any tilt in the finger plane, measured in the hand's frame. Rotation about the approach axis is just the object squaring up to the fingers, so I log it separately and don't count it. A box yawed 10° on a rigid pad squares up by 9.5° with only 0.25° of twist, and a total-angle measure had wrongly called that a near-failure.

## D3. Only grasps within 30° of straight down, for every method

The slip model needs the closing axis close to horizontal, so that friction rather than a pad's normal force carries the weight. AnyGrasp's authors use exactly this filter in their real-robot example code ("within a 30-degree angle with the vertical pose"), so B0 is still the usual way of using the detector. Tilts of 15° and 30° are checked in appendix R.3. The filter is `physics.approach_ok()`.

## D4. A realistic fingertip: 4×4 contact grid over the real 17×17 mm pad

Each fingertip is 16 contact points spread over the real Franka fingertip face (17 × 17 mm, from the Menagerie model's pad box).

The size of the contact patch decides when an object twists out. A real fingertip touches across its whole face on a flat side, along a line on a handle, and along one edge when the object is slightly turned or the grasp is off-centre. A grid reproduces all of that, while a single point with a fixed patch radius (the proposal's a = 6 mm) can't. With the grid, twisting resistance comes out of the contact geometry (A.12), so side question (a) becomes a real test rather than something the simulator satisfies by construction.

On a flat face this pad holds up to d\* = 7.53 cm at 0.6 kg, μ = 0.5 and 70 N; the proposal's point-pad value was 4.74 cm. A handle-like line contact holds 5.74 cm, and edge contacts much less. That meant the 0.6 kg objects with 0–8 cm shifts would barely reach their failure point, which D15 deals with.

Softness barely matters when the fingertip sits flat on the object, but it decides what happens when it doesn't. A box yawed 3° starts slipping at 1.4 cm with a rigid pad, 2.8 cm with rubber (0.11 mm sink) and 5.7 cm with a soft pad (0.6 mm). It fails at 1.6 cm with the rigid pad and 5.2 cm with rubber. GSNet's grasp angles come in 15° steps, so misalignments of a few degrees are normal. I took the default from a physical estimate rather than tuning it: a 2 mm rubber layer with E ≈ 3 MPa under 70 N over 17×17 mm (0.24 MPa) compresses about 0.16 mm. Hard plastic pads are run as a sensitivity case, like μ = 0.8. (D20 later replaced this rubber model with one that behaves correctly on misaligned contact.)

The patch model assumes the fingertip is parallel to the object face, so predicting from the point cloud which pad points will touch is the main open modelling question for M2.

I keep the single-point pad only to verify the simulator: MuJoCo implements the textbook soft-finger model exactly, and 8 of 8 parameter combinations fall in the predicted band.

## D5. Objects own their contact parameters

Every object geom has `priority="1"` and carries its own friction, condim, solref and solimp. With equal priority, MuJoCo takes the larger friction of the two geoms and averages their solref/solimp. Before this fix my "calibrated" pad settings never reached the contact (it ran at solref 0.012, not 0.004), and an object could never be less slippery than the pad, which would break the glossy-handle trick object and M3's idea of lowering the friction guess where something slipped. It also removed an earlier rule that the visible shell keep at least 10% of the mass, which had only been a side effect of the averaged, softer contact.

## D6. Stiff, converged contact settings

The step-1 settings were solref 0.01 1 and solimp 0.95 0.99 0.001 for the rubber pad, and solref 0.004 1, solimp 0.99 0.999 0.001 for the rigid one, with an elliptic friction cone, impratio 10, 2000 noslip iterations and a 2 ms time step. (D20 replaced the rubber setting.)

MuJoCo's defaults let a 70 N contact sink millimetres and let friction creep under load. With them, both pad types fail near 4.6 cm whatever the pad, which hides the pad physics. The noslip solver stops at its tolerance, so the iteration count is only a cap: 20 left 0.8° of false creep on the grid pad, 500 was enough for aligned grasps but not misaligned ones, and 2000, 5000 and 20,000 give identical results. solref 0.004 is the smallest time constant MuJoCo allows at a 2 ms step (twice the step); a larger step would silently clamp it.

A grasp trial with the grid pad costs about 1–6 s of computer time, and misaligned grasps are the slow ones.

## D7. Force-controlled gripper at 70 N per pad

The gripper is a stiff position servo on the finger tendon with its force clamped at 2 × 70 N, closing at 0.1 m/s, like Franka's `grasp(width, speed, force)`. Menagerie's gripper is a weak position spring: it squeezed a 4 cm object with 1.0 N per finger and dropped it even with no weight shift (B.3). My first version chattered because its damping was too large for the 2 ms step once the force hit its clamp; MuJoCo gives a clamped force no implicit damping (B.4). In the full scene, Menagerie's ±100 N actuator range (50 N per pad) is replaced as well.

## D8. M3 test lift: 2 s hold at full grip

The CoM accuracy I estimated in the proposal notes was 3.3 times too optimistic, because it assumed random joint axes and ignored the coupling inside the arm's Jacobian (E.5). Computed on the real Menagerie Panda at 27 lift poses (`scripts/panda_m3_numbers.py`), the median CoM error is:

| Torque noise | Hold 0.3 s (150 samples) | Hold 1.5 s |
|---|---|---|
| 0.1 N·m | 0.5 cm | — |
| 0.5 N·m | 2.5 cm | 1.1 cm |
| 1.0 N·m | — | 2.2 cm |

Getting below 1 cm at 0.5 N·m takes about 940 samples, a 1.9 s hold, so I hold for 2 s and prefer lift poses whose Jacobian measures the CoM well. The 1 cm target stays.

The test lift runs at the full 70 N. At the gentle-grip end (30 N) the slip limit is only 2 cm, so objects would twist during the measurement. The grip can be lowered afterwards for transport using A.10 (the proposal's formula dropped the ¼ term).

In simulation, twisting happens during the lift rather than during the hold, so slip is detected from the change in pose or moment between the readings before and after the lift, not from drift during the hold.

## D9. M2 score = detector score × P(hold)

I score each grasp by the detector's score times the probability that it holds, rather than by the margin ρ mentioned in the proposal. Before the test lift, the CoM is uniform over the footprint and P(hold) is computed exactly (C.5); a Gaussian with the same spread was off by up to 0.11. After the lift the belief is Gaussian (C.3). The margin ρ is the high-certainty limit of the same rule (F.2).

## D10. All statistics clustered by object

Trials on the same object are correlated. The proposal's "±13 points at worst" assumed independent trials; with 6 trials per object and a within-object correlation of 0.2–0.5, the real worst case is 18–24 points. McNemar's test also assumes independent pairs, and it tests for any difference rather than at least 15 points.

So the 95% intervals come from a bootstrap over objects (as the proposal planned), the paired test uses that bootstrap or a clustered McNemar test, and the B0 failure-point fit uses a bounded or penalized (Firth) logistic regression, because simulated outcomes are nearly deterministic in d and an unbounded fit won't converge. I count a threshold like "at least 15 points" as met when the point estimate reaches it and the bootstrap interval excludes zero.

## D11. Wording fixes to the proposal

- d is the horizontal distance from the CoM to the vertical plane containing the closing axis, not to "the line between the fingers".
- ρ = 1 − |μ_d|/d\*, with an absolute value.
- d\* = 4.74 cm for the exact soft-finger formula and 4.76 cm for the simple form; neither applies to the realistic pad (D4).
- For the gentle grip, use A.10: f_n,min = (W/μ)·√(¼ + (3d/4a)²).
- An 8 cm shift needs at least 8.9 cm from the centroid to the hidden weight, so objects at least about 18 cm long.
- `proposal/derivations.tex` Sec. 4.4 has a sign error: r⊥ = (n × e_z)/(mg), not (e_z × n)/(mg).

## D12. Rules for the full-arm scene

No `gravcomp` on the arm links: the joint torques have to contain the hand's and the object's weight, and the 5% link-mass error has to reach the readings. The scene copies step 1's contact options, the object class with its priority, the grid pads and the force gripper. The hand's real acceleration during the lift is measured, because the arm won't follow the minimum-jerk profile exactly and the predicted band depends on it. Every contact that isn't a pad (finger body, palm) is logged as its own outcome.

## D13. New control: B0 + blind retry

My professor's feedback on the proposal: AnyGrasp++ gets a second attempt that B0 doesn't, so part of any gain could just be the retry. The fix is a "B0 + blind retry" control that regrasps with the next-best grasp without re-ranking, to be defined before the midpoint.

My definition: B0's first grasp, then the same test lift and hold as AnyGrasp++. If the object has slipped (more than 10° or dropped), it is set down and regrasped once with the detector's next-best reachable grasp, in the detector's own order. Slip is read from the simulator, which is an oracle trigger, so blind retrying gets its best possible chance. If AnyGrasp++ still beats it, the gain isn't just from retrying.

That adds one more method on every object set (B0, B0 + retry, B1, B2, AnyGrasp++, AnyGrasp++ with an ideal sensor). The retry needs a new detection after the set-down, since the object may have moved.

## D14. Detector environment, caching, and the Panda width rule

`gpu/setup_gpu.sh` pins Python 3.10, PyTorch 2.1.2 with CUDA 11.8, gcc 11, numpy 1.23.5 and MinkowskiEngine from the GSNet authors' fork (`chenxi-wang/MinkowskiEngine`, master), with all four source repos fixed to a commit. That's the stack the authors' code was written for; the rented image's own PyTorch 2.14 and CUDA 13 are newer than MinkowskiEngine has been tested with, so I leave them alone. The compiled extensions are saved as wheels for sm_80/86/89, so a later session on any RTX 30xx/40xx or A-series card installs them in minutes.

One extension build serves both detectors. GSNet's copy of `pointnet2` differs from graspnet-baseline's only in line endings and comments, and GSNet's `knn` still uses PyTorch's removed THC API but is only used to build training labels, so I build the baseline's updated copy.

`gpu/detect_grasps.py` follows the authors' inference scripts: GSNet with 15000 points and 5 mm voxels, the baseline with 20000 points, 300 views, 12 angles and 4 depths, then the authors' collision check (1 cm voxels, 5 cm approach, 1 cm threshold) and graspnetAPI's NMS (3 cm, 30°). The cache keeps the raw network output, the collision mask and the NMS order, so filters like D3 can be applied later without rerunning the network. Point sampling is seeded per scene, and a rerun reproduces the output bit for bit.

Both detectors report width as 1.2 × the predicted object thickness, clamped at 10 cm (they were trained for a 10 cm gripper), while the Panda opens to 8 cm. On the authors' example scene only 6 of each detector's top 20 grasps report 8 cm or less. Every method keeps a grasp only if its reported width is at most 8 cm, which keeps the authors' 20% clearance and allows objects up to 6.7 cm thick.

The retry in D13 and AnyGrasp++'s regrasp both come after a set-down that can move the object, so detection runs in two batched passes: the initial scenes, then the scenes after set-down.

## D15. Objects weigh 1.0 kg (was 0.6 kg)

The proposal chose 0.6 kg so that the slip distance d\* (4.7 cm with the point-contact model) sat in the middle of the 0–8 cm shifts, with B0 holding on one side and failing on the other. The realistic fingertip of D4 resists twisting better: at 0.6 kg its flat-face d\* is 7.5 cm at rest and 7.2 cm at the lift's peak acceleration, near the top of the range, so B0 would rarely fail.

Flat-face d\* (`physics.patch_dstar`, 4×4 pad, 70 N, at rest / at the lift's 0.40 m/s² peak):

| mass | μ = 0.5 | μ = 0.8 |
|---|---|---|
| 0.6 kg | 7.53 / 7.23 cm | 12.08 / 11.61 cm |
| 0.8 kg | 5.62 / 5.39 cm | 9.05 / 8.69 cm |
| 1.0 kg | 4.47 / 4.29 cm | 7.22 / 6.93 cm |
| 1.2 kg | 3.70 / 3.55 cm | 6.00 / 5.76 cm |

At 1.0 kg, d\* is back in the middle of the range at μ = 0.5, and the μ = 0.8 repeat still has it inside the range. 1 kg is an ordinary household weight and far below the Panda's 3 kg payload. I wanted the test to be hard and realistic, and this is the mass that does that. (D21 later revisits the number using the slip distances the full pipeline actually measured.)

## D16. The hidden weight is a dense core; 10 objects out of 1108 scans

The hidden weight has to fit inside the object physically. The proposal placed an invisible point mass (B.2), but at 1.0 kg an 8 cm shift needs about 0.8 kg of hidden mass, and even tungsten (19.3 g/cm³) needs about 41 cm³ for that, a slug about 4 cm across. The proposal's own example of a weight inside a spatula blade couldn't exist. So the hidden weight is now a core: the end of the object beyond a cut plane, inside a 2 mm skin, filled with denser material, like a hammer head, the batteries at one end of a flashlight, or sand at the bottom of a bottle. The rest is a uniform shell. For each object I take the outermost cut that reaches the 8 cm shift with the core no denser than tungsten and at least 100 g left in the shell. The same core is used at every shift and only the two densities change, so the object looks identical at every shift (H.2; the voxel CoM lands on the target shift to 1e-12 m).

I selected objects from all 1030 MuJoCo scanned objects with `scripts/select_objects.py` and `anygrasp_pp/objects.py`:

1. Length 18–45 cm and some part thin enough for the width rule (D14): 289 objects.
2. On the collision solid, the core is feasible (244) and at least 5 cm of the length is graspable (181).
3. A single rigid piece (153). Sets of loose items such as furniture sets, game pieces and tea sets are found automatically by counting separate pieces after closing gaps under 2 mm.
4. Not footwear (31; shoes deform) and not in `results/objects/exclusions.csv` (17). That file gives my reason for each object I excluded by eye: soft items, cords and strings that the scan made rigid, and stacks of loose pieces scanned while touching.

Relaxing the length to 15 cm or the shell to 50 g adds nothing; the 8 cm shift at 1 kg is the binding limit.

The last test (`scripts/graspability_test.py`, results in `results/objects/gtest/`) places each of the 17 in its most probable resting pose that keeps the CoM 5 mm inside the support at every shift (I.3), simulates it for 1 s at shift 0 and 8 cm (it must move less than 5 mm), renders it from the overhead camera at four yaws, and runs both detectors with the authors' workspace crop (I.1). It passes if every yaw gives at least 3 grasps that are collision-free, report at most 8 cm, sit on the object and approach within the angle limit.

Physics removes three: the teapot has no pose that's stable at every shift, one track piece rolls 395 mm, and one action figure tips over at 8 cm. The angle limit decides the rest:

| approach limit | 30° (D3) | 45° | 60° | none |
|---|---|---|---|---|
| GSNet | 5 | 6 | 8 | 12 |
| graspnet-baseline | 5 | 8 | 12 | 12 |

At 30° only an elephant figure, an excavator toy, a shark toy, a bookend and a wooden track piece pass. The mallet, flashlight, dustpan and boxes do get grasps, but mostly steep side grasps (median 73° from vertical) or ones that collide; the flashlight's body is 7.5 cm, too wide for the hand. A straight-down camera, as in the GSNet authors' real-robot setup, gives the passing objects more usable grasps (the elephant at least 9 per yaw instead of 4) but the same objects fail (`results/objects/gtest_tilt0/`).

So I added the YCB set (Calli et al. 2015, CC-BY 4.0; 78 objects with textured scans) and kept every rule unchanged. Stage 1 gives 12 YCB candidates, and the core and one-piece checks leave 4. All objects then get CoACD collision pieces and the D435 depth model (D17).

The final set is every object that passes every check, 10 in all (`results/objects/final_set.csv`, from `scripts/final_set.py`): the YCB hammer, power drill, toy airplane and Duplo brick, and from the scanned set a bookend, an excavator toy, an action figure, an elephant figure, a shark toy and a wooden track piece. Each is stable at every shift and gets at least one usable top-down grasp from GSNet at every yaw. The depth model is what lets in the YCB hammer and drill: with ideal depth GSNet gives them no usable grasp at some yaw, which is the proposal's risk 2 (odd grasps on simulated depth) and its mitigation, now measured.

Why 10 and not 20: padding the set with objects that fail these checks would make B0 fail for reasons that have nothing to do with the centre of mass. The funnel from 1108 scans down to 10 is a result in itself. The trial count stays at the midpoint's 300 (10 × 5 shifts × 6 seeds; the proposal had 20 × 5 × 3). The statistics are clustered over 10 objects instead of 20, so object-level intervals are wider, and I report them that way. Objects whose largest feasible shift is a little under 8 cm (mustard bottle 7.1 cm, spatula 7.3 cm) could join in the Target tier after their own graspability test. At least one usable grasp per yaw is the threshold because B0 needs one grasp to execute; requiring three would keep 6 of the 10.

The scans' own V-HACD collision pieces stick out a median 6.7 mm (95th percentile) beyond the visible surface, enough to put the fingers on material the camera can't see. Every finalist and YCB candidate gets new pieces from CoACD (`scripts/make_collision.py`, same settings for all, concavity threshold 0.04), which brings the 95th-percentile bulge down to 1.8–4.8 mm (V-HACD was 5.7–28.2 mm on the scanned ones). The 2 mm voxels and surface sampling put a floor of about 2–3 mm on that measure.

## D17. Simulated depth behaves like a RealSense D435

Both detectors were trained on real D435 depth, and the proposal calls for a sim-to-real noise model. With ideal rendered depth, GSNet gave no usable top-down grasp at some yaw on the YCB hammer and drill.

The model (`scene.realsense_depth`, I.4) follows how the sensor works, active stereo with a 50 mm baseline and depth reported for the left imager:

1. Disparity noise of 0.08 px RMS (Intel's typical subpixel figure), so depth error grows as z², 0.73 mm at the pick area.
2. Stereo shadows: pixels the right imager can't see get no depth. I find them by rendering the right imager, which also gives the real sensor's invalid band along the left edge.
3. Pixels at depth jumps over 1.5 cm are invalid half the time.
4. Depth is rounded to 1 mm, the sensor's (and GraspNet's) unit.

The noise is seeded per scene. The checks: σ_z matches z²σ_d/(fb) within 0.4% by Monte Carlo, the left band is 71 columns (71.3 predicted), and the shadow behind a 10 cm box is 13 px wide (13.0 predicted) with no false holes on open table. With the model on, the YCB hammer and drill get usable grasps at every yaw; the scanned objects pass or fail the same either way.

## D18. How a B0 trial runs

I fixed this before the first full B0 run (`scripts/run_b0.py`, `anygrasp_pp/robot.py`, derivations J).

The scene has the Menagerie Panda (pinned commit) at the origin with step 1's hand (4×4 rubber pads, force-clamped gripper at 70 N per pad), the table at z = 0, and the camera from D17. The bin is 52 cm from the pick area (the proposal said 40 cm), with a 40 × 40 cm interior and 10 cm walls, so the longest object (40 cm) fits and the bin stays out of the camera's crop.

A trial is one object at one shift (0, 2, 4, 6 or 8 cm) with one seed. The seed sets the yaw and a ±3 cm jitter of the object's most probable resting pose, and the same seed gives the same pose at every shift. The object settles for 1 s with the arm parked out of view, the D435 image is rendered, and GSNet runs on it. That's 10 objects × 5 shifts × 6 seeds = 300 trials.

B0 is the proposal's "best-scoring grasp that IK can reach". Going down the detector's list (collision-free, after NMS, best score first), it takes the first grasp that approaches within 30° of vertical, reports at most 8 cm width, sits on the object, and has IK solutions for the pre-grasp (10 cm back along the approach), the grasp, the lift (10 cm up) and the pose above the bin, where the wrist may turn about the vertical. Both closing directions of each grasp are tried. Whole-arm collision checking belongs to M1, not B0.

The controller is computed-torque feedforward plus joint impedance with libfranka's gains, clipped at the Panda's torque limits; it knows the arm and hand but not the payload. The arm reaches in 2.5 s (minimum jerk in joint space) and descends along the approach in 1.5 s. The gripper closes only once the hand is within 1 mm (waiting at most 1 s), at 0.1 m/s per finger until 70 N per pad, then squeezes for 0.5 s, exactly like step 1. Then a 10 cm lift in 1.2 s, a 2 s hold, a 2.5 s carry to the bin, lowering until the object is 4 cm above the floor, and release.

A trial succeeds if the object ends in the bin, wasn't dropped before release, and never turned more than 10° in the hand between the lift and the release (turning about the approach axis doesn't count). Failures are labelled by whatever happened first: no reachable grasp, no grip (blocked on approach, meaning the hand stopped more than 5 mm short, or closed on air), turned in hand, dropped, or missed the bin. The lever arm d is measured at the end of the squeeze from the pad midpoint, the closing axis and the true CoM (A.2).

## D19. First B0 run: the fingertip was too stiff

The first run (step-1 pad, 1.0 kg; `results/b0/trials_pad_v1.csv`) succeeded in 37 of 300 trials. By shift (0, 2, 4, 6, 8 cm) that's 9, 9, 3, 7 and 9 of 60; the no-shift rate of 15% is far from the proposal's sanity check of at least 80%. 223 objects turned in the hand, 8 dropped, 9 missed the bin, 15 were never gripped and 8 had no reachable grasp. The pipeline itself worked end to end and was deterministic: a rerun reproduced every outcome.

To understand it, I computed each executed grasp's own slip distance from the pad contacts it actually made at the end of the squeeze (positions and normal forces, A.12, load shared equally, A.5). The physics model predicted the outcomes: holding through the lift and hold agreed with |d| < d\* in 79% of trials, 86% held when it predicted a hold (|d|/d\* < 0.9), and only 19% held when |d|/d\* > 2. But d\* was tiny. The weaker pad touched at a median of 2 points on every object, even the flat-sided Duplo brick and bookend, and even on straight-down grasps, so the median d\* was 1.0 cm instead of 4.3 cm. B0's grasps sat a median 5.1 cm from the CoM even with no shift, so almost everything twisted.

The step-1 pad was too stiff to conform. It was calibrated to sink 0.11 mm on a perfectly aligned flat face, so any misalignment above atan(0.11/17 mm) ≈ 0.4° loaded only one edge row. A real 2 mm rubber layer under 70 N on one row compresses about ten times more and spreads the load; on a curved surface it forms a patch about 8 mm wide. Step 1 had already noticed that misaligned grasps slip early, and the full pipeline showed it dominates. So I recalibrated the pad as a thin rubber layer (D20) and re-ran B0, planning to re-pick the mass with D15's rule if the per-grasp slip distance still sat far from the middle of the shift range.

## D20. The fingertip as a thin rubber layer

The rubber layer of D4 (E = 3 MPa, h = 2 mm) is now modelled as a bed of independent springs, one per grid cell, each k_c = E·A_cell/h = 27.1 kN/m (B.6). In MuJoCo, a constant impedance (solimp 0.95 0.95) and a direct stiffness (solref −5537.89 −148.834) make each pad point a linear spring of K/((1−d)Â) (B.5). I set the stiffness so that a free 1 kg object, the study's condition, sees exactly k_c per point, which gives a flat sink of 0.161 mm at 70 N. The step-1 setting is kept as `CONTACT_V1`.

I checked it against the thin-layer solution with nothing refitted, pressing at 70 N against a fixed object (`scripts/calibrate_pad_layer.py`, `results/pad/layer.json`, check B.6):

| case | step-1 pad: points / d\* | new pad: points / d\* | thin-layer theory |
|---|---|---|---|
| flat | 16 / 4.47 cm | 16 / 4.47 cm | 16 / 4.47 cm |
| misaligned 1° | 12 / 3.08 cm | 16 / 3.52 cm | 16 / 3.52 cm |
| misaligned 2° | 8 / 2.70 cm | 12 / 2.95 cm | 12 / 2.95 cm |
| misaligned 4° | 4 / 2.38 cm | 8 / 2.68 cm | 8 / 2.67 cm |
| cylinder, R = 15 mm | 8 / 3.43 cm | 8 / 3.43 cm | 8 / 3.40 cm |

Per-point forces match the theory within 0.05 N of 70 N. Both pads behave as spring beds at their own stiffness; the old one was just about 2.2 times stiffer than the physical layer (60 vs 27 kN/m per point).

On B0 at 1.0 kg (`results/b0/trials.csv`) it barely changed the success rate (34 of 300, against 37), but the contact patch grew: the weaker pad's median contact count went from 2 to 4 and the median per-grasp d\* from 1.0 to 1.9 cm. The physics also predicted each grasp better, agreeing with the outcome in 86% of trials (up from 79%). The held fractions by |d|/d\* were 25/31 below 0.5, 12/15 at 0.5–0.9, 2/9 at 0.9–1.1, 9/28 at 1.1–2 and 12/175 above 2, with a fitted 50% point at 0.87 against the model's 1.

Tracing six trials showed where the contacts land: mostly on the pad's lower rows and middle columns, because GSNet approaches 7–28° off vertical, which tilts the pad against vertical sides, and because the objects' sides are curved, studded or tapered. Real grasps on these objects make small patches; it isn't a modelling artefact. I re-ran the step-1 calibration (R.1–R.3) with this pad as the nominal one.

## D21. Object mass re-chosen with the measured slip distance: 0.45 kg

D15's rule is that the typical slip distance should sit in the middle of the 0–8 cm shift range, so that grasps hold on one side and fail on the other. D15 applied it with the flat-face d\* (4.47 cm at 1 kg). The full pipeline showed that real grasps form small patches, with a median per-grasp d\* of 1.8 cm at 1 kg. Applying the same rule to that number (d\* scales as 1/m) gives 1.8 cm × 1 kg / m ≈ 4 cm, so m ≈ 0.45 kg, an ordinary household weight (the YCB hammer is 0.67 kg). The hidden cores scale with the mass; the CoM positions, resting poses and grasps don't change. The 1.0 kg runs stay on record as a heavy condition.

The power drill can't come along. At 0.45 kg an 8 cm shift needs 82% of its mass in the core, and D16's rule (at least 100 g of shell, set before any run) caps that at 78%. I kept the rule, so the drill appears only in the 1 kg condition, and the 0.45 kg run has 9 objects × 5 shifts × 6 seeds = 270 trials, on the same poses and noise as the 1 kg run so the two can be compared trial by trial. `run_b0.py` skips any object whose core is infeasible at the run's mass.

The finding for the midpoint: the proposal assumed B0 would grasp near the visible centroid with a flat-face d\* of about 4.7 cm, and set "B0 ≥ 80% with no shift" as a sanity check. Instead, B0's grasps sit a median 5 cm from the true CoM even with no shift, because a shape-only detector spreads its grasps along the object, and their contact patches are small. So the hidden shift changes B0's success much less than expected; some grasps end up closer to the CoM and others farther. What does predict each grasp's outcome is the physics model applied to that grasp's own contacts, which is exactly what M2 needs for re-ranking.

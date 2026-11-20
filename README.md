# AnyGrasp++: checking the weight before trusting the grasp

Final project for DA434 (Embodied Intelligence), Dawit Chun.

Grasp detectors like AnyGrasp and GSNet pick grasps from the object's shape. That works until the weight sits somewhere the shape doesn't show: a hammer head, a drill battery, a half-full bottle. Then a grasp that looks fine twists out of the gripper as soon as the object leaves the table.

AnyGrasp++ leaves the detector alone and adds three steps around it:

- **M1** keeps only grasps the arm can actually reach without hitting anything.
- **M2** scores each grasp by how likely it is to hold, using a friction model of the fingertips and the current guess of where the centre of mass is.
- **M3** does a short test lift, reads the object's mass and centre of mass from the arm's joint torques, and regrasps if the first grasp looks unsafe.

Everything runs in MuJoCo with a simulated Franka Panda.

## Where it stands

**Step 1, contact model (done).** I calibrated the gripper and fingertips so the simulator's slip behaviour matches the friction model the scoring relies on. The fingertip is modelled as a thin rubber layer and checked against layer theory.

**Step 2, scene and objects (done).** The scene has a Panda arm, a table, a bin and an overhead depth camera that mimics a RealSense D435, noise and shadows included. There are 10 test objects from the Google Scanned Objects and YCB sets, each with a hidden dense core that shifts its centre of mass by 0–8 cm without changing how it looks. I picked them from all 1108 scans with fixed rules: the core has to be physically possible, the object has to rest stably, and GSNet has to find grasps on it.

**Step 3, baseline (done for the midpoint).** B0 takes GSNet's best grasp that the arm can reach and executes it: grasp, lift, hold, carry, drop in the bin. With 0.45 kg objects it succeeds in 36% of 270 trials; with 1 kg objects, 16% of 300. Almost every failure is the object twisting out of the fingers. GSNet's grasps land about 5 cm from the true centre of mass, and on real-shaped objects the fingertips only touch a small patch.

The useful part is that the friction model predicts these outcomes grasp by grasp. Each grasp's slip distance can be computed from the patch it actually touches. Objects held when the lever arm was below that distance and fell when it was above: the fitted 50% point is at 1.08 times the predicted slip distance (95% CI 0.62–1.69), and the prediction matches the outcome in 80% of trials.

**Step 4, re-ranking (started).** To see how much a better choice of grasp could help, I executed each scene's top 10 reachable grasps. Picking the best of them would succeed 64% of the time, against B0's 36%. My first M2, scored from physics alone, doesn't beat B0, because predicting the contact patch from a noisy depth image is hard. A small calibrated model over the same physical quantities reaches 43% on objects it wasn't trained on. Frozen and run on 270 fresh scenes, it gets 40% against B0's 33% (+7 points, 95% CI 0 to 14), with no loss when nothing is shifted. A blind retry alone would gain about as much, which is why M3 matters.

**Step 5 (next).** M1 and M3, the "B0 + blind retry" control, stress tests, and the final comparison on fresh scenes.

**Project page.** Open `page/index.html` in a browser for side-by-side rollout videos (B0 against the re-ranked grasp on the same scene), the results, an explorer for every development scene and the object gallery. `python scripts/build_page.py` rebuilds it from `results/midpoint/`.

The reasoning behind every change from the proposal is in `docs/decisions.md`. Every equation the code uses is derived in `docs/derivations.pdf`, and `scripts/check_derivations.py` tests each one (89 checks).

## Running it

You need conda, ffmpeg for the videos and, for the detector, a Linux machine with an NVIDIA GPU.

```bash
conda create -n da434 python=3.12 && conda activate da434
pip install -r requirements.txt
bash scripts/setup_assets.sh        # downloads the Franka model at a pinned commit
python scripts/check_derivations.py # about 5 minutes, should print 89/89
```

The detector weights can't be redistributed, so download them yourself and put them in `assets/weights/`:

- GSNet: `minkuresunet_realsense.tar` from github.com/graspnet/graspness_unofficial
- graspnet-baseline (backup): `checkpoint-rs.tar` from github.com/graspnet/graspnet-baseline

`setup_assets.sh` verifies their checksums.

### The GPU part

GSNet needs CUDA and MinkowskiEngine, so I run it on a rented GPU (tested on an RTX 3080 with Ubuntu 24.04). Copy `gpu/` to the machine and the weights to `/root/weights`, then run `bash setup_gpu.sh`. The first run compiles everything in about 15 minutes. If you keep the wheels it builds, `bash setup_gpu.sh <wheel folder>` sets up a fresh machine in about 5 minutes. `python example_check.py make`, then `detect_grasps.py`, then `example_check.py check` should end with `SANITY PASS`.

### The baseline runs

These are heavy, so I run them on the same rented machine (it has many cores):

```bash
MUJOCO_GL=egl python scripts/run_b0.py scenes --out results/b0
python gpu/detect_grasps.py --detector gsnet --scenes results/b0/scenes --out results/b0/cache
python scripts/run_b0.py run --out results/b0
python scripts/analyze_b0.py --trials results/b0/trials.csv --out results/b0
```

Set `B0_MASS=0.45` for the nominal object mass. Without it the objects weigh 1 kg.

### Midpoint results, figures and videos

```bash
python scripts/run_b0.py candidates --out results/b0_m045_v2 --k 10      # each scene's first 10 reachable grasps
python scripts/eval_rerank.py --run results/b0_m045_v2                    # M2 features
python scripts/midpoint_results.py --test results/m045_fresh              # every policy, with object-bootstrap CIs
python scripts/make_figures.py
MUJOCO_GL=egl python scripts/render_rollouts.py                           # needs ffmpeg
MUJOCO_GL=egl python scripts/render_gallery.py
python scripts/build_page.py
```

Fresh scenes for a held-out check use `--seeds`, e.g. `run_b0.py scenes --out results/m045_fresh --seeds 6:12`.

## What's where

- `anygrasp_pp/` is the library: the friction physics, the contact rig from step 1, the hidden-weight objects, the scene and camera, and the robot and its controller.
- `scripts/` holds everything that produces a result: calibration, object selection, the baseline trials and the analysis.
- `gpu/` holds the detector setup and the wrapper that runs it once per scene.
- `docs/` holds the decision log and the derivations.
- `results/` holds generated tables, CSVs and figures.

## Third-party parts

- Franka model from MuJoCo Menagerie (Apache-2.0), downloaded by the setup script.
- Google Scanned Objects (Downs et al., ICRA 2022, CC-BY 4.0), using K. Zakka's MuJoCo conversions (MIT).
- YCB object set (Calli et al., 2015, CC-BY 4.0).
- CoACD for collision shapes (MIT).
- GSNet and graspnet-baseline are licensed for noncommercial research only. The setup script clones them at fixed commits; their weights and compiled packages are not in this repository.
- MinkowskiEngine and graspnetAPI (MIT).

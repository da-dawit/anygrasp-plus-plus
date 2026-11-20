# AnyGrasp++: checking the weight before trusting the grasp

DA434 (Embodied Intelligence) final project, Dawit Chun.

A grasp detector picks objects from shape alone, so it fails when the weight sits where the shape does not show it (a hammer head, a half-full bottle). AnyGrasp++ keeps the detector frozen and adds three steps:

- **M1 (reach filter):** inverse kinematics and whole-arm collision checks.
- **M2 (physics score):** how likely each grasp is to hold, given the centre-of-mass estimate.
- **M3 (test lift):** a short lift that reads the object's mass and centre of mass from the arm's joint torques, then regrasps if the grasp looks unsafe.

Everything runs in MuJoCo with a Franka Panda.

## Status

| Step | What | State |
|---|---|---|
| 1 | Contact calibration: gripper, fingertip and contact model checked against the slip physics | done |
| 2 | Full arm, table, bin, camera, hidden-weight objects | next |
| 3 | GSNet detector (cached per scene) and baseline B0 | detectors running and checked (D14); B0 next |
| 4 | M1–M3 and the B0 + blind-retry control | |
| 5 | Stress tests and evaluation | |

## Layout

```
anygrasp_pp/          library
  physics.py          slip distance, required grip, patch limit surface, P(hold), margin (derivations A, C)
  contact_rig.py      floating Franka hand + hidden-weight object; pad and contact models (B)
  rig_trial.py        one grasp-lift-hold trial with full contact monitoring; twist-onset search
scripts/
  calibrate_contact.py    step 1 sweeps and the parameter check (appendix R.1, R.2)
  characterize_contact.py non-ideal grasps: yaw, offsets, curvature, tilt, lift speed (R.3)
  panda_m3_numbers.py     test-lift accuracy and tare numbers from the real Panda arm (E.2, E.5)
  check_derivations.py    independent checks of every derivation (69 checks)
  setup_assets.sh         downloads the robot model; verifies the detector weights
gpu/                      runs on a rented Linux GPU, not on the Mac
  setup_gpu.sh            pinned detector environment (full build, or reuse saved wheels)
  detect_grasps.py        GSNet or graspnet-baseline, once per scene, cached
  example_check.py        end-to-end check on the authors' example scene
  requirements-gpu.lock.txt  exact package versions of the checked environment
docs/
  derivations.tex/.pdf    every equation, derived step by step, with the check that tests it
  decisions.md            every change from the proposal, with the reason and the evidence
results/                  generated tables, CSVs and figures (the PDF reads its tables from here)
```

## Setup

```bash
conda create -n da434 python=3.12 && conda activate da434
pip install -r requirements.txt
bash scripts/setup_assets.sh          # MuJoCo Menagerie Franka model, pinned commit
```

**Detector weights** (needed from step 3) are not in this repository; their licence forbids redistribution. Download them in a browser and put them in `assets/weights/`:

- GSNet, `minkuresunet_realsense.tar`, from github.com/graspnet/graspness_unofficial (Model Weights section)
- graspnet-baseline, `checkpoint-rs.tar`, from github.com/graspnet/graspnet-baseline (backup detector)

`scripts/setup_assets.sh` checks their SHA-256 sums if they are present.

## Detector machine (GPU)

The detectors need CUDA and MinkowskiEngine, so they run on a rented Linux GPU (tested: RTX 3080, Ubuntu 24.04, NVIDIA driver 580). Copy `gpu/` and the two weight files to `/root/app` and `/root/weights`, then:

```bash
bash setup_gpu.sh                    # first time: builds everything (~15 min, mostly the MinkowskiEngine compile)
bash setup_gpu.sh /root/wheels_saved # later: reuse the saved wheels from assets/gpu_wheels/ (~5 min, no compiling)
python example_check.py make && \
  python detect_grasps.py --detector gsnet --scenes /root/run/scenes --out /root/run/cache && \
  python example_check.py check      # expects SANITY PASS (reference: results/detector/example_check.txt)
```

The saved wheels run on RTX 30xx/40xx, A-series, A100 and L4/L40 cards (sm_80/86/89). Scene clouds go in as `<name>.npz` in the OpenCV camera frame; see the docstring of `detect_grasps.py`.

## Reproduce step 1

```bash
python scripts/check_derivations.py       # ~5 min, expects 69/69
python scripts/panda_m3_numbers.py        # seconds
python scripts/calibrate_contact.py       # ~15 min
python scripts/characterize_contact.py --contact rubber   # ~15 min
python scripts/characterize_contact.py --contact rigid
cd docs && tectonic -X compile derivations.tex
```

## Licences of third-party parts

- MuJoCo Menagerie (Franka models): Apache-2.0, downloaded by the setup script.
- GSNet and graspnet-baseline: academic or non-profit noncommercial research use only. They are used, not redistributed (sources are cloned at pinned commits by `gpu/setup_gpu.sh`; weights and compiled wheels stay out of git).
- MinkowskiEngine (GSNet authors' fork) and graspnetAPI: MIT.

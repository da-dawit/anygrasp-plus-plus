# AnyGrasp++

Grasp detectors like AnyGrasp pick a grasp from the object's shape alone, so they fail when the weight is somewhere the shape doesn't show (like the head of a hammer). AnyGrasp++ keeps the detector and adds a 2 cm test lift. If the grasp is slipping, it feels where the weight is, puts the object down and regrasps.

Final project for DA434 (Embodied Intelligence), Dawit Chun. Simulated in MuJoCo with a Franka Panda and GSNet as the detector.

**Project page with videos and results: https://da-dawit.github.io/anygrasp-plus-plus/**

## Midpoint results

270 new scenes, 9 objects with a hidden weight that moves the centre of mass by 0-8 cm, at most 2 grasps each:

| | success |
|---|---|
| B0 (GSNet's best reachable grasp) | 33% |
| B0 + blind retry | 42% |
| AnyGrasp++ | 50% |

For now the test lift uses an ideal wrist sensor. Next is reading the weight from the arm's joint torques.

## Running it

You need conda and ffmpeg. GSNet needs a Linux machine with an NVIDIA GPU.

```bash
conda create -n da434 python=3.12 && conda activate da434
pip install -r requirements.txt
bash scripts/setup_assets.sh    # Franka model
```

The GSNet weights can't be redistributed. Get `minkuresunet_realsense.tar` from github.com/graspnet/graspness_unofficial and put it in `assets/weights/`. On the GPU machine, copy `gpu/` over and run `bash setup_gpu.sh` (about 15 min the first time).

Main runs (I ran these on a rented 64-core GPU box):

```bash
export B0_MASS=0.45
MUJOCO_GL=egl python scripts/run_b0.py scenes --out results/b0_m045_v2
python gpu/detect_grasps.py --detector gsnet --scenes results/b0_m045_v2/scenes --out results/b0_m045_v2/cache
python scripts/run_b0.py run --out results/b0_m045_v2          # B0
python gpu/detect_service.py --queue /root/detq &               # GSNet for the regrasps
python scripts/run_m3.py --policy agpp --out results/m3_dev     # AnyGrasp++
python scripts/run_m3.py --policy retry --out results/m3_dev    # B0 + blind retry
python scripts/analyze_m3.py --m3 results/m3_dev
python scripts/make_figures.py && python scripts/build_page.py
```

For the fresh scenes, make them with `run_b0.py scenes --seeds 6:12 --out results/m045_fresh` and run the same steps on that folder (`--run results/m045_fresh` for `run_m3.py`).

## What's where

- `anygrasp_pp/` friction physics, objects, scene and camera, robot
- `scripts/` everything that makes a result
- `gpu/` detector setup
- `results/` CSVs, figures, videos
- `docs/decisions.md` every change from the proposal and why

## Credits

Franka model from MuJoCo Menagerie (Apache-2.0). Google Scanned Objects (CC BY 4.0) through K. Zakka's MuJoCo conversions (MIT). YCB objects (CC BY 4.0). CoACD (MIT). GSNet and graspnet-baseline are for noncommercial research only, so their weights and builds aren't in this repo. MinkowskiEngine and graspnetAPI (MIT).

#!/usr/bin/env bash
# Sets up both grasp detectors (GSNet + graspnet-baseline) on a rented Linux GPU box (tested: Vast.ai PyTorch image,
# Ubuntu 24.04, RTX 3080, driver 580). Idempotent: rerunning skips finished steps.
#
#   bash setup_gpu.sh                 # full build (about 15 min, dominated by the MinkowskiEngine compile)
#   bash setup_gpu.sh /path/wheels    # reuse wheels saved from an earlier build (about 5 min, no compiling)
#
# Pins: Python 3.10, PyTorch 2.1.2+cu118, CUDA 11.8 toolkit, gcc 11, numpy 1.23.5. The image's own CUDA/PyTorch stay
# untouched; everything lives in /root/envs/gn and /opt/cuda-11.8. Wheels target sm_80/86/89 (+PTX): A100/A30,
# RTX 30xx/A-series, RTX 40xx/L4/L40. Detector weights must already be in /root/weights (not redistributable).
set -euxo pipefail
WHEELS_IN=${1:-}
ENV=${ENV:-/root/envs/gn}; PY=$ENV/bin/python; CU=/opt/cuda-11.8; SRC=/root/src; W=${W:-/root/wheels}
export DEBIAN_FRONTEND=noninteractive

# 1. system compilers (CUDA 11.8 supports gcc <= 11), BLAS for MinkowskiEngine, parallel downloader
apt-get update -qq && apt-get install -y -qq gcc-11 g++-11 libopenblas-dev aria2 ninja-build

# 2. Python env + PyTorch (the wheel bundles its own CUDA 11.8 runtime)
source /opt/miniforge3/etc/profile.d/conda.sh
[ -x $PY ] || mamba create -y -q -p $ENV -c conda-forge python=3.10
uv pip install --python $PY torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
# nvidia-cusparse-cu11: MinkowskiEngine needs libcusparse.so.11, which this torch build does not bundle
uv pip install --python $PY numpy==1.23.5 ninja setuptools==69.5.1 wheel cython==0.29.37 \
  nvidia-cusparse-cu11==11.7.5.86 patchelf \
  open3d==0.18.0 scipy==1.10.1 transforms3d==0.4.1 trimesh==4.4.1 tqdm Pillow opencv-python-headless==4.8.1.78 \
  tensorboard h5py dill cvxopt autolab_core matplotlib==3.7.5

# 3. sources at the commits this was tested with
mkdir -p $SRC && cd $SRC
clone() { [ -d "$2" ] || git clone -q "https://github.com/$1.git" "$2"; git -C "$2" checkout -q "$3"; }
clone graspnet/graspnet-baseline       graspnet-baseline    280c215129f759ed8649cb4e89fc5dfee55f4f80
clone graspnet/graspness_unofficial    graspness_unofficial b5abf5aaaf3a797514f89161d8ccc8dfc2ec0eca
clone chenxi-wang/MinkowskiEngine      MinkowskiEngine      1f3ba4e5625eb5aae309008221ab2531e65f184d
clone graspnet/graspnetAPI             graspnetAPI          eb57dd2092d8dbe05312a29c3d0c22f3226efbfc
# graspnetAPI's install_requires pulls autolab-perception and more that inference never imports; install it bare
uv pip install --python $PY --no-deps -e $SRC/graspnetAPI
uv pip install --python $PY --no-build-isolation grasp_nms==1.0.2

if [ -n "$WHEELS_IN" ]; then
  # 4a. reuse compiled wheels (they carry their own CUDA kernels; no toolkit needed at run time)
  uv pip install --python $PY --no-deps --reinstall "$WHEELS_IN"/pointnet2-*.whl "$WHEELS_IN"/knn_pytorch-*.whl \
    "$WHEELS_IN"/MinkowskiEngine-*.whl
else
  # 4b. CUDA 11.8 toolkit from NVIDIA's runfile (conda kept resolving to 13.x), toolkit only, own directory
  if [ ! -x $CU/bin/nvcc ]; then
    mkdir -p /root/dl && cd /root/dl
    aria2c -q -c -x16 -s16 -k8M --file-allocation=none \
      https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    before=$(readlink -e /usr/local/cuda || true)   # empty if the image has no CUDA
    sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit --toolkitpath=$CU --override --no-man-page --tmpdir=/root/dl
    if [ -n "$before" ]; then ln -sfn "$before" /usr/local/cuda; fi   # keep the image's default CUDA
    rm -f cuda_11.8.0_520.61.05_linux.run
  fi
  export CUDA_HOME=$CU PATH=$CU/bin:$PATH CC=gcc-11 CXX=g++-11 TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX" MAX_JOBS=48
  mkdir -p $W
  # pointnet2 and knn from graspnet-baseline serve both detectors: GSNet's pointnet2 copy differs only in line
  # endings and comments, and GSNet's knn (still on the removed THC API) is used only for training labels.
  for d in pointnet2 knn; do
    (cd $SRC/graspnet-baseline/$d && rm -rf build dist ./*.egg-info && $PY setup.py -q bdist_wheel -d $W)
  done
  (cd $SRC/MinkowskiEngine && rm -rf build dist ./*.egg-info && \
    $PY setup.py bdist_wheel -d $W --force_cuda --blas=openblas --cuda_home=$CU)
  # point the backend at NVIDIA's pip copy of libcusparse.so.11 (relative path) and record the dependency
  rm -rf /root/dl/mew && mkdir -p /root/dl/mew/out && cd /root/dl/mew
  $PY -m wheel unpack $W/MinkowskiEngine-0.5.4-cp310-cp310-linux_x86_64.whl -d .
  $ENV/bin/patchelf --set-rpath '$ORIGIN/../nvidia/cusparse/lib' MinkowskiEngine-0.5.4/MinkowskiEngineBackend/_C*.so
  sed -i '0,/^Summary/s//Requires-Dist: nvidia-cusparse-cu11==11.7.5.86\n&/' \
    MinkowskiEngine-0.5.4/MinkowskiEngine-0.5.4.dist-info/METADATA
  $PY -m wheel pack MinkowskiEngine-0.5.4 -d out && mv out/*.whl $W/
  uv pip install --python $PY --no-deps --reinstall $W/pointnet2-*.whl $W/knn_pytorch-*.whl $W/MinkowskiEngine-*.whl
fi

cd /root && $PY -c "import torch, MinkowskiEngine as ME, pointnet2._ext, knn_pytorch, grasp_nms; \
from graspnetAPI import GraspGroup; print('READY torch', torch.__version__, 'ME', ME.__version__, 'cuda', torch.cuda.is_available())"

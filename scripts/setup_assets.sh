#!/usr/bin/env bash
# Downloads the MuJoCo Menagerie Franka model (pinned commit) and verifies the detector weights if present.
set -euo pipefail
cd "$(dirname "$0")/.."

MENAGERIE_COMMIT=370c43965323c0944b433b202aac5058b5f907a9
if [ ! -d assets/menagerie/.git ]; then
  mkdir -p assets
  git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git assets/menagerie
  git -C assets/menagerie sparse-checkout set franka_emika_panda
fi
git -C assets/menagerie fetch --quiet origin "$MENAGERIE_COMMIT" || true
git -C assets/menagerie checkout --quiet "$MENAGERIE_COMMIT"
echo "Menagerie at $(git -C assets/menagerie rev-parse --short HEAD)"

check() {  # file, expected sha256
  if [ -f "assets/weights/$1" ]; then
    got=$(shasum -a 256 "assets/weights/$1" | cut -d' ' -f1)
    [ "$got" = "$2" ] && echo "ok       $1" || { echo "MISMATCH $1"; exit 1; }
  else
    echo "missing  $1 (download it in a browser, see README)"
  fi
}
check minkuresunet_realsense.tar a0b9d85b300e40c76b66d092de62138d74757d6b7c2037f0d364f33960c54aa9
check checkpoint-rs.tar          60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868

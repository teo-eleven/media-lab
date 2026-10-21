#!/usr/bin/env bash
# Fetch Robust Video Matting (RVM) for the `matte-video` recipe.
#
# RVM is GPL-3.0 and is used as a source checkout on sys.path (never imported
# into media_lab's own process, never committed). torch/torchvision are already
# in .venv via kinocut[upscale], so there is no venv to build here.
#
#   tools/RobustVideoMatting/     - the checkout (gitignored)
#   $MEDIA_LAB_WEIGHTS_DIR/       - rvm_resnet50.pth, rvm_mobilenetv3.pth
#     (default: work/punto-edit/gen/weights/)
set -euo pipefail

RVM_REPO_URL="https://github.com/PeterL1n/RobustVideoMatting.git"
WEIGHTS_BASE="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0"
WEIGHTS=(rvm_resnet50.pth rvm_mobilenetv3.pth)

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$root/tools/RobustVideoMatting"
weights_dir="${MEDIA_LAB_WEIGHTS_DIR:-$root/work/punto-edit/gen/weights}"

echo "==> RVM checkout"
if [[ -f "$repo_dir/model/__init__.py" ]]; then
  echo "    already present: $repo_dir"
else
  mkdir -p "$(dirname "$repo_dir")"
  git clone --depth 1 "$RVM_REPO_URL" "$repo_dir"
fi

echo "==> weights -> $weights_dir"
mkdir -p "$weights_dir"
for w in "${WEIGHTS[@]}"; do
  if [[ -f "$weights_dir/$w" ]]; then
    echo "    have $w"
  else
    echo "    download $w"
    curl -fSL --progress-bar -o "$weights_dir/$w" "$WEIGHTS_BASE/$w"
  fi
done

echo
echo "done. RVM has no published checksums; the files come straight from the"
echo "PeterL1n/RobustVideoMatting v1.0.0 GitHub release."
echo "Verify with:  media-lab doctor"

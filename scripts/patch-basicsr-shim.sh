#!/usr/bin/env bash
# Restore the torchvision.transforms.functional_tensor shim that basicsr needs.
#
# kinocut[upscale] pulls basicsr, which imports
# `torchvision.transforms.functional_tensor` - a module removed in
# torchvision 0.17. Without the shim, `import basicsr` (and every `realesrgan`
# / `kino *upscale*` call, and docs/video-agent/pipeline/upscale_realesrgan.py)
# fails at import time. This writes a re-export module into the active venv so
# `make setup` is reproducible; the file lives under the gitignored .venv.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="$root/.venv/bin/python"

if [[ ! -x "$py" ]]; then
  echo "error: $py not found. Run 'uv sync --group dev' first." >&2
  exit 1
fi

transforms_dir="$("$py" -c 'import os, torchvision.transforms as t; print(os.path.dirname(t.__file__))')"
shim="$transforms_dir/functional_tensor.py"

if "$py" -c 'import torchvision.transforms.functional_tensor' 2>/dev/null; then
  echo "ok: torchvision.transforms.functional_tensor already importable"
  exit 0
fi

cat > "$shim" <<'PY'
# Shim: torchvision removed this module in 0.17; basicsr still imports it.
# Written by scripts/patch-basicsr-shim.sh. Safe to delete and regenerate.
from torchvision.transforms._functional_tensor import *  # noqa: F401,F403
from torchvision.transforms._functional_tensor import rgb_to_grayscale  # noqa: F401
PY

"$py" -c 'from basicsr.archs.rrdbnet_arch import RRDBNet' \
  && echo "done: wrote $shim (basicsr imports cleanly)"

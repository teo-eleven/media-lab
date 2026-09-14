"""RVM (Robust Video Matting) inference driver.

Runs as a CHILD PROCESS via `ml_runner` - the only place RVM (GPL-3.0) is
imported. Reads a directory of RGB frames, writes RGBA frames + a stats file.

Usage:
  python -m ... rvm_infer.py FRAMES_IN OUT_DIR MODEL WEIGHTS RVM_REPO
                             [--downsample R] [--alpha-lift L] [--alpha-gain G]

OUT_DIR gets f-0001.png ... (RGBA) and stats.json:
  {"model", "device", "frames": [{"frame", "alpha_mean", "alpha_var"}, ...],
   "stability_score": float}   # mean |Δalpha| over ever-foreground px, 0-255
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("frames_in")
    p.add_argument("out_dir")
    p.add_argument("model", choices=("resnet50", "mobilenetv3"))
    p.add_argument("weights")
    p.add_argument("rvm_repo")
    p.add_argument("--downsample", type=float, default=0.375)
    p.add_argument("--alpha-lift", type=float, default=0.0,
                   help="subtract this from alpha before gain (v23 used 18)")
    p.add_argument("--alpha-gain", type=float, default=1.0,
                   help="multiply alpha after the lift (v23 used 1.28)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)

    sys.path.insert(0, ns.rvm_repo)
    import numpy as np
    import torch
    from model import MattingNetwork  # RVM, from rvm_repo on sys.path
    from PIL import Image

    frames = sorted(glob.glob(os.path.join(ns.frames_in, "f-*.png")))
    if not frames:
        print(f"no frames matching f-*.png in {ns.frames_in}", file=sys.stderr)
        return 2
    os.makedirs(ns.out_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = MattingNetwork(ns.model).eval().to(device)
    net.load_state_dict(torch.load(ns.weights, map_location="cpu", weights_only=True))

    rec: list = [None] * 4
    per_frame = []
    prev_alpha = None
    max_alpha = None          # running per-pixel max, to find ever-foreground px
    sum_abs_delta = None      # running per-pixel sum of |alpha[t] - alpha[t-1]|
    t0 = time.time()

    for i, path in enumerate(frames, 1):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)
        with torch.no_grad():
            fgr, pha, *rec = net(x, *rec, downsample_ratio=ns.downsample)
        fg = (fgr[0].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        alpha = pha[0, 0].cpu().numpy().clip(0, 1) * 255.0
        if ns.alpha_lift or ns.alpha_gain != 1.0:
            alpha = np.clip((alpha - ns.alpha_lift) * ns.alpha_gain, 0, 255)
        a8 = alpha.astype(np.uint8)

        Image.fromarray(np.dstack([fg, a8]), "RGBA").save(
            os.path.join(ns.out_dir, f"f-{i:04d}.png")
        )
        per_frame.append(
            {"frame": i, "alpha_mean": float(a8.mean()), "alpha_var": float(a8.var())}
        )

        af = a8.astype(np.float32)
        if prev_alpha is None:
            max_alpha = af.copy()
            sum_abs_delta = np.zeros_like(af)
        else:
            np.maximum(max_alpha, af, out=max_alpha)
            sum_abs_delta += np.abs(af - prev_alpha)
        prev_alpha = af

    n_deltas = len(frames) - 1
    if n_deltas > 0 and max_alpha is not None and sum_abs_delta is not None:
        ever_fg = max_alpha > 10
        score = (
            float(sum_abs_delta[ever_fg].sum() / (n_deltas * ever_fg.sum()))
            if ever_fg.any()
            else 0.0
        )
    else:
        score = 0.0

    with open(os.path.join(ns.out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"model": ns.model, "device": device, "frames": per_frame,
             "stability_score": score},
            fh,
            indent=2,
        )
    print(f"rvm_infer: {len(frames)} frames, {ns.model} on {device}, "
          f"{time.time() - t0:.1f}s, stability_score {score:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

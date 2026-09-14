"""Real-ESRGAN upscaler for RGBA image sequences.

RGB channels are upscaled with RealESRGAN (RRDBNet), alpha channel is resized
with Lanczos. Preserves transparency.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
from realesrgan import RealESRGANer


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-ESRGAN sequence upscaler")
    parser.add_argument("frames_in", help="Directory containing source f-*.png frames")
    parser.add_argument("frames_out", help="Directory to write upscaled frames")
    parser.add_argument("weights", help="Path to RealESRGAN_x2plus.pth or RealESRGAN_x4plus.pth")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4], help="Upscale factor")
    parser.add_argument("--tile", type=int, default=512, help="Tile size (0 for no tiling)")
    parser.add_argument("--tile-pad", type=int, default=16, help="Tile padding")

    args = parser.parse_args()

    frames = sorted(glob.glob(os.path.join(args.frames_in, "f-*.png")))
    if not frames:
        frames = sorted(glob.glob(os.path.join(args.frames_in, "*.png")))
    if not frames:
        sys.stderr.write(f"No PNG frames found in {args.frames_in}\n")
        sys.exit(1)

    os.makedirs(args.frames_out, exist_ok=True)

    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"

    num_block = 23
    num_grow_ch = 32
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=num_block,
        num_grow_ch=num_grow_ch,
        scale=args.scale,
    )
    try:
        upsampler = RealESRGANer(
            scale=args.scale,
            model_path=args.weights,
            model=model,
            tile=args.tile,
            tile_pad=args.tile_pad,
            pre_pad=0,
            half=False,
            device=dev,
        )
    except Exception as exc:  # noqa: BLE001
        if dev != "cpu":
            sys.stderr.write(f"RealESRGANer init failed on {dev} ({exc}), falling back to CPU...\n")
            dev = "cpu"
            upsampler = RealESRGANer(
                scale=args.scale,
                model_path=args.weights,
                model=model,
                tile=args.tile,
                tile_pad=args.tile_pad,
                pre_pad=0,
                half=False,
                device="cpu",
            )
        else:
            raise

    t0 = time.time()
    for i, frame_path in enumerate(frames):
        im = Image.open(frame_path).convert("RGBA")
        arr = np.asarray(im)
        rgb = arr[..., :3][:, :, ::-1]  # BGR for RealESRGAN
        a = arr[..., 3]

        try:
            out_bgr, _ = upsampler.enhance(rgb, outscale=args.scale)
        except Exception as exc:  # noqa: BLE001
            if dev != "cpu":
                sys.stderr.write(
                    f"RealESRGAN enhance failed on {dev} ({exc}), falling back to CPU...\n"
                )
                dev = "cpu"
                model_cpu = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=num_block,
                    num_grow_ch=num_grow_ch,
                    scale=args.scale,
                )
                upsampler = RealESRGANer(
                    scale=args.scale,
                    model_path=args.weights,
                    model=model_cpu,
                    tile=args.tile,
                    tile_pad=args.tile_pad,
                    pre_pad=0,
                    half=False,
                    device="cpu",
                )
                out_bgr, _ = upsampler.enhance(rgb, outscale=args.scale)
            else:
                raise

        out_rgb = out_bgr[:, :, ::-1]

        target_w, target_h = out_rgb.shape[1], out_rgb.shape[0]
        a_up = np.asarray(Image.fromarray(a).resize((target_w, target_h), Image.LANCZOS))

        res = np.dstack([out_rgb, a_up]).astype(np.uint8)
        out_name = os.path.basename(frame_path)
        Image.fromarray(res, "RGBA").save(os.path.join(args.frames_out, out_name))

        if (i + 1) % 15 == 0 or (i + 1) == len(frames):
            print(f"{i + 1}/{len(frames)} frames upscaled ({time.time() - t0:.1f}s)", flush=True)

    print(f"DONE {len(frames)} frames in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

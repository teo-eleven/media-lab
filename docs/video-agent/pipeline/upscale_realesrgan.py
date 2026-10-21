"""Real-ESRGAN x2 upscale of the isnet cutout frames (RGB), alpha via Lanczos.

in : work/punto-edit/isnet/cut/f-*.png   (720x1280 RGBA)
out: work/punto-edit/isnet/up/f-*.png    (1440x2560 RGBA)
"""
import glob
import os
import time

import numpy as np
import torch
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

SRC = sorted(glob.glob("work/punto-edit/isnet/cut/f-*.png"))
OUT = "work/punto-edit/isnet/up"
os.makedirs(OUT, exist_ok=True)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
up = RealESRGANer(
    scale=2,
    model_path="work/punto-edit/gen/weights/RealESRGAN_x2plus.pth",
    model=model,
    tile=512,
    tile_pad=16,
    pre_pad=0,
    half=False,
    device=dev,
)

t0 = time.time()
for i, f in enumerate(SRC):
    im = Image.open(f).convert("RGBA")
    rgb = np.asarray(im)[..., :3][:, :, ::-1]  # to BGR for realesrgan
    a = np.asarray(im)[..., 3]
    out_bgr, _ = up.enhance(rgb, outscale=2)
    out_rgb = out_bgr[:, :, ::-1]
    a_up = np.asarray(Image.fromarray(a).resize((out_rgb.shape[1], out_rgb.shape[0]), Image.LANCZOS))
    res = np.dstack([out_rgb, a_up]).astype(np.uint8)
    Image.fromarray(res, "RGBA").save(os.path.join(OUT, os.path.basename(f)))
    if i % 15 == 0:
        print(f"{i}/{len(SRC)}  {time.time()-t0:.0f}s", flush=True)
print(f"DONE {len(SRC)} in {time.time()-t0:.0f}s", flush=True)

"""Per-frame foot detection + baked contact shadow under the isnet cutout.

For each RGBA cutout frame: find the foot contact point (lowest opaque row,
x = centroid of opaque pixels near the bottom), draw a soft dark ellipse on a
layer *under* the person, and write subj+shadow PNGs. The shadow then moves
with her feet through the walk cycle, which is what actually reads as
"walking on the ground".
"""

import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SRC = sorted(glob.glob("work/punto-edit/isnet/cut/f-*.png"))
OUT = "work/punto-edit/isnet/shad"
os.makedirs(OUT, exist_ok=True)

ALPHA_THRESH = 40
# shadow ellipse size in source pixels (720x1280 frame)
SH_W = 190
SH_H = 54

for f in SRC:
    im = Image.open(f).convert("RGBA")
    w, h = im.size
    a = np.asarray(im)[..., 3]
    ys, xs = np.where(a > ALPHA_THRESH)
    if len(ys) == 0:
        im.save(os.path.join(OUT, os.path.basename(f)))
        continue
    foot_y = int(ys.max())
    band = ys > (foot_y - 60)
    foot_x = int(np.median(xs[band])) if band.any() else int(np.median(xs))

    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(shadow)
    # tight dark contact AO
    d.ellipse(
        [foot_x - SH_W // 2, foot_y - SH_H // 2 + 6,
         foot_x + SH_W // 2, foot_y + SH_H // 2 + 6],
        fill=(0, 0, 0, 235),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(9))
    # wider soft falloff
    soft = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(soft)
    d2.ellipse(
        [foot_x - SH_W, foot_y - SH_H + 8,
         foot_x + SH_W, foot_y + SH_H + 8],
        fill=(0, 0, 0, 110),
    )
    soft = soft.filter(ImageFilter.GaussianBlur(22))

    base = Image.alpha_composite(soft, shadow)
    out = Image.alpha_composite(base, im)
    out.save(os.path.join(OUT, os.path.basename(f)))

print(f"baked {len(SRC)} frames -> {OUT}")

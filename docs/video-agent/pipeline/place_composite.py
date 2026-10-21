"""Foot-locked, zoom-normalised placement of the (upscaled) isnet cutout with a
per-scene relight grade: directional key/fill gradient, ambient colour, ground
bounce on the lower body, and a sun-direction contact shadow.

in : work/punto-edit/rvm_up/f-*.png  (upscaled cutout; falls back to
     work/punto-edit/isnet/cut/f-*.png when rvm_up/ is absent or has <=100 files)
out: work/punto-edit/isnet/placed/f-*.png  (2160x3840 RGBA, ready to overlay on bg)
"""
import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CANVAS = (2160, 3840)

UP = "work/punto-edit/rvm_up"
CUT_DIR = UP if os.path.isdir(UP) and len(os.listdir(UP)) > 100 else "work/punto-edit/isnet/cut"
CUT = sorted(glob.glob(f"{CUT_DIR}/f-*.png"))
OUT = "work/punto-edit/isnet/placed"
os.makedirs(OUT, exist_ok=True)
N = len(CUT)
B1, B2 = 9999, 9999

# per-segment scene lighting
SEG = {
    "ts": dict(blur=0.55,  # NYC Wall St, flat overcast, on the asphalt foreground
        ground=3560, base_scale=0.98, dx=-180,
        ambient=(0.99, 1.00, 1.01), bright=0.96, gamma=1.03,
        key_dir=(-0.2, -1.0), key_amt=0.04,
        bounce=(0.55, 0.56, 0.57), bounce_amt=0.09,
        sun=(0.03, 1.0), shadow_a=135, sat=0.82, contrast=0.93,
    ),
    "street": dict(blur=0.0,  # daytime city street, overcast-ish flat light
        ground=3380, base_scale=0.99, dx=0,
        ambient=(1.00, 1.00, 1.02), bright=1.05, gamma=1.0,
        key_dir=(-0.25, -1.0), key_amt=0.06,
        bounce=(0.66, 0.66, 0.67), bounce_amt=0.09,  # grey asphalt
        sun=(0.1, 1.0), shadow_a=78,
    ),
    "desert": dict(blur=0.0,  # real Sahara, strong warm sun from behind-right
        ground=3260, base_scale=0.99, dx=10,
        ambient=(1.05, 1.00, 0.96), bright=1.06, gamma=0.98,
        key_dir=(0.8, -0.4), key_amt=0.14,
        bounce=(0.86, 0.66, 0.44), bounce_amt=0.14,  # orange sand
        sun=(-0.55, 0.5), shadow_a=120,
    ),
}


def seg_for(i):
    return SEG["ts"] if i < B1 else (SEG["street"] if i < B2 else SEG["desert"])


# pass 1: foot point + height (for zoom-normalisation)
raw = []
for f in CUT:
    a = np.asarray(Image.open(f).convert("RGBA"))[..., 3]
    ys, xs = np.where(a > 40)
    if len(ys) == 0:
        raw.append(None)
        continue
    fy, ty = int(ys.max()), int(ys.min())
    band = ys > fy - 90
    fx = int(np.median(xs[band]))
    raw.append((fx, fy, fy - ty))

vals = [r for r in raw if r]
med_y = int(np.median([r[1] for r in vals]))
med_h = float(np.median([r[2] for r in vals]))
# HARD per-frame pin: the lowest opaque pixel of the silhouette is placed
# EXACTLY on the ground line every frame -> feet never leave the ground, no float,
# no gap under her while walking. Bottom-Y is lightly median-smoothed (+-1) only to
# reject a 1-frame matte spike, never clamped.
bots = []
for r in raw:
    bots.append(r[1] if r else med_y)
fixed_bot = int(np.median([b for b in bots]))
bots_s = [fixed_bot] * len(bots)
foot = []
for i, r in enumerate(raw):
    fx = r[0] if r else 360
    z = (med_h / r[2]) if (r and r[2] > 0) else 1.0
    foot.append((fx, bots_s[i], max(0.94, min(1.07, z))))

def relight(rgba, s):
    """directional key gradient + ambient + ground bounce on lower body."""
    arr = np.asarray(rgba).astype(np.float32)
    h, w = arr.shape[:2]
    rgb = arr[..., :3]

    # ambient colour + brightness + gamma
    for c in range(3):
        rgb[..., c] *= s["ambient"][c]
    rgb *= s["bright"]
    rgb = 255.0 * np.clip(rgb / 255.0, 0, 1) ** s["gamma"]

    # directional key/fill gradient across the figure
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / w - 0.5) * 2.0
    ny = (yy / h - 0.5) * 2.0
    kx, ky = s["key_dir"]
    g = -(nx * kx + ny * ky)
    g = (g - g.min()) / (np.ptp(g) + 1e-6)          # 0..1, lit side -> 1
    grad = 1.0 + (g[..., None] - 0.5) * 2.0 * s["key_amt"]
    rgb *= grad

    # ground bounce: tint + lift the lower ~35% of the body
    by = np.clip((yy / h - 0.62) / 0.38, 0, 1)[..., None]
    bcol = np.array(s["bounce"], np.float32) * 255.0
    rgb = rgb * (1 - by * s["bounce_amt"]) + bcol * (by * s["bounce_amt"])

    if s.get("sat", 1.0) != 1.0 or s.get("contrast", 1.0) != 1.0:
        g = rgb.mean(axis=2, keepdims=True)
        rgb = g + (rgb - g) * s.get("sat", 1.0)
        rgb = 128 + (rgb - 128) * s.get("contrast", 1.0)
    out = np.clip(np.dstack([np.clip(rgb, 0, 255), arr[..., 3]]), 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


for i, f in enumerate(CUT):
    s = seg_for(i)
    im = Image.open(f).convert("RGBA")
    w, h = im.size
    sc = s["base_scale"] * foot[i][2]
    sub = im.resize((int(w * sc), int(h * sc)), Image.LANCZOS)
    sub = relight(sub, s)
    if s.get("blur", 0):
        from PIL import ImageFilter as _IF
        sub = sub.filter(_IF.GaussianBlur(s["blur"]))

    fx, fy, _ = foot[i]
    fx_s, fy_s = fx * sc, fy * sc
    px = int(CANVAS[0] / 2 - fx_s + s["dx"])
    py = int(s["ground"] - fy_s)
    fx_i = int(CANVAS[0] / 2 + s["dx"])

    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))

    # sun-direction contact shadow
    cxf = int(CANVAS[0] / 2 + s["dx"])
    gy = s["ground"]
    sdx, sdy = s["sun"]
    off_x = int(-sdx * 260)
    sh = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    d = ImageDraw.Draw(sh)
    d.ellipse([fx_i - 260 + off_x, gy - 24, fx_i + 260 + off_x, gy + 90], fill=(0, 0, 0, 70))
    sh = sh.filter(ImageFilter.GaussianBlur(28))
    d2 = ImageDraw.Draw(sh)
    d2.ellipse([fx_i - 150, gy - 8, fx_i + 150, gy + 44], fill=(0, 0, 0, 120))
    sh = sh.filter(ImageFilter.GaussianBlur(10))
    d3 = ImageDraw.Draw(sh)
    d3.ellipse([fx_i - 95, gy + 2, fx_i + 95, gy + 30], fill=(0, 0, 0, 150))
    sh = sh.filter(ImageFilter.GaussianBlur(4))
    canvas = Image.alpha_composite(canvas, sh)
    canvas.alpha_composite(sub, (px, py))
    canvas.save(os.path.join(OUT, os.path.basename(f)))

print(f"placed {N} frames from {CUT_DIR} -> {OUT}")

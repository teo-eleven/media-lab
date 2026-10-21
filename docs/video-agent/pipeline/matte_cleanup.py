import glob, os
import numpy as np
from PIL import Image
from scipy import ndimage

SRC = sorted(glob.glob("work/punto-edit/isnet/up/f-*.png"))
OUT = "work/punto-edit/isnet/up_clean"
os.makedirs(OUT, exist_ok=True)
N = len(SRC)
WIN = 3
MAX_HOLE_PX = 16000

alphas = np.stack([np.asarray(Image.open(f).convert("RGBA"))[..., 3] for f in SRC])
print(f"loaded {N}", flush=True)

for i, f in enumerate(SRC):
    lo, hi = max(0, i - WIN), min(N, i + WIN + 1)
    a = np.median(alphas[lo:hi], axis=0).astype(np.uint8)
    binm = a > 35
    closed = ndimage.binary_closing(binm, iterations=3)
    holes = ndimage.binary_fill_holes(closed) & ~closed
    lbl, n = ndimage.label(holes)
    small = np.zeros_like(holes)
    for k in range(1, n + 1):
        m = lbl == k
        if int(m.sum()) < MAX_HOLE_PX:
            small |= m
    a2 = a.astype(np.uint16)
    a2[small] = 205
    a2 = ndimage.gaussian_filter(a2.astype(np.float32), sigma=1.4)
    a2 = np.clip(a2, 0, 255).astype(np.uint8)
    im = np.asarray(Image.open(f).convert("RGBA")).copy()
    im[..., 3] = a2
    Image.fromarray(im, "RGBA").save(os.path.join(OUT, os.path.basename(f)))
    if i % 30 == 0:
        print(f"{i}/{N}", flush=True)
print("DONE", flush=True)

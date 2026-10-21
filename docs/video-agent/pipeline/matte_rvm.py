import sys, time, glob, os
sys.path.insert(0, "/tmp/RVM")
import torch
from model import MattingNetwork
from PIL import Image
import numpy as np

dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = MattingNetwork("resnet50").eval().to(dev)
m.load_state_dict(torch.load("work/punto-edit/gen/weights/rvm_resnet50.pth", map_location="cpu"))

files = sorted(glob.glob("work/punto-edit/isnet/src/f-*.png"))
OUT = "work/punto-edit/rvm"; os.makedirs(OUT, exist_ok=True)
rec = [None]*4
t0 = time.time()
for i, f in enumerate(files):
    im = np.array(Image.open(f).convert("RGB"))
    x = torch.from_numpy(im.copy()).permute(2,0,1).unsqueeze(0).float().div(255).to(dev)
    with torch.no_grad():
        fgr, pha, *rec = m(x, *rec, downsample_ratio=0.375)
    fg = (fgr[0].permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8)
    a = (pha[0,0].cpu().numpy().clip(0,1)*255).astype(np.uint8)
    # slight alpha contrast to cut green/edge spill
    a = np.clip((a.astype(np.int16)-18)*1.28, 0, 255).astype(np.uint8)
    Image.fromarray(np.dstack([fg, a]), "RGBA").save(f"{OUT}/f-{i+1:04d}.png")
    if i % 40 == 0: print(i, flush=True)
print(f"DONE {len(files)} in {time.time()-t0:.0f}s", flush=True)

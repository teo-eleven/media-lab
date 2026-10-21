from rembg import remove, new_session
from PIL import Image
import glob, os, time

sess = new_session("isnet-general-use")
files = sorted(glob.glob("work/punto-edit/isnet/src/f-*.png"))
t0 = time.time()
for i, f in enumerate(files):
    o = f.replace("/src/", "/cut/")
    if os.path.exists(o):
        continue
    im = Image.open(f).convert("RGBA")
    cut = remove(im, session=sess, post_process_mask=True)
    Image.alpha_composite(Image.new("RGBA", im.size, (0, 0, 0, 0)), cut).save(o)
    if i % 20 == 0:
        print(f"{i}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
print(f"DONE {len(files)} frames in {time.time()-t0:.0f}s", flush=True)

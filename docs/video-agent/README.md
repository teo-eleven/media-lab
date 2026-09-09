# Video-agent toolkit

Reusable pieces for a future **local video/photo create-and-edit agent**. Built while
compositing one clip (`videoclip punto.mp4`: a person holding a handwritten sign,
placed into new locations). No media lives here — only the code, the recipe, and
the lessons. Media stays in the (gitignored) `in/ out/ work/` dirs.

Runs on: macOS Apple Silicon (MPS), **no CUDA**. `ffmpeg` 9.0 static in `../../bin/`.
Python: the project 3.12 `.venv`. It already carries `torch` / `torchvision` /
`basicsr` / `realesrgan` (via `kinocut[upscale]`), so RVM matting and
Real-ESRGAN run there. Only two things need a separate venv: rembg/isnet
(`.matte-venv`, Phase 2) and IC-Light (`.gen-venv`, GPU box only — does not run
here).

---

## Pipeline (what actually worked — v23)

```
source.mp4 (person)
  └─ matte_rvm.py ........... RVM video matting  → RGBA frames, per-frame, temporally stable
  └─ (matte_cleanup.py) ..... only if using a per-frame matte (isnet); RVM does not need it
  └─ upscale_realesrgan.py .. Real-ESRGAN x2 on the RGBA frames → sharper subject
  └─ place_composite.py ..... foot-lock placement + per-scene relight grade + contact shadow
                              → 2160x3840 RGBA frames, subject positioned on the canvas
compose_pipeline.sh
  ├─ background: a real-time PORTRAIT stock clip, scaled to 2160x3840, fps=30 (no retiming)
  ├─ overlay subject frames on the background
  ├─ depth occlusion: re-overlay the bottom strip of the bg on top of the subject
  │                   (front-row crowd passes in front of her — feathered top edge)
  ├─ grade: unify (eq + colorbalance + curves), atmosphere (downscaled gblur, screen 8%),
  │         mild film grain, mild vignette
  └─ encode: libx264 High, crf 17-18, +faststart, 30fps native
```

Every stage is a small standalone script. `compose_pipeline.sh` is the glue; edit the
variables at the top (bg clip, offsets, occlusion strip Y) per shot.

---

## Tools — verdict

| Tool                                                          | Verdict               | Notes                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RVM** (Robust Video Matting, `PeterL1n/RobustVideoMatting`) | **USE THIS**          | Recurrent → temporally stable, no flicker, no per-frame holes. ~0.06 s/frame on MPS (resnet50). Empirically kept the held sign on this clip (docs warn it can drop held objects — test per clip). `matte_rvm.py`. Weights: GitHub release `rvm_resnet50.pth` / `rvm_mobilenetv3.pth`. |
| **rembg / isnet-general-use**                                 | fallback              | Per-frame → flickers, drops holes. Needs `matte_cleanup.py` (temporal median ±3 + small-hole fill; NEVER fill the enclosed gap between raised arms — that must stay transparent). `pip install "rembg[cpu]"`, use the Python API not the CLI. `matte_isnet.py`.                       |
| **Real-ESRGAN** (`realesrgan`+`basicsr`)                      | useful                | x2 upscale of the cutout, cleaner edges/texture. `basicsr` breaks on new torchvision — add a `torchvision/transforms/functional_tensor.py` shim re-exporting from `_functional_tensor`. ~7.5 s/frame MPS. `upscale_realesrgan.py`. Weights: `RealESRGAN_x2plus.pth`.                  |
| **IC-Light** (diffusers, `lllyasviel/ic-light` fbc)           | **does NOT run here** | Relighting to match scene — the only thing that would get to "can't tell". >10 min/frame on 9 GB MPS, OOM at 512², no output. Needs a real NVIDIA GPU or a cloud service. `iclight_relight.py` kept for reference / future GPU box.                                                   |
| **MatAnyone** (`pq-yang/MatAnyone`, CVPR 2025)                | untried, promising    | Video-native + first-frame mask propagation → guaranteed to keep the sign if the first-frame mask includes it. `pip install git+https://github.com/pq-yang/MatAnyone.git`. Worth trying next.                                                                                         |
| **BiRefNet** / `transparent-background`                       | untried               | Dichotomous salient seg → keeps held objects; add a temporal median. `ZhengPeng7/BiRefNet-matting` on HF.                                                                                                                                                                             |
| `kino` (kinocut CLI, wrapped by this project)                 | not for compositing   | `composite-layers` forces H.264 no-CRF and caps 25 fps; `video-ai-upscale` broken on ffmpeg 9 (`-vsync`). Only `hyperframes-remove-background` was used (→ isnet).                                                                                                                    |

---

## Lessons (read before the next attempt)

1. **Background must be a real-time, portrait-native clip.** Slowing a hyperlapse
   (`setpts` + `minterpolate`) makes moving people warp (`mci`) or fade in/out
   (`blend`) — reads as "random people appearing". Get the right clip instead.
   Pexels has portrait real-time crowd clips (`videos.pexels.com/video-files/<id>/…`).
   Landscape 1080p cropped to 2160×3840 portrait = mush; must be portrait-native or ≥1440 wide.
2. **Foot–ground contact is the hard ceiling** of ffmpeg+PIL compositing. Options tried:
   per-frame foot-lock, rolling-min, hard rigid pin (one fixed Y), 3-layer contact
   shadow, ground-line matching per plate. Still slightly floats. A real fix needs
   frame-by-frame rotoscoping or a 3D/relight pass. Mitigations that help most:
   rigid pin + a firm dark contact shadow + hiding the exact contact point behind a
   foreground element (a parked car, the front row of a crowd).
3. **Depth**: compositing the subject on top of everyone makes her "walk over" the
   crowd. Re-overlaying the bottom ~200–400 px of the bg on top of the subject
   (feathered) puts the nearest people in front of her — cheap, effective occlusion.
4. **Match the plate**: desaturate + drop contrast + a touch of blur on the subject
   so she isn't sharper/punchier than the crowd around her; tint to the scene's
   white balance; add grain AFTER compositing so fg and bg share it.
5. **Choose a plate with visible foreground ground.** Times Square crops had the
   crowd at head-height with no pavement to stand on → nowhere to plant feet.
   The Wall St plate had an open asphalt foreground → feet had somewhere to go.
6. `minterpolate` at 3840 px is very slow (~0.7 fps) and warps on scene changes.
   For 30→60 fps use `me=ds:search_param=16:mb_size=16` (fast) — or skip 60 fps.
   Deliver 30 fps native unless smoothness is a hard requirement.
7. `zsh` does NOT word-split unquoted `$vars` — put ffmpeg pipelines in a `#!/bin/bash`
   script or use arrays.
8. ffmpeg `fade=t=in:st=N` whites-out everything _before_ N — cannot chain fades for
   mid-clip blinks. Use `xfade` between separate segments instead.
9. Static ffmpeg has no fontconfig → `drawtext` fails; render text to PNG with PIL.

---

## Install

```sh
# torch / torchvision / basicsr / realesrgan + the functional_tensor shim
# come from `make setup` (kinocut[upscale] + scripts/patch-basicsr-shim.sh).

# RVM matting: a source checkout on sys.path (GPL-3, never committed) + weights
git clone https://github.com/PeterL1n/RobustVideoMatting tools/RobustVideoMatting
#   weights: rvm_resnet50.pth / rvm_mobilenetv3.pth from that repo's releases,
#   into work/punto-edit/gen/weights/ (or MEDIA_LAB_WEIGHTS_DIR)

# Phase 2 only — rembg / isnet fallback (its own py3.9 venv):
python3 -m venv .matte-venv && .matte-venv/bin/pip install "rembg[cpu]" pillow onnxruntime scipy

# Out of scope — IC-Light generative relight (GPU box only, does not run here):
python3 -m venv .gen-venv && .gen-venv/bin/pip install torch torchvision diffusers transformers accelerate safetensors
```

`docs/punto-edit.md` in the repo root has the full blow-by-blow session log
(every version v1–v23, what changed and why).

"""IC-Light (fbc: foreground + background conditioned) relighting for the punto cutout.

Relights the isnet subject cutout so its lighting matches each scene background.
Per-frame; fixed seed + fixed bg per segment for stability. Keeps our own alpha
matte (IC-Light only relights RGB).

Usage:
  iclight.py test <frame_idx>          -> single frame to work/punto-edit/gen/test/
  iclight.py batch <a> <b> <bg_path>   -> frames [a,b) relit against bg_path
"""
import sys
import os
import numpy as np
import torch
import safetensors.torch as sf
from PIL import Image
from huggingface_hub import hf_hub_download
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    EulerAncestralDiscreteScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float32
PROC_W, PROC_H = 320, 576  # portrait, multiple of 64 - kept small for MPS memory
STEPS = 12
CFG = 1.8
SEED = 12345
PROMPT = "a woman holding a sign, standing outdoors, natural realistic lighting, photograph, high quality"
N_PROMPT = "cartoon, illustration, cgi, plastic, overexposed, bad shadows, deformed"

CUT_DIR = "work/punto-edit/isnet/cut"


def load_models():
    tok = CLIPTokenizer.from_pretrained(SD15, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(SD15, subfolder="text_encoder").to(DEVICE, DTYPE).eval()
    vae = AutoencoderKL.from_pretrained(SD15, subfolder="vae").to(DEVICE, DTYPE).eval()
    unet = UNet2DConditionModel.from_pretrained(SD15, subfolder="unet").to(DEVICE, DTYPE).eval()

    with torch.no_grad():
        new_conv = torch.nn.Conv2d(12, unet.conv_in.out_channels,
                                   unet.conv_in.kernel_size, unet.conv_in.stride,
                                   unet.conv_in.padding)
        new_conv.weight.zero_()
        new_conv.weight[:, :4].copy_(unet.conv_in.weight)
        new_conv.bias.copy_(unet.conv_in.bias)
        unet.conv_in = new_conv
    unet = unet.to(DEVICE, DTYPE)

    w = hf_hub_download("lllyasviel/ic-light", "iclight_sd15_fbc.safetensors")
    offset = sf.load_file(w)
    origin = unet.state_dict()
    merged = {k: origin[k] + offset[k].to(origin[k]) if k in offset else origin[k]
              for k in origin}
    unet.load_state_dict(merged, strict=True)

    try:
        unet.set_attention_slice(1)
    except Exception as e:
        print("attn slice unavailable:", e)
    try:
        vae.enable_slicing()
        vae.enable_tiling()
    except Exception:
        pass

    sched = EulerAncestralDiscreteScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", steps_offset=1,
    )
    return tok, te, vae, unet, sched


@torch.no_grad()
def encode_text(tok, te, prompt):
    ids = tok(prompt, padding="max_length", max_length=tok.model_max_length,
              truncation=True, return_tensors="pt").input_ids.to(DEVICE)
    return te(ids).last_hidden_state


@torch.no_grad()
def vae_encode(vae, img):  # img: PIL RGB -> latent
    a = np.asarray(img.resize((PROC_W, PROC_H), Image.LANCZOS)).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(DEVICE, DTYPE)
    return vae.encode(t).latent_dist.mode() * vae.config.scaling_factor


@torch.no_grad()
def vae_decode(vae, lat):
    img = vae.decode(lat / vae.config.scaling_factor).sample[0]
    img = ((img.float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(img.astype(np.uint8))


def relight(models, cut_png, bg_img):
    tok, te, vae, unet, sched = models
    cut = Image.open(cut_png).convert("RGBA")
    alpha = np.asarray(cut)[..., 3]
    ys, xs = np.where(alpha > 20)
    if len(ys) == 0:
        return cut
    pad = 40
    x0, x1 = max(0, xs.min() - pad), min(cut.width, xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(cut.height, ys.max() + pad)
    crop = cut.crop((x0, y0, x1, y1))
    ca = np.asarray(crop).astype(np.float32)
    # foreground on grey
    fg_rgb = ca[..., :3]
    a = (ca[..., 3:4] / 255.0)
    fg = fg_rgb * a + 127.0 * (1 - a)
    fg_img = Image.fromarray(fg.astype(np.uint8), "RGB")
    bg_crop = bg_img.resize((crop.width, crop.height), Image.LANCZOS)

    cond = encode_text(tok, te, PROMPT)
    uncond = encode_text(tok, te, N_PROMPT)
    fg_lat = vae_encode(vae, fg_img)
    bg_lat = vae_encode(vae, bg_crop)
    concat = torch.cat([fg_lat, bg_lat], dim=1)  # 8ch

    g = torch.Generator(device="cpu").manual_seed(SEED)
    lat = torch.randn((1, 4, PROC_H // 8, PROC_W // 8), generator=g).to(DEVICE, DTYPE)
    sched.set_timesteps(STEPS, device=DEVICE)
    lat = lat * sched.init_noise_sigma
    for t in sched.timesteps:
        li = sched.scale_model_input(lat, t)
        mi = torch.cat([li, concat], dim=1)  # 12ch
        n_c = unet(mi, t, encoder_hidden_states=cond).sample
        n_u = unet(mi, t, encoder_hidden_states=uncond).sample
        noise = n_u + CFG * (n_c - n_u)
        lat = sched.step(noise, t, lat).prev_sample
        del li, mi, n_c, n_u, noise

    out = vae_decode(vae, lat).resize((crop.width, crop.height), Image.LANCZOS)
    # paste relit RGB back, keep original alpha
    res = np.asarray(cut).copy()
    o = np.asarray(out).astype(np.uint8)
    res[y0:y1, x0:x1, :3] = o
    return Image.fromarray(res, "RGBA")


def main():
    mode = sys.argv[1]
    models = load_models()
    if mode == "test":
        idx = int(sys.argv[2])
        os.makedirs("work/punto-edit/gen/test", exist_ok=True)
        bg = Image.open(sys.argv[3]).convert("RGB")
        f = f"{CUT_DIR}/f-{idx:04d}.png"
        out = relight(models, f, bg)
        out.save(f"work/punto-edit/gen/test/relit-{idx:04d}.png")
        # side by side vs original
        orig = Image.open(f).convert("RGBA")
        cmp = Image.new("RGB", (orig.width * 2, orig.height), (128, 128, 128))
        cmp.paste(Image.alpha_composite(Image.new("RGBA", orig.size, (128,128,128,255)), orig).convert("RGB"), (0, 0))
        cmp.paste(Image.alpha_composite(Image.new("RGBA", orig.size, (128,128,128,255)), out).convert("RGB"), (orig.width, 0))
        cmp.save("work/punto-edit/gen/test/compare.png")
        print("wrote work/punto-edit/gen/test/compare.png")
    else:
        a, b, bgp = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        outdir = "work/punto-edit/gen/relit"
        os.makedirs(outdir, exist_ok=True)
        bg = Image.open(bgp).convert("RGB")
        import time
        t0 = time.time()
        for i in range(a, b):
            f = f"{CUT_DIR}/f-{i:04d}.png"
            relight(models, f, bg).save(f"{outdir}/f-{i:04d}.png")
            if (i - a) % 5 == 0:
                print(f"{i}  {time.time()-t0:.0f}s", flush=True)
        print(f"DONE {a}-{b} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

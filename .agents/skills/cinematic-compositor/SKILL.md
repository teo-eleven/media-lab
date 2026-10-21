---
name: cinematic-compositor
description: >-
  Professional film post-production and high-end visual effects guide. Covers neural RVM
  matting, statistical Lab colour transfer, multi-layer contact shadows, foot grounding,
  film grain emulation, and DaVinci-grade color harmony.
---

# Cinematic Compositor — High-End VFX & Color Science

This guide details professional post-production workflows inspired by DaVinci Resolve, Nuke, and Hollywood VFX compositing pipelines, adapted for `media-lab` local execution.

---

## 1. Clean Neural Matting & Alpha Precision
- **Model:** Robust Video Matting (`RVM`) using MobileNetV3/ResNet50 backbone.
- **Format:** Always output in **Apple ProRes 4444 (`.mov`)** or uncompressed RGBA PNG image sequences. Never encode alpha into VP9/WebM for intermediate VFX steps, as compositor pipelines can discard secondary alpha blocks.
- **Flicker mitigation:** Temporal median smoothing on alpha masks prevents edge chattering between adjacent frames.

---

## 2. Geometric Grounding & Perspective Alignment
- **Foot Pinning:** Locate the lowest alpha centroid for each frame. Lock the ground plane coordinate ($Y_{\text{ground}}$) to prevent subject floating during walking cycles.
- **Multi-Layer Contact Shadow Architecture:**
  1. *Ambient Occlusion (AO) Core:* Dark, tight ellipse ($opacity \approx 0.85$, $blur \approx 4\text{px}$) directly beneath the contact foot.
  2. *Diffuse Contact Shadow:* Medium ellipse scaled to body width ($opacity \approx 0.50$, $blur \approx 18\text{px}$).
  3. *Directional Cast Shadow:* Stretched ellipse angled opposite the primary scene light vector (`sun_dir`).

---

## 3. Statistical Color Harmonisation (Reinhard Lab)
- **Problem:** Subjects shot in front of green screens or neutral backdrops clash chromatically with natural backgrounds.
- **Solution:** Transform both subject and background plate into CIE $L^*a^*b^*$ color space. Align the mean ($\mu$) and standard deviation ($\sigma$) of the subject channels to match the background lighting zone:
  $$L^*_{\text{new}} = (L^*_{\text{subj}} - \mu_{L,\text{subj}}) \cdot \frac{\sigma_{L,\text{bg}}}{\sigma_{L,\text{subj}}} + \mu_{L,\text{bg}}$$
- **Ground Bounce Light:** Add a subtle tint and lift to the lower 30% of the subject reflecting ambient ground color.

---

## 4. Visual Quality Control (Render Doctor)
Before marking any shot as final deliverable:
- Run ffprobe inspection on bitrate, stream layout, and alpha integrity.
- Verify that color luminance ranges stay within legal broadcast limits (IRE 0–100).
- Extract and review contact sheets across minimum 6 evenly spaced time intervals.

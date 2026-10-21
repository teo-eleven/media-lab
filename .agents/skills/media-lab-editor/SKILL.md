---
name: media-lab-editor
description: >-
  Expert guide and command reference for the media-lab AI video, audio, and photo
  pipeline. Use whenever inspecting media, running CLI editing recipes, authoring EditSpec
  YAML pipelines, or executing natural language prompt-to-edit workflows.
---

# Media-Lab Expert Skill

`media-lab` is a 100% local, typed, non-destructive video/audio/photo editing pipeline built for Apple Silicon (Metal/MPS) with strict ffprobe verification on every render.

## Core Rules & Guardrails
1. **Non-destructive:** Never modify sources in `in/`. All outputs go to `out/`. Intermediate files belong in `work/`.
2. **Force guard:** Existing outputs in `out/` require `--force` to overwrite.
3. **Verify every render:** Always verify codec, duration, alpha channel, and audio loudness using `media_lab.verify` or `media-lab probe`.
4. **Use local ffmpeg:** Always rely on `./bin/ffmpeg` and `./bin/ffprobe` configured via `.env`.

---

## CLI Command Quick Reference

### Video & Pacing
- `media-lab short <input> -o <output>` : Reframe to vertical 9:16 social clip + quality gate.
- `media-lab smart-reframe <input> -o <output> [--smooth 0.8]` : Dynamic face-tracking vertical crop.
- `media-lab punch-zoom <input> -o <output> [--factor 1.15 --interval 4.0]` : Dynamic engagement zoom.
- `media-lab broll <input> --broll <overlay> -o <output> --start <sec> --duration <sec>` : Cutaway insert with continuous audio.
- `media-lab stabilize <input> -o <output> [--shakiness 5 --accuracy 9]` : 2-pass VidStab camera stabilization.
- `media-lab speed <input> -o <output> --speed <float>` : Retime video (0.1x to 10x) preserving audio pitch.
- `media-lab progress-bar <input> -o <output> [--color yellow --position bottom --height 8]` : Retention progress bar.
- `media-lab assemble <clip1> <clip2> ... -o <output> [--transition wipeleft --duration 0.5]` : Multi-clip montage with xfade.

### Audio Studio
- `media-lab audio-enhance <input> -o <output>` : Studio mastering (80Hz high-pass, voice EQ, de-esser 6.5kHz, compressor, gate).
- `media-lab cut-silence <input> -o <output> [--noise-db -32 --min-silence 0.4]` : Jump-cut pauses automatically.
- `media-lab stems <input> -o <output_dir> [--clean-speech]` : Demucs vocal/music separation or isolated clean voice.
- `media-lab music <input> --track <music.mp3> -o <output>` : EBU R128 ducking (music dips under speech).
- `media-lab sfx <input> -o <output> --sound whoosh --timestamp <sec>` : Procedural sound effects.
- `media-lab narrate --text "<text>" -o <output.wav> [--voice ro|en]` : Local offline TTS narrator.

### Visual Effects & Matting
- `media-lab matte <input> -o <output.mov>` : Robust Video Matting (RVM) neural human cutout (ProRes 4444 RGBA).
- `media-lab ground <input_cutout.mov> -o <output.mov> [--canvas 2160x3840 --sun-dir 1.0]` : Foot pinning and 3-layer contact shadow.
- `media-lab colour-match <subject> --reference <bg> -o <output>` : Reinhard Lab statistical color & lighting transfer.
- `media-lab upscale <input> -o <output> [--scale 2|4]` : Real-ESRGAN super-resolution.
- `media-lab filter <input> -o <output> --preset warm|cool|cyberpunk|noir` : Curated color grading looks.

### Subtitles & Graphics
- `media-lab subtitles <input> -o <output> [--preset tiktok|clean|box --burn]` : Whisper speech-to-text with styled ASS subtitles.
- `media-lab text-overlay <input> --text "<title>" -o <output> [--style badge|minimal]` : Kinetic text overlays.
- `media-lab inpaint <input> --mask <mask> -o <output>` : Content-aware removal and backdrop fill.
- `media-lab retouch <input> -o <output> [--blur-bg 15]` : Portrait skin softening & artificial depth of field.

### Declarative & Natural Language
- `media-lab inspect <media_path>` : Deep JSON report (LUFS, RMS, silence, face presence, resolution, fps).
- `media-lab edit <spec.yaml>` : Execute multi-step declared pipeline in a single optimized pass.
- `media-lab prompt <input> -o <output> --prompt "<natural language in RO/EN>"` : Auto-generate EditSpec and render.
- `media-lab studio [--port 8765]` : Local interactive Web Chat Studio UI with video player.
- `media-lab mcp` : Native Model Context Protocol server over stdio.

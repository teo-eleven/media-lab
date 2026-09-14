# DECISIONS.md

Technical decisions taken on this project, with the trade-off each one accepts.
Newest first.

## 2026-09-14 — Phase 5: Studio 360° Superpowers & Local Web Studio UI (Multi-Clip Assembly, Speed Ramping, Retention Progress Bar, Beat Sync, Local Narrator, and Real-Time Chat UI)

**Context.** Transforming Media Lab into a complete 360° Studio required:
1. Multi-clip timeline assembly with transition effects and seamless audio crossfades.
2. Motion dynamics / speed ramping (slow-motion / timelapse) with pitch-preserved audio.
3. Dynamic social retention progress bar (sliding animated bar for TikTok/Reels/Shorts).
4. Audio rhythm and beat detection (BPM and transient onsets) for musical cuts.
5. Local offline text-to-speech voiceover narration with native Romanian and English voices.
6. A local Web Studio interface with an interactive chat UI, HTML5 media player with HTTP 206 streaming, and real-time execution of automated editing pipelines.

**Chosen.**
1. `recipes/assembly.py`: Chained FFmpeg `xfade` (15 cinematic transition modes) with recursive cumulative offset calculation $\text{offset}_k = \text{offset}_{k-1} + d_{k-1} - \delta$, coupled with equal-power `acrossfade` audio transitions and pre-normalization to unified resolution, 30fps, SAR 1:1, and YUV420p.
2. `recipes/speed.py`: Exact timestamp scaling via `setpts=(1.0/speed)*(PTS-STARTPTS)` paired with dynamic factorization of arbitrary speeds into chained FFmpeg `atempo` filters bounded strictly to $[0.5, 2.0]$.
3. `recipes/progress_bar.py`: Dynamic sliding overlay pattern (`min(0, -w+w*(t/dur))`) over a static semi-transparent track, ensuring frame-accurate continuous filling without disappearing after clip duration.
4. `recipes/beat_sync.py`: Offline audio transient energy flux calculation and derivative peak picking with moving-window tempo estimation, exporting beat timestamps and BPM.
5. `recipes/narrator.py`: macOS native speech synthesis (`/usr/bin/say`) with Romanian 'Ioana' and English voices, audio mix attenuation compensation (`weights=1 1:normalize=0`), and timeline muxing with `-shortest` defense.
6. `server.py` + `static/index.html`: Zero-dependency, pure Python `http.server.ThreadingHTTPServer` implementing RFC 7233 byte-range streaming for seamless video seeking, path traversal defense via `is_relative_to()`, DOM XSS prevention, and a dark-mode studio interface.

**Trade-off accepted.** Using Python's standard library for the Web Studio avoids adding heavy web framework dependencies (FastAPI/Flask/Starlette/Node/React), keeping the application lightweight, standalone, and completely offline on macOS Apple Silicon.

---

## 2026-09-14 — Phase 4: Omnichannel Autonomous Studio (Audio Mastering, Silence Trim, SFX, Smart Reframe, Punch Zoom, B-Roll, Typography, Inpainting, Face Retouch, Natural Language Prompt Agent, and MCP Server)

**Context.** Creating an end-to-end autonomous media creation suite required extending beyond basic cutting to full-studio creative capabilities:
1. Audio Mastering & Pacing: vocal dynamics mastering (EQ, compand, de-esser, loudness normalization), silence/pause jump-cutting, and procedural sound effect synthesis (whooshes, pops, dings).
2. Video Pacing & Cutaways: face/salience tracking vertical reframing, retention punch-in zooms, and B-roll cutaway overlays preserving continuous dialogue audio (L/J-cuts).
3. Photo & Graphic Studio: dynamic typography and capsule badges with auto-wrap, content-aware object/blemish inpainting, edge-preserving portrait skin smoothing, and depth bokeh defocus blur.
4. Autonomous Prompt Agent & Model Context Protocol (MCP): bilingual natural language prompt understanding translating vague requests to executable `EditSpec` pipelines, plus an MCP stdio server allowing Antigravity, Claude, and Cursor to execute all operations natively.

**Chosen.**
1. `recipes/audio_enhance.py` + `recipes/silence_trim.py` + `recipes/sfx.py`: Multi-stage FFmpeg audio mastering filtergraphs (highpass 80Hz, profile EQ, de-esser, compand, loudnorm -16 LUFS), silencedetect interval tracking with padding and `aresample=async=1000` drift correction, and pure procedural mathematical synthesis via `lavfi` (`anoisesrc`, `sine`, envelopes) with stereo alignment.
2. `recipes/smart_reframe.py` + `recipes/punch_zoom.py` + `recipes/broll.py`: Median skin chrominance/contrast centroid tracking with fallback split-blur; dual-branch split-scale-overlay pattern for dynamic camera punch-ins bypassing FFmpeg static crop limitations; PTS-shifted cutaways with media probing to prevent missing stream crashes.
3. `recipes/typography.py` + `recipes/inpainting.py` + `recipes/face_retouch.py`: PIL typography engine with pill capsule badges, drop shadows, and `-loop 1` video overlays; OpenCV Navier-Stokes/Telea inpainting for object and text erasure; feathered YCrCb bilateral skin smoothing preserving facial features with optional background depth blur.
4. `prompt_agent.py` + `mcp_server.py`: Deterministic Romanian and English heuristic prompt interpreter parsing complex composite requests into typed `EditSpec` execution trees; JSON-RPC 2.0 stdio MCP server exposing media inspection, prompt orchestration, and 9 direct tool endpoints without external heavy framework dependencies.

**Trade-off accepted.** Complex filtergraphs require precise stream formatting (explicit stereo layouts, sample rate normalization, PTS offsetting, and even dimension rounding), but allow 100% local Apple Silicon execution with zero cloud lock-in.

---

## 2026-09-14 — Phase 3: Demucs audio separation, Whisper ASS subtitles, Photo batch processing, Deep Inspection, and Declarative EditSpec Orchestrator

**Context.** Creating social media content and enabling an autonomous media-lab agent
required:
1. Audio voice isolation (stems) and speech denoising without external cloud services.
2. Word-level subtitle generation with modern social styling (TikTok, clean, box) burned or exported.
3. High-quality single and batch photo processing with Reinhard color harmonization and shadow grounding.
4. Objective media ground truth (deep inspection metrics: duration, resolution, visual palette, silhouette, audio levels, speech detection) so an AI agent can make informed editing choices.
5. Declarative prompt-to-edit orchestration (`EditSpec` schema) that can chain speech cleaning, grading, reframing, subtitles, and music mixing in an optimized, single-pass pipeline.

**Chosen.**
1. `ml/demucs_infer.py` + `recipes/stems.py`: Demucs v4 (htdemucs) run in isolated child process via `ml_runner.py`, supporting MPS hardware acceleration and `--clean-speech` video remuxing.
2. `ml/whisper_infer.py` + `subtitles.py` + `recipes/subtitles.py`: OpenAI Whisper run locally in isolated child process with fallback, generating typed ASS subtitles with custom social styles or burning via ffmpeg `libass`.
3. `recipes/photo.py`: PIL-based image composition with smart aspect ratio cropping (1:1, 4:5, 9:16, 16:9), Reinhard Lab color matching, contact shadows, unsharp mask sharpening, and directory batch processing.
4. `inspect.py`: Deep inspection engine computing ground truth metrics directly from ffprobe and PIL analysis (brightness, contrast, dominant RGB/Lab palette, silhouette detection, audio RMS and speech presence) returned as typed dataclasses or JSON.
5. `edit_spec.py` + `recipes/edit.py`: Declarative edit specification schema (YAML/dict/CLI flags) and executor chaining native recipes with deterministic intermediate tracking and strict verification.

**Trade-off accepted.** Increased ML inference dependencies (demucs, openai-whisper) managed in sub-processes to prevent torch memory leaks and maintain strict mypy typing on the core package.

---

## 2026-09-14 — Phase 2: modular upscale, Reinhard colour-match, foot-pinning grounding replace legacy scripts

**Context.** Phase 1 staged frames around `docs/video-agent/pipeline/upscale_realesrgan.py`
and `place_composite.py` via ad-hoc directory moves (`isnet/up/` -> `rvm_up/`). The scripts
had hardcoded paths and rigid configurations.

**Chosen.** Folded all logic into four typed, modular, tested tools:
1. `recipes/upscale.py` + `ml/realesrgan_infer.py`: Real-ESRGAN running as a subprocess
   via `ml_runner`, alpha upscaled with Lanczos, supporting arbitrary directories and video files.
2. `colour_transfer.py` + `recipes/colour_match.py`: Reinhard statistical colour transfer in
   decorrelated Lab space + scene directional lighting and ground bounce.
3. `grounding.py` + `recipes/subject_ground.py`: Foot contact point detection, silhouette base
   locking (foot-pinning), zoom normalisation, and 3-layer directional contact shadow rendering.
4. `scale_plate.py` + `recipes/scale_plate.py`: Analytical scale and ground line estimation from
   background plate reference persons.

The `punto` runner now calls these native recipes directly, eliminating all directory-move hacks.

**Trade-off accepted.** More source files to maintain, but zero hardcoded workspace paths,
fully isolated ML boundaries, full unit test coverage, and reusable CLI tools.

---

## 2026-09-09 — compose-spec renders in one fused ffmpeg pass

**Context.** The v23 `compose_pipeline.sh` was three ffmpeg invocations with
two ProRes 422 intermediates (~720 MB each): background prep, composite +
occlusion, grade + encode. Spike S1 reproduced each stage from parameters.

**Chosen.** `compose_spec.build_filtergraph` emits a single `-filter_complex`
graph that does all three: `[0:v]fps,scale,split` -> feathered occlusion strip
via `geq` -> overlay the pre-placed subject at 0:0 -> re-overlay the strip ->
the v23 grade chain -> encode. One `ffmpeg` call, no intermediates.

**Trade-off accepted.** The graph string is long (~700 chars) and a failure in
it is harder to bisect than a failed stage. In exchange: no 1.4 GB of ProRes
per run, one pass instead of three, and — measured against the real v23
render — `mean|Δ| 2.3/255` (better than the 3-stage spike's 2.6, because the
ProRes round-trips are gone). Bit-exact is impossible anyway: `noise=allf=t`
reseeds per run.

---

## 2026-09-09 — the punto runner stages frames around the unchanged v23 scripts

**Context.** Phase 1 keeps `upscale_realesrgan.py` and `place_composite.py`
verbatim (they are a captured record). But their hardcoded I/O dirs do not
chain: matte writes `rvm/`, upscale reads `isnet/cut/` and writes `isnet/up/`,
place reads `rvm_up/` (or falls back to `isnet/cut/`). The v23 run only worked
via manual renames recorded nowhere (audit finding M2).

**Chosen.** `recipes/punto_v23.py` stages between them: matte frames ->
`isnet/cut/`; after upscale, **move** `isnet/up/` -> `rvm_up/` so `place`
picks the upscaled frames. `ml_runner.run` gained a `cwd` argument so the
scripts run from the repo root where their relative paths resolve. `--proxy`
skips the upscale and the move, so `place` composites the 720p frames and the
subject is ~half v23 scale — the CLI says so.

**Trade-off accepted.** The runner encodes knowledge of two scripts' internal
paths, so editing those scripts can break it silently. This is the documented
Phase-2 cleanup point (fold them into real `upscale` / `subject-ground`
tools). Until then the alternative — a one-line env-var patch to each script —
was rejected to keep them a faithful v23 record.

---

## 2026-09-09 — video-agent Phase 1: recipes + CLI, not skills or subagents

**Context.** The `docs/video-agent/ROADMAP.md` lists `compose-spec`,
`proxy-preview` and `matte-video` in a "Tools / skills" table and a separate
"Subagents" table. They could be standalone skills (like `verify-render`),
Claude Code subagents, or Python recipes in this package.

**Chosen.** Python recipes in `src/media_lab/recipes/` plus `media-lab`
subcommands, extending the existing package on `feat/video-agent-toolkit`.

**Trade-off accepted.** They are not invokable from outside a `media-lab`
checkout until a later phase wraps them. In exchange they reuse `config`,
`paths`, `verify`, `ffmpeg` and the test harness unchanged — one codebase, one
error model, one non-destructive contract — instead of re-implementing that
infrastructure per skill. A thin skill/agent wrapper is a Phase 4 concern.

---

## 2026-09-09 — Run RVM in a child process, reusing the project venv

**Context.** `matte-video` wraps Robust Video Matting. RVM is **GPL-3.0** and is
used as a source checkout (`from model import MattingNetwork`), not a package —
`import`ing it into `media_lab`'s own process would make our process a
derivative work. RVM needs `torch` + `torchvision`; an audit on 2026-09-09
found the project `.venv` **already carries both** (plus `basicsr` and
`realesrgan`) because `kinocut[upscale]` declares `torch>=2.0` — they are in
`uv.lock`. An earlier draft of this entry assumed the opposite and proposed a
second venv; that rationale was wrong.

**Chosen.** Clone RVM to `tools/RobustVideoMatting/` (git-ignored, never
committed — GPL). Invoke it through a new `ml_runner.py` module — the single
sanctioned exit for ML subprocesses, as `kino.py` is for kinocut and
`ffmpeg.py` for direct ffmpeg. `ml_runner` runs a driver script
(`src/media_lab/ml/rvm_infer.py`) as a **child process using the project
interpreter** (`sys.executable`); the driver puts the RVM clone on `sys.path`,
imports it there, and writes an RGBA PNG sequence + a JSON stats file. No
separate venv.

**Trade-off accepted.** One extra process boundary and a git-ignored source
checkout to recreate (alongside `bin/`). Data crosses as files, not tensors.
In exchange: GPL code is never imported into our process and never enters the
repo; `torch` (a ~1 s import) stays out of the main CLI process unless matting
is actually run; a torch/RVM break is contained to one module. Cost of the
correction: `ml_runner.py` is thinner than a full venv-switching runner, but
kept for the process boundary and the timeout/stderr/typed-error wrapper.

---

## 2026-09-09 — ML paths are lazily-validated config, not startup-validated

**Context.** `config.py` validates everything at startup so the project fails
loudly rather than mid-render. The new `rvm_repo` / `weights_dir` paths only
matter to `matte-video`.

**Chosen.** Add them to `Config` with project-root-relative defaults, but check
them in a `require_ml(config)` helper the recipe calls — not in `load_config`.
`doctor` reports ML state as `(ok)` / `(not configured)` without failing.

**Trade-off accepted.** A second validation style in the codebase (startup for
core, lazy for ML). In exchange, `doctor`, `cutout`, `backdrop`, `filter`,
`music`, `short` and `pipeline` all keep working on a machine that has never
installed torch.

---

## 2026-09-09 — matte-video writes ProRes 4444 .mov

**Context.** `pipeline/matte_rvm.py` writes a PNG sequence; `place_composite.py`
consumes a directory of PNGs. But the project's verification layer
(`verify_render`, `measure_alpha_spread`) is built around a single file with a
readable alpha channel, and `cutout.py` already established ProRes 4444 `.mov`
as the alpha-carrying interchange format.

**Chosen.** `matte-video` writes frames internally, then muxes a ProRes 4444
`.mov` as its deliverable. The PNG working directory is kept as an
intermediate.

**Trade-off accepted.** A mux step and a large file (ProRes is ~214 MB / 6.4 s).
In exchange, one artefact to verify with the existing tooling, consistent with
`cutout`.

---

## 2026-09-09 — compose-spec: YAML in, filtergraph emitted and run, subject pre-placed

**Context.** `compose-spec` replaces the hand-edited `compose_pipeline.sh`.
Open questions: spec format; whether it also positions the subject; how the
grade chain is exposed.

**Chosen.** A YAML spec (comments, readability). `compose-spec` writes the
`-filter_complex` string to a `.filtergraph.txt` sidecar **and** runs it. It
overlays a **pre-placed, full-canvas** RGBA subject sequence at `0:0` — it
never scales or moves the subject (that stays in `place_composite.py` / a
Phase-2 `subject-ground` tool). The grade is a fixed `v23` profile
(constants lifted from the v23 script) toggled on/off with a few scalar knobs
(atmosphere opacity, grain, vignette). The depth-occlusion strip is
parametrised (height, feather, y).

**Trade-off accepted.** The spec cannot express a camera move or a custom
grade without a code change. In exchange, the person writing a spec cannot
hit the `zsh` word-split or `fade` white-out pitfalls, and the grade stays a
named look rather than raw ffmpeg values — the same philosophy as
`recipes/filters.py`. Matches how v23 actually rendered (`overlay=0:0` on
full-canvas placed frames).

---

## 2026-09-09 — Phase 1 acceptance is the punto clip, checked to container spec

**Context.** "Phase 1 done" needed a concrete bar. Options ranged from
"synthetic tests only" to "bit-exact reproduction of v23".

**Chosen.** Both: each tool carries its own suite (happy + ≥2 edge + 1 error,
coverage ≥ 80%, ML mocked by default with one opt-in real-RVM test), **and** a
`media-lab punto` runner reproduces the v23 render through the three new tools.
The reproduction is judged a pass on container spec (2160×3840, 30 fps,
~6.43 s, no audio, H.264 High, yuv420p — via `verify-render`) plus a visual
contact-sheet match against
`out/punto-final_2160x3840_30fps_h264-crf17.mp4`. Not bit-exact.

**Trade-off accepted.** The visual half of the check is a human judgement, not
an assertion, so it cannot gate CI (there is no CI — single-user, local). In
exchange, the acceptance actually proves the three tools replace the v23
pipeline, which "synthetic tests only" would not.

---

## 2026-09-03 — Wrap Kinocut rather than write our own ffmpeg layer

**Context.** The goal was a local editor for short social clips: person
cutout, new backdrop, looks, background music, vertical export. A GitHub survey
turned up ~20 "ffmpeg MCP" projects, nearly all at 0-3 stars and unmaintained.
`KyaniteLabs/kinocut` (136 stars, Apache-2.0, active) was the only serious
candidate, and it is built on the same principle we wanted: typed tools with
preflight validation instead of agent-invented ffmpeg flags.

**Chosen.** Build a typed Python wrapper over kinocut's `kino` CLI, with our own
configuration, path safety and render verification on top.

**Trade-off accepted.** kinocut has a bus factor of one (1208 of its commits are
from a single author) and is ~6 months old, so its API may move. We pin
`kinocut==1.15.1` and keep every call behind `kino.py`, so a breaking change hits
one module rather than the whole codebase.

---

## 2026-09-03 — Drive kinocut through its CLI, not its MCP server

**Context.** kinocut exposes 196 MCP tools and 167 CLI commands over the same
engine.

**Chosen.** The CLI, invoked as a subprocess from `kino.py`.

**Trade-off accepted.** MCP would validate parameters before execution; the CLI
does not, so we validate in our own recipes. In exchange, tool definitions cost
no context on every message, output is trivially testable, and the same calls
work from code and from the shell.

---

## 2026-09-03 — Static ffmpeg in `bin/` instead of Homebrew

**Context.** ffmpeg is mandatory; the machine had neither ffmpeg nor Homebrew.

**Chosen.** Download pinned arm64 binaries into `./bin` via
`scripts/fetch-ffmpeg.sh`, with SHA256 verification.

**Trade-off accepted.** Updates are manual and the binaries are macOS arm64 only,
so the project is not portable as-is. In exchange nothing is installed globally,
the ffmpeg version is pinned and reproducible, and setup does not depend on a
package manager. Note that evermeet.cx, the best-known macOS source, ships
x86_64 builds that would have run under Rosetta; the arm64 builds come from
osxexperts.net.

---

## 2026-09-03 — Build the music bed with ffmpeg, not `kino audio-bed`

**Context.** `kino audio-bed` does exactly what we need — sidechain ducking plus
EBU R128 normalisation in one pass — but it cannot run on macOS. kinocut 1.15.1
gates it behind immutable source snapshots built on `os.memfd_create`, a
Linux-only API, and it fails with `source_identity_changed` before touching any
media. Verified directly: `hasattr(os, "memfd_create")` is `False` on Darwin.
Only `audio-bed` and `body-swap` depend on it.

**Chosen.** Implement ducking in `recipes/audio_bed.py` with ffmpeg's own
`sidechaincompress` and `loudnorm`, through a new `ffmpeg.py` module.

**Trade-off accepted.** One recipe no longer goes through kinocut, so the
"everything behind `kino.py`" rule now has a second sanctioned exit through
`ffmpeg.py`. The alternative — `kino add-audio --mix` — works but does no
ducking, which was the point. Measured result: -16.04 LUFS against a -16.0
target.

---

## 2026-09-03 — Write cutouts as ProRes 4444 `.mov`, not VP9 WebM

**Context.** `hyperframes-remove-background` defaults to WebM. That file really
does carry alpha, but only ffmpeg's `libvpx-vp9` decoder exposes it, and
kinocut's compositor does not request that decoder. The result composited the
subject as an opaque rectangle over the backdrop — a wrong result that looked
entirely plausible and that probing could not distinguish from a correct one.

**Chosen.** Ask for `.mov`, which the tool renders as ProRes 4444
(`yuva444p12le`); ffmpeg reads that alpha natively.

**Trade-off accepted.** ProRes is very large — 214 MB for 6.4 seconds — so
`work/` grows quickly and the test suite got noticeably slower. Correctness over
disk.

---

## 2026-09-03 — Clamp the compositor canvas to 25 fps

**Context.** kinocut's compositor renders at most 25 fps but tags the output
with whatever fps the canvas asked for. Measured across durations and rates:
a request for `duration x fps` frames always yields `duration x min(fps, 25)`.
A 30 fps source therefore came out 5.37s instead of 6.43s.

**Chosen.** `backdrop.py` clamps the canvas fps to 25 and reports the clamp.

**Trade-off accepted.** Composited output moves at 25 fps rather than the
source's frame rate. Running time and audio sync stay correct, which matters
more for short social clips than the extra 5 fps.

---

## 2026-09-03 — Verify that alpha varies, not merely that it exists

**Context.** The first alpha check only asked whether a file had an alpha
channel. A matte that is uniformly opaque separates nothing, yet produces a
file that passes that check.

**Chosen.** `ffmpeg.measure_alpha_spread` measures the actual min/max of the
alpha plane on one frame; `cutout` fails the render when the spread is zero.

**Trade-off accepted.** One extra ffmpeg pass per cutout. Worth it: this is the
failure mode that is hardest to notice by eye.

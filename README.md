# media-lab

Local video and photo editing for short social clips, built as a typed wrapper
over [Kinocut](https://github.com/KyaniteLabs/kinocut).

Everything runs on this machine. No cloud services, no API keys, no uploads.
Sources are never modified and no render is reported as done until it has been
probed and checked against what was asked for.

- Working contract: [REGULI.md](./REGULI.md)
- Implementation plan and deviations: [PLAN.md](./PLAN.md)

## What it does

| Command              | What it does                                                           |
| -------------------- | ---------------------------------------------------------------------- |
| `media-lab doctor`   | Report the resolved environment and run kinocut's own checks           |
| `media-lab cutout`   | Cut a person out of a still or video, keeping alpha                    |
| `media-lab matte`    | Matte a person out of a video with RVM (ProRes 4444 + flicker score)   |
| `media-lab backdrop` | Composite a cutout onto a new image or video backdrop                  |
| `media-lab upscale`      | Upscale video or PNG sequence with Real-ESRGAN (x2/x4, alpha-aware)    |
| `media-lab ground`       | Place cutout on canvas with foot-pinning and 3-layer contact shadow    |
| `media-lab scale-plate`  | Calculate scale factor and ground line from plate reference person    |
| `media-lab colour-match` | Transfer colour mood (Reinhard Lab) and apply directional relighting   |
| `media-lab compose`      | Render a shot from a compose-spec YAML (bg, subject, occlusion, grade) |
| `media-lab filter`       | Apply one of ten named looks, or chain two                             |
| `media-lab proxy`        | Fast low-res proxy + contact sheet of a render, optional side-by-side  |
| `media-lab music`        | Mix a music bed under the voice with sidechain ducking                 |
| `media-lab short`        | Reframe to 9:16 (or another ratio), export, quality-gate, thumbnail    |
| `media-lab stems`        | Separate audio stems (Demucs) or clean speech in a video clip          |
| `media-lab subtitles`    | Transcribe speech (Whisper) and style/burn dynamic social subtitles   |
| `media-lab photo`        | Batch/single photo edit, social crop, backdrop swap & colour match     |
| `media-lab inspect`      | Deep media inspection returning ground-truth JSON for autonomous agents|
| `media-lab edit`         | Declarative prompt-to-edit orchestrator (flags or edit-spec YAML)      |
| `media-lab audio-enhance`| Studio vocal mastering (rumble filter, EQ, de-esser, compand, LUFS)    |
| `media-lab cut-silence`  | Automatic dead-time and pause trimming (jump-cut video/audio)          |
| `media-lab sfx`          | Synthesize and mix procedural sound effects (whoosh, pop, ding, impact)|
| `media-lab smart-reframe`| Smart vertical 9:16 reframe with face/subject tracking or split-blur   |
| `media-lab punch-zoom`   | Dynamic retention punch-in camera zooms (1.1x–1.25x)                   |
| `media-lab broll`        | B-roll cutaways overlay preserving primary dialogue track (L/J-cut)    |
| `media-lab text-overlay` | Styled typography title badge and lower-thirds with pill background    |
| `media-lab inpaint`      | Content-aware object/blemish erasing (Telea / Navier-Stokes)           |
| `media-lab retouch`      | Portrait retouching: skin smoothing and bokeh depth-of-field blur      |
| `media-lab prompt`       | Autonomous agent: prompt-to-edit in natural language (RO & EN)         |
| `media-lab mcp`          | Model Context Protocol (MCP) server over stdio for AI assistant pairing|
| `media-lab pipeline`     | Run the whole edit in one pass                                         |
| `media-lab punto`        | Reproduce the punto v23 render through matte/compose/proxy             |
| `media-lab clean`        | Empty `work/` (add `--dry-run` to preview first)                       |

Available looks: `warm`, `cool`, `vintage`, `cinematic`, `noir`, `vignette`,
`glow`, `grain`, `vibrant`, `punchy`.

`matte` and `punto` need the RVM checkout: `./scripts/fetch-rvm.sh` clones it to
`tools/` and downloads the weights. See `docs/video-agent/` for the toolkit's
design (`SPEC.md`, `PLAN.md`) and the compose-spec format (`punto-v23.yaml`).

## Requirements

| Component | Version               | Notes                                      |
| --------- | --------------------- | ------------------------------------------ |
| macOS     | arm64 (Apple Silicon) | the bundled ffmpeg binaries are arm64 only |
| Python    | 3.12                  | installed automatically by `uv`            |
| Node.js   | 22+                   | needed by the Hyperframes CLI              |
| uv        | any recent            | https://docs.astral.sh/uv/                 |

Pinned: `kinocut==1.15.1`, `hyperframes@0.8.27`, static `ffmpeg`/`ffprobe` 9.0
(checksums pinned in `scripts/fetch-ffmpeg.sh`).

## Install

```sh
cd media-lab
./scripts/fetch-ffmpeg.sh   # static ffmpeg + ffprobe into ./bin
make setup                  # uv sync + npm install + basicsr shim
cp .env.example .env
make doctor                 # verify the environment
```

`bin/`, `.venv/` and `node_modules/` are gitignored; those three commands
recreate them from pinned versions. `make setup` also runs
`scripts/patch-basicsr-shim.sh` (see Known limitations).

## Use it

Put your footage in `in/`. Renders land in `out/`, intermediates in `work/`.

```sh
# the whole edit in one pass
uv run media-lab pipeline in/clip.mp4 \
    --bg in/backdrop.png \
    --look cinematic \
    --track in/song.mp3 \
    -o out/final.mp4

# or one stage at a time
uv run media-lab cutout   in/clip.mp4 -o out/cutout.webm
uv run media-lab backdrop out/cutout.webm --bg in/bg.png -o out/composed.mp4
uv run media-lab filter   out/composed.mp4 --look warm --then grain -o out/graded.mp4
uv run media-lab music    out/graded.mp4 --track in/song.mp3 -o out/mixed.mp4
uv run media-lab short    out/mixed.mp4 -o out/final.mp4

# video-agent compositing workflow (Phase 1 & Phase 2)
uv run media-lab matte    in/clip.mp4 -o work/matte.mov
uv run media-lab upscale  work/matte.mov -o work/matte-up.mov --scale 2
uv run media-lab scale-plate work/matte-up.mov --ref-height 480 --ref-ground 3560
uv run media-lab colour-match work/matte-up.mov -o work/relit.mov --bg in/bg.mp4
uv run media-lab ground   work/relit.mov -o work/placed.mov --ground-y 3560
uv run media-lab compose  work/shot.yaml -o out/final.mp4
uv run media-lab proxy    out/final.mp4 --compare out/prev.mp4

# autonomous agent & extensions workflow (Phase 3)
# 1. Separate audio stems or clean speech
uv run media-lab stems in/podcast.mp4 -o out/stems/ --two-stems vocals
uv run media-lab stems in/noisy_vlog.mp4 -o out/stems/ --clean-speech -c out/cleaned.mp4

# 2. Transcribe & burn dynamic social subtitles (tiktok, clean, box)
uv run media-lab subtitles in/clip.mp4 -o out/subtitled.mp4 --style tiktok --burn

# 3. Social photo editing (1:1 feed, 4:5 portrait, 9:16 story) with Reinhard backdrop swap
uv run media-lab photo in/model.jpg -o out/portrait.jpg --target 4:5 --bg in/studio.jpg --look warm

# 4. Deep inspection for agent decision-making
uv run media-lab inspect in/clip.mp4 --json

# 5. Declarative prompt-to-edit orchestration (one-pass audio cleaning, grading, reframing, subs, music)
uv run media-lab edit in/clip.mp4 -o out/reel.mp4 --target 9:16 --clean-speech --look cinematic --subtitles --sub-style tiktok --music in/beat.mp3
uv run media-lab edit --spec spec.yaml

# omnichannel autonomous studio (Phase 4)
# 1. Audio studio & speech mastering
uv run media-lab audio-enhance in/podcast.mp3 -o out/mastered.mp3 --profile podcast
uv run media-lab cut-silence   in/raw_talk.mp4 -o out/jumpcut.mp4 --min-silence 0.4
uv run media-lab sfx           in/teaser.mp4   -o out/punctuated.mp4 --cue "whoosh@0.0" --cue "impact@2.5"

# 2. Video pacing & reframing
uv run media-lab smart-reframe in/interview.mp4 -o out/vertical.mp4 --aspect 9:16 --mode smart
uv run media-lab punch-zoom    in/monologue.mp4 -o out/dynamic.mp4 --interval 4.0 --scale 1.15
uv run media-lab broll         in/host.mp4      -o out/with_broll.mp4 --cut in/demo.mp4@2.0:3.5

# 3. Photo & graphics studio
uv run media-lab text-overlay in/cover.jpg -o out/badge.jpg --text "Breaking News" --position top --badge
uv run media-lab inpaint      in/photo.png -o out/clean.png --bbox 120,80,60,40
uv run media-lab retouch      in/portrait.jpg -o out/glow.jpg --smooth 0.6 --depth-blur --radiance 0.2

# 4. Prompt agent & MCP server
uv run media-lab prompt in/vlog.mp4 -o out/viral.mp4 -p "transforma in short 9:16 cu subtitrari galbene, curata vocea, scoate pauzele si pune titlul 'Podcast #1' sus"
uv run media-lab prompt in/vlog.mp4 -p "make it a reel with tiktok subtitles and whoosh sound effect" --dry-run
uv run media-lab mcp
```

The pipeline runs: cutout, backdrop, look, restore the voice the compositor
drops, music bed, vertical export. Stages you do not ask for are skipped —
without `--bg` there is no cutout, and a clip that only needs reframing keeps
its original audio untouched.

Nothing overwrites an existing file unless you pass `--force`, and no render
can be written outside the project directory at all.

`pipeline` and `short` both accept `--fail-on-warning`, which turns a failed
quality gate into a non-zero exit instead of a printed warning.

## Layout

```
bin/       static ffmpeg + ffprobe (gitignored, see scripts/fetch-ffmpeg.sh)
tools/     third-party checkouts, e.g. RVM (gitignored, see scripts/fetch-rvm.sh)
in/        source media - READ ONLY, never modified (gitignored)
out/       renders (gitignored)
work/      pipeline intermediates, kept for inspection (gitignored)
scripts/   setup helpers
src/       package source
tests/     test suite
```

Inside `src/media_lab/`: `config.py` validates the environment at startup,
`kino.py` is the only module that shells out to kinocut, `ffmpeg.py` the only
one that shells out to ffmpeg, `probe.py` reads facts with ffprobe,
`verify.py` asserts a render matches expectations, `paths.py` enforces the
non-destructive contract, and `recipes/` holds one module per editing step.

## Development

```sh
make test        # pytest with coverage
make lint        # ruff
make typecheck   # mypy strict
make check       # all of the above
```

Test media is synthesised with ffmpeg at run time; no fixtures are committed.
The suite performs real renders, so it takes a few minutes.

## Known limitations

- **`kino audio-bed` does not work on macOS.** kinocut 1.15.1 gates it behind
  immutable source snapshots built on `os.memfd_create`, a Linux-only API, so
  it fails with `source_identity_changed` before touching any media. The music
  bed is therefore built with ffmpeg's own `sidechaincompress` + `loudnorm`
  instead, in `recipes/audio_bed.py`. Same result, different engine.
- **Object and product cutouts are not available.** They need the
  `kinocut[object-matte]` extra, which is not in the published 1.15.0 wheel.
  Person cutouts (`u2net_human_seg`) work and are hardware-accelerated
  through CoreML.
- **Cutouts are written as ProRes 4444 `.mov`, not WebM.** VP9-in-WebM does
  carry alpha, but only ffmpeg's `libvpx-vp9` decoder exposes it, and
  kinocut's compositor does not request that decoder - it would silently
  composite the subject as an opaque rectangle over the backdrop. ProRes
  alpha is read natively. The files are large; they live in `work/`.
- **The compositor caps at 25 fps.** kinocut 1.15.1 renders at most 25 fps but
  tags the output with whatever the canvas asked for, so a 30 fps source came
  out 5.37s instead of 6.43s. `backdrop` clamps the canvas fps and says so.
  Motion is 25 fps; the running time stays correct.
- **The cutout model keeps people, not what they hold.** `u2net_human_seg`
  segments the person; a sign, a product or a prop in their hands is cut away
  with the background. There is no person-plus-object model in this install.
- **Cutout speed** is roughly 125 ms per frame on an M2, so a 30-second clip
  at 30 fps takes about two minutes.
- `video-body-swap` exists in kinocut but is deliberately not exposed here.
- **`basicsr` needs a torchvision shim.** `kinocut[upscale]` pulls `basicsr`,
  which imports `torchvision.transforms.functional_tensor` - removed in
  torchvision 0.17. Without it, `import basicsr` / `realesrgan` and any
  `kino *upscale*` call fail at import. `make setup` runs
  `scripts/patch-basicsr-shim.sh`, which writes a re-export module into the
  venv; the file is under the gitignored `.venv`, so the script makes the
  patch reproducible. It is idempotent and safe to re-run.

## Environment variables

All configuration lives in `.env`; see [.env.example](./.env.example). This
project holds no secrets — the variables are directory paths and timeouts.

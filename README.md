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
| `media-lab compose`  | Render a shot from a compose-spec YAML (bg, subject, occlusion, grade) |
| `media-lab filter`   | Apply one of ten named looks, or chain two                             |
| `media-lab proxy`    | Fast low-res proxy + contact sheet of a render, optional side-by-side  |
| `media-lab music`    | Mix a music bed under the voice with sidechain ducking                 |
| `media-lab short`    | Reframe to 9:16 (or another ratio), export, quality-gate, thumbnail    |
| `media-lab pipeline` | Run the whole edit in one pass                                         |
| `media-lab punto`    | Reproduce the punto v23 render through matte/compose/proxy             |
| `media-lab clean`    | Empty `work/` (add `--dry-run` to preview first)                       |

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

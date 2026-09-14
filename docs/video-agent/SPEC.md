# SPEC — video-agent toolkit, Phase 1

Scope of this document: Phase 1 of the video-agent roadmap only
(`compose-spec`, `proxy-preview`, `matte-video`). Phases 2–4 are out of scope
and listed under Non-goals. Read `ROADMAP.md` for the whole picture and
`../punto-edit.md` (v1→v23) for the pain each tool removes.

---

## Problem

Compositing one 6.4-second clip (`punto.mp4`: a person holding a handwritten
sign, dropped into new locations) took 23 hand-driven iterations. Three
activities dominated the wasted time:

1. **Matting.** `rembg`/isnet flickered, dropped interior holes, and silently
   dropped the held sign; the failure was only caught on visual review.
   RVM (Robust Video Matting) fixed it but was found late.
2. **Hand-written ffmpeg filtergraphs.** Every version meant editing a long
   `compose_pipeline.sh` by hand, hitting `zsh` word-split bugs, the
   `fade=t=in:st=N` whites-out-before-N bug, and re-deriving the depth-occlusion
   `geq` each time.
3. **Preview loop.** Each version was a ~8-minute encode followed by a
   hand-built contact sheet and a manual read.

Without the toolkit, the next similar job repeats all three.

## Users

One user (the repo owner), on one machine (macOS, Apple Silicon, MPS, no CUDA),
driving the tools from the Claude Code chat. One job at a time. No CI, no
unattended/headless runs, no concurrency, no other consumers. No personal data,
no secrets, no network calls.

## Primary flow (the one thing to get right)

Reproduce the `punto` v23 render **in one pass** through the three new tools,
instead of 23 hand-driven iterations:

```
in/punto-source.mp4
  → matte-video (RVM)              → work/…/matte.mov      (ProRes 4444 RGBA)
  → upscale_realesrgan.py          → work/…/up/            (unchanged script)
  → place_composite.py             → work/…/placed/        (unchanged script)
  → compose-spec (YAML → ffmpeg)   → out/punto-final…mp4   (bg-prep, overlay,
                                                            depth occlusion,
                                                            v23 grade, encode)
  → proxy-preview                  → work/…/sheet.jpg      (contact sheet, and
                                                            side-by-side vs the
                                                            reference render)
```

A `--proxy` flag on the runner takes a fast path (RVM mobilenetv3, skip
upscale, 540 px proxy encode) for iteration; the default path is full quality.

## MVP

The smallest version that removes the three pains above:

- **`matte-video`** — `media-lab matte in/clip.mp4 -o work/matte.mov`.
  Single backend: RVM. `--model {resnet50,mobilenetv3}` (default resnet50).
  Output: ProRes 4444 `.mov` with alpha. Reports an **alpha-stability score**
  (temporal variance of the alpha plane; higher variance between adjacent
  frames = more flicker). Verified with `verify_render` + `measure_alpha_spread`.
- **`compose-spec`** — `media-lab compose spec.yaml -o out/final.mp4`.
  Loads a YAML shot spec, builds the ffmpeg filtergraph, writes it to a
  sidecar file for inspection, and runs it. Covers: background prep
  (trim/scale/fps), overlay of the pre-placed full-canvas subject sequence at
  `0:0`, parametrised depth-occlusion strip (bottom band of the bg re-overlaid
  with a feathered top edge), a fixed **"v23" grade profile** (on/off + a few
  scalar knobs: atmosphere opacity, grain, vignette), and the final encode.
  Handles the `zsh`-quoting and `fade` pitfalls by construction. Does **not**
  position or scale the subject (that stays upstream).
- **`proxy-preview`** — `media-lab proxy out/final.mp4 [--compare ref.mp4]`.
  Fast low-res proxy encode + an N-frame contact sheet. With `--compare`, an
  additional side-by-side sheet against the named file. No auto-discovery of a
  "previous version".
- **`punto` runner** — `media-lab punto -o out/punto-final.mp4 [--proxy]`.
  Chains the flow above. The final render is checked against the v23 container
  spec and a contact sheet is emitted for a visual check.

Definition of done for Phase 1:

- Each tool has its own test suite: happy path + ≥2 edge cases + 1 error case,
  `make check` green (ruff + mypy strict + pytest), coverage ≥ 80% on the new
  modules. ML calls are mocked by default; one opt-in slow test runs real RVM
  on a ~10-frame synthetic clip when the RVM clone and weights are present.
- `media-lab punto` reproduces the v23 render: the output matches
  `out/punto-final_2160x3840_30fps_h264-crf17.mp4` on **container spec**
  (2160×3840, 30 fps, ~6.43 s ± 0.1, no audio, H.264 High, yuv420p) as checked
  by `verify-render`, and the contact sheet is a visual match on inspection.
  **Not** bit-exact.
- `README.md` (root + `docs/video-agent/`), `.env.example`, `PLAN.md`
  checkboxes and `DECISIONS.md` updated.

## Non-goals (explicit)

- Multi-backend matting, backend auto-pick, `matte-qc`, held-object-survival
  check (Phase 2/3).
- `upscale` as a productised tool — the existing `upscale_realesrgan.py` script
  is called unchanged.
- `subject-ground` / foot-lock / relight as a tool — `place_composite.py` is
  called unchanged.
- IC-Light / generative relight — known hardware ceiling, does not run on MPS.
- `footage-scout`, `edit-planner`, `shot-compositor`, `render-doctor`,
  `deliver`, `scale-from-plate`, `colour-match`, `smart-retime`, and the
  `media-lab` orchestrator agent (Phases 2–4).
- Keyframe animation / camera move in `compose-spec` (v10+ dropped camera
  moves).
- 60 fps / `minterpolate` (v14+ delivered 30 fps native).
- Any GUI, MCP server, cloud service, or external API.
- Non-macOS / non-arm64 support.
- New scoping-session code beyond the three deliverable docs.

## Deferred (and what would block it)

- **Productising `upscale` and `subject-ground`.** Deferred to Phase 2.
  Blocking decision now: the `punto` runner stages files into the exact
  `work/punto-edit/…` paths those scripts hardcode — which do **not** chain
  as-is (audit M2: upscale writes `isnet/up/`, place reads `rvm_up/` or
  `isnet/cut/`), so the runner also moves `isnet/up/` → `rvm_up/` between the
  two. If Phase 2 parametrises the scripts, this staging shim is deleted.
- **`matte-video` multi-backend.** Deferred to Phase 2. The Phase-1 recipe
  keeps the backend behind a `--model` flag on one code path, so adding an
  `isnet` branch later is additive, not a rewrite.
- **A dedicated contact-sheet skill.** `proxy-preview` and the `punto` runner
  share a `contact_sheet` helper module now; promoting it to a standalone
  skill later is a wrapper, not new logic.

## Data model (entities, in prose)

No database. The "entities" are files on disk and one config object.

- **Config** — resolved from `.env` + environment at startup
  (`src/media_lab/config.py`). Phase 1 adds two fields for ML paths
  (`rvm_repo`, `weights_dir`), **validated lazily** — only when `matte-video`
  runs, so `doctor` and the non-ML commands work without the RVM clone or
  weights present. Source of truth: `.env` (git-ignored), documented in
  `.env.example`.
- **Source clip** — under `in/`, read-only, never modified. Source of truth:
  the user's `in/` directory. `in/punto-source.mp4` is the Phase-1 fixture
  (720×1280, 30 fps, 193 frames, 6.433 s, AAC audio).
- **Shot spec** — a YAML file the user writes (or the `punto` runner emits).
  Fields: background (path, trim start, target w/h/fps), subject sequence
  (path/glob of pre-placed RGBA frames), occlusion strip (height, feather,
  y-position), grade (`profile: v23 | none`, `atmosphere`, `grain`,
  `vignette`), output (path, codec/crf). Source of truth: the file itself,
  kept in `work/` or beside the output.
- **Intermediates** — `matte.mov`, upscaled frames, placed frames, the
  generated `.filtergraph.txt` sidecar. Under `work/`, kept for inspection,
  addressed by the final render's stem. Source of truth: regenerated on each
  run; safe to delete with `media-lab clean`.
- **Render** — under `out/`. Never overwritten without `--force`. Source of
  truth: the last successful run.
- **Reference render** — `out/punto-final_2160x3840_30fps_h264-crf17.mp4`
  (the committed-by-description v23 result). Used read-only by `--compare` and
  the acceptance check.
- **Reports** — the alpha-stability score and the `verify_render` result are
  returned in-process and printed by the CLI; not persisted.

## Ugly cases

- **Missing ML setup** (RVM clone or weights absent) — `matte-video` raises a
  typed `MlEnvError` naming the missing piece and the `scripts/fetch-rvm.sh`
  command that installs it. `doctor` reports ML state as "not configured"
  without failing. Other commands are unaffected.
- **Missing / empty source, output already exists, output outside the
  project** — handled by the existing `paths.py` contract (unchanged).
- **RVM drops the held sign** — Phase 1 does not detect this
  (`matte-qc` is Phase 3). The alpha-stability score does not catch it. The
  contact sheet from `proxy-preview` is the only guard, same as v23.
- **`compose-spec` given a subject sequence that is not full-canvas** — the
  filtergraph overlays at `0:0`; a mis-sized sequence renders mis-framed.
  `compose-spec` checks the first frame's dimensions against the canvas and
  raises before rendering.
- **Operation interrupted mid-run** — each stage writes to `work/` and is
  verified before the next starts (existing pipeline pattern); a failure names
  the stage and leaves completed intermediates on disk. No partial `out/` file
  (ffmpeg writes to a temp name via the existing `run_ffmpeg`).
- **Two runs at once** — not supported, not guarded. Single-user assumption.
  Intermediates are addressed by the output stem, so two runs targeting
  different outputs do not collide; two runs targeting the same output race on
  `work/` files (documented, not prevented).
- **`/tmp/RVM` is the only RVM checkout** — `/tmp` is cleared on reboot.
  `scripts/fetch-rvm.sh` clones to `tools/RobustVideoMatting/` (git-ignored)
  so the checkout survives; the spike migrates the existing `/tmp/RVM`.

## Non-functional requirements

- **Local only.** No network, no API keys, no uploads. Enforced by having no
  code path that opens a socket.
- **Non-destructive.** `in/` is never written; `out/` is never overwritten
  without `--force`; nothing is written outside the project. Enforced by the
  existing `paths.py`.
- **Speed.** Full `punto` run is ~25–30 min (RVM ~12 s/193 frames;
  Real-ESRGAN ~7.5 s/frame ≈ 24 min; compose ~2–3 min). `--proxy` path is
  ~2–3 min total. Neither number is a hard SLA — the point is the `--proxy`
  path exists for iteration.
- **Honesty of reporting.** No render is reported done before `verify_render`
  passes. The alpha-stability score and any `verify` warnings are always
  printed.
- **Reproducibility.** The RVM clone (`tools/RobustVideoMatting/`) and the
  weights are git-ignored but recreatable from `scripts/fetch-rvm.sh` +
  documented commands, the same contract as `bin/`. `torch` etc. are already
  pinned in `uv.lock` via `kinocut[upscale]`.
- **Failure isolation.** RVM runs as a child process behind a single module
  (`ml_runner.py`), so GPL-3 code is never imported into `media_lab` and a
  torch/RVM break hits one module — the same containment `kino.py` gives
  kinocut and `ffmpeg.py` gives direct ffmpeg.

## Success criteria

1. `make check` green; coverage ≥ 80% on `matte_video.py`, `compose_spec.py`,
   `proxy_preview.py`, `ml_runner.py`, `punto_v23.py`.
2. `media-lab matte in/punto-source.mp4 -o work/m.mov` produces a valid RGBA
   ProRes with a non-zero alpha spread and a printed stability score.
3. `media-lab compose <spec> -o out/x.mp4` produces the expected resolution /
   fps / duration, with an inspectable `.filtergraph.txt` sidecar.
4. `media-lab proxy out/x.mp4 --compare out/punto-final_…mp4` produces a
   proxy + a side-by-side contact sheet.
5. `media-lab punto -o out/punto-final.mp4` reproduces the v23 render to
   container spec + visual match on the contact sheet.
6. `docs/`, `README.md`, `.env.example` updated; `PLAN.md` steps checked off.

## Assumptions

- **A1** — one user, local, one job at a time, no CI, no headless runs.
- **A2** — RVM is a git-ignored source checkout at `tools/RobustVideoMatting/`
  (cloned by `scripts/fetch-rvm.sh`), put on `sys.path` by the child-process
  driver. No dedicated venv: `torch` / `torchvision` are already in `.venv`
  via `kinocut[upscale]`. The child process exists for the GPL-3 boundary.
- **A3** — weights stay where they are now
  (`work/punto-edit/gen/weights/{rvm_resnet50,rvm_mobilenetv3,RealESRGAN_x2plus}.pth`);
  the default `MEDIA_LAB_WEIGHTS_DIR` points there. Provenance (GitHub
  releases) is documented; the files never enter git.
- **A4** — RVM is cloned to `tools/RobustVideoMatting/` (git-ignored), not
  vendored, because it is GPL-3.0.
- **A5** — "reproduce v23" = container-spec match + visual contact-sheet
  match, not bit-exact.
- **A6** — `upscale_realesrgan.py` and `place_composite.py` are used
  unchanged; the `punto` runner stages inputs into the `work/punto-edit/…`
  paths they hardcode and bridges the gap between them (`isnet/up/` →
  `rvm_up/`). See PLAN Step 7.
- **A7** — `gh` and web tools are not needed in Phase 1.
- **A8** — the existing 25 fps compositor clamp in `backdrop.py` is
  irrelevant here: `compose-spec` runs ffmpeg directly, not
  `kino composite-layers`.

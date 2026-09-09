# DECISIONS.md

Technical decisions taken on this project, with the trade-off each one accepts.
Newest first.

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

## 2026-09-09 — Run RVM through a separate venv via subprocess

**Context.** `matte-video` wraps Robust Video Matting. RVM needs
`torch` + `torchvision`; the project `.venv` deliberately carries neither, and
RVM is **GPL-3.0** — vendoring its source into this (otherwise permissively
licensed) repo would be a licensing problem.

**Chosen.** Clone RVM to `tools/RobustVideoMatting/` (git-ignored), build a
git-ignored `.rvm-venv` at the project root, and invoke it only through a new
`ml_runner.py` module — the single sanctioned exit to that venv, exactly as
`kino.py` is for kinocut and `ffmpeg.py` for direct ffmpeg. Our driver code
that imports RVM (`src/media_lab/ml/rvm_infer.py`) runs inside `.rvm-venv` as a
subprocess.

**Trade-off accepted.** A third external process boundary and a second
recreatable-but-not-committed environment (alongside `bin/` and `.venv/`).
Data crosses the boundary as an RGBA PNG sequence + a JSON stats file rather
than in-process tensors. In exchange, torch stays out of `uv sync`, mypy strict
and ruff never see it, GPL code never enters the repo, and a torch/RVM break
hits one module.

---

## 2026-09-09 — ML paths are lazily-validated config, not startup-validated

**Context.** `config.py` validates everything at startup so the project fails
loudly rather than mid-render. The new `rvm_venv` / `rvm_repo` / `weights_dir`
paths only matter to `matte-video`.

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

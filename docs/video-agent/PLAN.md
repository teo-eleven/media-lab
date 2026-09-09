# PLAN — video-agent toolkit, Phase 1

Implements the Phase 1 slice of `ROADMAP.md`: `matte-video`, `compose-spec`,
`proxy-preview`, plus a `punto` runner that proves the three replace the v23
hand-driven pipeline. Read `SPEC.md` first.

Working contract: `../../REGULI.md`. One step at a time; each step ends with a
report and waits for confirmation. Every step is vertical — the project builds
and `make check` passes after it. Commits are atomic, conventional
(`feat:` / `fix:` / `chore:` / `docs:`), and end with the `Claude-Session:`
trailer.

Code, comments, commit messages: English. All new code extends the
`media_lab` package on branch `feat/video-agent-toolkit`.

---

## Architecture at a glance

| New file                                 | Role                                                                                                                                                                                                                                                                              |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/media_lab/ml_runner.py`             | The **only** module that runs an ML driver as a child process (project interpreter, `sys.executable`). Timeout, stderr capture, typed `MlEnvError`. Sibling of `kino.py` / `ffmpeg.py`. The child process is the GPL boundary — RVM is never imported into `media_lab`'s process. |
| `src/media_lab/ml/rvm_infer.py`          | Our driver script, run as a child process. `sys.path.insert(0, rvm_repo)`, imports RVM's `MattingNetwork`, writes an RGBA PNG sequence + a JSON stats blob (per-frame alpha mean/var).                                                                                            |
| `src/media_lab/recipes/matte_video.py`   | `matte-video` recipe: orchestrates `rvm_infer` via `ml_runner`, muxes ProRes 4444 `.mov`, computes the alpha-stability score, verifies.                                                                                                                                           |
| `src/media_lab/compose_spec.py`          | Pure filtergraph builder + YAML schema + validation. No I/O.                                                                                                                                                                                                                      |
| `src/media_lab/recipes/compose_spec.py`  | `compose-spec` recipe: load YAML → build → write `.filtergraph.txt` sidecar → run via `ffmpeg.py` → verify.                                                                                                                                                                       |
| `src/media_lab/contact_sheet.py`         | Shared: extract N frames, tile a sheet, tile a side-by-side. Used by proxy-preview and the punto runner.                                                                                                                                                                          |
| `src/media_lab/recipes/proxy_preview.py` | `proxy-preview` recipe: low-res proxy encode + contact sheet + optional `--compare` side-by-side.                                                                                                                                                                                 |
| `src/media_lab/recipes/punto_v23.py`     | `punto` runner: matte-video → upscale (unchanged script) → place (unchanged script) → compose-spec → contact sheet; `--proxy` fast path.                                                                                                                                          |
| `scripts/fetch-rvm.sh`                   | Clone RVM to `tools/RobustVideoMatting/` (git-ignored, GPL-3, never committed) and print the weights-download commands. No venv. Sibling of `scripts/fetch-ffmpeg.sh`.                                                                                                            |

| Changed file                                                            | Change                                                                                                          |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `src/media_lab/config.py`                                               | Add `rvm_repo`, `weights_dir` fields + `.env` keys. Lazily validated (a `require_ml()` helper), not at startup. |
| `src/media_lab/ffmpeg.py`                                               | Add `run_filtergraph(...)` helper (build `-filter_complex` + `-map` + encode args, go through `run_ffmpeg`).    |
| `src/media_lab/errors.py`                                               | Add `MlEnvError(MediaLabError)`, `SpecError(MediaLabError)`.                                                    |
| `src/media_lab/cli.py`                                                  | Add subcommands: `matte`, `compose`, `proxy`, `punto`. Extend `doctor` to print ML state.                       |
| `.env.example`, `README.md`, `docs/video-agent/README.md`, `.gitignore` | `tools/` ignored; new env vars documented.                                                                      |
| `DECISIONS.md` (root)                                                   | New dated entries as decisions land.                                                                            |

`pyproject.toml` gains one **runtime** dependency: **`pyyaml`** (compose-spec
parses YAML at run time), plus `types-PyYAML` in the dev group for mypy.
`torch`, `torchvision`, `basicsr` and `realesrgan` are **already** in `.venv`
and `uv.lock` via `kinocut[upscale]` — no new ML dependency, and no separate
venv. `rvm_infer.py` runs in a child process only to keep GPL-3 RVM code out
of `media_lab`'s own process.

---

## Spikes (do these first — they de-risk the rest)

### S0 — ML environment boundary · ~2–3 h · agent: `debugger` for the RVM run, otherwise manual

**Question it answers:** can we drive RVM headless as a child process using the
project interpreter (torch is already in `.venv`), with the RVM clone on
`sys.path`, and what is a sound alpha-stability score formula?

**Do:** migrate `/tmp/RVM` → `tools/RobustVideoMatting/`. Write a throwaway
script in `work/` that: takes a ~10-frame synthetic clip
(`ffmpeg lavfi testsrc`), runs RVM (`rvm_mobilenetv3.pth`) via
`subprocess.run([sys.executable, driver, ...])` where `driver` does
`sys.path.insert(0, "tools/RobustVideoMatting")` + `from model import
MattingNetwork`, writes RGBA PNGs, and prints per-frame alpha mean/variance and
a candidate single-number score. Confirm the frames carry a real matte.

**Output:** a note in this file (S0 findings) fixing: the `sys.path` / import
shape for RVM, the score formula, and whether `resnet50` is feasible on the
test machine in test time.

**Done when:** RGBA PNGs exist for the synthetic clip and the score is
computed; no project code written.

**S0 findings (run 2026-09-09, `work/spike-s0/`, 12 real punto frames):**

- Child-process mechanism **works**: `subprocess.run([sys.executable,
rvm_driver.py, rvm_repo, weights, model, in, out])`; the driver does
  `sys.path.insert(0, rvm_repo)` then `from model import MattingNetwork`
  (RVM's `model/__init__.py` re-exports it). `torch` / RVM are never imported
  in the orchestrator process. `torch.load(..., weights_only=True)` (RVM
  weights are pure state dicts — avoids the pickle-RCE warning).
- Real matte: alpha spread 255, fg fraction ~0.076/frame, `alpha_mean` drifts
  17.4 → 18.7 smoothly across 12 frames (subject moving, no flicker).
- **Score formula:** `stability_score` = mean absolute frame-to-frame alpha
  delta over ever-foreground pixels (`|Δα|` on pixels where `max_t α > 10`),
  in 0–255 units, lower = stabler. RVM/mobilenetv3 on this clip: **~6.5**.
  (All-pixel delta ~0.57 is diluted by the static-zero background; temporal
  std ~19 conflates real subject motion with flicker — rejected.) Phase 1
  only reports the number; a threshold gate is `matte-qc` (Phase 3).
- **Timing:** mobilenetv3 12 frames = 6.8 s cold (model load + MPS warmup
  dominate). Steady-state ~0.06 s/frame matches the README. `resnet50` not
  yet timed on this machine — Step 3's opt-in slow test will; mechanism is
  identical.
- Migration `/tmp/RVM` → `tools/RobustVideoMatting/` is Step 1
  (`scripts/fetch-rvm.sh`); the spike ran against `/tmp/RVM` directly.

### S1 — filtergraph port · ~2–3 h · agent: manual (ffmpeg work), then

`python-pro` review of the builder sketch

**Question it answers:** does a parametrised Python builder reproduce the v23
`compose_pipeline.sh` filtergraph — including the depth-occlusion `geq` and the
grade chain — closely enough on ffmpeg 9, and are the `zsh`/`fade` pitfalls
actually gone once it is a Python list of args?

**Do:** take the three ffmpeg invocations in
`docs/video-agent/pipeline/compose_pipeline.sh` (bg prep, composite with
occlusion, grade+encode). Sketch a builder that emits the same
`-filter_complex` string from parameters. Run it against the existing v23
intermediates in `work/punto-edit/` (or regenerate a short bg + a handful of
placed frames). Diff a few extracted frames against
`out/punto-final_2160x3840_30fps_h264-crf17.mp4` visually.

**Output:** S1 findings here: the parameter set the builder needs, the exact
occlusion `geq` expression, the grade profile constants lifted verbatim from
v23, and any ffmpeg-9 behaviour differences.

**Done when:** a throwaway builder produces a filtergraph that renders a
visually-matching short clip; no project code written.

**S1 findings (run 2026-09-09, `work/spike-s1/build.py`, real v23 inputs:
`in/backgrounds/nyc-wallst.mp4` + `work/punto-edit/isnet/placed/` 193 frames):**

- A frozen `Spec` dataclass → 3 arg lists (bg-prep, composite+occlusion,
  grade+encode). Runs on ffmpeg 9 **unchanged**. Wall time ~1m40s for 193
  frames at 2160×3840.
- Output: 2160×3840, 30 fps, 193 frames, 6.433 s, yuv420p, 64 MB — **exact
  container match** to `out/punto-final_2160x3840_30fps_h264-crf17.mp4`
  (66 MB; the delta is x264 + noise seed).
- Frame fidelity vs the real v23 render (4 samples): `mean|Δ| ≈ 2.6/255`,
  `p99|Δ| ≈ 9–10`, no diff spike at `y=3710` (the occlusion-strip seam) →
  the graph is structurally faithful; the residual is `noise=alls=4:allf=t`
  running a different seed. **Bit-exact is impossible; use `mean|Δ| < ~5/255`
  on sampled frames as the numeric half of the Step 7 acceptance, alongside
  the contact-sheet eyeball (confirms SPEC A5).**
- zsh/`fade` pitfalls: gone — it is a `list[str]`, no shell; v23 uses hard
  cuts, no `fade`.
- Occlusion `geq` ports verbatim: on `crop={w}:{occ_h}:0:{occ_y}`,
  `a='if(lt(Y,{feather}),255*Y/{feather},255)'`. Params: `occ_h=130`,
  `occ_y=3710`, `occ_feather=70`.
- Grade chain ported verbatim (atmosphere `format=gbrp,split` →
  `scale=iw/6:ih/6,gblur=sigma=7,scale=W:H:flags=bilinear,eq=brightness=0.03`
  → `blend=all_mode=screen:all_opacity=0.07` → `eq` / `colorbalance` /
  `curves=master='0/0.01 0.5/0.5 1/0.993'` / `unsharp=5:5:0.24` /
  `vignette=PI/5.8` / `noise=alls=4:allf=t` → `format=yuv420p`). **No
  ffmpeg-9 behaviour differences observed.** Step 4 moves these constants
  into `GRADE_PROFILES["v23"]`; the spike proves the emitted string is right.

Spikes are throwaway (`work/`), not committed. Findings are appended to this
file and folded into the steps below.

---

## Steps

Dependency order: **1 → 2 → 3**; **4 → 5**; **6**; then **7 → 8**.
Parallelisable: **4** and **6** are independent of the ML chain (2–3) and of
each other — they can be delegated to `fork` subagents running alongside step 3.

---

### Step 1 — Config for ML paths + `fetch-rvm.sh` + `doctor` ML state

**Depends on:** S0.

**Build:** `config.py` gains `rvm_repo`, `weights_dir` (defaults:
`./tools/RobustVideoMatting`, `./work/punto-edit/gen/weights`, resolved like
the existing dir fields). A `require_ml(config)` helper checks both exist (and
that the expected weight files are under `weights_dir`) and raises `MlEnvError`
with the `fetch-rvm.sh` hint — **called by recipes, never at startup**.
`scripts/fetch-rvm.sh`: clone RVM to `tools/RobustVideoMatting/` (GPL-3, kept
out of git) and echo the weights URLs — no venv (torch is already in `.venv`).
`doctor` prints `rvm repo`, `weights` with an `(ok)` / `(not configured)`
marker. `.gitignore` += `tools/`.

**Files:** `config.py`, `errors.py`, `cli.py` (`doctor`), `scripts/fetch-rvm.sh`,
`.env.example`, `.gitignore`.

**Tests:** `tests/test_config.py` extended — new fields parsed from env;
defaults; `require_ml` raises with the right message when a path is missing;
`require_ml` passes when both present (use `tmp_path` dirs with stub weight
files). `doctor` output contains the ML lines (extend `tests/test_cli.py`).

**Agent:** manual + `python-reviewer` at the end.

**Vertical check:** `make check` green; `media-lab doctor` runs and shows ML
state on a machine without the RVM clone.

---

### Step 2 — `ml_runner.py` (single sanctioned exit for ML subprocesses)

**Depends on:** 1.

**Build:** `MlRunner` dataclass, `from_config`, `.run(script, args, *, timeout_s)`
→ `subprocess.run([sys.executable, script, *args], …)` with the project env,
capture stderr, raise `MlEnvError` on non-zero / timeout. Mirrors `KinoRunner`.
No RVM knowledge here — it just runs a python script as a child process. The
child boundary exists so GPL-3 RVM code is never imported into `media_lab`'s
own process and so `torch` is not imported unless matting runs.

**Files:** `ml_runner.py`, `errors.py` (if not already in 1).

**Tests:** `tests/test_ml_runner.py` — mock `subprocess.run`: success returns
stdout; non-zero raises `MlEnvError` with stderr tail; `TimeoutExpired` raises
`MlEnvError` naming the timeout. One test uses a real 2-line python script in
`tmp_path` executed by the _project_ interpreter (via a `MlRunner` pointed at
`sys.executable`) to prove the happy path end to end.

**Agent:** `python-pro` (writes cleanly against the `KinoRunner` pattern),
`python-reviewer` after.

**Vertical check:** `make check` green.

---

### Step 3 — `matte-video` recipe + `rvm_infer.py` + CLI `matte`

**Depends on:** 2.

**Build:** `src/media_lab/ml/rvm_infer.py` (run as a child process; first line
after imports: `sys.path.insert(0, <rvm_repo>)`): argv = frames-dir in, dir
out, model name, weights path, rvm-repo path; writes `f-%04d.png` RGBA +
`stats.json` (`[{frame, alpha_mean, alpha_var}]`). Logic lifted from
`pipeline/matte_rvm.py`, parametrised, no hardcoded paths (drop its
`sys.path.insert(0, "/tmp/RVM")` and hardcoded weights/glob).
`recipes/matte_video.py`: `matte_video(source, output, config, ml_runner, *,
model="resnet50", force=False)` → `require_ml` → explode source to PNGs under
`work/<stem>-frames/` via `run_ffmpeg` → `ml_runner.run("ml/rvm_infer.py", …)`
→ mux `work/<stem>-rvm/f-%04d.png` to ProRes 4444 `.mov` → `verify_render`
(`requires_alpha=True`, `requires_audio=False`, duration from source) →
`measure_alpha_spread` (reuse) → compute **alpha-stability score** from
`stats.json` (S0 formula). Return a frozen `MatteResult`
(media, model, frame count, alpha_spread, stability_score). CLI `matte` prints
them.

**Files:** `ml/__init__.py`, `ml/rvm_infer.py`, `recipes/matte_video.py`,
`cli.py`.

**Tests:** `tests/test_matte_video.py` — **default: mock `MlRunner.run`** to
drop a canned RGBA PNG sequence + `stats.json` into the expected dir, then
assert the `.mov` is built, alpha is non-uniform, the score matches the canned
stats, and `force` / missing-source / bad-model paths behave. **Opt-in:** a
`@pytest.mark.slow` test (skipped unless the RVM clone + weights present) that
runs real `rvm_infer.py` on a 10-frame `testsrc` clip.

**Agent:** `python-pro` for the recipe; `tdd-guide` for the test file;
`python-reviewer` + `security-reviewer` (subprocess + path handling) after.

**Vertical check:** `make check` green (mocked); manual
`media-lab matte in/punto-source.mp4 -o work/m.mov` on the real machine
produces a valid matte + a printed score.

---

### Step 4 — `compose_spec.py` filtergraph builder (pure) + YAML schema

**Depends on:** S1. **Parallel with 2–3.**

**Build:** `compose_spec.py` — a `ComposeSpec` frozen dataclass mirroring the
YAML (background, subject, occlusion, grade, output), a `load_spec(path)` that
parses YAML and validates (raises `SpecError` with field-named messages), and
`build_filtergraph(spec) -> FilterGraph` (the `-filter_complex` string + the
`-map` target + encode args). Grade is a `GRADE_PROFILES = {"v23": {...}}`
table with the constants from S1; the spec picks `profile` + scalar overrides
for atmosphere opacity / grain / vignette. Depth-occlusion strip built from
`height` / `feather` / `y`. No file I/O, no ffmpeg call.

**Files:** `compose_spec.py`, `errors.py` (`SpecError`).

**Tests:** `tests/test_compose_spec.py` — golden filtergraph string for a
canonical spec; `profile: none` omits the grade chain; occlusion math
(feather expression) for two sets of params; `SpecError` on missing
`background.path`, on unknown `grade.profile`, on negative dimensions; a spec
round-trips through `load_spec` from a YAML file in `tmp_path`.

**Agent:** `python-pro` (delegate as a `fork` subagent alongside step 3);
`python-reviewer` after.

**Vertical check:** `make check` green.

---

### Step 5 — `compose-spec` recipe + CLI `compose` + `ffmpeg.run_filtergraph`

**Depends on:** 4.

**Build:** `ffmpeg.py` gains `run_filtergraph(inputs, filtergraph, map_target,
encode_args, output, config)` → assembles the arg list, calls `run_ffmpeg`.
`recipes/compose_spec.py`: `compose(spec_path, output, config, *, force=False)`
→ `load_spec` → resolve + `ensure_readable_source` every input path → check the
first subject frame's dimensions against the canvas (raise `SpecError` if
mismatched) → `build_filtergraph` → write `<output>.filtergraph.txt` sidecar in
`work/` → `run_filtergraph` → `verify_render` (w/h/fps/duration from the spec,
`requires_audio=False`). Return `ComposeResult` (media, spec_path,
filtergraph_path). CLI `compose` prints them.

**Files:** `ffmpeg.py`, `recipes/compose_spec.py`, `cli.py`.

**Tests:** `tests/test_compose_recipe.py` — synthesise a small bg
(`color`/`testsrc`, e.g. 256×455 @30) + a short pre-placed RGBA PNG sequence
(same size) with ffmpeg; a minimal spec → real render → assert resolution /
fps / duration and that the sidecar file exists and is non-empty; mismatched
subject size raises `SpecError`; existing output without `force` raises
`PathSafetyError`; `profile: none` still renders.

**Agent:** `python-pro`; `python-reviewer` after.

**Vertical check:** `make check` green;
`media-lab compose <spec> -o out/x.mp4` works on real inputs.

---

### Step 6 — `contact_sheet.py` + `proxy-preview` recipe + CLI `proxy`

**Depends on:** 1. **Parallel with 2–5.**

**Build:** `contact_sheet.py` — `extract_frames(src, n, config) -> list[Path]`
(even spacing via `run_ffmpeg` `-ss`), `tile(frames, cols) -> Path`,
`side_by_side(sheet_a, sheet_b) -> Path`. `recipes/proxy_preview.py`:
`proxy_preview(source, config, *, height=540, frames=8, compare=None,
force=False)` → proxy encode (`scale=-2:height`, fast x264) to
`work/<stem>-proxy.mp4` → `verify_render` → contact sheet to
`work/<stem>-sheet.jpg` → if `compare`: proxy+sheet it too and `side_by_side`
to `work/<stem>-vs.jpg`. Return `ProxyResult` (proxy, sheet, comparison|None).
CLI `proxy` prints the paths.

**Files:** `contact_sheet.py`, `recipes/proxy_preview.py`, `cli.py`.

**Tests:** `tests/test_proxy_preview.py` — synthetic clip → proxy exists at the
right height, sheet exists with expected dimensions (cols × frame size), frame
count honoured; `--compare` against a second synthetic clip produces the
side-by-side; `force` guards the proxy.

**Agent:** `python-pro` (delegate as a `fork` subagent); `python-reviewer`
after.

**Vertical check:** `make check` green;
`media-lab proxy out/punto-final_…mp4 --compare …` produces both sheets.

---

### Step 7 — `punto` runner (the acceptance) + CLI `punto`

**Depends on:** 3, 5, 6.

**Build:** `recipes/punto_v23.py`: `run_punto(config, ml_runner, output, *,
proxy=False, force=False)`.

The two unchanged scripts have hardcoded, mutually inconsistent I/O dirs (audit
M2) — the runner stages around them precisely (audit M3):

- `upscale_realesrgan.py` reads `work/punto-edit/isnet/cut/`, writes
  `work/punto-edit/isnet/up/`.
- `place_composite.py` reads `work/punto-edit/rvm_up/` **iff** it exists and
  holds > 100 files, else falls back to `work/punto-edit/isnet/cut/`; writes
  `work/punto-edit/isnet/placed/`.

Full path: `matte_video(in/punto-source.mp4)` → write its RGBA frames to
`work/punto-edit/isnet/cut/f-%04d.png` (720×1280) → `ml_runner.run` on
`pipeline/upscale_realesrgan.py` → **move** `work/punto-edit/isnet/up/` →
`work/punto-edit/rvm_up/` so `place_composite.py` picks the upscaled frames →
run `pipeline/place_composite.py` → emit a `punto-v23.yaml` spec pointing at
`work/punto-edit/isnet/placed/` + `in/backgrounds/nyc-wallst.mp4` with the v23
occlusion + grade → `compose(...)` → `contact_sheet` vs
`out/punto-final_2160x3840_30fps_h264-crf17.mp4` → `verify_render` against the
v23 container spec.

`--proxy`: `model="mobilenetv3"`, **skip upscale and the `rvm_up/` move** —
`place_composite.py` then falls back to the 720×1280 `isnet/cut/` frames, so
the placed subject is ~half v23 scale (acceptable for a rough preview; the
runner prints this caveat). `height=540` in the spec; skip the reference
compare.

The two scripts (`upscale_realesrgan.py`, `place_composite.py`) are called
**as-is** via `ml_runner` from the repo root (they resolve `work/…` relative
to cwd). This staging shim is the documented Phase-2 cleanup point (fold them
into real `upscale` / `subject-ground` tools).

**Files:** `recipes/punto_v23.py`, `cli.py`, a checked-in
`docs/video-agent/punto-v23.yaml` template.

**Tests:** `tests/test_punto_v23.py` — `--proxy` path with `MlRunner.run`
mocked (canned matte frames), on a trimmed synthetic stand-in for the source,
asserting the chain runs and the final `verify_render` spec is applied; an
`@pytest.mark.slow` full-path test guarded on ML availability + the real
source file.

**Agent:** `fullstack-developer` or `python-pro`; `code-reviewer` +
`python-reviewer` after; `debugger` on demand for the real end-to-end run.

**Vertical check:** `make check` green (mocked); on the real machine,
`media-lab punto -o out/punto-final.mp4` reproduces v23 to container spec +
visual match.

---

### Step 8 — Docs sync + review pass

**Depends on:** 1–7.

**Build:** `README.md` (root) — new commands table rows, `tools/` in the
layout, an ML-setup subsection. `docs/video-agent/README.md` — point the
"Pipeline" section at the new commands. `.env.example` — the two ML vars (done
in step 1, re-check). `PLAN.md` — tick every step, record deviations.
`DECISIONS.md` — fold in anything the spikes changed. Grep the repo for stale
references (old script paths, `/tmp/RVM`).

**Files:** the docs above.

**Tests:** none new; `make check` stays green. Manually follow the README
ML-setup steps from scratch (clone RVM to `tools/`, download weights).

**Agent:** `doc-updater`; final `code-review` skill over the whole branch.

**Vertical check:** a new reader can set up ML and run `media-lab punto` from
the README alone.

---

## Out of scope for this plan

Everything under SPEC "Non-goals". Notably: no changes to `upscale_realesrgan.py`
or `place_composite.py`; no multi-backend matting; no orchestrator agent; no
skill/subagent wrappers (Phase 4).

## Progress

- [x] S0 — ML environment boundary spike (findings above; `work/spike-s0/`)
- [x] S1 — filtergraph port spike (findings above; `work/spike-s1/`)
- [x] Step 1 — config + fetch-rvm.sh + doctor
- [ ] Step 2 — ml_runner.py

### Deviations

- **Step 1:** `scripts/fetch-rvm.sh` also `curl`s the two weight files (skipping
  any present), rather than only printing the URLs — more useful, idempotent,
  and it means `media-lab doctor` goes straight to `(ok)`. No checksum pinning:
  RVM publishes none; the script says so.
- [ ] Step 3 — matte-video + rvm_infer + CLI
- [ ] Step 4 — compose_spec builder (pure)
- [ ] Step 5 — compose-spec recipe + CLI
- [ ] Step 6 — contact_sheet + proxy-preview + CLI
- [ ] Step 7 — punto runner + CLI
- [ ] Step 8 — docs sync + review

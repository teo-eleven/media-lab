# Video-agent roadmap

A toolkit of small subagents + tools to build here, then wire into a big
**`media-lab`** orchestrator agent. Every item below is tied to a concrete place
where the `punto` compositing job (v1→v23, see `../punto-edit.md`) went slow or
wrong. Build **incrementally** — validate on the punto clip after each phase.

---

## Subagents (`~/.claude/agents/`)

| Agent               | Purpose                                                                                                                                                                                                                                                          | Pain it removes                                                              |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **footage-scout**   | Given a shot brief (location, orientation, motion, mood), query Pexels / Pixabay / Mixkit / Coverr, filter by orientation + duration + real-time-vs-timelapse heuristics, download candidates, build a contact sheet, return verified direct `.mp4` URLs ranked. | 2 manual agent runs + guessing CDN URLs this session; WebFetch kept failing. |
| **edit-planner**    | Turn a vague ask ("put her in Times Square walking through a crowd") into a concrete shot spec: plate type, scale reference, ground line, occlusion depth, grade direction, deliverable specs.                                                                   | Ambiguous asks → I guessed and redid.                                        |
| **shot-compositor** | The workhorse. Runs matte → upscale → scale-match → ground → colour-match → compose → grade → QC, returns a proxy + report. Orchestrates the tools below.                                                                                                        | I hand-drove 23 iterations.                                                  |
| **render-doctor**   | Given a finished render + the brief, diagnose what's off (floating feet, edge halo, colour mismatch, retime warping, matte holes) and name the exact parameter to change. A second pair of eyes.                                                                 | I eyeballed every contact sheet.                                             |

---

## Tools / skills

| Tool                 | Purpose                                                                                                                                                                                                                                                                                                                      | Pain it removes                                                                  |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **matte-video**      | Multi-backend matting: RVM (default), isnet + temporal cleanup, BiRefNet, MatAnyone. Auto-pick by "does the subject hold/wear something detached". Output: clean RGBA sequence + an alpha-stability score.                                                                                                                   | isnet flickered / dropped holes / dropped the sign; found RVM late.              |
| **matte-qc**         | Measure matte quality: temporal alpha variance (flicker), interior-hole count, edge spill, whether a held object survived. Gate the pipeline.                                                                                                                                                                                | Didn't catch the sign being dropped until visual review.                         |
| **subject-ground**   | Foot detection + rigid/soft foot-pin + 3-layer contact shadow + optional depth-occlusion strip lifted from the plate. One call, params for how buried.                                                                                                                                                                       | The entire feet-float saga.                                                      |
| **scale-from-plate** | Measure a reference person in the background plate (pixel height at a given y), return the subject scale + feet-Y so they sit at that depth.                                                                                                                                                                                 | Scale/depth mismatch every version → "walks over people".                        |
| **colour-match**     | Statistical transfer (mean/std per channel in Lab) from a bg region onto the subject + contrast/saturation match + grain match. Optional IC-Light hook when a CUDA box is configured.                                                                                                                                        | Manual `eq` / `colorbalance` tuning per scene.                                   |
| **smart-retime**     | Detect timelapse/hyperlapse (optical-flow magnitude vs framerate); if slowing is needed, pick frame-blend vs optical-flow vs "accept as-is"; warn when retiming will warp people.                                                                                                                                            | Hyperlapse → warped / fading pedestrians.                                        |
| **compose-spec**     | Take a YAML shot spec (bg, subject seq, occlusion strip, grade, output) → emit AND run the ffmpeg filtergraph. Handles the zsh-quoting / fade pitfalls. _(Phase 1: subject overlaid at 0:0 on pre-placed full-canvas frames — no x/y/scale keyframes; those wait until a keyframe/camera-move need is real. See `SPEC.md`.)_ | Hand-written filtergraphs; zsh word-split bugs; `fade` whites-out-before-st bug. |
| **proxy-preview**    | Fast low-res proxy render + auto contact-sheet (N frames) + side-by-side vs the previous version.                                                                                                                                                                                                                            | Every version = ~8 min encode + manual contact sheet + Read.                     |
| **upscale**          | Real-ESRGAN (or better), with the `basicsr` `functional_tensor` shim baked in, RGBA-aware (upscale RGB, carry alpha).                                                                                                                                                                                                        | basicsr breakage, manual shim, per-run boilerplate.                              |
| **text-plate**       | Recreate a sign/caption as a clean RGBA PNG (font, board style, diacritics) or key a real one from the source; return the overlay + a tracking path.                                                                                                                                                                         | Placard PNG + luma-key experiments.                                              |
| **deliver**          | Final encode presets per platform (IG reel, TikTok, YT Short); QC against spec (res / fps / duration / audio / aspect); copy to a findable location; generate a preview. Extends `verify-render`.                                                                                                                            | Manual `ffprobe` + verify each time; user couldn't find the output file.         |
| **assets-ledger**    | Track which source / plate / weights a render used, licences, and where intermediates live — so `work/` can be cleaned safely and a render reproduced.                                                                                                                                                                       | Orphan files, unclear provenance.                                                |

---

## Orchestrator

**`media-lab`** — routes a natural request:

```
edit-planner → footage-scout (if a new plate is needed)
             → shot-compositor
                 matte-video → upscale → scale-from-plate → subject-ground
                 → colour-match → compose-spec → [proxy-preview loop] → deliver
             (render-doctor in the loop)
```

Human approval gates: **plate choice**, **first proxy**, **final**.

---

## Build order

- **Phase 1** — unblocks ~80% of the pain: `compose-spec`, `proxy-preview`, `matte-video` (wrap RVM).
- **Phase 2** — `scale-from-plate`, `subject-ground`, `colour-match`, `footage-scout`.
- **Phase 3** — `smart-retime`, `matte-qc`, `render-doctor`, `deliver`.
- **Phase 4** — `shot-compositor` + `edit-planner`, then wire into `media-lab`.

**Tradeoff:** ~16 pieces. Building all the interfaces before an end-to-end run risks
over-engineering. After Phase 1, redo the punto clip in a single pass with those
three tools, then let what still hurts decide the rest.

---

## Known hard limit (not a tool problem)

Foot–ground contact and true light integration ("can't tell it's a composite")
need frame-by-frame rotoscoping or a GPU relight pass (IC-Light on CUDA, or a
cloud service). The tools above get to "a clean, produced composite" — not
photoreal — on this hardware. Plan a `media-lab` path that offloads the relight
step to a remote GPU when one is configured.

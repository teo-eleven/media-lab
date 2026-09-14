"""Natural language prompt interpreter and autonomous editing agent.

Translates human user requests (in Romanian or English) into fully structured,
validated declarative EditSpecs and executes them seamlessly across audio, video,
and photo pipelines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .edit_spec import (
    AudioEditSpec,
    EditSpec,
    PhotoEditSpec,
    SubtitleEditSpec,
    TypographyEditSpec,
    VideoEditSpec,
)
from .kino import KinoRunner
from .ml_runner import MlRunner
from .recipes.edit import EditResult, run_edit_spec

ROMANIAN_ENHANCE_KEYWORDS = {
    "curata vocea",
    "curăță vocea",
    "voce clara",
    "voce clară",
    "mastering",
    "elimina zgomotul",
    "elimină zgomotul",
}
ENGLISH_ENHANCE_KEYWORDS = {
    "clean speech",
    "enhance voice",
    "master audio",
    "denoise",
    "studio voice",
}

ROMANIAN_SILENCE_KEYWORDS = {
    "taie pauzele",
    "elimina tacerile",
    "elimină tăcerile",
    "jump cut",
    "fara pauze",
    "fără pauze",
}
ENGLISH_SILENCE_KEYWORDS = {"cut silence", "trim silence", "remove pauses", "jump cut"}

ROMANIAN_RECFRAME_KEYWORDS = {"short", "tiktok", "reels", "vertical", "9:16"}
ENGLISH_RECFRAME_KEYWORDS = {"short", "vertical", "reels", "tiktok", "9:16"}


@dataclass(frozen=True, slots=True)
class PromptPlan:
    """Summary of operations derived from natural language prompt."""

    prompt: str
    operations: tuple[str, ...]
    spec: EditSpec


def interpret_prompt(
    prompt: str,
    source: str | Path,
    output: str | Path,
) -> EditSpec:
    """Parse natural language editing prompt into a structured EditSpec.

    Supports natural phrasing in Romanian and English:
    - 'fă-mi un short cu subtitrări galbene, taie pauzele și curăță vocea'
    - 'make a vertical short, cut silence, master voice to podcast, and apply cinematic look'
    - 'retușează portretul: netezește tenul, pune titlu "Nou Episod" jos și adaugă bokeh'
    """
    p_lower = prompt.lower()

    # 1. Video aspect & reframing
    aspect = "original"
    smart_reframe = False

    if any(k in p_lower for k in ("9:16", "short", "reels", "tiktok", "vertical")):
        aspect = "9:16"
        smart_reframe = True
    elif "1:1" in p_lower or "patrat" in p_lower or "pătrat" in p_lower or "square" in p_lower:
        aspect = "1:1"
        smart_reframe = True
    elif "4:5" in p_lower or "portrait" in p_lower:
        aspect = "4:5"
        smart_reframe = True

    # 2. Subtitles
    subtitles_enabled = any(k in p_lower for k in ("subtitr", "subtitle", "caption"))
    sub_style = "tiktok"
    if "galben" in p_lower or "yellow" in p_lower:
        sub_style = "boxed_yellow"
    elif "cinematic" in p_lower and subtitles_enabled:
        sub_style = "cinematic"
    elif "minimal" in p_lower:
        sub_style = "minimal"

    # 3. Visual Look
    look: str | None = None
    if "cinematic" in p_lower and not (
        subtitles_enabled and "subtitr" in p_lower and "cinematic" in p_lower
    ):
        look = "cinematic"
    elif "cald" in p_lower or "warm" in p_lower:
        look = "warm"
    elif "rece" in p_lower or "cool" in p_lower:
        look = "cool"
    elif "vintage" in p_lower:
        look = "vintage"
    elif "noir" in p_lower or "alb-negru" in p_lower or "black and white" in p_lower:
        look = "noir"

    # 4. Audio: Clean speech & Vocal Mastering
    clean_speech = any(k in p_lower for k in ROMANIAN_ENHANCE_KEYWORDS | ENGLISH_ENHANCE_KEYWORDS)
    master_profile: str | None = None
    if "podcast" in p_lower:
        master_profile = "podcast"
    elif "radio" in p_lower:
        master_profile = "radio"
    elif "crisp" in p_lower or "clar" in p_lower:
        master_profile = "crisp"
    elif clean_speech:
        master_profile = "podcast"

    # 5. Silence Jump-cut
    silence_trim = any(k in p_lower for k in ROMANIAN_SILENCE_KEYWORDS | ENGLISH_SILENCE_KEYWORDS)

    # 6. Retention Punch-in Zoom
    punch_zoom = any(k in p_lower for k in ("punch zoom", "zoom", "retention", "apropiere"))
    zoom_interval = 5.0 if punch_zoom else None

    # 7. Typography Badge / Title
    typo_spec: TypographyEditSpec | None = None
    title_match = re.search(r'["\']([^"\']+)["\']', prompt)
    if any(k in p_lower for k in ("titlu", "text", "badge", "title", "caption")) or title_match:
        text_content = title_match.group(1) if title_match else "Media Lab"
        pos = "bottom"
        if "sus" in p_lower or "top" in p_lower:
            pos = "top"
        elif "mijloc" in p_lower or "center" in p_lower:
            pos = "center"
        elif "jos" in p_lower or "bottom" in p_lower:
            pos = "bottom"
        elif subtitles_enabled:
            # Default to top when subtitles are active to prevent badge collision
            pos = "top"
        typo_spec = TypographyEditSpec(
            text=text_content,
            position=pos,
            badge=True,
        )

    # 8. Sound Effects (SFX)
    sfx_cues: list[dict[str, Any]] = []
    if "whoosh" in p_lower:
        sfx_cues.append({"kind": "whoosh", "at": 0.0})
    if "pop" in p_lower:
        sfx_cues.append({"kind": "pop", "at": 0.5})
    if "ding" in p_lower:
        sfx_cues.append({"kind": "ding", "at": 1.0})
    if "impact" in p_lower:
        sfx_cues.append({"kind": "impact", "at": 0.0})

    # 9. Photo Retouch & Bokeh
    retouch = any(
        k in p_lower for k in ("retus", "retuș", "retouch", "piele", "ten", "smooth skin")
    )
    depth_blur = any(k in p_lower for k in ("bokeh", "defocus", "blur fundal", "background blur"))
    skin_strength = 0.6 if retouch else 0.0
    bokeh_sigma = 15.0 if depth_blur else 0.0
    radiance = (
        0.2 if ("radiance" in p_lower or "stralucire" in p_lower or "lumina" in p_lower) else 0.0
    )

    # 10. Speed Ramping
    speed_factor = 1.0
    if any(
        k in p_lower for k in ("slow motion", "slow-mo", "slowmo", "incetineste", "încetinește")
    ):
        speed_factor = 0.5
    elif any(
        k in p_lower
        for k in ("timelapse", "time-lapse", "fast forward", "accelereaza", "accelerează", "repede")
    ):
        speed_factor = 2.0
    speed_match = re.search(r"\b([0-9.]+)x\b", p_lower)
    if speed_match:
        try:
            val = float(speed_match.group(1))
            if 0.1 <= val <= 10.0:
                speed_factor = val
        except ValueError:
            pass

    # 11. Social Retention Progress Bar
    has_progress_bar = any(
        k in p_lower
        for k in (
            "progress bar",
            "bara de progres",
            "bara progres",
            "bara galbena",
            "bara rosie",
            "bară de progres",
        )
    )
    pb_color = "yellow"
    if "rosie" in p_lower or "red" in p_lower or "roșie" in p_lower:
        pb_color = "red"
    elif "alba" in p_lower or "white" in p_lower or "albă" in p_lower:
        pb_color = "white"
    elif "tiktok" in p_lower and "bara" in p_lower:
        pb_color = "tiktok"
    elif "cyan" in p_lower or "albastru" in p_lower:
        pb_color = "cyan"
    elif "verde" in p_lower or "green" in p_lower:
        pb_color = "green"

    pb_position = (
        "top" if (("sus" in p_lower or "top" in p_lower) and "bara" in p_lower) else "bottom"
    )

    video_spec = VideoEditSpec(
        aspect=aspect,
        smart_reframe=smart_reframe,
        reframe_mode="smart",
        punch_zoom=punch_zoom,
        zoom_interval=zoom_interval,
        look=look,
        subtitles=SubtitleEditSpec(enabled=subtitles_enabled, style=sub_style),
        typography=typo_spec,
        speed=speed_factor,
        progress_bar=has_progress_bar,
        progress_bar_color=pb_color,
        progress_bar_position=pb_position,
    )

    audio_spec = AudioEditSpec(
        clean_speech=clean_speech,
        master_profile=master_profile,
        silence_trim=silence_trim,
        sfx_cues=tuple(sfx_cues),
    )

    photo_spec = PhotoEditSpec(
        retouch=retouch or depth_blur,
        skin_strength=skin_strength,
        depth_blur=depth_blur,
        bokeh_sigma=bokeh_sigma,
        radiance=radiance,
    )

    return EditSpec(
        source=str(source),
        output=str(output),
        video=video_spec,
        audio=audio_spec,
        photo=photo_spec,
    )


def plan_prompt(prompt: str, source: str | Path, output: str | Path) -> PromptPlan:
    """Generate an inspectable plan of actions from a prompt without executing it."""
    spec = interpret_prompt(prompt, source, output)
    ops: list[str] = []

    if spec.audio.silence_trim:
        ops.append("Tăiere pauze/silențiu (jump-cut)")
    if spec.audio.clean_speech:
        ops.append("Izolare vocală curată (Demucs speech stem)")
    if spec.audio.master_profile:
        ops.append(f"Mastering audio vocal (profil {spec.audio.master_profile})")
    if spec.video.look:
        ops.append(f"Colorizare cinematică (look {spec.video.look})")
    if spec.video.smart_reframe:
        ops.append(f"Reîncadrare inteligentă {spec.video.aspect} cu urmărire subiect")
    elif spec.video.aspect != "original":
        ops.append(f"Decupare format {spec.video.aspect}")
    if spec.video.punch_zoom:
        ops.append("Punch-in zoom dinamic pentru retenție")
    if spec.video.typography:
        ops.append(f"Titlu grafic ({spec.video.typography.text!r})")
    if spec.video.subtitles.enabled:
        ops.append(f"Subtitrări automate Whisper ({spec.video.subtitles.style})")
    if spec.audio.sfx_cues:
        ops.append(f"Efecte sonore procedurale ({len(spec.audio.sfx_cues)} efecte)")
    if spec.video.speed != 1.0:
        label = "slow-motion" if spec.video.speed < 1.0 else "timelapse"
        ops.append(f"Modificare viteză ({spec.video.speed}x {label})")
    if spec.video.progress_bar:
        ops.append(
            f"Bară animată de retenție ({spec.video.progress_bar_color} "
            f"la {spec.video.progress_bar_position})"
        )
    if spec.photo.retouch:
        ops.append("Retușare ten și netezire facială")
    if spec.photo.depth_blur:
        ops.append("Simulare profunzime de câmp (bokeh)")

    if not ops:
        ops.append("Conversie și verificare de bază")

    return PromptPlan(prompt=prompt, operations=tuple(ops), spec=spec)


def execute_prompt(
    prompt: str,
    source: str | Path,
    output: str | Path,
    config: Config,
    runner: KinoRunner,
    ml_runner: MlRunner,
    *,
    force: bool = False,
) -> EditResult:
    """Interpret prompt and immediately execute the entire edit pipeline."""
    spec = interpret_prompt(prompt, source, output)
    return run_edit_spec(spec, config, runner, ml_runner, force=force)

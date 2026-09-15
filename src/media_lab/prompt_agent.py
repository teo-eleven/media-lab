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
from .probe import MediaInfo, probe
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


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Conversational reply and actionable plan from Media Lab Studio Agent."""

    reply: str
    intent: str
    plan_operations: tuple[str, ...] = ()
    suggested_prompts: tuple[str, ...] = ()
    spec: EditSpec | None = None
    executable: bool = False
    source_summary: str | None = None


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

    # 12. Geometric & Spatial Transforms (vflip, hflip, rotate, reverse, negate, grayscale)
    vflip_triggers = (
        "cu susul in jos",
        "cu susul în jos",
        "cu susu-n jos",
        "sus in jos",
        "sus în jos",
        "solul cu susul in jos",
        "solul cu susul în jos",
        "solul cu sus in jos",
        "solul cu sus în jos",
        "solul sus",
        "intoarce vertical",
        "întoarce vertical",
        "rasturnat",
        "răsturnat",
        "rastoarna",
        "răstoarnă",
        "flip vertical",
        "vflip",
        "upside down",
        "pe dos",
    )
    vflip = any(k in p_lower for k in vflip_triggers)

    hflip_triggers = (
        "oglind",
        "flip orizontal",
        "hflip",
        "mirror",
        "horizontal flip",
    )
    hflip = any(k in p_lower for k in hflip_triggers)

    rotate_deg = 0
    if "180" in p_lower and any(k in p_lower for k in ("rot", "grade", "deg")):
        rotate_deg = 180
    elif "90" in p_lower and any(k in p_lower for k in ("rot", "grade", "deg")):
        rotate_deg = 90
    elif "270" in p_lower and any(k in p_lower for k in ("rot", "grade", "deg")):
        rotate_deg = 270

    invert_colors = any(
        k in p_lower
        for k in ("inversare culori", "culori inversate", "negativ", "negative", "invert colors")
    )
    grayscale = any(
        k in p_lower for k in ("alb-negru", "alb negru", "grayscale", "monocrom", "black and white")
    )
    reverse_video = any(
        k in p_lower
        for k in (
            "inapoi",
            "înapoi",
            "reverse",
            "redare inversa",
            "redare inversă",
            "de la coada la cap",
            "de la coadă la cap",
        )
    )
    # 13. Color, Lighting, Blur & Clarity
    brightness = 0.0
    if any(k in p_lower for k in ("mai luminos", "lumineaz", "creste luminoz", "crește luminoz")):
        brightness = 0.08
    elif any(k in p_lower for k in ("mai intunecat", "mai întunecat", "intunec", "întunec")):
        brightness = -0.08

    contrast = 1.0
    if any(k in p_lower for k in ("mai mult contrast", "contrast mai mare", "creste contrast")):
        contrast = 1.25
    elif any(
        k in p_lower for k in ("mai putin contrast", "mai puțin contrast", "contrast mai mic")
    ):
        contrast = 0.80

    saturation = 1.0
    if any(k in p_lower for k in ("culori mai vii", "culori vii", "mai saturat", "vivid")):
        saturation = 1.35
    elif any(k in p_lower for k in ("culori pale", "desaturat", "mai putin colorat")):
        saturation = 0.70

    blur_val = 0.0
    if any(k in p_lower for k in ("blureaz", "estompeaz", "neclar")) or (
        "blur" in p_lower and "fundal" not in p_lower and not depth_blur
    ):
        blur_val = 4.0

    sharpen = any(k in p_lower for k in ("mai clar", "claritate", "sharpen", "detalii mai clare"))

    # 14. Audio Volume & Muting
    mute_audio = any(
        k in p_lower
        for k in ("fara sunet", "fără sunet", "fara audio", "fără audio", "mute", "mut")
    )
    volume_mult = 1.0
    if any(k in p_lower for k in ("mai tare", "volum mai mare", "creste volum", "crește volum")):
        volume_mult = 1.5
    elif any(k in p_lower for k in ("mai incet", "mai încet", "volum mai mic", "scade volum")):
        volume_mult = 0.6

    # 16. AI Neural Cutout (RVM) & Background Replacement
    cutout = any(
        k in p_lower
        for k in (
            "scoate fundalul",
            "elimina fundalul",
            "elimină fundalul",
            "fara fundal",
            "fără fundal",
            "transparent",
            "cutout",
            "rvm",
            "decupeaza",
            "decupează",
            "decupare",
            "remove background",
            "background removal",
            "izoleaza persoana",
            "izolează persoana",
        )
    )
    backdrop: str | None = None
    if any(
        k in p_lower
        for k in (
            "schimba fundalul",
            "schimbă fundalul",
            "pune fundal",
            "alt fundal",
            "inlocuieste fundalul",
            "înlocuiește fundalul",
            "replace background",
            "new background",
            "backdrop",
        )
    ):
        cutout = True
        backdrop = "in/punto-bg.png" if "punto" in p_lower else "in/demo-bg.png"

    # 17. AI Neural Super-Resolution (Real-ESRGAN Upscaling)
    upscale_val = 0
    if any(
        k in p_lower
        for k in (
            "upscale",
            "mareste rezolutia",
            "mărește rezoluția",
            "creste rezolutia",
            "crește rezoluția",
            "creste calitatea",
            "crește calitatea",
            "super rezolutie",
            "super-rezoluție",
            "super resolution",
            "realesrgan",
            "claritate maxima",
            "claritate maximă",
            "4k",
            "hd",
        )
    ):
        upscale_val = 4 if "4k" in p_lower else 2

    # 18. Local TTS Voiceover Narrator
    narrator_text: str | None = None
    narrator_voice = "Daniel" if ("english" in p_lower or "engleza" in p_lower) else "Ioana"
    if any(
        k in p_lower
        for k in (
            "narator",
            "naratiune",
            "narațiune",
            "voce de prezentare",
            "voiceover",
            "povestitor",
            "adaugă voce",
            "adauga voce",
            "spune",
        )
    ):
        narrator_match = re.search(r'["\']([^"\']+)["\']', prompt)
        if narrator_match:
            narrator_text = narrator_match.group(1)
        else:
            narrator_text = "Bun venit la producția Media Lab Studio!"

    # 19. Background Music Bed with Ducking
    music_track: str | None = None
    if any(
        k in p_lower
        for k in (
            "muzica",
            "muzică",
            "music",
            "pune muzica",
            "pune muzică",
            "adauga muzica",
            "adaugă muzică",
            "pe fundal",
            "soundtrack",
            "coloana sonora",
            "coloană sonoră",
        )
    ):
        music_track = "in/demo2-music.m4a" if "demo2" in p_lower else "in/demo-music.m4a"

    # 20. Comprehensive Studio Production autonomous trigger
    full_production_triggers = (
        "tot ce are nevoie",
        "tot ce avem",
        "tot ce am descarcat",
        "tot ce am descărcat",
        "clip complet",
        "productie completa",
        "producție completă",
        "video viral",
        "viral",
        "fa tot",
        "fă tot",
        "fa ceva misto",
        "fă ceva mișto",
        "studio 360",
        "fa-l complet",
        "fă-l complet",
    )
    if any(k in p_lower for k in full_production_triggers):
        if aspect == "original":
            aspect = "9:16"
            smart_reframe = True
        subtitles_enabled = True
        clean_speech = True
        master_profile = "podcast"
        silence_trim = True
        punch_zoom = True
        has_progress_bar = True
        sharpen = True
        if not look:
            look = "cinematic"
        if not music_track:
            music_track = "in/demo-music.m4a"

    # 15. Actionable prompt fallback
    action_roots = (
        "fa",
        "fă",
        "pune",
        "adauga",
        "adaugă",
        "modifica",
        "modifică",
        "schimba",
        "schimbă",
        "aplica",
        "aplică",
        "executa",
        "execută",
        "creeaza",
        "creează",
        "regleaza",
        "reglează",
        "ajusteaza",
        "ajustează",
        "editeaza",
        "editează",
        "proceseaza",
        "procesează",
        "transforma",
        "transformă",
        "lucru",
        "lucreaza",
        "lucrează",
    )
    has_action_verb = any(re.search(rf"\b{root}", p_lower) for root in action_roots)
    if has_action_verb and not (
        aspect != "original"
        or look
        or vflip
        or hflip
        or rotate_deg
        or invert_colors
        or grayscale
        or reverse_video
        or brightness != 0.0
        or contrast != 1.0
        or saturation != 1.0
        or blur_val > 0
        or sharpen
        or mute_audio
        or volume_mult != 1.0
        or clean_speech
        or silence_trim
        or punch_zoom
        or typo_spec
        or has_progress_bar
        or retouch
        or depth_blur
        or subtitles_enabled
        or cutout
        or backdrop
        or upscale_val > 0
        or narrator_text
        or music_track
    ):
        sharpen = True

    video_spec = VideoEditSpec(
        aspect=aspect,
        smart_reframe=smart_reframe,
        reframe_mode="smart",
        punch_zoom=punch_zoom,
        zoom_interval=zoom_interval,
        look=look,
        subtitles=SubtitleEditSpec(enabled=subtitles_enabled, style=sub_style),
        typography=typo_spec,
        cutout=cutout,
        backdrop=backdrop,
        speed=speed_factor,
        progress_bar=has_progress_bar,
        progress_bar_color=pb_color,
        progress_bar_position=pb_position,
        vflip=vflip,
        hflip=hflip,
        rotate=rotate_deg,
        invert_colors=invert_colors,
        grayscale=grayscale,
        reverse=reverse_video,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        blur=blur_val,
        sharpen=sharpen,
        mute_audio=mute_audio,
        volume_multiplier=volume_mult,
        upscale=upscale_val,
        narrator_text=narrator_text,
        narrator_voice=narrator_voice,
    )

    audio_spec = AudioEditSpec(
        clean_speech=clean_speech,
        master_profile=master_profile,
        silence_trim=silence_trim,
        music_track=music_track,
        sfx_cues=tuple(sfx_cues),
    )

    photo_spec = PhotoEditSpec(
        retouch=retouch or depth_blur,
        skin_strength=skin_strength,
        depth_blur=depth_blur,
        bokeh_sigma=bokeh_sigma,
        radiance=radiance,
        vflip=vflip,
        hflip=hflip,
        rotate=rotate_deg,
        invert_colors=invert_colors,
        grayscale=grayscale,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        blur=blur_val,
        sharpen=sharpen,
        upscale=upscale_val,
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
    if spec.video.vflip or spec.photo.vflip:
        ops.append("Răsturnare verticală (solul cu susul în jos / vflip)")
    if spec.video.hflip or spec.photo.hflip:
        ops.append("Oglindire orizontală (mirror / hflip)")
    if spec.video.rotate or spec.photo.rotate:
        deg = spec.video.rotate or spec.photo.rotate
        ops.append(f"Rotire ({deg}°)")
    if spec.video.invert_colors or spec.photo.invert_colors:
        ops.append("Inversare culori (efect negativ)")
    if spec.video.grayscale or spec.photo.grayscale:
        ops.append("Conversie alb-negru (grayscale)")
    if spec.video.reverse:
        ops.append("Redare inversă (reverse video)")
    if spec.video.brightness > 0:
        ops.append("Creștere luminozitate (+luminos)")
    elif spec.video.brightness < 0:
        ops.append("Reducere luminozitate (mai întunecat)")
    if spec.video.contrast > 1.0:
        ops.append("Amplificare contrast (+contrast)")
    elif spec.video.contrast < 1.0:
        ops.append("Reducere contrast (soft)")
    if spec.video.saturation > 1.0:
        ops.append("Culori vii și saturate (vivid)")
    elif spec.video.saturation < 1.0:
        ops.append("Desaturare culori (pastel)")
    if spec.video.blur > 0:
        ops.append("Estompare imagine (efect blur)")
    if spec.video.sharpen:
        ops.append("Optimizare claritate și detalii (sharpen)")
    if spec.video.mute_audio:
        ops.append("Eliminare pistă audio (mute)")
    elif spec.video.volume_multiplier > 1.0:
        ops.append("Creștere volum audio (+50%)")
    elif spec.video.volume_multiplier < 1.0:
        ops.append("Reducere volum audio (-40%)")
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

    if spec.video.cutout:
        ops.append("Decupare subiect AI (RVM - Robust Video Matting fără green screen)")
    if spec.video.backdrop:
        ops.append(f"Compoziție pe fundal ({spec.video.backdrop})")
    if spec.video.upscale > 0 or spec.photo.upscale > 0:
        scale = spec.video.upscale or spec.photo.upscale
        ops.append(f"Super-rezoluție AI Real-ESRGAN ({scale}x)")
    if spec.video.narrator_text:
        ops.append(f"Narator vocal local TTS ({spec.video.narrator_voice or 'Ioana'})")
    if spec.audio.music_track:
        ops.append(f"Coloană sonoră cu ducking ({spec.audio.music_track})")

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


def chat_agent(
    prompt: str,
    source: str | Path | None,
    config: Config,
) -> ChatResponse:
    """Conversational engine providing natural language interaction and plan generation."""
    p = prompt.strip()
    p_lower = p.lower()

    # Source info if available
    source_info: MediaInfo | None = None
    source_path: Path | None = None
    if source:
        try:
            sp = Path(source)
            if sp.is_file():
                source_path = sp
                source_info = probe(sp, config)
        except Exception:
            pass

    # 1. Greetings
    greetings = {
        "salut",
        "buna",
        "bună",
        "hei",
        "hello",
        "hi",
        "noroc",
        "servus",
        "buna ziua",
        "bună ziua",
        "neata",
        "neața",
        "alo",
        "start",
    }
    is_pure_greeting = p_lower in greetings or any(
        p_lower.startswith(g) and len(p_lower) <= len(g) + 5 for g in greetings
    )
    if is_pure_greeting:
        return ChatResponse(
            reply=(
                "Salut! Sunt **Media Lab Studio Assistant** 🎬 — asistentul tău creativ pentru "
                "editare video, audio și foto pe Apple Silicon.\n\n"
                "Te pot ajuta să transformi orice clip brut într-o producție optimizată pentru "
                "TikTok, Reels, Shorts, YouTube sau podcast-uri, complet local și privat.\n\n"
                "Selectează un fișier din stânga și spune-mi ce dorești să facem, sau alege o "
                "comandă rapidă de mai jos!"
            ),
            intent="greeting",
            suggested_prompts=(
                "Fă un short vertical 9:16 cu subtitrări galbene TikTok și taie pauzele",
                "Curăță vocea cu profil podcast, aplică de-esser și normalizează sunetul",
                "Adaugă o bară animată de progres galbenă la bază și zoom dinamic",
                "Analizează clipul selectat și dă-mi recomandări",
            ),
        )

    # 2. Help / Capabilities
    help_triggers = {
        "ce poti",
        "ce poți",
        "ce stii",
        "ce știi",
        "ajutor",
        "help",
        "comenzi",
        "optiuni",
        "opțiuni",
        "capabilitati",
        "capabilități",
        "cum functioneaza",
        "cum funcționează",
        "ce faci",
        "ce unelte",
        "functionalitati",
        "funcționalități",
    }
    if any(h in p_lower for h in help_triggers):
        return ChatResponse(
            reply=(
                "### 🛠️ Ce pot face pentru tine în Media Lab Studio:\n\n"
                "**1. 📱 Format & Pacing Social Media (Reels / TikTok / Shorts):**\n"
                "- Decupare verticală 9:16 sau pătrată 1:1 cu urmărire a subiectului.\n"
                "- Punch-in zoom dinamic pentru menținerea retenției audienței.\n"
                "- Bară animată de progres cu culori configurabile (galben, roșu, alb, cyan).\n"
                "- Speed ramping (slow-motion 0.5x sau timelapse 2x) cu păstrarea tonului vocii.\n"
                "- Asamblare clipuri cu 15 tranziții cinematice (wipe, fade, dissolve).\n\n"
                "**2. 🎙️ Audio Studio & Vocal Mastering:**\n"
                "- Izolare vocală AI cu Demucs v4 (extrage vorbirea curată din zgomot).\n"
                "- Mastering de voce (profil podcast, radio sau crisp, de-esser, -16 LUFS).\n"
                "- Tăiere automată a momentelor de silențiu și pauzelor lungi (jump-cuts).\n"
                "- Sinteză de efecte sonore procedurale: whoosh, pop, ding, impact.\n"
                "- Narator vocal offline în limba română (`Ioana`) și engleză (`Daniel`).\n\n"
                "**3. 🎨 Subtitrări & Grafică:**\n"
                "- Subtitrări Whisper pe cuvinte (stil TikTok, galben, minimalist).\n"
                "- Titluri grafice și badge-uri stil capsulă colorată cu umbră.\n"
                "- Retuș ten portret (skin smoothing) și adâncime de câmp (bokeh blur).\n"
                "- Inpainting pentru ștergerea watermark-urilor sau obiectelor nedorite.\n\n"
                "Alege orice combinație de mai sus și spune-mi în limbaj natural!"
            ),
            intent="help",
            suggested_prompts=(
                "Fă un short 9:16 cu subtitrări galbene TikTok, curăță vocea și taie pauzele",
                "Adaugă o bară animată de progres galbenă la bază și zoom dinamic la 4 secunde",
                "Curăță vocea cu profil podcast, aplică de-esser și normalizează sunetul",
                "Retușează tenul discret, adaugă bokeh pe fundal și un badge cu titlul sus",
            ),
        )

    # 3. File Inspection / "analizează" / "ce are clipul"
    inspect_triggers = {
        "analizeaza",
        "analizează",
        "ce are",
        "proprietati",
        "proprietăți",
        "info",
        "inspecteaza",
        "inspectează",
        "detalii",
        "rezolutie",
        "rezoluție",
        "durata",
        "durată",
    }
    if any(it in p_lower for it in inspect_triggers):
        if source_info and source_path:
            ratio_label = (
                "Vertical 9:16"
                if abs(source_info.aspect_ratio - 9 / 16) < 0.05
                else (
                    "Orizontal 16:9"
                    if abs(source_info.aspect_ratio - 16 / 9) < 0.05
                    else f"Aspect {source_info.aspect_ratio:.2f}"
                )
            )
            has_aud = "Da (pistă audio detectată)" if source_info.has_audio else "Nu (fără sunet)"
            rec1 = (
                "Decupare verticală 9:16 cu centrare pe persoană"
                if "Orizontal" in ratio_label
                else "Adăugare bară de retenție la bază"
            )
            rec2 = (
                "Mastering vocal podcast și subtitrări automate galbene"
                if source_info.has_audio
                else "Adăugare narator vocal de prezentare"
            )
            rec3 = "Tăiere automată a pauzelor (jump-cut) și punch-zoom dinamic"

            return ChatResponse(
                reply=(
                    f"### 🔍 Raport de Inspecție pentru `{source_path.name}`:\n\n"
                    f"- **Rezoluție:** {source_info.width}x{source_info.height} ({ratio_label})\n"
                    f"- **Durată:** {source_info.duration_s:.2f} secunde\n"
                    f"- **Cadre pe secundă:** {source_info.fps:.1f} fps\n"
                    f"- **Codec:** {source_info.codec_video or 'H.264'} "
                    f"({source_info.pixel_format})\n"
                    f"- **Audio:** {has_aud}\n\n"
                    "💡 **Recomandări de producție pentru acest clip:**\n"
                    f"1. {rec1}\n"
                    f"2. {rec2}\n"
                    f"3. {rec3}"
                ),
                intent="inspect",
                suggested_prompts=(
                    f"Fă un short 9:16 din {source_path.name} cu subtitrări TikTok și taie pauzele",
                    f"Curăță vocea din {source_path.name} și aplică mastering podcast",
                    f"Adaugă o bară animată de progres galbenă la baza clipului {source_path.name}",
                ),
            )
        return ChatResponse(
            reply=(
                "Selectează un fișier din lista din stânga pentru a-i analiza rezoluția, "
                "durata, pista audio și a primi recomandări optime de editare!"
            ),
            intent="inspect",
            suggested_prompts=(
                "Fă un short 9:16 cu subtitrări galbene TikTok și taie pauzele",
                "Curăță vocea cu profil podcast, aplică de-esser și normalizează sunetul",
            ),
        )

    # 4. Prompt Ideation / "dă-mi o idee" / "viral" / "cum să editez"
    idea_triggers = {
        "da-mi o idee",
        "dă-mi o idee",
        "idei",
        "ce prompt",
        "cum sa editez",
        "cum să editez",
        "viral",
        "sugestie",
        "recomanda",
        "recomandă",
        "fa un prompt",
        "fă un prompt",
        "ce recomanzi",
    }
    if any(it in p_lower for it in idea_triggers):
        return ChatResponse(
            reply=(
                "### 💡 3 Rețete Virale Recomandate pentru Social Media:\n\n"
                "**1. 📱 Retenție Maximă (TikTok / Reels / Shorts):**\n"
                "> *„Fă un short vertical 9:16 cu subtitrări galbene TikTok, curăță vocea, "
                "taie pauzele și adaugă o bară galbenă de progres la bază”*\n\n"
                "**2. 🎙️ Podcast / Discurs Clar & Dinamic:**\n"
                "> *„Curăță vocea cu profil podcast, aplică de-esser, normalizează sunetul, "
                "elimină tăcerile și fă punch zoom la fiecare 4 secunde”*\n\n"
                "**3. ⚡ Look Cinematic & Branding:**\n"
                "> *„Look cinematic, adaugă titlu 'Episodul #1' cu badge sus și zoom dinamic”*\n\n"
                "Apasă pe oricare dintre opțiunile de mai jos pentru a pregăti planul de producție!"
            ),
            intent="ideation",
            suggested_prompts=(
                "Fă un short 9:16 cu subtitrări galbene TikTok, curăță vocea și taie pauzele",
                "Curăță vocea podcast, elimină tăcerile și adaugă punch-in zoom la 4 secunde",
                "Aplică look cinematic, adaugă titlul 'Episodul 1' sus și bară de progres",
            ),
        )

    # 5. Concrete Editing Request
    target_out = (
        (source_path.parent.parent / "out" / f"studio_{source_path.stem}.mp4")
        if source_path
        else Path("out/studio_render.mp4")
    )
    plan = plan_prompt(prompt, source_path or "in/clip.mp4", target_out)
    is_baseline_only = len(plan.operations) == 1 and "Conversie" in plan.operations[0]

    if is_baseline_only:
        return ChatResponse(
            reply=(
                f"Am primit cererea ta: *„{p}”*.\n\n"
                "Pentru a orchestra fluxul de producție, spune-mi ce transformări dorești "
                "(ex: decupare 9:16, subtitrări, mastering voce, tăiere pauze, bară progres), "
                "sau alege una dintre sugestiile populare de mai jos:"
            ),
            intent="clarification",
            suggested_prompts=(
                "Fă un short 9:16 cu subtitrări galbene TikTok și taie pauzele",
                "Adaugă o bară animată de progres galbenă la bază și zoom dinamic",
                "Curăță vocea cu profil podcast și elimină zgomotul",
                "Modifică viteza la 0.5x slow-motion cu audio păstrat",
            ),
            plan_operations=plan.operations,
            spec=plan.spec,
            executable=False,
        )

    # Check for silent input media
    is_silent = source_info is not None and not source_info.has_audio
    has_audio_ops = (
        plan.spec.audio.silence_trim
        or plan.spec.audio.clean_speech
        or bool(plan.spec.audio.master_profile)
        or plan.spec.video.subtitles.enabled
    )
    has_visual_ops = (
        plan.spec.video.aspect != "original"
        or plan.spec.video.look is not None
        or plan.spec.video.punch_zoom
        or plan.spec.video.typography is not None
        or plan.spec.video.progress_bar
        or plan.spec.video.speed != 1.0
        or plan.spec.video.vflip
        or plan.spec.video.hflip
        or plan.spec.video.rotate != 0
        or plan.spec.video.invert_colors
        or plan.spec.video.grayscale
        or plan.spec.video.reverse
        or plan.spec.video.brightness != 0.0
        or plan.spec.video.contrast != 1.0
        or plan.spec.video.saturation != 1.0
        or plan.spec.video.blur > 0
        or plan.spec.video.sharpen
        or plan.spec.video.mute_audio
        or plan.spec.video.volume_multiplier != 1.0
        or plan.spec.photo.retouch
        or plan.spec.photo.depth_blur
        or plan.spec.photo.vflip
        or plan.spec.photo.hflip
        or plan.spec.photo.rotate != 0
        or plan.spec.photo.invert_colors
        or plan.spec.photo.grayscale
        or plan.spec.photo.brightness != 0.0
        or plan.spec.photo.contrast != 1.0
        or plan.spec.photo.saturation != 1.0
        or plan.spec.photo.blur > 0
        or plan.spec.photo.sharpen
        or plan.spec.video.cutout
        or bool(plan.spec.video.backdrop)
        or plan.spec.video.upscale > 0
        or bool(plan.spec.video.narrator_text)
        or plan.spec.photo.upscale > 0
        or bool(plan.spec.audio.sfx_cues)
        or plan.spec.audio.music_track is not None
    )

    if is_silent and has_audio_ops and not has_visual_ops:
        s_name = source_path.name if source_path else "Clipul"
        return ChatResponse(
            reply=(
                f"⚠️ Fișierul `{s_name}` **nu conține o pistă audio**.\n\n"
                "Operațiunile vocale cerute (curățare voce, tăiere silențiu, subtitrări) "
                "nu pot fi aplicate unui clip fără sunet.\n\n"
                "Alege o transformare video (decupare 9:16, zoom, titlu grafic, bară progres) "
                "sau selectează un clip cu sunet din lista din stânga!"
            ),
            intent="clarification",
            suggested_prompts=(
                f"Fă un short 9:16 din {s_name} cu zoom dinamic",
                f"Adaugă o bară animată de progres galbenă pe {s_name}",
                f"Aplică look cinematic și titlu 'Video' pe {s_name}",
            ),
            plan_operations=(),
            spec=plan.spec,
            executable=False,
        )

    # If operations are found, build a rich, personalized production plan
    final_ops: list[str] = []
    for op in plan.operations:
        if is_silent and any(
            k in op.lower() for k in ("pauze", "silențiu", "vocal", "demucs", "subtitrări")
        ):
            final_ops.append(f"{op} *(omis - fișier fără audio)*")
        else:
            final_ops.append(op)

    ops_md = "\n".join(f"- **{i + 1}.** {op}" for i, op in enumerate(final_ops))
    note = ""
    if is_silent and has_audio_ops:
        f_name = source_path.name if source_path else "selectat"
        note = (
            f"\n\n> ⚠️ **Notă:** Fișierul `{f_name}` nu are pistă audio. "
            "Etapele vocale vor fi omise automat, iar efectele video vor fi aplicate."
        )

    reply = (
        f"🎬 **Am configurat planul de producție pentru cererea ta:**\n\n"
        f"{ops_md}{note}\n\n"
        "Parametrii au fost validați. Apasă butonul de mai jos pentru a lansa randarea!"
    )
    return ChatResponse(
        reply=reply,
        intent="plan",
        plan_operations=tuple(final_ops),
        suggested_prompts=(),
        spec=plan.spec,
        executable=True,
    )

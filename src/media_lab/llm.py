"""Cognitive reasoning and multi-provider LLM brain for Media Lab Studio.

Provides semantic understanding of complex, conversational video/photo editing
requests in Romanian and English, with state continuity (referencing previous
edits), negation handling, camera motion defect neutralization (VidStab), and
autonomous multi-skill coordination.

Supported engines:
1. Google Gemini REST API (gemini-2.0-flash / gemini-1.5-flash via httpx)
2. OpenAI-compatible / Local Ollama (e.g. localhost:11434)
3. Local Semantic Engine (deterministic offline reasoning with zero API keys)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from .edit_spec import (
    AudioEditSpec,
    EditSpec,
    PhotoEditSpec,
    SubtitleEditSpec,
    VideoEditSpec,
    parse_edit_spec,
)
from .probe import MediaInfo

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMConfig:
    """Configuration for LLM reasoning engine."""

    provider: str = "auto"  # 'auto', 'gemini', 'openai', 'ollama', 'local'
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    temperature: float = 0.1
    timeout_s: float = 12.0

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Create LLMConfig by discovering environment variables."""
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        openai_base = os.getenv("OPENAI_BASE_URL")
        ollama_endpoint = os.getenv("OLLAMA_HOST") or "http://localhost:11434"

        provider = "auto"
        if os.getenv("MEDIA_LAB_LLM_PROVIDER"):
            provider = os.getenv("MEDIA_LAB_LLM_PROVIDER", "auto")
        elif gemini_key:
            provider = "gemini"
        elif openai_key:
            provider = "openai"

        return cls(
            provider=provider,
            gemini_api_key=gemini_key,
            openai_api_key=openai_key,
            openai_base_url=openai_base,
            ollama_url=ollama_endpoint,
        )


@dataclass(frozen=True, slots=True)
class ReasoningResult:
    """Output produced by cognitive brain reasoning pass."""

    reply: str
    intent: str
    operations: tuple[str, ...] = ()
    spec: EditSpec | None = None
    executable: bool = False
    provider_used: str = "local_semantic"
    suggested_prompts: tuple[str, ...] = ()


def spec_to_dict(spec: EditSpec) -> dict[str, Any]:
    """Convert an EditSpec dataclass into a clean dictionary."""
    return asdict(spec)


def dict_to_spec(data: dict[str, Any], default_source: str, default_output: str) -> EditSpec:
    """Convert a dictionary into a validated EditSpec."""
    payload = dict(data)
    if "source" not in payload or not payload["source"]:
        payload["source"] = default_source
    if "output" not in payload or not payload["output"]:
        payload["output"] = default_output
    return parse_edit_spec(payload)


def merge_specs(base: EditSpec, updates: dict[str, Any]) -> EditSpec:
    """Apply dictionary updates over an existing base EditSpec."""
    base_dict = spec_to_dict(base)

    for top_key in ("video", "audio", "photo"):
        if top_key in updates and isinstance(updates[top_key], dict):
            base_dict[top_key].update(updates[top_key])

    for scalar_key in ("source", "output"):
        if scalar_key in updates and updates[scalar_key]:
            base_dict[scalar_key] = updates[scalar_key]

    return parse_edit_spec(base_dict)


SYSTEM_PROMPT = """You are the AI Cognitive Brain for Media Lab Studio on Apple Silicon.
Your task is to analyze user editing requests (in Romanian or English), understand nuances,
camera defect descriptions, negations, and continuity, and output a valid JSON response.

CAPABILITIES:
1. Video:
   - aspect: "original", "9:16", "1:1", "4:5", "16:9"
   - smart_reframe: boolean (follow subject)
   - punch_zoom: boolean (dynamic retention zoom during talking)
   - stabilize: boolean (VidStab 2-pass camera stabilization, anti-shake, fixes zoom-out/jitter)
   - look: null or "cinematic", "warm", "cool", "vintage", "noir"
   - speed: float (0.5 for slow-mo, 2.0 for timelapse)
   - progress_bar: boolean, progress_bar_color: "yellow"|"red"|"white"|"cyan"|"green"
   - progress_bar_position: "bottom"|"top"
   - vflip: boolean (upside down / cu susul in jos), hflip: boolean (mirror / oglinda)
   - rotate: 0, 90, 180, 270, invert_colors: boolean, grayscale: boolean, reverse: boolean
   - brightness: float (-0.5 to 0.5), contrast: float (0.5 to 2.0), saturation: float (0.0 to 2.0)
   - blur: float (0.0 to 20.0), sharpen: boolean
   - cutout: boolean (RVM AI background matting), backdrop: null or image path
   - upscale: 0, 2, 4 (Real-ESRGAN super-resolution)
   - narrator_text: null or string, narrator_voice: "Ioana" or "Daniel"
   - subtitles: {"enabled": bool, "style": "tiktok"|"boxed_yellow"|"cinematic"|"minimal"}
   - typography: null or {"text": string, "position": "top"|"center"|"bottom", "badge": true}

2. Audio:
   - clean_speech: boolean (Demucs voice stem isolation)
   - master_profile: null or "podcast", "radio", "crisp"
   - silence_trim: boolean (jump cut silent pauses)
   - music_track: null or path (background music bed with ducking)

CRITICAL RULES:
- CONTINUITY: If the user says "ca inainte", "la fel ca inainte", "pastreaza",
  keep all previous properties from `previous_spec`!
- NEGATION: If user says "scoate zoom", "fara zoom", "nu mai da zoom", SET punch_zoom: false!
- CAMERA FIXES: If user mentions "am dat zoom out la filmare", "se vede prost", "tremura",
  SET stabilize: true and punch_zoom: false!
- AUDIO: If `media_info.has_audio` is false, silence_trim, clean_speech, subtitles MUST be off.

You MUST respond strictly with a valid JSON object adhering to this schema:
{
  "intent": "edit" | "plan" | "greeting" | "help" | "inspect",
  "explanation": "Markdown text in Romanian explaining what actions and fixes were determined.",
  "operations": ["Operation 1", "Operation 2"],
  "executable": true | false,
  "video": { ...updated video fields... },
  "audio": { ...updated audio fields... },
  "photo": { ...updated photo fields... }
}
"""


class CognitiveBrain:
    """Cognitive reasoning coordinator managing remote LLMs and local semantic engine."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()

    def reason(
        self,
        prompt: str,
        source: Path | None,
        output: Path | None,
        media_info: MediaInfo | None = None,
        previous_spec: EditSpec | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> ReasoningResult:
        """Process prompt through remote LLM or fallback seamlessly to local semantic engine."""
        p_clean = prompt.strip()

        # Check for remote LLM execution if configured
        if self.config.provider in ("gemini", "auto") and self.config.gemini_api_key:
            try:
                res = self._call_gemini(p_clean, source, output, media_info, previous_spec, history)
                if res:
                    return res
            except Exception as exc:
                logger.warning("Gemini LLM reasoning call failed, falling back to local: %s", exc)

        if self.config.provider in ("openai", "ollama") and (
            self.config.openai_api_key or self.config.ollama_url
        ):
            try:
                res = self._call_openai_compatible(
                    p_clean, source, output, media_info, previous_spec, history
                )
                if res:
                    return res
            except Exception as exc:
                logger.warning("OpenAI/Ollama LLM call failed, falling back to local: %s", exc)

        # Default guaranteed fallback: Local Semantic Diff Engine
        return self._local_semantic_reason(
            p_clean, source, output, media_info, previous_spec, history
        )

    def _call_gemini(
        self,
        prompt: str,
        source: Path | None,
        output: Path | None,
        media_info: MediaInfo | None,
        previous_spec: EditSpec | None,
        history: list[dict[str, str]] | None,
    ) -> ReasoningResult | None:
        """Call Google Gemini REST API using httpx."""
        api_key = self.config.gemini_api_key
        if not api_key:
            return None

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.gemini_model}:generateContent?key={api_key}"
        )

        user_content = self._build_context_prompt(
            prompt, source, output, media_info, previous_spec, history
        )

        payload = {
            "contents": [{"role": "user", "parts": [{"text": user_content}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": self.config.temperature,
            },
        }

        with httpx.Client(timeout=self.config.timeout_s) as client:
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                logger.warning("Gemini returned status %d: %s", resp.status_code, resp.text)
                return None

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return None
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return None

            text = parts[0].get("text", "")
            return self._parse_llm_json_response(
                text, source, output, previous_spec, provider="gemini"
            )

    def _call_openai_compatible(
        self,
        prompt: str,
        source: Path | None,
        output: Path | None,
        media_info: MediaInfo | None,
        previous_spec: EditSpec | None,
        history: list[dict[str, str]] | None,
    ) -> ReasoningResult | None:
        """Call OpenAI or Local Ollama chat completion endpoint."""
        is_ollama = self.config.provider == "ollama" or (not self.config.openai_api_key)
        base_url = (
            self.config.ollama_url.rstrip("/") + "/v1"
            if is_ollama
            else (self.config.openai_base_url or "https://api.openai.com/v1").rstrip("/")
        )
        model = self.config.ollama_model if is_ollama else self.config.openai_model
        endpoint = f"{base_url}/chat/completions"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.openai_api_key:
            headers["Authorization"] = f"Bearer {self.config.openai_api_key}"

        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-4:])
        user_content = self._build_context_prompt(
            prompt, source, output, media_info, previous_spec, None
        )
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": self.config.temperature,
        }

        with httpx.Client(timeout=self.config.timeout_s) as client:
            resp = client.post(endpoint, json=payload, headers=headers)
            if resp.status_code != 200:
                logger.warning("LLM endpoint returned status %d: %s", resp.status_code, resp.text)
                return None

            res_json = resp.json()
            choices = res_json.get("choices", [])
            if not choices:
                return None
            content = choices[0].get("message", {}).get("content", "")
            return self._parse_llm_json_response(
                content, source, output, previous_spec, provider="ollama" if is_ollama else "openai"
            )

    def _build_context_prompt(
        self,
        prompt: str,
        source: Path | None,
        output: Path | None,
        media_info: MediaInfo | None,
        previous_spec: EditSpec | None,
        history: list[dict[str, str]] | None,
    ) -> str:
        """Construct rich contextual user prompt for the remote LLM."""
        info_str = "None"
        if media_info:
            info_str = (
                f"Resolution: {media_info.width}x{media_info.height}, "
                f"Duration: {media_info.duration_s:.1f}s, "
                f"FPS: {media_info.fps:.1f}, "
                f"HasAudio: {media_info.has_audio}"
            )

        prev_str = "None"
        if previous_spec:
            prev_str = json.dumps(spec_to_dict(previous_spec), indent=2)

        history_str = ""
        if history:
            history_str = "\nConversation History:\n" + "\n".join(
                f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history[-3:]
            )

        return (
            f"User Prompt: {prompt}\n"
            f"Source File: {source.name if source else 'in/video.mp4'}\n"
            f"Output Destination: {output.name if output else 'out/render.mp4'}\n"
            f"Source Media Info: {info_str}\n"
            f"Previous EditSpec:\n{prev_str}\n"
            f"{history_str}\n\n"
            "Analyze the request and return the JSON edit specification."
        )

    def _parse_llm_json_response(
        self,
        text: str,
        source: Path | None,
        output: Path | None,
        previous_spec: EditSpec | None,
        provider: str,
    ) -> ReasoningResult | None:
        """Parse, validate, and merge LLM JSON response."""
        try:
            parsed = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception:
                    return None
            else:
                return None

        intent = parsed.get("intent", "edit")
        explanation = parsed.get("explanation", "Planul de editare a fost configurat.")
        operations = tuple(parsed.get("operations", []))
        executable = bool(parsed.get("executable", True))

        default_src = str(source) if source else "in/clip.mp4"
        default_out = str(output) if output else "out/studio_render.mp4"

        try:
            if previous_spec is not None:
                spec = merge_specs(previous_spec, parsed)
            else:
                spec = dict_to_spec(parsed, default_src, default_out)
        except Exception as err:
            logger.warning("Failed to construct EditSpec from LLM output: %s", err)
            return None

        return ReasoningResult(
            reply=explanation,
            intent=intent,
            operations=operations,
            spec=spec,
            executable=executable,
            provider_used=provider,
        )

    def _local_semantic_reason(
        self,
        prompt: str,
        source: Path | None,
        output: Path | None,
        media_info: MediaInfo | None,
        previous_spec: EditSpec | None,
        history: list[dict[str, str]] | None,
    ) -> ReasoningResult:
        """Deterministic semantic reasoning engine with state continuity and negation logic."""
        p_lower = prompt.lower()
        src_str = str(source) if source else "in/clip.mp4"
        out_str = str(output) if output else "out/studio_render.mp4"

        # Check for conversational / informational intents first
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
        }
        has_edit_action = any(
            k in p_lower
            for k in (
                "zoom",
                "short",
                "taie",
                "curata",
                "curăță",
                "subtitr",
                "fa ",
                "fă ",
                "pune ",
                "stabiliz",
                "reels",
            )
        )
        is_greeting = (
            any(re.match(rf"^{g}\b", p_lower) for g in greetings) or p_lower in greetings
        ) and not has_edit_action
        is_help = any(h in p_lower for h in help_triggers) and not has_edit_action

        if is_greeting or is_help:
            return ReasoningResult(
                reply=(
                    "Salut! Sunt **Media Lab Cognitive Brain** 🧠🎬 — creierul tău creativ pentru "
                    "editare video, audio și foto pe Apple Silicon.\n\n"
                    "Înțeleg comenzi complexe în limbaj natural, țin minte editările anterioare, "
                    "știu să elimin sau să compensez mișcările nedorite ale camerei (zoom manual, "
                    "tremur) și pot orchestra toate uneltele instalate.\n\n"
                    "Selectează un fișier și dă-mi direct instrucțiuni!"
                ),
                intent="greeting" if is_greeting else "help",
                suggested_prompts=(
                    "Fă un short vertical 9:16 cu subtitrări galbene TikTok și taie pauzele",
                    "Curăță vocea cu profil podcast, aplică de-esser și normalizează sunetul",
                    "Adaugă o bară animată de progres galbenă la bază și zoom dinamic",
                    "Analizează clipul selectat și dă-mi recomandări",
                ),
                provider_used="local_semantic",
            )

        # Continuity detection (e.g. "exact la fel ca înainte", "ca înainte",
        # "păstrează", "modifică doar")
        is_iterative = previous_spec is not None and (
            any(
                k in p_lower
                for k in (
                    "ca inainte",
                    "ca înainte",
                    "la fel ca inainte",
                    "la fel ca înainte",
                    "exact la fel",
                    "pastreaza restul",
                    "păstrează restul",
                    "modifica doar",
                    "modifică doar",
                    "scoate doar",
                    "fara sa schimbi",
                    "fără să schimbi",
                    "in continuare",
                    "în continuare",
                )
            )
        )

        # Base specs: either inherited from previous_spec or fresh defaults
        if is_iterative and previous_spec is not None:
            base_vid = previous_spec.video
            base_aud = previous_spec.audio
            base_pho = previous_spec.photo
        else:
            base_vid = VideoEditSpec()
            base_aud = AudioEditSpec()
            base_pho = PhotoEditSpec()

        # 1. Camera defect detection & Stabilization
        stabilize_triggers = (
            "am dat zoom out",
            "zoom out",
            "se vede prost",
            "stabilizeaza",
            "stabilizează",
            "stabilizare",
            "deshake",
            "tremur",
            "miscare brusca",
            "mișcare bruscă",
            "netezeste camera",
            "netezește camera",
        )
        has_stabilize_intent = any(k in p_lower for k in stabilize_triggers)
        stabilize = True if has_stabilize_intent else base_vid.stabilize

        # 2. Punch zoom & Negation logic
        no_zoom_triggers = (
            "scoate zoom",
            "scoate acel zoom",
            "fara zoom",
            "fără zoom",
            "elimina zoom",
            "elimină zoom",
            "anuleaza zoom",
            "anulează zoom",
            "fara apropiere",
            "fără apropiere",
            "nu pune zoom",
            "nu mai da zoom",
            "dezactiveaza zoom",
            "dezactivează zoom",
        )
        is_negated_zoom = any(k in p_lower for k in no_zoom_triggers) or bool(
            re.search(
                r"\b(scoate|fara|fără|elimina|elimină|anuleaza|anulează|nu|dezactiveaza|dezactivează)\b.*?\bzoom\b",
                p_lower,
            )
        )

        punch_zoom = base_vid.punch_zoom
        if is_negated_zoom:
            punch_zoom = False
        elif (
            any(k in p_lower for k in ("punch zoom", "zoom", "retention", "apropiere"))
            and not has_stabilize_intent
        ):
            punch_zoom = True
        elif has_stabilize_intent and not is_iterative:
            punch_zoom = False

        zoom_interval = 5.0 if punch_zoom else None

        # 2b. Question & Directorial Consultation Detection
        is_question = (
            any(
                p_lower.startswith(q)
                for q in (
                    "poti",
                    "poți",
                    "ai putea",
                    "se poate",
                    "ar trebui",
                    "ce parere ai",
                    "ce părere ai",
                    "cum facem",
                    "ce recomanzi",
                    "crezi ca",
                    "crezi că",
                    "de ce",
                    "oare",
                )
            )
            or "?" in prompt
            or any(
                k in p_lower
                for k in (
                    "poti face asta",
                    "poți face asta",
                    "se poate face",
                    "ai putea face",
                    "e posibil",
                )
            )
        )

        # 2c. Camera Motion & Dynamic Kinetic Tracking
        camera_track_triggers = (
            "urmareste miscarea",
            "urmărește mișcarea",
            "urmareste persoana",
            "urmărește persoana",
            "urmareste-o",
            "urmărește-o",
            "sa mearga cu miscarea",
            "să meargă cu mișcarea",
            "miscari consecvente cu camera",
            "mișcări consecvente cu camera",
            "miscari cu camera",
            "mișcări cu camera",
            "camera tracking",
            "camera follow",
            "pan track",
            "kinetic tracking",
            "urmarire dinamica",
            "urmărire dinamică",
        )
        has_camera_track = any(k in p_lower for k in camera_track_triggers)
        has_push_in = any(
            k in p_lower for k in ("push in", "slow zoom in", "apropiere lina", "apropiere lină")
        )
        has_pull_out = any(
            k in p_lower for k in ("pull out", "slow zoom out", "departare lina", "depărtare lină")
        )
        has_pan_left = any(k in p_lower for k in ("pan stanga", "pan stânga", "pan left"))
        has_pan_right = any(k in p_lower for k in ("pan dreapta", "pan right"))
        has_handheld = any(
            k in p_lower
            for k in (
                "handheld",
                "camera din mana",
                "cameră din mână",
                "miscare organica",
                "mișcare organică",
            )
        )

        camera_motion = base_vid.camera_motion
        reframe_mode = base_vid.reframe_mode
        if has_camera_track:
            camera_motion = "track"
            reframe_mode = "track"
            if has_stabilize_intent or "zoom out" in p_lower or "consecvent" in p_lower:
                stabilize = True
        elif has_push_in:
            camera_motion = "push_in"
        elif has_pull_out:
            camera_motion = "pull_out"
        elif has_pan_left:
            camera_motion = "pan_left"
        elif has_pan_right:
            camera_motion = "pan_right"
        elif has_handheld:
            camera_motion = "handheld"
        elif any(
            k in p_lower for k in ("scoate miscare camera", "fara tracking", "camera statica")
        ):
            camera_motion = None
            reframe_mode = "smart"

        # 3. Format & Reframe
        aspect = base_vid.aspect
        smart_reframe = base_vid.smart_reframe
        if any(k in p_lower for k in ("9:16", "short", "reels", "tiktok", "vertical")):
            aspect = "9:16"
            smart_reframe = True
            if has_camera_track:
                reframe_mode = "track"
        elif any(k in p_lower for k in ("1:1", "patrat", "pătrat", "square")):
            aspect = "1:1"
            smart_reframe = True
            if has_camera_track:
                reframe_mode = "track"
        elif any(k in p_lower for k in ("4:5", "portrait")):
            aspect = "4:5"
            smart_reframe = True
            if has_camera_track:
                reframe_mode = "track"
        elif "16:9" in p_lower or "orizontal" in p_lower or "horizontal" in p_lower:
            aspect = "16:9"
            smart_reframe = False

        # 4. Subtitles
        subtitles_enabled = base_vid.subtitles.enabled
        sub_style = base_vid.subtitles.style
        if any(k in p_lower for k in ("scoate subtitr", "fara subtitr", "fără subtitr")):
            subtitles_enabled = False
        elif any(k in p_lower for k in ("subtitr", "subtitle", "caption")):
            subtitles_enabled = True
            if "galben" in p_lower or "yellow" in p_lower:
                sub_style = "boxed_yellow"
            elif "cinematic" in p_lower:
                sub_style = "cinematic"
            elif "minimal" in p_lower:
                sub_style = "minimal"

        # 5. Visual Look
        look = base_vid.look
        if any(k in p_lower for k in ("scoate look", "fara look", "culori normale")):
            look = None
        elif "cinematic" in p_lower and not (subtitles_enabled and "subtitr" in p_lower):
            look = "cinematic"
        elif "cald" in p_lower or "warm" in p_lower:
            look = "warm"
        elif "rece" in p_lower or "cool" in p_lower:
            look = "cool"
        elif "vintage" in p_lower:
            look = "vintage"
        elif "noir" in p_lower or "alb-negru" in p_lower or "black and white" in p_lower:
            look = "noir"

        # 6. Audio: clean speech & mastering
        clean_speech = base_aud.clean_speech
        if any(k in p_lower for k in ("nu curata", "fara demucs", "sunet original")):
            clean_speech = False
        elif any(
            k in p_lower
            for k in (
                "curata vocea",
                "curăță vocea",
                "clean speech",
                "izoleaza vocea",
                "izolează vocea",
            )
        ):
            clean_speech = True

        master_profile = base_aud.master_profile
        if "podcast" in p_lower:
            master_profile = "podcast"
        elif "radio" in p_lower:
            master_profile = "radio"
        elif "crisp" in p_lower or "clar" in p_lower:
            master_profile = "crisp"

        # 7. Silence jump cut
        silence_trim = base_aud.silence_trim
        if any(k in p_lower for k in ("nu taia pauzele", "pastreaza pauzele", "păstrează pauzele")):
            silence_trim = False
        elif any(
            k in p_lower
            for k in (
                "taie pauzele",
                "elimina tacerile",
                "elimină tăcerile",
                "jump cut",
                "fara pauze",
            )
        ):
            silence_trim = True

        # 8. Spatial / Geometric flips
        vflip = base_vid.vflip
        if any(
            k in p_lower
            for k in (
                "cu susul in jos",
                "cu susul în jos",
                "solul cu susul in jos",
                "solul sus",
                "rastoarna",
                "răstoarnă",
                "vflip",
                "upside down",
            )
        ):
            vflip = True

        hflip = base_vid.hflip
        if any(k in p_lower for k in ("oglinda", "oglindă", "mirror", "hflip")):
            hflip = True

        rotate_deg = base_vid.rotate
        if "180" in p_lower and any(k in p_lower for k in ("rot", "grade", "deg")):
            rotate_deg = 180
        elif "90" in p_lower and any(k in p_lower for k in ("rot", "grade", "deg")):
            rotate_deg = 90

        # 9. Progress bar
        progress_bar = base_vid.progress_bar
        pb_color = base_vid.progress_bar_color
        pb_pos = base_vid.progress_bar_position
        if any(k in p_lower for k in ("scoate bara", "fara bara", "fără bară")):
            progress_bar = False
        elif any(k in p_lower for k in ("progress bar", "bara de progres", "bară de progres")):
            progress_bar = True
            if "rosie" in p_lower or "roșie" in p_lower:
                pb_color = "red"
            elif "alba" in p_lower or "albă" in p_lower:
                pb_color = "white"
            elif "cyan" in p_lower:
                pb_color = "cyan"
            if "sus" in p_lower or "top" in p_lower:
                pb_pos = "top"

        # 10. AI Neural Cutout (RVM) & Super-Resolution (Real-ESRGAN)
        cutout = base_vid.cutout
        backdrop = base_vid.backdrop
        if any(
            k in p_lower for k in ("scoate fundalul", "rvm", "cutout", "decupeaza", "decupează")
        ):
            cutout = True
        if any(k in p_lower for k in ("schimba fundalul", "schimbă fundalul", "pune fundal")):
            cutout = True
            backdrop = "in/punto-bg.png" if "punto" in p_lower else "in/demo-bg.png"

        upscale = base_vid.upscale
        if any(k in p_lower for k in ("upscale", "super rezolutie", "super-rezoluție", "4k")):
            upscale = 4 if "4k" in p_lower else 2

        # 11. TTS Narrator
        narrator_text = base_vid.narrator_text
        narrator_voice = base_vid.narrator_voice or "Ioana"
        if any(k in p_lower for k in ("narator", "voiceover", "voce de prezentare")):
            narrator_match = re.search(r'["\']([^"\']+)["\']', prompt)
            narrator_text = (
                narrator_match.group(1)
                if narrator_match
                else (narrator_text or "Bun venit la producția Media Lab Studio!")
            )

        # 12. Music bed with ducking
        music_track = base_aud.music_track
        if any(k in p_lower for k in ("muzica", "muzică", "soundtrack", "coloana sonora")):
            music_track = "in/demo2-music.m4a" if "demo2" in p_lower else "in/demo-music.m4a"

        # Handle silent input defensive logic
        is_silent = media_info is not None and not media_info.has_audio
        if is_silent:
            clean_speech = False
            master_profile = None
            silence_trim = False
            subtitles_enabled = False

        new_vid = VideoEditSpec(
            aspect=aspect,
            smart_reframe=smart_reframe,
            reframe_mode=reframe_mode,
            punch_zoom=punch_zoom,
            zoom_interval=zoom_interval,
            broll_cuts=base_vid.broll_cuts,
            look=look,
            second_look=base_vid.second_look,
            subtitles=SubtitleEditSpec(enabled=subtitles_enabled, style=sub_style),
            typography=base_vid.typography,
            cutout=cutout,
            backdrop=backdrop,
            speed=base_vid.speed,
            progress_bar=progress_bar,
            progress_bar_color=pb_color,
            progress_bar_position=pb_pos,
            vflip=vflip,
            hflip=hflip,
            rotate=rotate_deg,
            invert_colors=base_vid.invert_colors,
            grayscale=base_vid.grayscale,
            reverse=base_vid.reverse,
            brightness=base_vid.brightness,
            contrast=base_vid.contrast,
            saturation=base_vid.saturation,
            blur=base_vid.blur,
            sharpen=base_vid.sharpen,
            mute_audio=base_vid.mute_audio,
            volume_multiplier=base_vid.volume_multiplier,
            upscale=upscale,
            narrator_text=narrator_text,
            narrator_voice=narrator_voice,
            stabilize=stabilize,
            camera_motion=camera_motion,
            camera_smoothing=1.5,
        )

        new_aud = AudioEditSpec(
            clean_speech=clean_speech,
            master_profile=master_profile,
            silence_trim=silence_trim,
            music_track=music_track,
            target_lufs=base_aud.target_lufs,
            music_volume=base_aud.music_volume,
            sfx_cues=base_aud.sfx_cues,
        )

        new_pho = base_pho

        final_spec = EditSpec(
            source=src_str,
            output=out_str,
            video=new_vid,
            audio=new_aud,
            photo=new_pho,
        )

        ops = self._generate_operations_summary(final_spec, is_silent)
        explanation = self._generate_explanation(
            prompt,
            final_spec,
            ops,
            is_iterative,
            is_negated_zoom,
            has_stabilize_intent,
            is_question=is_question,
            has_camera_track=has_camera_track,
        )

        return ReasoningResult(
            reply=explanation,
            intent="consultation" if is_question else "edit",
            operations=tuple(ops),
            spec=final_spec,
            executable=True,
            provider_used="local_semantic",
            suggested_prompts=(
                "Fă un short vertical 9:16 cu subtitrări galbene TikTok și taie pauzele",
                "Curăță vocea cu profil podcast, aplică de-esser și normalizează sunetul",
                "Adaugă o bară animată de progres galbenă la bază și zoom dinamic",
            ),
        )

    def _generate_operations_summary(self, spec: EditSpec, is_silent: bool) -> list[str]:
        """Compile readable operations bullet points for the plan card."""
        ops: list[str] = []
        if spec.video.stabilize:
            ops.append(
                "Stabilizare cameră 2-pass VidStab (compensare zoom manual și eliminare tremur)"
            )
        if not spec.video.punch_zoom:
            ops.append("Punch-in zoom dezactivat (fără apropiere artificială pe subiect)")
        else:
            ops.append("Punch-in zoom dinamic activat pentru retenție (1.15x)")

        if spec.video.camera_motion == "track":
            ops.append(
                "Urmărire cinetică a camerei (Dynamic Camera Follow - "
                "camera glisează lin cu mișcarea persoanei)"
            )
        elif spec.video.camera_motion == "push_in":
            ops.append("Apropiere cinematică lină de cameră (Slow Push-In Zoom)")
        elif spec.video.camera_motion == "pull_out":
            ops.append("Depărtare cinematică lină de cameră (Slow Pull-Out Zoom)")
        elif spec.video.camera_motion == "pan_left":
            ops.append("Glisare cinematică spre stânga (Pan Left)")
        elif spec.video.camera_motion == "pan_right":
            ops.append("Glisare cinematică spre dreapta (Pan Right)")
        elif spec.video.camera_motion == "handheld":
            ops.append("Mișcare organică de cameră din mână (Cinematic Handheld Drift)")

        if spec.video.aspect != "original":
            reframe_label = (
                f"Reîncadrare verticală {spec.video.aspect} cu "
                "urmărire dinamică pe subiect (Kinetic Track)"
                if spec.video.reframe_mode in ("track", "dynamic", "follow")
                else f"Reîncadrare format {spec.video.aspect} (urmărire subiect)"
            )
            ops.append(reframe_label)

        if spec.video.look:
            ops.append(f"Colorizare cinematică (profil {spec.video.look})")
        if spec.video.subtitles.enabled:
            ops.append(f"Subtitrări automate Whisper ({spec.video.subtitles.style})")
        if spec.video.progress_bar:
            ops.append(f"Bară animată de progres ({spec.video.progress_bar_color})")
        if spec.video.vflip:
            ops.append("Răsturnare verticală (solul cu susul în jos / vflip)")
        if spec.video.hflip:
            ops.append("Oglindire orizontală (mirror / hflip)")
        if spec.video.rotate:
            ops.append(f"Rotire cadru ({spec.video.rotate}°)")
        if spec.video.cutout:
            ops.append("Decupare subiect AI (RVM Robust Video Matting)")
        if spec.video.backdrop:
            ops.append(f"Compoziție pe fundal nou ({spec.video.backdrop})")
        if spec.video.upscale > 0:
            ops.append(f"Super-rezoluție Real-ESRGAN {spec.video.upscale}x")
        if spec.video.narrator_text:
            ops.append(f"Narator vocal local TTS ({spec.video.narrator_voice or 'Ioana'})")
        if spec.audio.clean_speech:
            ops.append("Curățare zgomot și izolare voce Demucs")
        if spec.audio.master_profile:
            ops.append(f"Mastering audio vocal (profil {spec.audio.master_profile})")
        if spec.audio.silence_trim:
            ops.append("Tăiere automată pauze și tăceri (jump-cuts)")
        if spec.audio.music_track:
            ops.append(f"Coloană sonoră cu ducking ({spec.audio.music_track})")

        return ops

    def _generate_explanation(
        self,
        prompt: str,
        spec: EditSpec,
        ops: list[str],
        is_iterative: bool,
        is_negated_zoom: bool,
        has_stabilize: bool,
        is_question: bool = False,
        has_camera_track: bool = False,
    ) -> str:
        """Create a clear, contextual, human-like Romanian reply explaining decisions."""
        lines: list[str] = []

        if has_camera_track:
            if is_question:
                lines.append(
                    "🎬 **Da, absolut! Putem realiza mișcări consecvente și fluide cu camera, "
                    "care să meargă în armonie cu mișcarea persoanei.**\n\n"
                    "În loc de o decupare rigidă sau un cadru static care lasă persoana să iasă, "
                    "am configurat **Dynamic Kinetic Camera Tracking**:\n"
                    "- **1. Urmărire Cinetică Fluidă (Kinetic Pan Track):** Camera analizează "
                    "traiectoria persoanei pe axa timpului și ghidează lin fereastra de filmare "
                    "odată cu pașii și deplasarea ei.\n"
                    "- **2. Amortizare Tip Gimbal (Inerție 1.5s):** Mișcarea are o curbă organică "
                    "de accelerare și decelerare, astfel încât camera nu smucește la mișcări mici, "
                    "ci glisează elegant ca un operator profesionist.\n"
                    "- **3. Corectare Defect Filmare:** Dacă ai dat zoom-out din greșeală când "
                    "persoana s-a apropiat, urmărirea combinată cu stabilizarea 2-pass "
                    "compensează optic trepidațiile și salturile de perspectivă."
                )
            else:
                lines.append(
                    "🎬 **Am activat urmărirea consecventă a camerei "
                    "(Dynamic Kinetic Camera Tracking):**\n"
                    "Camera va glisa lin și organic urmărind deplasarea persoanei în cadru, "
                    "cu inerție de amortizare."
                )
        elif is_iterative:
            lines.append("🧠 **Am înțeles contextul:** Păstrez setările de la randarea precedentă.")
            if is_negated_zoom:
                lines.append(
                    "- **Zoom persoană:** L-am dezactivat complet conform instrucțiunii tale."
                )
            if has_stabilize:
                lines.append(
                    "- **Stabilizare:** Am activat algoritmul 2-pass VidStab pentru a compensa "
                    "zoom-out-ul manual din filmare și a netezi mișcarea camerei."
                )
        elif has_stabilize and is_negated_zoom:
            lines.append(
                "🧠 **Analiză prompt:** Am identificat problema din filmarea inițială "
                "(zoom-out manual neuniform). Am dezactivat efectul de punch-in zoom și "
                "am activat stabilizarea 2-pass VidStab pentru a netezi cadrul."
            )
        elif is_question:
            lines.append("💡 **Da, sigur! Iată cum abordăm această solicitare:**")
        else:
            lines.append("🎬 **Am configurat planul de producție optimizat pentru cererea ta:**")

        lines.append("\n**📋 Etape configurate în noul spec:**")
        for i, op in enumerate(ops, 1):
            lines.append(f"- **{i}.** {op}")

        lines.append(
            "\nTotul este pregătit și sincronizat. "
            "Apasă butonul de mai jos sau trimite comanda pentru randare!"
        )
        return "\n".join(lines)

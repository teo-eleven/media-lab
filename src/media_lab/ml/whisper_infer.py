"""Whisper ML driver for audio transcription.

Runs Whisper as a child process via `ml_runner` to isolate PyTorch and Whisper.
Extracts text segments and word-level timestamps and saves to structured JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import whisper


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper transcription driver")
    parser.add_argument("input_audio", help="Path to input audio or video file")
    parser.add_argument("output_json", help="Path to write structured transcript JSON")
    parser.add_argument("--model", default="base", help="Whisper model name (default: base)")
    parser.add_argument("--language", default=None, help="Language code (e.g. en, ro, es)")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cpu", "cuda"],
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--no-word-timestamps",
        dest="word_timestamps",
        action="store_false",
        help="Disable word-level timestamps",
    )

    args = parser.parse_args()

    input_path = Path(args.input_audio).resolve()
    if not input_path.is_file():
        sys.stderr.write(f"Input file not found: {input_path}\n")
        sys.exit(1)

    out_json = Path(args.output_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    start_time = time.monotonic()
    try:
        model = whisper.load_model(args.model, device=device)
    except Exception as exc:  # noqa: BLE001
        if device != "cpu":
            sys.stderr.write(f"Device {device} failed ({exc}), falling back to cpu...\n")
            model = whisper.load_model(args.model, device="cpu")
        else:
            sys.stderr.write(f"Failed to load Whisper model '{args.model}': {exc}\n")
            sys.exit(1)

    transcribe_kwargs: dict[str, object] = {
        "word_timestamps": args.word_timestamps,
    }
    if args.language:
        transcribe_kwargs["language"] = args.language

    try:
        result = model.transcribe(str(input_path), **transcribe_kwargs)
    except Exception as exc:  # noqa: BLE001
        if device == "mps":
            sys.stderr.write(f"Whisper inference failed on MPS ({exc}), falling back to CPU...\n")
            try:
                model = whisper.load_model(args.model, device="cpu")
                result = model.transcribe(str(input_path), **transcribe_kwargs)
            except Exception as cpu_exc:  # noqa: BLE001
                sys.stderr.write(f"Whisper transcription failed on CPU fallback: {cpu_exc}\n")
                sys.exit(1)
        else:
            sys.stderr.write(f"Whisper transcription failed: {exc}\n")
            sys.exit(1)

    clean_segments = []
    for i, s in enumerate(result.get("segments", [])):
        words = []
        for w in s.get("words", []):
            words.append(
                {
                    "word": str(w.get("word", "")).strip(),
                    "start": round(float(w.get("start", 0.0)), 3),
                    "end": round(float(w.get("end", 0.0)), 3),
                    "probability": round(float(w.get("probability", 1.0)), 3),
                }
            )
        clean_segments.append(
            {
                "id": s.get("id", i),
                "start": round(float(s.get("start", 0.0)), 3),
                "end": round(float(s.get("end", 0.0)), 3),
                "text": str(s.get("text", "")).strip(),
                "words": words,
            }
        )

    payload = {
        "text": str(result.get("text", "")).strip(),
        "language": str(result.get("language", "")),
        "duration_s": round(time.monotonic() - start_time, 2),
        "segments": clean_segments,
    }

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    sys.stdout.write(
        f"Transcribed {len(clean_segments)} segments ({payload['language']}) "
        f"in {payload['duration_s']}s\n"
    )


if __name__ == "__main__":
    main()

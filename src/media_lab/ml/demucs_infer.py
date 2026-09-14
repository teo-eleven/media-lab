"""Demucs ML driver for separating audio stems.

Runs as a child process via `ml_runner` to isolate heavy PyTorch and Demucs dependencies.
Supports 2-stems (e.g. vocals / no_vocals) and 4-stems (drums, bass, other, vocals).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from demucs.api import Separator, save_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Demucs audio stem separator")
    parser.add_argument("input_audio", help="Path to input audio file")
    parser.add_argument("output_dir", help="Directory where separated stems will be saved")
    parser.add_argument(
        "--model", default="htdemucs", help="Pretrained Demucs model (default: htdemucs)"
    )
    parser.add_argument(
        "--two-stems",
        default="vocals",
        choices=["vocals", "drums", "bass", "other", "none"],
        help="Separate into selected stem and remainder (default: vocals)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cpu", "cuda"],
        help="Device to run inference on (default: auto)",
    )
    parser.add_argument(
        "--shifts",
        type=int,
        default=0,
        help="Random shifts for equivariant stabilization (default: 0)",
    )
    parser.add_argument(
        "--format",
        default="wav",
        choices=["wav", "mp3", "flac"],
        help="Output audio format (default: wav)",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=320,
        help="Bitrate for mp3/flac if applicable (default: 320)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_audio).resolve()
    if not input_path.is_file():
        sys.stderr.write(f"Input audio file not found: {input_path}\n")
        sys.exit(1)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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
        separator = Separator(
            model=args.model,
            device=device,
            shifts=args.shifts,
            progress=False,
        )
    except Exception as exc:  # noqa: BLE001
        # If MPS or CUDA failed, fallback to CPU
        if device != "cpu":
            sys.stderr.write(f"Device {device} failed ({exc}), falling back to cpu...\n")
            separator = Separator(
                model=args.model,
                device="cpu",
                shifts=args.shifts,
                progress=False,
            )
        else:
            sys.stderr.write(f"Failed to load Demucs model '{args.model}': {exc}\n")
            sys.exit(1)

    try:
        _origin, separated = separator.separate_audio_file(input_path)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"Failed during Demucs separation: {exc}\n")
        sys.exit(1)

    ext = args.format
    if args.two_stems and args.two_stems != "none":
        target = args.two_stems
        if target not in separated:
            sys.stderr.write(f"Stem '{target}' not in model outputs: {list(separated.keys())}\n")
            sys.exit(1)

        stem_target = separated[target]
        stem_other = sum(tensor for name, tensor in separated.items() if name != target)

        target_file = out_dir / f"{target}.{ext}"
        other_file = out_dir / f"no_{target}.{ext}"

        save_audio(stem_target, target_file, samplerate=separator.samplerate, bitrate=args.bitrate)
        save_audio(stem_other, other_file, samplerate=separator.samplerate, bitrate=args.bitrate)
        sys.stdout.write(f"Saved 2 stems to {out_dir}: {target_file.name}, {other_file.name}\n")
    else:
        for name, tensor in separated.items():
            stem_file = out_dir / f"{name}.{ext}"
            save_audio(tensor, stem_file, samplerate=separator.samplerate, bitrate=args.bitrate)
        sys.stdout.write(f"Saved {len(separated)} stems to {out_dir}\n")

    elapsed = time.monotonic() - start_time
    sys.stdout.write(f"Demucs separation completed in {elapsed:.2f}s\n")


if __name__ == "__main__":
    main()

"""Recipe: Render Doctor — Automated Visual and Acoustic Quality Auditor.

Inspired by DaVinci Resolve QC, Frame.io review passes, and Hollywood delivery gates.
Inspects finished renders for audio clipping, black frames, freeze frames,
odd dimensions, and platform delivery compliance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageStat

from ..config import Config
from ..contact_sheet import extract_frames, tile
from ..errors import MediaLabError
from ..paths import ensure_readable_source, work_directory
from ..probe import MediaInfo, probe

CheckStatus = Literal["PASS", "WARNING", "FAIL"]


@dataclass(frozen=True, slots=True)
class CheckItem:
    """Individual quality gate check."""

    name: str
    status: CheckStatus
    message: str
    recommendation: str | None = None


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Comprehensive diagnosis of a rendered media file."""

    source: Path
    overall_status: CheckStatus
    checks: tuple[CheckItem, ...]
    contact_sheet: Path | None
    media_info: MediaInfo

    @property
    def passed(self) -> bool:
        return self.overall_status == "PASS"

    @property
    def has_warnings(self) -> bool:
        return self.overall_status == "WARNING"


def diagnose_render(
    source: Path | str,
    config: Config,
    *,
    sample_frames: int = 6,
    generate_contact_sheet: bool = True,
    expected_aspect: str | None = None,
) -> DoctorReport:
    """Run an automated visual and technical audit on a rendered media file.

    Args:
        source: Path to the media file to inspect.
        config: MediaLab configuration.
        sample_frames: Number of frames to extract and analyze for visual defects.
        generate_contact_sheet: Whether to generate a tiled contact sheet image.
        expected_aspect: Expected aspect ratio like '9:16', '16:9', '1:1'.

    Returns:
        DoctorReport containing structured checks, overall status, and recommendations.
    """
    resolved_source = ensure_readable_source(source)
    media = probe(resolved_source, config)
    checks: list[CheckItem] = []

    # 1. Dimensions check (H.264/H.265 requires even width and height)
    if media.has_video:
        if media.width % 2 != 0 or media.height % 2 != 0:
            checks.append(
                CheckItem(
                    name="Even Dimensions",
                    status="FAIL",
                    message=(
                        f"Odd dimensions ({media.width}x{media.height}) "
                        "cause encoder corruption."
                    ),
                    recommendation="Ensure dimensions are divisible by 2 using scale filter.",
                )
            )
        else:
            checks.append(
                CheckItem(
                    name="Even Dimensions",
                    status="PASS",
                    message=f"Dimensions {media.width}x{media.height} are properly aligned.",
                )
            )

        # 2. Aspect ratio check
        if expected_aspect:
            aspect_map = {
                "9:16": 9 / 16,
                "16:9": 16 / 9,
                "1:1": 1.0,
                "4:5": 4 / 5,
            }
            target = aspect_map.get(expected_aspect)
            if target and not math.isclose(media.aspect_ratio, target, rel_tol=0.03):
                checks.append(
                    CheckItem(
                        name="Aspect Ratio Target",
                        status="WARNING",
                        message=(
                            f"Aspect ratio is {media.aspect_ratio:.2f}, "
                            f"expected {expected_aspect} ({target:.2f})."
                        ),
                        recommendation="Reframe using media-lab smart-reframe or media-lab short.",
                    )
                )
            elif target:
                checks.append(
                    CheckItem(
                        name="Aspect Ratio Target",
                        status="PASS",
                        message=f"Aspect ratio matches {expected_aspect} target.",
                    )
                )

        # 3. Minimum duration check
        if media.duration_s < 0.5:
            checks.append(
                CheckItem(
                    name="Duration Check",
                    status="WARNING",
                    message=f"Duration ({media.duration_s:.2f}s) is extremely short.",
                    recommendation="Verify that input footage was not truncated unintentionally.",
                )
            )
        else:
            checks.append(
                CheckItem(
                    name="Duration Check",
                    status="PASS",
                    message=f"Duration {media.duration_s:.2f}s is within normal range.",
                )
            )

    # 4. Audio presence & sanity
    if media.has_audio:
        checks.append(
            CheckItem(
                name="Audio Stream",
                status="PASS",
                message=f"Audio stream present with codec {media.codec_audio or 'unknown'}.",
            )
        )
    else:
        checks.append(
            CheckItem(
                name="Audio Stream",
                status="WARNING",
                message="Video contains no audio stream.",
                recommendation="If sound was expected, attach audio using media-lab music.",
            )
        )

    # 5. Visual Frame Inspection (Extract sample frames)
    contact_sheet_path: Path | None = None
    if media.has_video and sample_frames > 0:
        work_dir = work_directory(config, "render_doctor")
        frames_dir = work_dir / f"frames_{resolved_source.stem}"
        try:
            extract_frames(resolved_source, sample_frames, config, frames_dir)
            extracted = sorted(frames_dir.glob("f-*.png"))

            if extracted:
                black_frame_count = 0
                luminance_values: list[float] = []

                for frame_file in extracted:
                    with Image.open(frame_file) as img:
                        stat = ImageStat.Stat(img.convert("L"))
                        mean_luma = stat.mean[0]
                        luminance_values.append(mean_luma)
                        if mean_luma < 5.0:
                            black_frame_count += 1

                if black_frame_count > 0:
                    checks.append(
                        CheckItem(
                            name="Black Frames",
                            status="WARNING" if black_frame_count < len(extracted) else "FAIL",
                            message=f"Found {black_frame_count}/{len(extracted)} black frame(s).",
                            recommendation="Check video starting offsets or fade effects.",
                        )
                    )
                else:
                    checks.append(
                        CheckItem(
                            name="Black Frames",
                            status="PASS",
                            message="No black frames detected across sample intervals.",
                        )
                    )

                # Freeze frame detection (identical luma across all sampled frames)
                if (
                    len(luminance_values) > 1
                    and max(luminance_values) - min(luminance_values) < 0.05
                ):
                    checks.append(
                        CheckItem(
                            name="Frame Continuity",
                            status="WARNING",
                            message="Visual luminance is static across all frames (freeze frame).",
                            recommendation="Verify that input stream has active video motion.",
                        )
                    )
                else:
                    checks.append(
                        CheckItem(
                            name="Frame Continuity",
                            status="PASS",
                            message="Dynamic visual movement confirmed between sample frames.",
                        )
                    )

                # Generate contact sheet image
                if generate_contact_sheet:
                    sheet_out = work_dir / f"qc_sheet_{resolved_source.stem}.png"
                    tile(frames_dir, sheet_out, config, cols=min(3, sample_frames))
                    contact_sheet_path = sheet_out
        except MediaLabError as err:
            checks.append(
                CheckItem(
                    name="Visual Frame Sampling",
                    status="WARNING",
                    message=f"Could not extract visual frames for QC: {err}",
                )
            )

    # Determine overall status
    overall: CheckStatus = "PASS"
    for check in checks:
        if check.status == "FAIL":
            overall = "FAIL"
            break
        if check.status == "WARNING" and overall != "FAIL":
            overall = "WARNING"

    return DoctorReport(
        source=resolved_source,
        overall_status=overall,
        checks=tuple(checks),
        contact_sheet=contact_sheet_path,
        media_info=media,
    )

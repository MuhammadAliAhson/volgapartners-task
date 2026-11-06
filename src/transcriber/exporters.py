"""Render a transcript as JSON, SRT, WebVTT, or plain text.

Timestamp formatting is isolated so hour rollover and millisecond rounding are
tested once and shared by SRT (comma) and VTT (dot).
"""

import math

from transcriber.models import TranscriptionResult


def format_timestamp(seconds: float, *, decimal_separator: str) -> str:
    """Format seconds as ``HH:MM:SS`` plus milliseconds.

    Rounding is half-up. Hours are not wrapped at 24, so a long recording
    stays unambiguous, and 3600 seconds is ``01:00:00`` rather than ``60:00``.
    """
    total_ms = _rounded_millis(seconds)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_separator}{millis:03d}"


def to_json(result: TranscriptionResult) -> str:
    """Pretty-print the output contract."""
    return result.model_dump_json(indent=2) + "\n"


def to_srt(result: TranscriptionResult) -> str:
    """SubRip captions. Cues are separated by a blank line."""
    blocks = [
        _cue(index, segment.start, segment.end, segment.text, ",")
        for index, segment in enumerate(result.segments, start=1)
    ]
    if not blocks:
        return ""
    return "\n".join(blocks)


def to_vtt(result: TranscriptionResult) -> str:
    """WebVTT captions using a dot before the milliseconds."""
    cues = [_body(segment.start, segment.end, segment.text, ".") for segment in result.segments]
    if not cues:
        return "WEBVTT\n"
    return "WEBVTT\n\n" + "\n\n".join(cues) + "\n"


def to_txt(result: TranscriptionResult) -> str:
    """The cleaned full text, with a trailing newline when it is non-empty."""
    if not result.text:
        return ""
    return f"{result.text}\n"


def _rounded_millis(seconds: float) -> int:
    if seconds <= 0:
        return 0
    return math.floor(seconds * 1000 + 0.5)


def _cue(index: int, start: float, end: float, text: str, separator: str) -> str:
    return f"{index}\n{_body(start, end, text, separator)}\n"


def _body(start: float, end: float, text: str, separator: str) -> str:
    opened = format_timestamp(start, decimal_separator=separator)
    closed = format_timestamp(end, decimal_separator=separator)
    return f"{opened} --> {closed}\n{text}"

"""Subtitle timestamps roll hours over and round milliseconds half-up."""

import json

from transcriber.exporters import format_timestamp, to_json, to_srt, to_txt, to_vtt
from transcriber.models import Segment, TranscriptionResult


def _result() -> TranscriptionResult:
    return TranscriptionResult(
        engine="whisper",
        language="en",
        duration=7200,
        word_count=2,
        text="Hello there.",
        segments=(Segment(start=3661.25, end=3662.5, text="Hello there."),),
    )


def test_hour_rollover_timestamps() -> None:
    """3600 seconds is one hour, and 3661.25 keeps the milliseconds."""
    assert format_timestamp(0, decimal_separator=",") == "00:00:00,000"
    assert format_timestamp(3600, decimal_separator=",") == "01:00:00,000"
    assert format_timestamp(3661.25, decimal_separator=",") == "01:01:01,250"
    assert format_timestamp(3662.5, decimal_separator=".") == "01:01:02.500"


def test_exact_srt_and_vtt() -> None:
    """The cue text matches the subtitle specs, including the millisecond separator."""
    result = _result()
    assert to_srt(result) == "1\n01:01:01,250 --> 01:01:02,500\nHello there.\n"
    assert to_vtt(result) == "WEBVTT\n\n01:01:01.250 --> 01:01:02.500\nHello there.\n"
    assert to_txt(result) == "Hello there.\n"
    payload = json.loads(to_json(result))
    assert payload["segments"][0]["start"] == 3661.25
    assert set(payload) == {"engine", "language", "duration", "word_count", "text", "segments"}


def test_two_srt_cues_are_separated_by_a_blank_line() -> None:
    """Players expect a blank line between cues."""
    result = TranscriptionResult(
        engine="whisper",
        language="en",
        duration=4,
        word_count=2,
        text="One Two",
        segments=(
            Segment(start=0, end=1, text="One"),
            Segment(start=1, end=2, text="Two"),
        ),
    )
    rendered = to_srt(result)
    assert "\n\n2\n" in rendered
    assert (
        to_vtt(
            TranscriptionResult(
                engine="whisper",
                language=None,
                duration=0,
                word_count=0,
                text="",
                segments=(),
            )
        )
        == "WEBVTT\n"
    )
    assert to_txt(result.model_copy(update={"text": ""})) == ""

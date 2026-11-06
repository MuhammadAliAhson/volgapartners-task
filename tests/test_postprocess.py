"""Postprocess produces text that downstream tools can trust."""

import pytest
from pydantic import ValidationError

from transcriber.models import Segment
from transcriber.postprocess import postprocess


def test_whitespace_duplicates_order_and_clamp() -> None:
    """Blanks and adjacent repeats go away, then segments are sorted and clamped."""
    segments = [
        Segment(start=1, end=2, text="  hello   there "),
        Segment(start=2, end=3, text="   "),
        Segment(start=3, end=4, text="hello there"),
        Segment(start=0.5, end=1, text="intro"),
        Segment(start=4, end=9, text="tail"),
        Segment(start=6, end=7, text="beyond"),
    ]
    result = postprocess(segments, engine="whisper", language="en", duration=5)
    assert [segment.text for segment in result.segments] == ["intro", "hello there", "tail"]
    assert result.segments[-1].end == 5
    assert result.text == "intro hello there tail"
    assert result.word_count == 4
    assert result.engine == "whisper"
    assert result.duration == 5


def test_segment_rejects_inverted_times() -> None:
    """A segment that runs backwards is a model bug, not something we store."""
    with pytest.raises(ValidationError):
        Segment(start=2, end=1, text="nope")

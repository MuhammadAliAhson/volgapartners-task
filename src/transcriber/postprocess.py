"""Clean a raw transcript before it leaves the pipeline.

Downstream search, subtitles, summarization, and analytics all assume text is
non-empty, de-duplicated, and clamped to the file duration. Doing that in one
pure function keeps the engines free of presentation rules.
"""

from collections.abc import Sequence

from transcriber.models import Segment, TranscriptionResult


def postprocess(
    segments: Sequence[Segment],
    *,
    engine: str,
    language: str | None,
    duration: float,
) -> TranscriptionResult:
    """Normalize, drop blanks and loop artefacts, then rebuild the full text.

    Duplicate removal happens before sorting because Whisper's repeat artefact
    is two adjacent copies, not two copies that merely share a start time.
    """
    normalized = [_normalize(segment) for segment in segments]
    nonempty = [segment for segment in normalized if segment.text]
    unique = _drop_adjacent_duplicates(nonempty)
    fitted = _fit_to_duration(unique, duration)
    text = " ".join(segment.text for segment in fitted)
    return TranscriptionResult(
        engine=engine,
        language=language,
        duration=duration,
        word_count=len(text.split()),
        text=text,
        segments=tuple(fitted),
    )


def _normalize(segment: Segment) -> Segment:
    text = " ".join(segment.text.split())
    if text == segment.text:
        return segment
    return segment.model_copy(update={"text": text})


def _drop_adjacent_duplicates(segments: Sequence[Segment]) -> list[Segment]:
    kept: list[Segment] = []
    for segment in segments:
        if kept and kept[-1].text == segment.text:
            continue
        kept.append(segment)
    return kept


def _fit_to_duration(segments: Sequence[Segment], duration: float) -> list[Segment]:
    fitted: list[Segment] = []
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        if segment.start > duration:
            continue
        end = min(segment.end, duration)
        if end < segment.start:
            continue
        if end == segment.end:
            fitted.append(segment)
        else:
            fitted.append(segment.model_copy(update={"end": end}))
    return fitted

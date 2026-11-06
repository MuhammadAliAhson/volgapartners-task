"""Error hierarchy for the transcription pipeline.

Callers catch these types instead of raw ``subprocess`` or ``httpx`` errors.
Transient engine failures are the only ones the retry helper will repeat.
"""


class TranscriberError(Exception):
    """Base class for every expected failure in the pipeline."""


class ConfigurationError(TranscriberError):
    """The process is misconfigured, for example a missing API key."""


class UnsupportedFormatError(TranscriberError):
    """The input extension is not allowed, or the file has no audio stream."""


class AudioProcessingError(TranscriberError):
    """ffmpeg or ffprobe failed, timed out, or produced an unreadable checkpoint."""


class EngineError(TranscriberError):
    """A speech-to-text engine failed."""


class TransientEngineError(EngineError):
    """A failure that may succeed if the same call is repeated.

    Typical causes are timeouts, HTTP 429, HTTP 5xx, and connection errors.
    """


class PermanentEngineError(EngineError):
    """A failure that will not succeed if repeated, such as HTTP 401 or bad audio."""

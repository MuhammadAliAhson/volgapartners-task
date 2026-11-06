"""Command-line entry point.

Logging is configured only here. Library modules log through the standard
logger and never print, so embedding the pipeline does not write to stdout.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from transcriber.config import Settings
from transcriber.engines import get_engine
from transcriber.exceptions import TranscriberError, UnsupportedFormatError
from transcriber.exporters import to_json, to_srt, to_txt, to_vtt
from transcriber.models import TranscriptionResult
from transcriber.pipeline import TranscriptionPipeline

_FORMATS = {"json": to_json, "srt": to_srt, "vtt": to_vtt, "txt": to_txt}


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and transcribe one file. Returns a process exit code."""
    parser = _parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        if code is None or code == 0:
            return 0
        return int(code) if isinstance(code, int) else 2
    try:
        return _transcribe(args)
    except UnsupportedFormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (TranscriberError, FileNotFoundError, OSError, ValidationError) as exc:
        print(f"error: {_public_message(exc)}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcriber",
        description="Transcribe an audio file to JSON, SRT, VTT, and plain text.",
    )
    parser.add_argument("audio", type=Path, help="Audio or video file to transcribe")
    parser.add_argument("--engine", choices=["whisper", "elevenlabs"], default=None)
    parser.add_argument("--model", default=None, help="Whisper size or ElevenLabs model id")
    parser.add_argument("--language", default=None, help="Force a language code")
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--workdir", type=Path, default=None, help="Keep checkpoints and resume")
    parser.add_argument(
        "--format", default="json,srt,txt", help="Comma-separated: json,srt,vtt,txt"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _transcribe(args: argparse.Namespace) -> int:
    _configure_logging(bool(args.verbose))
    settings = Settings()
    _ensure_supported_input(args.audio, settings)
    settings, engine_name = _apply_model(settings, args.engine, args.model)
    formats = _parse_formats(str(args.format))
    engine = get_engine(engine_name, settings)
    try:
        result = TranscriptionPipeline(engine, settings).run(
            args.audio,
            language=args.language,
            workdir=args.workdir,
        )
    finally:
        closer = getattr(engine, "close", None)
        if callable(closer):
            closer()
    paths = _write_outputs(result, args.out_dir, formats)
    print(_summary(result, paths))
    return 0


def _apply_model(settings: Settings, engine: str | None, model: str | None) -> tuple[Settings, str]:
    engine_name = engine or settings.default_engine
    if not model:
        return settings, engine_name
    field = "elevenlabs_model" if engine_name == "elevenlabs" else "whisper_model"
    return settings.model_copy(update={field: model}), engine_name


def _ensure_supported_input(path: Path, settings: Settings) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    extension = path.suffix.lower().lstrip(".")
    if extension not in settings.supported_extensions:
        raise UnsupportedFormatError(f"Unsupported audio extension: {extension or 'none'}")


def _parse_formats(raw: str) -> tuple[str, ...]:
    formats = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    unknown = [item for item in formats if item not in _FORMATS]
    if not formats or unknown:
        raise UnsupportedFormatError("Format must be a comma-separated list of json, srt, vtt, txt")
    return formats


def _write_outputs(
    result: TranscriptionResult,
    directory: Path,
    formats: Sequence[str],
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        destination = directory / f"transcript.{fmt}"
        destination.write_text(_FORMATS[fmt](result), encoding="utf-8")
        paths.append(destination)
    return paths


def _summary(result: TranscriptionResult, paths: Sequence[Path]) -> str:
    rendered = ", ".join(str(path) for path in paths)
    language = result.language or "unknown"
    return (
        f"language={language} duration={result.duration:.2f}s "
        f"words={result.word_count} files={rendered}"
    )


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _public_message(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "invalid configuration"
    return str(exc)


if __name__ == "__main__":
    raise SystemExit(main())

"""Per-job checkpoint directory.

Checkpoints are namespaced by engine so switching engines never reuses another
engine's transcript. Writes go to a temp file and are renamed into place, so a
crash cannot leave a half-written JSON file for the next run to trust.
"""

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

from transcriber.exceptions import AudioProcessingError
from transcriber.models import ChunkSpec, Segment

logger = logging.getLogger(__name__)


class JobWorkspace:
    """Checkpoint store. A missing root becomes a temporary directory.

    The temporary directory is removed by ``close``. A caller-supplied root is
    kept, which is what makes ``--workdir`` resume possible.
    """

    def __init__(self, root: Path | None, engine: str) -> None:
        if not engine:
            raise AudioProcessingError("Engine name is required to namespace checkpoints")
        if root is None:
            self._temp: tempfile.TemporaryDirectory[str] | None = tempfile.TemporaryDirectory(
                prefix="transcriber-"
            )
            self.root = Path(self._temp.name)
        else:
            self._temp = None
            self.root = root
        self.engine_dir = self.root / engine
        self.engine_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """Delete the directory when this workspace owns it."""
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None

    def __enter__(self) -> "JobWorkspace":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def normalized_path(self) -> Path:
        """Path of the 16 kHz mono WAV for this engine."""
        return self.engine_dir / "normalized.wav"

    def chunk_audio_path(self, index: int) -> Path:
        """Temporary WAV for one chunk. Deleted after the checkpoint is saved."""
        return self.engine_dir / "parts" / f"{index:04d}.wav"

    def clip_dir(self) -> Path:
        """Directory that holds short clips used for language detection."""
        return self.engine_dir / "clips"

    def load_plan(self) -> list[ChunkSpec] | None:
        """Return the saved chunk plan, or ``None`` when this run must plan."""
        raw = self._read_json(self.engine_dir / "plan.json")
        if raw is None:
            return None
        if not isinstance(raw, list):
            raise AudioProcessingError("Chunk plan checkpoint is unreadable")
        return [ChunkSpec.model_validate(item) for item in raw]

    def save_plan(self, chunks: Sequence[ChunkSpec]) -> None:
        """Atomically store the chunk plan."""
        payload = json.dumps([chunk.model_dump() for chunk in chunks])
        _write_atomic(self.engine_dir / "plan.json", payload)

    def load_chunk(self, index: int) -> list[Segment] | None:
        """Return a finished chunk, or ``None`` when it still needs work."""
        raw = self._read_json(self._chunk_path(index))
        if raw is None:
            return None
        if not isinstance(raw, list):
            raise AudioProcessingError(f"Chunk checkpoint {index} is unreadable")
        return [Segment.model_validate(item) for item in raw]

    def save_chunk(self, index: int, segments: Sequence[Segment]) -> None:
        """Atomically store one chunk's segments on the original timeline."""
        payload = json.dumps([segment.model_dump() for segment in segments])
        _write_atomic(self._chunk_path(index), payload)

    def load_language(self) -> str | None:
        """Return the language already chosen for this engine, if any."""
        raw = self._read_json(self.engine_dir / "language.json")
        if not isinstance(raw, dict):
            return None
        language = raw.get("language")
        if isinstance(language, str) and language.strip():
            return language.strip()
        return None

    def save_language(self, language: str) -> None:
        """Atomically store the language forced on every chunk."""
        _write_atomic(self.engine_dir / "language.json", json.dumps({"language": language}))

    def clear_chunks(self) -> None:
        """Drop chunk transcripts so a language change is not mixed with old text."""
        directory = self.engine_dir / "chunks"
        if directory.exists():
            shutil.rmtree(directory)
        logger.info("Cleared chunk checkpoints under %s", directory.name)

    def _chunk_path(self, index: int) -> Path:
        return self.engine_dir / "chunks" / f"{index:04d}.json"

    def _read_json(self, path: Path) -> object | None:
        if not path.is_file():
            return None
        try:
            parsed: object = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AudioProcessingError(f"Checkpoint {path.name} is unreadable") from exc
        return parsed


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise

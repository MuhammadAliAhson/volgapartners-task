# Transcription pipeline

Turn an audio or video file into a timestamped transcript. One pipeline normalizes the media, detects the language once, splits long files on silence, and writes JSON, SubRip, WebVTT, and plain text.

```text
audio → validate → ffmpeg normalize (16 kHz mono WAV) → detect language once → plan chunks at silences → transcribe each chunk with the language forced → add chunk offset → merge → postprocess → export JSON / SRT / VTT / TXT
```

Finished stages are written to a work directory. Running the same command again with that directory skips work that already succeeded.

## Quickstart

Install [ffmpeg](https://ffmpeg.org/) (both `ffmpeg` and `ffprobe` must be on `PATH`), then:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[api,dev]"
```

Copy `.env.example` to `.env` only if you want ElevenLabs. Whisper, the default, needs no key.

Transcribe a file. This writes `transcript.json`, `transcript.srt`, and `transcript.txt` in the output directory:

```bash
transcriber sample.mp3 --out-dir out
transcriber interview.wav --engine whisper --model tiny --language en --workdir work --format json,srt,vtt,txt -v
```

`--workdir` is what makes resume work. If the process stops after chunk 2, the same command continues at chunk 3.

Start the API (one job at a time by default):

```bash
uvicorn transcriber.api.app:app --host 0.0.0.0 --port 8000
```

```bash
curl -s -F "file=@sample.mp3" -F "engine=whisper" http://127.0.0.1:8000/v1/transcriptions
curl -s http://127.0.0.1:8000/v1/transcriptions/JOB_ID
curl -s "http://127.0.0.1:8000/v1/transcriptions/JOB_ID/export?format=srt" -o transcript.srt
curl -s -X DELETE http://127.0.0.1:8000/v1/transcriptions/JOB_ID
curl -s http://127.0.0.1:8000/health
```

`POST` returns `202` with an id. Poll until `status` is `completed`, then export. The upload is stored as `data/<id>/audio.<ext>`. The client's filename is never used as a path.

## Output schema

Timestamps are seconds from the start of the original file.

```json
{
  "engine": "whisper",
  "language": "en",
  "duration": 1834.52,
  "word_count": 4210,
  "text": "Full transcript ...",
  "segments": [{ "start": 0.0, "end": 4.2, "text": "Hello and welcome." }]
}
```

SRT uses `HH:MM:SS,mmm`. WebVTT uses a dot before the milliseconds. Hours are not wrapped at 24, so a two-hour file shows `02:00:00`. Milliseconds use half-up rounding.

## Design decisions

### Engine choice

Whisper is the default because a reviewer can run the project with no account, no GPU, and no audio leaving the machine. `faster-whisper` with `int8` is the CPU-friendly build of that model. ElevenLabs Scribe is optional: it is accurate and does not need a local model, but it is paid and the audio is sent to ElevenLabs.

| | Whisper (`faster-whisper`) | ElevenLabs Scribe |
|---|---|---|
| Cost | Free, local | Paid API |
| Privacy | Audio stays on the machine | Audio is uploaded |
| Setup | ffmpeg + a model download on first use | `ELEVENLABS_API_KEY` |
| Hardware | CPU is enough (`int8`); GPU is used when present | No local GPU |
| Timestamps | Segment-level from the model | Word-level, grouped into segments |
| Language | `detect_language` on several clips, then forced | No detect endpoint; the middle clip is transcribed and `language_code` is kept |

Both engines implement the same protocol (`detect_language`, `transcribe`). The pipeline does not know which one it has. Timestamps from either engine are relative to the chunk; only the pipeline adds the chunk offset.

### Audio formats

The extension is checked against an allow-list, then ffprobe decides whether an audio stream exists. A renamed `.txt` is rejected even if it ends in `.wav`. After that, ffmpeg decodes everything to 16 kHz mono PCM WAV (`-vn`, so a video file works). One layout downstream means the chunker, the language sampler, and both engines share a single code path. The cost is an extra decode. That is cheaper than teaching every stage about MP3, M4A, and MP4.

The normalized WAV is renamed into place. A crash during decode cannot leave a half-written file that a later resume would trust.

### Long audio

Target chunk length is 10 minutes (`chunk_seconds=600`). The planner walks forward by that amount and, within ±`silence_window_seconds`, snaps the cut to the nearest silence. Silences come from ffmpeg `silencedetect`; the value used is the midpoint, which is the point least likely to sit inside a word. Because a word is not split, chunks do not overlap and the merge step does not have to delete duplicated words.

If nothing is quiet near the goal, the cut is a hard cut at the goal. The next goal is one target length after the cut that was actually chosen, so an early snap does not make the following chunk longer than the window allows. Planning stops when the tail is at most `chunk_seconds + silence_window_seconds`, which bounds the last chunk. Cuts that would leave less than one second on either side are rejected.

Each chunk is transcribed with retries, its timestamps are shifted by `chunk.start`, and the segment list is saved atomically (`write temp` + `os.replace`). The chunk WAV is deleted after the checkpoint so disk use stays flat. A second run with the same `--workdir` loads the language, the plan, and any finished chunks and does not call the engine for those chunks. Checkpoints live under `workdir/<engine>/`, so switching engines cannot reuse another engine's text.

What this does not do: there is no overlap. Continuous speech with no silence falls back to hard cuts, which can split a word. Chunks are sequential, so a long file takes as long as the sum of its chunks. Memory stays roughly one chunk plus the model, not the whole file.

### Language detection

The language is chosen once, then forced on every chunk.

Per-chunk auto-detect looks attractive and is worse here. A quiet chunk, a music bed, or a short answer can flip the language, and the rest of that chunk becomes garbage. A fixed language is also faster: Whisper does not re-run detection inside every chunk, and Scribe is told `language_code` up front.

Detection uses several clips (by default at 25%, 50%, and 75%) because one clip can be silence or music. Short audio (no longer than one clip) uses a single clip from the start. Whisper sums language probabilities across clips and picks the winner. Scribe has no separate detect call, so the middle clip is transcribed and its `language_code` is stored.

A caller-supplied `--language` wins and is stored. If a later run asks for a different language, chunk checkpoints are dropped and transcribed again. The plan (where to cut) does not depend on language, so it is kept.

Known limitation: code-switched audio. One forced language cannot represent a conversation that changes language and stays there.

### Timestamps

Segment times in the output are seconds from the start of the original file, not from the start of a chunk. Engines return times relative to the chunk they were given. The pipeline adds `chunk.start` before the checkpoint is written, so a resume does not add the offset twice.

Whisper already returns segments. ElevenLabs returns words (`timestamps_granularity=word`). `words_to_segments` drops spacing and audio events, then starts a new segment on sentence-ending punctuation (`.`, `!`, `?`), on a gap longer than one second, or when the segment would exceed 15 seconds. Abbreviations that end with a period can split early. That is the trade-off for a pure function with no sentence parser.

### Downstream use

Search, subtitle players, LLM summarization, and analytics all want the same thing: text that is not blank, not repeated, and anchored to the file. `postprocess` does that in a fixed order:

1. Collapse whitespace inside each segment.
2. Drop empty segments.
3. Drop a segment whose text is identical to the previous one (a Whisper repeat loop).
4. Sort by start time and clamp `end` to the file duration.
5. Rebuild `text` and `word_count` from the segments that remain.

Exporters then render that object. They do not clean text again, so a subtitle and the JSON cannot disagree.

## Assumptions and known limitations

- The repository root is the project. The name `transcription-pipeline` is the distribution name; the import package is `transcriber`.
- "Short audio" for chunking means `duration <= chunk_seconds + silence_window_seconds` (one chunk, no silence scan). "Short" for language sampling means `duration <= language_clip_seconds` (one clip from the start).
- The next cut goal is one target length after the previous cut, not a fixed grid from t=0. When two silences are equally close, the earlier one wins.
- A file ffprobe cannot identify as audio is `UnsupportedFormatError`. Timeouts, a missing ffmpeg binary, and a failed decode are `AudioProcessingError`.
- Zero-duration audio is a successful empty transcript (`word_count == 0`). It does not call the engine.
- Retry jitter is uniform in `[0, retry_base_delay]`. Delay after attempt `n` (from 0) is `base * 2^n + jitter`. Only `TransientEngineError` is retried.
- The Whisper model is cached on the engine instance, behind a lock, not in a process-wide global. The API builds one engine per job. `max_concurrent_jobs` defaults to 1, so one job means one model.
- ElevenLabs `language_code` is stored as returned (ISO 639-1 or 639-3). The API key is read from `ELEVENLABS_API_KEY`.
- The API job index is in memory. Restarting the process forgets ids even if files remain under `data/`.
- Exporting a job that is not `completed` returns 409.
- No database, queue, authentication, or overlap-merge. No speaker diarization. No word-level timestamps for Whisper (Scribe words are grouped into segments before they leave the engine).

## Testing and layout

```bash
make check
```

That runs `ruff check`, `ruff format --check`, `mypy --strict` on `src`, and `pytest` with coverage required at 85% or higher. FFmpeg tests are marked `integration` and still run in the default suite. They are skipped only when `ffmpeg` or `ffprobe` is absent. Network, a real Whisper download, and an ElevenLabs key are not required: Whisper is exercised with a fake model, and Scribe is exercised with `respx`.

```text
src/transcriber/
  config.py  models.py  exceptions.py  retry.py
  audio/ffmpeg.py  audio/chunking.py
  engines/base.py  engines/whisper.py  engines/elevenlabs.py
  language.py  workspace.py  pipeline.py
  postprocess.py  exporters.py  cli.py
  api/  app, schemas, jobs, errors
tests/   one module per area, plus conftest.py
```

`TranscriptionPipeline` receives the engine and the settings. It never constructs an engine. Pure functions (`find_cuts`, `words_to_segments`, `postprocess`, the exporters, `format_timestamp`) hold the logic that is easy to get wrong.

## Future improvements

- Word-level timestamps for Whisper (`word_timestamps=True`) and the same grouping used for Scribe.
- Speaker diarization, which neither engine is asked for today.
- Parallel chunk transcription. Checkpoints already make the chunks independent; the default is serial so a laptop does not load the model twice.
- Per-chunk language detection for code-switched audio, as an explicit opt-in. The default should stay "detect once".

## Part 2: System Design

_To be added by the author._

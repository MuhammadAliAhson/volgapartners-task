# Transcription pipeline

Turn an audio or video file into a timestamped transcript from the command line.

```text
audio → validate → normalize to 16 kHz mono WAV → detect language once → split on silence → transcribe each chunk → shift timestamps → clean text → export JSON / SRT / VTT / TXT
```

Each finished stage is saved in a work directory. Run the same command again with that directory, and completed chunks are skipped.

## Quickstart

### Requirements

- Python 3.11 or newer
- `ffmpeg` and `ffprobe` on `PATH`

### Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Whisper is the default engine and does not need an API key. To use ElevenLabs later, set the key in `.env` at the project root:

```text
ELEVENLABS_API_KEY=your_key_here
```

`.env` is listed in `.gitignore`. `.env.example` lists every setting, with a short note on what each one does.

### Command line

Put sample files in the `audios` folder. These commands write `transcript.json`, `transcript.srt`, and `transcript.txt`:

```bash
transcriber sample.mp3 --out-dir out
transcriber interview.wav --engine whisper --model tiny --language en --workdir work --format json,srt,vtt,txt -v
```

`--workdir` keeps checkpoints. If the process stops after chunk 2, the same command continues at chunk 3.

## Output

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

- SRT timestamps use a comma: `HH:MM:SS,mmm`.
- WebVTT timestamps use a dot: `HH:MM:SS.mmm`.
- Hours are not wrapped at 24. Two hours is `02:00:00`.
- Milliseconds use half-up rounding.

## Design decisions

Each section is one choice: what the code does, why it does it that way, and what that costs.

### Engines

**Choice.** Whisper is the default. ElevenLabs is optional. Both sit behind the same two methods, `detect_language` and `transcribe`. The pipeline is given an engine. It never builds one itself.

**Why.** Whisper is free, runs on this machine, and needs no key, so the project can be tried without an account. The audio never leaves the machine. ElevenLabs is there when a cloud model is worth paying for and no local GPU is available. One interface means the chunking, timestamps, and exporters stay the same when the engine changes. Passing the engine in means tests can use a fake and never download a model.

| | Whisper | ElevenLabs Scribe |
|---|---|---|
| Cost | Free, local | Paid API |
| Privacy | Audio stays on the machine | Audio is uploaded |
| Setup | ffmpeg, then a model download on first use | `ELEVENLABS_API_KEY` in `.env` |
| Hardware | CPU is enough (`int8`); a GPU is used when present | No local GPU |
| Timestamps | Segments from the model | Words, then grouped into segments |
| Language | Probability vote across several clips, then forced | No detect call; the middle clip's `language_code` is kept |

**Trade-off.** ElevenLabs can be more accurate, but it costs money and sends the audio to a third party.

### Audio formats

**Choice.** Check the extension, confirm with `ffprobe` that an audio stream exists, then convert everything to 16 kHz mono WAV. Video is dropped.

**Why.** A renamed text file can end in `.mp3`, so the extension alone is not enough. After that check, every later step should see one layout. The chunker, the language clips, and both engines then share one path. Whisper also expects 16 kHz audio. Writing the WAV under a temporary name and renaming it means a crash cannot leave a half-written file that the next run would trust.

**Trade-off.** Every file is decoded once, even if it was already a WAV. That decode is cheaper than teaching each step to read MP3, M4A, and MP4.

### Long audio

**Choice.** Files longer than about 11 minutes are cut into pieces of about 10 minutes. The cut looks for a pause within ±60 seconds and uses the middle of that pause. If there is no pause, the cut is at the 10-minute mark. Pieces meet exactly. They do not overlap. Each finished piece is saved under `--workdir`.

**Why.** Whisper's model only reads about 30 seconds at a time. `faster-whisper` already walks a long piece in those windows, and it starts the next window at the last finished word, so a word on the 30-second edge is heard again and not dropped. This project does not cut every 30 seconds, because those windows are not saved. A 10-minute piece is the unit that can be retried and resumed. A pause is the cut point so the cut falls between words. Overlap is avoided because the same words would then appear twice and need a second merge step.

**Trade-off.** If someone talks through the whole minute around a 10-minute boundary, the cut can split a word. Pieces run one after another. Only one piece is in memory at a time.

### Language

**Choice.** Detect the language once, save it, and force that language on every piece. Three clips are taken, near 25%, 50%, and 75% of the file. A short file uses one clip from the start. `--language` skips detection.

**Why.** A quiet or noisy piece can be detected as the wrong language, and the rest of that piece comes out wrong. One decision is also faster, because detection does not run again inside every piece. Several clips are used because a single clip may be music or silence. Whisper adds up the language probabilities and keeps the winner. ElevenLabs has no separate detect call, so the middle clip is transcribed and its language code is stored. The saved language is reused on resume. A later run that asks for a different language transcribes the pieces again. The cut plan stays, because where to cut does not depend on language.

**Trade-off.** One language cannot follow a conversation that switches language and stays there.

### Timestamps

**Choice.** Every segment has a start and an end, in seconds from the start of the original file. The engine's times start at zero for the piece it was given. The pipeline adds the piece start once, then saves that result.

**Why.** Subtitles and search need a time in the original recording, not a time inside a 10-minute piece. Adding the offset before the save means a resumed run does not add it a second time. Whisper already returns segments. ElevenLabs returns words, so those words are grouped into a segment on sentence punctuation (`.`, `!`, `?`), on a pause longer than one second, or when the segment would pass 15 seconds. Spacing and non-speech events are ignored.

**Trade-off.** An abbreviation that ends with a period can start a new segment early. Whisper output is segment-level, not word-level.

### Prepared output

**Choice.** Clean the transcript in a fixed order, then export that same object as JSON, SRT, VTT, or TXT. The exporters do not clean the text again.

**Why.** Search, subtitles, summarization, and analytics all need text that is readable and tied to a time. Doing the cleanup once means the subtitle and the JSON cannot disagree.

1. Collapse extra whitespace.
2. Drop empty segments.
3. Drop a segment whose text is the same as the previous one. This removes a Whisper repeat.
4. Sort by start time and keep each end inside the file duration.
5. Rebuild the full text and the word count from the segments that remain.

**Trade-off.** Two different sentences that happen to use the same words, one after the other, are treated as a repeat and the second is dropped.

## Assumptions and limitations

- The installable project name is `transcription-pipeline`. The Python import name is `transcriber`.
- A file shorter than about 11 minutes is one piece. Silence detection is skipped because there is nothing to split.
- A file shorter than one language clip is sampled from the start only.
- The next cut is measured from the previous cut, not from a fixed grid. If two pauses are equally close, the earlier one is used.
- A file that is not audio is an unsupported format. A missing `ffmpeg`, a timeout, or a broken decode is an audio-processing error.
- Silent or empty audio returns an empty transcript. That is a success, not an error.
- Only a temporary engine failure is retried. A bad key or a bad request fails immediately. The wait doubles after each attempt.
- The Whisper model is loaded once per run and reused.
- faster-whisper 1.2 still passes an argument that PyAV 19 removed. The Whisper engine strips that argument before decoding.
- There are no speaker labels, and Whisper does not return word-level times.

## Testing and project structure

```bash
make check
```

`make check` runs Ruff, Ruff format, `mypy --strict` on `src`, and pytest. Coverage must stay at 85% or higher. Tests that need ffmpeg are skipped only when ffmpeg is missing. The suite does not download a Whisper model, call ElevenLabs, or use the network. Whisper is tested with a fake model. Scribe is tested with a mocked HTTP client.

```text
src/transcriber/
  config.py          settings from the environment and .env
  models.py          Segment, ChunkSpec, TranscriptionResult
  exceptions.py      error types
  retry.py           retry transient engine failures
  audio/ffmpeg.py    probe, normalize, cut, find silences
  audio/chunking.py  plan cut points
  engines/           Whisper and ElevenLabs
  language.py        sample clips and resolve one language
  workspace.py       checkpoints for resume
  pipeline.py        the steps in the diagram above
  postprocess.py     clean the transcript
  exporters.py       JSON, SRT, VTT, TXT
  cli.py             the transcriber command
tests/               one test module per area
```

## Future improvements

- Word-level timestamps for Whisper, grouped the same way as Scribe words.
- Speaker labels.
- Parallel chunk transcription. Chunks are already independent. The default stays serial so one machine loads the model once.
- Optional per-chunk language detection for audio that changes language. The default stays one language for the whole file.

## Related work

I have built this kind of system before.

- **Final year project.** A Whisper model I fine-tuned on agricultural speech, used in [Kisan Rabta](https://github.com/MuhammadAliAhson/Kisan_rabta_web).
- **Podcast AI.** A project I shipped at my most recent company. I can walk through that work in the next round.


---
license: cc-by-4.0
task_categories:
- text-to-speech
language:
- en
- te
tags:
- tts
- speech
- indian-languages
- telugu
- indian-english
- emotion
- single-speaker
pretty_name: Indian English + Telugu Single-Speaker TTS (emotion-tagged)
size_categories:
- n<1K
configs:
- config_name: indian_english
  data_files:
  - split: train
    path: indian_english/train-*
  - split: validation
    path: indian_english/validation-*
- config_name: telugu
  data_files:
  - split: train
    path: telugu/train-*
  - split: validation
    path: telugu/validation-*
---

# Indian English + Telugu Single-Speaker TTS Dataset (emotion-tagged)

Clean audio clips sourced from YouTube, transcribed with **Sarvam** ASR, segmented with
diarization, and labeled with emotion/style tags. Built as a data-quality / curation exercise.

> **"Single-speaker"** means **each clip contains exactly one speaker** (verified by
> diarization and speaker-embedding similarity). The dataset spans **11 distinct speakers
> total** (5 English, 6 Telugu), tracked via `speaker_id`.

## Contents
- **Indian English** (`indian_english`): 30.05 min, 142 clips, 5 speakers; emotions: {'neutral': 31, 'calm': 31, 'sad': 30, 'excited': 30, 'angry': 12, 'fearful': 5, 'happy': 3}
- **Telugu** (`telugu`): 30.25 min, 140 clips, 6 speakers; emotions: {'calm': 25, 'neutral': 24, 'angry': 24, 'excited': 24, 'sad': 24, 'fearful': 9, 'happy': 7, 'surprised': 3}

Total: **60.3 minutes**.

## Evaluation (evidence, not just claims)

- **Single-speaker check** (ECAPA-TDNN embeddings): same-speaker cosine 0.74 vs different-speaker 0.21 (separation 0.52, verification AUC 0.96 / EER 9.1%; 0/11 speakers flagged).
- **Transcript reliability**: English cross-ASR agreement with Whisper = 6.8% WER / 4.5% CER (n=40) — strong. Realtime ASR language-ID matched the target language on 100% of EN and 100% of TE clips. Telugu cross-ASR is not a valid proxy (Whisper is weak in Telugu); Telugu transcripts are best audited by human review.
- **Emotion-tag reliability** (sarvam-30b vs sarvam-105b on 120 clips): 65% agreement, Cohen's κ 0.55.
- **Phoneme coverage**: English 39 (100%), Telugu 45 (90%).

See the project report (GitHub repo) for full methodology and figures.

## Schema
`audio` (24 kHz mono), `text`, `normalized_text`, `language`, `language_code`,
`emotion` (neutral, happy, sad, angry, excited, calm, fearful, surprised), `style` (narrative, conversational, formal, expressive, whisper), `emotion_confidence`, `tag_source`
(`auto`/`human`), `speaker_id`, `duration`, `snr_db`, `source_video_id`,
`source_url`, `source_channel`, `license`, `segment_start`, `segment_end`,
`sample_rate`.

## How it was built
1. Curated single-speaker YouTube sources (audiobooks, lectures, news, storytelling).
2. **Sarvam batch STT** (`saaras:v3`) with diarization + timestamps for structure.
3. Silence-snapped segmentation into 3–25 s clips (single speaker only).
4. **Sarvam realtime STT** (`saarika:v2.5`) per clip for clip-accurate transcripts.
5. Automated quality gates (clipping, SNR, silence, music/noise bed, ASR confidence, dedup).
6. Hybrid emotion tagging: per-speaker-normalized acoustic features + Sarvam LLM,
   with an acoustic whisper override.
7. Human review (listen, fix transcripts, relabel) — **human labels override automated ones**.
8. Light loudness normalization (dynamics preserved), balanced emotion selection.

## Audio
24 kHz mono WAV. Loudness lightly normalized (~-20 LUFS, peak −1 dBFS) WITHOUT
limiting, so the prosodic dynamics that carry emotion are preserved.

## Ethics & licensing
Sourced from YouTube for research; clips are short and transformative. Per-clip
provenance (`source_url`, `source_channel`, `license`) is retained. Respect the
original creators' rights; remove clips on request.

## Limitations
Emotion tags are heuristic (acoustic + LLM, partly human-verified) and may be
imperfect for subtle prosody. See the project report for iteration notes.

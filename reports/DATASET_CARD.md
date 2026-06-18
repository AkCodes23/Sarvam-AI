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
  - split: test
    path: indian_english/test-*
- config_name: telugu
  data_files:
  - split: train
    path: telugu/train-*
  - split: validation
    path: telugu/validation-*
  - split: test
    path: telugu/test-*
---

# Indian English + Telugu Single-Speaker TTS Dataset (emotion-tagged)

Clean audio clips sourced from YouTube, transcribed with **Sarvam** ASR, segmented with
diarization, and labeled with emotion/style tags. Built as a data-quality / curation exercise.

> **"Single-speaker"** means **each clip contains exactly one speaker** (verified by
> diarization and speaker-embedding similarity). The dataset spans **9 distinct speakers
> total** (4 English, 5 Telugu), tracked via `speaker_id`.

## Contents
- **Indian English** (`indian_english`): 30.17 min, 160 clips, 4 speakers; emotions: {'angry': 30, 'neutral': 29, 'calm': 29, 'sad': 29, 'excited': 29, 'fearful': 8, 'happy': 6}
- **Telugu** (`telugu`): 30.08 min, 150 clips, 5 speakers; emotions: {'calm': 27, 'neutral': 27, 'angry': 27, 'excited': 27, 'sad': 27, 'fearful': 6, 'happy': 6, 'surprised': 3}

Total: **60.25 minutes**.

## Evaluation (evidence, not just claims)

- **Single-speaker check** (ECAPA-TDNN embeddings): same-speaker cosine 0.74 vs different-speaker 0.21 (separation 0.52, verification AUC 0.96 / EER 9.1%; 0/11 candidate speakers flagged).
- **Transcript reliability**: English cross-ASR agreement with Whisper = 5.5% WER / 3.4% CER (n=40), strong. Telugu cross-ASR is not a valid proxy (Whisper is weak in Telugu); Telugu transcripts are best audited by human review.
- **Emotion-tag reliability** (sarvam-30b vs sarvam-105b on 120 clips): 65% agreement, Cohen's κ 0.55.
- **Phoneme coverage**: English 39 (100%), Telugu 44 (88%).
- **Perceptual quality** (DNSMOS OVRL, published set): EN 3.09 (58% pass>3.0), TE 3.16 (81% pass>3.0). Filter `dnsmos_pass=True` for a stricter subset.
- **Transcript–audio alignment** (MMS forced-align): median confidence EN 0.954, TE 0.937.
- **Emotion-label agreement** (Krippendorff alpha): 0.4442 between the two LLM raters (0.4+ is the field norm). A 3-rater panel adding SER models drops near zero, since off-the-shelf SER clusters toward neutral and does not transfer to Telugu. Per-clip VAD (valence, arousal, dominance) is included.
- **LLM-as-judge cross-check** (independent model, 499 clips): 75% of transcripts judged clean and 81% suitable to train on. Each clip also has a topic; the set is mostly storytelling (mythology, folk tales, audiobook fiction).

See the project report (GitHub repo) for full methodology and figures.

## Schema
`audio` (24 kHz mono), `text` (raw transcript), `annotated_text` (English code-switch
spans bracketed, truncation marked with an em dash), `normalized_text`, `language`,
`emotion` (neutral, happy, sad, angry, excited, calm, fearful, surprised), `style` (narrative, conversational, formal, expressive, whisper), `emotion_confidence`, `tag_source`
(`auto`/`human`), `topic`, `speaker_id`, `gender`, `accent`, `duration`; quality scores
(`snr_db`, `dnsmos_ovrl/sig/bak`, `dnsmos_pass`, `squim_*`, `mms_align_score`,
`overlap_flag`, `llm_tts_suitable`); VAD (`valence`/`arousal`/`dominance`); annotation
flags (below); and provenance (`source_video_id/url/channel`, `license`,
`segment_start/end`, `sample_rate`).

## Annotation flags
Each clip records what is imperfect about it, so users can filter rather than trust blindly.
The two audio-quality flags are automatically inferred proxies, not verified audible-noise labels:
`quality_flag` (a quality concern: DNSMOS < 3.0, or SNR < 18 dB, or elevated energy in pauses) and
`low_quality_audio` (clearly degraded: DNSMOS < 2.8). The rest:
`has_truncation` (ends mid-utterance), `has_codemix` (preserved English in a regional clip; 0 in
practice, since Sarvam ASR transliterates English into Telugu script), `has_laughter` (audible
laughter, set by a listening pass), `emotion_low_confidence` (tag confidence < 0.55),
`transcript_review_needed` (judge-flagged or alignment < 0.85), `overlap_flag` (possible second
voice). `annotation_flags` is the pipe-joined list per clip.

## Filtering recommendations
- Studio-like subset: `dnsmos_pass == true and quality_flag == false and has_truncation == false`
- Expressive subset: `emotion_confidence > 0.7 and emotion != "neutral"`
- Storytelling subset: `topic in ('mythology', 'folktale', 'fiction')`
- Review queue: `transcript_review_needed == true or emotion_low_confidence == true`

## How it was built
1. Curated single-speaker YouTube sources (audiobooks, lectures, news, storytelling).
2. **Sarvam batch STT** (`saaras:v3`) with diarization + timestamps for structure.
3. Silence-snapped segmentation into 3–25 s clips (single speaker only).
4. **Sarvam realtime STT** (`saarika:v2.5`) per clip for clip-accurate transcripts.
5. Automated quality gates (clipping, SNR, silence, music/noise bed, ASR confidence, dedup).
6. Hybrid emotion tagging: per-speaker-normalized acoustic features + Sarvam LLM,
   with an acoustic whisper override.
7. Human review (listen, fix transcripts, relabel); **human labels override automated ones**.
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

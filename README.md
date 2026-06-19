# Indian English and Telugu speech dataset for TTS

A data-quality-first pipeline that builds a clean, single-speaker, emotion-tagged TTS dataset
(~30 min Indian English + ~30 min Telugu) from YouTube, using [Sarvam](https://docs.sarvam.ai) APIs
for ASR, diarization, and LLM tagging, and publishes it as a public HuggingFace dataset.

- **Dataset:** https://huggingface.co/datasets/AkCodes23/sarvam-tts-in-te-en
- **Report:** [`reports/report.pdf`](reports/report.pdf) — pipeline, iterations, quality analysis, decisions
- **Code:** https://github.com/AkCodes23/Sarvam-AI

The pipeline is the means; the deliverable is the data. The two stages that decide quality are source
curation (listening to and choosing clean single-speaker sources) and the human review loop. Each clip
contains a single voice, tracked by `speaker_id`; the set spans 9 speakers (4 English, 5 Telugu), with
several voices per language kept on purpose for accent and speaking-style variety.

## What's in it

| | Indian English | Telugu |
|---|---|---|
| Duration | 30.2 min | 30.1 min |
| Clips (train / val / test) | 160 (144 / 8 / 8) | 150 (134 / 8 / 8) |
| Speakers | 4 | 5 |
| Median DNSMOS (perceptual quality, 1–5) | 3.09 | 3.16 |

24 kHz mono WAV. Each row carries the raw transcript, a TTS-facing `normalized_text` (numbers and
abbreviations expanded), emotion and style tags with a confidence, speaker / gender / accent, the
quality scores (DNSMOS, SQUIM, SNR, forced-alignment), a topic, annotation flags, and full source
provenance. The dataset card has the complete schema.

## Pipeline

```
sources.yaml (curated, listen-verified)
   └─ download (yt-dlp, bestaudio + provenance)
       └─ masters: 24 kHz (dataset) + 16 kHz (ASR)
           └─ Sarvam batch STT (saaras:v3) + diarization + timestamps          [structure]
               └─ single-speaker, silence-snapped segmentation (3–25 s)
                   └─ Sarvam realtime STT (saarika:v2.5) per clip              [clip-accurate text]
                       └─ acoustic features (parselmouth + librosa), per-speaker z-score
                           └─ quality gates (clipping / SNR / silence / music-bed / confidence / dedup)
                               └─ emotion + style tagging (acoustic + sarvam-30b; whisper via acoustic rule)
                                   └─ topic + validation (sarvam-105b judges the 30b tags)
                                       └─ human review (HTML app; human edits override)
                                           └─ balance emotions + light loudness norm
                                               └─ HuggingFace dataset (public) + card
```

Design choices that matter (rationale in the report):

- **Double-pass ASR** — batch diarization for structure, realtime re-ASR for clip-accurate text.
- **Silence-snapped cuts** — boundaries land in pauses, so clips never start or end mid-word.
- **Per-speaker-relative emotion** — features z-scored against each voice's own baseline.
- **Acoustic whisper detection** — the whispered-speech style is set by an acoustic rule (voicing, HNR, energy), separately from the LLM.
- **Small model tags, large model validates** — sarvam-30b assigns the tags; sarvam-105b judges them independently.
- **Indic ASR for Telugu** — a Telugu-specialized recognizer is the transcript cross-check, since generic Whisper is unreliable on Telugu.
- **Human-in-the-loop** — a reviewer's relabel or transcript fix always wins (`tag_source=human`).
- **Light normalization** — gentle loudness, no hard limiting, so the dynamics that carry emotion survive.

## Evaluation summary

Every number below comes from a script in `scripts/`; figures and the full discussion are in
[`reports/report.pdf`](reports/report.pdf), with raw outputs under `data/manifests/`.

| Check | Result |
|---|---|
| Single-speaker (ECAPA-TDNN, 10,000 pairs) | AUC 0.96, EER 9.1%; same-speaker cosine 0.74 vs 0.21 |
| English transcripts (cross-ASR vs Whisper) | 5.5% word error; 100% language-ID match |
| Telugu transcripts (Indic recognizer vs Sarvam) | 47% word error, down from 76% with generic Whisper; alignment 0.94 + human listening pass |
| Emotion tags (sarvam-30b vs sarvam-105b) | 65% agreement, Cohen's κ 0.55; shipped as weakly supervised, with confidences |
| Phoneme coverage (g2p over transcripts) | English 100% (39/39), Telugu 88% (44/50) |
| Human listening audit (80 clips, 40 per language) | transcripts: 37/40 English, 35/40 Telugu exact; 0 clips judged unusable |
| Selection funnel | 499 clips kept after gates → 310 published (160 English, 150 Telugu); 189 balanced out |

One honest limitation found by spectral analysis: the source audio is band-limited (most energy below
~4 kHz), so although it is stored at 24 kHz the set suits standard 16–24 kHz TTS rather than full-band,
high-fidelity synthesis.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e .
# ffmpeg + ffprobe static binaries are resolved from tools/ or PATH
cp .env.example .env   # fill SARVAM_API_KEY, HF_TOKEN, HF_USERNAME / HF_DATASET_REPO
```

## Usage

```bash
ttsds smoke                 # verify Sarvam (en + te) + chat + HF auth
# 1) curate config/sources.yaml by LISTENING to candidates first
ttsds process-all           # download → diarize → segment → transcribe → tag, per source
ttsds stats                 # inspect the candidate distribution
ttsds review-build          # open data/review_app/index.html, curate, export review.csv
ttsds review-merge          # apply human decisions (human overrides win)
ttsds build                 # balance emotions + finalize audio + stats + dataset card
ttsds publish               # push public to the HuggingFace Hub
ttsds verify                # reload + sanity-check the published dataset
```

Reproduce the evaluation numbers:

```bash
python scripts/eval_speaker_eer.py     # speaker verification (AUC / EER)
python scripts/eval_asr_indic.py       # Telugu Indic ASR + English cross-ASR word error
python scripts/eval_emotion.py         # 30b vs 105b emotion agreement + confusion matrix
python scripts/eval_phoneme.py         # phoneme coverage
python scripts/analyze_dataset.py      # per-speaker, split integrity, valence-arousal, bandwidth, SNR vs DNSMOS
python scripts/human_audit.py sample   # draw the 80-clip listening-audit sheet
```

## Layout

```
config/        config.yaml (all thresholds) + sources.yaml (curated sources)
src/ttsds/     pipeline modules, one per stage (config, download, audio, segment,
               transcribe_*, features, quality, tag_emotion, normalize, review, build_dataset, publish, cli)
scripts/       eval_* (speaker, ASR, Indic ASR, emotion, phoneme, agreement),
               score_* (DNSMOS/SQUIM, alignment, overlap, SER), analyze_dataset, make_figures*, human_audit
data/          raw / master / segments / manifests / build  (gitignored except manifests)
reports/       report.md + report.pdf + figures
tests/         pure-logic unit tests (segmentation, gates, normalization, taxonomy, audio metrics)
```

## License

Code: MIT (see `LICENSE`). Dataset audio: sourced from YouTube for research use, with per-clip
provenance and license retained in the dataset. Respect the original creators.

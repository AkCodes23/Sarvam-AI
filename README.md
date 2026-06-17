# Sarvam TTS Dataset Pipeline — Indian English + Telugu

A data-quality-first pipeline that builds a **clean, single-speaker, emotion-tagged
TTS dataset** (~30 min Indian English + ~30 min Telugu) from YouTube, using
**[Sarvam](https://docs.sarvam.ai) APIs** for ASR, diarization, and LLM tagging,
and publishes it as a public **HuggingFace** dataset.

> The pipeline is a tool; the deliverable is *clean data*. The highest-leverage
> stage is **source curation** (listening to and choosing good single-speaker
> sources), followed by the **human review loop**. See `reports/` for the report.
>
> **"Single-speaker"** means **each segment contains exactly one speaker** (verified by
> diarization + speaker embeddings). The dataset spans **11 distinct speakers total**
> (5 English, 6 Telugu), tracked via `speaker_id`.

## Pipeline

```
sources.yaml (curated, listen-verified)
   └─ download (yt-dlp, bestaudio + provenance)
       └─ masters: 24 kHz (dataset) + 16 kHz (ASR)
           └─ Sarvam BATCH STT (saaras:v3) + diarization + timestamps      [structure]
               └─ single-speaker, silence-snapped segmentation (3–25 s)
                   └─ Sarvam REALTIME STT (saarika:v2.5) per clip           [accurate transcript]
                       └─ acoustic features (parselmouth + librosa), per-speaker z-score
                           └─ quality gates (clip/SNR/silence/music-bed/conf/dedup)
                               └─ emotion+style tagging (acoustic + Sarvam LLM; whisper = acoustic rule)
                                   └─ HUMAN REVIEW (HTML app; human edits override)
                                       └─ balance emotions + light loudness norm
                                           └─ HuggingFace dataset (public) + card
```

Design choices that matter (see report for rationale):
- **Double-pass ASR** — batch diarization for structure, realtime re-ASR for clip-accurate text.
- **Silence-snapped cuts** — never cut mid-word; boundaries land in pauses.
- **Per-speaker-relative emotion** — z-scored vs that voice's baseline.
- **Acoustic whisper override** — whisper is detected from voicing/HNR/energy, not LLM text-guessing.
- **Human-in-the-loop** — a reviewer's relabel/transcript fix always wins (`tag_source=human`).
- **Light normalization** — peak/gentle loudness (no −23 LUFS limiting) so emotional dynamics survive.

## Evaluation highlights (evidence, not claims)

The project proves quality rather than asserting it (full analysis + figures in `reports/report.pdf`):

| Check | Result |
|---|---|
| Single-speaker (ECAPA-TDNN) | separation 0.52, **AUC 0.96 / EER 9.1 %**, 0/11 flagged |
| English transcripts (cross-ASR vs Whisper) | **6.8 % WER / 4.5 % CER**; 100 % language-ID match |
| Emotion tags (sarvam-30b vs 105b) | 65 % agreement, **Cohen's κ 0.55** |
| Phoneme coverage | English **39/39 (100 %)**, Telugu 45/50 (90 %) |
| Rejections | 12 / 457, all low-SNR (0 music / clipping / multi-speaker) |

Reproduce with `python scripts/eval_*.py`; raw numbers in `data/manifests/eval_*.json`.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e .      # or: uv pip install -r requirements.txt (pinned)
# ffmpeg + ffprobe static binaries are auto-resolved from tools/ or PATH
cp .env.example .env   # fill SARVAM_API_KEY, HF_TOKEN, HF_USERNAME/HF_DATASET_REPO
```

## Usage

```bash
ttsds smoke                 # verify Sarvam (en+te) + chat + HF auth
# 1) curate config/sources.yaml by LISTENING to candidates first
ttsds process-all           # download → diarize → segment → transcribe → tag, per source
ttsds stats                 # inspect candidate distribution
ttsds review-build          # open data/review_app/index.html, curate, export review.csv
ttsds review-merge          # apply human decisions (human overrides win)
ttsds build                 # balance emotions + finalize audio + stats + dataset card
ttsds publish               # push public to the HuggingFace Hub
ttsds verify                # reload + sanity-check the published dataset
```

## Layout

```
config/        config.yaml (all tunables) + sources.yaml (curated sources)
src/ttsds/     pipeline modules (one per stage)
data/          raw / master / segments / manifests / build  (gitignored except manifests)
reports/       dataset card + project report + figures
tests/         pure-logic unit tests (segmentation, gates, taxonomy, audio metrics)
```

## License

Code: MIT (see `LICENSE`). Dataset audio: sourced from YouTube for research; per-clip
provenance and license are retained in the dataset. Respect original creators.

# Project Overview (review dossier)

A complete, self-contained summary of the project for review.

## 1. Task

Build a high-quality TTS training dataset of ~60 minutes total: ~30 min Indian English and ~30 min
of an Indian language (Telugu chosen). Each clip must be clean and single-speaker, sourced from
YouTube, with an accurate transcript and an emotion/style tag, published as a public HuggingFace
dataset. Use Sarvam APIs for ASR, diarization, and LLM calls. The stated grading focus is data-quality
judgment and curation, not scripting.

## 2. Deliverables (all live)

- HuggingFace dataset (public): https://huggingface.co/datasets/AkCodes23/sarvam-tts-in-te-en
- GitHub repository (public): https://github.com/AkCodes23/Sarvam-AI
- PDF report: `reports/report.pdf` in the repo (build + quality report, 6 sections).

## 3. Dataset specification

- Total 60.3 minutes: Indian English 30.2 min / 160 clips, Telugu 30.1 min / 150 clips.
- 9 speakers (4 English, 5 Telugu); each clip is single-speaker, tracked by `speaker_id`.
- Audio: 24 kHz mono WAV, lightly loudness-normalized (no hard limiting, to preserve emotional
  dynamics).
- Two HuggingFace configs: `indian_english`, `telugu`. Each has train / validation / test splits
  (English 144/8/8, Telugu 134/8/8), stratified so every split sees every speaker and emotion.
- Topic-focused: predominantly Indian storytelling (English 79% storytelling, Telugu 75%): mythology,
  folk tales, audiobook fiction. A `topic` field allows filtering.
- Emotion taxonomy (closed): neutral, happy, sad, angry, excited, calm, fearful, surprised. Style:
  narrative, conversational, formal, expressive, whisper. All eight emotions present in both
  languages; common ones capped near 30 clips, rare ones kept in full.

Per-row schema: `audio` (24 kHz), `text`, `normalized_text`, `language`, `language_code`, `emotion`,
`style`, `emotion_confidence`, `tag_source` (auto/human), `speaker_id`, `gender`, `accent`, `duration`,
`snr_db`, `dnsmos_ovrl`, `dnsmos_sig`, `dnsmos_bak`, `dnsmos_pass`, `squim_stoi`, `squim_pesq`,
`squim_sisdr`, `mms_align_score`, `overlap_flag`, `ser_emotion`, `valence`, `arousal`, `dominance`,
`topic`, `llm_tts_suitable`, `source_video_id`, `source_url`, `source_channel`, `license`,
`segment_start`, `segment_end`, `sample_rate`.

## 4. Pipeline (one source at a time)

1. Source curation (`config/sources.yaml`): hand-picked single-voice YouTube sources (audiobooks,
   storytelling, lectures, one talk); music-bed channels excluded.
2. Download (`yt-dlp`) + provenance; `ffmpeg` to 16 kHz (ASR) and 24 kHz mono master (dataset).
3. Sarvam batch ASR + diarization (`saaras:v3`): speaker-labelled, time-stamped chunks.
4. Single-speaker, silence-snapped segmentation into 3 to 25 s clips (target 5 to 15 s).
5. Sarvam realtime ASR (`saarika:v2.5`) per clip: clip-accurate transcript + word timings + language
   confidence (a deliberate double-pass for transcript accuracy).
6. Acoustic features (parselmouth, librosa), z-scored per speaker.
7. Quality gates: clipping (sustained), SNR, silence ratio, music/noise bed, ASR confidence,
   character rate, near-duplicate dedup, language mismatch, code-mix flag.
8. Hybrid emotion/style tagging (acoustic features + `sarvam-30b`), with an acoustic whisper override.
9. Topic + LLM-as-judge (one `sarvam-30b` call per clip).
10. Human review tool (static HTML; human edits override automated labels).
11. Balance (prefer storytelling, then cleanest by DNSMOS, emotion-balanced), light normalization,
    publish to HuggingFace (`datasets`, train/val/test).

## 5. Validation evidence (every claim tested)

- Single-speaker (ECAPA-TDNN): same-speaker cosine 0.74 vs different-speaker 0.21; verification
  AUC 0.96, EER 9.1% over 10,000 pairs; 0 of 9 speakers flagged.
- Transcripts, English: cross-ASR vs Whisper 6.8% WER / 4.5% CER; 100% language-ID match.
- Transcripts, Telugu: MMS forced-alignment confidence, median 0.94 (English 0.95). Whisper is weak
  in Telugu so cross-ASR is not used there. A stratified low/mid/high sample is saved for human CER.
- Emotion: two LLM raters agree at Krippendorff alpha 0.44 (human-annotator range). A 3-rater panel
  with emotion2vec + audeering SER drops near zero (SER models cluster neutral, do not transfer to
  Telugu). Independent LLM judge endorses emotion on 37% of clips. Conclusion: emotion is the soft
  dimension; shipped with confidence + label source + review tool.
- LLM-as-judge cross-check: 75% of transcripts judged clean, 81% judged suitable to train on.
- Overlap: intra-clip ECAPA cohesion, median 0.58; 9 of 325 clips flagged below 0.40 (pyannote gated).
- Perceptual quality (DNSMOS OVRL): published set 57% above 3.0 in English, 81% in Telugu; median
  3.08 / 3.16. `dnsmos_pass` flag exposes the studio-grade subset.
- Phoneme coverage: English 39/39 (100%), Telugu 44/50 (88%).
- Lexical/prosodic: type-token ratio English 0.33 / Telugu 0.59; speaking rate 149 / 117 wpm; internal
  silence 5.9%.

## 6. Key decisions and trade-offs

- Double-pass ASR (batch diarization for structure, realtime for clip-accurate transcript).
- Silence-snapped cuts; never trust coarse chunk boundaries.
- Per-speaker z-scored emotion; acoustic whisper override (the LLM cannot guess prosody).
- DNSMOS used as a gate signal: 47% below 3.0 triggered re-curation (dropped 3 perceptually-poor
  sources), not a blanket cut.
- Topic focus on storytelling. Honest trade-off: topic-first selection lowers English DNSMOS-pass from
  86% (quality-first) to 57%, because the cleanest English clips are off-topic (lecture, talk). Chose
  coherence; `dnsmos_pass` recovers the clean subset.
- Light loudness normalization, no hard limiting, to preserve emotion-carrying dynamics.
- Human edits override automated labels.

## 7. Iterations (found by inspecting output)

1. Clipping gate rejected clean hot-mastered audio: switched from peak to flat-topped-fraction (41/43
   rejected to 43/43 kept).
2. LLM copied the prompt template: removed the fill-in-the-blank, described fields instead.
3. Reasoning model truncated before answering: raised the token budget.
4. DNSMOS re-curation: dropped 3 noisy sources, added a clean lecture + narration (pool pass 53 to 63%).
5. Robustness: hardened forced-alignment, soundfile loading to dodge a librosa/speechbrain clash,
   pinned `datasets<4`, idempotent per-source recovery when Sarvam credits ran out.

## 8. Limitations

- Emotion labels are heuristic; only a human pass converts the proxy agreement into ground truth.
- Off-the-shelf SER does not transfer to Telugu; no fair Telugu third emotion rater yet.
- Clean studio-grade Indian English is scarce on YouTube; topic purity and high DNSMOS conflict on the
  English side.
- pyannote overlap detection was gated; an embedding-cohesion proxy was used instead.
- All transcript/emotion validation is automated (cross-ASR, alignment, multi-rater, LLM judge), not
  yet human-confirmed.

## 9. What I would improve given more time

Human listening pass (transcripts + emotion), a Telugu-capable SER model, word-level alignment
trimming, music separation, a cleaner Indian-English storytelling source, and language-aware text
normalization.

## 10. Repository structure

```
config/        config.yaml (all thresholds), sources.yaml (curated sources)
src/ttsds/     pipeline modules (config, download, audio, segment, transcribe_*, features,
               quality, tag_emotion, review, build_dataset, publish, cli)
scripts/       discover, eval_* (speaker EER, ASR, sources, phoneme, basic, agreement),
               score_* (audio_quality/DNSMOS+SQUIM, mms_align, overlap, ser, emotion2vec),
               llm_judge, enrich_metadata, make_figures*, make_report_pdf
tests/         unit tests incl. edge cases (segmentation, gates, normalization, whisper, dedup); 22 pass
data/manifests/ per-source segment manifests + eval_*.json / score_*.json (provenance, gitignored audio)
reports/       report.md, report.pdf, DATASET_CARD.md, figures/
```

## 11. Reproduction

```
uv venv --python 3.12 .venv
uv pip install -e .                       # or: uv pip install -r requirements.txt
cp .env.example .env                       # SARVAM_API_KEY, HF_TOKEN, HF_USERNAME
ttsds smoke                                # verify Sarvam (en+te) + chat + HF auth
ttsds process-all                          # download, diarize, segment, transcribe, tag (per source)
python scripts/score_audio_quality.py && python scripts/score_mms_align.py \
  && python scripts/score_overlap.py && python scripts/score_ser.py && python scripts/llm_judge.py
python scripts/enrich_metadata.py
ttsds build && ttsds publish && ttsds verify
```

## 12. Tech and APIs

Python 3.12 (uv). Sarvam: `saaras:v3` (batch ASR+diarization), `saarika:v2.5` (realtime ASR),
`sarvam-30b` (emotion tagging + topic + judge). Validation: speechbrain ECAPA-TDNN, torchaudio MMS
forced alignment + SQUIM, DNSMOS, faster-whisper (English cross-ASR), emotion2vec + audeering wav2vec
(SER), epitran + g2p_en (phonemes), krippendorff. HuggingFace `datasets` (<4) + `huggingface_hub`.

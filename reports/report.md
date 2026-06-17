# Building a Single-Speaker, Emotion-Tagged TTS Dataset
### Indian English + Telugu, sourced from YouTube, with Sarvam APIs

---

## 1. What I built and how the pipeline works

The goal was a **high-quality, single-speaker TTS dataset** — ~30 minutes of Indian
English and ~30 minutes of Telugu — with accurate transcriptions and per-segment
emotion/style tags, published publicly on HuggingFace. The grading emphasis is on
*data-quality judgment and curation*, not scripting, so I treated the code as a tool
in service of curation, and spent the real effort on **source selection** and on a
**listen-and-iterate** loop.

The pipeline is a small, modular Python package (`ttsds`) driven by a `typer` CLI.
It processes **one source at a time** so quality can be validated (and API credits
protected) before scaling. Stages:

1. **Source curation** (`config/sources.yaml`) — the highest-leverage step. Candidate
   YouTube sources are chosen by archetype (audiobooks, solo lectures, AIR talks,
   storytelling, TEDx) and channel reputation, deliberately avoiding compilation
   channels that overlay music. A discovery aid (`scripts/discover.py`) searches
   YouTube metadata to surface candidates; every one is still validated downstream
   and by ear.
2. **Download + normalize** — `yt-dlp` pulls best-quality audio plus provenance
   (channel, upload date, license, video id). `ffmpeg` produces two derivatives: a
   **16 kHz mono** copy for ASR and a **24 kHz mono master** for the published audio.
3. **Batch ASR + diarization** — Sarvam **`saaras:v3`** batch STT with
   `with_diarization` + `with_timestamps` gives speaker-labelled, time-stamped chunks.
   This provides *structure*: who spoke when.
4. **Single-speaker, silence-snapped segmentation** — the dominant speaker's chunks
   are merged into runs (a different speaker breaks a run, so clips never span a
   speaker change). Runs are split into **3–25 s** clips (target 5–15 s) with cuts
   **snapped to silences** via local energy detection, never mid-word.
5. **Per-segment realtime transcription** — each finished clip is re-transcribed with
   Sarvam **`saarika:v2.5`**. The batch transcript is coarse; this second pass yields
   a transcript exactly aligned to the published clip, plus word timing (→ speaking
   rate) and a language-confidence signal. This **double-pass ASR** is a deliberate
   quality investment.
6. **Acoustic features** — `parselmouth` (Praat) + `librosa` extract pitch (F0
   mean/range/variation), energy/RMS dynamics, harmonics-to-noise ratio (HNR),
   high/low spectral band ratio (breathiness), voicing fraction, and pauses. Features
   are **z-scored per speaker**, so "excited" means *elevated for that voice*, not an
   absolute threshold.
7. **Quality gates** — each clip gets metrics + a status (pass / flag / reject) with
   explicit reasons: sustained clipping, low SNR, excessive silence, a music/noise
   bed (gap-energy ratio), low ASR confidence, implausible character rate, and
   near-duplicate transcripts. Flags (e.g. music bed, low tag confidence) are kept but
   surfaced for human review; rejects are dropped.
8. **Emotion + style tagging** — a hybrid of acoustics and an LLM. The per-speaker
   acoustic profile is rendered into plain language and sent, with the transcript, to
   Sarvam **`sarvam-30b`** constrained to a closed taxonomy. **Whisper is decided by an
   acoustic rule** (low voicing + low HNR + low energy), not LLM text-guessing, and the
   LLM is instructed to ground emotion in the acoustics (flat prosody → neutral even if
   the words are emotional).
9. **Human-in-the-loop review** (`ttsds review-build`) — a self-contained static HTML
   app lists every candidate with an audio player, editable transcript, emotion/style
   dropdowns, and all metrics. The reviewer accepts/rejects/relabels; **a human edit
   always overrides the automated tag** (`tag_source` flips to `human`).
10. **Balance + finalize** — accepted clips are selected to hit ~30 min/language while
    **balancing the emotion histogram** (round-robin across emotion buckets so rare
    emotions are fully included and neutral is capped). Audio gets **light loudness
    normalization** (≈ −20 LUFS, peak −1 dBFS) *without* limiting, so the prosodic
    dynamics that carry emotion are preserved.
11. **Publish + verify** — the dataset is built with HuggingFace `datasets` (two configs,
    `indian_english` + `telugu`, each with train/validation), pushed **public**, and
    reloaded with `load_dataset` to confirm audio decodes and the schema is intact.

### Key design decisions (the judgment that matters)
- **Double-pass ASR** — diarization for structure, realtime re-ASR for clip-accurate text.
- **Silence-snapped cuts** — never trust coarse chunk boundaries; cut in pauses.
- **Per-speaker-relative emotion** — z-scores, not absolute prosody thresholds.
- **Acoustically-grounded emotion + acoustic whisper override** — the LLM can't text-guess prosody.
- **Light normalization** — aggressive −23 LUFS limiting would flatten the very dynamics being tagged.
- **Human override is final** — the dataset's labels are only as good as what a person confirms.
- **Incremental, per-source validation** — catch problems on one source before spending credits on twelve.

---

## 2. Iterations to improve data quality

The pipeline only looked finished after running it on real audio and *looking at the
output*. Three concrete iterations, each found by inspecting iteration-0 results:

**Iteration A — the clipping gate was rejecting clean audio.**
On the first Telugu audiobook, **41 of 43 candidate segments were rejected**, all for
"clipping". Inspection showed the audio was actually pristine (SNR 24–30 dB, no music
bed, clean transcripts) — but YouTube audio is mastered hot, so peaks *touch* full
scale. My gate rejected any peak > 0.99. Measuring the actual **clipped-sample fraction**
showed ≤0.03 % of samples at full scale — nowhere near audible clipping (real
flat-topping is >1 %). Fix: gate on *sustained* clipped fraction (>1 %), not peak.
Result: 43/43 kept.

**Iteration B — the LLM was parroting my prompt template.**
With clipping fixed, every clip came back `neutral`/`narrative`. The cause: 34/43
responses literally copied my JSON example — `"rationale": "<=20 words"`,
`confidence: 0.0` — instead of classifying. A copyable placeholder template plus
low reasoning effort made the model echo the example. Fix: removed the literal template,
described the fields instead, and explicitly told the model to pick real values.

**Iteration C — the reasoning model was truncating before it answered.**
`sarvam-30b`/`105b` are reasoning models that emit ~1.6–2.3k tokens of reasoning
*before* the answer. At `max_tokens=1500`, **7/8 responses truncated** (`finish_reason:
length`) with empty content, falling back to a default tag. Raising the ceiling to 4000
(billed on actual tokens, so the ceiling is free insurance) gave **8/8 valid, varied,
confident tags**. After this, the audiobook produced a rich emotion mix (sad, excited,
angry, calm, happy, neutral, fearful) at median confidence 0.85 with zero fallbacks.

Other refinements made along the way: a **gap-energy** music-bed detector (residual
energy in inter-phrase pauses) instead of an ambiguous spectral-flatness threshold;
**parallelized** the per-segment ASR and LLM calls (the reasoning model is slow
sequentially); pinned a **Python 3.12** environment because key audio libraries lack
3.14 wheels.

Three operational issues surfaced near the finish and were worked through:
- **Credit exhaustion mid-run.** The Sarvam quota ran out while processing Telugu (4
  sources failed with `insufficient_quota_error`). Because the pipeline is per-source and
  idempotent, topping up the key and re-running `process-all --lang te` cleanly recovered
  exactly the missing sources — no reprocessing of finished ones.
- **`datasets` 4/5 requires `torchcodec`** to encode the `Audio` feature, which is painful
  on Windows. Pinned `datasets<4` (3.x encodes via `soundfile`), verified before publishing.
- **Split-config bug in the dataset card.** The first push wrote all clips into a single
  `train` split because the card's `data_files` glob (`config/**`) matched both
  `train-*` and `validation-*` parquets. Fixed by mapping splits explicitly in the card
  YAML and re-uploading — `train`/`validation` then resolved correctly.

---

## 3. What worked and what didn't

**Worked well**
- Sarvam batch diarization gave clean single-speaker structure on solo audiobooks/lectures.
- Audiobook narration was the best source type: clean, single voice, wide emotional range.
- The double-pass ASR produced clip-accurate transcripts; Telugu transcription quality was strong.
- Per-speaker z-scoring made emotion tags coherent within a voice.
- The objective gates (clipping fraction, SNR, gap-energy) were a reliable first filter; the
  HTML review app made human verification fast.

**Didn't work / needed care**
- Naïve thresholds (peak-based clipping) over-rejected clean audio — only visible by inspecting data.
- Reasoning-model quirks (template parroting, token truncation) silently degraded tags until inspected.
- Compilation/"motivational" channels almost always carry a music bed — excluded at the source step.
- TEDx/discourse sources can include applause/audience; the gates flag these for review.

---

## 4. Results and quality observations

**Published dataset:** https://huggingface.co/datasets/AkCodes23/sarvam-tts-in-te-en
(two configs, each with `train`/`validation`).

| | Indian English | Telugu |
|---|---|---|
| Minutes | 30.05 | 30.25 |
| Clips (train/val) | 142 (134/8) | 140 (133/7) |
| Distinct speakers | 5 | 6 |
| Emotion mix | neutral 31, calm 31, sad 30, excited 30, angry 12, fearful 5, happy 3 | calm 25, neutral 24, angry 24, excited 24, sad 24, fearful 9, happy 7, surprised 3 |

**Total: 60.3 minutes.** Sources span audiobooks, solo lectures, All India Radio talks,
storytelling, a TEDx talk, and discourse — 11 sources, all single-speaker.

**Funnel:** 457 candidate segments → **445 kept** → balanced down to **282** selected.
Only **12 were rejected, all for low SNR** (from one noisier AIR source); clipping and
music-bed rejections were zero after the gate fixes. Per-source medians: SNR 19–45 dB,
inter-pause gap-energy 0.00–0.03 (no music beds anywhere — even the kathalu/motivation
sources I'd flagged as risky came back clean), emotion-tag confidence 0.85. The
round-robin balancer capped dominant emotions (Telugu sad 73→24, excited 65→24) and
included rare ones fully (surprised 3, happy 7).
- **Emotion validity** is the hardest dimension. Grounding the LLM in per-speaker
  acoustics (not text alone) and overriding whisper acoustically materially improved it,
  but subtle affect remains imperfect — hence the human review pass and the
  `emotion_confidence` + `tag_source` fields shipped with every row for transparency.
- **Licensing/ethics**: clips are short and transformative, used for research; full
  per-clip provenance (`source_url`, `source_channel`, `license`) is retained. Permissive
  / government / educational sources (AIR, NPTEL) were preferred.

---

## 5. What I'd improve with more time

- **Forced-alignment** (e.g. WhisperX/MFA) for word-level boundaries to trim clips even
  more tightly and to flag transcript/audio mismatches automatically.
- **A trained SER model** (speech-emotion-recognition) as a second opinion alongside the
  acoustic+LLM tagger, with agreement used to auto-confirm and disagreement routed to review.
- **Background-music separation** (e.g. Demucs) to rescue otherwise-good clips that carry
  a light bed, instead of rejecting them.
- **Speaker-embedding verification** (resemblyzer) to guarantee one voice per `speaker_id`
  across a whole source, beyond diarization.
- **Text normalization** for TTS (number/abbreviation expansion), language-aware, for the
  `normalized_text` field.
- **A larger, more balanced source pool** per emotion, and a second human reviewer for
  inter-rater agreement on the emotion labels.

---

---

**Links**
- Dataset: https://huggingface.co/datasets/AkCodes23/sarvam-tts-in-te-en
- Code: https://github.com/AkCodes23/Sarvam-AI

*Appendix: figures are generated by `scripts/make_figures.py`; all thresholds live in
`config/config.yaml`; the dataset card is in `reports/DATASET_CARD.md`.*

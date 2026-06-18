# Indian English + Telugu TTS Dataset: Build and Quality Report

A 60.3-minute, single-speaker, emotion-tagged TTS dataset (30.2 min Indian English, 30.1 min
Telugu), sourced from YouTube, transcribed and tagged with Sarvam APIs, published on HuggingFace.

- Dataset: https://huggingface.co/datasets/AkCodes23/sarvam-tts-in-te-en
- Code: https://github.com/AkCodes23/Sarvam-AI

"Single-speaker" means each clip contains exactly one voice. The dataset spans 9 speakers (4 English,
5 Telugu), tracked by `speaker_id`. The curated sources are storytelling-heavy, so the set is
predominantly Indian narrative speech (mythology, folk tales, audiobook fiction).

---

## 1. What I built and how the pipeline works

A modular Python package (`ttsds`) driven by a CLI, run one source at a time so quality is validated
and API credits are protected before scaling. The stages:

1. **Source curation** (`config/sources.yaml`). Hand-picked YouTube sources chosen to be single-voice
   by nature: solo audiobooks, storytelling, single-narrator lectures, one stage talk. Compilation
   channels with music beds were excluded at this step. This is the highest-leverage stage.
2. **Download and normalize.** `yt-dlp` pulls best-quality audio plus provenance (channel, date,
   license, video id). `ffmpeg` produces a 16 kHz copy for ASR and a 24 kHz mono master for the
   published audio.
3. **Batch ASR with diarization** (Sarvam `saaras:v3`). Produces speaker-labelled, time-stamped
   chunks. This gives the structure needed to keep only single-speaker stretches.
4. **Segmentation.** Keep the target speaker's chunks, merge into runs (a different speaker breaks a
   run), then split into 3 to 25 second clips (target 5 to 15). Cut points are snapped to silence
   using local energy detection, so clips never start or end mid-word.
5. **Per-segment ASR** (Sarvam `saarika:v2.5`). Each finished clip is re-transcribed. The batch
   transcript is coarse and chunk-aligned, so this second pass gives a transcript that matches the
   exact clip, plus word timings used for trimming and a language-confidence signal. The double pass
   is a deliberate accuracy investment.
6. **Acoustic features** (parselmouth, librosa): pitch, energy dynamics, HNR, spectral tilt, voicing
   fraction, speaking rate, pauses. Features are z-scored per speaker so "excited" means elevated for
   that voice, not an absolute threshold.
7. **Quality gates.** Per clip, with explicit reasons: sustained clipping, low SNR, excessive
   silence, music or noise bed (inter-pause energy ratio), low ASR confidence, implausible character
   rate, near-duplicate transcript, wrong detected language, and a code-mix flag for a Telugu clip
   that is majority English. Rejections are dropped; flags are kept and surfaced for review.
8. **Emotion and style tagging.** A hybrid of acoustics and an LLM (Sarvam `sarvam-30b`). The
   per-speaker acoustic profile is rendered to text and sent with the transcript under a closed
   taxonomy. Whisper is set by an acoustic rule, not LLM guessing. Each tag carries a confidence and
   a source (`auto` or `human`).
9. **Topic and LLM-as-judge.** One Sarvam call per clip assigns a topic and independently scores the
   transcript and emotion (details in section 4).
10. **Human review.** A static HTML app lists every candidate with audio, transcript, tag, and
    metrics for accept, reject, relabel, or transcript fix. A human edit always overrides the
    automated label.
11. **Balance, finalize, publish.** Selection prefers storytelling topics, then the cleanest clips by
    DNSMOS, while balancing the emotion histogram to 30 minutes per language. Audio gets light
    loudness normalization (no hard limiting, so emotional dynamics survive). The dataset is built
    with HuggingFace `datasets` (two configs, train/validation/test) and pushed public.

Final dataset:

| | Indian English | Telugu |
|---|---|---|
| Minutes | 30.2 | 30.1 |
| Clips (train / val / test) | 160 (144 / 8 / 8) | 150 (134 / 8 / 8) |
| Speakers | 4 | 5 |
| Storytelling clips | 79% | 75% |
| Median DNSMOS | 3.08 | 3.16 |
| Clips above DNSMOS 3.0 | 57% | 81% |
| Median alignment confidence | 0.95 | 0.94 |

Both languages carry all eight emotion labels, with common ones (neutral, calm, sad, excited, angry)
capped near 30 clips so none dominates and rare ones (happy, fearful, surprised) kept in full. Every
row also has a normalized transcript, language, style, gender and accent, the quality scores (DNSMOS,
SQUIM, SNR, alignment, intra-clip cohesion), topic, the judge's suitability score, full source
provenance, and timestamps.

## 2. Iterations to improve data quality

Each of these was found by looking at the output, not the code.

1. **Clipping gate rejected clean audio.** The first Telugu audiobook lost 41 of 43 clips to a
   "clipping" gate, but the audio was clean. The clips peaked at exactly 1.0 because YouTube masters
   loud. Real clipping is a run of flat-topped samples, not one sample at full scale. I changed the
   gate to measure the fraction of flat-topped samples. Result: 43 of 43 kept.
2. **The LLM copied my prompt template.** Every clip came back "neutral, narrative" with the rationale
   field reading "<=20 words", which was my instruction text. The model was filling in a
   fill-in-the-blank. I removed the template and described the fields instead, and the labels spread
   out and matched what I could hear.
3. **The reasoning model truncated before answering.** Labels were varied but half had low confidence.
   `sarvam-30b` reasons before answering, and at a 1500-token budget it spent all of it reasoning and
   never wrote the JSON. I raised the budget (billed on tokens used, not the ceiling, so it costs
   nothing) and the truncation stopped. Median tag confidence then reached 0.85.
4. **DNSMOS re-curation.** Perceptual quality (DNSMOS) flagged 47% of clips below 3.0, over my
   pre-set one-third threshold. Per-source analysis showed three sources dragging the set down: an
   archival AIR recording (2.31), a hall discourse (2.27), and an English audiobook (2.42) whose
   compression DNSMOS heard even though its SNR looked clean. I dropped all three and added a clean
   lecture and more narration. Pool pass rate rose from 53 to 63 percent.
5. **Topic focus.** The sources were mostly storytelling, so I made that the dataset's theme rather
   than a random mix, using LLM topic tags to prefer narrative clips in selection.
6. **Robustness fixes during scaling.** Hardened forced-alignment against token-longer-than-audio
   clips, switched audio loading to soundfile where librosa clashed with speechbrain, pinned
   `datasets<4` to avoid a torchcodec dependency on Windows, and recovered cleanly when Sarvam credits
   ran out mid-run (per-source idempotent processing meant only the missing sources re-ran).

## 3. What worked and what did not

Worked:

- Audiobook and storytelling sources gave the best mix of clean audio and emotional range.
- The double-pass ASR produced clip-accurate transcripts. English cross-checks at 6.8% word error.
- DNSMOS exposed bad sources that SNR alone missed (the compressed audiobook), and per-source analysis
  made re-curation precise rather than a blanket cut.
- ECAPA speaker verification confirmed single-speaker integrity objectively (AUC 0.96).
- MMS forced alignment gave a transcript check that works in Telugu, where an English-trained second
  recognizer does not.

Did not work or needed care:

- Automatic emotion labeling is soft. Off-the-shelf SER models cluster toward neutral and do not
  transfer to Telugu, and even an LLM judge endorses only 37 percent of the labels. Emotion stays the
  dimension a human should review.
- pyannote overlap detection is behind a license that cannot be accepted from a script, so I used an
  intra-clip embedding-cohesion check instead.
- Whisper is weak in Telugu, so cross-ASR is not a valid transcript check there.
- Clean studio-grade Indian English is scarce on YouTube. The cleanest English clips are off-topic
  (a lecture, a talk), which forced a trade-off described in section 4.
- Sarvam credits ran out twice mid-build and had to be topped up.

## 4. Quality observations and decisions

I tested every claim rather than asserting it.

**Single-speaker.** Embedded every clip with ECAPA-TDNN. Same-speaker cosine similarity is 0.74,
different-speaker 0.21. As a verification task over 10,000 clip pairs that is AUC 0.96 and an
equal-error rate of 9 percent. No speaker's clips leak into another's.

![Speaker verification: same vs different similarity with the equal-error threshold](figures/speaker_verification.png)

**Transcripts.** For English, an unrelated recognizer (Whisper) agrees to 6.8 percent word error and
4.5 percent character error, and the realtime recognizer identified the correct language on 100
percent of clips. For Telugu, where Whisper is unreliable, MMS forced alignment gives a median
confidence of 0.94 (English 0.95), which says the words are present where the transcript claims. A
stratified low, middle, and high sample is saved for a human CER audit.

![MMS forced-alignment confidence, per language](figures/mms_align_dist.png)

**Emotion.** Two Sarvam models labeling the same clips agree at Krippendorff alpha 0.44, which is the
range human annotators reach. Adding two speech-emotion models (emotion2vec, audeering) as outside
raters drops the three-way alpha near zero, because both SER models call most clips neutral and
neither handles Telugu. An independent LLM judge endorsed the emotion on only 37 percent of clips.
The consistent message across three methods is that text and acoustics underdetermine emotion, so I
ship a confidence and a label source on every clip and mark emotion for human review.

![Emotion-label agreement across raters](figures/agreement_bars.png)

**Overlap.** pyannote is gated, so I embedded short windows inside each clip and checked they all
sound like the same person. Median cohesion is 0.58 (the single-speaker level) and only 9 of 325
clips fall below 0.40, flagged for a listen.

**Perceptual quality and the topic trade-off.** DNSMOS scores how clean audio sounds. Selecting for
the storytelling topic, the published set is 57 percent above 3.0 in English and 81 percent in
Telugu. English is lower because its cleanest clips are a lecture and a talk, which are off-topic, so
preferring storytelling pulls in narration at DNSMOS 2.85 to 3.19. I chose topic coherence over the
last few points of polish, because the set is more useful as a focused storytelling corpus, and a
`dnsmos_pass` column recovers the cleanest subset in one filter. This is clean speech at 23 to 35 dB
SNR that the judge rated 0.90 suitable, not a noise problem.

![DNSMOS distribution with the 3.0 line](figures/dnsmos_dist.png)

**Per-source quality.** Emotion entropy measures how varied a source's emotions are (higher is more
varied).

| Source | Type | Clips | Min | Median SNR | Median DNSMOS | Emotion entropy |
|---|---|---|---|---|---|---|
| en_nptel | lecture | 53 | 6.9 | 35.7 | 3.23 | 2.36 |
| en_mahabharata | story | 50 | 8.5 | 35.4 | 3.19 | 2.60 |
| en_air_talk | talk | 44 | 9.9 | 28.6 | 3.02 | 2.24 |
| en_tedx_amina | talk | 46 | 9.2 | 43.9 | 2.97 | 2.03 |
| en_mahabharata_e69 | story | 40 | 8.1 | 24.0 | 2.88 | 2.52 |
| en_mahabharata_e67 | story | 40 | 8.1 | 22.9 | 2.85 | 2.39 |
| te_ramaaraavi | story | 47 | 8.6 | 44.9 | 3.23 | 2.34 |
| te_kalalavelugu | audiobook | 50 | 9.4 | 30.6 | 3.20 | 2.60 |
| te_kathalu_epic | story | 39 | 8.8 | 39.9 | 3.16 | 2.52 |
| te_motivation_kasyap | talk | 47 | 7.9 | 38.6 | 3.01 | 2.31 |
| te_bhumiputri | audiobook | 43 | 10.0 | 27.5 | 2.95 | 2.53 |

![Minutes contributed per source](figures/source_contribution.png)

**Rejection analysis.** At the per-clip gate stage, 12 of 457 candidates were rejected, all for low
SNR, all from one archival source. There were zero clipping, music-bed, or multi-speaker rejections,
because the bad sources were filtered upstream at curation and at the DNSMOS step. A low rejection
rate here means the bad data was never let in, not that nothing was checked.

**Other decisions.** Light loudness normalization instead of a hard loudness target, so the intensity
dynamics that carry emotion survive. Human edits always override automated labels. Gender is inferred
from median F0 with known speakers corrected by hand. The full 60 minutes is kept rather than applying
a hard DNSMOS cut, with `dnsmos_pass` exposing the studio-grade subset.

## 5. What I would improve given more time

- A human listening pass. Every check above is automatic. The real next step is one annotator going
  through the alignment-sorted transcript sample and a second labeling emotion, which turns the proxy
  numbers into ground truth. The review tool is built for this.
- A speech-emotion model that handles Telugu, so the third emotion rater is fair.
- Word-level forced-alignment trimming to tighten clip edges further.
- Background-music separation to rescue otherwise-good clips that carry a light bed.
- A cleaner Indian-English storytelling source, the one ingredient that stayed scarce, to get topic
  coherence and high DNSMOS at the same time.
- Language-aware text normalization (numbers, abbreviations) for the normalized-text field.

## 6. Cross-check against the brief

The brief asked for about 60 minutes split across Indian English and one Indian language, as clean
single-speaker YouTube clips with accurate transcripts and an emotion tag each, published on
HuggingFace and built with Sarvam. What shipped: 60.3 minutes (30.2 English, 30.1 Telugu); every clip
is one speaker, verified at AUC 0.96; transcripts come from Sarvam and hold at 6.8 percent cross-ASR
error in English and 0.94 alignment in Telugu; every clip has an emotion and style tag with a
confidence; the dataset is public; and the ASR, diarization, emotion, and judge calls all run on
Sarvam. Reproduction steps are in the repository README.

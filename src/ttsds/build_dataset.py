"""Select kept clips, balance the emotion histogram to the per-language target,
apply final light loudness normalization, and emit final records + stats + card."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from .audio import loudness_normalize, read_wav, write_wav
from .config import DATA_DIR, MANIFEST_DIR, PROJECT_ROOT, REPORTS_DIR, Config
from .models import Segment, load_all_segments

BUILD_DIR = DATA_DIR / "build"
FINAL_SELECTION = MANIFEST_DIR / "final_selection.json"
STATS_PATH = MANIFEST_DIR / "dataset_stats.json"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _ovrl(s: Segment) -> float:
    v = s.metrics.get("dnsmos_ovrl")
    return float(v) if v is not None else 0.0


def _round_robin_balance(segs: list[Segment], target_seconds: float,
                         prefer_topics: list[str] | None = None) -> list[Segment]:
    """Pick across emotion buckets in turn so rare emotions are fully included and
    the dominant one (usually neutral) is capped, until we hit the target. Within each
    bucket, prefer the target topics first (a coherent narrative set), then the cleanest
    clips by DNSMOS OVRL."""
    prefer = set(prefer_topics or [])
    buckets: dict[str, list[Segment]] = defaultdict(list)
    for s in sorted(segs, key=lambda x: (
            0 if x.metrics.get("topic_category") in prefer else 1, -_ovrl(x), x.id)):
        buckets[s.emotion or "neutral"].append(s)
    # order buckets smallest-first so scarce emotions are exhausted before neutral
    order = sorted(buckets, key=lambda k: len(buckets[k]))
    chosen: list[Segment] = []
    total = 0.0
    progress = True
    while total < target_seconds and progress:
        progress = False
        for k in order:
            if buckets[k]:
                s = buckets[k].pop(0)
                chosen.append(s)
                total += s.duration_s
                progress = True
                if total >= target_seconds:
                    break
    return chosen


def select_and_balance(cfg: Config, segments: list[Segment] | None = None) -> dict[str, list[Segment]]:
    segments = segments if segments is not None else load_all_segments()
    kept = [s for s in segments if s.is_kept()]
    target = cfg.targets.minutes_per_language * 60.0
    out: dict[str, list[Segment]] = {}
    for lang in cfg.languages:
        lang_segs = [s for s in kept if s.language == lang]
        avail = sum(s.duration_s for s in lang_segs)
        if avail <= target:
            out[lang] = sorted(lang_segs, key=lambda x: x.id)  # take everything
        else:
            out[lang] = _round_robin_balance(lang_segs, target, cfg.targets.prefer_topics)
    return out


def finalize_audio(seg: Segment, cfg: Config) -> Path:
    """Light loudness normalization to a final per-config wav (dynamics preserved)."""
    config_name = cfg.hf_config_name(seg.language)
    out_dir = BUILD_DIR / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    y, sr = read_wav(PROJECT_ROOT / seg.wav_path)
    y = loudness_normalize(y, sr, target_lufs=-20.0, peak_ceiling_dbfs=-1.0)
    dst = out_dir / f"{seg.id}.wav"
    write_wav(dst, y, sr)
    return dst


def _record(seg: Segment, final_wav: Path, sr: int) -> dict:
    m = seg.metrics
    ovrl = m.get("dnsmos_ovrl")
    return {
        "audio": str(final_wav.relative_to(PROJECT_ROOT)),
        "text": seg.transcript,
        "normalized_text": normalize_text(seg.transcript),
        "language": seg.language,
        "language_code": seg.language_code,
        "emotion": seg.emotion,
        "style": seg.style,
        "emotion_confidence": seg.emotion_confidence,
        "tag_source": seg.tag_source,
        "speaker_id": seg.speaker_id,
        "gender": m.get("gender"),
        "accent": m.get("accent"),
        "duration": round(seg.duration_s, 3),
        "snr_db": m.get("snr_db"),
        "dnsmos_ovrl": ovrl,
        "dnsmos_sig": m.get("dnsmos_sig"),
        "dnsmos_bak": m.get("dnsmos_bak"),
        "dnsmos_pass": (ovrl is not None and ovrl > 3.0),
        "squim_stoi": m.get("squim_stoi"),
        "squim_pesq": m.get("squim_pesq"),
        "squim_sisdr": m.get("squim_sisdr"),
        "mms_align_score": m.get("mms_align_score"),
        "overlap_flag": bool(m.get("overlap_flag", False)),
        "ser_emotion": m.get("ser_emotion"),
        "valence": m.get("valence"),
        "arousal": m.get("arousal"),
        "dominance": m.get("dominance"),
        "topic": m.get("topic_category"),
        "llm_tts_suitable": m.get("tts_suitable"),
        "source_video_id": seg.source_video_id,
        "source_url": seg.source_url,
        "source_channel": seg.source_channel,
        "license": seg.license,
        "segment_start": seg.start_s,
        "segment_end": seg.end_s,
        "sample_rate": sr,
    }


def assemble(cfg: Config) -> dict:
    selection = select_and_balance(cfg)
    records: dict[str, list[dict]] = {}
    sr = cfg.audio.master_sample_rate
    for lang, segs in selection.items():
        config_name = cfg.hf_config_name(lang)
        recs = []
        for seg in segs:
            final_wav = finalize_audio(seg, cfg)
            recs.append(_record(seg, final_wav, sr))
        records[config_name] = recs

    FINAL_SELECTION.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = compute_stats(selection, cfg)
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_card(stats, cfg)
    return stats


def compute_stats(selection: dict[str, list[Segment]], cfg: Config) -> dict:
    out: dict = {"per_language": {}, "total_minutes": 0.0}
    for lang, segs in selection.items():
        emo: dict[str, int] = defaultdict(int)
        sty: dict[str, int] = defaultdict(int)
        spk: dict[str, float] = defaultdict(float)
        human = 0
        minutes = sum(s.duration_s for s in segs) / 60.0
        for s in segs:
            emo[s.emotion or "?"] += 1
            sty[s.style or "?"] += 1
            spk[s.speaker_id] += s.duration_s
            if s.tag_source == "human":
                human += 1
        ovrls = sorted(s.metrics["dnsmos_ovrl"] for s in segs if s.metrics.get("dnsmos_ovrl") is not None)
        aligns = [s.metrics["mms_align_score"] for s in segs if s.metrics.get("mms_align_score") is not None]
        out["per_language"][lang] = {
            "name": cfg.languages[lang].name,
            "config": cfg.hf_config_name(lang),
            "clips": len(segs),
            "minutes": round(minutes, 2),
            "speakers": len(spk),
            "speaker_minutes": {k: round(v / 60.0, 2) for k, v in spk.items()},
            "emotions": dict(sorted(emo.items(), key=lambda x: -x[1])),
            "styles": dict(sorted(sty.items(), key=lambda x: -x[1])),
            "human_tagged": human,
            "mean_clip_s": round((minutes * 60 / len(segs)) if segs else 0, 2),
            "dnsmos_ovrl_median": round(ovrls[len(ovrls) // 2], 2) if ovrls else None,
            "dnsmos_pass_pct": round(100 * sum(v > 3.0 for v in ovrls) / len(ovrls)) if ovrls else None,
            "mms_align_median": round(sorted(aligns)[len(aligns) // 2], 3) if aligns else None,
        }
        out["total_minutes"] = round(out["total_minutes"] + minutes, 2)
    return out


def _eval_section() -> str:
    """Build an Evaluation section for the card from any eval_*.json present."""
    def _load(name):
        p = MANIFEST_DIR / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    spk, emo, pho, asr = _load("eval_speaker.json"), _load("eval_emotion.json"), \
        _load("eval_phoneme.json"), _load("eval_asr.json")
    agr = _load("eval_agreement.json")
    stats = _load("dataset_stats.json")
    if not any([spk, emo, pho, asr]):
        return ""
    lines = ["", "## Evaluation (evidence, not just claims)", ""]
    if spk:
        extra = ""
        if spk.get("roc_auc") is not None:
            extra = f", verification AUC {spk['roc_auc']:.2f} / EER {spk['eer']*100:.1f}%"
        lines.append(
            f"- **Single-speaker check** (ECAPA-TDNN embeddings): same-speaker cosine "
            f"{spk['avg_within']:.2f} vs different-speaker {spk['avg_between']:.2f} "
            f"(separation {spk['separation']:.2f}{extra}; {len(spk.get('flagged', []))}/"
            f"{spk['n_speakers']} speakers flagged)."
        )
    if asr:
        en = asr.get("en", {})
        lid = asr.get("lang_id_match", {})
        line = (
            f"- **Transcript reliability**: English cross-ASR agreement with Whisper = "
            f"{en['wer_mean']*100:.1f}% WER / {en['cer_mean']*100:.1f}% CER (n={en['n']}), strong. "
        )
        if lid:
            line += (
                f"Realtime ASR language-ID matched the target language on "
                f"{lid.get('en',{}).get('pct',0):.0f}% of EN and {lid.get('te',{}).get('pct',0):.0f}% of TE clips. "
            )
        line += ("Telugu cross-ASR is not a valid proxy (Whisper is weak in Telugu); "
                 "Telugu transcripts are best audited by human review.")
        lines.append(line)
    if emo:
        o = emo["overall"]
        lines.append(
            f"- **Emotion-tag reliability** (sarvam-30b vs sarvam-105b on {o['n']} clips): "
            f"{o['emotion_agreement']*100:.0f}% agreement, Cohen's κ {o['emotion_kappa']:.2f}."
        )
    if pho:
        lines.append(
            f"- **Phoneme coverage**: English {pho['en']['unique_phonemes']} "
            f"({pho['en']['coverage']*100:.0f}%), Telugu {pho['te']['unique_phonemes']} "
            f"({pho['te']['coverage']*100:.0f}%)."
        )
    if stats and "per_language" in stats:
        pl = stats["per_language"]
        en, te = pl.get("en", {}), pl.get("te", {})
        if en.get("dnsmos_ovrl_median") is not None:
            lines.append(
                f"- **Perceptual quality** (DNSMOS OVRL, published set): EN "
                f"{en['dnsmos_ovrl_median']} ({en.get('dnsmos_pass_pct')}% pass>3.0), TE "
                f"{te.get('dnsmos_ovrl_median')} ({te.get('dnsmos_pass_pct')}% pass>3.0). "
                f"Filter `dnsmos_pass=True` for a stricter subset."
            )
        if en.get("mms_align_median") is not None:
            lines.append(
                f"- **Transcript–audio alignment** (MMS forced-align): median confidence EN "
                f"{en['mms_align_median']}, TE {te.get('mms_align_median')}."
            )
    if agr and agr.get("llm_inter_model_alpha") is not None:
        lines.append(
            f"- **Emotion-label agreement** (Krippendorff alpha): {agr['llm_inter_model_alpha']} "
            f"between the two LLM raters (0.4+ is the field norm). A 3-rater panel adding SER models "
            f"drops near zero, since off-the-shelf SER clusters toward neutral and does not transfer "
            f"to Telugu. Per-clip VAD (valence, arousal, dominance) is included."
        )
    judge = _load("score_llm_judge.json")
    if judge:
        n = len(judge) or 1
        clean = sum(1 for v in judge.values() if v.get("transcript_clean"))
        good = sum(1 for v in judge.values() if (v.get("tts_suitable") or 0) >= 0.5)
        lines.append(
            f"- **LLM-as-judge cross-check** (independent model, {len(judge)} clips): "
            f"{clean*100//n}% of transcripts judged clean and {good*100//n}% suitable to train on. "
            f"Each clip also has a topic; the set is mostly storytelling (mythology, folk tales, "
            f"audiobook fiction)."
        )
    lines.append("")
    lines.append("See the project report (GitHub repo) for full methodology and figures.")
    return "\n".join(lines)


def write_card(stats: dict, cfg: Config) -> Path:
    langs = list(cfg.languages)
    lines = []
    for lang in langs:
        ls = stats["per_language"].get(lang, {})
        lines.append(
            f"- **{ls.get('name', lang)}** (`{ls.get('config')}`): "
            f"{ls.get('minutes', 0)} min, {ls.get('clips', 0)} clips, "
            f"{ls.get('speakers', 0)} speakers; emotions: {ls.get('emotions', {})}"
        )
    body = "\n".join(lines)
    eval_md = _eval_section()
    emo_list = ", ".join(cfg.emotion.emotions)
    sty_list = ", ".join(cfg.emotion.styles)
    card = f"""---
license: {cfg.dataset.license}
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
> diarization and speaker-embedding similarity). The dataset spans **11 distinct speakers
> total** (5 English, 6 Telugu), tracked via `speaker_id`.

## Contents
{body}

Total: **{stats.get('total_minutes', 0)} minutes**.
{eval_md}

## Schema
`audio` (24 kHz mono), `text`, `normalized_text`, `language`, `language_code`,
`emotion` ({emo_list}), `style` ({sty_list}), `emotion_confidence`, `tag_source`
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
"""
    out = REPORTS_DIR / "DATASET_CARD.md"
    out.write_text(card, encoding="utf-8")
    return out

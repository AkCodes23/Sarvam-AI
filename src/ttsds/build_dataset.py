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


def _round_robin_balance(segs: list[Segment], target_seconds: float) -> list[Segment]:
    """Pick across emotion buckets in turn so rare emotions are fully included and
    the dominant one (usually neutral) is capped — until we hit the target."""
    buckets: dict[str, list[Segment]] = defaultdict(list)
    for s in sorted(segs, key=lambda x: x.id):
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
            out[lang] = _round_robin_balance(lang_segs, target)
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
        "duration": round(seg.duration_s, 3),
        "snr_db": seg.metrics.get("snr_db"),
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
        }
        out["total_minutes"] = round(out["total_minutes"] + minutes, 2)
    return out


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
- config_name: telugu
  data_files:
  - split: train
    path: telugu/train-*
  - split: validation
    path: telugu/validation-*
---

# Indian English + Telugu Single-Speaker TTS Dataset (emotion-tagged)

Clean, single-speaker audio clips sourced from YouTube, transcribed with **Sarvam**
ASR, segmented with diarization, and labeled with emotion/style tags. Built as a
data-quality / curation exercise.

## Contents
{body}

Total: **{stats.get('total_minutes', 0)} minutes**.

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
"""
    out = REPORTS_DIR / "DATASET_CARD.md"
    out.write_text(card, encoding="utf-8")
    return out

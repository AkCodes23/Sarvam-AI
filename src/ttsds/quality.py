"""Automated quality gates. Each segment gets metrics + a status
(pass | flag | reject) with explicit reasons. Flags are kept but surfaced for
human review; rejects are dropped from candidates."""

from __future__ import annotations

import numpy as np

from .audio import (
    clipped_fraction,
    gap_energy_ratio,
    peak_level,
    read_wav,
    silence_ratio,
    snr_db,
    spectral_flatness_mean,
)
from .config import PROJECT_ROOT, Config
from .models import Segment


def _tokens(text: str) -> set[str]:
    return set(text.lower().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def apply_gates(seg: Segment, cfg: Config) -> Segment:
    q = cfg.quality
    y, sr = read_wav(PROJECT_ROOT / seg.wav_path)

    peak = peak_level(y)
    clip_frac = clipped_fraction(y)
    snr = snr_db(y, sr)
    sil = silence_ratio(y, sr, top_db=cfg.segmentation.silence_top_db)
    gap = gap_energy_ratio(y, sr, top_db=cfg.segmentation.silence_top_db)
    flat = spectral_flatness_mean(y)
    n_chars = len(seg.transcript.strip())
    cps = n_chars / seg.duration_s if seg.duration_s > 0 else 0.0

    seg.metrics.update({
        "peak": round(peak, 4),
        "clipped_fraction": round(clip_frac, 5),
        "snr_db": round(snr, 2),
        "silence_ratio": round(sil, 3),
        "gap_energy_ratio": round(gap, 3),
        "spectral_flatness": round(flat, 4),
        "chars_per_sec": round(cps, 2),
    })

    reasons: list[str] = []
    flags: list[str] = []

    if not seg.transcript.strip():
        reasons.append("empty_transcript")
    if clip_frac > q.max_clipped_fraction:
        reasons.append(f"clipping({clip_frac:.3f})")
    if snr < q.min_snr_db:
        reasons.append(f"low_snr({snr:.1f}dB)")
    if sil > q.max_silence_ratio:
        reasons.append(f"too_much_silence({sil:.2f})")
    if not (q.min_chars_per_sec <= cps <= q.max_chars_per_sec) and n_chars > 0:
        reasons.append(f"char_rate({cps:.1f})")
    if seg.duration_s < cfg.segmentation.min_duration_s or seg.duration_s > cfg.segmentation.max_duration_s:
        reasons.append("duration_out_of_range")
    if seg.asr_language_probability is not None and seg.asr_language_probability < q.min_language_probability:
        reasons.append(f"low_asr_conf({seg.asr_language_probability:.2f})")

    # flags (kept, but reviewer should listen)
    if gap > q.max_gap_energy_ratio:
        flags.append(f"music_or_noise_bed({gap:.2f})")
    if seg.asr_language_code and seg.asr_language_code != seg.language_code:
        flags.append(f"lang_mismatch({seg.asr_language_code})")

    seg.reject_reasons = reasons
    seg.flags = flags
    seg.status = "reject" if reasons else ("flag" if flags else "pass")
    return seg


def dedup(segments: list[Segment], cfg: Config) -> list[Segment]:
    """Mark near-duplicate transcripts (per language) as rejects, keeping the first."""
    seen: dict[str, list[set[str]]] = {}
    for seg in segments:
        if seg.status == "reject":
            continue
        toks = _tokens(seg.transcript)
        kept = seen.setdefault(seg.language, [])
        if any(_jaccard(toks, prev) >= cfg.quality.dedup_similarity for prev in kept):
            seg.status = "reject"
            seg.reject_reasons.append("duplicate")
        else:
            kept.append(toks)
    return segments


def run_quality(segments: list[Segment], cfg: Config) -> list[Segment]:
    for seg in segments:
        apply_gates(seg, cfg)
    dedup(segments, cfg)
    return segments

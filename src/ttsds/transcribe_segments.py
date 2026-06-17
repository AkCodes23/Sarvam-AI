"""Second ASR pass: transcribe each finished clip with the realtime API so the
transcript is exactly aligned to the published audio (the batch transcript is
coarse). Also yields word timing -> speaking rate, and a confidence score."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .audio import extract_to_wav
from .config import PROJECT_ROOT, Config
from .models import Segment
from .sarvam_client import transcribe_clip


def _transcribe_one(seg: Segment, cfg: Config) -> None:
    clip24 = PROJECT_ROOT / seg.wav_path
    tmp16 = clip24.with_name(clip24.stem + ".16k.wav")
    try:
        extract_to_wav(clip24, tmp16, sr=cfg.audio.asr_sample_rate, channels=1)
        res = transcribe_clip(tmp16, seg.language_code, model=cfg.asr.realtime_model)
    finally:
        tmp16.unlink(missing_ok=True)

    # prefer the clip-aligned transcript; fall back to the batch text if empty
    seg.transcript = res.transcript or seg.transcript_batch
    seg.asr_language_probability = res.language_probability
    seg.asr_language_code = res.language_code

    n_words = len(res.words)
    seg.features["n_words"] = n_words
    seg.features["speaking_rate_wps"] = (
        round(n_words / seg.duration_s, 3) if seg.duration_s > 0 else 0.0
    )


def transcribe_segments(segments: list[Segment], cfg: Config, workers: int = 6) -> list[Segment]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda s: _transcribe_one(s, cfg), segments))
    return segments

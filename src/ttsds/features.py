"""Acoustic feature extraction (parselmouth + librosa) and per-speaker
normalization. These descriptors ground the emotion tags so the LLM can't just
text-guess prosody."""

from __future__ import annotations

from collections import defaultdict

import librosa
import numpy as np
import parselmouth

from .audio import read_wav, silence_ratio
from .config import PROJECT_ROOT, Config
from .models import Segment

# features that get per-speaker z-scored (relative prosody is what signals emotion)
Z_FEATURES = [
    "f0_mean", "f0_std", "f0_range", "rms_mean", "rms_dynamic",
    "speaking_rate_wps", "hf_lf_ratio", "hnr_mean", "voiced_fraction",
]


def extract_features(y: np.ndarray, sr: int) -> dict:
    feats: dict[str, float] = {}
    if y.size < int(0.2 * sr):
        return feats

    snd = parselmouth.Sound(values=y.astype(np.float64), sampling_frequency=sr)

    # --- pitch / voicing ---
    try:
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=70, pitch_ceiling=500)
        f0 = pitch.selected_array["frequency"]
        voiced = f0[f0 > 0]
        feats["voiced_fraction"] = float(len(voiced) / max(1, len(f0)))
        if voiced.size:
            feats["f0_mean"] = float(np.mean(voiced))
            feats["f0_std"] = float(np.std(voiced))
            feats["f0_range"] = float(np.percentile(voiced, 95) - np.percentile(voiced, 5))
        else:
            feats["f0_mean"] = feats["f0_std"] = feats["f0_range"] = 0.0
    except Exception:  # noqa: BLE001
        feats.update(voiced_fraction=0.0, f0_mean=0.0, f0_std=0.0, f0_range=0.0)

    # --- harmonics-to-noise ratio (breathiness / whisper signal) ---
    try:
        harm = snd.to_harmonicity_cc()
        vals = harm.values[harm.values != -200.0]
        feats["hnr_mean"] = float(np.mean(vals)) if vals.size else 0.0
    except Exception:  # noqa: BLE001
        feats["hnr_mean"] = 0.0

    # --- energy dynamics ---
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms)) if rms.size else 0.0
    feats["rms_mean"] = rms_mean
    feats["rms_std"] = float(np.std(rms)) if rms.size else 0.0
    feats["rms_dynamic"] = float(feats["rms_std"] / (rms_mean + 1e-9))

    # --- spectral brightness + high/low band ratio (whisper has more HF noise) ---
    feats["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    S = np.abs(librosa.stft(y, n_fft=1024)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    low = S[freqs < 2000].sum()
    high = S[freqs >= 2000].sum()
    feats["hf_lf_ratio"] = float(high / (low + 1e-9))

    feats["pause_ratio"] = float(silence_ratio(y, sr, top_db=35))
    return feats


def compute_features(segments: list[Segment], cfg: Config) -> list[Segment]:
    for seg in segments:
        y, sr = read_wav(PROJECT_ROOT / seg.wav_path)
        feats = extract_features(y, sr)
        # keep speaking_rate_wps / n_words set during transcription
        feats.update({k: v for k, v in seg.features.items() if k not in feats})
        seg.features.update(feats)
    return segments


def normalize_per_speaker(segments: list[Segment]) -> list[Segment]:
    """Z-score Z_FEATURES within each speaker so tags are relative to that voice."""
    by_speaker: dict[str, list[Segment]] = defaultdict(list)
    for seg in segments:
        by_speaker[seg.speaker_id].append(seg)

    for segs in by_speaker.values():
        for feat in Z_FEATURES:
            vals = np.array([s.features.get(feat, 0.0) for s in segs], dtype=float)
            mu, sd = float(np.mean(vals)), float(np.std(vals))
            for s in segs:
                raw = s.features.get(feat, 0.0)
                s.features_z[feat] = round((raw - mu) / sd, 3) if sd > 1e-6 else 0.0
    return segments

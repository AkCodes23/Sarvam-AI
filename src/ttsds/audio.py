"""Audio I/O, ffmpeg conversion, silence-aware segmentation helpers, and the
acoustic quality metrics used by the gates. All clips are mono float32."""

from __future__ import annotations

import subprocess
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from .config import FFMPEG, FFPROBE

EPS = 1e-9


# --- ffmpeg / ffprobe ---------------------------------------------------------

def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def extract_to_wav(
    src: Path, dst: Path, *, sr: int, channels: int = 1,
    start_s: float | None = None, dur_s: float | None = None,
) -> Path:
    """Resample/downmix (and optionally trim) any audio to a PCM16 WAV."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    args = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error"]
    if start_s is not None:
        args += ["-ss", f"{start_s:.3f}"]
    args += ["-i", str(src)]
    if dur_s is not None:
        args += ["-t", f"{dur_s:.3f}"]
    args += ["-ac", str(channels), "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(args, check=True, capture_output=True)
    return dst


# --- wav I/O ------------------------------------------------------------------

def read_wav(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def write_wav(path: Path, y: np.ndarray, sr: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y.astype(np.float32), sr, subtype="PCM_16")
    return path


# --- silence / segmentation ---------------------------------------------------

def nonsilent_intervals(
    y: np.ndarray, sr: int, *, top_db: float, min_gap_s: float
) -> list[tuple[int, int]]:
    """Sample intervals of speech; gaps shorter than min_gap_s are bridged."""
    if y.size == 0 or float(np.max(np.abs(y))) < EPS:
        return []  # silent / empty -> no speech (librosa.split degenerates on all-zero)
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return []
    min_gap = int(min_gap_s * sr)
    merged: list[list[int]] = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(int(s), int(e)) for s, e in merged]


def trim_edges(y: np.ndarray, sr: int, *, top_db: float, pad_s: float) -> np.ndarray:
    """Trim leading/trailing silence, leaving pad_s of margin on each side."""
    iv = nonsilent_intervals(y, sr, top_db=top_db, min_gap_s=10.0)  # one span
    if not iv:
        return y
    pad = int(pad_s * sr)
    start = max(0, iv[0][0] - pad)
    end = min(len(y), iv[-1][1] + pad)
    return y[start:end]


# --- quality metrics ----------------------------------------------------------

def peak_level(y: np.ndarray) -> float:
    return float(np.max(np.abs(y))) if y.size else 0.0


def clipped_fraction(y: np.ndarray, thresh: float = 0.999) -> float:
    """Fraction of samples at/near full scale. Real clipping flat-tops many samples;
    a peak merely touching 1.0 (hot mastering) yields a tiny fraction."""
    return float(np.mean(np.abs(y) >= thresh)) if y.size else 0.0


def _frame_rms(y: np.ndarray, sr: int, frame_ms: float = 25.0) -> np.ndarray:
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, frame // 2)
    return librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]


def snr_db(y: np.ndarray, sr: int) -> float:
    """Crude but stable SNR: loud-frame RMS vs quiet-frame (noise floor) RMS."""
    if y.size < sr // 10:
        return 0.0
    r = _frame_rms(y, sr)
    if r.size == 0:
        return 0.0
    noise = np.percentile(r, 10)
    signal = np.percentile(r, 90)
    return float(20.0 * np.log10((signal + EPS) / (noise + EPS)))


def silence_ratio(y: np.ndarray, sr: int, *, top_db: float) -> float:
    if y.size == 0:
        return 1.0
    iv = nonsilent_intervals(y, sr, top_db=top_db, min_gap_s=0.0)
    speech = sum(e - s for s, e in iv)
    return float(1.0 - speech / len(y))


def gap_energy_ratio(y: np.ndarray, sr: int, *, top_db: float) -> float:
    """RMS energy inside inter-phrase pauses / RMS during speech.

    Clean speech -> near 0 (pauses are silent). A sustained music or noise bed
    keeps energy in the pauses -> ratio rises. Robust music/noise-bed signal.
    """
    if y.size == 0:
        return 0.0
    iv = nonsilent_intervals(y, sr, top_db=top_db, min_gap_s=0.0)
    if not iv:
        return 0.0
    mask = np.zeros(len(y), dtype=bool)
    for s, e in iv:
        mask[s:e] = True
    speech = y[mask]
    gaps = y[~mask]
    if speech.size == 0 or gaps.size < sr // 20:  # need a meaningful amount of gap
        return 0.0
    speech_rms = float(np.sqrt(np.mean(speech ** 2)) + EPS)
    gap_rms = float(np.sqrt(np.mean(gaps ** 2)))
    return gap_rms / speech_rms


def spectral_flatness_mean(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.mean(librosa.feature.spectral_flatness(y=y)))


# --- normalization (light — preserves emotional dynamics) ---------------------

def peak_normalize(y: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    peak = peak_level(y)
    if peak < EPS:
        return y
    gain = (10.0 ** (target_dbfs / 20.0)) / peak
    return (y * gain).astype(np.float32)


def loudness_normalize(
    y: np.ndarray, sr: int, target_lufs: float = -20.0, peak_ceiling_dbfs: float = -1.0
) -> np.ndarray:
    """Gentle gain to a loudness target, then a peak guard. No limiting/compression,
    so relative intensity *within* a clip (the emotional dynamics) is preserved."""
    if y.size < int(0.4 * sr):
        return peak_normalize(y, peak_ceiling_dbfs)
    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        if not np.isfinite(loudness):
            return peak_normalize(y, peak_ceiling_dbfs)
        y = pyln.normalize.loudness(y, loudness, target_lufs).astype(np.float32)
    except Exception:  # noqa: BLE001
        return peak_normalize(y, peak_ceiling_dbfs)
    # peak guard so the gain never clips
    peak = peak_level(y)
    ceiling = 10.0 ** (peak_ceiling_dbfs / 20.0)
    if peak > ceiling:
        y = (y * (ceiling / peak)).astype(np.float32)
    return y

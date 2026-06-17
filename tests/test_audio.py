import numpy as np

from ttsds.audio import (
    nonsilent_intervals,
    peak_level,
    peak_normalize,
    silence_ratio,
)

SR = 16000


def _tone(dur_s, freq=220, amp=0.3):
    t = np.linspace(0, dur_s, int(dur_s * SR), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_peak_normalize_hits_target():
    y = _tone(1.0, amp=0.1)
    out = peak_normalize(y, target_dbfs=-1.0)
    target = 10 ** (-1.0 / 20.0)
    assert abs(peak_level(out) - target) < 1e-3


def test_silence_ratio_extremes():
    assert silence_ratio(np.zeros(SR, dtype=np.float32), SR, top_db=35) == 1.0
    assert silence_ratio(_tone(1.0), SR, top_db=35) < 0.2


def test_nonsilent_intervals_finds_speech_island():
    sig = np.concatenate([np.zeros(SR // 2, np.float32), _tone(1.0), np.zeros(SR // 2, np.float32)])
    iv = nonsilent_intervals(sig, SR, top_db=35, min_gap_s=0.1)
    assert len(iv) == 1
    s, e = iv[0]
    assert s > 0 and e < len(sig)

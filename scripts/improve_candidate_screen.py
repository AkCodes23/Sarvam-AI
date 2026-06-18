"""Automated screening of candidate English voices (#2/#3) — NOT merged/published.

For each candidate source: clip yield, DNSMOS cleanliness, emotion spread, and
ECAPA speaker-distinctness vs the 9 published speakers (a low max-cosine means a
genuinely new voice). Honest caveat: this is automated screening; final
single-speaker / clean judgement still needs a by-ear pass.

Writes data/manifests/improvements/candidate_voices.json.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechmos import dnsmos

from ttsds.config import MANIFEST_DIR, PROJECT_ROOT
from ttsds.models import load_all_segments

CANDIDATES = ["en_estories", "en_audiobook_kafan", "en_air_century"]
OUT = MANIFEST_DIR / "improvements" / "candidate_voices.json"
SR = 16000
SAMPLE_EXISTING = 12


def _load16k(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g).astype(np.float32)
    return np.clip(y, -1.0, 1.0)


def _dnsmos_ovrl(y: np.ndarray) -> float:
    r = dnsmos.run(y, sr=SR)
    for k in ("ovrl_mos", "OVRL", "ovrl"):
        if k in r:
            return float(r[k])
    return float(next(iter(r.values())))


def main() -> None:
    # Warm up librosa's lazy submodules (used by dnsmos) BEFORE importing speechbrain,
    # whose lazy-import machinery otherwise hijacks librosa's lazy load and fails on k2.
    import librosa
    librosa.feature.melspectrogram(y=np.zeros(SR, dtype=np.float32), sr=SR)
    from speechbrain.inference import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})

    segs = [s for s in load_all_segments() if s.is_kept()]
    cand = {c: [s for s in segs if s.source_id == c] for c in CANDIDATES}
    published = defaultdict(list)
    for s in segs:
        if s.source_id not in CANDIDATES:
            published[s.speaker_id].append(s)

    def embed(s) -> np.ndarray:
        y = _load16k(str(PROJECT_ROOT / s.wav_path))
        with torch.no_grad():
            e = ecapa.encode_batch(torch.from_numpy(y).float().unsqueeze(0)).squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    def centroid(group, k=None) -> np.ndarray:
        g = group[:: max(1, len(group) // k)][:k] if k else group
        v = np.mean([embed(s) for s in g], axis=0)
        return v / (np.linalg.norm(v) + 1e-9)

    pub_centroids = {spk: centroid(g, SAMPLE_EXISTING) for spk, g in published.items()}

    report = {}
    for c, group in cand.items():
        if not group:
            report[c] = {"error": "no kept clips"}
            continue
        cc = centroid(group)
        sims = {spk: round(float(np.dot(cc, v)), 3) for spk, v in pub_centroids.items()}
        nearest = max(sims, key=sims.get)
        ovrls = [_dnsmos_ovrl(_load16k(str(PROJECT_ROOT / s.wav_path))) for s in group]
        snr = [s.metrics.get("snr_db") for s in group if s.metrics.get("snr_db") is not None]
        report[c] = {
            "clips": len(group),
            "minutes": round(sum(s.duration_s for s in group) / 60, 1),
            "dnsmos_median": round(float(np.median(ovrls)), 2),
            "dnsmos_pass_pct": round(100 * np.mean([o > 3.0 for o in ovrls])),
            "snr_db_median": round(float(np.median(snr)), 1) if snr else None,
            "emotions": dict(Counter(s.emotion for s in group).most_common()),
            "nearest_published_speaker": nearest,
            "max_cosine_to_published": sims[nearest],
            "distinct_voice": sims[nearest] < 0.50,   # same-spk ~0.74, diff-spk ~0.21 (eval_speaker)
            "all_cosines": dict(sorted(sims.items(), key=lambda x: -x[1])),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for c, r in report.items():
        if "error" in r:
            print(f"{c}: {r['error']}"); continue
        print(f"{c}: {r['clips']} clips / {r['minutes']}min | DNSMOS {r['dnsmos_median']} "
              f"(pass {r['dnsmos_pass_pct']}%) | SNR {r['snr_db_median']} | "
              f"distinct={r['distinct_voice']} (max cos {r['max_cosine_to_published']} to {r['nearest_published_speaker']})")
        print(f"    emotions: {r['emotions']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

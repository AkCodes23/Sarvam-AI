"""Per-clip single-speaker integrity via INTRA-CLIP ECAPA cohesion.

pyannote's overlapped-speech model is gated (license can't be accepted
programmatically), so instead of frame-level overlap we test the more directly
relevant property for TTS clips: is the clip ONE consistent voice end-to-end?
We embed sliding windows within each clip with ECAPA-TDNN and take the mean
pairwise cosine similarity (cohesion). A second speaker / speaker-change splits
the embeddings -> low cohesion -> overlap_flag. Solo clips score ~0.8+.

Writes data/manifests/score_overlap.json keyed by segment id.
"""

from __future__ import annotations

import json
import statistics as st
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference import EncoderClassifier

from ttsds.config import MANIFEST_DIR, PROJECT_ROOT
from ttsds.models import load_all_segments

WIN_S, HOP_S, SR = 2.0, 1.0, 16000
COHESION_FLAG_BELOW = 0.55   # below this -> likely a 2nd speaker / speaker change


def load_16k(path: str) -> np.ndarray:
    """Load mono 16 kHz without librosa (avoids a librosa/speechbrain lazy-import clash)."""
    y, sr0 = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr0 != SR:
        y = resample_poly(y, SR, sr0).astype(np.float32)
    return y


def clip_cohesion(model, y: np.ndarray) -> float | None:
    win, hop = int(WIN_S * SR), int(HOP_S * SR)
    if len(y) < win:
        return None  # too short to window -> trust single-speaker segmentation
    embs = []
    for a in range(0, len(y) - win + 1, hop):
        seg = torch.from_numpy(y[a:a + win]).float().unsqueeze(0)
        with torch.no_grad():
            e = model.encode_batch(seg).squeeze().numpy()
        embs.append(e / (np.linalg.norm(e) + 1e-9))
    if len(embs) < 2:
        return None
    E = np.stack(embs)
    sims = [float(E[i] @ E[j]) for i in range(len(E)) for j in range(i + 1, len(E))]
    return float(np.mean(sims))


def main() -> None:
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )
    segs = [s for s in load_all_segments() if s.is_kept()]
    scores: dict[str, dict] = {}
    by_lang: dict[str, list[bool]] = defaultdict(list)
    cohs: list[float] = []
    for i, s in enumerate(segs, 1):
        y = load_16k(str(PROJECT_ROOT / s.wav_path))
        coh = clip_cohesion(model, y)
        if coh is None:
            scores[s.id] = {"intra_clip_cohesion": None, "overlap_flag": False}
        else:
            flag = coh < COHESION_FLAG_BELOW
            scores[s.id] = {"intra_clip_cohesion": round(coh, 3), "overlap_flag": flag}
            by_lang[s.language].append(flag)
            cohs.append(coh)
        if i % 50 == 0:
            print(f"{i}/{len(segs)} processed", flush=True)

    MANIFEST_DIR.joinpath("score_overlap.json").write_text(
        json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    flagged = sum(1 for v in scores.values() if v["overlap_flag"])
    print(f"\n=== Intra-clip single-speaker cohesion (flag if < {COHESION_FLAG_BELOW}) ===")
    print(f"clips: {len(scores)}  median cohesion: {st.median(cohs):.3f}  flagged: {flagged}")
    for lang in sorted(by_lang):
        print(f"  {lang}: flagged {sum(by_lang[lang])}/{len(by_lang[lang])}")


if __name__ == "__main__":
    main()

"""Deliverable (B) [OPTIONAL]: categorical SER with emotion2vec_plus_large.

Runs inside the ISOLATED .venv_ser interpreter (funasr/modelscope), NOT the main
.venv. A higher-quality multilingual categorical rater over a stratified subset
(~100 clips, 50/lang deterministic stride). If anything here fails (install,
download, runtime), the caller skips it gracefully and the agreement eval falls
back to the VAD-derived SER label.

Reads the kept-clip list straight from the manifests on disk (no ttsds import, so
the isolated venv needs nothing from the main package). Writes
data/manifests/score_emotion2vec.json -> { id: {"label": <raw emotion2vec label>} }.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
N_PER_LANG = 50
MODEL = "emotion2vec/emotion2vec_plus_large"

# Canonical emotion buckets; anything else collapses to "other".
KNOWN_EMOTIONS = {
    "angry", "happy", "sad", "neutral", "fearful", "surprised", "disgusted",
}


def load_kept() -> list[dict]:
    """Read every *.segments.json, return kept clips as plain dicts."""
    out = []
    for p in sorted(MANIFEST_DIR.glob("*.segments.json")):
        for r in json.loads(p.read_text(encoding="utf-8")):
            review = r.get("review_decision")
            status = r.get("status")
            kept = (review == "accept") or (review != "reject" and status in ("pass", "flag"))
            if kept:
                out.append(r)
    return out


def sample_per_language(segs: list[dict], n: int) -> list[dict]:
    by_lang = defaultdict(list)
    for s in segs:
        by_lang[s["language"]].append(s)
    out = []
    for lang in sorted(by_lang):
        items = sorted(by_lang[lang], key=lambda s: s["id"])
        if len(items) <= n:
            out.extend(items)
            continue
        k = len(items) / n
        out.extend(items[int(i * k)] for i in range(n))
    return out


def map_label(raw: str) -> str:
    """emotion2vec labels look like 'angry/生气' or '生气/angry'; keep the english
    token and collapse anything outside the canonical set to 'other'."""
    parts = [p.strip().lower() for p in str(raw).split("/") if p.strip()]
    eng = next((p for p in parts if p in KNOWN_EMOTIONS), None)
    if eng:
        return eng
    # fallback: last token, lowercased
    tail = parts[-1] if parts else ""
    return tail if tail in KNOWN_EMOTIONS else "other"


def load_16k_mono(path: str):
    """Read audio at 16 kHz mono without librosa's soxr resampler (broken on this
    box). soundfile reads native sample rate; scipy.signal.resample_poly handles
    rate conversion. Returns float32 mono waveform."""
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(wav, "ndim", 1) > 1:  # downmix to mono
        wav = wav.mean(axis=1)
    if sr != 16000:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), 16000)
        wav = resample_poly(wav, 16000 // g, int(sr) // g).astype("float32")
    return np.ascontiguousarray(wav, dtype="float32")


def main() -> None:
    from funasr import AutoModel

    kept = load_kept()
    sample = sample_per_language(kept, N_PER_LANG)
    print(f"sampled {len(sample)} clips; loading {MODEL} ...", flush=True)

    # Use the HuggingFace hub to avoid modelscope.cn reachability issues.
    model = AutoModel(model=MODEL, hub="hf", disable_update=True)

    out: dict[str, dict] = {}
    for i, seg in enumerate(sample, 1):
        wav_path = str(PROJECT_ROOT / seg["wav_path"])
        try:
            wav = load_16k_mono(wav_path)
            res = model.generate(
                wav, granularity="utterance", extract_embedding=False,
            )
            rec = res[0] if isinstance(res, list) and res else res
            labels = rec.get("labels") or []
            scores = rec.get("scores") or []
            if labels and scores:
                top = max(range(len(scores)), key=lambda j: scores[j])
                label = map_label(labels[top])
            else:
                label = "other"
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {seg['id']}: {exc}", flush=True)
            continue
        out[seg["id"]] = {"label": label}
        if i % 20 == 0 or i == len(sample):
            print(f"  scored {i}/{len(sample)}", flush=True)

    dest = MANIFEST_DIR / "score_emotion2vec.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dest} ({len(out)} clips)", flush=True)


if __name__ == "__main__":
    main()

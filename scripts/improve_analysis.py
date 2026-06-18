"""#1 emotion machine cross-check + #6 forced-alignment edge-trim opportunity.

Both are analysis-only (no model downloads, no API, no dataset mutation). Outputs
go to data/manifests/improvements/. Honest framing: the emotion cross-check is
three automatic signals triangulating, NOT a human relabel; the edge-trim figure
is the silence a word-level trim could remove, measured on the published clips.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf

from ttsds.config import PROJECT_ROOT, MANIFEST_DIR
from ttsds.build_dataset import FINAL_SELECTION

IMP = MANIFEST_DIR / "improvements"
IMP.mkdir(parents=True, exist_ok=True)
POS = {"happy", "excited", "calm"}
HIGH_AROUSAL = {"angry", "excited", "fearful", "surprised"}


def emotion_crosscheck(fs: dict) -> dict:
    agree = total = 0
    conf = defaultdict(Counter)          # LLM emotion -> SER emotion counts
    vbe = defaultdict(lambda: {"v": [], "a": []})
    for cfg, rs in fs.items():
        for r in rs:
            tag, ser = r.get("emotion"), r.get("ser_emotion")
            v, a = r.get("valence"), r.get("arousal")
            if tag and ser:
                total += 1
                agree += (tag == ser)
                conf[tag][ser] += 1
            if tag and v is not None:
                vbe[tag]["v"].append(v); vbe[tag]["a"].append(a)
    # VAD ordering sanity: mean valence/arousal per tag
    vad = {t: {"valence": round(float(np.mean(d["v"])), 3),
               "arousal": round(float(np.mean(d["a"])), 3),
               "n": len(d["v"])}
           for t, d in vbe.items()}
    # directional checks
    pos_val = np.mean([vad[t]["valence"] for t in POS if t in vad])
    neg_val = np.mean([vad[t]["valence"] for t in ("sad", "angry", "fearful") if t in vad])
    hi_ar = np.mean([vad[t]["arousal"] for t in HIGH_AROUSAL if t in vad])
    lo_ar = np.mean([vad[t]["arousal"] for t in ("calm", "neutral", "sad") if t in vad])
    return {
        "n": total,
        "ser_categorical_agreement_pct": round(100 * agree / total) if total else None,
        "note_ser_labelset": "SER model emits 6 classes (no fearful/surprised) and rarely predicts calm",
        "confusion_LLM_to_SER": {t: dict(c.most_common()) for t, c in conf.items()},
        "vad_mean_by_tag": vad,
        "vad_directional_check": {
            "positive_tags_mean_valence": round(float(pos_val), 3),
            "negative_tags_mean_valence": round(float(neg_val), 3),
            "valence_orders_correctly": bool(pos_val > neg_val),
            "high_arousal_tags_mean_arousal": round(float(hi_ar), 3),
            "low_arousal_tags_mean_arousal": round(float(lo_ar), 3),
            "arousal_orders_correctly": bool(hi_ar > lo_ar),
        },
    }


def edge_trim(fs: dict) -> dict:
    """Leading/trailing low-energy that a word-level forced-alignment trim could cut."""
    out = {}
    for cfg, rs in fs.items():
        lead, trail = [], []
        for r in rs:
            try:
                y, sr = sf.read(str(PROJECT_ROOT / r["audio"]), dtype="float32")
            except Exception:  # noqa: BLE001
                continue
            if y.ndim > 1:
                y = y.mean(axis=1)
            fl = int(0.025 * sr); hop = int(0.010 * sr)
            if len(y) < fl:
                continue
            frames = [y[i:i + fl] for i in range(0, len(y) - fl, hop)]
            energy = np.array([float(np.sqrt(np.mean(f * f))) for f in frames])
            peak = energy.max() or 1e-9
            voiced = energy > (peak * 0.10)        # -20 dB of clip peak
            if not voiced.any():
                continue
            idx = np.where(voiced)[0]
            lead.append(idx[0] * hop / sr)
            trail.append((len(frames) - 1 - idx[-1]) * hop / sr)
        lead, trail = np.array(lead), np.array(trail)
        out[cfg] = {
            "clips": len(lead),
            "lead_ms_median": round(float(np.median(lead) * 1000)),
            "lead_ms_mean": round(float(lead.mean() * 1000)),
            "trail_ms_median": round(float(np.median(trail) * 1000)),
            "trail_ms_mean": round(float(trail.mean() * 1000)),
            "pct_clips_gt_150ms_edge": round(100 * float(np.mean((lead > 0.15) | (trail > 0.15)))),
            "total_trimmable_s": round(float((lead + trail).sum()), 1),
        }
    return out


def main() -> None:
    fs = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))
    emo = emotion_crosscheck(fs)
    dis = {}
    p = MANIFEST_DIR / "emotion_disagreement.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        dis = {"sarvam30b_vs_105b_agree_pct": round(100 * d["agree"] / d["n"]) if d.get("n") else None,
               "n": d.get("n")}
    emo["sarvam_30b_vs_105b"] = dis
    (IMP / "emotion_crosscheck.json").write_text(json.dumps(emo, ensure_ascii=False, indent=2), encoding="utf-8")
    print("EMOTION:", json.dumps({k: emo[k] for k in ("n", "ser_categorical_agreement_pct",
          "vad_directional_check", "sarvam_30b_vs_105b")}, ensure_ascii=False, indent=2))
    print("calm -> SER:", emo["confusion_LLM_to_SER"].get("calm"))
    print("neutral -> SER:", emo["confusion_LLM_to_SER"].get("neutral"))

    trim = edge_trim(fs)
    (IMP / "edge_trim.json").write_text(json.dumps(trim, ensure_ascii=False, indent=2), encoding="utf-8")
    print("EDGE-TRIM:", json.dumps(trim, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

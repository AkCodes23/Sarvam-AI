"""Multi-rater emotion agreement (Krippendorff's alpha), credit-free.

Three independent raters over the clips that all three labelled:
  Rater 1 = sarvam-30b  -> the shipped `seg.emotion` (LLM, text + acoustics)
  Rater 2 = emotion2vec -> multilingual speech-emotion model (score_emotion2vec.json)
  Rater 3 = audeering   -> wav2vec VAD-derived label (score_ser.json)

All labels are folded onto the project's 8-emotion taxonomy. We report Krippendorff's
nominal alpha across the 3 raters and pairwise percent agreement. The 30b-vs-105b LLM
inter-model alpha from the earlier run (when credits were available) is preserved.

Output: data/manifests/eval_agreement.json
"""

from __future__ import annotations

import json
from collections import Counter

import krippendorff

from ttsds.config import MANIFEST_DIR, load_config
from ttsds.models import load_all_segments

E2V = MANIFEST_DIR / "score_emotion2vec.json"
SER = MANIFEST_DIR / "score_ser.json"
OUT = MANIFEST_DIR / "eval_agreement.json"

# fold any model's raw label onto the project taxonomy; unknown -> neutral
MAP = {
    "angry": "angry", "anger": "angry", "disgusted": "angry", "disgust": "angry",
    "happy": "happy", "happiness": "happy", "excited": "excited", "excitement": "excited",
    "sad": "sad", "sadness": "sad", "neutral": "neutral", "calm": "calm",
    "surprised": "surprised", "surprise": "surprised",
    "fearful": "fearful", "fear": "fearful", "other": "neutral", "unknown": "neutral",
}


def fold(label: str) -> str:
    return MAP.get(str(label).split("/")[0].lower().strip(), "neutral")


def main() -> None:
    cfg = load_config()
    emotions = cfg.emotion.emotions
    idx = {e: i for i, e in enumerate(emotions)}

    by_id = {s.id: s for s in load_all_segments() if s.is_kept()}
    e2v = json.loads(E2V.read_text(encoding="utf-8")) if E2V.exists() else {}
    e2v = {k: v for k, v in e2v.items() if isinstance(v, dict) and "label" in v}
    ser = json.loads(SER.read_text(encoding="utf-8")) if SER.exists() else {}

    r1, r2, r3 = [], [], []          # 30b, emotion2vec, audeering
    for sid in sorted(e2v):
        seg = by_id.get(sid)
        if seg is None or seg.emotion not in emotions:
            continue
        l2 = fold(e2v[sid]["label"])
        l3 = fold(ser.get(sid, {}).get("ser_emotion", "")) if sid in ser else None
        r1.append(seg.emotion)
        r2.append(l2 if l2 in emotions else "neutral")
        r3.append(l3 if l3 in emotions else None)

    def alpha(rows):
        rel = [[idx[v] if v in idx else None for v in row] for row in rows]
        try:
            return round(float(krippendorff.alpha(reliability_data=rel,
                         level_of_measurement="nominal")), 4)
        except Exception:  # noqa: BLE001
            return None

    def pct(a, b):
        p = [(x, y) for x, y in zip(a, b) if x and y]
        return round(sum(x == y for x, y in p) / len(p), 4) if p else 0.0

    prior = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    llm_alpha = prior.get("llm_inter_model_alpha") or prior.get("krippendorff_alpha_llms")

    out = {
        "n": len(r1),
        "raters": ["sarvam-30b (LLM)", "emotion2vec (multilingual SER)", "audeering (VAD SER)"],
        "krippendorff_alpha_3raters": alpha([r1, r2, r3]),
        "krippendorff_alpha_30b_vs_emotion2vec": alpha([r1, r2]),
        "llm_inter_model_alpha": llm_alpha,   # sarvam-30b vs sarvam-105b, earlier run
        "pairwise_agreement": {
            "30b_vs_emotion2vec": pct(r1, r2),
            "30b_vs_audeering": pct(r1, r3),
            "emotion2vec_vs_audeering": pct(r2, r3),
        },
        "emotion2vec_label_dist": dict(Counter(r2).most_common()),
        "shipped_30b_label_dist": dict(Counter(r1).most_common()),
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

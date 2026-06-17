"""Source-level analysis + rejection breakdown (judgment evidence).

Per-source: clips, minutes, median SNR, median gap-energy, emotion diversity
(distinct count + Shannon entropy), median tag confidence. Plus the dataset-wide
rejection-reason table. Writes data/manifests/eval_sources.json.
"""

from __future__ import annotations

import json
import math
import statistics as st
from collections import Counter, defaultdict

from ttsds.config import MANIFEST_DIR
from ttsds.models import load_all_segments, load_sources
from ttsds.config import CONFIG_DIR


def entropy(counts: list[int]) -> float:
    tot = sum(counts)
    if tot == 0:
        return 0.0
    return round(-sum((c / tot) * math.log2(c / tot) for c in counts if c), 3)


def main() -> None:
    specs = {s.id: s for s in load_sources(CONFIG_DIR / "sources.yaml").sources}
    segs = load_all_segments()

    by_src: dict[str, list] = defaultdict(list)
    for s in segs:
        by_src[s.source_id].append(s)

    sources = []
    for sid, group in by_src.items():
        kept = [s for s in group if s.is_kept()]
        if not kept:
            continue
        spec = specs.get(sid)
        emo = Counter(s.emotion for s in kept if s.emotion)
        snrs = [s.metrics.get("snr_db") for s in kept if s.metrics.get("snr_db") is not None]
        gaps = [s.metrics.get("gap_energy_ratio") for s in kept if s.metrics.get("gap_energy_ratio") is not None]
        confs = [s.emotion_confidence for s in kept if s.emotion_confidence is not None]
        sources.append({
            "source_id": sid,
            "language": kept[0].language,
            "content_type": spec.content_type if spec else "?",
            "clips": len(kept),
            "minutes": round(sum(s.duration_s for s in kept) / 60, 2),
            "snr_db_median": round(st.median(snrs), 1) if snrs else None,
            "gap_energy_median": round(st.median(gaps), 3) if gaps else None,
            "distinct_emotions": len(emo),
            "emotion_entropy": entropy(list(emo.values())),
            "conf_median": round(st.median(confs), 2) if confs else None,
            "emotions": dict(emo.most_common()),
        })
    sources.sort(key=lambda r: (r["language"], -r["minutes"]))

    # rejection breakdown (all candidates)
    rej = Counter()
    for s in segs:
        if s.status == "reject":
            for r in s.reject_reasons:
                rej[r.split("(")[0]] += 1
    # include zeroed canonical reasons so the "deliberate gates" story is visible
    for canon in ["low_snr", "clipping", "music_or_noise_bed", "too_much_silence",
                  "low_asr_conf", "char_rate", "empty_transcript", "duplicate", "duration_out_of_range"]:
        rej.setdefault(canon, 0)

    candidates = len(segs)
    kept_total = sum(1 for s in segs if s.status != "reject")
    out = {
        "sources": sources,
        "rejection": dict(sorted(rej.items(), key=lambda x: -x[1])),
        "funnel": {"candidates": candidates, "kept": kept_total, "rejected": candidates - kept_total},
    }
    (MANIFEST_DIR / "eval_sources.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # observations (data-driven)
    best_snr = max(sources, key=lambda r: r["snr_db_median"] or 0)
    worst_snr = min(sources, key=lambda r: r["snr_db_median"] or 99)
    most_div = max(sources, key=lambda r: r["emotion_entropy"])
    print(f"sources: {len(sources)} | candidates {candidates} -> kept {kept_total}")
    print(f"highest SNR : {best_snr['source_id']} ({best_snr['snr_db_median']} dB, {best_snr['content_type']})")
    print(f"lowest SNR  : {worst_snr['source_id']} ({worst_snr['snr_db_median']} dB, {worst_snr['content_type']})")
    print(f"most emotionally diverse: {most_div['source_id']} (entropy {most_div['emotion_entropy']}, {most_div['content_type']})")
    print("rejection:", out["rejection"])
    print(json.dumps(sources, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

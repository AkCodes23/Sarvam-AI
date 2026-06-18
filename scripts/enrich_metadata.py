"""Merge all per-clip score files into the segment manifests and derive
speaker-level gender (from median F0) + accent. Re-runnable; merges whatever
score_*.json files exist.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict

from ttsds.config import MANIFEST_DIR
from ttsds.models import load_all_segments, save_segments
from ttsds.models import Segment

SCORE_FILES = [
    "score_audio_quality.json", "score_mms_align.json",
    "score_overlap.json", "score_ser.json", "score_llm_judge.json",
]
ACCENT = {"en": "Indian English", "te": "Telugu"}
F0_GENDER_BOUNDARY_HZ = 165.0
# F0-based gender is unreliable for expressive/high-pitched male speakers, so override
# the speakers whose identity is known (named public figures / clear narration).
KNOWN_GENDER = {
    "te_chaganti": "male",            # Chaganti Koteswara Rao
    "te_motivation_kasyap": "male",   # MVN Kasyap
    "te_ramaaraavi": "female",        # Ramaa Raavi
    "en_tedx_amina": "female",        # Amina Nijam
    "en_audiobook_kafan": "male",
    "te_audiobook_bhumiputri": "female",
}


def main() -> None:
    segs = load_all_segments()
    by_id = {s.id: s for s in segs}

    # 1) merge score files into seg.metrics
    merged = 0
    for fname in SCORE_FILES:
        p = MANIFEST_DIR / fname
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for sid, vals in data.items():
            s = by_id.get(sid)
            if s is None:
                continue
            for k, v in vals.items():
                if k == "language":
                    continue
                s.metrics[k] = v
            merged += 1
        print(f"merged {fname} ({len(data)} entries)")

    # 2) speaker gender from median F0, accent from language
    by_spk: dict[str, list[Segment]] = defaultdict(list)
    for s in segs:
        by_spk[s.speaker_id].append(s)
    gender_map = {}
    for spk, group in by_spk.items():
        f0s = [g.features.get("f0_mean") for g in group if g.features.get("f0_mean")]
        med = st.median(f0s) if f0s else 0.0
        gender = KNOWN_GENDER.get(spk, "male" if 0 < med < F0_GENDER_BOUNDARY_HZ else "female")
        gender_map[spk] = (gender, round(med, 1))
        for g in group:
            g.metrics["gender"] = gender
            g.metrics["accent"] = ACCENT.get(g.language, g.language)

    # 3) save back per source
    by_src: dict[str, list[Segment]] = defaultdict(list)
    for s in segs:
        by_src[s.source_id].append(s)
    for sid, group in by_src.items():
        save_segments(sid, group)

    print("\nspeaker gender (from median F0):")
    for spk, (g, f0) in sorted(gender_map.items()):
        print(f"  {spk:<28} {g:<7} (median F0 {f0} Hz)")


if __name__ == "__main__":
    main()

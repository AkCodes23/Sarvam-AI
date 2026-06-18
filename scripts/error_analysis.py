"""Concrete error analysis from REAL detected issues (no fabrication):
  - transcript corrections the double-pass made (batch ASR -> realtime ASR)
  - transcripts the LLM judge flagged as not clean, with its stated issue
  - emotion labels the LLM judge did not endorse, with its stated issue

Writes data/manifests/error_analysis.json and a markdown table for the report.
"""

from __future__ import annotations

import json
import re

from ttsds.config import MANIFEST_DIR
from ttsds.models import load_all_segments


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t.lower())).strip()


def main():
    segs = {s.id: s for s in load_all_segments() if s.is_kept()}
    judge = json.loads((MANIFEST_DIR / "score_llm_judge.json").read_text(encoding="utf-8"))

    # 1) double-pass transcript corrections: batch vs realtime differ, realtime longer/cleaner
    corrections = []
    for s in segs.values():
        b, r = norm(s.transcript_batch), norm(s.transcript)
        if b and r and b != r and 0 < abs(len(r.split()) - len(b.split())) <= 4 and s.language == "en":
            corrections.append((s.id, s.transcript_batch.strip(), s.transcript.strip()))
    corrections.sort(key=lambda x: len(x[2]))
    corrections = corrections[:3]

    # 2) transcripts the judge flagged unclean (distinct issues)
    tflag, seen = [], set()
    for sid, v in judge.items():
        if not v.get("transcript_clean") and v.get("issue") and v["issue"] not in seen:
            seen.add(v["issue"]); tflag.append((sid, v["language"], v["issue"]))
        if len(tflag) >= 3:
            break

    # 3) emotion labels the judge did not endorse
    eflag, seen2 = [], set()
    for sid, v in judge.items():
        s = segs.get(sid)
        if s and not v.get("emotion_supported") and v.get("issue") and v["issue"] not in seen2:
            seen2.add(v["issue"]); eflag.append((sid, s.emotion, v["issue"]))
        if len(eflag) >= 3:
            break

    out = {"transcript_corrections": [{"id": i, "before": b, "after": a} for i, b, a in corrections],
           "transcript_flags": [{"id": i, "lang": l, "issue": x} for i, l, x in tflag],
           "emotion_flags": [{"id": i, "emotion": e, "issue": x} for i, e, x in eflag]}
    (MANIFEST_DIR / "error_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== transcript corrections (double-pass: batch -> realtime) ===")
    for i, b, a in corrections:
        print(f"  {i}\n    before: {b[:70]}\n    after : {a[:70]}")
    print("\n=== transcripts flagged by judge ===")
    for i, l, x in tflag:
        print(f"  {i} ({l}): {x}")
    print("\n=== emotion labels not endorsed by judge ===")
    for i, e, x in eflag:
        print(f"  {i} [{e}]: {x}")


if __name__ == "__main__":
    main()

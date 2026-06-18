"""Edge-case annotation pass (no pipeline rebuild, no Sarvam calls).

Derives per-clip annotation flags from scores already in the manifests and writes
them back: has_noise, has_truncation, has_codemix, has_laughter,
emotion_low_confidence, transcript_review_needed, overlap_suspected,
low_quality_audio, a combined `annotation_flags` string, and an `annotated_text`
(English code-switch spans bracketed, truncation marked with an em dash).

Flags that require listening (laughter, cough, breath) are part of the documented
convention but are NOT auto-asserted here; they default to false.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from ttsds.models import Segment, load_all_segments, save_segments

TERMINAL = ".!?।॥\"')]}”"   # incl. Telugu danda/double-danda
LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)*")


def latin_fraction(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() for c in letters) / len(letters) if letters else 0.0


def annotated_text(seg: Segment, truncated: bool) -> str:
    t = seg.transcript.strip()
    if seg.language != "en":                      # bracket English (Latin) spans in regional text
        t = LATIN_RUN.sub(lambda m: f"[{m.group(0)}]", t)
    if truncated:
        t = t.rstrip(" .") + " —"            # em dash marks a cut-off utterance
    return t


def main() -> None:
    segs = [s for s in load_all_segments() if s.is_kept()]
    stats: dict[str, Counter] = defaultdict(Counter)

    for s in segs:
        m = s.metrics
        ovrl = m.get("dnsmos_ovrl")
        snr = m.get("snr_db")
        txt = s.transcript.strip()
        latin = latin_fraction(txt)

        has_truncation = bool(txt) and txt[-1] not in TERMINAL
        has_codemix = s.language != "en" and bool(re.search(r"[A-Za-z]{2,}", txt))
        has_noise = (ovrl is not None and ovrl < 3.0) or (snr is not None and snr < 18) \
            or (m.get("gap_energy_ratio", 0) or 0) > 0.30
        low_quality_audio = ovrl is not None and ovrl < 2.8
        emotion_low_conf = (s.emotion_confidence or 1.0) < 0.55   # the tag's own confidence
        transcript_review = m.get("transcript_clean") is False \
            or (m.get("mms_align_score") is not None and m["mms_align_score"] < 0.85)
        overlap = bool(m.get("overlap_flag", False))

        flags = []
        if has_noise: flags.append("noise")
        if has_truncation: flags.append("truncation")
        if has_codemix: flags.append("codemix")
        if emotion_low_conf: flags.append("emotion_low_confidence")
        if transcript_review: flags.append("transcript_review_needed")
        if overlap: flags.append("overlap_suspected")
        if low_quality_audio: flags.append("low_quality_audio")

        m.update({
            "has_noise": has_noise, "has_truncation": has_truncation,
            "has_codemix": has_codemix, "has_laughter": False,
            "emotion_low_confidence": emotion_low_conf,
            "transcript_review_needed": transcript_review,
            "overlap_suspected": overlap, "low_quality_audio": low_quality_audio,
            "annotation_flags": "|".join(flags),
            "annotated_text": annotated_text(s, has_truncation),
        })
        for f in flags:
            stats[s.language][f] += 1
        stats[s.language]["_clips"] += 1

    by_src: dict[str, list[Segment]] = defaultdict(list)
    for s in segs:
        by_src[s.source_id].append(s)
    for sid, group in by_src.items():
        save_segments(sid, group)

    print(f"{'flag':<26}{'EN':>8}{'TE':>8}")
    keys = ["noise", "truncation", "codemix", "emotion_low_confidence",
            "transcript_review_needed", "overlap_suspected", "low_quality_audio"]
    for k in keys:
        en, te = stats["en"][k], stats["te"][k]
        print(f"{k:<26}{en:>8}{te:>8}")
    print(f"{'(clips)':<26}{stats['en']['_clips']:>8}{stats['te']['_clips']:>8}")
    # show a couple of code-mix annotated examples
    for s in segs:
        if s.metrics.get("has_codemix"):
            print("codemix eg:", s.metrics["annotated_text"][:80]); break


if __name__ == "__main__":
    main()

"""LLM-as-judge + topic classification, one Sarvam call per clip.

For each kept clip the judge sees the transcript and the per-speaker acoustic
summary (it cannot hear audio, so this judges the transcript text and whether the
acoustics support the emotion label, not raw audio fidelity). It returns:
  topic_category : closed set, for aggregating the dataset's subject matter
  topic          : a short free-form phrase
  tts_suitable   : 0-1, is this a good TTS training clip (clean, fluent, one thought)
  transcript_clean : does the transcript read as complete, fluent, single-language
  emotion_supported: does the acoustic summary support the assigned emotion
  issue          : short note if anything is off

Writes data/manifests/score_llm_judge.json keyed by segment id.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from ttsds.config import MANIFEST_DIR, load_config
from ttsds.models import load_all_segments
from ttsds.sarvam_client import chat_json
from ttsds.tag_emotion import describe_acoustics

TOPICS = ["mythology", "folktale", "fiction", "education", "news",
          "motivation", "devotional", "conversation", "other"]

SYSTEM = (
    "You are a strict data-quality judge for a text-to-speech training set. You are given a "
    "clip's transcript and a description of its acoustics (you cannot hear the audio). Judge the "
    "TRANSCRIPT and whether the acoustics support the stated emotion. Be skeptical.\n"
    f"topic_category must be one of: {', '.join(TOPICS)}.\n"
    "Respond with ONLY a JSON object with keys: topic_category (from the list), topic (a short "
    "phrase, <=6 words), tts_suitable (number 0 to 1: is this a clean, fluent, single-language "
    "clip of one coherent thought, good to train TTS on), transcript_clean (true/false), "
    "emotion_supported (true/false: does the acoustic description fit the emotion), issue (a short "
    "phrase naming any problem, or empty string)."
)


def judge(seg, cfg) -> dict | None:
    user = (
        f'Transcript: "{seg.transcript}"\n'
        f"Stated emotion: {seg.emotion}\n"
        f"Acoustics:\n{describe_acoustics(seg)}"
    )
    r = chat_json(SYSTEM, user, model=cfg.llm.model, temperature=0.1,
                  max_tokens=4000, reasoning_effort="low")
    if not r:
        return None
    cat = str(r.get("topic_category", "other")).lower().strip()
    try:
        tts = max(0.0, min(1.0, float(r.get("tts_suitable", 0.0))))
    except (TypeError, ValueError):
        tts = 0.0
    return {
        "topic_category": cat if cat in TOPICS else "other",
        "topic": str(r.get("topic", ""))[:60],
        "tts_suitable": round(tts, 2),
        "transcript_clean": bool(r.get("transcript_clean", False)),
        "emotion_supported": bool(r.get("emotion_supported", False)),
        "issue": str(r.get("issue", ""))[:80],
        "language": seg.language,
    }


def main() -> None:
    cfg = load_config()
    segs = [s for s in load_all_segments() if s.is_kept()]
    out: dict[str, dict] = {}

    def work(s):
        return s.id, judge(s, cfg)

    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, (sid, res) in enumerate(ex.map(work, segs), 1):
            if res:
                out[sid] = res
            if i % 50 == 0:
                print(f"{i}/{len(segs)} judged", flush=True)

    (MANIFEST_DIR / "score_llm_judge.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    fails = sum(1 for v in out.values() if not v["transcript_clean"])
    emo_fail = sum(1 for v in out.values() if not v["emotion_supported"])
    low = sum(1 for v in out.values() if v["tts_suitable"] < 0.5)
    print(f"\njudged {len(out)}/{len(segs)} (failed JSON: {len(segs)-len(out)})")
    print(f"transcript flagged unclean: {fails} | emotion unsupported: {emo_fail} | tts_suitable<0.5: {low}")
    for lang in ("en", "te"):
        c = Counter(v["topic_category"] for v in out.values() if v["language"] == lang)
        print(f"{lang} topics: {dict(c.most_common())}")


if __name__ == "__main__":
    main()

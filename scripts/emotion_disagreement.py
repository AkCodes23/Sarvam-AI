"""Are sarvam-30b vs sarvam-105b emotion disagreements between NEIGHBORING classes
(ambiguity) or CONTRADICTORY ones (error)? Same-taxonomy comparison over a sample.

Neighboring = close in valence-arousal space (euclidean distance < 0.6).
Writes data/manifests/emotion_disagreement.json.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from ttsds.config import MANIFEST_DIR, load_config
from ttsds.models import load_all_segments
from ttsds.sarvam_client import chat_json
from ttsds.tag_emotion import _system_prompt, describe_acoustics, is_whisper

# valence, arousal coordinates for the 8-class taxonomy
VA = {
    "neutral": (0.0, 0.0), "calm": (0.3, -0.5), "sad": (-0.6, -0.4), "happy": (0.7, 0.4),
    "excited": (0.6, 0.8), "angry": (-0.6, 0.7), "fearful": (-0.5, 0.5), "surprised": (0.2, 0.8),
}
THRESH = 0.6
N = 80


def dist(a, b):
    (x1, y1), (x2, y2) = VA[a], VA[b]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def retag(seg, cfg, sysp):
    user = (f'Transcript: "{seg.transcript}"\n\nAcoustic prosody (relative to this speaker):\n'
            f'{describe_acoustics(seg)}\n\nWhisper indicated acoustically: '
            f'{"YES" if is_whisper(seg, cfg) else "no"}')
    r = chat_json(sysp, user, model="sarvam-105b", temperature=0.1, max_tokens=4000, reasoning_effort="low")
    if r:
        c = str(r.get("emotion", "")).lower().strip()
        return c if c in cfg.emotion.emotions else None
    return None


def main():
    cfg = load_config()
    sysp = _system_prompt(cfg)
    segs = [s for s in load_all_segments() if s.is_kept() and s.emotion in VA]
    segs = sorted(segs, key=lambda s: s.id)
    sample = segs[:: max(1, len(segs) // N)][:N]
    with ThreadPoolExecutor(max_workers=6) as ex:
        labels = list(ex.map(lambda s: retag(s, cfg, sysp), sample))

    dis = neigh = agree = fails = 0
    examples = []
    for s, l2 in zip(sample, labels):
        if l2 is None:
            fails += 1; continue
        if s.emotion == l2:
            agree += 1
        else:
            dis += 1
            n = dist(s.emotion, l2) < THRESH
            neigh += n
            if len(examples) < 6:
                examples.append({"id": s.id, "a": s.emotion, "b": l2, "neighboring": bool(n)})
    out = {"n": agree + dis, "agree": agree, "disagree": dis, "failures": fails,
           "neighboring": neigh, "contradictory": dis - neigh,
           "neighboring_pct": round(neigh / dis * 100) if dis else None,
           "examples": examples}
    (MANIFEST_DIR / "emotion_disagreement.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

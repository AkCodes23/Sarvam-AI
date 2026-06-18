"""#7 Speech-event detection (improvement experiment, not wired into the pipeline).

Runs an AudioSet classifier (AST) over the published clips to replace the
convention-only has_laughter / cough / breath flags with measured ones, and to
detect music beds. Writes data/manifests/improvements/speech_events.json.

CPU-friendly: windows each clip into <=10s hops and takes the max class
probability across windows. Honest about being automatic (a detector, not ears).
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from ttsds.config import PROJECT_ROOT
from ttsds.build_dataset import FINAL_SELECTION

OUT = PROJECT_ROOT / "data/manifests/improvements/speech_events.json"
MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000
WIN = 10 * SR
THRESH = 0.5

# AudioSet display-name groups -> our flag
GROUPS = {
    "laughter": ["Laughter", "Giggle", "Chuckle, chortle", "Snicker", "Belly laugh", "Baby laughter"],
    "cough": ["Cough", "Throat clearing"],
    "breath": ["Breathing", "Gasp", "Sniff", "Sneeze", "Pant"],
    "music": ["Music"],
}


def load_16k(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        y = resample_poly(y, SR // g, sr // g).astype("float32")
    return y


def main() -> None:
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    model = AutoModelForAudioClassification.from_pretrained(MODEL).eval()
    id2label = model.config.id2label
    name2id = {v: k for k, v in id2label.items()}
    group_ids = {g: [name2id[n] for n in names if n in name2id] for g, names in GROUPS.items()}

    recs = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))
    per_clip: dict[str, dict] = {}
    counts: dict[str, Counter] = defaultdict(Counter)
    n_done = 0
    for cfg, rs in recs.items():
        lang = "en" if cfg == "indian_english" else "te"
        for r in rs:
            sid = os.path.splitext(os.path.basename(r["audio"]))[0]
            try:
                y = load_16k(str(PROJECT_ROOT / r["audio"]))
            except Exception as e:  # noqa: BLE001
                per_clip[sid] = {"error": str(e)[:60]}
                continue
            # window into <=10s hops, take max sigmoid prob per class
            wins = [y[i:i + WIN] for i in range(0, max(1, len(y)), WIN)] or [y]
            best = np.zeros(len(id2label), dtype="float32")
            for w in wins:
                if len(w) < SR // 2:  # skip <0.5s tails
                    continue
                inp = fe(w, sampling_rate=SR, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**inp).logits[0]
                probs = torch.sigmoid(logits).numpy()
                best = np.maximum(best, probs)
            ev = {g: round(float(max((best[i] for i in ids), default=0.0)), 3) for g, ids in group_ids.items()}
            flags = {f"has_{g}": (ev[g] >= THRESH) for g in ("laughter", "cough", "breath")}
            flags["music_bed"] = ev["music"] >= THRESH
            per_clip[sid] = {"lang": lang, "probs": ev, **flags}
            for k, v in flags.items():
                if v:
                    counts[lang][k] += 1
            counts[lang]["_clips"] += 1
            n_done += 1
            if n_done % 25 == 0:
                print(f"  ...{n_done} clips")

    summary = {lang: dict(c) for lang, c in counts.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": MODEL, "threshold": THRESH,
                               "summary": summary, "per_clip": per_clip},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY:", json.dumps(summary, ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

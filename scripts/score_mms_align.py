"""MMS forced-alignment confidence — a transcript-validation signal that works for
BOTH English and Telugu (via uroman romanization). For each clip we force-align the
Sarvam transcript to the audio with torchaudio's MMS_FA aligner and take the mean
alignment probability as confidence. Low confidence => transcript/audio mismatch.

This directly addresses the Telugu transcript-validation gap (no English-only Whisper).
Writes data/manifests/score_mms_align.json and prints a stratified sample for human CER audit.
"""

from __future__ import annotations

import json
import re
import statistics as st
from collections import defaultdict

import librosa
import numpy as np
import torch
import torchaudio
from torchaudio.pipelines import MMS_FA as bundle
from uroman import Uroman

from ttsds.config import MANIFEST_DIR, PROJECT_ROOT
from ttsds.models import load_all_segments

LCODE = {"en": "eng", "te": "tel"}


def romanize(ur: Uroman, text: str, lang: str) -> str:
    try:
        rom = ur.romanize_string(text, lcode=LCODE.get(lang))
    except Exception:  # noqa: BLE001
        rom = ur.romanize_string(text)
    return re.sub(r"\s+", " ", rom.lower()).strip()


def align_confidence(model, dictionary, waveform: torch.Tensor, rom: str) -> float | None:
    # exclude index 0 (the blank token, e.g. '-') which forced_align rejects in targets
    tokens = [dictionary[c] for w in rom.split() for c in w if c in dictionary and dictionary[c] != 0]
    if not tokens:
        return None
    try:
        with torch.inference_mode():
            emission, _ = model(waveform)
        # CTC forced alignment requires the token sequence to fit in the emission
        # frames; longer-than-audio targets (or any align failure) -> skip the clip.
        if len(tokens) >= emission.size(1):
            return None
        targets = torch.tensor([tokens], dtype=torch.int32)
        _, scores = torchaudio.functional.forced_align(emission, targets, blank=0)
        return float(scores.exp().mean().item())
    except Exception:  # noqa: BLE001 - never let one clip kill the whole run
        return None


def main() -> None:
    model = bundle.get_model()
    model.eval()
    dictionary = bundle.get_dict()
    ur = Uroman()

    segs = [s for s in load_all_segments() if s.is_kept()]
    out: dict[str, dict] = {}
    by_lang: dict[str, list[float]] = defaultdict(list)
    rows = []
    skipped = 0
    for i, s in enumerate(segs, 1):
        try:
            y, _ = librosa.load(str(PROJECT_ROOT / s.wav_path), sr=bundle.sample_rate, mono=True)
            y = np.clip(y, -1.0, 1.0)
            wav = torch.from_numpy(y).float().unsqueeze(0)
            conf = align_confidence(model, dictionary, wav, romanize(ur, s.transcript, s.language))
        except Exception:  # noqa: BLE001 - skip a bad clip, keep going
            conf = None
        if conf is None:
            skipped += 1
            continue
        out[s.id] = {"mms_align_score": round(conf, 4), "language": s.language}
        by_lang[s.language].append(conf)
        rows.append((s.id, s.language, conf))
        if i % 50 == 0:
            print(f"{i}/{len(segs)} aligned", flush=True)

    (MANIFEST_DIR / "score_mms_align.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== MMS forced-alignment confidence (skipped {skipped}) ===")
    for lang, vals in by_lang.items():
        vals_s = sorted(vals)
        print(f"{lang}: n={len(vals)} median={st.median(vals):.3f} "
              f"p10={np.percentile(vals,10):.3f} p90={np.percentile(vals,90):.3f}")

    # stratified sample for HUMAN CER audit (low / mid / high alignment per language)
    print("\n=== stratified human-CER audit sample (listen to these) ===")
    sample = {}
    for lang in by_lang:
        lr = sorted([r for r in rows if r[1] == lang], key=lambda x: x[2])
        n = len(lr)
        picks = {"low": [r[0] for r in lr[:3]],
                 "mid": [r[0] for r in lr[n//2 - 1: n//2 + 2]],
                 "high": [r[0] for r in lr[-3:]]}
        sample[lang] = picks
        print(f"{lang}: {picks}")
    (MANIFEST_DIR / "mms_audit_sample.json").write_text(
        json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

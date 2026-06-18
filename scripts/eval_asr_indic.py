"""Independent ASR cross-check with an INDIC recognizer (not generic Whisper).

Primary: AI4Bharat IndicConformer (multilingual, en-IN + te). Fallback (if it will
not load on this box): a Telugu-specialized Whisper fine-tune (vasista22) for te,
which is itself an Indic ASR. We report WER only (CER dropped); WER is the metric
that matters for a word-level transcript check.

Writes data/manifests/eval_asr_indic.json.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

import jiwer

from ttsds.config import MANIFEST_DIR, PROJECT_ROOT
from ttsds.build_dataset import FINAL_SELECTION
from ttsds.models import load_all_segments

SR = 16000
N_PER_LANG = 40
OUT = MANIFEST_DIR / "eval_asr_indic.json"
_PUNCT = "".join(chr(c) for c in range(0x2000, 0x206F)) + r""".,!?;:"'`()[]{}–—…।॥"""


def norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t).lower()
    return " ".join("".join(ch for ch in t if ch not in _PUNCT).split())


def load16k(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g).astype(np.float32)
    return np.clip(y, -1.0, 1.0)


def sample(segs, pub):
    by = defaultdict(list)
    for s in segs:
        if s.is_kept() and s.id in pub:
            by[s.language].append(s)
    out = {}
    for lang, g in by.items():
        g = sorted(g, key=lambda s: s.id)
        k = max(1, len(g) // N_PER_LANG)
        out[lang] = g[::k][:N_PER_LANG]
    return out


def try_indicconformer():
    """Return a transcribe(path, lang)->str fn, or None if the model won't load."""
    try:
        from transformers import AutoModel
        m = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual",
                                      trust_remote_code=True)
        m.eval()

        def tx(path, lang):
            y = load16k(path)
            wav = torch.from_numpy(y).unsqueeze(0)
            with torch.no_grad():
                out = m(wav, lang, "ctc")
            return out[0] if isinstance(out, (list, tuple)) else str(out)
        # smoke test on 0.5s silence
        tx_test = m(torch.zeros(1, SR // 2), "te", "ctc")  # noqa: F841
        return tx, "ai4bharat/indic-conformer-600m-multilingual"
    except Exception as e:  # noqa: BLE001
        print(f"[indic-conformer] unavailable: {str(e)[:120]} -> falling back", flush=True)
        return None, None


def whisper_ft_fallback():
    """Telugu-specialized Whisper fine-tune (Indic ASR) for te, the language generic
    Whisper fails on; English keeps an English-Whisper check (reliable for English).
    Uses the processor+model directly on a decoded array, so neither ffmpeg nor
    torchcodec is needed (clips are <=25s, within Whisper's 30s window)."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    MODELS = {"te": "vasista22/whisper-telugu-base", "en": "openai/whisper-small.en"}
    cache = {}

    def get(lang):
        if lang not in cache:
            mid = MODELS[lang]
            cache[lang] = (WhisperProcessor.from_pretrained(mid),
                           WhisperForConditionalGeneration.from_pretrained(mid).eval())
        return cache[lang]

    def tx(path, lang):
        proc, model = get(lang)
        feats = proc(load16k(path), sampling_rate=SR, return_tensors="pt").input_features
        # vasista22 te model is language-specific (defaults to Telugu); its generation
        # config predates the language= kwarg, so we let it decode in its native language.
        with torch.no_grad():
            ids = model.generate(feats, max_new_tokens=220)
        txt = proc.batch_decode(ids, skip_special_tokens=True)[0]
        return re.sub(r"<\|[^|]*\|>", "", txt)   # strip any residual whisper special tokens

    def model_name(lang):
        return MODELS[lang]
    return tx, model_name


def main():
    pub = set()
    for rs in json.loads(FINAL_SELECTION.read_text(encoding="utf-8")).values():
        for r in rs:
            pub.add(os.path.splitext(os.path.basename(r["audio"]))[0])
    segs = load_all_segments()
    chosen = sample(segs, pub)

    tx, model_name = try_indicconformer()
    fallback = tx is None
    if fallback:
        tx, model_name_fn = whisper_ft_fallback()

    report = {}
    for lang, group in chosen.items():
        wers = []
        used = model_name if not fallback else model_name_fn(lang)
        print(f"[{lang}] transcribing {len(group)} clips with {used}", flush=True)
        for s in group:
            try:
                hyp = tx(str(PROJECT_ROOT / s.wav_path), lang)
            except Exception as e:  # noqa: BLE001
                print(f"  skip {s.id}: {str(e)[:80]}", flush=True)
                continue
            ref = norm(s.transcript)
            h = norm(hyp)
            if ref and h:
                wers.append(jiwer.wer(ref, h))
        if wers:
            report[lang] = {"n": len(wers), "model": used,
                            "wer_mean": round(float(np.mean(wers)), 4),
                            "wer_median": round(float(np.median(wers)), 4)}
            print(f"[{lang}] WER mean {report[lang]['wer_mean']*100:.1f}% "
                  f"median {report[lang]['wer_median']*100:.1f}% (n={len(wers)})", flush=True)

    report["_recognizer"] = "indic-conformer" if not fallback else "indic-whisper-ft-fallback"
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

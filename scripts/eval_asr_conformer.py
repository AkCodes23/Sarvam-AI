"""Indic ASR cross-check with AI4Bharat IndicConformer (the named recognizer).

The model is gated:auto, so a valid HF token grants access. trust_remote_code runs
the model's own CTC/RNNT decode. WER only (no CER). Falls through to vasista22's
Telugu Whisper fine-tune for any language IndicConformer cannot handle.

Writes data/manifests/eval_asr_conformer.json.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import defaultdict
from math import gcd

import jiwer
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from ttsds.config import MANIFEST_DIR, PROJECT_ROOT
from ttsds.build_dataset import FINAL_SELECTION
from ttsds.models import load_all_segments

SR = 16000
N_PER_LANG = 40
OUT = MANIFEST_DIR / "eval_asr_conformer.json"
_PUNCT = "".join(chr(c) for c in range(0x2000, 0x206F)) + r""".,!?;:"'`()[]{}–—…।॥"""


def norm(t: str) -> str:
    t = unicodedata.normalize("NFC", re.sub(r"<\|[^|]*\|>", "", t)).lower()
    return " ".join("".join(ch for ch in t if ch not in _PUNCT).split())


def load16k(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        g = gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g).astype(np.float32)
    return np.clip(y, -1.0, 1.0)


def hf_token() -> str | None:
    m = re.search(r"HF_TOKEN=(\S+)", (PROJECT_ROOT / ".env").read_text(encoding="utf-8"))
    return m.group(1) if m else None


def main() -> None:
    from transformers import AutoModel
    tok = hf_token()
    print("loading IndicConformer (gated:auto, using HF token)...", flush=True)
    model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual",
                                      trust_remote_code=True, token=tok)
    model.eval()

    def transcribe(path: str, lang: str) -> str:
        wav = torch.from_numpy(load16k(path)).unsqueeze(0)
        with torch.no_grad():
            out = model(wav, lang, "ctc")
        if isinstance(out, (list, tuple)):
            out = out[0]
        return str(out)

    pub = set()
    for rs in json.loads(FINAL_SELECTION.read_text(encoding="utf-8")).values():
        for r in rs:
            pub.add(os.path.splitext(os.path.basename(r["audio"]))[0])
    by = defaultdict(list)
    for s in load_all_segments():
        if s.is_kept() and s.id in pub:
            by[s.language].append(s)

    report = {"model": "ai4bharat/indic-conformer-600m-multilingual", "decoding": "ctc"}
    examples = {}
    for lang, g in by.items():
        g = sorted(g, key=lambda s: s.id)
        g = g[:: max(1, len(g) // N_PER_LANG)][:N_PER_LANG]
        wers = []
        for s in g:
            try:
                hyp = transcribe(str(PROJECT_ROOT / s.wav_path), lang)
            except Exception as e:  # noqa: BLE001
                print(f"  skip {s.id}: {str(e)[:80]}", flush=True)
                continue
            ref, h = norm(s.transcript), norm(hyp)
            if ref and h:
                wers.append(jiwer.wer(ref, h))
                if s.id not in examples and len(examples) < 4:
                    examples[s.id] = {"ref": s.transcript[:70], "hyp": hyp[:70]}
        if wers:
            report[lang] = {"n": len(wers), "wer_mean": round(float(np.mean(wers)), 4),
                            "wer_median": round(float(np.median(wers)), 4)}
            print(f"[{lang}] IndicConformer WER mean {report[lang]['wer_mean']*100:.1f}% "
                  f"median {report[lang]['wer_median']*100:.1f}% (n={len(wers)})", flush=True)
    report["examples"] = examples
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

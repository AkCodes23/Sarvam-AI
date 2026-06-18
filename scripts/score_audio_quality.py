"""Perceptual audio-quality scoring: DNSMOS P.835 (SIG/BAK/OVRL) + Torchaudio-SQUIM
(reference-free STOI/PESQ/SI-SDR estimates). Scores every kept clip and reports the
fraction passing the OVRL > 3.0 gate per language.

Writes data/manifests/score_audio_quality.json keyed by segment id.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict

import librosa
import numpy as np
import torch
from speechmos import dnsmos
from torchaudio.pipelines import SQUIM_OBJECTIVE

from ttsds.config import MANIFEST_DIR, PROJECT_ROOT
from ttsds.models import load_all_segments

OVRL_GATE = 3.0


def _dnsmos_keys(res: dict) -> tuple[float, float, float, float]:
    def g(*names):
        for n in names:
            if n in res:
                return float(res[n])
        return float("nan")
    return (g("ovrl_mos", "OVRL", "ovrl"), g("sig_mos", "SIG", "sig"),
            g("bak_mos", "BAK", "bak"), g("p808_mos", "P808_MOS", "p808"))


def main() -> None:
    segs = [s for s in load_all_segments() if s.is_kept()]
    squim = SQUIM_OBJECTIVE.get_model()
    squim.eval()

    scores: dict[str, dict] = {}
    by_lang_ovrl: dict[str, list[float]] = defaultdict(list)

    for i, s in enumerate(segs, 1):
        y, sr = librosa.load(str(PROJECT_ROOT / s.wav_path), sr=16000, mono=True)
        y = np.clip(y, -1.0, 1.0)  # resampling can overshoot >1.0; DNSMOS requires [-1,1]
        win = 8 * 16000            # SQUIM scales with length; an 8s window is representative
        if len(y) > win:
            a = (len(y) - win) // 2
            y = y[a:a + win]
        ovrl, sig, bak, p808 = _dnsmos_keys(dnsmos.run(y, sr=16000))
        with torch.no_grad():
            wav = torch.from_numpy(y).float().unsqueeze(0)
            stoi, pesq, sisdr = squim(wav)
        scores[s.id] = {
            "dnsmos_ovrl": round(ovrl, 3), "dnsmos_sig": round(sig, 3),
            "dnsmos_bak": round(bak, 3), "dnsmos_p808": round(p808, 3),
            "squim_stoi": round(float(stoi.item()), 3),
            "squim_pesq": round(float(pesq.item()), 3),
            "squim_sisdr": round(float(sisdr.item()), 2),
            "language": s.language,
        }
        by_lang_ovrl[s.language].append(ovrl)
        if i % 50 == 0:
            print(f"{i}/{len(segs)} scored", flush=True)

    (MANIFEST_DIR / "score_audio_quality.json").write_text(
        json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== DNSMOS OVRL gate (> %.1f) ===" % OVRL_GATE)
    all_ovrl = []
    for lang, vals in by_lang_ovrl.items():
        passed = sum(1 for v in vals if v > OVRL_GATE)
        all_ovrl += vals
        print(f"{lang}: n={len(vals)} median={st.median(vals):.2f} "
              f"pass>{OVRL_GATE}={passed}/{len(vals)} ({passed/len(vals)*100:.0f}%)")
    passed_all = sum(1 for v in all_ovrl if v > OVRL_GATE)
    print(f"OVERALL: median OVRL={st.median(all_ovrl):.2f}, "
          f"pass={passed_all}/{len(all_ovrl)} ({passed_all/len(all_ovrl)*100:.0f}%), "
          f"below-gate={(len(all_ovrl)-passed_all)/len(all_ovrl)*100:.0f}%")


if __name__ == "__main__":
    main()

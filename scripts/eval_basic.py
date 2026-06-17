"""Lightweight dataset-quality evaluations (no heavy ML deps):
WPM distribution, pause distribution (leading/trailing/internal), lexical diversity
(TTR), confidence histogram, and a quality-gate ablation (SNR raw->kept->selected).

Outputs figures to reports/figures/ and a summary to data/manifests/eval_basic.json.
"""

from __future__ import annotations

import json
import statistics as st
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ttsds.audio import nonsilent_intervals, read_wav  # noqa: E402
from ttsds.build_dataset import FINAL_SELECTION  # noqa: E402
from ttsds.config import FIGURES_DIR, MANIFEST_DIR, PROJECT_ROOT  # noqa: E402
from ttsds.models import load_all_segments  # noqa: E402

LANG_COLOR = {"indian_english": "#2563eb", "telugu": "#e0701a", "en": "#2563eb", "te": "#e0701a"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, name):
    fig.tight_layout(); fig.savefig(FIGURES_DIR / name, bbox_inches="tight"); plt.close(fig)
    print("wrote", name)


def main():
    records = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))
    summary = {}

    # ---- WPM, TTR, duration, confidence, SNR per config (published clips) ----
    wpm_by, dur_by, conf_by, snr_by = {}, {}, {}, {}
    for cfg_name, recs in records.items():
        wpm, durs, confs, snrs = [], [], [], []
        words, all_tokens = 0, []
        for r in recs:
            d = r["duration"]; toks = r["text"].split()
            durs.append(d); words += len(toks); all_tokens += [t.lower() for t in toks]
            if d > 0:
                wpm.append(len(toks) / d * 60)
            if r.get("emotion_confidence") is not None:
                confs.append(r["emotion_confidence"])
            if r.get("snr_db") is not None:
                snrs.append(r["snr_db"])
        wpm_by[cfg_name], dur_by[cfg_name], conf_by[cfg_name], snr_by[cfg_name] = wpm, durs, confs, snrs
        ttr = len(set(all_tokens)) / max(1, words)
        summary[cfg_name] = {
            "clips": len(recs), "total_words": words, "unique_words": len(set(all_tokens)),
            "type_token_ratio": round(ttr, 4),
            "wpm_mean": round(st.mean(wpm), 1) if wpm else 0,
            "wpm_median": round(st.median(wpm), 1) if wpm else 0,
            "duration_mean_s": round(st.mean(durs), 2) if durs else 0,
            "emotion_conf_median": round(st.median(confs), 3) if confs else None,
            "snr_db_median": round(st.median(snrs), 1) if snrs else None,
        }

    # WPM histogram
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    for k, v in wpm_by.items():
        ax.hist(v, bins=18, alpha=0.6, label=k, color=LANG_COLOR.get(k, "#888"))
    ax.set_title("Speaking rate (words/min)"); ax.set_xlabel("WPM"); ax.set_ylabel("clips"); ax.legend()
    _save(fig, "wpm_hist.png")

    # confidence histogram
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    for k, v in conf_by.items():
        if v:
            ax.hist(v, bins=18, alpha=0.6, label=k, color=LANG_COLOR.get(k, "#888"))
    ax.set_title("Emotion-tag confidence"); ax.set_xlabel("confidence"); ax.set_ylabel("clips"); ax.legend()
    _save(fig, "confidence_hist.png")

    # ---- pause distribution (measure on published clips) ----
    pauses = {"leading": [], "trailing": [], "internal_ratio": []}
    for cfg_name, recs in records.items():
        for r in recs:
            y, sr = read_wav(PROJECT_ROOT / r["audio"])
            iv = nonsilent_intervals(y, sr, top_db=35, min_gap_s=0.0)
            if not iv or len(y) == 0:
                continue
            lead = iv[0][0] / sr
            trail = (len(y) - iv[-1][1]) / sr
            speech = sum(e - s for s, e in iv)
            internal = 1.0 - speech / len(y) - lead / (len(y) / sr) - trail / (len(y) / sr)
            pauses["leading"].append(lead); pauses["trailing"].append(trail)
            pauses["internal_ratio"].append(max(0.0, internal))
    summary["pauses"] = {
        "mean_leading_s": round(float(np.mean(pauses["leading"])), 3),
        "mean_trailing_s": round(float(np.mean(pauses["trailing"])), 3),
        "mean_internal_silence_frac": round(float(np.mean(pauses["internal_ratio"])), 3),
    }
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    ax.hist(pauses["leading"], bins=20, alpha=0.6, label="leading", color="#16a34a")
    ax.hist(pauses["trailing"], bins=20, alpha=0.6, label="trailing", color="#7c3aed")
    ax.set_title("Edge-pause distribution (published clips)"); ax.set_xlabel("seconds"); ax.set_ylabel("clips"); ax.legend()
    _save(fig, "pause_dist.png")

    # ---- ablation: SNR raw(all candidates) -> kept -> selected ----
    segs = load_all_segments()
    by_lang_all, by_lang_kept = {}, {}
    for s in segs:
        by_lang_all.setdefault(s.language, []).append(s.metrics.get("snr_db"))
        if s.is_kept():
            by_lang_kept.setdefault(s.language, []).append(s.metrics.get("snr_db"))
    abl = {}
    for lang in by_lang_all:
        allv = [x for x in by_lang_all[lang] if x is not None]
        keptv = [x for x in by_lang_kept.get(lang, []) if x is not None]
        abl[lang] = {
            "candidates": len(allv), "kept": len(keptv),
            "snr_all_median": round(st.median(allv), 1) if allv else None,
            "snr_kept_median": round(st.median(keptv), 1) if keptv else None,
        }
    summary["snr_ablation"] = abl
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    langs = list(abl)
    x = range(len(langs)); w = 0.38
    ax.bar([i - w/2 for i in x], [abl[l]["snr_all_median"] for l in langs], w, label="all candidates", color="#94a3b8")
    ax.bar([i + w/2 for i in x], [abl[l]["snr_kept_median"] for l in langs], w, label="kept (post-gate)", color="#16a34a")
    ax.set_xticks(list(x)); ax.set_xticklabels(langs); ax.set_ylabel("median SNR (dB)")
    ax.set_title("Quality-gate ablation: SNR before/after filtering"); ax.legend()
    _save(fig, "snr_ablation.png")

    (MANIFEST_DIR / "eval_basic.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

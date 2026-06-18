"""Figures for the perceptual-quality / alignment / SER additions:
  dnsmos_dist.png     DNSMOS OVRL distribution per language with the 3.0 gate line
  mms_align_dist.png  MMS forced-alignment confidence per language
  vad_space.png       valence-arousal scatter colored by emotion
  agreement_bars.png  Krippendorff alpha + pairwise agreement
"""

from __future__ import annotations

import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ttsds.config import FIGURES_DIR, MANIFEST_DIR  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
LC = {"en": "#2563eb", "te": "#e0701a"}


def _load(name):
    p = MANIFEST_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _save(fig, name):
    fig.tight_layout(); fig.savefig(FIGURES_DIR / name, bbox_inches="tight"); plt.close(fig)
    print("wrote", name)


def dnsmos_dist():
    d = _load("score_audio_quality.json")
    if not d:
        return
    by = defaultdict(list)
    for v in d.values():
        by[v["language"]].append(v["dnsmos_ovrl"])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for lang, vals in by.items():
        ax.hist(vals, bins=22, alpha=0.6, label=f"{lang} (med {np.median(vals):.2f})", color=LC.get(lang))
    ax.axvline(3.0, color="#dc2626", ls="--", lw=1.5, label="OVRL=3.0 gate")
    ax.set_title("DNSMOS OVRL distribution (in-the-wild YouTube)")
    ax.set_xlabel("DNSMOS OVRL"); ax.set_ylabel("clips"); ax.legend()
    _save(fig, "dnsmos_dist.png")


def mms_dist():
    d = _load("score_mms_align.json")
    if not d:
        return
    by = defaultdict(list)
    for v in d.values():
        by[v["language"]].append(v["mms_align_score"])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for lang, vals in by.items():
        ax.hist(vals, bins=22, alpha=0.6, label=f"{lang} (med {np.median(vals):.2f})", color=LC.get(lang))
    ax.set_title("MMS forced-alignment confidence (transcript validation)")
    ax.set_xlabel("mean alignment probability"); ax.set_ylabel("clips"); ax.legend()
    _save(fig, "mms_align_dist.png")


def vad_space():
    d = _load("score_ser.json")
    if not d:
        return
    emos = sorted({v["ser_emotion"] for v in d.values()})
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for i, e in enumerate(emos):
        xs = [v["valence"] for v in d.values() if v["ser_emotion"] == e]
        ys = [v["arousal"] for v in d.values() if v["ser_emotion"] == e]
        ax.scatter(xs, ys, s=12, alpha=0.55, color=cmap(i % 10), label=e)
    ax.set_xlabel("valence"); ax.set_ylabel("arousal")
    ax.set_title("VAD emotion space (audeering wav2vec2)")
    ax.legend(fontsize=8, markerscale=1.5, loc="best")
    _save(fig, "vad_space.png")


def agreement_bars():
    d = _load("eval_agreement.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    labels = ["a 2-LLM", "a 3-rater", "30b vs e2v", "30b vs aud", "e2v vs aud"]
    pw = d["pairwise_agreement"]
    vals = [d.get("llm_inter_model_alpha") or 0, d.get("krippendorff_alpha_3raters") or 0,
            pw["30b_vs_emotion2vec"], pw["30b_vs_audeering"], pw["emotion2vec_vs_audeering"]]
    colors = ["#16a34a", "#dc2626", "#2563eb", "#94a3b8", "#94a3b8"]
    b = ax.bar(labels, vals, color=colors)
    ax.bar_label(b, fmt="%.2f", fontsize=9)
    ax.axhline(0.4, color="#16a34a", ls=":", lw=1, label="field norm α≥0.4")
    ax.axhline(0.2, color="#dc2626", ls=":", lw=1, label="α<0.2 re-examine")
    ax.set_title("Emotion-label agreement (Krippendorff α + pairwise)")
    ax.set_ylabel("alpha / agreement"); ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, "agreement_bars.png")


def main():
    dnsmos_dist(); mms_dist(); vad_space(); agreement_bars()


if __name__ == "__main__":
    main()

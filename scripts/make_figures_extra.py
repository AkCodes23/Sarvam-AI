"""Extra figures for the report: cross-ASR WER comparison and emotion distribution.
Reads the published selection and the eval JSONs; writes PNGs into reports/figures/."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ttsds.config import FIGURES_DIR, MANIFEST_DIR  # noqa: E402
from ttsds.build_dataset import FINAL_SELECTION  # noqa: E402

INK = "#14346b"
ACC = "#c0561f"


def _load(name):
    p = MANIFEST_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def wer_figure():
    asr = _load("eval_asr.json")           # generic Whisper (en small.en, te large-v3)
    indic = _load("eval_asr_indic.json")   # vasista22 Telugu FT, en small.en
    conf = _load("eval_asr_conformer.json")  # IndicConformer (if it ran)
    bars = []
    te_generic = (asr.get("te") or {}).get("wer_mean")
    if te_generic is not None:
        bars.append(("Telugu\ngeneric Whisper", te_generic * 100, "#9aa4b2"))
    te_indic = (indic.get("te") or {}).get("wer_mean")
    if te_indic is not None:
        bars.append(("Telugu\nIndic Whisper FT", te_indic * 100, ACC))
    te_conf = (conf.get("te") or {}).get("wer_mean")
    if te_conf is not None:
        bars.append(("Telugu\nIndicConformer", te_conf * 100, "#1f7a3d"))
    en = (indic.get("en") or {}).get("wer_mean") or (asr.get("en") or {}).get("wer_mean")
    if en is not None:
        bars.append(("English\nWhisper", en * 100, INK))

    labels = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    cols = [b[2] for b in bars]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    xs = np.arange(len(labels))
    ax.bar(xs, vals, color=cols, width=0.62)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=10, color="#1a1a1a")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("word error rate vs Sarvam (%)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("Cross-ASR word error: an Indic recognizer agrees far more on Telugu", fontsize=10.5, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "wer_comparison.png", dpi=150); plt.close(fig)
    print("wrote wer_comparison.png", [(l.replace(chr(10), ' '), round(v)) for l, v in zip(labels, vals)])


def emotion_figure():
    recs = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))
    order = ["neutral", "calm", "sad", "excited", "angry", "happy", "fearful", "surprised"]
    counts = {"indian_english": {}, "telugu": {}}
    for cfg, rs in recs.items():
        for r in rs:
            counts[cfg][r["emotion"]] = counts[cfg].get(r["emotion"], 0) + 1
    en = [counts["indian_english"].get(e, 0) for e in order]
    te = [counts["telugu"].get(e, 0) for e in order]
    xs = np.arange(len(order)); w = 0.4
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.bar(xs - w / 2, en, w, label="Indian English", color=INK)
    ax.bar(xs + w / 2, te, w, label="Telugu", color=ACC)
    ax.set_xticks(xs); ax.set_xticklabels(order, fontsize=9, rotation=20)
    ax.set_ylabel("clips")
    ax.set_title("Emotion distribution per language (common ones capped, rare ones kept)", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "emotion_dist.png", dpi=150); plt.close(fig)
    print("wrote emotion_dist.png  en=", en, "te=", te)


if __name__ == "__main__":
    wer_figure()
    emotion_figure()

"""TTS-readiness analysis on the published set: duration, transcript length, speech
rate, lexical diversity (vocab, top words, Zipf), and phoneme coverage detail.

Figures -> reports/figures/; stats -> data/manifests/tts_readiness.json.
"""

from __future__ import annotations

import json
import re
import statistics as st
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ttsds.build_dataset import FINAL_SELECTION  # noqa: E402
from ttsds.config import FIGURES_DIR, MANIFEST_DIR  # noqa: E402

LANG = {"indian_english": ("en", "#2563eb"), "telugu": ("te", "#e0701a")}
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, name):
    fig.tight_layout(); fig.savefig(FIGURES_DIR / name, bbox_inches="tight"); plt.close(fig)
    print("wrote", name)


def main():
    recs = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))
    summary = {}

    durs, words, chars, wpm = {}, {}, {}, {}
    for cfg, rs in recs.items():
        lang = LANG[cfg][0]
        durs[lang] = [r["duration"] for r in rs]
        words[lang] = [len(r["text"].split()) for r in rs]
        chars[lang] = [len(r["text"]) for r in rs]
        wpm[lang] = [len(r["text"].split()) / r["duration"] * 60 for r in rs if r["duration"] > 0]

    # --- duration histogram ---
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    for cfg, (lang, color) in LANG.items():
        ax.hist(durs[lang], bins=18, alpha=0.6, label=f"{lang} (median {st.median(durs[lang]):.1f}s)", color=color)
    ax.axvline(3, color="#dc2626", ls=":", lw=1); ax.axvline(25, color="#dc2626", ls=":", lw=1, label="3-25s bounds")
    ax.set_title("Clip duration distribution"); ax.set_xlabel("seconds"); ax.set_ylabel("clips"); ax.legend(fontsize=8)
    _save(fig, "tts_duration.png")

    # --- transcript length (words) ---
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    for cfg, (lang, color) in LANG.items():
        ax.hist(words[lang], bins=18, alpha=0.6, label=f"{lang} (median {int(st.median(words[lang]))} words)", color=color)
    ax.set_title("Transcript length (words per clip)"); ax.set_xlabel("words"); ax.set_ylabel("clips"); ax.legend(fontsize=8)
    _save(fig, "tts_transcript_len.png")

    # --- speech rate, with slow/medium/fast bands ---
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    for cfg, (lang, color) in LANG.items():
        ax.hist(wpm[lang], bins=18, alpha=0.6, label=f"{lang} (median {int(st.median(wpm[lang]))} wpm)", color=color)
    ax.axvline(110, color="#16a34a", ls=":", lw=1); ax.axvline(160, color="#16a34a", ls=":", lw=1, label="slow|medium|fast")
    ax.set_title("Speech rate distribution (words/min)"); ax.set_xlabel("wpm"); ax.set_ylabel("clips"); ax.legend(fontsize=8)
    _save(fig, "tts_speech_rate.png")

    # --- lexical diversity: vocab, top-20, Zipf ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for cfg, (lang, color) in LANG.items():
        toks = [w.lower().strip(".,!?;:\"'") for r in recs[cfg] for w in r["text"].split()]
        toks = [t for t in toks if t]
        freq = Counter(toks)
        ranks = np.arange(1, len(freq) + 1)
        counts = np.array(sorted(freq.values(), reverse=True))
        axes[0].loglog(ranks, counts, marker=".", ls="none", ms=3, alpha=0.6, label=lang, color=color)
        summary.setdefault(lang, {})
        summary[lang].update({
            "clips": len(recs[cfg]),
            "duration_s": {"min": round(min(durs[lang]), 1), "median": round(st.median(durs[lang]), 1), "max": round(max(durs[lang]), 1)},
            "words_per_clip_median": int(st.median(words[lang])),
            "chars_per_clip_median": int(st.median(chars[lang])),
            "wpm_median": int(st.median(wpm[lang])),
            "wpm_bands": {
                "slow_<110": round(sum(w < 110 for w in wpm[lang]) / len(wpm[lang]) * 100),
                "medium_110_160": round(sum(110 <= w <= 160 for w in wpm[lang]) / len(wpm[lang]) * 100),
                "fast_>160": round(sum(w > 160 for w in wpm[lang]) / len(wpm[lang]) * 100),
            },
            "total_tokens": len(toks),
            "vocab_size": len(freq),
            "type_token_ratio": round(len(freq) / max(1, len(toks)), 3),
            "top20_words": [w for w, _ in freq.most_common(20)],
        })
    axes[0].set_title("Word frequency (Zipf, log-log)"); axes[0].set_xlabel("rank"); axes[0].set_ylabel("frequency"); axes[0].legend(fontsize=8)
    # vocab bar
    langs = [LANG[c][0] for c in recs]
    axes[1].bar(langs, [summary[l]["vocab_size"] for l in langs], color=[LANG[c][1] for c in recs])
    for i, l in enumerate(langs):
        axes[1].text(i, summary[l]["vocab_size"], str(summary[l]["vocab_size"]), ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Vocabulary size (unique words)"); axes[1].set_ylabel("unique words")
    _save(fig, "tts_lexical.png")

    # --- phoneme coverage detail (reuse eval_phoneme if present) ---
    pho = MANIFEST_DIR / "eval_phoneme.json"
    if pho.exists():
        summary["phoneme"] = json.loads(pho.read_text(encoding="utf-8"))

    (MANIFEST_DIR / "tts_readiness.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({l: {k: summary[l][k] for k in ("duration_s", "words_per_clip_median", "wpm_median", "wpm_bands", "vocab_size", "type_token_ratio")} for l in ("en", "te")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

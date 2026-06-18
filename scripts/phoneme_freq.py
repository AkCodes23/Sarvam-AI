"""Phoneme frequency detail: counts per phoneme, rarest phonemes, and a frequency
figure. English via g2p_en (ARPAbet), Telugu via epitran + panphon (IPA).

Figure -> reports/figures/phoneme_freq.png; detail -> data/manifests/phoneme_freq.json.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ttsds.build_dataset import FINAL_SELECTION  # noqa: E402
from ttsds.config import FIGURES_DIR, MANIFEST_DIR  # noqa: E402


def english_phonemes(texts):
    from g2p_en import G2p
    g2p = G2p()
    c = Counter()
    for t in texts:
        for p in g2p(t):
            p = re.sub(r"\d", "", p).strip()
            if p and p.isalpha():
                c[p] += 1
    return c


def telugu_phonemes(texts):
    import epitran
    import panphon
    epi = epitran.Epitran("tel-Telu")
    ft = panphon.FeatureTable()
    c = Counter()
    for t in texts:
        ipa = epi.transliterate(t)
        for seg in ft.ipa_segs(ipa):
            c[seg] += 1
    return c


def main():
    recs = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))
    en_txt = [r["text"] for r in recs.get("indian_english", [])]
    te_txt = [r["text"] for r in recs.get("telugu", [])]

    en = english_phonemes(en_txt)
    te = telugu_phonemes(te_txt)

    out = {}
    for lang, c in (("en", en), ("te", te)):
        total = sum(c.values()) or 1
        rarest = c.most_common()[-8:]
        out[lang] = {
            "unique": len(c),
            "total_tokens": total,
            "rarest_8": [{"phoneme": p, "count": n, "pct": round(n / total * 100, 2)} for p, n in rarest],
            "top_8": [{"phoneme": p, "count": n} for p, n in c.most_common(8)],
        }

    (MANIFEST_DIR / "phoneme_freq.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (lang, c, color) in zip(axes, [("English", en, "#2563eb"), ("Telugu", te, "#e0701a")]):
        items = c.most_common()
        ax.bar(range(len(items)), [n for _, n in items], color=color)
        ax.set_title(f"{lang} phoneme frequency ({len(items)} phonemes)")
        ax.set_xlabel("phoneme (sorted by frequency)"); ax.set_ylabel("count")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "phoneme_freq.png", bbox_inches="tight"); plt.close(fig)
    print("wrote phoneme_freq.png")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

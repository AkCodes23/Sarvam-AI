"""Phoneme-coverage evaluation for the TTS dataset.

English: g2p_en -> ARPAbet phonemes (stress digits stripped); coverage vs ~39.
Telugu: epitran tel-Telu -> IPA, segmented with panphon; coverage vs ~50.

Outputs reports/figures/phoneme_coverage.png and data/manifests/eval_phoneme.json.
"""

from __future__ import annotations

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ttsds.build_dataset import FINAL_SELECTION  # noqa: E402
from ttsds.config import FIGURES_DIR, MANIFEST_DIR  # noqa: E402

EN_INVENTORY = 39  # ARPAbet phonemes
TE_INVENTORY = 50  # approximate Telugu inventory
STRESS_RE = re.compile(r"\d+$")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})


def english_phonemes(texts: list[str]) -> set[str]:
    from g2p_en import G2p

    g2p = G2p()
    phones: set[str] = set()
    for text in texts:
        for tok in g2p(text):
            base = STRESS_RE.sub("", tok)
            if base.isalpha():  # keep ARPAbet phonemes, drop punctuation/spaces
                phones.add(base)
    return phones


def telugu_phonemes(texts: list[str]) -> tuple[set[str], str | None]:
    try:
        import epitran
        import panphon

        epi = epitran.Epitran("tel-Telu")
        ft = panphon.FeatureTable()
        phones: set[str] = set()
        for text in texts:
            phones.update(ft.ipa_segs(epi.transliterate(text)))
        return phones, None
    except Exception as exc:  # noqa: BLE001 - report cleanly, do not crash
        return set(), f"{type(exc).__name__}: {exc}"


def main() -> None:
    records = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))
    en_texts = [r["text"] for r in records["indian_english"]]
    te_texts = [r["text"] for r in records["telugu"]]

    en_set = english_phonemes(en_texts)
    te_set, te_err = telugu_phonemes(te_texts)

    en_cov = min(1.0, len(en_set) / EN_INVENTORY)
    te_cov = min(1.0, len(te_set) / TE_INVENTORY)

    summary = {
        "en": {"unique_phonemes": len(en_set), "coverage": round(en_cov, 4), "phonemes": sorted(en_set)},
        "te": {"unique_phonemes": len(te_set), "coverage": round(te_cov, 4), "phonemes": sorted(te_set)},
    }
    if te_err:
        summary["te"]["error"] = te_err

    # bar figure: unique-phoneme count per language
    labels = ["English", "Telugu"]
    counts = [len(en_set), len(te_set)]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bars = ax.bar(labels, counts, color=["#2563eb", "#e0701a"])
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(n),
                ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("distinct phonemes")
    ax.set_ylim(0, max(counts) * 1.18 if counts else 1)
    ax.set_title("Distinct phonemes covered")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phoneme_coverage.png", bbox_inches="tight")
    plt.close(fig)

    (MANIFEST_DIR / "eval_phoneme.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"English: {len(en_set)} unique phonemes, coverage {en_cov:.3f} (vs {EN_INVENTORY})")
    if te_err:
        print(f"Telugu: ERROR -> {te_err}")
    else:
        print(f"Telugu: {len(te_set)} unique phonemes, coverage {te_cov:.3f} (vs {TE_INVENTORY})")
    print("wrote reports/figures/phoneme_coverage.png and data/manifests/eval_phoneme.json")


if __name__ == "__main__":
    main()

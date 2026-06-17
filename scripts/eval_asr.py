"""Transcript-reliability eval via cross-ASR agreement (Sarvam vs Whisper).

We validate Sarvam ASR transcripts against an INDEPENDENT ASR (faster-whisper)
on a deterministic subset per language. Low WER/CER divergence => high transcript
reliability. This is an inter-ASR agreement proxy, NOT human ground truth; note
that Whisper itself is weaker in Telugu, so Telugu divergence is an upper bound.

Outputs reports/figures/asr_agreement.png and data/manifests/eval_asr.json.
Slow on CPU (15-25 min) by design; let it run.
"""

from __future__ import annotations

import json
import re
import statistics as st
import unicodedata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import jiwer  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

from ttsds.config import FIGURES_DIR, MANIFEST_DIR, PROJECT_ROOT  # noqa: E402

FINAL_SELECTION = MANIFEST_DIR / "final_selection.json"
SAMPLE = {"indian_english": ("en", 40), "telugu": ("te", 25)}
# (config_key) -> (faster-whisper model id); large-v3 ~3GB download once.
WHISPER_MODEL = {"en": "small.en", "te": "large-v3"}
EXTRA_PUNCT = "।॥‌‍“”‘’–—…​‌‍"  # Telugu danda + zero-width/quotes


def normalize(text: str) -> str:
    """Lowercase, strip ASCII+Telugu punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFC", text).lower()
    out = []
    for ch in text:
        if ch in EXTRA_PUNCT:
            continue
        cat = unicodedata.category(ch)
        out.append(" " if cat.startswith("P") or cat.startswith("S") else ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def sample_records(recs: list[dict], n: int) -> list[dict]:
    """Sort by audio path, take every k-th to get ~n deterministic clips."""
    ordered = sorted(recs, key=lambda r: r["audio"])
    k = max(1, len(ordered) // n)
    return ordered[::k][:n]


def transcribe(model: WhisperModel, path: str, lang: str) -> str:
    segments, _ = model.transcribe(path, language=lang, beam_size=1)
    return " ".join(seg.text for seg in segments)


def eval_language(cfg_key: str, lang: str, n: int, recs: list[dict]) -> dict:
    model_id = WHISPER_MODEL[lang]
    print(f"[{lang}] loading whisper '{model_id}' (cpu/int8) ...", flush=True)
    model = WhisperModel(model_id, device="cpu", compute_type="int8")
    subset = sample_records(recs, n)
    wers, cers, skipped = [], [], 0
    for i, r in enumerate(subset, 1):
        try:
            ref = normalize(r["text"])
            hyp = normalize(transcribe(model, str(PROJECT_ROOT / r["audio"]), lang))
            if not ref:
                skipped += 1
                continue
            wers.append(jiwer.wer(ref, hyp))
            cers.append(jiwer.cer(ref, hyp))
        except Exception as e:  # noqa: BLE001 - robustness: skip a bad clip
            skipped += 1
            print(f"[{lang}] skip {r['audio']}: {e}", flush=True)
        if i % 10 == 0 or i == len(subset):
            print(f"[{lang}] {i}/{len(subset)} done (skipped={skipped})", flush=True)
    return {
        "n": len(wers),
        "skipped": skipped,
        "whisper_model": model_id,
        "wer_mean": round(st.mean(wers), 4) if wers else None,
        "wer_median": round(st.median(wers), 4) if wers else None,
        "cer_mean": round(st.mean(cers), 4) if cers else None,
        "cer_median": round(st.median(cers), 4) if cers else None,
    }


def make_figure(results: dict) -> None:
    langs = list(results)
    x = range(len(langs))
    w = 0.38
    wer = [results[l]["wer_mean"] * 100 for l in langs]
    cer = [results[l]["cer_mean"] * 100 for l in langs]
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=130)
    b1 = ax.bar([i - w / 2 for i in x], wer, w, label="mean WER", color="#2563eb")
    b2 = ax.bar([i + w / 2 for i in x], cer, w, label="mean CER", color="#e0701a")
    ax.bar_label(b1, fmt="%.1f%%", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.1f%%", padding=2, fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{l} (n={results[l]['n']})" for l in langs])
    ax.set_ylabel("divergence (%)")
    ax.set_title("Cross-ASR agreement (Sarvam vs Whisper)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "asr_agreement.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote asr_agreement.png", flush=True)


def main() -> None:
    records = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))
    results = {}
    for cfg_key, (lang, n) in SAMPLE.items():
        results[lang] = eval_language(cfg_key, lang, n, records[cfg_key])
    make_figure(results)
    (MANIFEST_DIR / "eval_asr.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

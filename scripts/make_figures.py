"""Generate report figures from the manifests + dataset stats.

Outputs PNGs to reports/figures/:
  funnel.png            candidate -> kept reject-reason breakdown
  duration_hist.png     kept-clip duration distribution
  emotion_dist.png      emotion counts per language (final selection)
  style_dist.png        style counts per language (final selection)
  snr_hist.png          SNR distribution of kept clips
  speaker_minutes.png   minutes per speaker
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ttsds.config import FIGURES_DIR, MANIFEST_DIR  # noqa: E402
from ttsds.models import load_all_segments  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
LANG_COLOR = {"en": "#2563eb", "te": "#e0701a"}


def _save(fig, name: str):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES_DIR / name)


def funnel(segs):
    rc = collections.Counter()
    for s in segs:
        if s.status == "reject":
            for r in s.reject_reasons:
                rc[r.split("(")[0]] += 1
    kept = sum(1 for s in segs if s.status != "reject")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = ["candidates", "kept"] + list(rc)
    vals = [len(segs), kept] + [rc[k] for k in rc]
    colors = ["#64748b", "#16a34a"] + ["#dc2626"] * len(rc)
    ax.bar(labels, vals, color=colors)
    ax.set_title("Pipeline funnel: candidates -> kept, with reject reasons")
    ax.set_ylabel("segments")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, "funnel.png")


def duration_hist(segs):
    durs = [s.duration_s for s in segs if s.is_kept()]
    if not durs:
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(durs, bins=20, color="#2563eb", alpha=0.8)
    ax.set_title(f"Kept clip durations (n={len(durs)})")
    ax.set_xlabel("seconds"); ax.set_ylabel("clips")
    _save(fig, "duration_hist.png")


def _grouped(segs, attr, name):
    data: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for s in segs:
        if s.is_kept():
            data[s.language][getattr(s, attr) or "?"] += 1
    cats = sorted({c for d in data.values() for c in d})
    if not cats:
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = range(len(cats))
    width = 0.38
    for i, lang in enumerate(sorted(data)):
        ax.bar([xi + i * width for xi in x], [data[lang][c] for c in cats],
               width, label=lang, color=LANG_COLOR.get(lang, "#888"))
    ax.set_xticks([xi + width / 2 for xi in x]); ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_title(f"{name} distribution (kept)"); ax.set_ylabel("clips"); ax.legend()
    _save(fig, f"{attr}_dist.png")


def snr_hist(segs):
    snrs = [s.metrics.get("snr_db") for s in segs if s.is_kept() and s.metrics.get("snr_db")]
    if not snrs:
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(snrs, bins=20, color="#16a34a", alpha=0.8)
    ax.set_title(f"SNR of kept clips (n={len(snrs)})")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("clips")
    _save(fig, "snr_hist.png")


def speaker_minutes(segs):
    mins: dict[str, float] = collections.defaultdict(float)
    for s in segs:
        if s.is_kept():
            mins[s.speaker_id] += s.duration_s / 60.0
    if not mins:
        return
    items = sorted(mins.items(), key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(items))))
    ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color="#7c3aed")
    ax.set_title("Minutes per speaker (kept)"); ax.set_xlabel("minutes")
    _save(fig, "speaker_minutes.png")


def main():
    segs = load_all_segments()
    if not segs:
        print("no segments yet"); return
    funnel(segs)
    duration_hist(segs)
    _grouped(segs, "emotion", "Emotion")
    _grouped(segs, "style", "Style")
    snr_hist(segs)
    speaker_minutes(segs)
    stats = MANIFEST_DIR / "dataset_stats.json"
    if stats.exists():
        print("stats:", json.dumps(json.loads(stats.read_text()), indent=2)[:400])


if __name__ == "__main__":
    main()

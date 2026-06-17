"""Generate three polished report figures from the dataset manifests.

Outputs PNGs to reports/figures/:
  sankey_funnel.png       candidate -> kept/rejected -> selected/surplus -> per-language
  quality_radar.png       normalized quality radar (English vs Telugu)
  source_contribution.png minutes contributed per source, grouped by language

Run:
  PYTHONUTF8=1 .venv/Scripts/python.exe scripts/make_figures2.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths and shared style
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EN_COLOR = "#2563eb"  # blue  -> English
TE_COLOR = "#e0701a"  # orange -> Telugu
REJECT_COLOR = "#dc2626"
KEPT_COLOR = "#16a34a"
SURPLUS_COLOR = "#94a3b8"
CAND_COLOR = "#475569"
TEXT_COLOR = "#1e293b"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _load(name: str) -> dict[str, Any]:
    """Load a manifest JSON file, returning {} if it is missing."""
    path = MANIFEST_DIR / name
    if not path.exists():
        print(f"  warning: missing manifest {path}")
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _get(d: dict, *keys, default=None):
    """Defensive nested lookup: _get(d, 'a', 'b') -> d['a']['b'] or default."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# --------------------------------------------------------------------------- #
# 1. Sankey funnel
# --------------------------------------------------------------------------- #
def _funnel_counts(sources: dict, stats: dict) -> dict[str, int]:
    funnel = _get(sources, "funnel", default={})
    rejection = _get(sources, "rejection", default={})
    candidates = int(funnel.get("candidates", 457))
    kept = int(funnel.get("kept", 445))
    low_snr = int(rejection.get("low_snr", candidates - kept))

    en_sel = int(_get(stats, "per_language", "en", "clips", default=142))
    te_sel = int(_get(stats, "per_language", "te", "clips", default=140))
    selected = en_sel + te_sel
    surplus = max(kept - selected, 0)
    return {
        "candidates": candidates,
        "kept": kept,
        "low_snr": low_snr,
        "selected": selected,
        "surplus": surplus,
        "en_sel": en_sel,
        "te_sel": te_sel,
    }


def sankey_funnel(sources: dict, stats: dict) -> str:
    """Render the selection funnel as a Sankey. Returns 'plotly' or 'matplotlib'."""
    c = _funnel_counts(sources, stats)
    out = FIGURES_DIR / "sankey_funnel.png"

    try:
        import plotly.graph_objects as go  # noqa: E402

        labels = [
            f"Candidates ({c['candidates']})",          # 0
            f"Kept ({c['kept']})",                       # 1
            f"Rejected: low SNR ({c['low_snr']})",       # 2
            f"Selected ({c['selected']})",               # 3
            f"Surplus ({c['surplus']})",                 # 4
            f"English ({c['en_sel']})",                  # 5
            f"Telugu ({c['te_sel']})",                   # 6
        ]
        node_colors = [
            CAND_COLOR, KEPT_COLOR, REJECT_COLOR,
            "#0d9488", SURPLUS_COLOR, EN_COLOR, TE_COLOR,
        ]
        src = [0, 0, 1, 1, 3, 3]
        dst = [1, 2, 3, 4, 5, 6]
        val = [c["kept"], c["low_snr"], c["selected"], c["surplus"], c["en_sel"], c["te_sel"]]
        link_colors = [
            "rgba(22,163,74,0.40)", "rgba(220,38,38,0.45)",
            "rgba(13,148,136,0.40)", "rgba(148,163,184,0.40)",
            "rgba(37,99,235,0.40)", "rgba(224,112,26,0.40)",
        ]

        fig = go.Figure(
            go.Sankey(
                arrangement="snap",
                node=dict(
                    label=labels,
                    color=node_colors,
                    pad=24,
                    thickness=22,
                    line=dict(color="white", width=1.5),
                ),
                link=dict(source=src, target=dst, value=val, color=link_colors),
            )
        )
        fig.update_layout(
            title=dict(
                text="<b>Selection funnel</b>  —  candidates to per-language dataset",
                font=dict(size=20, color=TEXT_COLOR),
                x=0.02,
            ),
            font=dict(size=14, color=TEXT_COLOR, family="DejaVu Sans"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=20, r=20, t=70, b=20),
        )
        fig.write_image(str(out), scale=2, width=900, height=500)
        print("wrote", out, "(plotly)")
        return "plotly"
    except Exception as exc:  # noqa: BLE001 - fall back on any kaleido/plotly failure
        print(f"  plotly/kaleido export failed ({exc!r}); using matplotlib fallback")
        return _sankey_fallback_mpl(c, out)


def _stage_bar(ax, x, segments, bar_w, total):
    """Draw one vertical stacked stage at center-x; return list of (label,val,y0,y1,color)."""
    placed = []
    y = 0.0
    for label, val, color in segments:
        h = val / total
        ax.add_patch(
            plt.Rectangle(
                (x - bar_w / 2, y), bar_w, h,
                facecolor=color, edgecolor="white", linewidth=1.2,
            )
        )
        placed.append((label, val, y, y + h, color))
        y += h
    return placed


def _sankey_fallback_mpl(c: dict, out: Path) -> str:
    """Clean stepped horizontal funnel when kaleido is unavailable."""
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    total = c["candidates"]
    bar_w = 0.12
    xs = [0.06, 0.38, 0.70]

    s0 = _stage_bar(ax, xs[0], [("Candidates", c["candidates"], CAND_COLOR)], bar_w, total)
    s1 = _stage_bar(
        ax, xs[1],
        [("Kept", c["kept"], KEPT_COLOR), ("Rejected: low SNR", c["low_snr"], REJECT_COLOR)],
        bar_w, total,
    )
    s2 = _stage_bar(
        ax, xs[2],
        [("English", c["en_sel"], EN_COLOR), ("Telugu", c["te_sel"], TE_COLOR),
         ("Surplus", c["surplus"], SURPLUS_COLOR)],
        bar_w, total,
    )

    def ribbon(x0, y0a, y0b, x1, y1a, y1b, color):
        verts_x = np.linspace(x0, x1, 60)
        t = (verts_x - x0) / (x1 - x0)
        s = t * t * (3 - 2 * t)  # smoothstep
        top = y0a + (y1a - y0a) * s
        bot = y0b + (y1b - y0b) * s
        ax.fill_between(verts_x, bot, top, color=color, alpha=0.28, linewidth=0)

    # Candidates -> Kept (+ rejected)
    ribbon(xs[0] + bar_w / 2, s0[0][2], s0[0][2] + c["kept"] / total,
           xs[1] - bar_w / 2, s1[0][2], s1[0][3], KEPT_COLOR)
    ribbon(xs[0] + bar_w / 2, s0[0][3] - c["low_snr"] / total, s0[0][3],
           xs[1] - bar_w / 2, s1[1][2], s1[1][3], REJECT_COLOR)
    # Kept -> English / Telugu / Surplus
    kept_y = s1[0][2]
    for (lbl, val, y0, y1, color) in s2:
        ribbon(xs[1] + bar_w / 2, kept_y, kept_y + val / total,
               xs[2] - bar_w / 2, y0, y1, color)
        kept_y += val / total

    for stage in (s0, s1, s2):
        for (label, val, y0, y1, color) in stage:
            ax.text(
                (xs[[s0, s1, s2].index(stage)]) , (y0 + y1) / 2,
                f"{label}\n{val}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white",
            )

    ax.set_xlim(0, 0.82)
    ax.set_ylim(-0.03, 1.03)
    ax.axis("off")
    ax.set_title("Selection funnel  —  candidates to per-language dataset",
                 fontsize=15, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, "(matplotlib fallback)")
    return "matplotlib"


# --------------------------------------------------------------------------- #
# 2. Quality radar
# --------------------------------------------------------------------------- #
def _emotion_balance(emotions: dict | None) -> float:
    """1 - normalized stdev of emotion counts (1.0 = perfectly even). Clipped 0-1."""
    if not emotions:
        return 0.0
    vals = np.array(list(emotions.values()), dtype=float)
    if vals.sum() == 0 or len(vals) < 2:
        return 0.0
    p = vals / vals.sum()
    n = len(p)
    std = float(np.std(p))
    # max stdev for a distribution over n bins (all mass in one bin)
    max_std = math.sqrt((1.0 - 1.0 / n) / n)
    if max_std == 0:
        return 1.0
    return float(np.clip(1.0 - std / max_std, 0.0, 1.0))


def quality_radar(basic: dict, phoneme: dict, speaker: dict, asr: dict, stats: dict) -> None:
    axes_labels = [
        "SNR",
        "Phoneme\ncoverage",
        "Lexical\ndiversity",
        "Emotion\nbalance",
        "Speaker\nconsistency",
        "Transcript\nreliability",
    ]

    sep = float(_get(speaker, "separation", default=0.5243))
    speaker_score = float(np.clip(sep / 0.6, 0.0, 1.0))

    en_snr = float(_get(basic, "indian_english", "snr_db_median", default=28.9))
    te_snr = float(_get(basic, "telugu", "snr_db_median", default=28.0))
    en_ttr = float(_get(basic, "indian_english", "type_token_ratio", default=0.3339))
    te_ttr = float(_get(basic, "telugu", "type_token_ratio", default=0.5906))

    en_emos = _get(stats, "per_language", "en", "emotions", default={})
    te_emos = _get(stats, "per_language", "te", "emotions", default={})

    en_wer = _get(asr, "en", "wer_mean", default=None)
    en_transcript = float(np.clip(1.0 - en_wer, 0.0, 1.0)) if en_wer is not None else None

    en_vals = [
        float(np.clip(en_snr / 40.0, 0, 1)),
        float(np.clip(_get(phoneme, "en", "coverage", default=1.0), 0, 1)),
        float(np.clip(en_ttr / 0.6, 0, 1)),
        _emotion_balance(en_emos),
        speaker_score,
        en_transcript if en_transcript is not None else 0.0,
    ]
    te_vals = [
        float(np.clip(te_snr / 40.0, 0, 1)),
        float(np.clip(_get(phoneme, "te", "coverage", default=0.9), 0, 1)),
        float(np.clip(te_ttr / 0.6, 0, 1)),
        _emotion_balance(te_emos),
        speaker_score,
        None,  # Telugu has no comparable cross-ASR WER -> plotted as n/a
    ]

    n = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(7.6, 7.4), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_facecolor("white")

    ax.set_xticks(angles)
    ax.set_xticklabels(axes_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#64748b")
    ax.set_rlabel_position(180 / n)
    ax.grid(color="#cbd5e1", alpha=0.7, linewidth=0.8)
    ax.spines["polar"].set_color("#cbd5e1")

    def _plot(vals, color, label):
        # Replace None (n/a) with 0 for the line, but remember which to drop.
        filled = [v if v is not None else 0.0 for v in vals]
        closed = filled + filled[:1]
        ax.plot(angles_closed, closed, color=color, linewidth=2.4, label=label, zorder=4)
        ax.fill(angles_closed, closed, color=color, alpha=0.18, zorder=3)
        for ang, v in zip(angles, vals):
            if v is None:
                continue
            ax.scatter([ang], [v], color=color, s=34, zorder=5, edgecolors="white", linewidths=0.8)

    _plot(en_vals, EN_COLOR, "Indian English")
    _plot(te_vals, TE_COLOR, "Telugu")

    # Annotate Telugu transcript-reliability as n/a (single-ASR caveat).
    transcript_angle = angles[5]
    ax.annotate(
        "Telugu: n/a\n(see report)",
        xy=(transcript_angle, 0.06),
        xytext=(transcript_angle, 0.46),
        ha="center", va="center", fontsize=8.5, color=TE_COLOR, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=TE_COLOR, alpha=0.9, lw=1.0),
    )

    ax.set_title(
        "Dataset quality profile\n(normalized 0–1, higher = better)",
        fontsize=14, fontweight="bold", pad=34,
    )
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2,
        frameon=True, fontsize=11,
    )

    note = (
        "Transcript reliability = 1 − Whisper WER (English only).\n"
        "Telugu cross-ASR WER is not comparable, so that axis is marked n/a."
    )
    fig.text(0.5, -0.06, note, ha="center", fontsize=8.5, color="#64748b")

    out = FIGURES_DIR / "quality_radar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# 3. Source contribution
# --------------------------------------------------------------------------- #
def source_contribution(sources: dict) -> None:
    srcs = _get(sources, "sources", default=[])
    if not srcs:
        print("  warning: no sources found; skipping source_contribution")
        return

    # Sort by language (en first), then by minutes descending within language.
    srcs = sorted(
        srcs,
        key=lambda s: (0 if s.get("language") == "en" else 1, -float(s.get("minutes", 0.0))),
    )

    labels = [s.get("source_id", "?") for s in srcs]
    minutes = [float(s.get("minutes", 0.0)) for s in srcs]
    langs = [s.get("language", "?") for s in srcs]
    ctypes = [s.get("content_type", "") for s in srcs]
    colors = [EN_COLOR if lg == "en" else TE_COLOR for lg in langs]

    y = np.arange(len(srcs))[::-1]  # top item first

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    bars = ax.barh(y, minutes, color=colors, edgecolor="white", height=0.74)

    max_min = max(minutes) if minutes else 1.0
    for yi, val, ct in zip(y, minutes, ctypes):
        ax.text(val + max_min * 0.012, yi, f"{val:.1f} min",
                va="center", ha="left", fontsize=9, color=TEXT_COLOR, fontweight="bold")
        if ct:
            ax.text(max_min * 0.015, yi, ct, va="center", ha="left",
                    fontsize=8.5, color="white", fontstyle="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Minutes of audio")
    ax.set_xlim(0, max_min * 1.16)
    ax.set_title("Source contribution by language  (minutes per source)",
                 fontsize=14, fontweight="bold", loc="left", pad=12)
    ax.grid(axis="x", color="#e2e8f0", alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    en_total = sum(m for m, lg in zip(minutes, langs) if lg == "en")
    te_total = sum(m for m, lg in zip(minutes, langs) if lg == "te")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=EN_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=TE_COLOR),
    ]
    ax.legend(
        handles,
        [f"English ({en_total:.0f} min)", f"Telugu ({te_total:.0f} min)"],
        loc="lower right", frameon=True, fontsize=10, title="Language",
    )

    fig.tight_layout()
    out = FIGURES_DIR / "source_contribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    sources = _load("eval_sources.json")
    basic = _load("eval_basic.json")
    speaker = _load("eval_speaker.json")
    asr = _load("eval_asr.json")
    phoneme = _load("eval_phoneme.json")
    stats = _load("dataset_stats.json")

    sankey_mode = sankey_funnel(sources, stats)
    quality_radar(basic, phoneme, speaker, asr, stats)
    source_contribution(sources)

    print(f"\nSankey rendered via: {sankey_mode}")


if __name__ == "__main__":
    main()

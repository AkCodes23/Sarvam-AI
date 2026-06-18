"""Deeper dataset analyses for the report (no API, runs off published data + audio):
  1. split integrity (clip overlap, speaker coverage, emotion balance, transcript dup)
  2. per-speaker distribution and quality
  4. valence-arousal consistency across emotion labels
  5. audio bandwidth / spectral quality (is 24 kHz genuine wideband or upsampled?)
  6. SNR vs DNSMOS relationship
(#3 emotion confusion matrix is produced by eval_emotion.py -> emotion_confusion.png)

Writes figures into reports/figures/ and a summary to data/manifests/dataset_analysis.json.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import welch  # noqa: E402

from ttsds.config import FIGURES_DIR, MANIFEST_DIR, PROJECT_ROOT, load_config  # noqa: E402
from ttsds.build_dataset import FINAL_SELECTION  # noqa: E402
from ttsds.publish import _stratified_split  # noqa: E402

INK, ACC = "#14346b", "#c0561f"
EMO_ORDER = ["neutral", "calm", "sad", "excited", "angry", "happy", "fearful", "surprised"]
VA = {"neutral": (0.0, 0.0), "calm": (0.3, -0.5), "sad": (-0.6, -0.4), "happy": (0.7, 0.4),
      "excited": (0.6, 0.8), "angry": (-0.6, 0.7), "fearful": (-0.5, 0.5), "surprised": (0.2, 0.8)}


def _sid(r):
    return os.path.splitext(os.path.basename(r["audio"]))[0]


def split_integrity(recs, cfg):
    out = {}
    for cfg_name, rs in recs.items():
        tr, va, te = _stratified_split(rs, cfg.targets.val_fraction)
        ids = {"train": {_sid(r) for r in tr}, "val": {_sid(r) for r in va}, "test": {_sid(r) for r in te}}
        overlap = (ids["train"] & ids["val"]) | (ids["train"] & ids["test"]) | (ids["val"] & ids["test"])
        spk = {k: sorted({r["speaker_id"] for r in g}) for k, g in (("train", tr), ("val", va), ("test", te))}
        # transcript duplication across splits
        norm = lambda r: " ".join((r.get("normalized_text") or r["text"]).lower().split())
        by_split_text = {k: {norm(r) for r in g} for k, g in (("train", tr), ("val", va), ("test", te))}
        cross_dups = len((by_split_text["val"] | by_split_text["test"]) & by_split_text["train"])
        emo = {k: dict(Counter(r["emotion"] for r in g)) for k, g in (("train", tr), ("val", va), ("test", te))}
        out[cfg_name] = {
            "sizes": {"train": len(tr), "val": len(va), "test": len(te)},
            "clip_overlap_across_splits": len(overlap),
            "speakers_train": len(spk["train"]), "speakers_val": len(spk["val"]), "speakers_test": len(spk["test"]),
            "all_speakers_seen_in_train": set(spk["val"] + spk["test"]).issubset(set(spk["train"])),
            "cross_split_transcript_dups": cross_dups,
            "emotion_per_split": emo,
        }
    return out


def per_speaker(recs):
    rows = []
    for cfg_name, rs in recs.items():
        by = defaultdict(list)
        for r in rs:
            by[r["speaker_id"]].append(r)
        for spk, g in by.items():
            ov = [x["dnsmos_ovrl"] for x in g if x.get("dnsmos_ovrl") is not None]
            snr = [x["snr_db"] for x in g if x.get("snr_db") is not None]
            rows.append({
                "speaker_id": spk, "language": g[0]["language"], "gender": g[0].get("gender"),
                "clips": len(g), "minutes": round(sum(x["duration"] for x in g) / 60, 1),
                "median_dnsmos": round(float(np.median(ov)), 2) if ov else None,
                "median_snr_db": round(float(np.median(snr)), 1) if snr else None,
                "n_emotions": len({x["emotion"] for x in g}),
            })
    rows.sort(key=lambda r: (-r["minutes"]))
    # figure: minutes per speaker
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    labs = [r["speaker_id"] for r in rows]
    cols = [INK if r["language"] == "en" else ACC for r in rows]
    ax.barh(range(len(rows)), [r["minutes"] for r in rows], color=cols)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labs, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("minutes"); ax.set_title("Minutes per speaker (blue = English, orange = Telugu)", fontsize=10.5, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "per_speaker_minutes.png", dpi=150); plt.close(fig)
    return rows


def vad_consistency(recs):
    by = defaultdict(lambda: {"v": [], "a": []})
    for cfg_name, rs in recs.items():
        for r in rs:
            if r.get("valence") is not None:
                by[r["emotion"]]["v"].append(r["valence"]); by[r["emotion"]]["a"].append(r["arousal"])
    stats = {e: {"valence": round(float(np.mean(d["v"])), 3), "arousal": round(float(np.mean(d["a"])), 3),
                 "n": len(d["v"])} for e, d in by.items()}
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for e in EMO_ORDER:
        if e not in stats:
            continue
        v, a = stats[e]["valence"], stats[e]["arousal"]
        ax.scatter(v, a, s=80, color=INK)
        ax.annotate(e, (v, a), fontsize=9, xytext=(5, 4), textcoords="offset points")
    ax.axhline(np.mean([stats[e]["arousal"] for e in stats]), color="#ccc", lw=0.8)
    ax.axvline(np.mean([stats[e]["valence"] for e in stats]), color="#ccc", lw=0.8)
    ax.set_xlabel("measured valence (audeering SER)"); ax.set_ylabel("measured arousal")
    ax.set_title("Measured valence-arousal per emotion label", fontsize=10.5, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "vad_by_emotion.png", dpi=150); plt.close(fig)
    return stats


def spectral_bandwidth(recs, per_lang=50):
    out = {}
    roll_all = {"en": [], "te": []}
    above8 = {"en": [], "te": []}
    for cfg_name, rs in recs.items():
        lang = "en" if cfg_name == "indian_english" else "te"
        rs = sorted(rs, key=lambda r: r["audio"])
        for r in rs[:: max(1, len(rs) // per_lang)][:per_lang]:
            try:
                y, sr = sf.read(str(PROJECT_ROOT / r["audio"]), dtype="float32")
            except Exception:
                continue
            if y.ndim > 1:
                y = y.mean(axis=1)
            f, p = welch(y, sr, nperseg=min(2048, len(y)))
            csum = np.cumsum(p) / (p.sum() + 1e-12)
            roll = float(f[np.searchsorted(csum, 0.99)])           # 99% energy rolloff
            e8 = float(p[f >= 8000].sum() / (p.sum() + 1e-12))     # energy fraction above 8 kHz
            roll_all[lang].append(roll); above8[lang].append(e8)
    for lang in ("en", "te"):
        r = roll_all[lang]
        out[lang] = {"n": len(r),
                     "rolloff99_hz_median": round(float(np.median(r))) if r else None,
                     "energy_above_8khz_median_pct": round(float(np.median(above8[lang])) * 100, 2) if r else None,
                     "clips_truly_wideband_pct": round(100 * float(np.mean([x > 0.005 for x in above8[lang]]))) if r else None}
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for lang, c in (("en", INK), ("te", ACC)):
        if roll_all[lang]:
            ax.hist(np.array(roll_all[lang]) / 1000, bins=20, alpha=0.6, color=c, label="English" if lang == "en" else "Telugu")
    ax.axvline(8, color="#888", ls="--", lw=1); ax.text(8.05, ax.get_ylim()[1]*0.9, "8 kHz", fontsize=8, color="#888")
    ax.set_xlabel("99% energy roll-off frequency (kHz)"); ax.set_ylabel("clips")
    ax.set_title("Spectral roll-off: energy concentrated below ~4 kHz, little above 8 kHz", fontsize=9.5, color=INK)
    ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "spectral_bandwidth.png", dpi=150); plt.close(fig)
    return out


def snr_vs_dnsmos(recs):
    xs, ys, cols = [], [], []
    for cfg_name, rs in recs.items():
        for r in rs:
            if r.get("snr_db") is not None and r.get("dnsmos_ovrl") is not None:
                xs.append(r["snr_db"]); ys.append(r["dnsmos_ovrl"])
                cols.append(INK if r["language"] == "en" else ACC)
    xs, ys = np.array(xs), np.array(ys)
    pear = float(np.corrcoef(xs, ys)[0, 1])
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.scatter(xs, ys, s=12, c=cols, alpha=0.5)
    ax.axhline(3.0, color="#888", ls="--", lw=1); ax.text(xs.min(), 3.02, "DNSMOS 3.0 gate", fontsize=8, color="#888")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("DNSMOS OVRL")
    ax.set_title(f"SNR vs DNSMOS (Pearson r = {pear:.2f}): they measure different things", fontsize=10, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "snr_vs_dnsmos.png", dpi=150); plt.close(fig)
    return {"pearson_r": round(pear, 3), "n": len(xs)}


def main():
    cfg = load_config()
    recs = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))
    res = {
        "split_integrity": split_integrity(recs, cfg),
        "per_speaker": per_speaker(recs),
        "vad_by_emotion": vad_consistency(recs),
        "spectral_bandwidth": spectral_bandwidth(recs),
        "snr_vs_dnsmos": snr_vs_dnsmos(recs),
    }
    (MANIFEST_DIR / "dataset_analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("split_integrity", "snr_vs_dnsmos", "spectral_bandwidth")}, ensure_ascii=False, indent=1)[:1400])
    print("VAD:", json.dumps(res["vad_by_emotion"], ensure_ascii=False))
    print("per-speaker rows:", len(res["per_speaker"]))
    print("wrote figures: per_speaker_minutes, vad_by_emotion, spectral_bandwidth, snr_vs_dnsmos")


if __name__ == "__main__":
    main()

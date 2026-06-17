"""Emotion-label RELIABILITY eval: re-tag a deterministic subset with a larger
model (sarvam-105b) and measure cross-model agreement vs the existing sarvam-30b
labels. PROXY for reliability; true human agreement is collected separately via
the review app. Reuses ttsds.tag_emotion so the 105b call uses the exact prompt.
Outputs: reports/figures/emotion_confusion.png, data/manifests/eval_emotion.json"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import cohen_kappa_score, confusion_matrix  # noqa: E402

from ttsds.config import FIGURES_DIR, MANIFEST_DIR, load_config  # noqa: E402
from ttsds.models import load_all_segments  # noqa: E402
from ttsds.sarvam_client import chat_json  # noqa: E402
from ttsds.tag_emotion import _system_prompt, describe_acoustics, is_whisper  # noqa: E402

N_PER_LANG = 60
MODEL = "sarvam-105b"


def sample_per_language(segs, n):
    """Deterministic stride sample: sort by id, take every k-th up to n."""
    out = []
    by_lang = defaultdict(list)
    for s in segs:
        by_lang[s.language].append(s)
    for lang in sorted(by_lang):
        items = sorted(by_lang[lang], key=lambda s: s.id)
        if len(items) <= n:
            out.extend(items); continue
        k = len(items) / n
        out.extend(items[int(i * k)] for i in range(n))
    return out


def retag(seg, cfg, sys_prompt):
    """Reproduce the exact pipeline prompt and call the larger model.
    Returns (old_emotion, new_emotion, old_style, new_style) with new_* None on failure."""
    whisper = is_whisper(seg, cfg)
    user = (
        f'Transcript: "{seg.transcript}"\n\n'
        f"Acoustic prosody (relative to this speaker):\n{describe_acoustics(seg)}\n\n"
        f"Whisper indicated acoustically: {'YES' if whisper else 'no'}"
    )
    result = chat_json(
        sys_prompt, user, model=MODEL, temperature=0.1,
        max_tokens=4000, reasoning_effort="low",
    )
    new_emotion = new_style = None
    if result:
        cand_e = str(result.get("emotion", "")).lower().strip()
        cand_s = str(result.get("style", "")).lower().strip()
        if cand_e in cfg.emotion.emotions:
            new_emotion = cand_e
        if cand_s in cfg.emotion.styles:
            new_style = cand_s
    # whisper override applies to STYLE only, never emotion (matches pipeline)
    if whisper and new_style is not None:
        new_style = "whisper"
    return seg.language, seg.emotion, new_emotion, seg.style, new_style


def metrics(pairs):
    """pairs: list of (old, new). Returns agreement fraction and Cohen's kappa."""
    if not pairs:
        return {"agreement": None, "kappa": None}
    old, new = zip(*pairs)
    agree = sum(o == n for o, n in pairs) / len(pairs)
    try:
        kappa = float(cohen_kappa_score(old, new))
    except Exception:  # noqa: BLE001 — single-class edge case
        kappa = None
    return {"agreement": round(agree, 4), "kappa": round(kappa, 4) if kappa is not None else None}


def main():
    cfg = load_config()
    emotions = cfg.emotion.emotions
    sys_prompt = _system_prompt(cfg)

    kept = [s for s in load_all_segments() if s.is_kept()]
    sample = sample_per_language(kept, N_PER_LANG)
    print(f"sampled {len(sample)} segments (target {N_PER_LANG}/lang); re-tagging with {MODEL}")

    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(lambda s: retag(s, cfg, sys_prompt), sample))

    failures = sum(1 for r in rows if r[2] is None)
    e_all, s_all = [], []
    e_lang, s_lang = defaultdict(list), defaultdict(list)
    for lang, oe, ne, os_, ns in rows:
        if ne is not None:
            e_all.append((oe, ne)); e_lang[lang].append((oe, ne))
        if ns is not None:
            s_all.append((os_, ns)); s_lang[lang].append((os_, ns))

    def block(epairs, spairs):
        em, sm = metrics(epairs), metrics(spairs)
        return {"n": len(epairs),
                "emotion_agreement": em["agreement"], "emotion_kappa": em["kappa"],
                "style_agreement": sm["agreement"], "style_kappa": sm["kappa"]}

    out = {"overall": {**block(e_all, s_all), "failures": failures}}
    for lang in sorted(e_lang):
        out[lang] = block(e_lang[lang], s_lang.get(lang, []))

    # confusion matrix (emotion) overall: rows=30b, cols=105b
    old, new = zip(*e_all) if e_all else ([], [])
    cm = confusion_matrix(old, new, labels=emotions)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(emotions))); ax.set_yticks(range(len(emotions)))
    ax.set_xticklabels(emotions, rotation=45, ha="right"); ax.set_yticklabels(emotions)
    ax.set_xlabel("sarvam-105b"); ax.set_ylabel("sarvam-30b")
    ax.set_title("Emotion label agreement (30b vs 105b)")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(emotions)):
        for j in range(len(emotions)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "emotion_confusion.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    (MANIFEST_DIR / "eval_emotion.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote reports/figures/emotion_confusion.png and data/manifests/eval_emotion.json")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

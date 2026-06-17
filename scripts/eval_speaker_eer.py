"""Speaker-verification EER + ROC-AUC evaluation using ECAPA-TDNN embeddings.

Embeds every kept clip with speechbrain ECAPA-TDNN (cached to an .npz), builds a
balanced trial set of same-speaker (positive) and different-speaker (negative)
clip pairs, scores each by cosine similarity, then reports ROC-AUC and the Equal
Error Rate. Writes an overlaid score-histogram figure and merges the new metrics
into data/manifests/eval_speaker.json (existing keys preserved).
"""

from __future__ import annotations

import itertools
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio.functional as AF  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from ttsds.config import FIGURES_DIR, MANIFEST_DIR, PROJECT_ROOT  # noqa: E402
from ttsds.models import load_all_segments  # noqa: E402

TARGET_SR = 16000
MAX_POS_PAIRS = 5000
CACHE_PATH = MANIFEST_DIR / "_spk_emb.npz"
rng = np.random.default_rng(0)


def load_clip(abs_path) -> torch.Tensor | None:
    """Load a WAV as a mono 16 kHz float tensor shaped [1, time]."""
    wav, sr = sf.read(str(abs_path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if wav.size == 0:
        return None
    t = torch.from_numpy(wav)
    if sr != TARGET_SR:
        t = AF.resample(t, sr, TARGET_SR)
    return t.unsqueeze(0)


def compute_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ids, speakers, embeddings); reuse cache if present."""
    if CACHE_PATH.exists():
        d = np.load(CACHE_PATH, allow_pickle=True)
        print(f"loaded {len(d['ids'])} cached embeddings from {CACHE_PATH.name}")
        return d["ids"], d["speakers"], d["embeddings"]

    from speechbrain.inference.speaker import EncoderClassifier

    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"}
    )
    ids, speakers, embs = [], [], []
    for seg in (s for s in load_all_segments() if s.is_kept()):
        try:
            wav = load_clip(PROJECT_ROOT / seg.wav_path)
            if wav is None:
                raise ValueError("empty audio")
            emb = model.encode_batch(wav).squeeze().detach().cpu().numpy()
            norm = np.linalg.norm(emb)
            if norm == 0:
                raise ValueError("zero embedding")
            ids.append(seg.id); speakers.append(seg.speaker_id)
            embs.append((emb / norm).astype(np.float32))
        except Exception as exc:  # skip a clip that fails to load/encode
            print(f"skip {seg.id}: {exc}")
    ids = np.array(ids); speakers = np.array(speakers)
    embeddings = np.vstack(embs)
    np.savez(CACHE_PATH, ids=ids, speakers=speakers, embeddings=embeddings)
    print(f"embedded {len(ids)} clips -> cached to {CACHE_PATH.name}")
    return ids, speakers, embeddings


def build_pairs(speakers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Balanced positive (same-speaker) / negative (diff-speaker) index pairs."""
    n = len(speakers)
    pos = [(i, j) for i, j in itertools.combinations(range(n), 2)
           if speakers[i] == speakers[j]]
    pos = np.array(pos)
    if len(pos) > MAX_POS_PAIRS:
        pos = pos[rng.choice(len(pos), MAX_POS_PAIRS, replace=False)]
    neg = set()
    target = len(pos)
    while len(neg) < target:
        i, j = rng.integers(0, n, size=2)
        if i != j and speakers[i] != speakers[j]:
            neg.add((min(i, j), max(i, j)))
    return pos, np.array(sorted(neg))


def main() -> None:
    _, speakers, embeddings = compute_embeddings()
    pos, neg = build_pairs(speakers)

    pos_scores = np.sum(embeddings[pos[:, 0]] * embeddings[pos[:, 1]], axis=1)
    neg_scores = np.sum(embeddings[neg[:, 0]] * embeddings[neg[:, 1]], axis=1)
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    eer_thr = float(thr[idx])

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(scores.min(), -0.2), 1.0, 60)
    ax.hist(pos_scores, bins=bins, alpha=0.6, color="#2a9d8f", label="same speaker")
    ax.hist(neg_scores, bins=bins, alpha=0.6, color="#e76f51", label="different speaker")
    ax.axvline(eer_thr, ls="--", color="#264653", lw=1.5,
               label=f"EER threshold = {eer_thr:.3f}")
    ax.set_xlabel("cosine similarity"); ax.set_ylabel("pair count")
    ax.set_title("Speaker verification (ECAPA-TDNN)")
    ax.legend(loc="upper center")
    ax.annotate(f"EER = {eer * 100:.2f}%   AUC = {auc:.4f}",
                xy=(0.02, 0.96), xycoords="axes fraction", va="top",
                fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "speaker_verification.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    json_path = MANIFEST_DIR / "eval_speaker.json"
    summary = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    summary.update({
        "roc_auc": round(auc, 4),
        "eer": round(eer, 4),
        "eer_threshold": round(eer_thr, 4),
        "n_positive_pairs": int(len(pos)),
        "n_negative_pairs": int(len(neg)),
    })
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nROC-AUC        = {auc:.4f}")
    print(f"EER            = {eer * 100:.2f}%")
    print(f"EER threshold  = {eer_thr:.4f}")
    print(f"positive pairs = {len(pos)}  negative pairs = {len(neg)}")
    print("wrote reports/figures/speaker_verification.png and updated data/manifests/eval_speaker.json")


if __name__ == "__main__":
    main()

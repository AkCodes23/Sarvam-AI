"""Speaker-verification evaluation using a pretrained ECAPA-TDNN embedder.

For every kept clip we extract a 192-d speaker embedding, then measure how
tightly each speaker's clips cluster (within-speaker similarity) versus how
distinct the speakers are from one another (between-speaker centroid
similarity). A high separation (within - between) means the dataset's
speaker_id labels are acoustically consistent; speakers whose own clips are
no more similar to each other than to other speakers are flagged as possible
label contamination.

Outputs a centroid similarity heatmap to reports/figures/speaker_similarity.png
and a summary to data/manifests/eval_speaker.json.
"""

from __future__ import annotations

import itertools
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio.functional as AF  # noqa: E402

from ttsds.config import FIGURES_DIR, MANIFEST_DIR, PROJECT_ROOT  # noqa: E402
from ttsds.models import load_all_segments  # noqa: E402

TARGET_SR = 16000
MAX_PAIRS = 200            # cap pairwise comparisons per speaker
FLAG_MARGIN = 0.1          # flag if within_cohesion < avg_between + FLAG_MARGIN
random.seed(0)


def load_clip(abs_path) -> torch.Tensor | None:
    """Load a WAV as a mono 16 kHz float tensor shaped [1, time]."""
    wav, sr = sf.read(str(abs_path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # downmix to mono
    if wav.size == 0:
        return None
    t = torch.from_numpy(wav)
    if sr != TARGET_SR:
        t = AF.resample(t, sr, TARGET_SR)
    return t.unsqueeze(0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # embeddings are pre-normalized


def mean_pairwise(embs: list[np.ndarray]) -> float:
    pairs = list(itertools.combinations(range(len(embs)), 2))
    if not pairs:
        return 1.0  # single clip: perfectly cohesive by definition
    if len(pairs) > MAX_PAIRS:
        pairs = random.sample(pairs, MAX_PAIRS)
    return float(np.mean([cosine(embs[i], embs[j]) for i, j in pairs]))


def main() -> None:
    from speechbrain.inference.speaker import EncoderClassifier

    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"}
    )

    segs = [s for s in load_all_segments() if s.is_kept()]
    by_speaker: dict[str, list[np.ndarray]] = {}
    n_ok, n_fail = 0, 0
    for seg in segs:
        try:
            wav = load_clip(PROJECT_ROOT / seg.wav_path)
            if wav is None:
                raise ValueError("empty audio")
            emb = model.encode_batch(wav).squeeze().detach().cpu().numpy()
            norm = np.linalg.norm(emb)
            if norm == 0:
                raise ValueError("zero embedding")
            by_speaker.setdefault(seg.speaker_id, []).append(emb / norm)
            n_ok += 1
        except Exception as exc:  # skip a clip that fails to load/encode
            n_fail += 1
            print(f"skip {seg.id}: {exc}")

    speakers = sorted(by_speaker)
    n_spk = len(speakers)
    if n_spk < 2:
        raise SystemExit(f"need >=2 speakers, got {n_spk} ({n_ok} clips, {n_fail} failed)")

    # within-speaker cohesion + centroids
    within = {sid: mean_pairwise(by_speaker[sid]) for sid in speakers}
    centroids = {sid: (c := np.mean(by_speaker[sid], axis=0)) / np.linalg.norm(c)
                 for sid in speakers}

    # between-speaker: centroid cosine matrix + off-diagonal mean
    sim = np.eye(n_spk)
    for i, j in itertools.combinations(range(n_spk), 2):
        v = cosine(centroids[speakers[i]], centroids[speakers[j]])
        sim[i, j] = sim[j, i] = v
    off = sim[~np.eye(n_spk, dtype=bool)]
    avg_between = float(np.mean(off))
    avg_within = float(np.mean(list(within.values())))
    separation = avg_within - avg_between
    flagged = [sid for sid in speakers if within[sid] < avg_between + FLAG_MARGIN]

    # heatmap
    fig, ax = plt.subplots(figsize=(1.0 * n_spk + 2, 1.0 * n_spk + 1.5))
    im = ax.imshow(sim, cmap="viridis", vmin=-0.2, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n_spk)); ax.set_yticks(range(n_spk))
    ax.set_xticklabels(speakers, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(speakers, fontsize=8)
    for i in range(n_spk):
        for j in range(n_spk):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                    color="white" if sim[i, j] < 0.6 else "black", fontsize=7)
    ax.set_title("Speaker-centroid cosine similarity (ECAPA-TDNN)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "speaker_similarity.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_clips": n_ok, "n_speakers": n_spk,
        "avg_within": round(avg_within, 4),
        "avg_between": round(avg_between, 4),
        "separation": round(separation, 4),
        "per_speaker": {sid: {"clips": len(by_speaker[sid]),
                              "within_cohesion": round(within[sid], 4)} for sid in speakers},
        "flagged": flagged,
    }
    (MANIFEST_DIR / "eval_speaker.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nclips ok={n_ok} failed={n_fail}  speakers={n_spk}")
    print(f"avg_within ={avg_within:.4f}")
    print(f"avg_between={avg_between:.4f}")
    print(f"separation ={separation:.4f}  (>0.3 is strong)")
    print(f"flagged    ={flagged or 'none'}")
    print("wrote reports/figures/speaker_similarity.png and data/manifests/eval_speaker.json")


if __name__ == "__main__":
    main()

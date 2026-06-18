"""Deliverable (A): dimensional VAD ratings for ALL kept clips.

Runs the audeering dimensional SER model
(audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) over every kept clip.
The model has a custom regression head on top of wav2vec2 that emits three
continuous scores in [0,1]: [arousal, dominance, valence] (MSP-Podcast dims).
We reproduce the model-card architecture explicitly (no trust_remote_code) so the
load is reproducible and pinned to the local transformers version.

For each clip: resample to 16 kHz mono, clip to [-1, 1], take a centered 8 s
window (speed), then map the VAD point to the project's 8-emotion taxonomy via a
valence/arousal quadrant rule.

Output: data/manifests/score_ser.json
  { id: {"valence", "arousal", "dominance", "ser_emotion"} }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Config,
    Wav2Vec2PreTrainedModel,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttsds.config import MANIFEST_DIR, PROJECT_ROOT  # noqa: E402
from ttsds.models import load_all_segments  # noqa: E402

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
TARGET_SR = 16000
WINDOW_S = 8.0          # centered analysis window (speed)
WINDOW_SAMPLES = int(WINDOW_S * TARGET_SR)


# --- model-card architecture (custom regression head over wav2vec2) -----------

class RegressionHead(nn.Module):
    """Classification/regression head, identical to the model card."""

    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    """wav2vec2 encoder + mean-pooled regression head -> [arousal, dominance, valence]."""

    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)

    def _init_weights(self, module) -> None:  # noqa: ANN001
        """No-op: we always load pretrained weights, never init from scratch."""

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.wav2vec2(input_values)
        hidden = outputs[0]
        hidden = torch.mean(hidden, dim=1)  # mean pool over time
        return self.classifier(hidden)


def load_model() -> EmotionModel:
    """Build the architecture and load weights manually.

    transformers 5.x's `from_pretrained` finalization path is incompatible with
    this old-style custom `PreTrainedModel` subclass (it expects
    `all_tied_weights_keys`, populated only on the standard init path). We sidestep
    it entirely: download the safetensors checkpoint and load the state dict by
    hand. This keeps the load deterministic and pinned to the local versions.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config = Wav2Vec2Config.from_pretrained(MODEL_NAME)
    model = EmotionModel(config)

    weights_path = hf_hub_download(repo_id=MODEL_NAME, filename="model.safetensors")
    state = load_file(weights_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # the only acceptable gaps are wav2vec2 buffers re-derived at runtime
    critical = [k for k in missing if k.startswith("classifier.")]
    if critical:
        raise RuntimeError(f"missing regression-head weights: {critical}")
    model.eval()
    return model


# --- taxonomy mapping ---------------------------------------------------------

def vad_to_emotion(valence: float, arousal: float) -> str:
    """Map a VAD point to the project's 8-emotion closed set via an A/V quadrant
    rule. Scores are in [0,1]; 0.5 is the neutral midpoint.

    high arousal + high valence -> excited       (happy is the softer variant)
    high arousal + low  valence -> angry
    low  arousal + low  valence -> sad
    low  arousal + high valence -> calm
    mid everything              -> neutral
    """
    hi_a, lo_a = 0.60, 0.40
    hi_v, lo_v = 0.55, 0.45

    # clearly low-arousal regions
    if arousal <= lo_a:
        if valence <= lo_v:
            return "sad"
        if valence >= hi_v:
            return "calm"
        return "calm"  # low arousal, neutral valence reads as calm/relaxed

    # clearly high-arousal regions
    if arousal >= hi_a:
        if valence >= hi_v:
            return "excited"
        if valence <= lo_v:
            return "angry"
        # high arousal, neutral valence: bright -> happy, else surprised-ish
        return "happy"

    # mid arousal band
    if valence >= hi_v:
        return "happy"
    if valence <= lo_v:
        return "sad"
    return "neutral"


# --- inference ----------------------------------------------------------------

def load_audio(wav_path: Path) -> np.ndarray | None:
    """Load -> mono -> 16 kHz -> clip to [-1,1] -> centered 8 s window."""
    try:
        y, _ = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    except Exception as exc:  # noqa: BLE001 — skip unreadable clip, keep going
        print(f"  ! failed to load {wav_path}: {exc}")
        return None
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    if y.size == 0:
        return None
    if y.size > WINDOW_SAMPLES:
        start = (y.size - WINDOW_SAMPLES) // 2
        y = y[start:start + WINDOW_SAMPLES]
    return y


@torch.no_grad()
def predict_vad(model: EmotionModel, y: np.ndarray) -> tuple[float, float, float]:
    """Returns (valence, arousal, dominance) in [0,1].
    Model output order is [arousal, dominance, valence]."""
    x = torch.from_numpy(y).unsqueeze(0)  # (1, T)
    out = model(x).squeeze(0).cpu().numpy()
    arousal, dominance, valence = float(out[0]), float(out[1]), float(out[2])
    return valence, arousal, dominance


def main() -> None:
    print(f"loading {MODEL_NAME} (CPU, float32) ...")
    model = load_model()

    kept = [s for s in load_all_segments() if s.is_kept()]
    print(f"{len(kept)} kept clips to score")

    out: dict[str, dict] = {}
    for i, seg in enumerate(kept, 1):
        wav = (PROJECT_ROOT / seg.wav_path)
        y = load_audio(wav)
        if y is None:
            continue
        valence, arousal, dominance = predict_vad(model, y)
        out[seg.id] = {
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "dominance": round(dominance, 4),
            "ser_emotion": vad_to_emotion(valence, arousal),
        }
        if i % 50 == 0 or i == len(kept):
            print(f"  scored {i}/{len(kept)}")

    dest = MANIFEST_DIR / "score_ser.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dest} ({len(out)} clips)")

    # quick per-language VAD summary for the report
    by_lang: dict[str, list[dict]] = {}
    id_lang = {s.id: s.language for s in kept}
    for sid, rec in out.items():
        by_lang.setdefault(id_lang[sid], []).append(rec)
    for lang in sorted(by_lang):
        recs = by_lang[lang]
        mv = np.mean([r["valence"] for r in recs])
        ma = np.mean([r["arousal"] for r in recs])
        md = np.mean([r["dominance"] for r in recs])
        print(f"  {lang}: n={len(recs)} meanV={mv:.3f} meanA={ma:.3f} meanD={md:.3f}")


if __name__ == "__main__":
    main()

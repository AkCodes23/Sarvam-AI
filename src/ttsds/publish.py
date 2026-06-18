"""Build HuggingFace datasets from the final selection and push them public."""

from __future__ import annotations

import json

from datasets import Audio, Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi

from .build_dataset import FINAL_SELECTION
from .config import PROJECT_ROOT, REPORTS_DIR, Config, load_secrets

_FEATURES = Features({
    "audio": Audio(),  # sampling_rate set after, via cast
    "text": Value("string"),
    "normalized_text": Value("string"),
    "language": Value("string"),
    "language_code": Value("string"),
    "emotion": Value("string"),
    "style": Value("string"),
    "emotion_confidence": Value("float32"),
    "tag_source": Value("string"),
    "speaker_id": Value("string"),
    "gender": Value("string"),
    "accent": Value("string"),
    "duration": Value("float32"),
    # audio quality
    "snr_db": Value("float32"),
    "dnsmos_ovrl": Value("float32"),
    "dnsmos_sig": Value("float32"),
    "dnsmos_bak": Value("float32"),
    "dnsmos_pass": Value("bool"),
    "squim_stoi": Value("float32"),
    "squim_pesq": Value("float32"),
    "squim_sisdr": Value("float32"),
    # transcript + emotion validation
    "mms_align_score": Value("float32"),
    "overlap_flag": Value("bool"),
    "ser_emotion": Value("string"),
    "valence": Value("float32"),
    "arousal": Value("float32"),
    "dominance": Value("float32"),
    # topic + LLM-judge cross-check
    "topic": Value("string"),
    "llm_tts_suitable": Value("float32"),
    # edge-case annotation
    "annotated_text": Value("string"),
    "annotation_flags": Value("string"),
    "has_noise": Value("bool"),
    "has_truncation": Value("bool"),
    "has_codemix": Value("bool"),
    "has_laughter": Value("bool"),
    "emotion_low_confidence": Value("bool"),
    "transcript_review_needed": Value("bool"),
    "low_quality_audio": Value("bool"),
    # provenance
    "source_video_id": Value("string"),
    "source_url": Value("string"),
    "source_channel": Value("string"),
    "license": Value("string"),
    "segment_start": Value("float32"),
    "segment_end": Value("float32"),
    "sample_rate": Value("int32"),
})


def _stratified_split(recs: list[dict], val_fraction: float):
    """Stratified 3-way split: every k-th -> test, next -> validation, rest -> train."""
    k = max(3, round(1.0 / val_fraction)) if val_fraction > 0 else 0
    ordered = sorted(recs, key=lambda r: (r["emotion"] or "", r["speaker_id"], r["audio"]))
    if k == 0:
        return ordered, [], []
    train, val, test = [], [], []
    for i, r in enumerate(ordered):
        if i % k == 0:
            test.append(r)
        elif i % k == 1:
            val.append(r)
        else:
            train.append(r)
    return train, val, test


def _to_dataset(recs: list[dict], sr: int) -> Dataset:
    cols: dict[str, list] = {key: [] for key in _FEATURES}
    for r in recs:
        for key in _FEATURES:
            if key == "audio":
                cols[key].append(str(PROJECT_ROOT / r["audio"]))
            else:
                cols[key].append(r.get(key))
    ds = Dataset.from_dict(cols, features=_FEATURES)
    return ds.cast_column("audio", Audio(sampling_rate=sr))


def build_dataset_dicts(cfg: Config) -> dict[str, DatasetDict]:
    records = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))
    sr = cfg.audio.master_sample_rate
    out: dict[str, DatasetDict] = {}
    for config_name, recs in records.items():
        if not recs:
            continue
        train, val, test = _stratified_split(recs, cfg.targets.val_fraction)
        dd = {"train": _to_dataset(train, sr)}
        if val:
            dd["validation"] = _to_dataset(val, sr)
        if test:
            dd["test"] = _to_dataset(test, sr)
        out[config_name] = DatasetDict(dd)
    return out


def push(cfg: Config) -> str:
    secrets = load_secrets()
    repo_id = secrets.repo_id
    if not repo_id or not secrets.hf_token:
        raise RuntimeError("HF_TOKEN and HF_USERNAME/HF_DATASET_REPO required to publish.")

    dicts = build_dataset_dicts(cfg)
    for config_name, dd in dicts.items():
        dd.push_to_hub(repo_id, config_name=config_name, private=False, token=secrets.hf_token)

    # upload our dataset card as the repo README (after data push so it isn't clobbered)
    card = REPORTS_DIR / "DATASET_CARD.md"
    if card.exists():
        HfApi().upload_file(
            path_or_fileobj=str(card), path_in_repo="README.md",
            repo_id=repo_id, repo_type="dataset", token=secrets.hf_token,
        )
    return repo_id


def verify(cfg: Config) -> dict:
    from datasets import load_dataset

    secrets = load_secrets(require_sarvam=False)
    repo_id = secrets.repo_id
    report: dict = {"repo_id": repo_id, "configs": {}}
    for lang in cfg.languages:
        config_name = cfg.hf_config_name(lang)
        try:
            ds = load_dataset(repo_id, config_name, split="train")
            sample = ds[0]
            report["configs"][config_name] = {
                "rows": ds.num_rows,
                "audio_ok": sample["audio"]["array"] is not None,
                "sr": sample["audio"]["sampling_rate"],
                "first_text": sample["text"][:60],
            }
        except Exception as e:  # noqa: BLE001
            report["configs"][config_name] = {"error": str(e)}
    return report

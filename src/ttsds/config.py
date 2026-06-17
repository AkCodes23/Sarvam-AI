"""Central configuration: paths, tunables (config.yaml), secrets (.env), ffmpeg."""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MASTER_DIR = DATA_DIR / "master"
SEGMENTS_DIR = DATA_DIR / "segments"
MANIFEST_DIR = DATA_DIR / "manifests"
REVIEW_DIR = DATA_DIR / "review_app"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TOOLS_DIR = PROJECT_ROOT / "tools"

for _d in (RAW_DIR, MASTER_DIR, SEGMENTS_DIR, MANIFEST_DIR, REVIEW_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- ffmpeg resolution: prefer bundled static binary, fall back to PATH --------

def _resolve_binary(name: str) -> str:
    bundled = TOOLS_DIR / f"{name}.exe"
    if bundled.exists():
        return str(bundled)
    bundled_nix = TOOLS_DIR / name
    if bundled_nix.exists():
        return str(bundled_nix)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"{name} not found in {TOOLS_DIR} or on PATH. Run scripts/setup or place a binary there."
    )


FFMPEG = _resolve_binary("ffmpeg")
FFPROBE = _resolve_binary("ffprobe")


# --- typed config from config.yaml --------------------------------------------

class AudioCfg(BaseModel):
    asr_sample_rate: int
    master_sample_rate: int
    channels: int


class SegmentationCfg(BaseModel):
    min_duration_s: float
    max_duration_s: float
    target_min_s: float
    target_max_s: float
    silence_top_db: float
    min_silence_gap_s: float
    edge_pad_s: float
    merge_gap_s: float


class QualityCfg(BaseModel):
    max_clipped_fraction: float
    min_snr_db: float
    max_silence_ratio: float
    min_language_probability: float
    max_gap_energy_ratio: float
    min_chars_per_sec: float
    max_chars_per_sec: float
    dedup_similarity: float


class AsrCfg(BaseModel):
    batch_model: str
    batch_mode: str
    batch_timeout_s: int
    realtime_model: str


class LlmCfg(BaseModel):
    model: str
    temperature: float
    max_tokens: int


class WhisperGateCfg(BaseModel):
    max_voiced_fraction: float
    max_energy_zscore: float
    max_hnr_db: float


class EmotionCfg(BaseModel):
    emotions: list[str]
    styles: list[str]
    low_confidence_threshold: float
    whisper: WhisperGateCfg


class TargetsCfg(BaseModel):
    minutes_per_language: float
    val_fraction: float


class LanguageCfg(BaseModel):
    code: str
    name: str
    config: str


class DatasetCfg(BaseModel):
    license: str


class Config(BaseModel):
    audio: AudioCfg
    segmentation: SegmentationCfg
    quality: QualityCfg
    asr: AsrCfg
    llm: LlmCfg
    emotion: EmotionCfg
    targets: TargetsCfg
    languages: dict[str, LanguageCfg]
    dataset: DatasetCfg

    def lang_code(self, lang: str) -> str:
        return self.languages[lang].code

    def hf_config_name(self, lang: str) -> str:
        return self.languages[lang].config


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> Config:
    path = path or (CONFIG_DIR / "config.yaml")
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Config(**data)


# --- secrets ------------------------------------------------------------------

class Secrets(BaseModel):
    sarvam_api_key: str
    hf_token: str | None = None
    hf_username: str | None = None
    hf_dataset_repo: str | None = None

    @property
    def repo_id(self) -> str | None:
        if self.hf_dataset_repo:
            return self.hf_dataset_repo
        if self.hf_username:
            return f"{self.hf_username}/sarvam-tts-in-te-en"
        return None


@lru_cache(maxsize=1)
def load_secrets(require_sarvam: bool = True) -> Secrets:
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("SARVAM_API_KEY", "").strip()
    if require_sarvam and not key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return Secrets(
        sarvam_api_key=key,
        hf_token=(os.getenv("HF_TOKEN") or "").strip() or None,
        hf_username=(os.getenv("HF_USERNAME") or "").strip() or None,
        hf_dataset_repo=(os.getenv("HF_DATASET_REPO") or "").strip() or None,
    )

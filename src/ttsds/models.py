"""Data models: source specs (sources.yaml) and per-segment records (manifests)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .config import MANIFEST_DIR


class SourceSpec(BaseModel):
    """One curated YouTube source from config/sources.yaml."""

    id: str                       # slug, unique, e.g. "te_kathalu_01"
    url: str
    language: str                 # "en" | "te"
    content_type: str             # audiobook | lecture | news | story | discourse | talk ...
    speaker_id: str               # stable voice label for the dataset
    license: str = "unknown"
    title: str | None = None
    notes: str | None = None
    expected_speakers: int | None = None   # diarization hint; None -> auto
    enabled: bool = True
    max_minutes: float | None = None        # cap audio taken from this source
    start_offset_s: float = 0.0              # skip intro
    end_trim_s: float = 0.0                  # skip outro (seconds from end)


class SourceSet(BaseModel):
    sources: list[SourceSpec]

    def by_language(self, lang: str) -> list[SourceSpec]:
        return [s for s in self.sources if s.language == lang and s.enabled]

    def get(self, source_id: str) -> SourceSpec:
        for s in self.sources:
            if s.id == source_id:
                return s
        raise KeyError(source_id)


def load_sources(path: Path) -> SourceSet:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return SourceSet(**data)


class Segment(BaseModel):
    """A single candidate/accepted clip. Flows through the whole pipeline."""

    id: str
    source_id: str
    language: str
    language_code: str
    speaker_id: str

    wav_path: str                 # relative to PROJECT_ROOT
    start_s: float                # position in the source timeline
    end_s: float
    duration_s: float

    # transcripts
    transcript: str = ""          # authoritative (realtime pass, or human-fixed)
    transcript_batch: str = ""    # from batch diarization pass
    asr_language_probability: float | None = None
    asr_language_code: str | None = None

    # acoustic features (raw) and per-speaker z-scores
    features: dict = Field(default_factory=dict)
    features_z: dict = Field(default_factory=dict)

    # quality metrics + gate outcome
    metrics: dict = Field(default_factory=dict)
    status: str = "candidate"     # candidate | pass | flag | reject
    reject_reasons: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)

    # emotion / style tags
    emotion: str | None = None
    style: str | None = None
    emotion_confidence: float | None = None
    emotion_rationale: str | None = None
    tag_source: str = "auto"      # auto | human

    # human review
    review_decision: str | None = None   # accept | reject | None

    # provenance
    source_url: str = ""
    source_video_id: str = ""
    source_channel: str = ""
    source_title: str = ""
    license: str = "unknown"
    upload_date: str = ""

    def is_kept(self) -> bool:
        """Final inclusion: rejected clips and human-rejected clips are dropped."""
        if self.review_decision == "reject":
            return False
        if self.review_decision == "accept":
            return True
        return self.status in ("pass", "flag")


# --- manifest IO --------------------------------------------------------------

def manifest_path(source_id: str) -> Path:
    return MANIFEST_DIR / f"{source_id}.segments.json"


def save_segments(source_id: str, segments: list[Segment]) -> Path:
    p = manifest_path(source_id)
    p.write_text(
        json.dumps([s.model_dump() for s in segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p


def load_segments(source_id: str) -> list[Segment]:
    p = manifest_path(source_id)
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [Segment(**r) for r in raw]


def load_all_segments() -> list[Segment]:
    out: list[Segment] = []
    for p in sorted(MANIFEST_DIR.glob("*.segments.json")):
        raw = json.loads(p.read_text(encoding="utf-8"))
        out.extend(Segment(**r) for r in raw)
    return out

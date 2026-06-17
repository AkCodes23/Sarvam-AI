"""Download best-quality audio from YouTube with yt-dlp and capture provenance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yt_dlp

from .config import RAW_DIR, TOOLS_DIR


@dataclass
class SourceMeta:
    video_id: str
    title: str
    channel: str
    upload_date: str
    license: str
    url: str
    duration_s: float
    audio_path: str   # relative to project root


def _meta_path(source_id: str) -> Path:
    return RAW_DIR / f"{source_id}.meta.json"


def download_source(source_id: str, url: str, *, declared_license: str = "unknown") -> SourceMeta:
    """Download bestaudio (no re-encode) and persist a provenance record."""
    outtmpl = str(RAW_DIR / f"{source_id}.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "ffmpeg_location": str(TOOLS_DIR),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "writeinfojson": False,
        "retries": 5,
        "fragment_retries": 5,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = Path(ydl.prepare_filename(info))

    lic = info.get("license") or declared_license or "unknown"
    meta = SourceMeta(
        video_id=info.get("id", source_id),
        title=info.get("title", ""),
        channel=info.get("uploader") or info.get("channel") or "",
        upload_date=info.get("upload_date", ""),
        license=str(lic),
        url=info.get("webpage_url", url),
        duration_s=float(info.get("duration") or 0.0),
        audio_path=str(audio_path),
    )
    _meta_path(source_id).write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def load_meta(source_id: str) -> SourceMeta | None:
    p = _meta_path(source_id)
    if not p.exists():
        return None
    return SourceMeta(**json.loads(p.read_text(encoding="utf-8")))

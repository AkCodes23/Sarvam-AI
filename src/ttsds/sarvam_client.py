"""Thin, retrying wrappers around the Sarvam SDK: batch ASR+diarization,
per-segment realtime transcription, and chat completions that return JSON."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from sarvamai import SarvamAI
from sarvamai import (
    InternalServerError,
    ServiceUnavailableError,
    TooManyRequestsError,
)

from .config import load_secrets

# transient = Sarvam 429/5xx + raw network drops (disconnects, read/connect timeouts)
_TRANSIENT = (
    TooManyRequestsError, InternalServerError, ServiceUnavailableError,
    httpx.TransportError,
)


@lru_cache(maxsize=1)
def get_client() -> SarvamAI:
    return SarvamAI(api_subscription_key=load_secrets().sarvam_api_key)


def _retry(fn, *, tries: int = 5, base_delay: float = 3.0):
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except _TRANSIENT as e:  # rate limit / transient server error
            last = e
            time.sleep(base_delay * (2 ** attempt))
        except Exception as e:  # noqa: BLE001 — surface non-transient immediately
            raise e
    raise last  # type: ignore[misc]


# --- batch ASR + diarization --------------------------------------------------

@dataclass
class DiarChunk:
    speaker_id: str
    start_s: float
    end_s: float
    text: str


@dataclass
class BatchResult:
    language_code: str | None
    full_transcript: str
    chunks: list[DiarChunk]


def run_batch_diarization(
    audio_path: Path,
    language_code: str,
    *,
    model: str,
    mode: str,
    num_speakers: int | None,
    out_dir: Path,
    timeout_s: int = 2400,
) -> BatchResult:
    """Upload one audio file, run a diarized batch job, parse the result JSON."""
    client = get_client()

    def _create():
        return client.speech_to_text_job.create_job(
            model=model,
            mode=mode,
            with_diarization=True,
            with_timestamps=True,
            language_code=language_code,
            num_speakers=num_speakers,
        )

    job = _retry(_create)
    job.upload_files(file_paths=[str(audio_path)])
    job.start()
    job.wait_until_complete(poll_interval=10, timeout=timeout_s)
    if not job.is_successful():
        status = job.get_status()
        raise RuntimeError(f"Batch job {job.job_id} failed: {status}")

    out_dir.mkdir(parents=True, exist_ok=True)
    job.download_outputs(output_dir=str(out_dir))
    return _parse_batch_output(out_dir, audio_path.stem)


def _parse_batch_output(out_dir: Path, stem: str) -> BatchResult:
    """The job writes one JSON per input audio; find and parse it."""
    candidates = sorted(out_dir.glob("*.json"))
    # prefer the file whose name matches the input stem
    match = [p for p in candidates if stem in p.stem] or candidates
    if not match:
        raise FileNotFoundError(f"No batch output JSON in {out_dir}")
    data = json.loads(match[0].read_text(encoding="utf-8"))

    diar = data.get("diarized_transcript") or {}
    entries = diar.get("entries") if isinstance(diar, dict) else None
    chunks: list[DiarChunk] = []
    for e in entries or []:
        chunks.append(
            DiarChunk(
                speaker_id=str(e.get("speaker_id", "0")),
                start_s=float(e.get("start_time_seconds", 0.0)),
                end_s=float(e.get("end_time_seconds", 0.0)),
                text=(e.get("transcript") or "").strip(),
            )
        )
    return BatchResult(
        language_code=data.get("language_code"),
        full_transcript=(data.get("transcript") or "").strip(),
        chunks=chunks,
    )


# --- realtime per-segment transcription ---------------------------------------

@dataclass
class RealtimeResult:
    transcript: str
    language_code: str | None
    language_probability: float | None
    words: list[str]
    word_start: list[float]
    word_end: list[float]


def transcribe_clip(audio_path: Path, language_code: str, *, model: str) -> RealtimeResult:
    client = get_client()

    def _call():
        with open(audio_path, "rb") as f:
            return client.speech_to_text.transcribe(
                file=f, model=model, language_code=language_code
            )

    resp = _retry(_call)
    ts = getattr(resp, "timestamps", None)
    words = list(getattr(ts, "words", []) or []) if ts else []
    wstart = list(getattr(ts, "start_time_seconds", []) or []) if ts else []
    wend = list(getattr(ts, "end_time_seconds", []) or []) if ts else []
    return RealtimeResult(
        transcript=(resp.transcript or "").strip(),
        language_code=getattr(resp, "language_code", None),
        language_probability=getattr(resp, "language_probability", None),
        words=words,
        word_start=wstart,
        word_end=wend,
    )


# --- chat -> JSON -------------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any] | None:
    """Parse JSON, tolerating code fences and JSON embedded in prose/reasoning."""
    if not text:
        return None
    cleaned = re.sub(r"```$", "", re.sub(r"^```(?:json)?", "", text.strip()).strip()).strip()
    try:
        return json.loads(cleaned)
    except Exception:  # noqa: BLE001
        pass
    # scan for the first balanced {...} that parses
    depth, start = 0, None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(cleaned[start:i + 1])
                except Exception:  # noqa: BLE001
                    start = None
    return None


def chat_json(
    system: str,
    user: str,
    *,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 1000,
    tries: int = 3,
    reasoning_effort: str = "low",
) -> dict[str, Any] | None:
    """Call chat and return parsed JSON. Sarvam 30b/105b are reasoning models, so
    we keep reasoning low and a generous token budget, and fall back to the
    reasoning channel if the content channel comes back empty."""
    client = get_client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for _ in range(tries):
        def _call():
            return client.chat.completions(
                messages=messages, model=model, temperature=temperature,
                max_tokens=max_tokens, reasoning_effort=reasoning_effort,
            )

        resp = _retry(_call)
        msg = resp.choices[0].message if resp.choices else None
        content = (getattr(msg, "content", None) or "").strip() if msg else ""
        parsed = _extract_json(content)
        if parsed is None and msg is not None:  # fall back to reasoning channel
            parsed = _extract_json(getattr(msg, "reasoning_content", None) or "")
        if parsed is not None:
            return parsed
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": "Respond with ONLY valid JSON, no prose."})
    return None

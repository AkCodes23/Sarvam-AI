"""Turn diarized batch chunks + the audio master into single-speaker, silence-
snapped candidate segments in the target duration window."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .audio import nonsilent_intervals, read_wav, trim_edges, write_wav
from .config import PROJECT_ROOT, SEGMENTS_DIR, Config
from .download import SourceMeta
from .models import Segment, SourceSpec
from .sarvam_client import DiarChunk


def dominant_speaker(chunks: list[DiarChunk]) -> str:
    dur: dict[str, float] = defaultdict(float)
    for c in chunks:
        dur[c.speaker_id] += max(0.0, c.end_s - c.start_s)
    return max(dur, key=dur.get) if dur else "0"


def build_runs(chunks: list[DiarChunk], speaker: str, merge_gap_s: float) -> list[tuple[float, float]]:
    """Contiguous spans of the target speaker, bridging gaps < merge_gap_s.

    A chunk by a *different* speaker breaks the run, so runs never span a
    speaker change (protects against cutting across overlapping/other voices).
    """
    runs: list[list[float]] = []
    for c in sorted(chunks, key=lambda x: x.start_s):
        if c.speaker_id != speaker:
            continue
        if runs and c.start_s - runs[-1][1] <= merge_gap_s:
            runs[-1][1] = max(runs[-1][1], c.end_s)
        else:
            runs.append([c.start_s, c.end_s])
    return [(s, e) for s, e in runs if e > s]


def _hard_split(s: int, e: int, sr: int, target_max_s: float) -> list[tuple[int, int]]:
    dur = (e - s) / sr
    n = max(1, int(np.ceil(dur / target_max_s)))
    step = (e - s) // n
    out = []
    for i in range(n):
        a = s + i * step
        b = e if i == n - 1 else s + (i + 1) * step
        out.append((a, b))
    return out


def pack_segments(islands: list[tuple[int, int]], sr: int, cfg: Config) -> list[tuple[int, int]]:
    """Greedily accumulate speech islands into clips, cutting in the silences
    between them, aiming for [target_min, target_max] and bounded by [min, max]."""
    sc = cfg.segmentation
    if not islands:
        return []
    segs: list[tuple[int, int]] = []
    cur_s, cur_e = islands[0]
    for s, e in islands[1:]:
        prospective = (e - cur_s) / sr
        cur_dur = (cur_e - cur_s) / sr
        if prospective <= sc.target_max_s:
            cur_e = e
        elif cur_dur >= sc.target_min_s:
            segs.append((cur_s, cur_e)); cur_s, cur_e = s, e
        else:
            cur_e = e  # too short to emit; extend past target to reach min
    # flush leftover
    if (cur_e - cur_s) / sr >= sc.min_duration_s:
        segs.append((cur_s, cur_e))
    elif segs and (cur_e - segs[-1][0]) / sr <= sc.max_duration_s:
        segs[-1] = (segs[-1][0], cur_e)

    final: list[tuple[int, int]] = []
    for s, e in segs:
        if (e - s) / sr <= sc.max_duration_s:
            final.append((s, e))
        else:
            final.extend(_hard_split(s, e, sr, sc.target_max_s))
    return [(s, e) for s, e in final if (e - s) / sr >= sc.min_duration_s]


def segment_source(
    spec: SourceSpec, meta: SourceMeta, master_wav: Path,
    chunks: list[DiarChunk], cfg: Config, time_offset_s: float = 0.0,
) -> list[Segment]:
    y, sr = read_wav(master_wav)
    sc = cfg.segmentation
    lang_code = cfg.lang_code(spec.language)

    # choose the target voice
    if spec.expected_speakers == 1:
        speaker = dominant_speaker(chunks) if chunks else "0"
        kept = chunks  # trust single-speaker source; keep all chunks
    else:
        speaker = dominant_speaker(chunks)
        kept = [c for c in chunks if c.speaker_id == speaker]

    runs = build_runs(kept, speaker, sc.merge_gap_s)
    out_dir = SEGMENTS_DIR / spec.id
    out_dir.mkdir(parents=True, exist_ok=True)

    segments: list[Segment] = []
    idx = 0
    for run_start, run_end in runs:
        a = int(run_start * sr)
        b = min(len(y), int(run_end * sr))
        if b - a < int(sc.min_duration_s * sr):
            continue
        run_y = y[a:b]
        islands = nonsilent_intervals(
            run_y, sr, top_db=sc.silence_top_db, min_gap_s=sc.min_silence_gap_s
        )
        for s, e in pack_segments(islands, sr, cfg):
            clip = trim_edges(run_y[s:e], sr, top_db=sc.silence_top_db, pad_s=sc.edge_pad_s)
            dur = len(clip) / sr
            if dur < sc.min_duration_s or dur > sc.max_duration_s:
                continue
            seg_id = f"{spec.id}_{idx:04d}"
            wav_path = out_dir / f"{seg_id}.wav"
            write_wav(wav_path, clip, sr)
            abs_start = time_offset_s + run_start + s / sr
            m0, m1 = run_start + s / sr, run_start + e / sr
            batch_text = " ".join(
                c.text for c in kept if c.start_s < m1 and c.end_s > m0 and c.text
            ).strip()
            segments.append(Segment(
                id=seg_id, source_id=spec.id, language=spec.language,
                language_code=lang_code, speaker_id=spec.speaker_id,
                wav_path=str(wav_path.relative_to(PROJECT_ROOT)),
                start_s=round(abs_start, 3), end_s=round(abs_start + dur, 3),
                duration_s=round(dur, 3), transcript_batch=batch_text,
                source_url=meta.url, source_video_id=meta.video_id,
                source_channel=meta.channel, source_title=meta.title,
                license=meta.license or spec.license, upload_date=meta.upload_date,
            ))
            idx += 1
    return segments

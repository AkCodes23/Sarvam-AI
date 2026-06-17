"""ttsds — command-line orchestrator for the TTS dataset pipeline.

Typical flow:
  ttsds smoke
  ttsds process-all            # download -> diarize -> segment -> transcribe -> tag, per source
  ttsds review-build           # open data/review_app/index.html, curate, export review.csv
  ttsds review-merge           # apply human decisions (human overrides win)
  ttsds build                  # balance + finalize audio + stats + card
  ttsds publish && ttsds verify
"""

from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .audio import extract_to_wav, probe_duration
from .config import CONFIG_DIR, MANIFEST_DIR, MASTER_DIR, load_config, load_secrets
from .download import download_source, load_meta
from .features import compute_features, normalize_per_speaker
from .models import Segment, SourceSpec, load_all_segments, load_segments, load_sources, save_segments
from .quality import run_quality
from .sarvam_client import run_batch_diarization
from .segment import segment_source
from .tag_emotion import tag_segments
from .transcribe_segments import transcribe_segments

app = typer.Typer(add_completion=False, help="Sarvam TTS dataset pipeline")
console = Console()
SOURCES_PATH = CONFIG_DIR / "sources.yaml"


def _sources() -> list[SourceSpec]:
    return load_sources(SOURCES_PATH).sources


def _trim_window(spec: SourceSpec, duration: float) -> tuple[float, float]:
    start = max(0.0, spec.start_offset_s)
    end = max(start, duration - max(0.0, spec.end_trim_s))
    dur = end - start
    if spec.max_minutes:
        dur = min(dur, spec.max_minutes * 60.0)
    return start, dur


def process_source(spec: SourceSpec, force_download: bool = False) -> list[Segment]:
    cfg = load_config()
    lang_code = cfg.lang_code(spec.language)
    console.print(f"[bold cyan]▶ {spec.id}[/] ({spec.language}, {spec.content_type})")

    meta = load_meta(spec.id)
    if meta is None or force_download:
        console.print("  downloading…")
        meta = download_source(spec.id, spec.url, declared_license=spec.license)
    raw = Path(meta.audio_path)

    start, dur = _trim_window(spec, meta.duration_s or probe_duration(raw))
    master24 = MASTER_DIR / f"{spec.id}.wav"
    asr16 = MASTER_DIR / f"{spec.id}.16k.wav"
    console.print(f"  converting masters (offset={start:.0f}s, dur={dur/60:.1f}min)…")
    extract_to_wav(raw, master24, sr=cfg.audio.master_sample_rate, channels=1, start_s=start, dur_s=dur)
    extract_to_wav(raw, asr16, sr=cfg.audio.asr_sample_rate, channels=1, start_s=start, dur_s=dur)

    console.print("  batch diarization (Sarvam)…")
    batch = run_batch_diarization(
        asr16, lang_code, model=cfg.asr.batch_model, mode=cfg.asr.batch_mode,
        num_speakers=spec.expected_speakers,
        out_dir=MANIFEST_DIR / f"{spec.id}.batch", timeout_s=cfg.asr.batch_timeout_s,
    )
    console.print(f"  diarized chunks: {len(batch.chunks)}")

    segs = segment_source(spec, meta, master24, batch.chunks, cfg, time_offset_s=start)
    console.print(f"  candidate segments: {len(segs)}")
    if not segs:
        save_segments(spec.id, [])
        return []

    console.print("  per-segment transcription (Sarvam realtime)…")
    transcribe_segments(segs, cfg)
    console.print("  acoustic features…")
    compute_features(segs, cfg)
    normalize_per_speaker(segs)
    console.print("  quality gates…")
    run_quality(segs, cfg)
    console.print("  emotion tagging (acoustic + LLM)…")
    tag_segments(segs, cfg)

    save_segments(spec.id, segs)
    kept = sum(1 for s in segs if s.status != "reject")
    console.print(f"  [green]done[/]: {kept}/{len(segs)} kept "
                  f"({sum(s.duration_s for s in segs if s.status!='reject')/60:.1f} min)")
    return segs


# --- commands -----------------------------------------------------------------

@app.command()
def smoke():
    """Verify Sarvam STT (en-IN + te-IN), chat, and HF auth."""
    from .sarvam_client import chat_json, transcribe_clip
    from .config import FFMPEG
    cfg = load_config()
    secrets = load_secrets()
    tone = MASTER_DIR / "_smoke.wav"
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=frequency=220:duration=2", "-ac", "1", "-ar", "16000",
                    str(tone)], check=True)
    for lc in (cfg.lang_code("en"), cfg.lang_code("te")):
        r = transcribe_clip(tone, lc, model=cfg.asr.realtime_model)
        console.print(f"  STT {lc}: OK (lang={r.language_code}, transcript={r.transcript!r})")
    j = chat_json("Return JSON.", 'Reply with {"ok": true}', model=cfg.llm.model, max_tokens=50)
    console.print(f"  chat: {j}")
    if secrets.hf_token:
        from huggingface_hub import whoami
        who = whoami(token=secrets.hf_token)
        console.print(f"  HF: OK as {who.get('name')} -> repo {secrets.repo_id}")
    else:
        console.print("  HF: token not set (skipping)")
    tone.unlink(missing_ok=True)
    console.print("[green]smoke OK[/]")


@app.command()
def process(source_id: str, force_download: bool = typer.Option(False, "--force")):
    """Run the full pipeline for one source id."""
    spec = next((s for s in _sources() if s.id == source_id), None)
    if spec is None:
        raise typer.BadParameter(f"unknown source id: {source_id}")
    process_source(spec, force_download)


@app.command("process-all")
def process_all(lang: str = typer.Option("", "--lang"), force: bool = typer.Option(False, "--force")):
    """Process every enabled source (optionally filtered by --lang en|te)."""
    for spec in _sources():
        if not spec.enabled or (lang and spec.language != lang):
            continue
        if not force and load_segments(spec.id):
            console.print(f"[dim]skip {spec.id} (already processed; --force to redo)[/]")
            continue
        try:
            process_source(spec, force)
        except Exception as e:  # noqa: BLE001
            console.print(f"  [red]FAILED {spec.id}: {e}[/]")


@app.command()
def retag():
    """Recompute per-speaker normalization across ALL segments and re-tag emotion."""
    cfg = load_config()
    segs = load_all_segments()
    normalize_per_speaker(segs)
    tag_segments(segs, cfg)
    by_source: dict[str, list[Segment]] = defaultdict(list)
    for s in segs:
        by_source[s.source_id].append(s)
    for sid, group in by_source.items():
        save_segments(sid, group)
    console.print(f"[green]retagged[/] {len(segs)} segments")


@app.command()
def requalify(retag: bool = typer.Option(True, "--retag/--no-retag")):
    """Re-run quality gates (local, free) on existing manifests, then re-normalize
    + re-tag. Use after tuning thresholds — avoids re-spending ASR credits."""
    cfg = load_config()
    by_source: dict[str, list[Segment]] = defaultdict(list)
    for s in load_all_segments():
        by_source[s.source_id].append(s)
    for sid, segs in by_source.items():
        run_quality(segs, cfg)
        if retag:
            normalize_per_speaker(segs)
            tag_segments(segs, cfg)
        save_segments(sid, segs)
        kept = sum(1 for s in segs if s.status != "reject")
        console.print(f"  {sid}: {kept}/{len(segs)} kept")


@app.command("review-build")
def review_build():
    """Generate the static HTML review app."""
    from .review import build_review_app
    out = build_review_app(load_config())
    console.print(f"[green]review app:[/] {out}\n  open it, curate, Export review.csv → data/manifests/review.csv")


@app.command("review-merge")
def review_merge(csv: Path = typer.Argument(MANIFEST_DIR / "review.csv")):
    """Apply human review decisions (human edits override automated tags)."""
    from .review import merge_decisions
    stats = merge_decisions(csv)
    console.print(f"[green]merged[/] {stats}")


@app.command()
def stats():
    """Show current candidate distribution (pre-build)."""
    segs = load_all_segments()
    table = Table(title="Candidate segments")
    for col in ("lang", "status", "clips", "minutes"):
        table.add_column(col)
    agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for s in segs:
        agg[(s.language, s.status)].append(s.duration_s)
    for (lang, st), durs in sorted(agg.items()):
        table.add_row(lang, st, str(len(durs)), f"{sum(durs)/60:.1f}")
    console.print(table)
    # emotion mix among kept
    emo: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in segs:
        if s.is_kept():
            emo[s.language][s.emotion or "?"] += 1
    for lang, d in emo.items():
        console.print(f"  {lang} emotions: {dict(sorted(d.items(), key=lambda x:-x[1]))}")


@app.command()
def build():
    """Select + balance + finalize audio + write stats and dataset card."""
    from .build_dataset import assemble
    stats = assemble(load_config())
    console.print(f"[green]assembled[/] total {stats['total_minutes']} min")
    for lang, d in stats["per_language"].items():
        console.print(f"  {lang}: {d['minutes']} min, {d['clips']} clips, "
                      f"{d['speakers']} spk, emotions {d['emotions']}")


@app.command()
def publish():
    """Push the dataset to the HuggingFace Hub (public)."""
    from .publish import push
    repo = push(load_config())
    console.print(f"[green]published:[/] https://huggingface.co/datasets/{repo}")


@app.command()
def verify():
    """Reload the published dataset and sanity-check it."""
    from .publish import verify as _verify
    console.print(_verify(load_config()))


if __name__ == "__main__":
    app()

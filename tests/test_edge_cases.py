"""Edge-case coverage: degenerate and adversarial inputs to the pure-logic stages
(segmentation, dedup, text normalization, whisper gate, code-mix detection)."""

from ttsds.build_dataset import normalize_text
from ttsds.config import load_config
from ttsds.quality import _jaccard, _tokens, dedup, latin_fraction
from ttsds.sarvam_client import DiarChunk
from ttsds.segment import build_runs, dominant_speaker, pack_segments
from ttsds.tag_emotion import is_whisper

SR = 16000


def _seg(**kw):
    from ttsds.models import Segment
    base = dict(id="s", source_id="src", language="te", language_code="te-IN",
                speaker_id="spk", wav_path="x.wav", start_s=0.0, end_s=5.0, duration_s=5.0)
    base.update(kw)
    return Segment(**base)


# --- text normalization edge cases ---
def test_normalize_collapses_whitespace_keeps_content():
    assert normalize_text("  హాయ్   అండి \n\t world  ") == "హాయ్ అండి world"
    # now language-aware: numbers/symbols expand to spoken form (see test_normalize.py)
    assert normalize_text("count 42 items, 3.5%") == "count forty-two items, three point five percent"
    assert normalize_text("") == ""
    assert normalize_text("\n\n\t  ") == ""


# --- segmentation edge cases ---
def test_pack_segments_empty_and_tiny():
    cfg = load_config()
    assert pack_segments([], SR, cfg) == []
    assert pack_segments([(0, int(0.5 * SR))], SR, cfg) == []          # below min -> dropped


def test_pack_segments_huge_island_is_split():
    cfg = load_config()
    pieces = pack_segments([(0, int(60 * SR))], SR, cfg)               # one 60s run-on
    assert len(pieces) >= 3
    assert all((e - s) / SR <= cfg.segmentation.max_duration_s for s, e in pieces)


def test_dominant_speaker_empty():
    assert dominant_speaker([]) == "0"


def test_build_runs_no_target_speaker_is_empty():
    chunks = [DiarChunk("1", 0, 4, "x"), DiarChunk("2", 4, 8, "y")]
    assert build_runs(chunks, "0", merge_gap_s=0.4) == []


# --- dedup edge cases ---
def test_dedup_ignores_empty_transcripts():
    cfg = load_config()
    a = _seg(id="a", transcript="", status="pass")
    b = _seg(id="b", transcript="", status="pass")
    dedup([a, b], cfg)
    assert a.status == "pass" and b.status == "pass"   # empties are not duplicates


def test_jaccard_bounds():
    assert _jaccard(set(), set()) == 0.0
    assert _jaccard(_tokens("a a a"), _tokens("a")) == 1.0


# --- whisper gate edge cases ---
def test_is_whisper_handles_missing_features():
    cfg = load_config()
    assert is_whisper(_seg(features={}, features_z={}), cfg) is False   # no crash, defaults


# --- code-mixing (English inside Telugu) ---
def test_latin_fraction():
    assert latin_fraction("hello world") > 0.9
    assert latin_fraction("హాయ్ అండి") < 0.05
    assert 0.2 < latin_fraction("హాయ్ welcome టు") < 0.8
    assert latin_fraction("") == 0.0

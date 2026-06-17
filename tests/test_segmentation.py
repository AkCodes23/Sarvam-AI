from ttsds.config import load_config
from ttsds.sarvam_client import DiarChunk
from ttsds.segment import build_runs, dominant_speaker, pack_segments, _hard_split

SR = 16000


def test_dominant_speaker():
    chunks = [
        DiarChunk("0", 0, 5, "a"),
        DiarChunk("1", 5, 6, "b"),
        DiarChunk("0", 6, 12, "c"),
    ]
    assert dominant_speaker(chunks) == "0"


def test_build_runs_merges_small_gaps_and_breaks_on_speaker_change():
    chunks = [
        DiarChunk("0", 0.0, 2.0, "x"),
        DiarChunk("0", 2.1, 4.0, "y"),   # gap 0.1 < merge -> merge
        DiarChunk("1", 4.0, 5.0, "other"),
        DiarChunk("0", 10.0, 12.0, "z"),  # gap from prev kept end (4.0) is huge -> new run
    ]
    runs = build_runs(chunks, "0", merge_gap_s=0.4)
    assert runs == [(0.0, 4.0), (10.0, 12.0)]


def test_pack_segments_groups_into_target_window():
    cfg = load_config()
    # four 4s islands separated by 1s gaps
    islands = []
    pos = 0
    for _ in range(4):
        islands.append((pos, pos + 4 * SR))
        pos += 5 * SR
    segs = pack_segments(islands, SR, cfg)
    durs = [(e - s) / SR for s, e in segs]
    assert len(segs) >= 1
    assert all(cfg.segmentation.min_duration_s <= d <= cfg.segmentation.max_duration_s for d in durs)
    # the first packed clip should be within target_max
    assert durs[0] <= cfg.segmentation.target_max_s + 1e-6


def test_hard_split_bounds():
    pieces = _hard_split(0, 60 * SR, SR, target_max_s=15.0)
    assert len(pieces) == 4
    assert all((e - s) / SR <= 16.0 for s, e in pieces)


def test_pack_drops_too_short():
    cfg = load_config()
    islands = [(0, int(1.0 * SR))]  # 1s < min 3s
    assert pack_segments(islands, SR, cfg) == []

from ttsds.config import load_config
from ttsds.models import Segment
from ttsds.quality import _jaccard, _tokens, dedup
from ttsds.tag_emotion import _level, is_whisper


def _seg(**kw) -> Segment:
    base = dict(
        id="s", source_id="src", language="te", language_code="te-IN",
        speaker_id="spk", wav_path="data/segments/x.wav",
        start_s=0.0, end_s=5.0, duration_s=5.0,
    )
    base.update(kw)
    return Segment(**base)


def test_jaccard_and_tokens():
    assert _jaccard(_tokens("a b c"), _tokens("a b c")) == 1.0
    assert _jaccard(_tokens("a b"), _tokens("c d")) == 0.0


def test_dedup_marks_duplicates():
    cfg = load_config()
    a = _seg(id="a", transcript="hello world this is a test", status="pass")
    b = _seg(id="b", transcript="hello world this is a test", status="pass")
    c = _seg(id="c", transcript="completely different sentence here", status="pass")
    dedup([a, b, c], cfg)
    assert a.status == "pass"
    assert b.status == "reject" and "duplicate" in b.reject_reasons
    assert c.status == "pass"


def test_is_kept_matrix():
    assert _seg(status="pass").is_kept()
    assert _seg(status="flag").is_kept()
    assert not _seg(status="reject").is_kept()
    assert not _seg(status="pass", review_decision="reject").is_kept()
    assert _seg(status="reject", review_decision="accept").is_kept()


def test_level_thresholds():
    assert _level(1.5) == "much higher than usual"
    assert _level(0.5) == "higher than usual"
    assert _level(0.0) == "around this speaker's average"
    assert _level(-0.5) == "lower than usual"
    assert _level(-1.5) == "much lower than usual"


def test_is_whisper_gate():
    cfg = load_config()
    whisper_seg = _seg(
        features={"voiced_fraction": 0.1, "hnr_mean": 3.0},
        features_z={"rms_mean": -1.2},
    )
    normal_seg = _seg(
        features={"voiced_fraction": 0.8, "hnr_mean": 18.0},
        features_z={"rms_mean": 0.2},
    )
    assert is_whisper(whisper_seg, cfg)
    assert not is_whisper(normal_seg, cfg)

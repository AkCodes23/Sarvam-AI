"""Hybrid emotion/style tagging: describe per-speaker-relative prosody, apply an
acoustic whisper override, then let the LLM choose a label grounded in the
acoustics (not just the text). Low-confidence tags are flagged for human review."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .config import Config
from .models import Segment
from .sarvam_client import chat_json


def _level(z: float) -> str:
    if z >= 1.0:
        return "much higher than usual"
    if z >= 0.4:
        return "higher than usual"
    if z <= -1.0:
        return "much lower than usual"
    if z <= -0.4:
        return "lower than usual"
    return "around this speaker's average"


def describe_acoustics(seg: Segment) -> str:
    z = seg.features_z
    f = seg.features
    lines = [
        f"- Pitch level: {_level(z.get('f0_mean', 0))}",
        f"- Loudness/energy: {_level(z.get('rms_mean', 0))}",
        f"- Pitch variation (expressiveness): {_level(z.get('f0_range', 0))}",
        f"- Energy variation within clip: {_level(z.get('rms_dynamic', 0))}",
        f"- Speaking pace: {_level(z.get('speaking_rate_wps', 0))}",
        f"- Voice clarity (HNR): {f.get('hnr_mean', 0):.1f} dB"
        f" (voiced fraction {f.get('voiced_fraction', 0):.2f})",
        f"- Breathiness (high/low band ratio): {_level(z.get('hf_lf_ratio', 0))}",
    ]
    return "\n".join(lines)


def is_whisper(seg: Segment, cfg: Config) -> bool:
    w = cfg.emotion.whisper
    return (
        seg.features.get("voiced_fraction", 1.0) < w.max_voiced_fraction
        and seg.features_z.get("rms_mean", 0.0) < w.max_energy_zscore
        and seg.features.get("hnr_mean", 99.0) < w.max_hnr_db
    )


def _system_prompt(cfg: Config) -> str:
    e = ", ".join(cfg.emotion.emotions)
    s = ", ".join(cfg.emotion.styles)
    return (
        "You are an expert speech annotator labeling short single-speaker clips for a "
        "TTS dataset. You receive the transcript and an ACOUSTIC prosody description "
        "(pitch, loudness, pace, voice quality) already normalized to this speaker's own "
        "baseline. Choose exactly ONE emotion and ONE style.\n"
        f"Emotions: {e}\nStyles: {s}\n"
        "Rules:\n"
        "- Ground emotion PRIMARILY in the acoustics; use text only as support. If prosody "
        "is flat/average, label 'neutral' even when the words sound emotional.\n"
        "- calm=low steady arousal; excited=high arousal + high pitch/energy variation; "
        "angry=high tense energy; sad=low energy + low/falling pitch; happy=bright positive "
        "prosody; fearful/surprised only with clear acoustic cues.\n"
        "- Style: narrative=storytelling; conversational=casual; formal=measured/announcer; "
        "expressive=dramatic/dynamic; whisper=breathy/unvoiced.\n"
        "- confidence in [0,1] = how clearly the acoustics support the label.\n\n"
        "Reason briefly about the acoustic evidence for THIS clip, then respond with "
        "ONLY a JSON object (no markdown, no extra text) containing exactly these keys:\n"
        "  emotion  : one of the emotion options above\n"
        "  style    : one of the style options above\n"
        "  confidence: a number between 0 and 1\n"
        "  rationale: a short English phrase (max 20 words) citing the specific acoustic evidence\n"
        "Pick real values for this clip. Do not copy these descriptions verbatim."
    )


def tag_segment(seg: Segment, cfg: Config) -> Segment:
    whisper = is_whisper(seg, cfg)
    user = (
        f"Transcript: \"{seg.transcript}\"\n\n"
        f"Acoustic prosody (relative to this speaker):\n{describe_acoustics(seg)}\n\n"
        f"Whisper indicated acoustically: {'YES' if whisper else 'no'}"
    )
    result = chat_json(
        _system_prompt(cfg), user,
        model=cfg.llm.model, temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens, reasoning_effort="medium",
    )

    emotion, style, conf, rationale = "neutral", "narrative", 0.3, "fallback (no/invalid LLM output)"
    if result:
        cand_e = str(result.get("emotion", "")).lower().strip()
        cand_s = str(result.get("style", "")).lower().strip()
        if cand_e in cfg.emotion.emotions:
            emotion = cand_e
        if cand_s in cfg.emotion.styles:
            style = cand_s
        try:
            conf = max(0.0, min(1.0, float(result.get("confidence", 0.3))))
        except (TypeError, ValueError):
            conf = 0.3
        rationale = str(result.get("rationale", ""))[:160]

    # acoustic whisper override always wins on style
    if whisper:
        style = "whisper"
        rationale = (rationale + " | whisper set acoustically").strip(" |")

    seg.emotion = emotion
    seg.style = style
    seg.emotion_confidence = round(conf, 3)
    seg.emotion_rationale = rationale
    seg.tag_source = "auto"

    if conf < cfg.emotion.low_confidence_threshold and "low_emotion_confidence" not in seg.flags:
        seg.flags.append("low_emotion_confidence")
        if seg.status == "pass":
            seg.status = "flag"
    return seg


def tag_segments(segments: list[Segment], cfg: Config, workers: int = 6) -> list[Segment]:
    todo = [s for s in segments if s.status != "reject"]  # don't tag dropped clips
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda s: tag_segment(s, cfg), todo))
    return segments

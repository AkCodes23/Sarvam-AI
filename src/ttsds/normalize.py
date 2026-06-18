"""Language-aware text normalization for the `normalized_text` field.

Turns written forms into how a TTS model should *say* them: digits to words,
ordinals, currency, percentages, and a conservative set of abbreviations.
English and Telugu both expand numbers through num2words; abbreviation and
symbol maps are per language. The raw `text` field is kept untouched as the
verbatim ASR output; `normalized_text` is the TTS-facing reading.

Most of the curated content is narrative, so numbers are the live case
(currency/abbreviation handling is a robustness provision that is a no-op on
the current corpus, by design).
"""

from __future__ import annotations

import re

from num2words import num2words

# language tag accepted by num2words; anything unknown falls back to English numbers
_NUM_LANG = {"en": "en", "te": "te"}

# conservative, unambiguous abbreviation expansions (case-insensitive), English only.
# Ambiguous forms ("St." street/saint, "No." number/no) are deliberately left alone.
_ABBR_EN = [
    (r"\be\.g\.", "for example"),
    (r"\bi\.e\.", "that is"),
    (r"\bDr\.", "doctor"),
    (r"\bMr\.", "mister"),
    (r"\bMrs\.", "missus"),
    (r"\bMs\.", "miss"),
    (r"\bJr\.", "junior"),
    (r"\bSr\.", "senior"),
    (r"\bvs\.", "versus"),
    (r"\betc\.", "et cetera"),
]

_SYMBOL = {
    "en": [("%", " percent"), ("&", " and ")],
    "te": [("%", " శాతం"), ("&", " మరియు ")],
}

_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?")  # 10,000 grouped OR plain/decimal
_WS = re.compile(r"\s+")


def _say(num, nl: str, ordinal: bool = False) -> str:
    """num2words with a safe fallback to the original digits on any failure."""
    try:
        return num2words(num, lang=nl, to="ordinal") if ordinal else num2words(num, lang=nl)
    except Exception:  # noqa: BLE001 - never let an unsupported value drop the text
        return str(num)


def _expand_currency_en(text: str) -> str:
    text = re.sub(r"\$\s?(\d[\d,]*(?:\.\d+)?)", lambda m: m.group(1) + " dollars", text)
    text = re.sub(r"(?:Rs\.?|₹)\s?(\d[\d,]*(?:\.\d+)?)", lambda m: m.group(1) + " rupees", text)
    return text


def _expand_numbers(text: str, nl: str) -> str:
    text = _ORDINAL.sub(lambda m: _say(int(m.group(1)), nl, ordinal=True), text)

    def cardinal(m: "re.Match") -> str:
        tok = m.group(0).replace(",", "")
        return _say(float(tok) if "." in tok else int(tok), nl)

    return _NUMBER.sub(cardinal, text)


def normalize_text(text: str, language: str = "en") -> str:
    """Collapse whitespace and expand numbers/abbreviations into spoken form.

    `language` is the short tag (``"en"`` / ``"te"``); unknown tags normalize
    numbers as English and skip the English-only abbreviation pass.
    """
    nl = _NUM_LANG.get(language, "en")
    if language == "en":
        text = _expand_currency_en(text)
        for pattern, repl in _ABBR_EN:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = _expand_numbers(text, nl)
    for symbol, repl in _SYMBOL.get(language, _SYMBOL["en"]):
        text = text.replace(symbol, repl)
    return _WS.sub(" ", text).strip()

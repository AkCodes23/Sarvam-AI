"""Unit tests for language-aware normalization of the normalized_text field."""

import re

from ttsds.normalize import normalize_text


def test_expands_cardinal_numbers_english():
    assert normalize_text("I had 16 horses", "en") == "I had sixteen horses"


def test_expands_grouped_thousands():
    assert normalize_text("10,000 men marched", "en") == "ten thousand men marched"


def test_trailing_comma_is_not_a_thousands_separator():
    # "10," is the number ten followed by a comma, not 10,000
    assert normalize_text("there were 10, maybe 12", "en") == "there were ten, maybe twelve"


def test_expands_ordinals():
    assert normalize_text("on the 3rd and 21st day", "en") == "on the third and twenty-first day"


def test_expands_decimals():
    assert normalize_text("a 3.5 km walk", "en") == "a three point five km walk"


def test_expands_unambiguous_abbreviations():
    assert normalize_text("Dr. Rao met Mr. Shyam", "en") == "doctor Rao met mister Shyam"


def test_expands_percent_and_eg():
    assert normalize_text("50% done, e.g. mostly", "en") == "fifty percent done, for example mostly"


def test_expands_currency_into_spoken_order():
    assert normalize_text("Rs. 100 and $5", "en") == "one hundred rupees and five dollars"


def test_collapses_whitespace():
    assert normalize_text("two   spaces\there", "en") == "two spaces here"


def test_text_without_numbers_is_unchanged_apart_from_whitespace():
    assert normalize_text("no numbers here.", "en") == "no numbers here."


def test_telugu_numbers_become_telugu_words_no_digits_remain():
    out = normalize_text("16 సైనికులు 100 గుర్రాలు", "te")
    assert not re.search(r"\d", out)          # every digit was expanded
    assert "సైనికులు" in out and "గుర్రాలు" in out  # surrounding text preserved


def test_unknown_language_falls_back_to_english_numbers():
    assert normalize_text("5 apples", "xx") == "five apples"


def test_num2words_failure_falls_back_to_digits(monkeypatch):
    # if num2words ever raises, the original digits survive rather than dropping text
    import ttsds.normalize as nz

    def boom(*a, **k):
        raise ValueError("unsupported")

    monkeypatch.setattr(nz, "num2words", boom)
    assert normalize_text("7 birds", "en") == "7 birds"

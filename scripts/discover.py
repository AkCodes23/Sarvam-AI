"""Source-discovery aid: search YouTube (metadata only, no download) for candidate
single-speaker sources, filter by duration, dedupe by channel, and print a table.

Usage:
  python scripts/discover.py te        # Telugu archetype queries
  python scripts/discover.py en        # Indian English archetype queries
  python scripts/discover.py te "custom query" 15

These are CANDIDATES only — every one must still be verified by listening before
it goes into config/sources.yaml.
"""

from __future__ import annotations

import sys

import yt_dlp

QUERIES = {
    "te": [
        "telugu kathalu single narrator story",
        "telugu audiobook navala",
        "telugu podcast solo monologue",
        "telugu motivational speech single speaker",
        "telugu news bulletin anchor",
        "telugu pravachana speech",
    ],
    "en": [
        "NPTEL lecture full",
        "indian english audiobook narration",
        "tedx india talk",
        "indian english solo podcast monologue",
        "all india radio english talk",
        "indian motivational speech english single speaker",
    ],
}

MIN_S, MAX_S = 5 * 60, 60 * 60  # 5–60 min


def search(query: str, n: int) -> list[dict]:
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    return info.get("entries", []) or []


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    lang = sys.argv[1] if len(sys.argv) > 1 else "te"
    queries = [sys.argv[2]] if len(sys.argv) > 2 else QUERIES.get(lang, [])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    seen_channel: set[str] = set()
    rows: list[tuple] = []
    for q in queries:
        for e in search(q, n):
            dur = e.get("duration") or 0
            if not (MIN_S <= dur <= MAX_S):
                continue
            ch = (e.get("channel") or e.get("uploader") or "?").strip()
            vid = e.get("id", "")
            key = ch.lower()
            if key in seen_channel:
                continue  # one per channel for voice diversity
            seen_channel.add(key)
            rows.append((round(dur / 60, 1), ch, e.get("title", "")[:70],
                         f"https://www.youtube.com/watch?v={vid}", q))

    rows.sort(key=lambda r: r[1])
    print(f"\n{len(rows)} candidates for '{lang}' (one per channel, {MIN_S//60}-{MAX_S//60} min):\n")
    for mins, ch, title, url, q in rows:
        print(f"{mins:>5}m | {ch[:24]:<24} | {title}")
        print(f"        {url}   [{q}]")


if __name__ == "__main__":
    main()

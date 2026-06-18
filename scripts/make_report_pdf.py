"""Render reports/report.md (+ figures) to reports/report.pdf via markdown + xhtml2pdf.

Pure-Python (no native deps). Figures in reports/figures/*.png are appended as an
appendix. Run `python scripts/make_figures.py` first.
"""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

from ttsds.config import FIGURES_DIR, REPORTS_DIR


def _inline_body_images(html: str) -> tuple[str, set[str]]:
    """Replace <img src="figures/x.png"> in the body with base64 data URIs so
    xhtml2pdf renders them. Returns (html, set of inlined figure filenames)."""
    inlined: set[str] = set()

    def repl(m):
        src = m.group(1)
        p = (REPORTS_DIR / src).resolve()
        if not p.exists():
            return m.group(0)
        inlined.add(p.name)
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return f'<img class="inlinefig" src="data:image/png;base64,{b64}"'

    html = re.sub(r'<img\s+[^>]*?src="((?:reports/)?figures/[^"]+)"', repl, html)
    return html, inlined

CSS = """
@page { size: A4; margin: 1.8cm 1.6cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 19pt; color: #0f2a52; margin: 0 0 2pt; }
h2 { font-size: 14pt; color: #14346b; border-bottom: 1.5px solid #c9d6ee; padding-bottom: 2pt; margin-top: 16pt; }
h3 { font-size: 11.5pt; color: #1f2937; margin-top: 10pt; }
p, li { font-size: 10.5pt; }
code { font-family: Courier, monospace; background: #f1f3f7; font-size: 9.5pt; }
hr { border: 0; border-top: 1px solid #d6dce6; }
strong { color: #0f2a52; }
img { max-width: 100%; }
.inlinefig { width: 13cm; margin: 6pt 0; }
.fig { margin: 8pt 0 14pt; }
.figcap { font-size: 9pt; color: #555; }
table { -pdf-keep-with-next: true; width: 100%; }
td, th { border: 0.5px solid #d6dce6; padding: 2pt 5pt; font-size: 9.5pt; }
th { background: #eef2fb; }
"""


def _add_colgroups(html: str) -> str:
    """xhtml2pdf does not auto-size table columns and collapses an empty first
    header to near-zero width, overlapping cells. Force explicit column widths via a
    colgroup on every table (first column wider for label/metric tables)."""
    def repl(m: "re.Match") -> str:
        table = m.group(0)
        first_row = re.search(r"<tr>(.*?)</tr>", table, re.DOTALL)
        if not first_row:
            return table
        ncols = len(re.findall(r"<t[hd][ >]", first_row.group(1)))
        if ncols < 2:
            return table
        if ncols == 3:
            widths = [40, 30, 30]            # label/metric column wider
        else:                                # first column (names/ids) gets ~2x weight
            first = round(200 / (ncols + 1))
            rest = round((100 - first) / (ncols - 1))
            widths = [first] + [rest] * (ncols - 1)
        cols = "".join(f'<col width="{w}%"/>' for w in widths)
        return table.replace("<table>", f"<table><colgroup>{cols}</colgroup>", 1)
    return re.sub(r"<table>.*?</table>", repl, html, flags=re.DOTALL)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    md_text = (REPORTS_DIR / "report.md").read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["extra", "sane_lists"])
    body, _ = _inline_body_images(body)
    body = _add_colgroups(body)
    # only the figures referenced inline in the report appear; no figure dump.
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    out = REPORTS_DIR / "report.pdf"
    with open(out, "wb") as fh:
        result = pisa.CreatePDF(html, dest=fh, encoding="utf-8")
    if result.err:
        print("PDF generation had errors")
        sys.exit(1)
    print("wrote", out, f"({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

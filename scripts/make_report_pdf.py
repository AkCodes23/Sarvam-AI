"""Render reports/report.md (+ figures) to reports/report.pdf via markdown + xhtml2pdf.

Pure-Python (no native deps). Figures in reports/figures/*.png are appended as an
appendix. Run `python scripts/make_figures.py` first.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

from ttsds.config import FIGURES_DIR, REPORTS_DIR

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
.fig { margin: 8pt 0 14pt; }
.figcap { font-size: 9pt; color: #555; }
"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    md_text = (REPORTS_DIR / "report.md").read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["extra", "sane_lists"])

    figs = sorted(FIGURES_DIR.glob("*.png"))
    fig_html = ""
    if figs:
        fig_html = "<h2>Appendix: Figures</h2>"
        for f in figs:
            b64 = base64.b64encode(f.read_bytes()).decode("ascii")
            fig_html += (
                f'<div class="fig"><img src="data:image/png;base64,{b64}"/>'
                f'<div class="figcap">{f.stem}</div></div>'
            )

    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}{fig_html}</body></html>"
    out = REPORTS_DIR / "report.pdf"
    with open(out, "wb") as fh:
        result = pisa.CreatePDF(html, dest=fh, encoding="utf-8")
    if result.err:
        print("PDF generation had errors")
        sys.exit(1)
    print("wrote", out, f"({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

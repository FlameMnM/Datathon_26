"""Render DESIGN_CARD.md to docs/design_card.html (print it to DESIGN_CARD.pdf with any browser, or see README)."""
from pathlib import Path
import markdown  # pip install markdown (only needed for this helper)

ROOT = Path(__file__).resolve().parent.parent
CSS = """@page{size:Letter;margin:0.45in 0.5in}body{font-family:'Segoe UI',Arial,sans-serif;font-size:9.9pt;line-height:1.33;color:#1d2321}
h1{font-size:15pt;margin:0 0 7px;border-bottom:2px solid #c82132;padding-bottom:3px}p{margin:0 0 6px}
table{border-collapse:collapse;width:100%;margin:2px 0 7px;font-size:9.2pt}th,td{border:1px solid #cfd6cd;padding:3px 5px;text-align:left;vertical-align:top}
th{background:#edf0e9}code{font-family:Consolas,monospace;font-size:8.8pt;background:#f1f3ed;padding:0 2px}"""
body = markdown.markdown((ROOT / "DESIGN_CARD.md").read_text(encoding="utf-8"), extensions=["tables"])
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "design_card.html").write_text(f"<!doctype html><meta charset='utf-8'><title>Design card</title><style>{CSS}</style>{body}", encoding="utf-8")
print("wrote docs/design_card.html")

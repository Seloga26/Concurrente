#!/usr/bin/env python3
"""Convierte report/informe_tecnico.md a PDF (informe_tecnico.pdf) en Python puro.

Pipeline: Markdown -> HTML (lib `markdown`, con tablas) -> PDF (xhtml2pdf). Resuelve las
imagenes relativas (`../results/plots/*.png`) via un link_callback que mapea a rutas absolutas.
Pensado como respaldo cuando no hay pandoc/LaTeX. Uso:  python report/build_pdf.py
"""
from __future__ import annotations

import os

import markdown
from xhtml2pdf import pisa

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MD_PATH = os.path.join(HERE, "informe_tecnico.md")
PDF_PATH = os.path.join(HERE, "informe_tecnico.pdf")

_CSS = """
@page { size: A4; margin: 1.8cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.35; }
h1 { font-size: 17pt; margin-top: 14pt; border-bottom: 1px solid #999; }
h2 { font-size: 13pt; margin-top: 12pt; }
h3 { font-size: 11pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; }
th, td { border: 1px solid #bbb; padding: 3px 6px; font-size: 9pt; text-align: left; }
th { background: #eee; }
img { width: 17cm; }
code { background: #f2f2f2; font-family: Courier, monospace; font-size: 9pt; }
blockquote { color: #555; border-left: 3px solid #ccc; padding-left: 8px; margin-left: 0; }
"""


# Las fuentes base del PDF (Helvetica/Courier) carecen de glyphs sub/superindice Unicode;
# se mapean a ASCII (10^5, W1) solo para el PDF. El Markdown fuente conserva el Unicode.
_SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
_SUB = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
        "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9"}
# Simbolos matematicos ausentes en la fuente Courier (spans de codigo) -> ASCII.
_SYM = {"−": "-"}  # signo menos Unicode (U+2212)


def _ascii_scripts(text):
    """Convierte superindices (10^d), subindices (Wd) y simbolos Unicode a ASCII para el PDF."""
    out = []
    for ch in text:
        if ch in _SUP:
            out.append("^" + _SUP[ch])
        elif ch in _SUB:
            out.append(_SUB[ch])
        else:
            out.append(_SYM.get(ch, ch))
    return "".join(out)


def _strip_front_matter(text):
    """Quita el front-matter YAML (--- ... ---) y devuelve (meta_html, cuerpo)."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            fm = text[3:end].strip().splitlines()
            body = text[end + 4:]
            title = next((l.split(":", 1)[1].strip().strip('"') for l in fm
                          if l.startswith("title:")), "Informe técnico")
            subtitle = next((l.split(":", 1)[1].strip().strip('"') for l in fm
                             if l.startswith("subtitle:")), "")
            head = f"<h1>{title}</h1>"
            if subtitle:
                head += f"<p><em>{subtitle}</em></p>"
            return head, body
    return "", text


def _link_callback(uri, _rel):
    """Mapea rutas relativas del Markdown (../results/...) a rutas absolutas en disco."""
    if uri.startswith(("http://", "https://", "data:")):
        return uri
    path = os.path.normpath(os.path.join(HERE, uri))
    return path if os.path.exists(path) else uri


def build():
    with open(MD_PATH, "r", encoding="utf-8") as fh:
        text = fh.read()
    head, body = _strip_front_matter(_ascii_scripts(text))
    body_html = markdown.markdown(body, extensions=["tables", "fenced_code"])
    html = f"<html><head><style>{_CSS}</style></head><body>{head}{body_html}</body></html>"
    with open(PDF_PATH, "wb") as out:
        result = pisa.CreatePDF(html, dest=out, link_callback=_link_callback)
    if result.err:
        raise SystemExit(f"build_pdf: fallaron {result.err} elementos al renderizar")
    print(f"build_pdf: PDF generado -> {os.path.relpath(PDF_PATH, ROOT)}")


if __name__ == "__main__":
    build()

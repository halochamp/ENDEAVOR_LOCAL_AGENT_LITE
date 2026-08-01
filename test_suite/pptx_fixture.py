# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""Small dependency-free OOXML fixture builder for document tests."""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape
import zipfile


def write_pptx(path: Path, slide_texts: list[str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for number, text in enumerate(slide_texts, 1):
            slide = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
                'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
                '<p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>'
                f'{escape(text)}'
                '</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>'
            )
            package.writestr(f"ppt/slides/slide{number}.xml", slide)

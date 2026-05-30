"""
ingestion/pdf_parser.py

Handles PDF loading and text extraction.
Uses PyMuPDF (fitz) for layout-aware extraction — it preserves
paragraph structure better than simple text dumps.
"""

import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ParsedPage:
    """One page worth of extracted content."""
    page_number: int       # 1-indexed
    text: str
    source_file: str


def parse_pdf(pdf_path: str) -> list[ParsedPage]:
    """
    Extract text from every page of a PDF.

    Returns a list of ParsedPage objects — one per page.
    We keep pages separate so we can track exact source locations
    for citations later.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(path))
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # get_text("text") gives clean plain text while respecting layout
        text = page.get_text("text").strip()

        if not text:          # skip blank/image-only pages
            continue

        pages.append(ParsedPage(
            page_number=page_num + 1,   # human-readable 1-indexed
            text=text,
            source_file=path.name,
        ))

    doc.close()
    return pages


def parse_multiple_pdfs(pdf_paths: list[str]) -> list[ParsedPage]:
    """Parse several PDFs and return all pages in one flat list."""
    all_pages = []
    for path in pdf_paths:
        all_pages.extend(parse_pdf(path))
    return all_pages

"""
ingestion/chunker.py

Splits parsed pages into overlapping chunks for embedding.

Why overlap matters:
  A sentence that straddles a chunk boundary would be missed by a
  search that only hits one side. 200-token overlap ensures the
  boundary content always appears in at least one full chunk.

Why RecursiveCharacterTextSplitter:
  It tries paragraph → sentence → word splits in order, so chunks
  respect natural text boundaries instead of cutting mid-sentence.
"""

from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ingestion.pdf_parser import ParsedPage


@dataclass
class Chunk:
    """A single retrievable unit of text."""
    chunk_id: str          # unique ID: "filename_p3_c2"
    text: str
    source_file: str
    page_number: int
    chunk_index: int       # which chunk on that page


def chunk_pages(
    pages: list[ParsedPage],
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """
    Convert a list of ParsedPages into overlapping text chunks.

    chunk_size=800  → ~600 tokens, fits well within LLM context
    chunk_overlap=200 → prevents boundary information loss
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # priority order
    )

    chunks: list[Chunk] = []

    for page in pages:
        raw_chunks = splitter.split_text(page.text)

        for i, text in enumerate(raw_chunks):
            if not text.strip():
                continue

            # Build a deterministic ID for retrieval/citation tracking
            safe_name = page.source_file.replace(" ", "_").replace(".pdf", "")
            chunk_id = f"{safe_name}_p{page.page_number}_c{i}"

            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=text.strip(),
                source_file=page.source_file,
                page_number=page.page_number,
                chunk_index=i,
            ))

    return chunks

"""
ingestion/entity_extractor.py

Extracts named entities from chunk text using spaCy's NER model.
These entities become nodes in the Neo4j knowledge graph.

Why spaCy instead of an LLM for extraction:
  - 100x faster (milliseconds per chunk vs seconds)
  - Zero API cost
  - Deterministic — same input always gives same entities
  - Accurate enough for the entity types we care about

We use the small English model (en_core_web_sm, ~12MB).
For better accuracy on scientific text, swap in en_core_sci_sm
from the scispaCy package.

Setup (one-time):
    python -m spacy download en_core_web_sm
"""

import spacy
from dataclasses import dataclass
from ingestion.chunker import Chunk

# Entity types we care about for a research knowledge graph.
# Full list: https://spacy.io/api/annotation#named-entities
RELEVANT_ENTITY_TYPES = {
    "PERSON",   # people, researchers, authors
    "ORG",      # organisations, companies, institutions
    "GPE",      # countries, cities, states
    "PRODUCT",  # products, models, tools
    "EVENT",    # named events, conferences
    "WORK_OF_ART",  # paper titles, book names
    "LAW",      # laws, regulations, standards
    "NORP",     # nationalities, political groups
}


@dataclass
class Entity:
    text: str          # normalised entity text (lowercased, stripped)
    label: str         # spaCy entity type, e.g. "PERSON"
    chunk_id: str      # which chunk this came from
    source_file: str


def _load_model():
    """Load spaCy model, with a helpful error if not downloaded."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise OSError(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        )


class EntityExtractor:
    def __init__(self):
        self._nlp = _load_model()
        # Disable unused pipeline components for speed
        # We only need NER, not dependency parsing or tagging
        self._nlp.select_pipes(enable=["ner"])

    def extract(self, chunks: list[Chunk]) -> list[Entity]:
        """
        Extract entities from all chunks in batch.

        spaCy's pipe() processes in batches — much faster than
        calling nlp() one chunk at a time.
        """
        entities: list[Entity] = []
        texts = [c.text for c in chunks]

        for chunk, doc in zip(chunks, self._nlp.pipe(texts, batch_size=50)):
            seen = set()  # deduplicate within a chunk

            for ent in doc.ents:
                if ent.label_ not in RELEVANT_ENTITY_TYPES:
                    continue

                normalised = ent.text.strip().lower()

                # Skip very short or already-seen entities in this chunk
                if len(normalised) < 3 or normalised in seen:
                    continue

                seen.add(normalised)
                entities.append(Entity(
                    text=normalised,
                    label=ent.label_,
                    chunk_id=chunk.chunk_id,
                    source_file=chunk.source_file,
                ))

        return entities
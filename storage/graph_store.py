"""
storage/graph_store.py

Stores entities and their relationships in Neo4j.

Graph schema:
  Nodes:
    (:Chunk  {chunk_id, text, source_file, page_number})
    (:Entity {text, label})          e.g. ("attention mechanism", "PRODUCT")

  Edges:
    (:Entity)-[:APPEARS_IN]->(:Chunk)   — entity found in this chunk
    (:Entity)-[:CO_OCCURS_WITH]->(:Entity)  — both in same chunk (implies relationship)

Why Neo4j:
  Vector search finds semantically similar chunks.
  Graph traversal finds *related* chunks through shared entities.
  Example: query matches chunk A about "transformer architecture".
  Graph traversal finds chunk B also containing "transformer architecture"
  even if B isn't semantically similar to the query — it's structurally linked.

Free tier: https://neo4j.com/cloud/aura-free/ (1 free instance, no credit card)
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from ingestion.entity_extractor import Entity
from ingestion.chunker import Chunk

load_dotenv()


class GraphStore:
    def __init__(self):
        uri      = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        if not uri or not password:
            raise ValueError(
                "NEO4J_URI and NEO4J_PASSWORD must be set in .env\n"
                "Get a free instance at https://neo4j.com/cloud/aura-free/"
            )

        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._ensure_indexes()

    # ── Schema setup ─────────────────────────────────────────────────────────

    def _ensure_indexes(self):
        """Create indexes on first run for fast lookups."""
        with self._driver.session() as session:
            session.run("CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id)")
            session.run("CREATE INDEX entity_text IF NOT EXISTS FOR (e:Entity) ON (e.text)")
           
    # ── Ingestion ─────────────────────────────────────────────────────────────

    def store_chunks(self, chunks: list[Chunk]) -> None:
        """
        Upsert Chunk nodes into Neo4j.
        MERGE prevents duplicates if you re-index the same documents.
        """
        with self._driver.session() as session:
            for chunk in chunks:
                session.run(
                    """
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.text        = $text,
                        c.source_file = $source_file,
                        c.page_number = $page_number
                    """,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text[:500],   # truncate for graph storage
                    source_file=chunk.source_file,
                    page_number=chunk.page_number,
                )

    def store_entities(self, entities: list[Entity]) -> None:
        """
        Upsert Entity nodes and APPEARS_IN edges.
        Then build CO_OCCURS_WITH edges between entities in the same chunk.

        CO_OCCURS_WITH is the key relationship for graph traversal:
        if entity A and entity B appear together frequently,
        asking about A should surface chunks about B.
        """
        with self._driver.session() as session:
            # 1. Upsert entities + APPEARS_IN edges
            for entity in entities:
                session.run(
                    """
                    MERGE (e:Entity {text: $text})
                    SET e.label = $label
                    WITH e
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (e)-[:APPEARS_IN]->(c)
                    """,
                    text=entity.text,
                    label=entity.label,
                    chunk_id=entity.chunk_id,
                )

            # 2. Build CO_OCCURS_WITH edges per chunk
            # Find all entity pairs that share a chunk and connect them
            session.run(
                """
                MATCH (e1:Entity)-[:APPEARS_IN]->(c:Chunk)<-[:APPEARS_IN]-(e2:Entity)
                WHERE e1.text < e2.text   // avoid duplicate A-B and B-A edges
                MERGE (e1)-[r:CO_OCCURS_WITH]-(e2)
                ON CREATE SET r.count = 1
                ON MATCH  SET r.count = r.count + 1
                """
            )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_related_chunk_ids(
        self,
        entity_texts: list[str],
        hops: int = 1,
        limit: int = 10,
    ) -> list[str]:
        """
        Graph traversal: given a list of entity names found in the query,
        find chunk IDs reachable within `hops` CO_OCCURS_WITH edges.

        hops=1 means: find entities that co-occur with query entities,
        then return the chunks those related entities appear in.

        This surfaces chunks that are *topically related* even if they
        weren't returned by vector/BM25 search.
        """
        if not entity_texts:
            return []
        print("Query entities:", entity_texts)
            
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (e:Entity)
                WHERE e.text IN $entity_texts
                MATCH (e)-[:CO_OCCURS_WITH*1..{hops}]-(related:Entity)
                MATCH (related)-[:APPEARS_IN]->(c:Chunk)
                RETURN DISTINCT c.chunk_id AS chunk_id
                LIMIT $limit
                """,
                entity_texts=entity_texts,
                limit=limit,
            )
            print("Graph-related chunks:", chunk_ids) #***************REMOVE THIS SHITTTTTTTTTTTT ISHA ISHA ISHAAAAAAAA ***********
            return [row["chunk_id"] for row in result]

    def get_entity_graph(self, limit: int = 100) -> dict:
        """
        Return the full entity-entity graph for visualisation.
        Used by the Streamlit graph view tab.

        Returns {"nodes": [...], "edges": [...]} in streamlit-agraph format.
        """
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e1:Entity)-[r:CO_OCCURS_WITH]-(e2:Entity)
                WHERE r.count >= 1
                RETURN e1.text AS source, e1.label AS source_label,
                       e2.text AS target, e2.label AS target_label,
                       r.count AS weight
                ORDER BY r.count DESC
                LIMIT $limit
                """,
                limit=limit,
            )

            nodes_seen = set()
            nodes, edges = [], []

            for row in result:
                for name, label in [(row["source"], row["source_label"]),
                                     (row["target"], row["target_label"])]:
                    if name not in nodes_seen:
                        nodes.append({"id": name, "label": name, "group": label})
                        nodes_seen.add(name)

                edges.append({
                    "source": row["source"],
                    "target": row["target"],
                    "weight": row["weight"],
                })

            return {"nodes": nodes, "edges": edges}

    def close(self):
        self._driver.close()
"""
app/streamlit_app.py

Week 3: Agentic RAG + Knowledge Graph visualisation.

New in Week 3:
  - Agent loop (plan → retrieve → verify → [refine →] generate)
  - Neo4j graph store built during indexing
  - Knowledge Graph tab with entity visualisation
  - Sub-questions and retry count shown in chat
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import streamlit as st

from ingestion.pdf_parser import parse_pdf
from ingestion.chunker import chunk_pages
from ingestion.entity_extractor import EntityExtractor
from storage.vector_store import VectorStore
from storage.bm25_store import BM25Store
from storage.graph_store import GraphStore
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from agent.rag_agent import run_agent

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GraphRAG Research Assistant",
    page_icon="🔬",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    .source-card {
        background: #1e2130;
        border-left: 3px solid #4f8ef7;
        padding: 10px 14px;
        border-radius: 4px;
        margin: 6px 0;
        font-size: 0.85rem;
        color: #b0b8d0;
    }
    .agent-step {
        background: #181f2e;
        border-left: 3px solid #9c6ef7;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 4px 0;
        font-size: 0.82rem;
        color: #a0a8c0;
    }
    .confidence-high   { color: #4caf50; font-weight: 600; }
    .confidence-medium { color: #ff9800; font-weight: 600; }
    .confidence-low    { color: #f44336; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

for key, default in [
    ("retriever", None), ("graph_store", None), ("entity_extractor", None),
    ("messages", []), ("indexed_files", []), ("reranker", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 GraphRAG Assistant")
    st.caption("Week 3 · Agentic RAG + Knowledge Graph")

    st.divider()
    st.subheader("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True,
    )

    top_k = st.slider("Chunks to retrieve", 3, 10, 5)

    if st.button("🚀 Build Knowledge Base", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            progress = st.progress(0, text="Parsing PDFs...")
            all_chunks = []
            file_names = []

            for i, uploaded_file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                pages = parse_pdf(tmp_path)
                for page in pages:
                    page.source_file = uploaded_file.name
                chunks = chunk_pages(pages)
                all_chunks.extend(chunks)
                file_names.append(uploaded_file.name)
                os.unlink(tmp_path)
                progress.progress((i + 1) / len(uploaded_files) / 3,
                                   text=f"Parsed {uploaded_file.name}")

            progress.progress(0.4, text="Building vector + BM25 indexes...")
            vs = VectorStore()
            vs.build(all_chunks)
            bm25 = BM25Store()
            bm25.build(all_chunks)

            if st.session_state.reranker is None:
                progress.progress(0.55, text="Loading reranker (~80MB first run)...")
                st.session_state.reranker = Reranker()

            progress.progress(0.65, text="Extracting entities for knowledge graph...")
            extractor = EntityExtractor()
            entities = extractor.extract(all_chunks)

            progress.progress(0.80, text="Storing graph in Neo4j...")
            try:
                gs = GraphStore()
                gs.store_chunks(all_chunks)
                gs.store_entities(entities)
                st.session_state.graph_store = gs
                graph_ok = True
            except Exception as e:
                st.warning(f"Neo4j skipped (running without graph): {e}")
                st.session_state.graph_store = None
                graph_ok = False

            st.session_state.retriever = HybridRetriever(
                vs, bm25, st.session_state.reranker
            )
            st.session_state.entity_extractor = extractor
            st.session_state.indexed_files = file_names
            st.session_state.messages = []

            progress.progress(1.0, text="Done!")
            graph_msg = f" + {len(entities)} entities in graph" if graph_ok else ""
            st.success(f"✅ {len(all_chunks)} chunks indexed{graph_msg}")

    if st.session_state.indexed_files:
        st.divider()
        st.subheader("📚 Indexed Files")
        for f in st.session_state.indexed_files:
            st.markdown(f"• `{f}`")

    st.divider()
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_chat, tab_graph = st.tabs(["💬 Chat", "🕸 Knowledge Graph"])

# ── Chat tab ──────────────────────────────────────────────────────────────────

with tab_chat:
    st.title("Research Assistant")

    if st.session_state.retriever is None:
        st.info("👈 Upload PDFs and click **Build Knowledge Base** to start.")
        st.stop()

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["answer"])

                # Agent trace — show sub-questions and retry info
                sub_qs = msg.get("sub_questions", [])
                retries = msg.get("retries", 0)
                if sub_qs:
                    with st.expander("🧠 Agent reasoning trace"):
                        st.markdown(f"**Query decomposed into {len(sub_qs)} sub-question(s):**")
                        for q in sub_qs:
                            st.markdown(f'<div class="agent-step">🔍 {q}</div>',
                                        unsafe_allow_html=True)
                        if retries:
                            st.markdown(
                                f'<div class="agent-step">🔄 Retrieval refined {retries}x '
                                f'(context was insufficient)</div>',
                                unsafe_allow_html=True,
                            )

                # Confidence
                conf = msg.get("confidence", "Medium")
                st.markdown(
                    f'<span class="confidence-{conf.lower()}">Confidence: {conf}</span>',
                    unsafe_allow_html=True,
                )

                # Sources
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📎 Sources ({len(sources)} chunks)"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(
                                f'<div class="source-card">'
                                f'<strong>[{i}] {src["source_file"]}</strong>'
                                f' · Page {src["page_number"]}'
                                f' · Score: {src["relevance_score"]}<br>'
                                f'<em>{src["text_preview"]}</em>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

    question = st.chat_input("Ask a question about your documents...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Agent thinking..."):
            result = run_agent(
                question=question,
                retriever=st.session_state.retriever,
                graph_store=st.session_state.graph_store,
                entity_extractor=st.session_state.entity_extractor,
            )

        st.session_state.messages.append({"role": "assistant", **result})
        st.rerun()

# ── Knowledge Graph tab ───────────────────────────────────────────────────────

with tab_graph:
    st.title("Knowledge Graph")
    st.caption("Entity co-occurrence network extracted from your documents.")

    if st.session_state.graph_store is None:
        st.info("Build a knowledge base with Neo4j credentials set to see the graph.")
    else:
        try:
            from streamlit_agraph import agraph, Node, Edge, Config

            graph_data = st.session_state.graph_store.get_entity_graph(limit=80)

            if not graph_data["nodes"]:
                st.info("No entity relationships found yet. Upload more documents.")
            else:
                # Colour nodes by entity type
                LABEL_COLORS = {
                    "PERSON": "#4f8ef7", "ORG": "#f7a54f", "GPE": "#4fd9a5",
                    "PRODUCT": "#f74f7a", "EVENT": "#c04ff7", "WORK_OF_ART": "#f7e24f",
                }

                nodes = [
                    Node(
                        id=n["id"], label=n["label"][:20],
                        size=18,
                        color=LABEL_COLORS.get(n.get("group", ""), "#8899aa"),
                    )
                    for n in graph_data["nodes"]
                ]
                edges = [
                    Edge(source=e["source"], target=e["target"],
                         width=min(e["weight"], 5))
                    for e in graph_data["edges"]
                ]

                config = Config(
                    width=900, height=600,
                    directed=False,
                    physics=True,
                    hierarchical=False,
                )

                st.markdown(f"**{len(nodes)} entities · {len(edges)} relationships**")
                agraph(nodes=nodes, edges=edges, config=config)

                # Legend
                cols = st.columns(len(LABEL_COLORS))
                for col, (label, color) in zip(cols, LABEL_COLORS.items()):
                    col.markdown(
                        f'<span style="color:{color}">■</span> {label}',
                        unsafe_allow_html=True,
                    )
        except ImportError:
            st.error("Run: pip install streamlit-agraph")
        except Exception as e:
            st.error(f"Graph error: {e}")

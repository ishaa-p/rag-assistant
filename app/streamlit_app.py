import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import json
import streamlit as st
import pandas as pd

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
st.set_page_config(page_title="GraphRAG Research Assistant", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    .source-card {
        background: #1e2130; border-left: 3px solid #4f8ef7;
        padding: 10px 14px; border-radius: 4px; margin: 6px 0;
        font-size: 0.85rem; color: #b0b8d0;
    }
    .agent-step {
        background: #181f2e; border-left: 3px solid #9c6ef7;
        padding: 8px 12px; border-radius: 4px; margin: 4px 0;
        font-size: 0.82rem; color: #a0a8c0;
    }
    .metric-card {
        background: #1e2130; border-radius: 8px;
        padding: 16px; text-align: center; margin: 4px;
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
    ("eval_results", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 GraphRAG Assistant")
    st.caption("Week 4 · Complete · Eval + Docker + Deploy")
    st.divider()
    st.subheader("📄 Upload Documents")

    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    top_k = st.slider("Chunks to retrieve", 3, 10, 5)

    if st.button("🚀 Build Knowledge Base", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            progress = st.progress(0, text="Parsing PDFs...")
            all_chunks = []
            for i, uf in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uf.read()); tmp_path = tmp.name
                pages = parse_pdf(tmp_path)
                for p in pages: p.source_file = uf.name
                all_chunks.extend(chunk_pages(pages))
                os.unlink(tmp_path)
                progress.progress((i+1)/len(uploaded_files)/3, text=f"Parsed {uf.name}")

            progress.progress(0.4, text="Building vector + BM25 indexes...")
            vs = VectorStore(); vs.build(all_chunks)
            bm25 = BM25Store(); bm25.build(all_chunks)

            if st.session_state.reranker is None:
                progress.progress(0.55, text="Loading reranker (~80MB first run)...")
                st.session_state.reranker = Reranker()

            progress.progress(0.65, text="Extracting entities...")
            extractor = EntityExtractor()
            entities = extractor.extract(all_chunks)

            progress.progress(0.80, text="Storing graph in Neo4j...")
            try:
                gs = GraphStore(); gs.store_chunks(all_chunks); gs.store_entities(entities)
                st.session_state.graph_store = gs; graph_ok = True
            except Exception as e:
                st.warning(f"Neo4j skipped: {e}"); st.session_state.graph_store = None; graph_ok = False

            st.session_state.retriever = HybridRetriever(vs, bm25, st.session_state.reranker)
            st.session_state.entity_extractor = extractor
            st.session_state.indexed_files = [f.name for f in uploaded_files]
            st.session_state.messages = []
            progress.progress(1.0, text="Done!")
            st.success(f"✅ {len(all_chunks)} chunks indexed")

    if st.session_state.indexed_files:
        st.divider(); st.subheader("📚 Indexed")
        for f in st.session_state.indexed_files: st.markdown(f"• `{f}`")

    st.divider()
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []; st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_graph, tab_eval = st.tabs(["💬 Chat", "🕸 Knowledge Graph", "📊 Evaluation"])

# ════════════════════════════════════════════════════════════════════════════
# CHAT TAB
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.title("Research Assistant")
    if st.session_state.retriever is None:
        st.info("👈 Upload PDFs and click **Build Knowledge Base** to start.")
        st.stop()

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"): st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["answer"])
                sub_qs = msg.get("sub_questions", [])
                retries = msg.get("retries", 0)
                if sub_qs:
                    with st.expander("🧠 Agent reasoning trace"):
                        st.markdown(f"**Decomposed into {len(sub_qs)} sub-question(s):**")
                        for q in sub_qs:
                            st.markdown(f'<div class="agent-step">🔍 {q}</div>', unsafe_allow_html=True)
                        if retries:
                            st.markdown(f'<div class="agent-step">🔄 Retrieval refined {retries}x</div>', unsafe_allow_html=True)
                conf = msg.get("confidence", "Medium")
                st.markdown(f'<span class="confidence-{conf.lower()}">Confidence: {conf}</span>', unsafe_allow_html=True)
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📎 Sources ({len(sources)} chunks)"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(
                                f'<div class="source-card"><strong>[{i}] {src["source_file"]}</strong>'
                                f' · Page {src["page_number"]} · Score: {src["relevance_score"]}<br>'
                                f'<em>{src["text_preview"]}</em></div>', unsafe_allow_html=True)

    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Agent thinking..."):
            result = run_agent(question, st.session_state.retriever,
                               st.session_state.graph_store, st.session_state.entity_extractor)
        st.session_state.messages.append({"role": "assistant", **result})
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH TAB
# ════════════════════════════════════════════════════════════════════════════
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
                st.info("No entity relationships found yet.")
            else:
                LABEL_COLORS = {
                    "PERSON": "#4f8ef7", "ORG": "#f7a54f", "GPE": "#4fd9a5",
                    "PRODUCT": "#f74f7a", "EVENT": "#c04ff7", "WORK_OF_ART": "#f7e24f",
                }
                nodes = [Node(id=n["id"], label=n["label"][:20], size=18,
                              color=LABEL_COLORS.get(n.get("group", ""), "#8899aa"))
                         for n in graph_data["nodes"]]
                edges = [Edge(source=e["source"], target=e["target"], width=min(e["weight"], 5))
                         for e in graph_data["edges"]]
                config = Config(width=900, height=600, directed=False, physics=True, hierarchical=False)
                st.markdown(f"**{len(nodes)} entities · {len(edges)} relationships**")
                agraph(nodes=nodes, edges=edges, config=config)
        except ImportError:
            st.error("Run: pip install streamlit-agraph")
        except Exception as e:
            st.error(f"Graph error: {e}")

# ════════════════════════════════════════════════════════════════════════════
# EVALUATION TAB  (Week 4)
# ════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.title("📊 RAGAS Evaluation")
    st.markdown("""
Measure your pipeline's quality with three metrics:
- **Faithfulness** — does the answer only use facts from the retrieved context? *(target > 0.80)*
- **Context Precision** — are retrieved chunks actually relevant? *(target > 0.75)*
- **Context Recall** — did we retrieve everything needed to answer? *(target > 0.70)*
""")

    if st.session_state.retriever is None:
        st.info("Build a knowledge base first, then run evaluation.")
        st.stop()

    st.divider()
    st.subheader("1. Prepare your test set")
    st.markdown("""
Edit the JSON below — add 10–20 real question/answer pairs from your documents.
The `ground_truth` is the correct answer; RAGAS compares it against what the pipeline returns.
""")

    default_testset = json.dumps([
        {"question": "What is the main topic of the document?",
         "ground_truth": "Replace with the actual answer from your document."},
        {"question": "What methodology is described?",
         "ground_truth": "Replace with the actual answer from your document."},
        {"question": "What are the key findings or conclusions?",
         "ground_truth": "Replace with the actual answer from your document."},
    ], indent=2)

    testset_json = st.text_area("Test set (JSON)", value=default_testset, height=220)
    top_k_eval = st.slider("top_k for evaluation", 3, 10, 5, key="eval_topk")

    col1, col2 = st.columns([1, 3])
    with col1:
        run_eval = st.button("▶ Run Evaluation", type="primary", use_container_width=True)
    with col2:
        st.caption("⏱ Takes ~1–2 min (makes OpenAI calls for each question)")

    if run_eval:
        try:
            testset = json.loads(testset_json)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}"); st.stop()

        with st.spinner(f"Evaluating {len(testset)} questions with RAGAS..."):
            try:
                from evaluation.ragas_eval import evaluate_pipeline
                df = evaluate_pipeline(st.session_state.retriever, testset, top_k=top_k_eval)
                st.session_state.eval_results = df
            except ImportError as e:
                st.error(f"Missing dependency: {e}\nRun: pip install ragas")
                st.stop()
            except Exception as e:
                st.error(f"Evaluation error: {e}"); st.stop()

    if st.session_state.eval_results is not None:
        df = st.session_state.eval_results
        avg_row = df[df["question"] == "── AVERAGE ──"].iloc[0]

        st.divider()
        st.subheader("2. Results")

        # Big metric cards
        c1, c2, c3 = st.columns(3)
        def metric_card(col, label, value, target):
            color = "#4caf50" if value >= target else "#f44336"
            col.markdown(
                f'<div class="metric-card">'
                f'<div style="font-size:0.85rem;color:#888">{label}</div>'
                f'<div style="font-size:2rem;font-weight:700;color:{color}">{value:.3f}</div>'
                f'<div style="font-size:0.75rem;color:#666">target ≥ {target}</div>'
                f'</div>', unsafe_allow_html=True)

        metric_card(c1, "Faithfulness",       float(avg_row["faithfulness"]),       0.80)
        metric_card(c2, "Context Precision",  float(avg_row["context_precision"]),  0.75)
        metric_card(c3, "Context Recall",     float(avg_row["context_recall"]),     0.70)

        # Per-question table
        st.divider()
        st.subheader("3. Per-question breakdown")
        display_cols = ["question", "faithfulness", "context_precision", "context_recall"]
        available = [c for c in display_cols if c in df.columns]
        display_df = df[available].copy()
        if "question" in display_df.columns:
             display_df = display_df[display_df["question"] != "── AVERAGE ──"]
        display_df.index = range(1, len(display_df) + 1)   # start row numbers at 1, not 0
        st.dataframe(display_df, use_container_width=True)

        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇ Download full results CSV",
            data=csv,
            file_name="ragas_results.csv",
            mime="text/csv",
        )

        st.divider()
        st.subheader("4. How to improve your scores")
        faith = float(avg_row["faithfulness"])
        prec  = float(avg_row["context_precision"])
        rec   = float(avg_row["context_recall"])

        tips = []
        if faith < 0.80:
            tips.append("**Faithfulness low** → reduce `temperature` in generator (try 0.0), "
                        "tighten the system prompt to forbid facts not in context")
        if prec < 0.75:
            tips.append("**Context Precision low** → increase reranker strictness, "
                        "reduce `top_k` so only the best chunks are sent to the LLM")
        if rec < 0.70:
            tips.append("**Context Recall low** → increase `top_k`, "
                        "reduce chunk size so individual chunks are more focused, "
                        "check your BM25 tokenizer covers domain terms")
        if not tips:
            tips.append("✅ All targets met — your pipeline is performing well!")

        for tip in tips:
            st.markdown(f"- {tip}")
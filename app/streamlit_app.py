"""
app/streamlit_app.py

Week 1 UI: upload PDFs, ask questions, see answers with citations.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import streamlit as st

from ingestion.pdf_parser import parse_pdf
from ingestion.chunker import chunk_pages
from storage.vector_store import VectorStore
from retrieval.retriever import Retriever
from agent.generator import generate_answer

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GraphRAG Research Assistant",
    page_icon="🔬",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0f1117; }
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
    .confidence-high   { color: #4caf50; font-weight: 600; }
    .confidence-medium { color: #ff9800; font-weight: 600; }
    .confidence-low    { color: #f44336; font-weight: 600; }
    .chat-user     { background: #1e2130; padding: 12px; border-radius: 8px; margin: 8px 0; }
    .chat-assistant{ background: #151822; border: 1px solid #2a2f45;
                     padding: 12px; border-radius: 8px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 GraphRAG Assistant")
    st.caption("Week 1 MVP · Vector search + citations")

    st.divider()
    st.subheader("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs to build your knowledge base.",
    )

    top_k = st.slider("Chunks to retrieve", min_value=3, max_value=10, value=5,
                      help="Higher = more context but slower answers")

    if st.button("🚀 Build Knowledge Base", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            with st.spinner("Parsing and embedding documents..."):
                all_chunks = []
                file_names = []

                for uploaded_file in uploaded_files:
                    # Save to temp file so pdf_parser can open it
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    pages = parse_pdf(tmp_path)
                    # Override source_file name with original filename
                    for page in pages:
                        page.source_file = uploaded_file.name

                    chunks = chunk_pages(pages)
                    all_chunks.extend(chunks)
                    file_names.append(uploaded_file.name)
                    os.unlink(tmp_path)   # clean up temp file

                # Build FAISS index
                vs = VectorStore()
                vs.build(all_chunks)

                st.session_state.vector_store = vs
                st.session_state.retriever = Retriever(vs)
                st.session_state.indexed_files = file_names
                st.session_state.messages = []   # reset chat

            st.success(f"✅ Indexed {len(all_chunks)} chunks from {len(file_names)} file(s)")

    if st.session_state.indexed_files:
        st.divider()
        st.subheader("📚 Indexed Files")
        for fname in st.session_state.indexed_files:
            st.markdown(f"• `{fname}`")

    st.divider()
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────

st.title("Research Assistant")

if st.session_state.retriever is None:
    st.info("👈 Upload PDFs and click **Build Knowledge Base** to start.")
    st.stop()

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">🧑 **You:** {msg["content"]}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-assistant">🤖 **Assistant:**\n\n{msg["answer"]}</div>',
                    unsafe_allow_html=True)

        # Confidence badge
        conf = msg.get("confidence", "Medium")
        conf_class = f"confidence-{conf.lower()}"
        st.markdown(f'<span class="{conf_class}">Confidence: {conf}</span>',
                    unsafe_allow_html=True)

        # Collapsible sources panel
        with st.expander(f"📎 Sources used ({len(msg['sources'])} chunks)"):
            for i, src in enumerate(msg["sources"], start=1):
                st.markdown(
                    f'<div class="source-card">'
                    f'<strong>[{i}] {src["source_file"]}</strong> · Page {src["page_number"]} '
                    f'· Score: {src["relevance_score"]}<br>'
                    f'<em>{src["text_preview"]}</em>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ── Chat input ────────────────────────────────────────────────────────────────

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Retrieving and generating answer..."):
        retriever = st.session_state.retriever
        chunks_with_scores = retriever.retrieve(question, top_k=top_k)
        result = generate_answer(question, chunks_with_scores)

    st.session_state.messages.append({
        "role": "assistant",
        **result,
    })

    st.rerun()

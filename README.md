# 🔬 Agentic GraphRAG Research Assistant

A production-style RAG system built in 4 weeks. Chat with your PDF documents using semantic search, knowledge graphs, and agentic retrieval.

## Current Status: Week 1 MVP ✅

- PDF upload and parsing
- Semantic chunking with overlap
- FAISS vector search
- GPT-4o-mini answer generation
- Citations with source tracking
- Clean Streamlit UI

## Setup

### 1. Clone and install

```bash
git clone <your-repo>
cd graphrag-assistant
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Run

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501

## Project Structure

```
graphrag-assistant/
├── ingestion/
│   ├── pdf_parser.py      # PDF → pages (PyMuPDF)
│   └── chunker.py         # pages → overlapping chunks
├── storage/
│   └── vector_store.py    # FAISS index + OpenAI embeddings
├── retrieval/
│   └── retriever.py       # search interface (upgradeable)
├── agent/
│   └── generator.py       # LLM answer generation + citations
├── app/
│   └── streamlit_app.py   # UI
└── evaluation/            # Week 4: RAGAS evaluation
```

## Week-by-Week Roadmap

| Week | What you're adding | Key concepts |
|------|--------------------|-------------|
| **1** ✅ | PDF chat MVP | Vector search, embeddings, chunking |
| **2** | Hybrid search + reranking | BM25, cross-encoders, citations |
| **3** | Agentic loop + GraphRAG | LangGraph, Neo4j, query decomposition |
| **4** | Evaluation + deployment | RAGAS, Docker, Render |

## Interview Talking Points

- **Why RecursiveCharacterTextSplitter?** Respects paragraph/sentence boundaries vs fixed-size cuts
- **Why chunk_overlap=200?** Boundary sentences appear in at least one full chunk
- **Why IndexFlatIP + L2 normalize?** Inner product after normalization == cosine similarity
- **Why top_k=20 from FAISS then take 5?** Pre-fetching for reranker (Week 2)
- **Why temperature=0.1?** Factual tasks need deterministic output, not creativity

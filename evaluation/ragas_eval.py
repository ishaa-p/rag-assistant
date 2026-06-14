"""
evaluation/ragas_eval.py

Evaluates the RAG pipeline using RAGAS metrics.

What RAGAS measures:
  - answer_faithfulness:  does the answer only use facts from the context?
                          catches hallucinations. Target: > 0.80
  - context_precision:    are the retrieved chunks actually relevant?
                          measures retrieval quality. Target: > 0.75
  - context_recall:       did we retrieve all chunks needed to answer?
                          measures how much we missed. Target: > 0.70

Why evaluation matters for your resume:
  Anyone can build a RAG system. Very few measure it.
  Being able to say "I improved faithfulness from 0.61 → 0.83 by adding
  reranking and graph traversal" is a concrete, quantifiable achievement
  that recruiters and senior engineers remember.

Usage:
    python -m evaluation.ragas_eval \
        --pdf path/to/doc.pdf \
        --testset evaluation/sample_testset.json \
        --output evaluation/results.csv

Or import and call evaluate_pipeline() directly from the Streamlit eval tab.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ingestion.pdf_parser import parse_pdf
from ingestion.chunker import chunk_pages
from storage.vector_store import VectorStore
from storage.bm25_store import BM25Store
from retrieval.reranker import Reranker
from retrieval.hybrid_retriever import HybridRetriever
from agent.generator import generate_answer

load_dotenv()


# ── Test set helpers ──────────────────────────────────────────────────────────

SAMPLE_TESTSET = [
    # Add your own question-answer pairs here based on your documents.
    # These are placeholders — replace with real Q&A from your PDFs.
    {
        "question": "What is the main topic of the document?",
        "ground_truth": "Replace this with the actual answer from your document.",
    },
    {
        "question": "What methodology is described?",
        "ground_truth": "Replace this with the actual answer from your document.",
    },
    {
        "question": "What are the key findings or conclusions?",
        "ground_truth": "Replace this with the actual answer from your document.",
    },
]


def load_testset(path: str) -> list[dict]:
    """
    Load a JSON testset file.

    Format:
    [
      {"question": "...", "ground_truth": "..."},
      ...
    ]
    """
    with open(path) as f:
        return json.load(f)


def save_sample_testset(path: str = "evaluation/sample_testset.json") -> None:
    """Write the sample testset to disk so the user can edit it."""
    Path(path).parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(SAMPLE_TESTSET, f, indent=2)
    print(f"Sample testset saved to {path} — edit it with real Q&A pairs.")


# ── Pipeline builder ──────────────────────────────────────────────────────────

def build_pipeline(pdf_paths: list[str]) -> HybridRetriever:
    """Build the full retrieval pipeline from PDF paths."""
    all_chunks = []
    for path in pdf_paths:
        pages = parse_pdf(path)
        all_chunks.extend(chunk_pages(pages))

    vs = VectorStore()
    vs.build(all_chunks)

    bm25 = BM25Store()
    bm25.build(all_chunks)

    reranker = Reranker()

    return HybridRetriever(vs, bm25, reranker)


# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate_pipeline(
    retriever: HybridRetriever,
    testset: list[dict],
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Run RAGAS evaluation over a testset.

    For each question:
      1. Retrieve top_k chunks
      2. Generate answer
      3. Record (question, answer, contexts, ground_truth)

    Then feed all records to RAGAS for batch scoring.

    Args:
        retriever:  built HybridRetriever (from build_pipeline or the app)
        testset:    list of {"question": ..., "ground_truth": ...} dicts
        top_k:      chunks to retrieve per question

    Returns:
        DataFrame with per-question scores + aggregate means.
    """
    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"Running pipeline on {len(testset)} questions...")
    for i, item in enumerate(testset, 1):
        q = item["question"]
        gt = item["ground_truth"]

        chunks_with_scores = retriever.retrieve(q, top_k=top_k)
        result = generate_answer(q, chunks_with_scores)

        questions.append(q)
        answers.append(result["answer"])
        contexts.append([c.text for c, _ in chunks_with_scores])
        ground_truths.append(gt)

        print(f"  [{i}/{len(testset)}] Done: {q[:60]}...")

    # Build HuggingFace Dataset (RAGAS expects this format)
    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    print("\nRunning RAGAS metrics (this calls OpenAI — takes ~1 min)...")
    llm = ChatOpenAI(model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    df = result.to_pandas()

    # Add aggregate row
    means = df[["faithfulness", "context_precision", "context_recall"]].mean()
    summary_row = pd.DataFrame([{
        "question": "── AVERAGE ──",
        "answer": "",
        "faithfulness": round(means["faithfulness"], 3),
        "context_precision": round(means["context_precision"], 3),
        "context_recall": round(means["context_recall"], 3),
    }])
    df = pd.concat([df, summary_row], ignore_index=True)

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a clean summary table to stdout."""
    avg = df[df["question"] == "── AVERAGE ──"].iloc[0]
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Faithfulness      : {avg['faithfulness']:.3f}  (target > 0.80)")
    print(f"  Context Precision : {avg['context_precision']:.3f}  (target > 0.75)")
    print(f"  Context Recall    : {avg['context_recall']:.3f}  (target > 0.70)")
    print("=" * 50)

    faith = avg["faithfulness"]
    if faith >= 0.80:
        print("✅ Faithfulness target met!")
    else:
        print(f"⚠️  Faithfulness {faith:.2f} below 0.80 — consider improving chunking or reranker top_k")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with RAGAS")
    parser.add_argument("--pdf", nargs="+", required=True, help="PDF file paths to index")
    parser.add_argument("--testset", default="evaluation/sample_testset.json",
                        help="Path to JSON testset file")
    parser.add_argument("--output", default=None,
                        help="CSV output path (default: evaluation/results_<timestamp>.csv)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--init-testset", action="store_true",
                        help="Write sample_testset.json and exit")
    args = parser.parse_args()

    if args.init_testset:
        save_sample_testset()
        exit(0)

    testset_path = args.testset
    if not Path(testset_path).exists():
        print(f"Testset not found at {testset_path}. Creating sample...")
        save_sample_testset(testset_path)
        print("Edit it with real Q&A pairs, then re-run.")
        exit(0)

    testset = load_testset(testset_path)
    retriever = build_pipeline(args.pdf)
    df = evaluate_pipeline(retriever, testset, top_k=args.top_k)

    print_summary(df)

    output_path = args.output or f"evaluation/results_{datetime.now():%Y%m%d_%H%M%S}.csv"
    Path(output_path).parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nFull results saved to {output_path}")

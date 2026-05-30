"""
agent/generator.py

Takes retrieved chunks and a question, sends them to the LLM,
and returns a structured answer with citations.

The prompt engineering here is deliberate:
- We number each chunk so the LLM can reference [1], [2] etc.
- We instruct it to ONLY use the provided context (reduces hallucination)
- We ask for a confidence level (foundation for Week 3 self-correction)
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from ingestion.chunker import Chunk

load_dotenv()

SYSTEM_PROMPT = """You are a precise research assistant. Answer questions using ONLY the provided context chunks.

Rules:
1. Cite sources inline using [chunk_number] notation, e.g. "The model uses attention [1]."
2. If the context doesn't contain enough information, say "The provided documents don't cover this."
3. Never invent facts not present in the context.
4. End your answer with a Confidence line: Confidence: High / Medium / Low
   - High: context directly answers the question
   - Medium: context partially answers it
   - Low: you're inferring or context is thin
"""


def build_context_block(chunks_with_scores: list[tuple[Chunk, float]]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    lines = []
    for i, (chunk, score) in enumerate(chunks_with_scores, start=1):
        lines.append(
            f"[{i}] Source: {chunk.source_file}, Page {chunk.page_number}\n"
            f"{chunk.text}\n"
        )
    return "\n---\n".join(lines)


def generate_answer(
    question: str,
    chunks_with_scores: list[tuple[Chunk, float]],
    model: str = "gpt-4o-mini",   # cheap + capable for RAG tasks
) -> dict:
    """
    Generate an answer grounded in the retrieved chunks.

    Returns:
        {
            "answer": str,           # full LLM response
            "sources": list[dict],   # chunk metadata for UI citations
            "confidence": str,       # "High" / "Medium" / "Low"
        }
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    context = build_context_block(chunks_with_scores)
    user_message = f"""Context chunks:
{context}

Question: {question}

Answer (cite chunks with [number]):"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,    # low temperature = more factual, less creative
        max_tokens=1000,
    )

    answer_text = response.choices[0].message.content.strip()

    # Extract confidence level from the answer
    confidence = "Medium"
    for line in answer_text.split("\n"):
        if line.startswith("Confidence:"):
            confidence = line.replace("Confidence:", "").strip()
            break

    # Build source list for UI rendering
    sources = [
        {
            "chunk_id": chunk.chunk_id,
            "source_file": chunk.source_file,
            "page_number": chunk.page_number,
            "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            "relevance_score": round(score, 3),
        }
        for chunk, score in chunks_with_scores
    ]

    return {
        "answer": answer_text,
        "sources": sources,
        "confidence": confidence,
    }

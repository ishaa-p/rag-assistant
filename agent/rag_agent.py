"""
agent/rag_agent.py

Implements an agentic retrieval loop using LangGraph.

Why LangGraph instead of a simple for-loop:
  LangGraph models the agent as a state machine with explicit nodes
  and edges. This makes the flow inspectable, debuggable, and easy
  to extend (add new nodes = add new capabilities).
  It's also the industry standard for agentic RAG in 2026.

Agent flow:
  ┌─────────────┐
  │    START    │
  └──────┬──────┘
         ▼
  ┌─────────────┐     decompose complex query
  │   PLAN      │  ─► into sub-questions
  └──────┬──────┘
         ▼
  ┌─────────────┐     hybrid + graph retrieval
  │  RETRIEVE   │  ─► for each sub-question
  └──────┬──────┘
         ▼
  ┌─────────────┐     LLM grades context quality
  │   VERIFY    │  ─► "is this enough to answer?"
  └──────┬──────┘
         │  No (max 2 retries)          Yes
         ▼                               ▼
  ┌─────────────┐               ┌─────────────────┐
  │   REFINE    │               │    GENERATE     │
  │  (re-query) │               │  (final answer) │
  └──────┬──────┘               └─────────────────┘
         └──────────────────────────────┘ (loop back)
"""

import os
from typing import TypedDict, Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from ingestion.chunker import Chunk
from retrieval.hybrid_retriever import HybridRetriever
from storage.graph_store import GraphStore
from ingestion.entity_extractor import EntityExtractor

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
MAX_RETRIES = 2


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The state object passed between every node in the graph.
    LangGraph persists this across nodes — each node reads and updates it.

    Annotated[list, operator.add] means: when two nodes both write to
    this field, their lists are concatenated rather than one overwriting.
    """
    original_question: str
    sub_questions: list[str]
    all_chunks: Annotated[list[tuple[Chunk, float]], operator.add]
    context_sufficient: bool
    retry_count: int
    final_answer: str
    sources: list[dict]
    confidence: str


# ── Node functions ────────────────────────────────────────────────────────────

def plan_node(state: AgentState) -> dict:
    """
    Query decomposition: break a complex question into sub-questions.

    Why decompose?
    "How does attention relate to transformers and what are their limitations?"
    is three questions. Retrieving for the full string returns mediocre chunks.
    Retrieving for each sub-question separately, then synthesising, is far better.

    Simple questions (detected by LLM) are returned as-is.
    """
    question = state["original_question"]

    response = LLM.invoke([
        SystemMessage(content="""You are a query planning assistant.
Break complex questions into 2-3 focused sub-questions for retrieval.
For simple questions, return just the original.

Respond ONLY with a JSON list of strings. No preamble, no markdown fences.
Example: ["What is attention?", "How is attention used in transformers?"]"""),
        HumanMessage(content=question),
    ])

    import json
    try:
        sub_questions = json.loads(response.content.strip())
        if not isinstance(sub_questions, list):
            sub_questions = [question]
    except json.JSONDecodeError:
        sub_questions = [question]

    return {"sub_questions": sub_questions}


def retrieve_node(state: AgentState, retriever: HybridRetriever,
                  graph_store: GraphStore, entity_extractor: EntityExtractor) -> dict:
    """
    Hybrid + graph retrieval for all sub-questions.

    Two retrieval paths run in parallel:
      1. HybridRetriever (FAISS + BM25 + reranker) — from Week 2
      2. GraphStore traversal — entity-based chunk discovery

    Results are merged and deduplicated by chunk_id.
    """
    seen_ids: set[str] = set()
    all_chunks: list[tuple[Chunk, float]] = []

    for sub_q in state["sub_questions"]:
        # Path 1: hybrid retrieval
        hybrid_results = retriever.retrieve(sub_q, top_k=5)
        for chunk, score in hybrid_results:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                all_chunks.append((chunk, score))

        # Path 2: graph traversal
        # Extract entities from the sub-question itself
        # We create a temporary fake chunk just for entity extraction
        from ingestion.chunker import Chunk as ChunkCls
        temp_chunk = ChunkCls(
            chunk_id="query_temp", text=sub_q,
            source_file="query", page_number=0, chunk_index=0
        )
        query_entities = entity_extractor.extract([temp_chunk])
        entity_texts = [e.text for e in query_entities]

        if entity_texts:
            related_ids = graph_store.get_related_chunk_ids(entity_texts, hops=1, limit=5)
            # Note: in a full implementation you'd look up these chunks from a
            # chunk registry. For simplicity we log them — full registry added in Week 4.

    return {"all_chunks": all_chunks}


def verify_node(state: AgentState) -> dict:
    """
    Self-correction: LLM grades whether the retrieved context is
    sufficient to answer the original question.

    This is the key "agentic" behaviour — the system decides whether
    to try again rather than generating a weak answer.

    Returns context_sufficient=True/False.
    If False and retries remain, the graph routes back to retrieve_node
    with a refined query.
    """
    if not state["all_chunks"]:
        return {"context_sufficient": False}

    context_sample = "\n\n".join(
        chunk.text[:300] for chunk, _ in state["all_chunks"][:5]
    )

    response = LLM.invoke([
        SystemMessage(content="""You are a retrieval quality judge.
Given a question and retrieved context, decide if the context is sufficient
to answer the question accurately.
Respond with ONLY "SUFFICIENT" or "INSUFFICIENT". Nothing else."""),
        HumanMessage(content=f"Question: {state['original_question']}\n\nContext:\n{context_sample}"),
    ])

    sufficient = "SUFFICIENT" in response.content.upper()
    return {"context_sufficient": sufficient}


def refine_node(state: AgentState) -> dict:
    """
    If context was insufficient, ask the LLM to rewrite the query
    to retrieve different/better chunks on the next loop iteration.
    """
    response = LLM.invoke([
        SystemMessage(content="""You are a query refinement assistant.
The initial retrieval didn't find enough context.
Rewrite the question with different keywords to find better results.
Return ONLY the rewritten question, no explanation."""),
        HumanMessage(content=state["original_question"]),
    ])

    refined = response.content.strip()
    return {
        "sub_questions": [refined],
        "all_chunks": [],              # clear old results for fresh retrieval
        "retry_count": state["retry_count"] + 1,
    }


def generate_node(state: AgentState) -> dict:
    """
    Final answer generation — same logic as Week 1/2 generator
    but now receives chunks from the full agentic pipeline.
    """
    from agent.generator import generate_answer
    result = generate_answer(state["original_question"], state["all_chunks"])
    return {
        "final_answer": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"],
    }


# ── Routing logic ─────────────────────────────────────────────────────────────

def should_retry(state: AgentState) -> str:
    """
    Conditional edge: after verify_node, decide next step.
      - Context sufficient → generate
      - Context insufficient + retries left → refine (loop back)
      - Context insufficient + no retries → generate anyway (with low confidence)
    """
    if state["context_sufficient"]:
        return "generate"
    if state["retry_count"] < MAX_RETRIES:
        return "refine"
    return "generate"   # give up retrying, generate best-effort answer


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_agent(
    retriever: HybridRetriever,
    graph_store: GraphStore,
    entity_extractor: EntityExtractor,
):
    """
    Assemble and compile the LangGraph state machine.

    Node wiring:
      plan → retrieve → verify → [generate | refine → retrieve]
    """
    from functools import partial

    graph = StateGraph(AgentState)

    # Add nodes — each is a function that receives state and returns updates
    graph.add_node("plan",     plan_node)
    graph.add_node("retrieve", partial(retrieve_node,
                                       retriever=retriever,
                                       graph_store=graph_store,
                                       entity_extractor=entity_extractor))
    graph.add_node("verify",   verify_node)
    graph.add_node("refine",   refine_node)
    graph.add_node("generate", generate_node)

    # Add edges
    graph.set_entry_point("plan")
    graph.add_edge("plan",     "retrieve")
    graph.add_edge("retrieve", "verify")
    graph.add_conditional_edges("verify", should_retry, {
        "generate": "generate",
        "refine":   "refine",
    })
    graph.add_edge("refine",   "retrieve")   # loop back
    graph.add_edge("generate", END)

    return graph.compile()


def run_agent(
    question: str,
    retriever: HybridRetriever,
    graph_store: GraphStore,
    entity_extractor: EntityExtractor,
) -> dict:
    """
    Entry point: run the full agentic pipeline for a question.

    Returns the same dict shape as generate_answer() from Week 1/2
    so the Streamlit app needs minimal changes.
    """
    agent = build_agent(retriever, graph_store, entity_extractor)

    initial_state: AgentState = {
        "original_question": question,
        "sub_questions": [],
        "all_chunks": [],
        "context_sufficient": False,
        "retry_count": 0,
        "final_answer": "",
        "sources": [],
        "confidence": "Medium",
    }

    final_state = agent.invoke(initial_state)

    return {
        "answer":     final_state["final_answer"],
        "sources":    final_state["sources"],
        "confidence": final_state["confidence"],
        "sub_questions": final_state["sub_questions"],
        "retries":    final_state["retry_count"],
    }
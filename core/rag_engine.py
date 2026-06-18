import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from .parser import Message, parse_conversations
from .topic_detector import TopicSegment, detect_topics
from .summarizer import summarize_topic_segment, summarize_100_block
from .vector_store import VectorStore
from .persona import PersonaProfile, extract_persona, persona_to_json


@dataclass
class TopicCheckpoint:
    topic_id: int
    start_global_idx: int
    end_global_idx: int
    keywords: list[str]
    summary: str
    message_count: int


@dataclass
class HundredCheckpoint:
    block_num: int
    start_global_idx: int
    end_global_idx: int
    summary: str


@dataclass
class RAGState:
    messages: list[Message] = field(default_factory=list)
    topic_checkpoints: list[TopicCheckpoint] = field(default_factory=list)
    hundred_checkpoints: list[HundredCheckpoint] = field(default_factory=list)
    persona: Optional[PersonaProfile] = None
    vector_store: Optional[VectorStore] = field(default=None, repr=False)
    is_built: bool = False


def build_rag(csv_path: str, verbose: bool = True) -> RAGState:
    if verbose:
        print("[RAG] Parsing conversations...")
    messages = parse_conversations(csv_path)
    if verbose:
        print(f"[RAG] {len(messages)} messages loaded")

    state = RAGState(messages=messages, vector_store=VectorStore())

    if verbose:
        print("[RAG] Detecting topic segments...")
    segments = detect_topics(messages, window=10, stride=5, threshold=0.20, min_segment_len=6)
    if verbose:
        print(f"[RAG] {len(segments)} topic segments found")

    topic_texts, topic_kinds, topic_metas = [], [], []

    for seg in segments:
        seg.summary = summarize_topic_segment(seg)
        cp = TopicCheckpoint(
            topic_id=seg.topic_id,
            start_global_idx=seg.start_idx,
            end_global_idx=seg.end_idx,
            keywords=seg.keywords,
            summary=seg.summary,
            message_count=len(seg.messages),
        )
        state.topic_checkpoints.append(cp)
        topic_texts.append(seg.summary)
        topic_kinds.append("topic")
        topic_metas.append({"topic_id": seg.topic_id, "keywords": seg.keywords, "start": seg.start_idx, "end": seg.end_idx})

    state.vector_store.add(topic_texts, topic_kinds, topic_metas)

    if verbose:
        print("[RAG] Building 100-message checkpoints...")

    block_texts, block_kinds, block_metas = [], [], []
    block_size = 100
    for block_num, start in enumerate(range(0, len(messages), block_size)):
        block_msgs = messages[start:start + block_size]
        summary = summarize_100_block(block_msgs, block_num)
        cp = HundredCheckpoint(
            block_num=block_num,
            start_global_idx=block_msgs[0].global_idx,
            end_global_idx=block_msgs[-1].global_idx,
            summary=summary,
        )
        state.hundred_checkpoints.append(cp)
        block_texts.append(summary)
        block_kinds.append("block")
        block_metas.append({"block_num": block_num, "start": block_msgs[0].global_idx, "end": block_msgs[-1].global_idx})

    state.vector_store.add(block_texts, block_kinds, block_metas)

    if verbose:
        print("[RAG] Indexing raw message chunks (window=5)...")
    chunk_texts, chunk_kinds, chunk_metas = [], [], []
    for start in range(0, len(messages), 5):
        chunk = messages[start:start + 5]
        chunk_text = " | ".join(f"{m.speaker}: {m.text}" for m in chunk)
        chunk_texts.append(chunk_text)
        chunk_kinds.append("chunk")
        chunk_metas.append({"start_idx": chunk[0].global_idx, "end_idx": chunk[-1].global_idx})

    if chunk_texts:
        state.vector_store.add(chunk_texts, chunk_kinds, chunk_metas)

    if verbose:
        print("[RAG] Extracting user persona...")
    state.persona = extract_persona(messages)

    state.is_built = True
    if verbose:
        print(f"[RAG] Done. Vector store has {state.vector_store.total} entries.")

    return state


def query_rag(state: RAGState, question: str, top_topic: int = 3, top_chunk: int = 3) -> str:
    if not state.is_built:
        return "RAG system not built yet."

    topic_results = state.vector_store.search_by_kind(question, "topic", top_k=top_topic)
    chunk_results = state.vector_store.search_by_kind(question, "chunk", top_k=top_chunk)

    context_parts = []
    if topic_results:
        context_parts.append("=== Relevant Topic Summaries ===")
        for chunk, score in topic_results:
            context_parts.append(f"[Score:{score:.3f}] {chunk.text}")

    if chunk_results:
        context_parts.append("\n=== Relevant Message Excerpts ===")
        for chunk, score in chunk_results:
            context_parts.append(f"[Score:{score:.3f}] {chunk.text}")

    context = "\n".join(context_parts)
    return context


def get_retrieval_context(state: RAGState, question: str) -> dict:
    topic_results = state.vector_store.search_by_kind(question, "topic", top_k=3)
    chunk_results = state.vector_store.search_by_kind(question, "chunk", top_k=3)
    block_results = state.vector_store.search_by_kind(question, "block", top_k=2)

    return {
        "topic_summaries": [{"text": c.text, "score": round(s, 4), "meta": c.meta} for c, s in topic_results],
        "message_chunks": [{"text": c.text, "score": round(s, 4), "meta": c.meta} for c, s in chunk_results],
        "block_summaries": [{"text": c.text, "score": round(s, 4), "meta": c.meta} for c, s in block_results],
    }

import os
import sys
import json
import pickle
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.rag_engine import build_rag, get_retrieval_context, RAGState
from core.persona import persona_to_json

CACHE_PATH = Path(__file__).parent.parent / "data" / "rag_state.pkl"
CSV_PATH   = Path(__file__).parent.parent / "data" / "conversations.csv"
FRONT_PATH = Path(__file__).parent.parent / "frontend"

_state: RAGState = None


def _load_or_build() -> RAGState:
    global _state
    if CACHE_PATH.exists():
        print("[API] Loading cached RAG state …")
        with open(CACHE_PATH, "rb") as f:
            _state = pickle.load(f)
        print("[API] Loaded.")
    else:
        if not CSV_PATH.exists():
            raise FileNotFoundError(f"Put conversations.csv at {CSV_PATH}")
        _state = build_rag(str(CSV_PATH), verbose=True)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(_state, f)
        print("[API] State cached.")
    return _state


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_or_build)
    yield


app = FastAPI(title="ConversaLens", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONT_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(FRONT_PATH)), name="static")


class QueryRequest(BaseModel):
    question: str
    top_topic: int = 3
    top_chunk: int = 3


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    index = FRONT_PATH / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"status": "ConversaLens API running"}


@app.get("/health")
def health():
    return {"status": "ok", "rag_built": _state is not None and _state.is_built}


@app.get("/stats")
def stats():
    if not _state or not _state.is_built:
        raise HTTPException(503, "RAG not ready")
    return {
        "total_messages":     len(_state.messages),
        "topic_checkpoints":  len(_state.topic_checkpoints),
        "hundred_checkpoints": len(_state.hundred_checkpoints),
        "vector_store_size":  _state.vector_store.total,
    }


@app.get("/checkpoints/topics")
def get_topic_checkpoints(limit: int = 50, offset: int = 0):
    if not _state or not _state.is_built:
        raise HTTPException(503, "RAG not ready")
    items = _state.topic_checkpoints[offset : offset + limit]
    return {"total": len(_state.topic_checkpoints), "items": [asdict(cp) for cp in items]}


@app.get("/checkpoints/blocks")
def get_block_checkpoints(limit: int = 20, offset: int = 0):
    if not _state or not _state.is_built:
        raise HTTPException(503, "RAG not ready")
    items = _state.hundred_checkpoints[offset : offset + limit]
    return {"total": len(_state.hundred_checkpoints), "items": [asdict(cp) for cp in items]}


@app.post("/query")
def query_endpoint(req: QueryRequest):
    if not _state or not _state.is_built:
        raise HTTPException(503, "RAG not ready")
    context = get_retrieval_context(_state, req.question)
    return {"question": req.question, "retrieved_context": context}


@app.get("/persona")
def persona():
    if not _state or not _state.is_built:
        raise HTTPException(503, "RAG not ready")
    return json.loads(persona_to_json(_state.persona))


def _build_answer(question: str, persona_data: dict, context: dict) -> str:
    q = question.lower()
    topic_snippets = "\n".join(f"  • {r['text'][:180]}" for r in context["topic_summaries"])
    chunk_snippets = "\n".join(f"  • {r['text'][:180]}" for r in context["message_chunks"])

    is_person   = any(w in q for w in ["person","kind of","who is","personality","overall","describe"])
    is_habit    = any(w in q for w in ["habit","sleep","food","eat","routine","morning","diet","night"])
    is_talk     = any(w in q for w in ["talk","communicat","style","write","emoji","message","language","speak"])
    is_topic    = any(w in q for w in ["topic","discuss","talk about","interest","subject"])
    is_job      = any(w in q for w in ["job","work","career","profession","occupation"])
    is_relation = any(w in q for w in ["relation","family","friend","partner","married","kids","children"])

    if is_person:
        traits   = persona_data.get("personality_traits", [])
        jobs     = persona_data.get("jobs", [])
        hobbies  = persona_data.get("hobbies", [])
        tone     = persona_data.get("tone_distribution", {})
        dominant = max(tone, key=tone.get) if tone else "positive"
        lines = [
            "**Personality overview**",
            f"Traits      : {', '.join(traits) if traits else 'balanced and conversational'}",
            f"Occupation  : {', '.join(jobs[:3]) if jobs else 'not explicitly mentioned'}",
            f"Hobbies     : {', '.join(hobbies[:5]) if hobbies else 'varied interests'}",
            f"Tone        : predominantly {dominant}",
            "",
            "**From conversation context**",
            topic_snippets or "  (no specific topic context found)",
        ]
        return "\n".join(lines)

    if is_habit:
        sleep = persona_data.get("sleep_signals", [])
        food  = persona_data.get("food_habits", [])
        lines = [
            "**Detected habits**",
            f"Sleep patterns : {', '.join(sleep) if sleep else 'no strong sleep signals'}",
            f"Food habits    : {', '.join(food[:4]) if food else 'none explicitly mentioned'}",
            "",
            "**From conversation excerpts**",
            chunk_snippets or "  (no specific habit mentions found)",
        ]
        return "\n".join(lines)

    if is_talk:
        style = persona_data.get("communication_style", {})
        lines = [
            "**Communication style analysis**",
            f"Avg words/msg  : {style.get('avg_words_per_message', '?')}",
            f"Message length : {style.get('message_length_style', 'medium')}",
            f"Emoji usage    : {style.get('emoji_usage', 'rare')}",
            f"Questions      : {style.get('question_style', 'mostly statements')}",
            f"Traits         : {', '.join(persona_data.get('personality_traits', ['balanced']))}",
            "",
            "**Example exchanges**",
            chunk_snippets or "  (no excerpts found)",
        ]
        return "\n".join(lines)

    if is_topic:
        lines = [
            "**Top discussion topics**",
            topic_snippets or "  (no topic data)",
            "",
            "**Example excerpts**",
            chunk_snippets or "  (none found)",
        ]
        return "\n".join(lines)

    if is_job:
        jobs = persona_data.get("jobs", [])
        lines = [
            f"**Work / Occupation**",
            f"Detected roles : {', '.join(jobs) if jobs else 'not explicitly stated'}",
            "",
            "**From conversations**",
            chunk_snippets or topic_snippets or "  (no relevant excerpts)",
        ]
        return "\n".join(lines)

    if is_relation:
        rels = persona_data.get("relationships", [])
        lines = [
            "**Relationships mentioned**",
            f"{', '.join(rels) if rels else 'no explicit mentions'}",
            "",
            "**From conversations**",
            chunk_snippets or topic_snippets or "  (none found)",
        ]
        return "\n".join(lines)

    total_msgs   = len(_state.messages) if _state else "?"
    total_topics = len(_state.topic_checkpoints) if _state else "?"
    lines = [
        "**Relevant context retrieved**",
        topic_snippets or "  (no topic matches)",
        "",
        "**Message excerpts**",
        chunk_snippets or "  (no chunk matches)",
        "",
        f"Dataset: {total_msgs:,} messages · {total_topics:,} topic segments",
    ]
    return "\n".join(lines)


@app.post("/chat")
async def chat(req: ChatRequest):
    if not _state or not _state.is_built:
        raise HTTPException(503, "RAG not ready")

    persona_data = json.loads(persona_to_json(_state.persona))
    context      = get_retrieval_context(_state, req.message)
    answer       = _build_answer(req.message, persona_data, context)

    return {
        "answer": answer,
        "sources": {
            "topic_summaries_used": len(context["topic_summaries"]),
            "message_chunks_used":  len(context["message_chunks"]),
        },
    }

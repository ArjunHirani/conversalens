# ConversaLens — Conversation Intelligence RAG System

> KaStack Labs AI/ML Intern Assessment — End-to-end RAG with topic segmentation, persona extraction, and a zero-dependency vector store.

---

## Quick Start

```bash
pip install -r requirements.txt
python scripts/build_index.py          # ~45s, builds & caches index
cd api && uvicorn main:app --port 8000  # start API
open frontend/index.html               # open chatbot UI
```

---

## Architecture

```
conversations.csv (191,839 messages)
        │
        ▼
   parser.py          → flat chronological Message list
        │
        ├──► topic_detector.py  → 12,232 topic segments
        │    sliding-window cosine drift + conv-boundary snapping
        │         │
        │         └──► summarizer.py  → extractive TF-IDF sentence scoring
        │
        ├──► 100-msg blocks      → 1,919 fixed checkpoints
        │
        ├──► chunk indexer       → 5-msg sliding windows
        │
        ├──► vector_store.py     → pure TF-IDF cosine (52,519 entries, zero deps)
        │
        └──► persona.py          → regex + frequency signal mining
```

---

## Part 1 — Topic Detection

**How topic changes are detected (chronological, message-by-message):**

1. A sliding window of 15 messages moves forward in stride-5 steps
2. At each pivot, left-window and right-window BoW TF-IDF vectors are computed
3. If cosine similarity < 0.12, a topic boundary is flagged at that pivot
4. Each flagged boundary is snapped to the nearest real conversation boundary (±15 msgs) — this prevents mid-conversation splits
5. Micro-segments shorter than `min_segment_len` are merged into the prior segment

**Output:** 12,232 topic checkpoints, each with `start_idx`, `end_idx`, `keywords`, `summary`

**100-Message Checkpoints:** Every 100 messages (independent of topics), an extractive summary is generated and stored.

---

## Part 2 — Persona Extraction

All signals come from actual conversation text — no guessing:

| Field | Signal |
|---|---|
| `jobs` | Regex: "I'm a [role]", "I am a [role]" |
| `locations` | Regex: "I live in", "I'm from", "grew up in" |
| `relationships` | Frequency count of "my wife/dog/mom/son..." |
| `hobbies` | Regex on 30+ known activity keywords |
| `food_habits` | Regex on 12 specific food categories |
| `sleep_signals` | Keyword match: "night owl", "morning person", etc. |
| `avg_message_length` | Word count across all user turns |
| `emoji_frequency` | Unicode block detection per message |
| `question_rate` | Sentence-terminal `?` detection |
| `tone_distribution` | Positive/negative/humor word frequency ratio |

---

## Part 3 — Chatbot

`POST /chat` with `{"message": "..."}` combines:
- TF-IDF retrieval (topic summaries + raw chunks + block summaries)
- Persona JSON
- Intent classification (persona / habits / communication / general)

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status check |
| GET | `/stats` | Message/checkpoint/vector counts |
| GET | `/checkpoints/topics` | Paginated topic checkpoints |
| GET | `/checkpoints/blocks` | Paginated 100-msg checkpoints |
| POST | `/query` | Raw retrieval context |
| GET | `/persona` | Full persona JSON |
| POST | `/chat` | Chatbot answer |

---

## Project Structure

```
kastack-rag/
├── core/
│   ├── parser.py          # CSV → flat Message list
│   ├── topic_detector.py  # Sliding-window drift + boundary snapping
│   ├── summarizer.py      # Extractive TF-IDF sentence scoring
│   ├── persona.py         # Regex + frequency signal mining
│   ├── vector_store.py    # Pure TF-IDF cosine retrieval
│   └── rag_engine.py      # Orchestrator: build, cache, query
├── api/
│   └── main.py            # FastAPI backend
├── frontend/
│   └── index.html         # Single-file chatbot SPA
├── scripts/
│   ├── build_index.py     # Build + cache RAG state
│   └── query_cli.py       # Terminal chat interface
├── tests/
│   └── test_pipeline.py   # 7 unit tests (all pass)
└── requirements.txt
```

---

## Run Tests

```bash
python tests/test_pipeline.py
# 7/7 tests passed
```

---

## Key Design Choices

**Zero external model dependencies** — The entire retrieval stack is custom TF-IDF cosine similarity. No sentence-transformers, no OpenAI, no API calls. Fully offline, deterministic, and explainable.

**Conversation-boundary-aware topic splitting** — Naive drift detectors split mid-conversation. This system snaps pivot points to real conversation boundaries, producing semantically clean segments.

**Extractive summarization** — Sentences are scored by TF-IDF weight against the segment vocabulary and top-k are selected. No hallucination, instant generation, fully reproducible.

**Cached pickle state** — Full RAG state serialized after first build. API loads in <1s on subsequent starts.

import sys
import pickle
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.rag_engine import get_retrieval_context
from core.persona import persona_to_json

CACHE = Path(__file__).parent.parent / "data" / "rag_state.pkl"

def load_state():
    if not CACHE.exists():
        print("Run scripts/build_index.py first.")
        sys.exit(1)
    with open(CACHE, "rb") as f:
        return pickle.load(f)

def main():
    state = load_state()
    print(f"\nConversaLens CLI — {len(state.messages)} msgs | {len(state.topic_checkpoints)} topics | {state.vector_store.total} vectors")
    print("Type your question. 'persona' to show profile. 'quit' to exit.\n")
    while True:
        try:
            q = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() == "quit":
            break
        if q.lower() == "persona":
            print(persona_to_json(state.persona))
            continue
        ctx = get_retrieval_context(state, q)
        print("\n--- Topic Summaries ---")
        for r in ctx["topic_summaries"]:
            print(f"  [{r['score']:.3f}] {r['text'][:200]}")
        print("\n--- Message Chunks ---")
        for r in ctx["message_chunks"]:
            print(f"  [{r['score']:.3f}] {r['text'][:200]}")
        print()

if __name__ == "__main__":
    main()

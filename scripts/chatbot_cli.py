import sys, pickle, json, textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.rag_engine import get_retrieval_context
from core.persona import persona_to_json


BANNER = r"""
  ____                          _             _
 / ___|___  _ ____   _____ _ __| |_ __ _  ___| |     ___ _ __  ___
| |   / _ \| '_ \ \ / / _ \ '__| __/ _` |/ _ \ |    / _ \ '_ \/ __|
| |__| (_) | | | \ V /  __/ |  | || (_| |  __/ |___|  __/ | | \__ \
 \____\___/|_| |_|\_/ \___|_|   \__\__,_|\___|_____\___|_| |_|___/
                                                  RAG Chatbot v1.0
"""

CACHE = Path(__file__).parent.parent / "data" / "rag_state.pkl"

QUICK = [
    "What kind of person is this user?",
    "What are their habits?",
    "How do they talk and communicate?",
    "What topics do they discuss most?",
    "What are their hobbies and interests?",
]


def _fmt(text: str, width: int = 88) -> str:
    lines = text.split("\n")
    out = []
    for line in lines:
        if line.startswith("**") or line.startswith("  •"):
            out.append(line)
        else:
            out.extend(textwrap.wrap(line, width) or [""])
    return "\n".join(out)


def main():
    print(BANNER)

    if not CACHE.exists():
        print(f"[!] No cached index at {CACHE}")
        print("    Run:  python scripts/build_index.py")
        sys.exit(1)

    print("Loading index …", end=" ", flush=True)
    with open(CACHE, "rb") as f:
        state = pickle.load(f)
    persona_data = json.loads(persona_to_json(state.persona))
    print(f"done  ({len(state.messages):,} msgs · {len(state.topic_checkpoints):,} topics)\n")

    print("Quick prompts (type a number or your own question):")
    for i, q in enumerate(QUICK, 1):
        print(f"  {i}. {q}")
    print("  q. quit\n")

    def answer(question: str) -> str:
        ctx = get_retrieval_context(state, question)
        q = question.lower()

        topic_snippets = "\n".join(f"  • {r['text'][:180]}" for r in ctx["topic_summaries"])
        chunk_snippets = "\n".join(f"  • {r['text'][:180]}" for r in ctx["message_chunks"])

        is_person   = any(w in q for w in ["person","kind of","who is","personality","overall","describe"])
        is_habit    = any(w in q for w in ["habit","sleep","food","eat","routine","morning","diet"])
        is_talk     = any(w in q for w in ["talk","communicat","style","write","emoji","message","speak"])
        is_topic    = any(w in q for w in ["topic","discuss","talk about","interest","subject"])

        if is_person:
            traits  = persona_data.get("personality_traits", [])
            jobs    = persona_data.get("jobs", [])
            hobbies = persona_data.get("hobbies", [])
            tone    = persona_data.get("tone_distribution", {})
            dominant = max(tone, key=tone.get) if tone else "positive"
            return (
                f"**Personality overview**\n"
                f"Traits      : {', '.join(traits) or 'balanced, conversational'}\n"
                f"Occupation  : {', '.join(jobs[:3]) or 'not explicitly mentioned'}\n"
                f"Hobbies     : {', '.join(hobbies[:5]) or 'varied'}\n"
                f"Tone        : predominantly {dominant}\n\n"
                f"**From conversation context**\n{topic_snippets or '  (none found)'}"
            )
        if is_habit:
            sleep = persona_data.get("sleep_signals", [])
            food  = persona_data.get("food_habits", [])
            return (
                f"**Detected habits**\n"
                f"Sleep : {', '.join(sleep) or 'no strong signals'}\n"
                f"Food  : {', '.join(food[:4]) or 'none mentioned'}\n\n"
                f"**From excerpts**\n{chunk_snippets or '  (none found)'}"
            )
        if is_talk:
            style = persona_data.get("communication_style", {})
            return (
                f"**Communication style**\n"
                f"Avg words/msg : {style.get('avg_words_per_message','?')}\n"
                f"Length style  : {style.get('message_length','?')}\n"
                f"Emoji usage   : {style.get('emoji_usage','?')}\n"
                f"Questions     : {style.get('question_asking','?')}\n"
                f"Dominant tone : {style.get('dominant_tone','?')}\n\n"
                f"**Example exchanges**\n{chunk_snippets or '  (none found)'}"
            )
        if is_topic:
            return (
                f"**Top discussion topics**\n{topic_snippets or '  (none found)'}\n\n"
                f"**Example excerpts**\n{chunk_snippets or '  (none found)'}"
            )
        return (
            f"**Retrieved context**\n{topic_snippets or '  (none)'}\n\n"
            f"**Message excerpts**\n{chunk_snippets or '  (none)'}\n\n"
            f"Dataset: {len(state.messages):,} msgs · {len(state.topic_checkpoints):,} topics"
        )

    while True:
        try:
            raw = input("You › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if raw.lower() in ("q", "quit", "exit"):
            print("Bye!")
            break
        if not raw:
            continue
        if raw.isdigit() and 1 <= int(raw) <= len(QUICK):
            question = QUICK[int(raw) - 1]
            print(f"  → {question}")
        else:
            question = raw

        print("\nConversaLens ›")
        print(_fmt(answer(question)))
        print()


if __name__ == "__main__":
    main()

import sys
import pickle
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.parser import parse_conversations, Message
from core.topic_detector import detect_topics
from core.summarizer import extractive_summary, summarize_topic_segment
from core.persona import extract_persona
from core.vector_store import VectorStore
from core.rag_engine import build_rag, get_retrieval_context


MINI_CSV = Path(__file__).parent / "mini_test.csv"


def make_mini_csv():
    lines = [
        "conversation",
        '"User 1: Hey! How are you?\nUser 2: Great! Just got back from hiking.\nUser 1: Nice! Where did you hike?\nUser 2: Rocky Mountains in Colorado.\nUser 1: I love hiking too. Especially trail running.\nUser 2: Do you run marathons?\nUser 1: Yes I ran two marathons last year.\nUser 2: Impressive! I prefer yoga myself."',
        '"User 1: What music are you into?\nUser 2: I love jazz and classical.\nUser 1: Nice. I am a musician myself, I play guitar.\nUser 2: What genre?\nUser 1: Mostly blues and rock.\nUser 2: Cool! Have you performed live?\nUser 1: Yes, small venues around New York."',
    ]
    MINI_CSV.write_text("\n".join(lines))


def test_parser():
    make_mini_csv()
    msgs = parse_conversations(str(MINI_CSV))
    assert len(msgs) > 0, "Parser returned no messages"
    assert msgs[0].global_idx == 0
    assert msgs[0].speaker != ""
    assert msgs[0].text != ""
    print(f"  parser: {len(msgs)} messages OK")


def test_topic_detector():
    make_mini_csv()
    msgs = parse_conversations(str(MINI_CSV))
    segs = detect_topics(msgs, window=3, stride=1, threshold=0.05, min_segment_len=2)
    assert len(segs) >= 1
    total_msgs = sum(len(s.messages) for s in segs)
    assert total_msgs == len(msgs), f"Lost messages: {total_msgs} != {len(msgs)}"
    print(f"  topic_detector: {len(segs)} segments, no messages lost OK")


def test_summarizer():
    make_mini_csv()
    msgs = parse_conversations(str(MINI_CSV))
    summary = extractive_summary(msgs[:8])
    assert len(summary) > 10
    print(f"  summarizer: '{summary[:80]}...' OK")


def test_persona():
    make_mini_csv()
    msgs = parse_conversations(str(MINI_CSV))
    profile = extract_persona(msgs)
    assert profile.avg_message_length > 0
    assert isinstance(profile.personality_traits, list)
    print(f"  persona: traits={profile.personality_traits} OK")


def test_vector_store():
    store = VectorStore()
    store.add(
        ["hiking mountains trail running", "jazz music guitar blues"],
        ["topic", "topic"],
        [{}, {}],
    )
    results = store.search("music guitar", top_k=2)
    assert len(results) > 0
    assert results[0][0].text == "jazz music guitar blues"
    print(f"  vector_store: retrieval correct (score={results[0][1]:.3f}) OK")


def test_rag_retrieval():
    make_mini_csv()
    state = build_rag(str(MINI_CSV), verbose=False)
    assert state.is_built
    assert len(state.topic_checkpoints) >= 1
    ctx = get_retrieval_context(state, "music guitar")
    assert "topic_summaries" in ctx
    assert "message_chunks" in ctx
    print(f"  rag_retrieval: {len(ctx['topic_summaries'])} topic hits OK")


def test_chronological_order():
    make_mini_csv()
    msgs = parse_conversations(str(MINI_CSV))
    for i in range(1, len(msgs)):
        assert msgs[i].global_idx == msgs[i-1].global_idx + 1, "Chronological order violated"
    print(f"  chronological_order: {len(msgs)} messages in order OK")


if __name__ == "__main__":
    print("Running ConversaLens test suite...\n")
    tests = [
        test_parser,
        test_topic_detector,
        test_summarizer,
        test_persona,
        test_vector_store,
        test_rag_retrieval,
        test_chronological_order,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    MINI_CSV.unlink(missing_ok=True)

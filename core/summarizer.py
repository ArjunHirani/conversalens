import math
import re
from collections import Counter

from .parser import Message
from .topic_detector import TopicSegment, _tokenize, STOPWORDS


def _sentence_scores(sentences: list[str], word_freq: Counter) -> list[float]:
    scores = []
    for sent in sentences:
        tokens = _tokenize(sent)
        if not tokens:
            scores.append(0.0)
            continue
        score = sum(word_freq.get(t, 0) for t in tokens) / len(tokens)
        scores.append(score)
    return scores


def extractive_summary(messages: list[Message], max_sentences: int = 4) -> str:
    combined = " ".join(m.text for m in messages)
    sentences = re.split(r"(?<=[.!?])\s+", combined)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return " ".join(m.text for m in messages[:3])

    all_tokens = []
    for s in sentences:
        all_tokens.extend(_tokenize(s))
    word_freq = Counter(all_tokens)

    scores = _sentence_scores(sentences, word_freq)
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    top_indices = sorted(ranked[:max_sentences])
    return " ".join(sentences[i] for i in top_indices)


def summarize_topic_segment(segment: TopicSegment) -> str:
    topic_line = f"[Topic: {', '.join(segment.keywords[:4])}] "
    body = extractive_summary(segment.messages, max_sentences=3)
    return topic_line + body


def summarize_100_block(messages: list[Message], block_num: int) -> str:
    start = messages[0].global_idx
    end = messages[-1].global_idx
    header = f"[Messages {start}–{end}] "
    body = extractive_summary(messages, max_sentences=4)
    return header + body

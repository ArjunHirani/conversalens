import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from .parser import Message


STOPWORDS = frozenset({
    "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she",
    "it", "they", "them", "this", "that", "these", "those", "am", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "to", "of", "in", "for", "on", "with", "at",
    "by", "from", "as", "an", "the", "a", "and", "but", "or", "so",
    "if", "then", "than", "too", "very", "just", "also", "well",
    "oh", "yeah", "ok", "okay", "yes", "no", "not", "its", "it's",
    "don't", "doesn't", "can't", "won't", "about", "what", "how",
    "when", "where", "who", "which", "that's", "i'm", "i've", "i'll",
    "there", "here", "get", "got", "like", "think", "know", "really",
    "going", "go", "want", "love", "good", "great", "nice", "thing",
    "things", "lot", "much", "many", "some", "any", "all", "one",
    "two", "three", "first", "time", "way", "make", "made", "look",
    "say", "said", "told", "tell", "come", "came", "see", "saw",
    "take", "took", "give", "gave", "sure", "sounds", "pretty",
    "awesome", "cool", "wow", "well", "actually", "definitely",
    "totally", "honestly", "basically", "thanks", "thank", "doing",
    "done", "right", "even", "never", "always", "every", "still",
    "re", "ve", "ll", "d",
})


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z']+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def _bag_of_words(tokens: list[str]) -> Counter:
    return Counter(tokens)


def _cosine_bow(a: Counter, b: Counter) -> float:
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _top_keywords(tokenized: list[list[str]], top_n: int = 6) -> list[str]:
    freq = Counter()
    for tokens in tokenized:
        freq.update(tokens)
    return [w for w, _ in freq.most_common(top_n)]


@dataclass
class TopicSegment:
    topic_id: int
    start_idx: int
    end_idx: int
    messages: list[Message]
    keywords: list[str]
    summary: str = ""

    @property
    def label(self) -> str:
        return f"Topic {self.topic_id}: {', '.join(self.keywords[:4])}"


def detect_topics(
    messages: list[Message],
    window: int = 15,
    stride: int = 5,
    threshold: float = 0.12,
    min_segment_len: int = 5,
) -> list[TopicSegment]:
    if not messages:
        return []

    tokenized = [_tokenize(m.text) for m in messages]
    n = len(messages)

    if n < window * 2:
        return [TopicSegment(
            topic_id=1,
            start_idx=0,
            end_idx=n - 1,
            messages=messages,
            keywords=_top_keywords(tokenized),
        )]

    break_points: set[int] = {0}

    for pivot in range(window, n - window, stride):
        left_tokens = []
        for toks in tokenized[max(0, pivot - window):pivot]:
            left_tokens.extend(toks)
        right_tokens = []
        for toks in tokenized[pivot:min(n, pivot + window)]:
            right_tokens.extend(toks)

        bow_left = _bag_of_words(left_tokens)
        bow_right = _bag_of_words(right_tokens)
        sim = _cosine_bow(bow_left, bow_right)

        if sim < threshold:
            actual_pivot = pivot - (stride // 2)
            break_points.add(max(0, actual_pivot))

    conv_boundaries: set[int] = set()
    for i in range(1, n):
        if messages[i].conv_idx != messages[i - 1].conv_idx:
            conv_boundaries.add(i)

    merged_breaks = set()
    sorted_content = sorted(break_points)
    for bp in sorted_content:
        nearest_boundary = min(conv_boundaries, key=lambda x: abs(x - bp), default=bp)
        if abs(nearest_boundary - bp) <= window:
            merged_breaks.add(nearest_boundary)
        else:
            merged_breaks.add(bp)

    merged_breaks.add(0)
    merged_breaks.add(n)

    sorted_breaks = sorted(merged_breaks)

    raw_segments = []
    for b_idx in range(len(sorted_breaks) - 1):
        seg_start = sorted_breaks[b_idx]
        seg_end = sorted_breaks[b_idx + 1]
        if seg_end - seg_start > 0:
            raw_segments.append((seg_start, seg_end))

    merged_segments = []
    buf_start, buf_end = raw_segments[0]
    for seg_start, seg_end in raw_segments[1:]:
        if (seg_end - buf_start) < min_segment_len * 2:
            buf_end = seg_end
        else:
            merged_segments.append((buf_start, buf_end))
            buf_start, buf_end = seg_start, seg_end
    merged_segments.append((buf_start, buf_end))

    segments: list[TopicSegment] = []
    for topic_id, (seg_start, seg_end) in enumerate(merged_segments, start=1):
        seg_messages = messages[seg_start:seg_end]
        seg_tokens = tokenized[seg_start:seg_end]
        keywords = _top_keywords(seg_tokens)
        segments.append(TopicSegment(
            topic_id=topic_id,
            start_idx=seg_start,
            end_idx=seg_end - 1,
            messages=seg_messages,
            keywords=keywords,
        ))

    return segments

import re
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict

from .parser import Message

_JOB_RE = re.compile(
    r"\bi(?:'m| am) (?:a |an )([\w\s]{2,30}?(?:engineer|doctor|nurse|teacher|developer|designer|manager|lawyer|chef|student|analyst|professor|artist|writer|musician|scientist|researcher|architect|accountant|programmer|radiolog\w+))\b",
    re.IGNORECASE,
)
_LOC_RE = re.compile(
    r"\b(?:i(?:'m| am) from |i live in |i grew up in |living in |based in )([A-Z][a-zA-Z]{2,18}(?:\s[A-Z][a-zA-Z]{2,12})?)\b"
)
_REL_RE = re.compile(
    r"\bmy (wife|husband|partner|boyfriend|girlfriend|mom|dad|mother|father|sister|brother|son|daughter|dog|cat|kids?)\b",
    re.IGNORECASE,
)
_HOBBY_RE = re.compile(
    r"\bi (?:love|enjoy|really like|am into) ((?:hiking|reading|cooking|gaming|swimming|cycling|yoga|running|painting|drawing|photography|gardening|dancing|singing|writing|coding|traveling|climbing|skiing|fishing|skateboarding|surfing|boxing|martial arts|chess|knitting|woodworking|baking|camping|football|basketball|baseball|soccer|tennis|golf|volleyball|cricket|rugby)[\w\s]{0,15}?)\b",
    re.IGNORECASE,
)
_FOOD_RE = re.compile(
    r"\bi (?:love|like|enjoy|prefer) (?:eating |cooking |making )?(pizza|sushi|tacos|pasta|coffee|tea|ramen|burgers|seafood|salads?|sandwiches?|indian food|mexican food|italian food|thai food|chinese food|vegan food|vegetarian food|spicy food)\b",
    re.IGNORECASE,
)
_SLEEP_RE = re.compile(
    r"\b(stay(?:s)? up late|night owl|go to bed late|sleep late|can't sleep|wake up early|morning person|early riser|up until \d+)\b",
    re.IGNORECASE,
)
_MOOD = {
    "positive": ["happy","excited","love","great","amazing","wonderful","awesome","fantastic","joy","glad","fun","enjoy"],
    "negative": ["sad","angry","frustrated","upset","depressed","anxious","stressed","worried","terrible","awful"],
    "humor": ["haha","lol","lmao","funny","joke","laugh","hilarious","humor"],
}


@dataclass
class PersonaProfile:
    jobs: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    hobbies: list[str] = field(default_factory=list)
    sleep_signals: list[str] = field(default_factory=list)
    food_habits: list[str] = field(default_factory=list)
    avg_message_length: float = 0.0
    emoji_frequency: float = 0.0
    question_rate: float = 0.0
    tone_distribution: dict = field(default_factory=dict)
    personality_traits: list[str] = field(default_factory=list)
    communication_style: dict = field(default_factory=dict)
    raw_signal_count: int = 0


def extract_persona(messages: list[Message]) -> PersonaProfile:
    p = PersonaProfile()
    user_msgs = [m for m in messages if "1" in m.speaker or m.speaker.lower() in ("user","a")]
    if not user_msgs:
        user_msgs = messages
    text = " ".join(m.text for m in user_msgs)

    p.jobs = sorted({m.group(1).strip().lower() for m in _JOB_RE.finditer(text)})[:5]
    p.locations = sorted({m.group(1).strip() for m in _LOC_RE.finditer(text) if 2 < len(m.group(1).strip()) < 25})[:5]
    rel_c = Counter(m.group(1).lower() for m in _REL_RE.finditer(text))
    p.relationships = [r for r,_ in rel_c.most_common(8)]
    p.hobbies = sorted({m.group(1).strip().lower() for m in _HOBBY_RE.finditer(text)})[:8]
    p.sleep_signals = list({m.group(1).lower() for m in _SLEEP_RE.finditer(text)})[:4]
    p.food_habits = sorted({m.group(1).strip().lower() for m in _FOOD_RE.finditer(text)})[:8]

    n = len(user_msgs)
    p.avg_message_length = round(sum(m.word_count for m in user_msgs) / max(n,1), 1)
    p.emoji_frequency = round(sum(1 for m in user_msgs if m.has_emoji) / max(n,1), 3)
    p.question_rate = round(sum(1 for m in user_msgs if m.ends_with_question) / max(n,1), 3)

    tc = defaultdict(int)
    lt = text.lower()
    for tone, words in _MOOD.items():
        tc[tone] = sum(lt.count(w) for w in words)
    tot = sum(tc.values()) or 1
    p.tone_distribution = {k: round(v/tot,3) for k,v in tc.items()}

    traits = []
    if p.emoji_frequency > 0.3: traits.append("expressive / emoji-heavy")
    elif p.emoji_frequency < 0.05: traits.append("text-focused communicator")
    if p.avg_message_length < 8: traits.append("concise / brief replies")
    elif p.avg_message_length > 20: traits.append("verbose / detailed writer")
    if p.question_rate > 0.3: traits.append("inquisitive / asks many questions")
    if tc["humor"] > tc["negative"] * 1.5: traits.append("humorous / lighthearted")
    if tc["positive"] > tc["negative"] * 3: traits.append("optimistic / positive tone")
    elif tc["negative"] > tc["positive"]: traits.append("expressive about struggles")
    p.personality_traits = traits

    p.communication_style = {
        "avg_words_per_message": p.avg_message_length,
        "emoji_usage": "frequent" if p.emoji_frequency > 0.25 else "rare" if p.emoji_frequency < 0.05 else "moderate",
        "question_style": "frequently asks questions" if p.question_rate > 0.3 else "mostly statements",
        "message_length_style": "short" if p.avg_message_length < 8 else "long" if p.avg_message_length > 20 else "medium",
    }
    p.raw_signal_count = n
    return p


def persona_to_json(profile: PersonaProfile) -> str:
    return json.dumps(asdict(profile), indent=2)

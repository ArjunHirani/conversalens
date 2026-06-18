import re
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    global_idx: int
    conv_idx: int
    speaker: str
    text: str
    char_count: int = field(init=False)
    word_count: int = field(init=False)
    has_emoji: bool = field(init=False)
    ends_with_question: bool = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())
        emoji_pattern = re.compile(
            "[\U00010000-\U0010ffff\U0001F300-\U0001F9FF\u2600-\u26FF\u2700-\u27BF]",
            flags=re.UNICODE,
        )
        self.has_emoji = bool(emoji_pattern.search(self.text))
        self.ends_with_question = self.text.strip().endswith("?")


def parse_conversations(csv_path: str) -> list[Message]:
    df = pd.read_csv(csv_path)
    col = df.columns[0]

    speaker_pattern = re.compile(r"^(User \d+|Assistant|Bot|A|B):\s*", re.IGNORECASE)

    messages: list[Message] = []
    global_idx = 0

    for conv_idx, raw in enumerate(df[col]):
        if not isinstance(raw, str):
            continue
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        for line in lines:
            m = speaker_pattern.match(line)
            if m:
                speaker = m.group(1)
                text = line[m.end():].strip()
            else:
                speaker = "Unknown"
                text = line
            if not text:
                continue
            messages.append(Message(global_idx=global_idx, conv_idx=conv_idx, speaker=speaker, text=text))
            global_idx += 1

    return messages

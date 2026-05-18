"""分句工具：将文本按中文标点切分，并保留每句的字符偏移量。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# 中文/英文句子分隔符（与项目 nl_parser.py 保持一致）
_SENT_SPLIT_RE = re.compile(r"[，。；;！!？?\n\r]+")

# 过短的句子直接丢弃（去噪）
_MIN_SENTENCE_LEN = 2


@dataclass(frozen=True)
class Sentence:
    """带字符偏移量的一个句子。"""

    text: str
    char_start: int   # 相对于所在 section 的起始偏移
    char_end: int     # 相对于所在 section 的结束偏移（不含）

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "char_start": self.char_start, "char_end": self.char_end}


def split_sentences(text: str, offset: int = 0) -> list[Sentence]:
    """将 text 切分为句子列表，每句带绝对字符偏移（以 offset 为基准）。

    Args:
        text: 待切分的文本。
        offset: 该 text 在整篇文档中的起始偏移，用于计算绝对位置。

    Returns:
        按顺序排列的句子列表，不含空句和过短句子。
    """
    results: list[Sentence] = []
    pos = 0
    for match in _SENT_SPLIT_RE.finditer(text):
        chunk = text[pos:match.start()].strip()
        if len(chunk) >= _MIN_SENTENCE_LEN:
            abs_start = offset + pos
            abs_end = offset + match.start()
            results.append(Sentence(text=chunk, char_start=abs_start, char_end=abs_end))
        pos = match.end()

    # 处理末尾没有标点结束的剩余部分
    tail = text[pos:].strip()
    if len(tail) >= _MIN_SENTENCE_LEN:
        results.append(Sentence(
            text=tail,
            char_start=offset + pos,
            char_end=offset + len(text),
        ))
    return results

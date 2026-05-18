"""第一级消歧：字符串标准化 + 精确匹配。

输入  data/staging/mentions.jsonl 中 status='llm_candidate' 的 mention。
对每条 mention 的 normalized 词面做：
  1. 加载缩写展开表（abbr_expansion.json），规范化
  2. 与现有 alias 词典精确比对
  3. 命中 → linked_entity_id 直接填充，status 改为 'auto_confirmed'，method='alias_exact'
  4. 未命中 → 保持 llm_candidate，进入下一级消歧

精确匹配约可处理 30-40% 的 llm_candidate，剩余交给嵌入层。

运行方式
--------
    python -m optimize.disambiguation.string_normalize
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from optimize.config import cfg
from optimize.utils.file_utils import read_json, read_jsonl, write_jsonl
from optimize.utils.logging_utils import get_pipeline_logger

logger = get_pipeline_logger("disambiguation.string_normalize")


def _build_normalized_alias_index(alias_dict: dict[str, list[str]]) -> dict[str, str]:
    """将 skill_aliases.json 中所有别名归一化后建立反向索引：normalized → entity_id。"""
    index: dict[str, str] = {}
    for entity_id, aliases in alias_dict.items():
        # 以最长别名优先（防止短串覆盖长串）
        for alias in sorted(aliases, key=len, reverse=True):
            norm = alias.lower().strip()
            if norm and norm not in index:
                index[norm] = entity_id
    return index


def _normalize(surface: str, abbr_table: dict[str, str]) -> str:
    """对词面做标准化：小写、去前后空格、缩写展开。"""
    s = surface.lower().strip()
    # 去除常见前缀修饰词（"熟悉"、"了解"、"掌握"等），提取核心词面
    s = re.sub(r"^(熟悉|了解|掌握|精通|使用|基于|熟练使用|熟练掌握)\s*", "", s)
    return abbr_table.get(s, s)


def run() -> dict[str, int]:
    """对 llm_candidate mention 做第一级精确消歧。

    直接修改 mentions.jsonl 中命中 mention 的 linked_entity_id 和 status。
    """
    mentions_path = cfg.paths.staging_mentions
    alias_dict    = read_json(cfg.paths.dict_skill_aliases)
    abbr_table    = read_json(cfg.paths.abbr_expansion)

    alias_index = _build_normalized_alias_index(alias_dict)
    all_mentions: list[dict[str, Any]] = read_jsonl(mentions_path)

    confirmed = 0
    total_llm  = 0
    for m in all_mentions:
        if m.get("status") != "llm_candidate":
            continue
        total_llm += 1
        norm = _normalize(m.get("normalized", m.get("surface", "")), abbr_table)
        entity_id = alias_index.get(norm)
        if entity_id:
            m["linked_entity_id"] = entity_id
            m["link_method"]      = "alias_exact"
            m["link_confidence"]  = 0.99
            m["status"]           = "auto_confirmed"
            m["normalized"]       = norm
            confirmed += 1

    write_jsonl(mentions_path, all_mentions)
    remaining = total_llm - confirmed
    logger.info(
        "第一级精确消歧完成：总 llm_candidate=%d  命中=%d（%.0f%%）  剩余=%d",
        total_llm, confirmed, confirmed / max(total_llm, 1) * 100, remaining,
    )
    return {"total_llm": total_llm, "confirmed": confirmed, "remaining": remaining}


if __name__ == "__main__":
    stats = run()
    print(f"命中 {stats['confirmed']}/{stats['total_llm']}，剩余 {stats['remaining']} 条进入嵌入消歧")

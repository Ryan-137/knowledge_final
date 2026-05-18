"""L3 合并脚本：将 L1 和 L2b 的 mention 合并去重，输出最终 mentions.jsonl。

合并规则
--------
1. L1 (rule_match) 和 L2b (llm_candidate) 的 mention 按 doc_id + char_start + char_end 去重
2. 相同位置上 L1 优先（已有高精度链接），L2b 的 linked_entity_id=None 只补充新发现
3. 输出按 (doc_id, char_start) 排序，方便后续消歧模块顺序处理

运行方式
--------
    python -m optimize.ner.merge_mentions
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from optimize.config import cfg
from optimize.utils.file_utils import read_jsonl, write_jsonl
from optimize.utils.logging_utils import get_pipeline_logger

logger = get_pipeline_logger("ner.merge_mentions")


def run() -> dict[str, int]:
    """合并去重所有 mention。"""
    mentions_path = cfg.paths.staging_mentions
    if not mentions_path.exists():
        logger.warning("mentions.jsonl 不存在，跳过合并")
        return {"total": 0}

    all_mentions: list[dict[str, Any]] = read_jsonl(mentions_path)

    # 按 (doc_id, char_start, char_end) 去重，L1 优先
    seen: dict[tuple[str, int, int], dict[str, Any]] = {}
    for m in all_mentions:
        key = (m["doc_id"], m["char_start"], m["char_end"])
        existing = seen.get(key)
        if existing is None:
            seen[key] = m
        elif m["status"] == "rule_match" and existing["status"] != "rule_match":
            # L1 结果覆盖 L2 结果（同位置上 L1 更可信）
            seen[key] = m

    merged = sorted(seen.values(), key=lambda x: (x["doc_id"], x["char_start"]))

    # 统计
    by_status: dict[str, int] = {}
    for m in merged:
        s = m.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1

    # 覆盖写回（有序、去重后的版本）
    n = write_jsonl(mentions_path, merged)
    logger.info("合并完成：%d → %d 条（%s）", len(all_mentions), n, by_status)
    return {"before": len(all_mentions), "after": n, "by_status": by_status}


if __name__ == "__main__":
    result = run()
    print(f"合并前 {result['before']} 条 → 合并后 {result['after']} 条")
    print("状态分布:", result.get("by_status", {}))

"""半自动审阅 needs_review 队列。

高分段（score >= auto_accept_threshold）自动接受为新 alias，
中低分段逐条打印并写入人工审阅结果文件 pipeline_data/canonical/review_decisions.json。

运行方式
--------
    # 自动处理高分段（0.85+），交互审阅其余部分
    python -m optimize.evaluation.review_queue

    # 只自动处理高分段，不做交互（CI 友好）
    python -m optimize.evaluation.review_queue --auto-only

    # 查看结果统计
    python -m optimize.evaluation.review_queue --stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from optimize.config import cfg
from optimize.utils.file_utils import read_json, read_jsonl, write_json, write_jsonl
from optimize.utils.logging_utils import get_pipeline_logger

logger = get_pipeline_logger("evaluation.review_queue")

_DECISIONS_PATH = cfg.paths.canonical_root / "review_decisions.json"

# 高分段自动接受阈值
_AUTO_ACCEPT_THRESHOLD = 0.85

# 明显错误的链接（surface 和 best_entity 语义完全不相关，直接拒绝）
_AUTO_REJECT_PATTERNS = [
    # ndk → tool_redis 这类距离过远的误链接
    ("ndk", "tool_redis"),
    ("sqlite", "skill_sql"),    # sqlite 是独立工具，不是 sql 语言的 alias
]


def _is_auto_reject(surface: str, entity_id: str) -> bool:
    s = surface.lower().strip()
    return (s, entity_id) in _AUTO_REJECT_PATTERNS


def _decide_batch(records: list[dict[str, Any]], auto_only: bool) -> dict[str, Any]:
    """对所有 needs_review 记录做决策，返回决策字典。"""
    decisions: dict[str, Any] = {}   # surface → {"action": accept|reject|new_entity, "alias_for": ...}

    for r in records:
        surface  = r["surface"]
        entity   = r["best_entity"]
        score    = r.get("best_score", 0.0)

        # 已有决策则跳过
        if surface in decisions:
            continue

        if _is_auto_reject(surface, entity):
            decisions[surface] = {"action": "reject", "alias_for": None, "score": score,
                                  "note": "auto_reject: semantic mismatch"}
            continue

        if score >= _AUTO_ACCEPT_THRESHOLD:
            decisions[surface] = {"action": "accept", "alias_for": entity, "score": score,
                                  "note": "auto_accept: score >= 0.85"}
            continue

        if auto_only:
            decisions[surface] = {"action": "pending", "alias_for": entity, "score": score,
                                  "note": "pending_human_review"}
        else:
            # 交互式审阅
            decision = _interactive_review(surface, entity, score, r.get("candidates", []), r.get("entity_type_hint", ""))
            decisions[surface] = decision

    return decisions


def _interactive_review(
    surface: str,
    best_entity: str,
    score: float,
    candidates: list[dict[str, Any]],
    type_hint: str,
) -> dict[str, Any]:
    """单条交互审阅。"""
    print(f"\n{'='*60}")
    print(f"词面:     {surface}")
    print(f"建议链接: {best_entity}  (score={score:.3f})  [{type_hint}]")
    print("候选列表:")
    for c in candidates[:3]:
        print(f"  {c['score']:.3f}  {c['entity_id']}")
    print()
    print("操作: [a] 接受为 alias  [r] 拒绝  [n] 标记为新实体候选  [s] 跳过")

    while True:
        choice = input(">>> ").strip().lower()
        if choice == "a":
            return {"action": "accept", "alias_for": best_entity, "score": score, "note": "human_accepted"}
        elif choice == "r":
            return {"action": "reject", "alias_for": None, "score": score, "note": "human_rejected"}
        elif choice == "n":
            return {"action": "new_entity", "alias_for": None, "score": score, "note": "human_new_entity"}
        elif choice == "s":
            return {"action": "pending", "alias_for": best_entity, "score": score, "note": "skipped"}
        else:
            print("请输入 a/r/n/s")


def apply_decisions(decisions: dict[str, Any]) -> dict[str, int]:
    """将审阅结果应用到 aliases_enriched.json（追加新 alias）。"""
    aliases_out = cfg.paths.output_aliases
    if not aliases_out.exists():
        logger.warning("aliases_enriched.json 不存在，跳过应用")
        return {}

    aliases_data = read_json(aliases_out)
    extra: dict[str, list[str]] = aliases_data.get("extra_aliases", {})

    added = 0
    for surface, decision in decisions.items():
        if decision["action"] != "accept":
            continue
        eid = decision["alias_for"]
        if not eid:
            continue
        existing = [a.lower() for a in extra.get(eid, [])]
        if surface.lower() not in existing:
            extra.setdefault(eid, []).append(surface)
            added += 1

    aliases_data["extra_aliases"] = extra
    write_json(aliases_out, aliases_data)
    logger.info("已将 %d 条新 alias 写入 aliases_enriched.json", added)
    return {"aliases_added": added}


def print_stats(decisions: dict[str, Any]) -> None:
    from collections import Counter
    action_counter = Counter(d["action"] for d in decisions.values())
    print("\n审阅结果统计：")
    for action, count in sorted(action_counter.items()):
        print(f"  {action:<15} {count}")
    print(f"  total          {sum(action_counter.values())}")


def run(auto_only: bool = False, stats_only: bool = False) -> None:
    """主流程。"""
    # 加载已有决策（支持断点续做）
    existing: dict[str, Any] = {}
    if _DECISIONS_PATH.exists():
        existing = read_json(_DECISIONS_PATH)

    if stats_only:
        print_stats(existing)
        return

    log = read_jsonl(cfg.paths.disambig_log)
    nrs = [r for r in log if r.get("status") == "needs_review"]
    logger.info("needs_review 队列共 %d 条（已有决策 %d 条）", len(nrs), len(existing))

    # 过滤掉已有决策的记录
    pending = [r for r in nrs if r["surface"] not in existing]
    logger.info("待审阅 %d 条", len(pending))

    new_decisions = _decide_batch(pending, auto_only=auto_only)
    all_decisions  = {**existing, **new_decisions}

    write_json(_DECISIONS_PATH, all_decisions)
    logger.info("决策已写入 %s", _DECISIONS_PATH)

    apply_stats = apply_decisions(all_decisions)
    print_stats(all_decisions)

    accepted = sum(1 for d in new_decisions.values() if d["action"] == "accept")
    rejected = sum(1 for d in new_decisions.values() if d["action"] == "reject")
    pending_count = sum(1 for d in new_decisions.values() if d["action"] == "pending")
    print(f"\n本次新增：accept={accepted}  reject={rejected}  pending={pending_count}")
    print(f"alias 写入：{apply_stats.get('aliases_added', 0)} 条")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--auto-only", action="store_true", help="只处理高分段，不做交互审阅")
    p.add_argument("--stats", action="store_true", help="只显示当前决策统计")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(auto_only=args.auto_only, stats_only=args.stats)

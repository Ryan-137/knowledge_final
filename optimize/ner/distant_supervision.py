"""L2a 远程监督：从语料中挖掘未被 L1 覆盖的高频技术候选词面。

原理
----
传统远程监督假设"知识库里的实体对若在同一句中出现，该句可能表达该关系"。
对本项目的实体识别场景，我们做的是其变体：

  候选词面挖掘（candidate surface mining）
    1. 用正则提取语料中"像技术实体"的 n-gram（英文 CamelCase、全大写缩写、
       "XXX框架"/"XXX库"/"XXX平台"等中文模式）
    2. 过滤掉 L1 词典已覆盖的词面（归一化后比对）
    3. 按语料频次排序，输出 top-N"候选新词面"
    4. 将这批候选词面提供给：
       - L2b LLM NER 优先处理
       - Step 6 消歧模块（决定是新 alias 还是新节点）

同时输出 entity_cooccurrence_candidates.jsonl（同文档共现对），
供 wxs & sx 关系抽取组使用。

产出文件
--------
data/canonical/candidate_surfaces.json          候选新词面列表（排序后）
data/canonical/entity_cooccurrence_candidates.jsonl  实体共现对（不含权重）

运行方式
--------
    python -m optimize.ner.distant_supervision
    python -m optimize.ner.distant_supervision --top-n 300
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from optimize.config import cfg
from optimize.utils.file_utils import (
    append_jsonl_batch,
    ensure_dir,
    read_json,
    read_jsonl,
    write_json,
)
from optimize.utils.logging_utils import get_pipeline_logger

logger = get_pipeline_logger("ner.distant_supervision")

# 候选新词面输出路径
_CANDIDATES_PATH   = cfg.paths.canonical_root / "candidate_surfaces.json"
_COOCCUR_PATH      = cfg.paths.cooccurrence

# 提取"英文技术词"的正则：CamelCase、全大写缩写、含数字的技术词、含./#/+的技术词
_EN_TECH_RE = re.compile(
    r"\b(?:"
    r"[A-Z][a-z]+(?:[A-Z][a-z]+)+"      # CamelCase: SpringBoot
    r"|[A-Z]{2,}"                         # 全大写缩写: NLP, BERT, YOLO
    r"|[A-Za-z]+(?:\.[A-Za-z]+)+"        # 带点: Vue.js, Node.js
    r"|[A-Za-z]+[0-9]+(?:\.[0-9]+)*"     # 带版本号: Python3, CUDA11
    r"|[A-Za-z]+[+#@]"                   # C++, C#
    r")\b"
)

# 提取"中文技术词"的正则：XXX框架/库/平台/引擎/算法/系统/模型/工具
_ZH_TECH_SUFFIX_RE = re.compile(
    r"[\u4e00-\u9fa5A-Za-z0-9]{2,10}"
    r"(?:框架|库|平台|引擎|算法|系统|模型|工具|语言|数据库|中间件|协议|架构|方案)"
)

# 过滤噪声词（通用词、停用词）
_NOISE_WORDS = {
    "the", "and", "for", "with", "using", "API", "SDK", "App", "Web",
    "IT", "AI", "UI", "UX", "ID", "OK", "PR", "PM", "HR", "CTO", "CEO",
    "BC", "BA", "OA", "ERP", "CRM", "MIS", "EDM",
    "Java", "Python", "Go",  # 这些在 L1 已覆盖，远程监督阶段排除
}


def _extract_candidate_surfaces(text: str) -> list[str]:
    """从一段文本中提取候选技术词面（英文 + 中文双策略）。"""
    candidates: list[str] = []

    # 策略1：英文技术词
    for m in _EN_TECH_RE.finditer(text):
        word = m.group(0)
        if word in _NOISE_WORDS or len(word) < 2:
            continue
        candidates.append(word.lower())

    # 策略2：中文"XXX+技术后缀"组合
    for m in _ZH_TECH_SUFFIX_RE.finditer(text):
        candidates.append(m.group(0))

    return candidates


def _build_alias_set(alias_dict: dict[str, list[str]]) -> set[str]:
    """将 skill_aliases.json 中所有别名归一化为小写集合，用于快速过滤。"""
    result: set[str] = set()
    for aliases in alias_dict.values():
        result.update(a.lower().strip() for a in aliases)
    return result


def run(top_n: int = 500) -> dict[str, Any]:
    """执行候选词面挖掘主流程。

    Args:
        top_n: 最终保留的候选词面数量。

    Returns:
        统计字典。
    """
    ensure_dir(cfg.paths.canonical_root)

    # 加载 L1 已覆盖的词面（归一化后）
    alias_dict  = read_json(cfg.paths.dict_skill_aliases)
    covered     = _build_alias_set(alias_dict)
    logger.info("L1 词典已覆盖词面数（归一化）：%d", len(covered))

    # 加载 staging 文档
    staged_path = cfg.paths.staging_root / "staged_documents.jsonl"
    docs        = read_jsonl(staged_path)

    # 候选词面频次统计
    surface_counter: Counter[str] = Counter()
    # 文档级实体出现记录（用于共现）
    doc_entity_map: dict[str, set[str]] = defaultdict(set)
    # L1 mentions 中文档已识别的实体
    l1_mentions  = read_jsonl(cfg.paths.staging_mentions)
    for m in l1_mentions:
        entity_id = m.get("linked_entity_id")
        if entity_id:
            doc_entity_map[m["doc_id"]].add(entity_id)

    # 只处理 NER 价值高的章节类型
    target_sections = {"tech_skills", "requirements", "full_text", "projects"}

    for doc in docs:
        doc_id = doc["doc_id"]
        for sec in doc.get("sections", []):
            if sec["section_type"] not in target_sections:
                continue
            candidates = _extract_candidate_surfaces(sec["text"])
            for c in candidates:
                # 跳过 L1 已覆盖的
                if c in covered:
                    continue
                surface_counter[c] += 1

    # 过滤：频次 ≥ 2（单次出现很可能是噪声）
    filtered = {k: v for k, v in surface_counter.items() if v >= 2}
    # 按频次排序，取 top_n
    ranked = sorted(filtered.items(), key=lambda x: -x[1])[:top_n]

    candidates_output: list[dict[str, Any]] = [
        {"surface": s, "normalized": s.lower(), "frequency": f,
         "status": "candidate", "linked_entity_id": None}
        for s, f in ranked
    ]

    write_json(_CANDIDATES_PATH, {
        "total":      len(candidates_output),
        "top_n":      top_n,
        "candidates": candidates_output,
    })
    logger.info("候选新词面写入 %s（共 %d 条）", _CANDIDATES_PATH, len(candidates_output))

    # 生成文档级实体共现对（供关系抽取组参考）
    cooccur_records: list[dict[str, Any]] = []
    for doc_id, entity_ids in doc_entity_map.items():
        ids = sorted(entity_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                cooccur_records.append({
                    "entity_a":  ids[i],
                    "entity_b":  ids[j],
                    "doc_id":    doc_id,
                    "source":    "l1_cooccurrence",
                })

    # 合并相同 pair 的文档列表（减少冗余）
    pair_docs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rec in cooccur_records:
        pair_docs[(rec["entity_a"], rec["entity_b"])].append(rec["doc_id"])

    cooccur_flat = [
        {
            "entity_a": a,
            "entity_b": b,
            "doc_ids":  sorted(set(dids)),
            "count":    len(dids),
        }
        for (a, b), dids in sorted(pair_docs.items(), key=lambda x: -len(x[1]))
    ]

    if _COOCCUR_PATH.exists():
        _COOCCUR_PATH.unlink()
    append_jsonl_batch(_COOCCUR_PATH, cooccur_flat)
    logger.info("实体共现对写入 %s（共 %d 对）", _COOCCUR_PATH, len(cooccur_flat))

    stats = {
        "candidate_surfaces": len(candidates_output),
        "cooccurrence_pairs": len(cooccur_flat),
        "total_unique_surfaces_found": len(filtered),
    }
    logger.info("远程监督完成：%s", stats)
    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top-n", type=int, default=500, help="保留的候选词面数（默认 500）")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run(top_n=args.top_n)
    print(f"完成：候选新词面 {result['candidate_surfaces']} 条，实体共现对 {result['cooccurrence_pairs']} 对")

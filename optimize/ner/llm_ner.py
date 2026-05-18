"""L2b LLM NER：调用大模型抽取规则层遗漏的长尾实体。

设计目标
--------
1. 发现规则词典之外的新技术实体（LangChain、YOLO、Flink 等长尾工具）
2. 识别约束/否定表达（"不擅长 C++"、"C++ 基础薄弱"）
3. 捕获隐式技能（"参与推荐系统开发" → project_recommendation_system）

关键设计决策
------------
- 只处理 NER 价值最高的章节：tech_skills、requirements、projects
- 结合候选词面列表（distant_supervision 输出），在 prompt 中提示 LLM 重点关注
- 用 JSON Schema 约束输出格式，减少解析错误
- 去重：与 L1 词典命中的已知实体做归一化对比，只保留新发现
- 幂等：doc_id + section_id 缓存，避免重复调用 API

产出
----
新 mention 追加写入 data/staging/mentions.jsonl（status="llm_candidate"）
去重统计写入 data/canonical/llm_ner_stats.json

运行方式
--------
需先在环境变量中设置 API key：
    $env:DEEPSEEK_API_KEY = "sk-..."

    python -m optimize.ner.llm_ner
    python -m optimize.ner.llm_ner --max-docs 100 --sources fairCV
    python -m optimize.ner.llm_ner --dry-run          （只打印 prompt，不调用 API）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from optimize.config import cfg
from optimize.utils.file_utils import (
    append_jsonl_batch,
    ensure_dir,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from optimize.utils.hash_utils import text_sha256
from optimize.utils.logging_utils import get_pipeline_logger

logger = get_pipeline_logger("ner.llm_ner")

# 统计输出路径
_STATS_PATH    = cfg.paths.canonical_root / "llm_ner_stats.json"

# 只处理这些章节类型（NER 价值最高）
_TARGET_SECTION_TYPES = {"tech_skills", "requirements", "projects"}

# 每次发给 LLM 的最大字符数（约 600 token）
_MAX_CHARS_PER_CALL = 1200

# L1 词典命中阈值：归一化后长度 ≥ 2 才视为有效 alias
_MIN_ALIAS_LEN = 2

# 实体类型枚举（与 config 保持一致）
_ENTITY_TYPES = (
    "skill", "tool", "knowledge", "project",
    "soft_skill", "constraint", "interest", "language",
)

# Prompt 模板（中英文双语，减少 LLM 幻觉）
_SYSTEM_PROMPT = """你是技术岗位知识图谱的实体标注专家。请从输入文本中抽取技术实体。

重要说明：
1. 重点寻找【不常见的长尾技术实体】：新兴框架、特定平台、小众工具（如 LangChain、YOLO v8、Ray、Flink、TensorRT 等）
2. 【务必标注否定/约束表达】：例如"不擅长C++"→constraint，"数学较弱"→constraint
3. 【识别隐式技能】：例如"参与推荐系统开发"→project(推荐系统项目)
4. 常见实体（Python/Java/Docker/MySQL 等）【无需返回】，规则层已覆盖
5. 只返回 JSON，格式严格按下方 schema

实体类型说明：
- skill      : 编程语言、脚本语言
- tool       : 框架/库/中间件/平台/云服务/IDE/工具软件
- knowledge  : 理论知识、算法、设计模式、系统概念
- project    : 项目类型或实践经历（需含"项目"语义）
- soft_skill : 软技能（沟通/文档/协作/领导力等）
- constraint : 明确短板或不擅长（文本需有否定词或"薄弱"等词）
- interest   : 工作偏好或方向倾向
- language   : 自然语言能力（英语/日语等）"""

_USER_PROMPT_TEMPLATE = """\
{candidate_hint}文本：
{text}

请以如下 JSON 格式输出（只输出 JSON，不要任何解释）：
{{
  "entities": [
    {{
      "surface": "原文中的文字片段",
      "type": "<枚举值之一>",
      "is_negative": false,
      "confidence": 0.85,
      "context": "前后各约 15 字的上下文"
    }}
  ]
}}"""


def _build_alias_set(alias_dict: dict[str, list[str]]) -> set[str]:
    """构建 L1 词典归一化词面集合（用于过滤已知实体）。"""
    result: set[str] = set()
    for aliases in alias_dict.values():
        result.update(a.lower().strip() for a in aliases if len(a) >= _MIN_ALIAS_LEN)
    return result


def _load_candidate_hint(top_k: int = 20) -> str:
    """加载候选新词面，拼成 prompt 中的提示段落（让 LLM 重点关注这些词）。"""
    cand_path = cfg.paths.canonical_root / "candidate_surfaces.json"
    if not cand_path.exists():
        return ""
    data = read_json(cand_path)
    top = [c["surface"] for c in data.get("candidates", [])[:top_k]]
    if not top:
        return ""
    return f"请重点关注以下词面（可能是新实体）：{', '.join(top)}\n\n"


def _matches_sources(doc: dict[str, Any], sources: list[str] | None) -> bool:
    """按 source_group 或 source_name 过滤 staged 文档。"""
    if not sources:
        return True
    source_group = doc.get("source_group", "")
    source_name = doc.get("source_name", "")
    return source_group in sources or source_name in sources


def _find_surface_span(section_text: str, surface: str) -> tuple[int, int] | None:
    """在 section 原文中定位 LLM 返回的 surface，找不到则拒绝入库。"""
    norm = surface.lower()
    pos = section_text.lower().find(norm)
    if pos == -1:
        return None
    return pos, pos + len(surface)


def _call_llm(
    prompt_text: str,
    api_key: str,
    api_base: str,
    model: str,
    retry: int = 2,
) -> list[dict[str, Any]]:
    """调用 LLM（OpenAI 兼容接口），返回解析后的实体列表。

    每次调用都新建 httpx.Client，使用 20 秒读超时。
    网络不稳定时跳过（最多重试 2 次），不阻塞整个流程。
    """
    try:
        import httpx
        from openai import OpenAI  # type: ignore[import]
    except ImportError:
        logger.error("openai/httpx 包未安装，请运行：pip install openai httpx")
        return []

    for attempt in range(retry):
        # 每次尝试都新建连接，不复用可能已损坏的旧连接
        http_client = httpx.Client(
            timeout=httpx.Timeout(connect=8.0, read=20.0, write=8.0, pool=5.0)
        )
        try:
            client = OpenAI(api_key=api_key, base_url=api_base, http_client=http_client)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt_text},
                ],
                temperature=cfg.llm.temperature,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            parsed = json.loads(raw)
            return parsed.get("entities", [])
        except Exception as exc:
            wait = 2 * (attempt + 1)
            logger.warning("LLM 调用失败（第 %d/%d 次）：%s，%.0fs 后重试",
                           attempt + 1, retry, exc, wait)
            time.sleep(wait)
        finally:
            http_client.close()

    logger.warning("LLM 调用均失败，跳过本批次")
    return []


def _parse_llm_entities(
    raw_entities: list[dict[str, Any]],
    doc_id: str,
    section_id: str,
    section_type: str,
    section_text: str,
    section_char_start: int,
    covered_aliases: set[str],
    stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """将 LLM 返回的实体列表转换为 mention 格式，过滤已知实体。"""
    mentions: list[dict[str, Any]] = []
    for ent in raw_entities:
        surface = str(ent.get("surface", "")).strip()
        etype   = str(ent.get("type", "")).strip()
        if not surface or etype not in _ENTITY_TYPES:
            continue

        # 过滤 L1 词典已覆盖的实体（归一化比对）
        norm = surface.lower().strip()
        if norm in covered_aliases:
            continue

        span = _find_surface_span(section_text, surface)
        if span is None:
            if stats is not None:
                stats["dropped_unanchored"] = stats.get("dropped_unanchored", 0) + 1
            continue
        pos, end_pos = span

        context = str(ent.get("context", section_text[:60]))
        try:
            conf = float(ent.get("confidence", 0.80))
        except (TypeError, ValueError):
            conf = 0.0
        if conf < cfg.ner.llm_min_confidence or conf > 1.0:
            if stats is not None:
                stats["dropped_low_confidence"] = stats.get("dropped_low_confidence", 0) + 1
            continue
        is_neg  = bool(ent.get("is_negative", False))
        abs_start = section_char_start + pos
        abs_end   = section_char_start + end_pos

        mid = f"m_llm_{doc_id}_{section_id}_{abs_start}_{abs_end}"
        mentions.append({
            "mention_id":       mid,
            "doc_id":           doc_id,
            "section_id":       section_id,
            "section_type":     section_type,
            "surface":          surface,
            "normalized":       norm,
            "char_start":       abs_start,
            "char_end":         abs_end,
            "context_snippet":  context,
            "candidates":       [],        # 待消歧步骤填充
            "linked_entity_id": None,      # 尚未链接
            "link_method":      "llm_extracted",
            "link_confidence":  round(conf, 4),
            "status":           "llm_candidate",
            "is_negative":      is_neg,
            "intensity":        "negative" if is_neg else "neutral",
            "entity_type_hint": etype,     # LLM 猜测的类型（供消歧参考）
        })
    return mentions


def run(
    max_docs: int = 200,
    sources: list[str] | None = None,
    dry_run: bool = False,
    append: bool = False,
) -> dict[str, Any]:
    """执行 L2b LLM NER 主流程。

    Args:
        max_docs: 最多处理的文档数（控制 API 调用成本）。
        sources:  只处理指定来源，None 表示全部。
        dry_run:  仅打印 prompt 样例，不调用 API。

    Returns:
        统计字典。
    """
    ensure_dir(cfg.paths.canonical_root)
    ensure_dir(cfg.paths.staging_root)

    # 获取 API key
    api_key = os.environ.get(cfg.llm.api_key_env, "")
    if not api_key and not dry_run:
        logger.error(
            "未设置 API key，请执行：$env:%s = 'sk-...'",
            cfg.llm.api_key_env,
        )
        return {"error": "api_key_missing"}

    # 加载资源
    alias_dict      = read_json(cfg.paths.dict_skill_aliases)
    covered_aliases = _build_alias_set(alias_dict)
    candidate_hint  = _load_candidate_hint(top_k=20)
    cache: set[str] = set()
    docs            = read_jsonl(cfg.paths.staging_root / "staged_documents.jsonl")

    stats = {
        "docs_processed": 0,
        "sections_sent":  0,
        "new_mentions":   0,
        "skipped_cache":  0,
        "api_calls":      0,
        "dropped_unanchored": 0,
        "dropped_low_confidence": 0,
    }

    if not append and not dry_run and cfg.paths.staging_mentions.exists():
        existing_mentions = read_jsonl(cfg.paths.staging_mentions)
        rule_mentions = [m for m in existing_mentions if m.get("status") == "rule_match"]
        write_jsonl(cfg.paths.staging_mentions, rule_mentions)
        logger.info("已清理旧 LLM mention，保留规则 mention %d 条", len(rule_mentions))

    source_count: dict[str, int] = {}

    for doc in docs:
        src    = doc.get("source_name", "unknown")
        doc_id = doc["doc_id"]

        # 来源过滤
        if not _matches_sources(doc, sources):
            continue

        # 每个来源的文档上限
        if max_docs and source_count.get(src, 0) >= max_docs:
            continue

        # 筛选有价值的章节
        target_sections = [
            s for s in doc.get("sections", [])
            if s["section_type"] in _TARGET_SECTION_TYPES
        ]
        if not target_sections:
            continue

        doc_new_mentions: list[dict[str, Any]] = []

        for sec in target_sections:
            sec_sha = text_sha256(sec["text"])

            # 幂等检查
            if sec_sha in cache:
                stats["skipped_cache"] += 1
                continue

            # 文本过长则截断
            text = sec["text"][:_MAX_CHARS_PER_CALL]

            prompt = _USER_PROMPT_TEMPLATE.format(
                candidate_hint=candidate_hint,
                text=text,
            )

            if dry_run:
                # dry-run 只打印 prompt 样例，不写入缓存（避免污染正式运行）
                if stats["sections_sent"] == 0:
                    print("=== DRY RUN：prompt 样例 ===")
                    print(_SYSTEM_PROMPT[:300], "...")
                    print(prompt[:400], "...")
                    print("===========================")
                stats["sections_sent"] += 1
                continue

            raw_entities = _call_llm(
                prompt_text=prompt,
                api_key=api_key,
                api_base=cfg.llm.api_base,
                model=cfg.llm.model,
            )
            stats["api_calls"] += 1

            # 解析 + 过滤
            new_mentions = _parse_llm_entities(
                raw_entities    = raw_entities,
                doc_id          = doc_id,
                section_id      = sec["section_id"],
                section_type    = sec["section_type"],
                section_text    = sec["text"],
                section_char_start = sec["char_start"],
                covered_aliases = covered_aliases,
                stats           = stats,
            )
            doc_new_mentions.extend(new_mentions)
            cache.add(sec_sha)
            stats["sections_sent"] += 1

            # API 调用限速
            time.sleep(cfg.llm.request_interval)

        if doc_new_mentions:
            append_jsonl_batch(cfg.paths.staging_mentions, doc_new_mentions)
            stats["new_mentions"] += len(doc_new_mentions)

        source_count[src] = source_count.get(src, 0) + 1
        stats["docs_processed"] += 1

        if stats["docs_processed"] % 50 == 0:
            logger.info(
                "  已处理 %d 篇，新 mention %d 条，API 调用 %d 次",
                stats["docs_processed"], stats["new_mentions"], stats["api_calls"],
            )

    if not dry_run:
        write_json(_STATS_PATH, stats)
    logger.info(
        "L2b LLM NER 完成：docs=%d  sections=%d  new_mentions=%d  api_calls=%d",
        stats["docs_processed"], stats["sections_sent"],
        stats["new_mentions"], stats["api_calls"],
    )
    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-docs", type=int, default=200,
                   help="每个来源最多处理的文档数（默认 200，控制 API 成本）")
    p.add_argument("--sources", nargs="*", default=None,
                   help="指定数据来源，可用 source_group（fairCV/jd）或具体 source_name（如 csv_import）")
    p.add_argument("--dry-run", action="store_true",
                   help="仅打印 prompt 样例，不调用 API")
    p.add_argument("--append", action="store_true",
                   help="保留已有 LLM mention 并追加新结果（默认清理旧 LLM 结果）")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run(max_docs=args.max_docs, sources=args.sources, dry_run=args.dry_run, append=args.append)
    if "error" not in result:
        print(f"完成：new_mentions={result['new_mentions']}  api_calls={result['api_calls']}")

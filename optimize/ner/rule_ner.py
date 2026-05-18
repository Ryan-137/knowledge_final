"""L1 规则 NER：基于别名词典 + 正则模式扫描，提取技术实体 mention。

流程
----
1. 从 data/dictionaries/skill_aliases.json 构建「归一化别名 → entity_id」索引。
2. 对每个别名编译正则模式，长别名优先（贪婪最长匹配），
   同时保留中英文两种词边界规则。
3. 对 staged_documents.jsonl 的每一个句子逐一扫描：
   a. abbr_expansion 展开缩写（先于匹配执行，原始偏移通过映射还原）。
   b. 正则依次尝试，命中则记录 surface / char_start / char_end / entity_id。
   c. 在命中位置前后 ±_CONTEXT_WINDOW 字符内判断强度/否定上下文。
   d. phrase_rules 二次扫描，命中特殊语义（如"英语约束"）。
4. 将所有 mention 追加写入 data/staging/mentions.jsonl（幂等：sha256 缓存）。

输出格式（每行一条 mention）
-------------------------------
{
  "mention_id":       "m_fairCV_000000_tech_skills_0_286_292",
  "doc_id":           "fairCV_000000",
  "section_id":       "fairCV_000000_tech_skills_0",
  "section_type":     "tech_skills",
  "surface":          "Python",
  "normalized":       "python",
  "char_start":       286,
  "char_end":         292,
  "context_snippet":  "- Python（基础掌握）",
  "candidates": [
    {"entity_id": "skill_python", "method": "alias_exact", "score": 0.93}
  ],
  "linked_entity_id": "skill_python",
  "link_method":      "alias_exact",
  "link_confidence":  0.93,
  "status":           "rule_match",
  "is_negative":      false,
  "intensity":        "light_positive"
}

运行方式
--------
    python -m optimize.ner.rule_ner
    python -m optimize.ner.rule_ner --limit 50
    python -m optimize.ner.rule_ner --sources fairCV
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from optimize.config import cfg
from optimize.utils.file_utils import (
    append_jsonl_batch,
    ensure_dir,
    read_json,
    read_jsonl,
)
from optimize.utils.hash_utils import text_sha256
from optimize.utils.logging_utils import get_pipeline_logger

logger = get_pipeline_logger("ner.rule_ner")

# 输出路径
_MENTIONS_PATH = cfg.paths.staging_mentions

# 上下文窗口（字符数），用于判断强度/否定
_CONTEXT_WINDOW = 15

# 规则层 mention 的基础置信度
_BASE_CONFIDENCE = 0.93

_OffsetSpan = tuple[int, int]


@dataclass(frozen=True)
class AliasPattern:
    """一条编译好的别名匹配规则。"""
    entity_id: str
    alias:     str
    pattern:   re.Pattern


@dataclass
class Mention:
    """单条实体 mention，对应输出 JSONL 中的一行。"""
    mention_id:       str
    doc_id:           str
    section_id:       str
    section_type:     str
    surface:          str
    normalized:       str
    char_start:       int
    char_end:         int
    context_snippet:  str
    entity_id:        str
    method:           str
    confidence:       float
    is_negative:      bool
    intensity:        str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mention_id":       self.mention_id,
            "doc_id":           self.doc_id,
            "section_id":       self.section_id,
            "section_type":     self.section_type,
            "surface":          self.surface,
            "normalized":       self.normalized,
            "char_start":       self.char_start,
            "char_end":         self.char_end,
            "context_snippet":  self.context_snippet,
            "candidates": [
                {"entity_id": self.entity_id, "method": self.method, "score": round(self.confidence, 4)}
            ],
            "linked_entity_id": self.entity_id,
            "link_method":      self.method,
            "link_confidence":  round(self.confidence, 4),
            "status":           "rule_match",
            "is_negative":      self.is_negative,
            "intensity":        self.intensity,
        }


class RuleNER:
    """基于别名词典的规则 NER 引擎。"""

    def __init__(self) -> None:
        # 加载资源
        all_aliases: dict[str, list[str]]   = read_json(cfg.paths.dict_skill_aliases)
        self._prefs: dict[str, list[str]]   = read_json(cfg.paths.dict_pref_patterns)
        self._parsing: dict[str, Any]       = read_json(cfg.paths.dict_parsing_patterns)
        self._abbr: dict[str, str]          = read_json(cfg.paths.abbr_expansion)

        # 只保留 evidence 层节点的 alias，排除 ability/composite/direction/role
        # 原因：后者是推导层，不是用户能直接表达的实体
        evidence_node_ids = self._load_evidence_node_ids()
        self._aliases = {
            nid: aliases
            for nid, aliases in all_aliases.items()
            if nid in evidence_node_ids
        }
        logger.info(
            "alias 过滤：全量 %d → evidence 层 %d 个节点",
            len(all_aliases), len(self._aliases),
        )

        # 构建匹配索引
        self._alias_patterns: list[AliasPattern] = self._build_alias_patterns()
        self._phrase_rules: list[dict[str, Any]] = self._parsing.get("phrase_rules", [])

        logger.info(
            "规则 NER 初始化完成：alias_patterns=%d  phrase_rules=%d",
            len(self._alias_patterns), len(self._phrase_rules),
        )

    @staticmethod
    def _load_evidence_node_ids() -> set[str]:
        """从 seeds/nodes.json 中读取所有 evidence 层节点 ID。"""
        nodes = read_json(cfg.paths.seeds_nodes)
        return {n["id"] for n in nodes if n.get("layer") == "evidence"}

    def _build_alias_patterns(self) -> list[AliasPattern]:
        """为每条别名编译正则模式，按别名长度降序排列（贪婪最长匹配）。"""
        patterns: list[AliasPattern] = []
        for entity_id, alias_list in self._aliases.items():
            for alias in alias_list:
                if not alias.strip():
                    continue
                pat = self._compile_pattern(alias)
                patterns.append(AliasPattern(entity_id=entity_id, alias=alias, pattern=pat))
        # 长别名优先，防止"Spring"遮住"Spring Boot"
        patterns.sort(key=lambda p: len(p.alias), reverse=True)
        return patterns

    @staticmethod
    def _compile_pattern(alias: str) -> re.Pattern:
        """为一个 alias 编译正则。

        纯 ASCII 字母/数字串使用词边界（\\b），避免误匹配"C"在"CSDN"中；
        含中文或特殊字符的 alias 不加词边界（中文本无空格分词边界）。
        """
        escaped = re.escape(alias)
        if re.fullmatch(r"[a-z0-9][a-z0-9 .+#\-]*", alias, re.IGNORECASE):
            # 纯 ASCII 技术词：要求前后不是字母或数字
            return re.compile(
                rf"(?<![a-z0-9]){escaped}(?![a-z0-9])",
                flags=re.IGNORECASE,
            )
        # 含中文或混合内容：直接匹配，不加词边界
        return re.compile(escaped, flags=re.IGNORECASE)

    def _expand_abbr_with_offsets(self, text: str) -> tuple[str, list[_OffsetSpan]]:
        """展开缩写，并记录展开后每个字符对应的原文字符区间。"""
        if not self._abbr:
            return text, [(i, i + 1) for i in range(len(text))]

        abbr_items = sorted(self._abbr.items(), key=lambda item: len(item[0]), reverse=True)
        pattern = re.compile(
            "|".join(rf"(?<![a-z]){re.escape(abbr)}(?![a-z])" for abbr, _ in abbr_items),
            flags=re.IGNORECASE,
        )
        expansion_by_key = {abbr.lower(): expansion for abbr, expansion in abbr_items}

        expanded_parts: list[str] = []
        offset_map: list[_OffsetSpan] = []
        cursor = 0
        for match in pattern.finditer(text):
            if match.start() > cursor:
                raw_part = text[cursor:match.start()]
                expanded_parts.append(raw_part)
                offset_map.extend((i, i + 1) for i in range(cursor, match.start()))

            raw_abbr = match.group(0)
            expansion = expansion_by_key.get(raw_abbr.lower(), raw_abbr)
            expanded_parts.append(expansion)
            offset_map.extend((match.start(), match.end()) for _ in expansion)
            cursor = match.end()

        if cursor < len(text):
            raw_part = text[cursor:]
            expanded_parts.append(raw_part)
            offset_map.extend((i, i + 1) for i in range(cursor, len(text)))

        return "".join(expanded_parts), offset_map

    @staticmethod
    def _map_expanded_span(
        offset_map: list[_OffsetSpan],
        start: int,
        end: int,
    ) -> tuple[int, int] | None:
        """将展开文本中的 span 映射回原文 span，无法映射时返回 None。"""
        if start < 0 or end <= start or end > len(offset_map):
            return None
        spans = offset_map[start:end]
        orig_start = min(span[0] for span in spans)
        orig_end = max(span[1] for span in spans)
        if orig_end <= orig_start:
            return None
        return orig_start, orig_end

    def _detect_intensity(self, context: str) -> tuple[bool, str]:
        """根据上下文判断否定标记和强度标签。

        Returns:
            (is_negative, intensity_label)
        """
        ctx_lower = context.lower()

        # 否定优先判断
        for kw in self._prefs.get("negative", []):
            if kw in ctx_lower:
                return True, "negative"

        # 强度标签（从强到弱）
        for level in ("strong_positive", "medium_positive", "light_positive", "weak_positive"):
            for kw in self._prefs.get(level, []):
                if kw in ctx_lower:
                    return False, level

        return False, "neutral"

    def _build_context(self, text: str, start: int, end: int) -> str:
        """截取 mention 前后 _CONTEXT_WINDOW 字符作为上下文片段。"""
        ctx_start = max(0, start - _CONTEXT_WINDOW)
        ctx_end   = min(len(text), end + _CONTEXT_WINDOW)
        return text[ctx_start:ctx_end]

    def scan_sentence(
        self,
        sentence_text: str,
        sentence_abs_start: int,
        doc_id: str,
        section_id: str,
        section_type: str,
    ) -> list[Mention]:
        """对单个句子扫描所有 mention。

        Args:
            sentence_text:       句子原始文本。
            sentence_abs_start:  句子在整篇文档 full_text 中的起始字符偏移。
            doc_id:              文档 ID。
            section_id:          所属章节 ID。
            section_type:        章节类型。

        Returns:
            该句子中发现的 mention 列表。
        """
        # 先展开缩写，并保留展开文本到原文的偏移映射。
        expanded, offset_map = self._expand_abbr_with_offsets(sentence_text)

        mentions: list[Mention] = []
        # 已被更长 mention 覆盖的字符位置（防止双重计数）
        covered: set[int] = set()

        for ap in self._alias_patterns:
            for m in ap.pattern.finditer(expanded):
                span_start = m.start()
                span_end   = m.end()

                # 跳过已被更长 pattern 覆盖的位置
                if any(pos in covered for pos in range(span_start, span_end)):
                    continue

                original_span = self._map_expanded_span(offset_map, span_start, span_end)
                if original_span is None:
                    continue
                original_start, original_end = original_span

                surface    = sentence_text[original_start:original_end]
                normalized = ap.alias.lower()
                context    = self._build_context(expanded, span_start, span_end)
                is_neg, intensity = self._detect_intensity(context)

                # 置信度修正
                confidence = _BASE_CONFIDENCE
                if intensity == "strong_positive":
                    confidence = min(1.0, confidence + 0.05)
                elif intensity in ("weak_positive", "neutral"):
                    confidence = max(0.5, confidence - 0.08)

                abs_start = sentence_abs_start + original_start
                abs_end   = sentence_abs_start + original_end
                mid = f"m_{doc_id}_{section_id}_{abs_start}_{abs_end}"

                mentions.append(Mention(
                    mention_id      = mid,
                    doc_id          = doc_id,
                    section_id      = section_id,
                    section_type    = section_type,
                    surface         = surface,
                    normalized      = normalized,
                    char_start      = abs_start,
                    char_end        = abs_end,
                    context_snippet = context,
                    entity_id       = ap.entity_id,
                    method          = "alias_exact",
                    confidence      = confidence,
                    is_negative     = is_neg,
                    intensity       = intensity,
                ))
                # 标记已覆盖位置
                covered.update(range(span_start, span_end))

        # phrase_rules 二次扫描（捕获词典未覆盖的特殊语义）
        for rule in self._phrase_rules:
            label   = rule.get("label", "")
            phrases = rule.get("phrases", [])
            for phrase in phrases:
                if phrase in expanded.lower():
                    signals = rule.get("signals", [])
                    for sig in signals:
                        nid  = sig.get("node_id", "")
                        score = float(sig.get("score", 0.7))
                        if not nid or score < 0.3:
                            continue
                        mid = f"m_{doc_id}_{section_id}_phrase_{label}_{nid}"
                        mentions.append(Mention(
                            mention_id      = mid,
                            doc_id          = doc_id,
                            section_id      = section_id,
                            section_type    = section_type,
                            surface         = phrase,
                            normalized      = phrase.lower(),
                            char_start      = sentence_abs_start,
                            char_end        = sentence_abs_start + len(sentence_text),
                            context_snippet = sentence_text[:60],
                            entity_id       = nid,
                            method          = "phrase_rule",
                            confidence      = score,
                            is_negative     = False,
                            intensity       = "neutral",
                        ))

        return mentions

    def process_document(self, staged_doc: dict[str, Any]) -> list[Mention]:
        """处理单篇 staged 文档，返回全部 mention。"""
        doc_id  = staged_doc["doc_id"]
        all_mentions: list[Mention] = []

        for section in staged_doc.get("sections", []):
            section_id   = section["section_id"]
            section_type = section["section_type"]

            # 个人信息章节 NER 价值极低，跳过
            if section_type == "personal_info":
                continue

            for sent in section.get("sentences", []):
                text      = sent["text"]
                abs_start = sent["char_start"]
                if not text.strip():
                    continue
                found = self.scan_sentence(
                    sentence_text       = text,
                    sentence_abs_start  = abs_start,
                    doc_id              = doc_id,
                    section_id          = section_id,
                    section_type        = section_type,
                )
                all_mentions.extend(found)

        return all_mentions


def _matches_sources(doc: dict[str, Any], sources: list[str] | None) -> bool:
    """按 source_group 或 source_name 过滤 staged 文档。"""
    if not sources:
        return True
    source_group = doc.get("source_group", "")
    source_name = doc.get("source_name", "")
    return source_group in sources or source_name in sources


def run(
    sources: list[str] | None = None,
    limit: int | None = None,
    append: bool = False,
) -> dict[str, int]:
    """执行 L1 规则 NER 全量流程。

    Args:
        sources: 只处理指定来源（'fairCV' / 'jd'），None 表示全部。
        limit:   每个来源最多处理的文档数。

    Returns:
        统计字典：docs_processed / mentions_total / docs_skipped
    """
    ensure_dir(cfg.paths.staging_root)
    if not append and _MENTIONS_PATH.exists():
        _MENTIONS_PATH.unlink()

    ner     = RuleNER()
    cache: set[str] = set()
    docs    = read_jsonl(_MENTIONS_PATH.parent / "staged_documents.jsonl")

    stats   = {"docs_processed": 0, "docs_skipped": 0, "mentions_total": 0}
    counts_by_source: dict[str, int] = {}

    source_limit: dict[str, int] = {}  # 记录每个 source 已处理数量

    for doc in docs:
        src = doc.get("source_name", "unknown")

        # 来源过滤
        if not _matches_sources(doc, sources):
            continue

        # limit 控制
        if limit is not None:
            count = source_limit.get(src, 0)
            if count >= limit:
                continue
            source_limit[src] = count + 1

        # 幂等：用文档 sha256 跳过已处理的
        doc_sha = doc.get("sha256", "")
        if doc_sha and doc_sha in cache:
            stats["docs_skipped"] += 1
            continue

        mentions = ner.process_document(doc)
        if mentions:
            append_jsonl_batch(_MENTIONS_PATH, (m.as_dict() for m in mentions))

        cache.add(doc_sha)
        stats["docs_processed"] += 1
        stats["mentions_total"] += len(mentions)
        counts_by_source[src] = counts_by_source.get(src, 0) + 1

        if stats["docs_processed"] % 200 == 0:
            logger.info(
                "  已处理 %d 篇，累计 mention %d 条",
                stats["docs_processed"], stats["mentions_total"],
            )

    logger.info(
        "L1 NER 完成：processed=%d  mentions=%d  skipped=%d",
        stats["docs_processed"], stats["mentions_total"], stats["docs_skipped"],
    )
    logger.info("各来源处理量：%s", counts_by_source)
    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="*", default=None,
                   help="指定数据来源，可用 source_group（fairCV/jd）或具体 source_name（如 csv_import）")
    p.add_argument("--limit", type=int, default=None,
                   help="每个来源最多处理的文档数（调试用）")
    p.add_argument("--append", action="store_true",
                   help="保留已有 mentions.jsonl 并追加新结果（默认覆盖当前步骤输出）")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run(sources=args.sources, limit=args.limit, append=args.append)
    print(f"完成：docs={result['docs_processed']}  mentions={result['mentions_total']}")

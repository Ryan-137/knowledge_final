"""Staging 清洗主脚本：将 raw zone 文档规范化为带章节和分句偏移的 staged_documents.jsonl。

主要流程
--------
1. 扫描 data/raw/fairCV/ 和 data/raw/jd/*/ 下的所有 JSON 文件。
2. 对每份文档：
   - FairCV 简历：解析 Markdown，按 ### 标题拆分章节，清洗格式字符。
   - JD 招聘文档：按字段（job_title / requirements / preferred 等）拆分章节。
3. 对每个章节调用 segment_sentences 生成带字符偏移的句子列表。
4. 以 JSONL 格式追加写入 data/staging/staged_documents.jsonl。
   已处理过的文档（sha256 命中缓存）自动跳过，保证幂等。
5. 汇报章节类型分布和统计信息。

输出格式（每行一个文档）
------------------------
{
  "doc_id":      "fairCV_000001",
  "source_name": "fairCV",
  "doc_type":    "resume",
  "language":    "zh",
  "sections": [
    {
      "section_id":   "fairCV_000001_tech_skills",
      "section_type": "tech_skills",
      "raw_field":    "resume_text",
      "text":         "Java, Python, MySQL ...",
      "char_start":   42,
      "char_end":     90,
      "sentences": [
        {"text": "Java, Python, MySQL", "char_start": 42, "char_end": 61}
      ]
    }
  ],
  "full_text": "（完整清洗后文本，拼接自所有章节）",
  "sha256":    "...",
  "metadata":  { "position": "后端开发工程师", "skill_level": "中等" }
}

运行方式
--------
    python -m optimize.staging.clean_documents
    python -m optimize.staging.clean_documents --sources fairCV jd
    python -m optimize.staging.clean_documents --limit 100
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

from optimize.config import cfg
from optimize.staging.segment_sentences import split_sentences
from optimize.utils.file_utils import (
    append_jsonl,
    ensure_dir,
    read_json,
)
from optimize.utils.hash_utils import text_sha256
from optimize.utils.logging_utils import get_pipeline_logger

logger = get_pipeline_logger("staging.clean_documents")

# 输出文件路径
_OUTPUT_PATH = cfg.paths.staging_root / "staged_documents.jsonl"

# FairCV Markdown 章节标题 → section_type 映射
_FAIRCV_SECTION_MAP: dict[str, str] = {
    "教育背景":   "education",
    "技术技能":   "tech_skills",
    "专业技能":   "tech_skills",
    "实践经历":   "experience",
    "工作经历":   "experience",
    "项目经历":   "projects",
    "项目经验":   "projects",      # FairCV 实际使用的标题
    "自我评价":   "self_eval",
    "其他亮点":   "highlights",    # FairCV 实际使用的标题
    "个人信息":   "personal_info",
    "荣誉奖励":   "awards",
    "证书资质":   "certificates",
}

# 对 NER 意义不大的章节类型（staging 时保留，NER 时跳过）
_LOW_PRIORITY_SECTIONS = {"personal_info", "awards"}

# Markdown 清洗正则
_MD_BOLD_RE    = re.compile(r"\*{1,3}(.+?)\*{1,3}")   # **粗体** 或 *斜体*
_MD_HEADER_RE  = re.compile(r"^#{1,6}\s+")             # ### 标题前缀
_MD_HR_RE      = re.compile(r"^-{3,}$")               # --- 分隔线
_MD_BULLET_RE  = re.compile(r"^[-*]\s+")              # - 列表项前缀
_MD_BLANK_RE   = re.compile(r"\n{3,}")                 # 多余空行


def _clean_markdown(text: str) -> str:
    """移除 Markdown 格式字符，保留纯文本内容。"""
    # 去掉粗体/斜体标记，保留内容
    text = _MD_BOLD_RE.sub(r"\1", text)
    lines = []
    for line in text.splitlines():
        # 去掉标题前缀（### xxx → xxx）
        line = _MD_HEADER_RE.sub("", line)
        # 去掉水平分隔线
        if _MD_HR_RE.match(line.strip()):
            continue
        # 去掉列表符号（- xxx → xxx）
        line = _MD_BULLET_RE.sub("", line)
        lines.append(line)
    # 折叠多余空行
    cleaned = _MD_BLANK_RE.sub("\n\n", "\n".join(lines))
    return cleaned.strip()


def _parse_faircv_sections(resume_text: str) -> list[dict[str, Any]]:
    """将 FairCV 简历 Markdown 按 ### 标题拆分为章节列表。

    每个章节包含 section_type、raw_header、text（已去除 Markdown）
    以及在整篇清洗文本中的字符偏移。
    """
    # 按 ### 级别标题切割
    header_pattern = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
    sections: list[dict[str, Any]] = []

    matches = list(header_pattern.finditer(resume_text))
    if not matches:
        # 没有 Markdown 标题，整体作为一个章节
        cleaned = _clean_markdown(resume_text)
        if cleaned:
            sections.append({
                "section_type": "full_text",
                "raw_header":   "",
                "raw_text":     cleaned,
            })
        return sections

    for i, match in enumerate(matches):
        header_text = match.group(1).strip()
        section_start = match.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(resume_text)
        raw_content = resume_text[section_start:section_end]
        cleaned = _clean_markdown(raw_content).strip()

        if not cleaned:
            continue

        # 通过关键词匹配 section_type
        section_type = "other"
        for keyword, stype in _FAIRCV_SECTION_MAP.items():
            if keyword in header_text:
                section_type = stype
                break

        sections.append({
            "section_type": section_type,
            "raw_header":   header_text,
            "raw_text":     cleaned,
        })

    return sections


def _parse_jd_sections(content: dict[str, Any]) -> list[dict[str, Any]]:
    """将 JD 的各字段映射为章节列表。"""
    field_map = [
        ("job_title",        "title",        "job_title"),
        ("requirements",     "requirements", "requirements"),
        ("responsibilities", "description",  "responsibilities"),
        ("preferred",        "preferred",    "preferred"),
        ("full_text",        "full_text",    "full_text"),
    ]
    sections: list[dict[str, Any]] = []
    seen_texts: set[str] = set()  # 去重：requirements 和 full_text 常常相同

    for field_key, section_type, raw_field in field_map:
        text = str(content.get(field_key, "") or "").strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        sections.append({
            "section_type": section_type,
            "raw_header":   field_key,
            "raw_text":     text,
        })

    return sections


def _build_staged_doc(raw_doc: dict[str, Any]) -> dict[str, Any] | None:
    """将单份 raw 文档转换为 staged 文档（带章节和分句偏移）。

    Returns:
        staged 文档字典；若文档没有有效文本则返回 None。
    """
    doc_id      = raw_doc["doc_id"]
    source_name = raw_doc.get("source_name", "unknown")
    doc_type    = raw_doc.get("doc_type", "unknown")
    language    = raw_doc.get("language", "zh")
    content     = raw_doc.get("content", {})

    # 根据文档类型拆分章节
    if doc_type == "resume":
        raw_sections = _parse_faircv_sections(content.get("resume_text", ""))
        metadata = {
            "position":   content.get("position", ""),
            "skill_level": content.get("skill_level", ""),
        }
    elif doc_type == "jd":
        raw_sections = _parse_jd_sections(content)
        metadata = {
            "job_title":    content.get("job_title", ""),
            "company_name": content.get("company_name", ""),
            "location":     content.get("location", ""),
            "salary_range": content.get("salary_range", ""),
        }
    else:
        return None

    if not raw_sections:
        return None

    # 拼接全文，并为每个章节计算绝对字符偏移
    full_text_parts: list[str] = []
    sections: list[dict[str, Any]] = []
    cursor = 0  # 当前在 full_text 中的写入位置

    for idx, sec in enumerate(raw_sections):
        text = sec["raw_text"]
        section_id = f"{doc_id}_{sec['section_type']}_{idx}"
        char_start = cursor
        char_end   = cursor + len(text)

        # 对章节文本进行分句
        sentences = [s.as_dict() for s in split_sentences(text, offset=char_start)]

        sections.append({
            "section_id":   section_id,
            "section_type": sec["section_type"],
            "raw_field":    sec["raw_header"],
            "text":         text,
            "char_start":   char_start,
            "char_end":     char_end,
            "sentences":    sentences,
        })

        full_text_parts.append(text)
        cursor = char_end + 1  # +1 预留章节间分隔符的空间

    full_text = "\n".join(full_text_parts)
    if not full_text.strip():
        return None

    return {
        "doc_id":      doc_id,
        "source_name": source_name,
        "source_group": "fairCV" if doc_type == "resume" else "jd",
        "doc_type":    doc_type,
        "language":    language,
        "sections":    sections,
        "full_text":   full_text,
        "sha256":      text_sha256(full_text),
        "metadata":    metadata,
    }


def process(
    sources: list[str] | None = None,
    limit: int | None = None,
    append: bool = False,
) -> dict[str, int]:
    """执行 staging 清洗主流程。

    Args:
        sources: 数据源列表，可选 'fairCV' / 'jd'，None 表示全部。
        limit: 每个数据源最多处理的文档数，None 表示不限。

    Returns:
        统计字典，包含 processed/skipped/section_type_counts 等。
    """
    ensure_dir(cfg.paths.staging_root)

    # 默认重新生成当前步骤产物，避免旧 JSONL 混入新 run；--append 仅用于调试追加。
    if not append and _OUTPUT_PATH.exists():
        _OUTPUT_PATH.unlink()

    processed_cache: set[str] = set()

    # 收集待处理的 raw 文件路径；JD 支持多个来源子目录，如 csv_import / cn_skillspan_lkst
    source_dirs: dict[str, Path] = {
        "fairCV": cfg.paths.raw_fairCV,
        "jd":     cfg.paths.raw_jd,
    }
    active_sources = sources or list(source_dirs.keys())

    stats: dict[str, int] = {"processed": 0, "skipped_cache": 0, "skipped_empty": 0}
    section_type_counts: Counter = Counter()

    for source_name in active_sources:
        src_dir = source_dirs.get(source_name)
        if src_dir is None or not src_dir.exists():
            logger.warning("数据源目录不存在，跳过：%s", source_name)
            continue

        if source_name == "jd":
            raw_files = sorted(src_dir.glob("*/*.json"))
        else:
            raw_files = sorted(src_dir.glob("*.json"))
        if limit:
            raw_files = raw_files[:limit]

        logger.info("处理 %s：%d 份文档", source_name, len(raw_files))

        for raw_path in raw_files:
            raw_doc = read_json(raw_path)

            # 幂等检查：sha256 已在缓存中则跳过
            raw_sha = raw_doc.get("sha256", "")
            if raw_sha and raw_sha in processed_cache:
                stats["skipped_cache"] += 1
                continue

            staged = _build_staged_doc(raw_doc)
            if staged is None:
                stats["skipped_empty"] += 1
                continue

            append_jsonl(_OUTPUT_PATH, staged)
            processed_cache.add(raw_sha)
            stats["processed"] += 1

            for sec in staged["sections"]:
                section_type_counts[sec["section_type"]] += 1

            if stats["processed"] % 500 == 0:
                logger.info("  已清洗 %d 份文档…", stats["processed"])

    stats["section_type_counts"] = dict(section_type_counts)
    logger.info(
        "Staging 完成：processed=%d  skipped_cache=%d  skipped_empty=%d",
        stats["processed"], stats["skipped_cache"], stats["skipped_empty"],
    )
    logger.info("章节类型分布：%s", section_type_counts)
    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="*", choices=["fairCV", "jd"],
                   default=None, help="指定数据源，默认处理全部")
    p.add_argument("--limit", type=int, default=None,
                   help="每个数据源最多处理的文档数（调试用）")
    p.add_argument("--append", action="store_true",
                   help="保留已有 staged_documents.jsonl 并追加新结果（默认覆盖当前步骤输出）")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = process(sources=args.sources, limit=args.limit, append=args.append)
    print(f"完成：processed={result['processed']}  skipped={result['skipped_cache']}")
    print("章节类型分布：", result.get("section_type_counts", {}))

"""将 CN_skillspan train 集按岗位聚合为候选 JD raw 文档。

该脚本只处理 train 数据，输出作为中文招聘文本补充源进入 raw/staging/NER
主流程；test 数据集必须保留给独立评测，不能进入扩充语料。

运行方式：
    python -m optimize.data_collection.import_skillspan_jd
    python -m optimize.data_collection.import_skillspan_jd --input optimize/CN_skillspan_lkst_train.json
"""

from __future__ import annotations

import argparse
import html
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from optimize.config import cfg
from optimize.utils.file_utils import ensure_dir, read_json, save_raw_doc
from optimize.utils.hash_utils import dict_sha256, text_sha256
from optimize.utils.logging_utils import get_pipeline_logger

logger = get_pipeline_logger("data_collection.import_skillspan_jd")

SOURCE_NAME = "cn_skillspan_lkst"
_HTML_BR_RE = re.compile(r"(?:&lt;|<)\s*br\s*(?:&gt;|>)", flags=re.IGNORECASE)
_BROKEN_BR_TOKENS = {"&lt", "lt", "<", "br&gt", "br>", "br", "&gt", "gt", ">"}


@dataclass(frozen=True)
class SkillSpanSentence:
    """train 集中的一条句子记录。"""

    global_id: str
    sent_id: int
    sentence: str
    source_domain: str


def _as_train_sentence(row: dict[str, Any]) -> SkillSpanSentence | None:
    """兼容 train 文件的直接字段结构，忽略缺少关键字段的行。"""
    global_id = row.get("global_id")
    sent_id = row.get("sent_id")
    sentence = str(row.get("sentence", "") or "").strip()
    if global_id is None or sent_id is None or not sentence:
        return None
    return SkillSpanSentence(
        global_id=str(global_id),
        sent_id=int(sent_id),
        sentence=sentence,
        source_domain=str(row.get("source_domain", "") or ""),
    )


def _clean_sentence(text: str) -> str:
    """清理 SkillSpan 切句中常见的 HTML 换行残片。"""
    text = html.unescape(text).strip()
    text = _HTML_BR_RE.sub("\n", text)
    if text.strip().lower() in _BROKEN_BR_TOKENS:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def build_raw_docs(rows: list[dict[str, Any]], snapshot_date: str | None = None) -> list[dict[str, Any]]:
    """将 train 行按 global_id 聚合为 raw JD 文档。"""
    snapshot = snapshot_date or date.today().isoformat()
    grouped: dict[str, list[SkillSpanSentence]] = defaultdict(list)

    for row in rows:
        sent = _as_train_sentence(row)
        if sent is not None:
            grouped[sent.global_id].append(sent)

    docs: list[dict[str, Any]] = []
    for global_id, sentences in sorted(grouped.items(), key=lambda item: item[0]):
        ordered = sorted(sentences, key=lambda s: s.sent_id)
        cleaned_parts = [_clean_sentence(s.sentence) for s in ordered]
        cleaned_parts = [part for part in cleaned_parts if part]
        full_text = "\n".join(cleaned_parts).strip()
        if not full_text:
            continue

        source_domains = sorted({s.source_domain for s in ordered if s.source_domain})
        content = {
            "job_title": f"SkillSpan 招聘文本 {global_id}",
            "company_name": "",
            "location": "",
            "salary_range": "",
            "requirements": full_text,
            "responsibilities": "",
            "preferred": "",
            "full_text": full_text,
        }
        digest = text_sha256(full_text)[:8]
        doc_id = f"jd_{SOURCE_NAME}_{global_id}_{digest}"
        payload = {
            "doc_id": doc_id,
            "source_name": SOURCE_NAME,
            "source_url": "https://arxiv.org/abs/2604.23009",
            "snapshot_time": f"{snapshot}T00:00:00Z",
            "language": "zh",
            "license_note": "Chinese-SkillSpan research dataset; verify dataset license before redistribution.",
            "doc_type": "jd",
            "content": content,
            "metadata": {
                "global_id": global_id,
                "source_domains": source_domains,
                "sentence_count": len(ordered),
                "valid_sentence_count": len(cleaned_parts),
                "usage_note": "train split only; low-weight JD supplement, not gold evaluation data.",
            },
        }
        payload["sha256"] = dict_sha256(content)
        docs.append(payload)

    return docs


def run(input_path: Path, output_dir: Path | None = None, limit: int | None = None) -> dict[str, int]:
    """执行导入并返回统计信息。"""
    rows = read_json(input_path)
    if not isinstance(rows, list):
        raise ValueError(f"{input_path} 必须是 JSON 数组")

    docs = build_raw_docs(rows)
    if limit is not None:
        docs = docs[:limit]

    out_dir = ensure_dir(output_dir or (cfg.paths.raw_jd / SOURCE_NAME))
    for doc in docs:
        save_raw_doc(out_dir, doc["doc_id"], doc)

    stats = {
        "input_rows": len(rows),
        "raw_docs": len(docs),
    }
    logger.info("SkillSpan train 导入完成：%s", stats)
    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("optimize/CN_skillspan_lkst_train.json"),
                   help="CN_skillspan_lkst_train.json 路径")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="输出 raw JD 目录，默认 optimize/pipeline_data/raw/jd/cn_skillspan_lkst")
    p.add_argument("--limit", type=int, default=None, help="调试时限制输出文档数")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run(input_path=args.input, output_dir=args.output_dir, limit=args.limit)
    print(f"完成：input_rows={result['input_rows']} raw_docs={result['raw_docs']}")

"""生成答辩用实体抽取与质量案例 JSON 报告。

该脚本只生成 JSON 和终端摘要，不生成 Markdown 文件。

运行方式：
    python -m optimize.evaluation.case_report
    python -m optimize.evaluation.case_report --limit-per-section 5
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

from optimize.config import cfg
from optimize.evaluation.validate_entity_quality import validate_entity_quality
from optimize.utils.file_utils import ensure_dir, read_json, read_jsonl, write_json


_DEFAULT_OUTPUT_PATH = cfg.paths.canonical_root / "entity_case_report.json"
_QUALITY_REPORT_PATH = cfg.paths.canonical_root / "entity_quality_report.json"
_PUNCT_ONLY = set("，。；、,.!?！？（）()[]【】{}<>《》:：;；-_/\\|")


def _try_read_json(path: Path, missing: list[str]) -> Any:
    if not path.exists():
        missing.append(str(path))
        return None
    return read_json(path)


def _try_read_jsonl(path: Path, missing: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        missing.append(str(path))
        return []
    return read_jsonl(path)


def _take_unique(records: list[dict[str, Any]], key: str, limit: int) -> list[dict[str, Any]]:
    """按 key 去重取样，让案例覆盖更多不同词面或实体。"""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for record in records:
        marker = str(record.get(key, ""))
        if marker in seen:
            continue
        seen.add(marker)
        result.append(record)
        if len(result) >= limit:
            break
    return result


def _is_displayable_surface(value: Any) -> bool:
    """答辩样本过滤空串、纯标点等不可展示词面。"""
    if not isinstance(value, str):
        return False
    text = value.strip()
    return bool(text) and not all(char in _PUNCT_ONLY for char in text)


def _rule_match_examples(mentions: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    records = []
    for mention in mentions:
        if mention.get("status") != "rule_match":
            continue
        if not _is_displayable_surface(mention.get("surface")):
            continue
        records.append(
            {
                "surface": mention.get("surface"),
                "entity_id": mention.get("linked_entity_id"),
                "doc_id": mention.get("doc_id"),
                "section_id": mention.get("section_id"),
                "confidence": mention.get("confidence"),
                "link_method": mention.get("link_method"),
            }
        )
    return _take_unique(records, "entity_id", limit)


def _auto_confirmed_alias_examples(mentions: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    records = []
    for mention in mentions:
        if mention.get("status") != "auto_confirmed" and mention.get("link_method") != "embedding_high":
            continue
        if not _is_displayable_surface(mention.get("surface")):
            continue
        records.append(
            {
                "surface": mention.get("surface"),
                "entity_id": mention.get("linked_entity_id"),
                "doc_id": mention.get("doc_id"),
                "score": mention.get("score"),
                "confidence": mention.get("confidence"),
                "link_method": mention.get("link_method"),
            }
        )
    return _take_unique(records, "surface", limit)


def _needs_review_examples(disambig_log: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    records = []
    for item in disambig_log:
        if item.get("status") != "needs_review":
            continue
        if not _is_displayable_surface(item.get("surface")):
            continue
        records.append(
            {
                "surface": item.get("surface"),
                "best_entity": item.get("best_entity"),
                "best_score": item.get("best_score"),
                "entity_type_hint": item.get("entity_type_hint"),
                "candidates": item.get("candidates", [])[:3],
            }
        )
    return _take_unique(records, "surface", limit)


def _flatten_extracted_entities(skills_payload: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(skills_payload, dict):
        return []
    records = []
    for category, nodes in skills_payload.items():
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if not isinstance(node, dict) or node.get("origin") != "extracted":
                continue
            records.append(
                {
                    "category": category,
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "aliases": node.get("aliases", []),
                    "source_note": node.get("source_note"),
                    "review_status": node.get("review_status"),
                }
            )
    return records[:limit]


def _external_alignment_examples(skills_payload: Any, limit: int) -> dict[str, list[dict[str, Any]]]:
    refs = []
    if isinstance(skills_payload, dict):
        for category, nodes in skills_payload.items():
            if not isinstance(nodes, list):
                continue
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                for ref in node.get("external_refs", []) if isinstance(node.get("external_refs"), list) else []:
                    if not isinstance(ref, dict):
                        continue
                    refs.append(
                        {
                            "entity_id": node.get("id"),
                            "entity_name": node.get("name"),
                            "category": category,
                            "source": ref.get("source"),
                            "label": ref.get("label") or ref.get("tool_name") or ref.get("name") or ref.get("title"),
                            "similarity": ref.get("similarity"),
                        }
                    )
    numeric_refs = [r for r in refs if isinstance(r.get("similarity"), (int, float))]
    return {
        "high_similarity": sorted(numeric_refs, key=lambda r: float(r["similarity"]), reverse=True)[:limit],
        "low_similarity": sorted(numeric_refs, key=lambda r: float(r["similarity"]))[:limit],
    }


def _profile_mapping_examples(profiles: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(profiles, list):
        return []
    records = []
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        mapped = profile.get("mapped_node_ids", [])
        records.append(
            {
                "profile_id": profile.get("profile_id"),
                "source_type": profile.get("source_type"),
                "source_title": profile.get("source_title"),
                "mapped_count": len(mapped) if isinstance(mapped, list) else 0,
                "mapped_node_ids_sample": mapped[:10] if isinstance(mapped, list) else [],
            }
        )
    return records[:limit]


def _quality_samples(quality_report: dict[str, Any] | None, code: str, limit: int) -> list[dict[str, Any]]:
    if not quality_report:
        return []
    records = []
    for item in quality_report.get("warnings", []) + quality_report.get("errors", []):
        if item.get("code") == code or (code.endswith("*") and str(item.get("code", "")).startswith(code[:-1])):
            records.append(item)
    return records[:limit]


def build_case_report(
    *,
    limit_per_section: int = 8,
    output_path: Path | None = _DEFAULT_OUTPUT_PATH,
    mentions_path: Path = cfg.paths.staging_mentions,
    disambig_log_path: Path = cfg.paths.disambig_log,
    clusters_path: Path = cfg.paths.canonical_root / "new_entity_clusters.json",
    alignment_path: Path = cfg.paths.canonical_root / "external_alignment.json",
    skills_path: Path = cfg.paths.output_skills,
    profiles_path: Path = cfg.paths.output_profiles,
    quality_report_path: Path = _QUALITY_REPORT_PATH,
) -> dict[str, Any]:
    """生成答辩案例报告。"""
    missing_inputs: list[str] = []
    mentions = _try_read_jsonl(mentions_path, missing_inputs)
    disambig_log = _try_read_jsonl(disambig_log_path, missing_inputs)
    clusters = _try_read_json(clusters_path, missing_inputs)
    alignment = _try_read_json(alignment_path, missing_inputs)
    skills_payload = _try_read_json(skills_path, missing_inputs)
    profiles = _try_read_json(profiles_path, missing_inputs)

    if quality_report_path.exists():
        quality_report = read_json(quality_report_path)
    else:
        quality_report = validate_entity_quality(output_path=quality_report_path)

    report = {
        "generated_at": date.today().isoformat(),
        "inputs": {
            "mentions": str(mentions_path),
            "disambig_log": str(disambig_log_path),
            "clusters": str(clusters_path),
            "alignment": str(alignment_path),
            "skills": str(skills_path),
            "profiles": str(profiles_path),
            "quality_report": str(quality_report_path),
        },
        "missing_inputs": missing_inputs,
        "summary": {
            "limit_per_section": limit_per_section,
            "mentions_loaded": len(mentions),
            "disambig_records_loaded": len(disambig_log),
            "new_entity_clusters": clusters.get("total_clusters") if isinstance(clusters, dict) else None,
            "alignment_nodes": len(alignment.get("alignment", {})) if isinstance(alignment, dict) else 0,
            "quality_errors": quality_report.get("summary", {}).get("error_count", 0) if quality_report else 0,
            "quality_warnings": quality_report.get("summary", {}).get("warning_count", 0) if quality_report else 0,
        },
        "rule_match_examples": _rule_match_examples(mentions, limit_per_section),
        "auto_confirmed_alias_examples": _auto_confirmed_alias_examples(mentions, limit_per_section),
        "needs_review_examples": _needs_review_examples(disambig_log, limit_per_section),
        "extracted_entity_examples": _flatten_extracted_entities(skills_payload, limit_per_section),
        "suspicious_alias_examples": _quality_samples(quality_report, "alias_*", limit_per_section),
        "alias_conflict_examples": _quality_samples(quality_report, "alias_conflict", limit_per_section),
        "external_alignment_examples": _external_alignment_examples(skills_payload, limit_per_section),
        "profile_mapping_examples": _profile_mapping_examples(profiles, limit_per_section),
    }

    if output_path is not None:
        ensure_dir(output_path.parent)
        write_json(output_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit-per-section", type=int, default=8, help="每类案例最多输出多少条")
    parser.add_argument("--output-path", type=Path, default=_DEFAULT_OUTPUT_PATH, help="案例报告 JSON 输出路径")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_case_report(limit_per_section=args.limit_per_section, output_path=args.output_path)
    print(
        "答辩案例报告完成："
        f"rule={len(report['rule_match_examples'])} "
        f"auto_alias={len(report['auto_confirmed_alias_examples'])} "
        f"needs_review={len(report['needs_review_examples'])} "
        f"extracted={len(report['extracted_entity_examples'])} "
        f"missing_inputs={len(report['missing_inputs'])}"
    )
    print(f"报告已写入：{args.output_path}")


if __name__ == "__main__":
    main()

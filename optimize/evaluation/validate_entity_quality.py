"""校验 optimize/output/ 中实体、别名、外部对齐和 profile 映射质量。

该脚本用于答辩前质量门禁：硬错误会阻断合并，软风险只进入 warning。

运行方式：
    python -m optimize.evaluation.validate_entity_quality
    python -m optimize.evaluation.validate_entity_quality --fail-on-warning
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import json
import re
from pathlib import Path
from typing import Any

from optimize.config import cfg
from optimize.utils.file_utils import ensure_dir, write_json


_DEFAULT_OUTPUT_PATH = cfg.paths.canonical_root / "entity_quality_report.json"

_CATEGORY_PREFIX = {
    "skill": "skill_",
    "tool": "tool_",
    "knowledge": "knowledge_",
    "language": "language_",
    "soft_skill": "soft_",
    "constraint": "constraint_",
    "project": "project_",
    "interest": "interest_",
}

_REQUIRED_NODE_FIELDS = ("id", "name", "aliases", "description", "origin")
_REQUIRED_PROFILE_FIELDS = (
    "profile_id",
    "source_type",
    "source_id",
    "source_url",
    "source_title",
    "snapshot_date",
    "evidence_snippet",
    "mapped_node_ids",
)
_ORG_OR_RECRUITING_KEYWORDS = (
    "有限公司",
    "交易所",
    "中心",
    "市场",
    "招聘",
    "岗位职责",
    "任职要求",
    "公司信息",
)
_NOISE_KEYWORDS = (
    "负责",
    "参与",
    "熟悉",
    "掌握",
    "经验",
    "能力",
    "项目",
    "系统",
)
_SENTENCE_PUNCTUATION = ("，", "。", "；", "、", ";")
_DISPLAY_REF_FIELDS = ("label", "tool_name", "name", "title")


def _read_json_file(path: Path, errors: list[dict[str, Any]], label: str) -> Any:
    """读取 JSON；失败时记录硬错误并返回 None。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(_issue("file_missing", f"{label} 文件不存在", str(path)))
    except json.JSONDecodeError as exc:
        errors.append(_issue("json_invalid", f"{label} 不是合法 JSON：{exc}", str(path)))
    return None


def _read_graph_node_ids(path: Path) -> set[str]:
    """读取当前已编译图谱节点 ID，用于校验 alias/profile 指向。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()
    if not isinstance(payload, list):
        return set()
    return {str(item.get("id", "")).strip() for item in payload if isinstance(item, dict) and item.get("id")}


def _issue(code: str, message: str, context: str, sample: Any | None = None) -> dict[str, Any]:
    item = {"code": code, "message": message, "context": context}
    if sample is not None:
        item["sample"] = sample
    return item


def _norm_alias(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _is_suspicious_alias(alias: str) -> tuple[bool, str]:
    """返回 alias 是否像噪声，以及最主要的原因。"""
    text = alias.strip()
    if len(text) > 24:
        return True, "alias_too_long"
    if any(keyword in text for keyword in _ORG_OR_RECRUITING_KEYWORDS):
        return True, "alias_looks_like_org_or_recruiting_text"
    if sum(text.count(mark) for mark in _SENTENCE_PUNCTUATION) >= 2:
        return True, "alias_looks_like_sentence"
    if len(text) > 8 and any(keyword in text for keyword in _NOISE_KEYWORDS):
        return True, "alias_contains_noise_keyword"
    return False, ""


def _has_cluster_id(source_note: str) -> bool:
    return re.search(r"\bcluster_id\s*=", source_note) is not None


def _flatten_skill_nodes(
    skills: Any,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    """把按类别分组的 skills_enriched.json 展开成节点列表。"""
    if not isinstance(skills, dict):
        errors.append(_issue("skills_top_level_invalid", "skills_enriched.json 顶层必须是对象", "skills_enriched.json"))
        return [], set()

    nodes: list[dict[str, Any]] = []
    known_ids: set[str] = set()
    seen_ids: dict[str, str] = {}

    for category, items in skills.items():
        context = f"skills_enriched.json.{category}"
        if category not in _CATEGORY_PREFIX:
            errors.append(_issue("unsupported_category", f"不支持的实体类别：{category}", context))
            continue
        if not isinstance(items, list):
            errors.append(_issue("category_not_list", f"{category} 必须是列表", context))
            continue

        expected_prefix = _CATEGORY_PREFIX[category]
        for index, node in enumerate(items):
            node_context = f"{context}[{index}]"
            if not isinstance(node, dict):
                errors.append(_issue("node_not_object", "实体节点必须是对象", node_context))
                continue

            missing = [field for field in _REQUIRED_NODE_FIELDS if field not in node or node.get(field) in ("", None)]
            if missing:
                errors.append(_issue("node_missing_required_fields", f"实体缺少必要字段：{missing}", node_context, node.get("id")))

            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            if node_id in seen_ids:
                errors.append(_issue("duplicate_entity_id", f"entity_id 重复：{node_id}", node_context, seen_ids[node_id]))
            else:
                seen_ids[node_id] = node_context
                known_ids.add(node_id)

            if not node_id.startswith(expected_prefix):
                errors.append(
                    _issue(
                        "entity_prefix_mismatch",
                        f"{category} 类实体 ID 应以 {expected_prefix} 开头",
                        node_context,
                        node_id,
                    )
                )

            aliases = node.get("aliases")
            if not isinstance(aliases, list):
                errors.append(_issue("aliases_not_list", "aliases 必须是列表", node_context, node_id))
            else:
                for alias in aliases:
                    if not isinstance(alias, str) or not alias.strip():
                        errors.append(_issue("alias_invalid", "alias 必须是非空字符串", node_context, node_id))

            if node.get("origin") == "extracted":
                source_note = str(node.get("source_note", "")).strip()
                if not source_note or not _has_cluster_id(source_note):
                    errors.append(_issue("extracted_missing_cluster_trace", "extracted 实体必须能追溯到 cluster_id", node_context, node_id))
                if node.get("review_status") != "needs_review":
                    errors.append(_issue("extracted_review_status_invalid", "extracted 实体必须默认 review_status=needs_review", node_context, node_id))
                if isinstance(aliases, list):
                    valid_aliases = [a for a in aliases if isinstance(a, str) and a.strip()]
                    suspicious_count = sum(1 for alias in valid_aliases if _is_suspicious_alias(alias)[0])
                    if not valid_aliases or suspicious_count == len(valid_aliases):
                        warnings.append(_issue("extracted_aliases_risky", "extracted 实体没有可靠 alias", node_context, node_id))

            nodes.append({"category": category, "index": index, "node": node})

    return nodes, known_ids


def _validate_aliases(
    nodes: list[dict[str, Any]],
    aliases_payload: Any,
    known_ids: set[str],
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    """检查 alias 指向、冲突与噪声。"""
    alias_to_entities: dict[str, set[str]] = defaultdict(set)
    alias_examples: dict[str, str] = {}
    alias_sources: dict[str, list[str]] = defaultdict(list)

    def register(alias: Any, entity_id: str, source: str) -> None:
        if not isinstance(alias, str) or not alias.strip():
            return
        norm = _norm_alias(alias)
        alias_to_entities[norm].add(entity_id)
        alias_examples.setdefault(norm, alias.strip())
        alias_sources[norm].append(source)
        suspicious, reason = _is_suspicious_alias(alias)
        if suspicious:
            warnings.append(_issue(reason, f"alias 可能包含噪声：{alias.strip()}", source, entity_id))

    for item in nodes:
        node = item["node"]
        entity_id = str(node.get("id", "")).strip()
        source = f"skills_enriched.json.{item['category']}[{item['index']}]"
        register(node.get("name"), entity_id, source + ".name")
        for alias in node.get("aliases", []) if isinstance(node.get("aliases"), list) else []:
            register(alias, entity_id, source + ".aliases")

    if not isinstance(aliases_payload, dict):
        errors.append(_issue("aliases_top_level_invalid", "aliases_enriched.json 顶层必须是对象", "aliases_enriched.json"))
        extra_aliases = {}
    else:
        extra_aliases = aliases_payload.get("extra_aliases", {})
        if not isinstance(extra_aliases, dict):
            errors.append(_issue("extra_aliases_invalid", "extra_aliases 必须是对象", "aliases_enriched.json.extra_aliases"))
            extra_aliases = {}

    for entity_id, aliases in extra_aliases.items():
        entity_id = str(entity_id).strip()
        context = f"aliases_enriched.json.extra_aliases.{entity_id}"
        if entity_id not in known_ids:
            errors.append(_issue("alias_unknown_entity", "extra_aliases 指向不存在的实体 ID", context, entity_id))
        if not isinstance(aliases, list):
            errors.append(_issue("extra_aliases_value_invalid", "extra_aliases 的值必须是列表", context, entity_id))
            continue
        for alias in aliases:
            register(alias, entity_id, context)

    conflicts = []
    for norm, entity_ids in sorted(alias_to_entities.items()):
        if len(entity_ids) <= 1:
            continue
        conflict = {
            "alias": alias_examples[norm],
            "normalized_alias": norm,
            "entity_ids": sorted(entity_ids),
            "sources": alias_sources[norm][:8],
        }
        conflicts.append(conflict)
        errors.append(_issue("alias_conflict", "同一 alias 映射到多个实体", "aliases", conflict))

    return {
        "unique_aliases": len(alias_to_entities),
        "alias_conflicts": conflicts,
    }


def _validate_external_refs(
    nodes: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, int]:
    total_refs = 0
    low_similarity = 0

    for item in nodes:
        node = item["node"]
        refs = node.get("external_refs")
        if refs is None:
            continue
        context = f"skills_enriched.json.{item['category']}[{item['index']}].external_refs"
        if not isinstance(refs, list):
            errors.append(_issue("external_refs_not_list", "external_refs 必须是列表", context, node.get("id")))
            continue
        for ref_index, ref in enumerate(refs):
            ref_context = f"{context}[{ref_index}]"
            total_refs += 1
            if not isinstance(ref, dict):
                errors.append(_issue("external_ref_not_object", "external_ref 必须是对象", ref_context, node.get("id")))
                continue
            if not ref.get("source"):
                errors.append(_issue("external_ref_missing_source", "external_ref 缺少 source", ref_context, node.get("id")))
            if not any(ref.get(field) for field in _DISPLAY_REF_FIELDS):
                errors.append(_issue("external_ref_missing_display_name", "external_ref 缺少可展示名称字段", ref_context, node.get("id")))
            similarity = ref.get("similarity")
            if not isinstance(similarity, (int, float)) or not (0 <= float(similarity) <= 1):
                errors.append(_issue("external_ref_similarity_invalid", "external_ref.similarity 必须是 0 到 1 的数字", ref_context, node.get("id")))
                continue
            if float(similarity) < 0.75:
                low_similarity += 1
                warnings.append(_issue("external_ref_low_similarity", "external_refs 相似度低于 0.75", ref_context, {"entity_id": node.get("id"), "similarity": similarity}))

    return {"total_refs": total_refs, "low_similarity_refs": low_similarity}


def _validate_profiles(
    profiles: Any,
    known_ids: set[str],
    errors: list[dict[str, Any]],
) -> dict[str, int]:
    if not isinstance(profiles, list):
        errors.append(_issue("profiles_top_level_invalid", "imported_profiles_new.json 顶层必须是列表", "imported_profiles_new.json"))
        return {"profiles": 0, "mapped_node_ids": 0}

    mapped_count = 0
    for index, profile in enumerate(profiles):
        context = f"imported_profiles_new.json[{index}]"
        if not isinstance(profile, dict):
            errors.append(_issue("profile_not_object", "profile 必须是对象", context))
            continue
        missing = [field for field in _REQUIRED_PROFILE_FIELDS if field not in profile or profile.get(field) in ("", None, [])]
        if missing:
            errors.append(_issue("profile_missing_required_fields", f"profile 缺少必要字段：{missing}", context, profile.get("profile_id")))
        mapped = profile.get("mapped_node_ids")
        if not isinstance(mapped, list):
            errors.append(_issue("profile_mapped_node_ids_invalid", "mapped_node_ids 必须是列表", context, profile.get("profile_id")))
            continue
        for node_id in mapped:
            mapped_count += 1
            if str(node_id) not in known_ids:
                errors.append(_issue("profile_unknown_mapped_node", "profile 映射了不存在的实体 ID", context, node_id))

    return {"profiles": len(profiles), "mapped_node_ids": mapped_count}


def validate_entity_quality(
    *,
    skills_path: Path = cfg.paths.output_skills,
    aliases_path: Path = cfg.paths.output_aliases,
    profiles_path: Path = cfg.paths.output_profiles,
    graph_nodes_path: Path = cfg.paths.seeds_nodes,
    output_path: Path | None = _DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """执行质量校验并返回报告。"""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    skills = _read_json_file(skills_path, errors, "skills_enriched.json")
    aliases_payload = _read_json_file(aliases_path, errors, "aliases_enriched.json")
    profiles = _read_json_file(profiles_path, errors, "imported_profiles_new.json")

    nodes, known_ids = _flatten_skill_nodes(skills, errors, warnings) if skills is not None else ([], set())
    graph_node_ids = _read_graph_node_ids(graph_nodes_path)
    resolvable_ids = known_ids | graph_node_ids
    alias_stats = _validate_aliases(nodes, aliases_payload, resolvable_ids, errors, warnings) if aliases_payload is not None else {"unique_aliases": 0, "alias_conflicts": []}
    external_stats = _validate_external_refs(nodes, errors, warnings)
    profile_stats = _validate_profiles(profiles, resolvable_ids, errors) if profiles is not None else {"profiles": 0, "mapped_node_ids": 0}

    categories = Counter(item["category"] for item in nodes)
    report = {
        "generated_at": date.today().isoformat(),
        "inputs": {
            "skills": str(skills_path),
            "aliases": str(aliases_path),
            "profiles": str(profiles_path),
            "graph_nodes": str(graph_nodes_path),
        },
        "summary": {
            "passed": len(errors) == 0,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "entity_count": len(nodes),
            "category_counts": dict(sorted(categories.items())),
            "unique_entity_ids": len(known_ids),
            "graph_node_ids": len(graph_node_ids),
            "unique_aliases": alias_stats["unique_aliases"],
            "alias_conflicts": len(alias_stats["alias_conflicts"]),
            "external_refs": external_stats["total_refs"],
            "low_similarity_external_refs": external_stats["low_similarity_refs"],
            "profiles": profile_stats["profiles"],
            "profile_mapped_node_ids": profile_stats["mapped_node_ids"],
        },
        "checks": {
            "entity_id_unique": not any(item["code"] == "duplicate_entity_id" for item in errors),
            "entity_prefix_valid": not any(item["code"] == "entity_prefix_mismatch" for item in errors),
            "alias_conflict_free": not alias_stats["alias_conflicts"],
            "extracted_traceable": not any(item["code"] == "extracted_missing_cluster_trace" for item in errors),
            "extracted_needs_review": not any(item["code"] == "extracted_review_status_invalid" for item in errors),
            "external_refs_valid": not any(item["code"].startswith("external_ref_") and item["code"] != "external_ref_low_similarity" for item in errors),
            "profiles_resolve_mapped_nodes": not any(item["code"] == "profile_unknown_mapped_node" for item in errors),
        },
        "errors": errors,
        "warnings": warnings,
        "samples": {
            "alias_conflicts": alias_stats["alias_conflicts"][:20],
            "suspicious_aliases": [item for item in warnings if item["code"].startswith("alias_")][:20],
            "low_similarity_external_refs": [item for item in warnings if item["code"] == "external_ref_low_similarity"][:20],
        },
    }

    if output_path is not None:
        ensure_dir(output_path.parent)
        write_json(output_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-path", type=Path, default=_DEFAULT_OUTPUT_PATH, help="质量报告 JSON 输出路径")
    parser.add_argument("--fail-on-warning", action="store_true", help="存在 warning 时也返回失败退出码")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = validate_entity_quality(output_path=args.output_path)
    summary = report["summary"]
    print(
        "实体质量检测完成："
        f"errors={summary['error_count']} warnings={summary['warning_count']} "
        f"entities={summary['entity_count']} aliases={summary['unique_aliases']} "
        f"profiles={summary['profiles']}"
    )
    print(f"报告已写入：{args.output_path}")
    if summary["error_count"] > 0 or (args.fail_on_warning and summary["warning_count"] > 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Step 8 输出生成：将管道成果写入 optimize/output/ 供手动审阅后合并。

产出三份文件（均为原项目 data/sources/ 对应文件的增量副本）：

  skills_enriched.json
    - 包含所有原有节点（完整保留）
    - 新增：从新实体簇中提取的高质量 evidence 节点
    - 新增：O*NET external_refs 字段（现有节点获得外部标准锚点）
    - 修改：origin 由 'curated' 更新为 'canonical'（有溯源）

  aliases_enriched.json
    - 包含所有原有 extra_aliases
    - 新增：消歧流程中发现的新别名（auto_confirmed embedding_high 命中）

  imported_profiles_new.json
    - 新增 FairCV 和 51job/智联数据集的 profile 条目

合并方式（供 ztt 执行）：
  Step 1：对比审阅 optimize/output/ 与 data/sources/ 中的差异
  Step 2：确认无误后将 skills_enriched.json 替换 data/sources/skills.json
  Step 3：将 aliases_enriched.json 替换 data/sources/aliases.json
  Step 4：将 imported_profiles_new.json 的新条目追加到 data/sources/imported_profiles.json
  Step 5：运行 python scripts/build_graph.py 重新编译图谱
  Step 6：运行 python scripts/validate_graph.py 验证无报错

运行方式
--------
    python -m optimize.output.generate_output
    python -m optimize.output.generate_output --min-cluster-size 4
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path
from typing import Any

from optimize.config import cfg, OPTIMIZE_ROOT
from optimize.utils.file_utils import ensure_dir, read_json, read_jsonl, write_json
from optimize.utils.logging_utils import get_pipeline_logger

logger = get_pipeline_logger("output.generate_output")

# 自动新建节点时接受的实体类型（tool/skill 最可靠；knowledge 需更谨慎）
_AUTO_ACCEPT_TYPES = {"tool", "skill", "language"}

# 词面长度过滤：过短或过长的词面不适合作为节点名
_MIN_SURFACE_LEN = 2
_MAX_SURFACE_LEN = 20

# 过滤词面中包含的噪声词
_NOISE_PATTERNS = re.compile(r"(经验|能力|水平|习惯|负责|参与|了解|具备|掌握|开发|设计|实现|维护|管理|优化|提升|系统|项目|平台|方案|架构|模块|功能|代码|业务|服务|基础|高级)")


def _is_clean_surface(surface: str) -> bool:
    """判断词面是否适合作为新节点的 name/alias。"""
    if not (_MIN_SURFACE_LEN <= len(surface) <= _MAX_SURFACE_LEN):
        return False
    # 纯中文技术词可以接受，但不接受长描述性短语
    if _NOISE_PATTERNS.search(surface) and len(surface) > 6:
        return False
    return True


def _surface_to_node_id(surface: str, node_type: str) -> str:
    """将词面转换为 ASCII snake_case node_id。

    对中文词面，尝试只保留其中的 ASCII 子串；若无 ASCII 子串则跳过（不生成节点）。
    """
    prefix = cfg.entity_types.id_prefix.get(node_type, node_type + "_")
    s = surface.lower().strip()
    # 去除"框架"、"引擎"等后缀助词
    s = re.sub(r"(框架|引擎|库|平台|工具|开发|环境|系统)$", "", s).strip()
    # 只保留 ASCII 字母数字（去掉点号等）
    ascii_only = re.sub(r"[^a-z0-9 ]", " ", s).strip()
    ascii_only = re.sub(r"\s+", "_", ascii_only).strip("_")
    # 若无有效 ASCII 片段则返回 None 信号
    if not ascii_only or ascii_only == "_":
        return ""
    return prefix + ascii_only


def generate_skills(min_cluster_size: int = 3) -> dict[str, Any]:
    """生成 skills_enriched.json。"""
    original  = read_json(cfg.paths.sources_skills)
    clusters  = read_json(cfg.paths.canonical_root / "new_entity_clusters.json")
    alignment = read_json(cfg.paths.canonical_root / "external_alignment.json")
    ext_refs  = alignment.get("alignment", {})

    # 建立现有节点 ID 集合（用于去重）
    existing_ids:   set[str] = set()
    existing_names: set[str] = set()

    # 深拷贝原始数据，并逐节点注入 external_refs + 更新 origin
    enriched: dict[str, list[dict[str, Any]]] = {}
    for category, nodes in original.items():
        enriched_nodes = []
        for node in nodes:
            nid = node["id"]
            existing_ids.add(nid)
            existing_names.add(node["name"].lower())

            # 注入 O*NET external_refs
            refs = ext_refs.get(nid, [])
            enriched_node = dict(node)
            if refs:
                enriched_node["external_refs"] = refs
            # 更新 origin 标记（来自管道，有溯源）
            if node.get("origin") == "curated":
                enriched_node["origin"] = "canonical"

            enriched_nodes.append(enriched_node)
        enriched[category] = enriched_nodes

    # 从新实体簇中提取高质量新节点
    new_nodes_added = 0
    for cluster in clusters.get("clusters", []):
        if cluster["size"] < min_cluster_size:
            continue
        stype = cluster.get("suggested_type", "")
        if stype not in _AUTO_ACCEPT_TYPES:
            continue

        # 选最短、最干净的词面作为 canonical name（英文优先，其次中文）
        surfaces = cluster["surfaces"]
        clean = [s for s in surfaces if _is_clean_surface(s)]
        if not clean:
            continue

        # 英文词面优先
        en_surfs = [s for s in clean if re.match(r"^[a-z0-9][a-z0-9 .+#\-]*$", s, re.I)]
        name = en_surfs[0] if en_surfs else clean[0]
        canonical_name = name.title() if re.match(r"^[a-z]+$", name) else name

        nid = _surface_to_node_id(name, stype)
        # 跳过无法生成合法 ASCII ID 的词面
        if not nid:
            continue
        # 去重
        if nid in existing_ids or canonical_name.lower() in existing_names:
            continue

        aliases = list({s.lower() for s in clean if s.lower() != canonical_name.lower()})[:6]
        new_node = {
            "id":          nid,
            "name":        canonical_name,
            "aliases":     aliases,
            "description": f"{canonical_name} — 由语料挖掘新增（原始词面：{', '.join(surfaces[:3])}）",
            "origin":      "extracted",
            "source_note": f"cluster_id={cluster['cluster_id']} size={cluster['size']}",
            "review_status": "needs_review",
        }

        # 归入对应 category
        category = stype if stype in enriched else "skill"
        enriched.setdefault(category, []).append(new_node)
        existing_ids.add(nid)
        existing_names.add(canonical_name.lower())
        new_nodes_added += 1
        logger.info("新节点：%s (%s)  aliases=%s", nid, stype, aliases[:3])

    ensure_dir(cfg.paths.output_dir)
    write_json(cfg.paths.output_skills, enriched)
    logger.info("skills_enriched.json 已写入：原有类别 %d，新增节点 %d", len(enriched), new_nodes_added)
    return {"categories": len(enriched), "new_nodes": new_nodes_added}


def generate_aliases() -> dict[str, Any]:
    """生成 aliases_enriched.json：追加消歧发现的新别名。"""
    original = read_json(cfg.paths.sources_aliases)
    extra    = dict(original.get("extra_aliases", {}))

    # 从 auto_confirmed (embedding_high) 的 mention 中收集新别名
    mentions = read_jsonl(cfg.paths.staging_mentions)
    new_alias_count = 0
    for m in mentions:
        if m.get("link_method") not in ("embedding_high",):
            continue
        eid     = m.get("linked_entity_id", "")
        surface = m.get("surface", "").strip()
        if not eid or not surface or len(surface) < 2:
            continue
        norm = surface.lower()
        current = extra.get(eid, [])
        if norm not in [a.lower() for a in current]:
            extra.setdefault(eid, []).append(surface)
            new_alias_count += 1

    result = {"extra_aliases": extra}
    write_json(cfg.paths.output_aliases, result)
    logger.info("aliases_enriched.json 已写入：新别名 %d 条", new_alias_count)
    return {"new_aliases": new_alias_count}


def generate_profiles() -> dict[str, Any]:
    """生成 imported_profiles_new.json：FairCV 和 JD 数据集的 profile 条目。"""
    snapshot_date = date.today().isoformat()

    # 汇总 FairCV 覆盖的节点（从 mentions.jsonl 中统计 fairCV 来源的命中节点）
    mentions    = read_jsonl(cfg.paths.staging_mentions)
    jd_dir      = cfg.paths.raw_jd / "csv_import"
    fc_dir      = cfg.paths.raw_fairCV

    fc_node_ids: set[str] = set()
    jd_node_ids: set[str] = set()

    for m in mentions:
        eid = m.get("linked_entity_id", "")
        if not eid:
            continue
        if "fairCV" in m.get("doc_id", ""):
            fc_node_ids.add(eid)
        else:
            jd_node_ids.add(eid)

    profiles = [
        {
            "profile_id":      "fairCV_dataset_v1",
            "source_type":     "fairCV",
            "source_id":       "OhMyKing/FairCV",
            "source_url":      "https://huggingface.co/datasets/OhMyKing/FairCV",
            "source_title":    "OhMyKing/FairCV: Chinese Simulated Resume Dataset",
            "snapshot_date":   snapshot_date,
            "evidence_snippet": (
                "Simulated Chinese resumes covering backend, frontend, ML, NLP, CV, "
                "security, DevOps, Android, iOS and other CS roles. "
                "Contains skills, education, project experience and self-evaluation sections."
            ),
            "sample_job_titles": [
                "后端开发工程师", "前端开发工程师", "机器学习工程师",
                "NLP工程师", "计算机视觉工程师", "安全架构师",
            ],
            "mapped_node_ids": sorted(fc_node_ids)[:50],
            "profile_tags":    ["resume", "zh", "fairCV", "simulated"],
        },
        {
            "profile_id":      "jd_51job_zhilian_v1",
            "source_type":     "jd_crawler",
            "source_id":       "51job+zhilian_2018",
            "source_url":      "https://www.kaggle.com (Job Information for IT)",
            "source_title":    "51job & 智联招聘 CS Job Descriptions (2018)",
            "snapshot_date":   snapshot_date,
            "evidence_snippet": (
                "Computer science job postings from 51job and Zhaopin covering "
                "backend, frontend, data, AI/ML, DevOps, security, QA and embedded roles. "
                "Contains requirements, responsibilities and preferred qualifications."
            ),
            "sample_job_titles": [
                "Python后端工程师", "Java开发工程师", "机器学习工程师",
                "安全工程师", "测试开发工程师", "运维工程师",
            ],
            "mapped_node_ids": sorted(jd_node_ids)[:50],
            "profile_tags":    ["jd", "zh", "51job", "zhilian", "2018"],
        },
    ]

    write_json(cfg.paths.output_profiles, profiles)
    logger.info("imported_profiles_new.json 已写入：%d 个 profile", len(profiles))
    return {"profiles": len(profiles)}


def run(min_cluster_size: int = 3) -> dict[str, Any]:
    """执行 Step 8 全量输出生成。"""
    ensure_dir(cfg.paths.output_dir)
    skills_stats  = generate_skills(min_cluster_size)
    aliases_stats = generate_aliases()
    profile_stats = generate_profiles()

    stats = {**skills_stats, **aliases_stats, **profile_stats}
    logger.info("Step 8 完成：%s", stats)

    # 打印合并指引
    print()
    print("=" * 60)
    print("输出文件已生成到 optimize/output/：")
    for name in ("skills_enriched.json", "aliases_enriched.json", "imported_profiles_new.json"):
        p = cfg.paths.output_dir / name
        size_kb = p.stat().st_size // 1024 if p.exists() else 0
        print(f"  {name}  ({size_kb} KB)")
    print()
    print("合并步骤（确认无误后执行）：")
    print("  1. cp optimize/output/skills_enriched.json data/sources/skills.json")
    print("  2. cp optimize/output/aliases_enriched.json data/sources/aliases.json")
    print("  3. 手动将 imported_profiles_new.json 的条目追加到 data/sources/imported_profiles.json")
    print("  4. python scripts/build_graph.py")
    print("  5. python scripts/validate_graph.py")
    print("=" * 60)
    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-cluster-size", type=int, default=3, help="新节点最小簇大小（默认 3）")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(min_cluster_size=args.min_cluster_size)

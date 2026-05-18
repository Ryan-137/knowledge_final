"""第二级消歧：多语言嵌入相似度 + DBSCAN 新实体聚类。

对第一级未命中的 llm_candidate mention：
  - 为每个唯一 normalized 词面计算嵌入向量
  - 与现有 canonical entity 嵌入做余弦相似度检索
  - cosine ≥ auto_link_threshold（0.88）→ auto_confirmed，method='embedding_high'
  - review_threshold ≤ cosine < auto_link_threshold → needs_review
  - cosine < review_threshold → 收集为 orphan，做 DBSCAN 聚类

DBSCAN 聚类：
  - 每个簇代表一个潜在的新 evidence 节点
  - 输出到 data/canonical/new_entity_clusters.json
  - 由人工审阅后决定是否新建 entity（Step 8 写入 skills.json）

同时写入 data/canonical/disambiguation_log.jsonl（审计链）。

运行方式
--------
    python -m optimize.disambiguation.embedding_disambiguate
    python -m optimize.disambiguation.embedding_disambiguate --batch-size 64
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from optimize.config import cfg
from optimize.utils.file_utils import (
    append_jsonl_batch,
    ensure_dir,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from optimize.utils.hash_utils import file_sha256
from optimize.utils.logging_utils import get_pipeline_logger

logger = get_pipeline_logger("disambiguation.embedding_disambiguate")

_LOG_PATH        = cfg.paths.disambig_log
_CLUSTERS_PATH   = cfg.paths.canonical_root / "new_entity_clusters.json"
_EMB_CACHE_PATH  = cfg.paths.embedding_cache / "entity_embeddings.npy"
_EMB_IDS_PATH    = cfg.paths.embedding_cache / "entity_ids.json"
_EMB_MANIFEST_PATH = cfg.paths.embedding_cache / "entity_embeddings_manifest.json"


def _build_entity_corpus(alias_dict: dict[str, list[str]], nodes: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """为每个 evidence 节点构建嵌入语料：'name alias1 alias2...' 拼接。

    Returns:
        (entity_ids, entity_texts) 两个等长列表。
    """
    # evidence 层节点 ID 集合
    evidence_ids = {n["id"] for n in nodes if n.get("layer") == "evidence"}
    entity_ids:   list[str] = []
    entity_texts: list[str] = []
    for eid in sorted(evidence_ids):
        node = next((n for n in nodes if n["id"] == eid), None)
        if node is None:
            continue
        aliases = alias_dict.get(eid, [])
        text = node["name"] + " " + " ".join(aliases[:5])
        entity_ids.append(eid)
        entity_texts.append(text.strip())
    return entity_ids, entity_texts


def _build_embedding_manifest() -> dict[str, str]:
    """记录实体嵌入缓存依赖的输入指纹。"""
    return {
        "model": cfg.disambig.embedding_model,
        "nodes_sha256": file_sha256(cfg.paths.seeds_nodes),
        "aliases_sha256": file_sha256(cfg.paths.dict_skill_aliases),
    }


def _load_or_compute_entity_embeddings(
    model: Any,
    alias_dict: dict[str, list[str]],
    nodes: list[dict[str, Any]],
    batch_size: int,
) -> tuple[list[str], np.ndarray]:
    """加载缓存的实体嵌入，或重新计算并缓存。"""
    ensure_dir(cfg.paths.embedding_cache)
    expected_manifest = _build_embedding_manifest()

    if _EMB_CACHE_PATH.exists() and _EMB_IDS_PATH.exists() and _EMB_MANIFEST_PATH.exists():
        entity_ids = read_json(_EMB_IDS_PATH)
        embeddings = np.load(str(_EMB_CACHE_PATH))
        manifest = read_json(_EMB_MANIFEST_PATH)
        if manifest == expected_manifest and len(entity_ids) == int(embeddings.shape[0]):
            logger.info("加载嵌入缓存：%d 个实体", len(entity_ids))
            return entity_ids, embeddings
        logger.info("实体嵌入缓存已过期，重新计算")

    logger.info("首次计算实体嵌入（共 %d 个 evidence 节点）…", sum(1 for n in nodes if n.get("layer") == "evidence"))
    entity_ids, entity_texts = _build_entity_corpus(alias_dict, nodes)
    embeddings = model.encode(
        entity_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(str(_EMB_CACHE_PATH), embeddings)
    write_json(_EMB_IDS_PATH, entity_ids)
    write_json(_EMB_MANIFEST_PATH, expected_manifest)
    logger.info("实体嵌入已缓存：%s", _EMB_CACHE_PATH)
    return entity_ids, embeddings


def _cosine_top_k(query_emb: np.ndarray, corpus_embs: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    """计算 query 与 corpus 中每条的余弦相似度，返回 top-k (index, score)。

    corpus_embs 需已 L2 归一化（model.encode 时 normalize_embeddings=True）。
    """
    scores = corpus_embs @ query_emb  # (N,)
    top_k_idx = np.argpartition(scores, -k)[-k:]
    top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
    return [(int(i), float(scores[i])) for i in top_k_idx]


def run(batch_size: int = 64) -> dict[str, Any]:
    """执行嵌入消歧主流程。"""
    ensure_dir(cfg.paths.canonical_root)

    # 加载资源
    alias_dict = read_json(cfg.paths.dict_skill_aliases)
    nodes      = read_json(cfg.paths.seeds_nodes)
    all_mentions: list[dict[str, Any]] = read_jsonl(cfg.paths.staging_mentions)

    # 收集待消歧的唯一词面（仍为 llm_candidate）
    candidates: dict[str, dict[str, Any]] = {}   # normalized → mention
    for m in all_mentions:
        if m.get("status") == "llm_candidate":
            norm = m.get("normalized", "")
            if norm and norm not in candidates:
                candidates[norm] = m

    if not candidates:
        logger.info("无待消歧的 llm_candidate，跳过嵌入消歧")
        return {"auto_confirmed": 0, "needs_review": 0, "orphans": 0}

    logger.info("待嵌入消歧词面数：%d", len(candidates))

    # 加载嵌入模型
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
    except ImportError:
        logger.error("sentence-transformers 未安装，请运行：pip install sentence-transformers")
        return {}

    model = SentenceTransformer(cfg.disambig.embedding_model)
    entity_ids, entity_embs = _load_or_compute_entity_embeddings(
        model, alias_dict, nodes, batch_size
    )

    # 对所有候选词面计算嵌入
    surfaces = list(candidates.keys())
    logger.info("计算 %d 个候选词面的嵌入…", len(surfaces))
    surface_embs: np.ndarray = model.encode(
        surfaces,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # 分层消歧
    auto_confirmed: list[dict[str, Any]] = []
    needs_review:   list[dict[str, Any]] = []
    orphan_surfaces: list[str] = []
    orphan_embs:     list[np.ndarray] = []

    log_records: list[dict[str, Any]] = []

    for i, surface in enumerate(surfaces):
        emb   = surface_embs[i]
        top_k = _cosine_top_k(emb, entity_embs, k=cfg.disambig.top_k_candidates)
        best_idx, best_score = top_k[0]
        best_eid = entity_ids[best_idx]

        candidates_list = [
            {"entity_id": entity_ids[j], "score": round(s, 4)}
            for j, s in top_k
        ]

        if best_score >= cfg.disambig.auto_link_threshold:
            status = "auto_confirmed"
            auto_confirmed.append({"surface": surface, "entity_id": best_eid, "score": best_score})
        elif best_score >= cfg.disambig.review_threshold:
            status = "needs_review"
            needs_review.append({"surface": surface, "entity_id": best_eid, "score": best_score,
                                  "candidates": candidates_list})
        else:
            status = "orphan"
            orphan_surfaces.append(surface)
            orphan_embs.append(emb)

        log_records.append({
            "surface":      surface,
            "status":       status,
            "best_entity":  best_eid,
            "best_score":   round(best_score, 4),
            "candidates":   candidates_list,
            "entity_type_hint": candidates[surface].get("entity_type_hint", ""),
        })

    # 写消歧日志
    if _LOG_PATH.exists():
        _LOG_PATH.unlink()
    append_jsonl_batch(_LOG_PATH, log_records)
    logger.info("消歧日志已写入：%s", _LOG_PATH)

    # 更新 mention 状态（auto_confirmed → linked）
    confirmed_map = {item["surface"]: (item["entity_id"], item["score"]) for item in auto_confirmed}
    review_map    = {item["surface"]: (item["entity_id"], item["score"]) for item in needs_review}

    for m in all_mentions:
        if m.get("status") != "llm_candidate":
            continue
        norm = m.get("normalized", "")
        if norm in confirmed_map:
            eid, score = confirmed_map[norm]
            m["linked_entity_id"] = eid
            m["link_method"]      = "embedding_high"
            m["link_confidence"]  = round(score, 4)
            m["status"]           = "auto_confirmed"
        elif norm in review_map:
            eid, score = review_map[norm]
            m["linked_entity_id"] = eid
            m["link_confidence"]  = round(score, 4)
            m["status"]           = "needs_review"

    write_jsonl(cfg.paths.staging_mentions, all_mentions)

    # DBSCAN 聚类发现新实体
    new_clusters = _cluster_orphans(orphan_surfaces, orphan_embs, candidates)
    write_json(_CLUSTERS_PATH, {
        "total_clusters": len(new_clusters),
        "total_orphans":  len(orphan_surfaces),
        "clusters":       new_clusters,
    })
    logger.info("新实体聚类写入：%s（%d 簇，%d 个孤立词面）",
                _CLUSTERS_PATH, len(new_clusters), len(orphan_surfaces))

    stats = {
        "auto_confirmed": len(auto_confirmed),
        "needs_review":   len(needs_review),
        "orphans":        len(orphan_surfaces),
        "new_clusters":   len(new_clusters),
    }
    logger.info("嵌入消歧完成：%s", stats)
    return stats


def _cluster_orphans(
    surfaces: list[str],
    embs: list[np.ndarray],
    candidates_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """对无法链接到已有实体的孤立词面做 DBSCAN 聚类，每簇代表一个候选新节点。"""
    if len(embs) < cfg.disambig.dbscan_min_samples:
        return []

    from sklearn.cluster import DBSCAN  # type: ignore[import]

    emb_matrix = np.stack(embs)   # (N, D)
    # cosine 距离 = 1 - cosine_similarity（归一化后等于欧氏距离²/2）
    labels = DBSCAN(
        eps=cfg.disambig.dbscan_eps,
        min_samples=cfg.disambig.dbscan_min_samples,
        metric="cosine",
    ).fit_predict(emb_matrix)

    # 按簇汇总
    clusters: dict[int, list[str]] = defaultdict(list)
    for surface, label in zip(surfaces, labels):
        if label == -1:
            continue   # 噪声点
        clusters[int(label)].append(surface)

    result = []
    for label, cluster_surfaces in sorted(clusters.items(), key=lambda x: -len(x[1])):
        type_hints = [candidates_meta[s].get("entity_type_hint", "") for s in cluster_surfaces]
        most_common_type = max(set(type_hints), key=type_hints.count) if type_hints else ""
        result.append({
            "cluster_id":      label,
            "size":            len(cluster_surfaces),
            "surfaces":        cluster_surfaces,
            "suggested_type":  most_common_type,
            "suggested_id":    None,   # 人工填写
            "status":          "pending_review",
        })
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=64, help="嵌入计算批大小")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    stats = run(batch_size=args.batch_size)
    print("消歧完成：", stats)

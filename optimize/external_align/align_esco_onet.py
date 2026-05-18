"""外部标准对齐：将 evidence 层节点与 ESCO / O*NET 术语做嵌入检索对齐。

对每个 evidence 节点：
  1. 用其名称 + aliases 拼接查询文本
  2. 与 ESCO skills_index（首选标签 + 别名）做嵌入相似度检索
  3. 与 O*NET tech_skills_index（工具名称）做嵌入相似度检索
  4. 相似度 ≥ 对齐阈值（0.85）的结果写入节点的 external_refs 字段

产出文件
--------
data/canonical/external_alignment.json   对齐结果（每个节点的外部标准引用）

运行方式
--------
    python -m optimize.external_align.align_esco_onet
    python -m optimize.external_align.align_esco_onet --skip-esco  （仅对齐 O*NET）
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from optimize.config import cfg
from optimize.utils.file_utils import ensure_dir, read_json, write_json
from optimize.utils.logging_utils import get_pipeline_logger

logger = get_pipeline_logger("external_align.align_esco_onet")

_ALIGN_OUT_PATH = cfg.paths.canonical_root / "external_alignment.json"


def _load_esco_corpus() -> list[dict[str, Any]]:
    """加载 ESCO 技能索引，构建查询语料。"""
    idx_path = cfg.paths.raw_esco / "skills_index.json"
    if not idx_path.exists():
        logger.warning("ESCO 索引不存在（需手动下载），跳过 ESCO 对齐：%s", idx_path)
        return []
    data = read_json(idx_path)
    skills = []
    for s in data.get("skills", []):
        uri   = s.get("uri", "")
        pref  = s.get("preferred_label", "")
        alts  = s.get("alt_labels", [])
        if not uri or not pref:
            continue
        text = pref + " " + " ".join(alts[:3])
        skills.append({"uri": uri, "preferred_label": pref, "text": text.strip()})
    logger.info("ESCO 技能语料：%d 条", len(skills))
    return skills


def _load_onet_corpus() -> list[dict[str, Any]]:
    """加载 O*NET 技术工具索引。"""
    idx_path = cfg.paths.raw_onet / "tech_skills_index.json"
    if not idx_path.exists():
        logger.warning("O*NET 索引不存在，跳过 O*NET 对齐：%s", idx_path)
        return []
    data = read_json(idx_path)
    tools = []
    for t in data.get("tools", []):
        name = t.get("tool_name", "")
        cat  = t.get("onet_commodity_title", "")
        if not name:
            continue
        text = name + (" " + cat if cat else "")
        tools.append({"tool_name": name, "category": cat, "text": text.strip()})
    logger.info("O*NET 工具语料：%d 条", len(tools))
    return tools


def _encode_corpus(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)


def _top_k_matches(query_emb: np.ndarray, corpus_embs: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    scores = corpus_embs @ query_emb
    top_k  = np.argpartition(scores, -k)[-k:]
    top_k  = top_k[np.argsort(scores[top_k])[::-1]]
    return [(int(i), float(scores[i])) for i in top_k]


def run(
    skip_esco: bool = False,
    skip_onet: bool = False,
    batch_size: int = 64,
) -> dict[str, Any]:
    """执行外部标准对齐主流程。"""
    ensure_dir(cfg.paths.canonical_root)

    alias_dict = read_json(cfg.paths.dict_skill_aliases)
    nodes      = read_json(cfg.paths.seeds_nodes)
    evidence_nodes = [n for n in nodes if n.get("layer") == "evidence"]

    if not evidence_nodes:
        logger.error("未找到 evidence 层节点")
        return {}

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
    except ImportError:
        logger.error("sentence-transformers 未安装")
        return {}

    model = SentenceTransformer(cfg.disambig.embedding_model)

    # 计算节点查询嵌入（名称 + 前5个别名）
    node_texts = []
    node_ids   = []
    for n in evidence_nodes:
        aliases = alias_dict.get(n["id"], [])
        text    = n["name"] + " " + " ".join(aliases[:5])
        node_ids.append(n["id"])
        node_texts.append(text.strip())

    logger.info("计算 %d 个 evidence 节点的嵌入…", len(node_ids))
    node_embs: np.ndarray = model.encode(node_texts, batch_size=batch_size,
                                          normalize_embeddings=True, show_progress_bar=True)

    alignment: dict[str, list[dict[str, Any]]] = {nid: [] for nid in node_ids}
    threshold = cfg.external_align.esco_auto_align_threshold

    # ESCO 对齐
    if not skip_esco:
        esco_corpus = _load_esco_corpus()
        if esco_corpus:
            logger.info("计算 ESCO 嵌入（%d 条）…", len(esco_corpus))
            esco_texts = [c["text"] for c in esco_corpus]
            esco_embs  = _encode_corpus(model, esco_texts, batch_size)
            matched = 0
            for i, node_id in enumerate(node_ids):
                for idx, score in _top_k_matches(node_embs[i], esco_embs, k=3):
                    if score >= threshold:
                        alignment[node_id].append({
                            "source":      "esco",
                            "uri":         esco_corpus[idx]["uri"],
                            "label":       esco_corpus[idx]["preferred_label"],
                            "similarity":  round(score, 4),
                        })
                        matched += 1
            logger.info("ESCO 对齐完成：%d 个节点命中（共 %d 条引用）",
                        sum(1 for refs in alignment.values() if refs), matched)

    # O*NET 对齐
    if not skip_onet:
        onet_corpus = _load_onet_corpus()
        if onet_corpus:
            logger.info("计算 O*NET 嵌入（%d 条）…", len(onet_corpus))
            onet_texts = [c["text"] for c in onet_corpus]
            onet_embs  = _encode_corpus(model, onet_texts, batch_size)
            matched = 0
            for i, node_id in enumerate(node_ids):
                for idx, score in _top_k_matches(node_embs[i], onet_embs, k=3):
                    if score >= threshold:
                        alignment[node_id].append({
                            "source":      "onet",
                            "tool_name":   onet_corpus[idx]["tool_name"],
                            "category":    onet_corpus[idx]["category"],
                            "similarity":  round(score, 4),
                        })
                        matched += 1
            logger.info("O*NET 对齐完成：%d 个节点命中（共 %d 条引用）",
                        sum(1 for k, v in alignment.items() if any(r["source"] == "onet" for r in v)), matched)

    # 裁剪每个节点的引用数上限
    max_refs = cfg.external_align.max_external_refs
    for node_id in alignment:
        alignment[node_id] = sorted(alignment[node_id], key=lambda x: -x["similarity"])[:max_refs]

    # 统计
    nodes_with_refs = sum(1 for refs in alignment.values() if refs)
    total_refs      = sum(len(refs) for refs in alignment.values())

    output = {
        "pipeline_version": cfg.pipeline_version,
        "evidence_nodes":   len(evidence_nodes),
        "nodes_with_refs":  nodes_with_refs,
        "total_refs":       total_refs,
        "alignment":        alignment,
    }
    write_json(_ALIGN_OUT_PATH, output)
    logger.info("外部对齐结果写入：%s（%d/%d 节点有外部引用）",
                _ALIGN_OUT_PATH, nodes_with_refs, len(evidence_nodes))

    return {
        "nodes_with_refs": nodes_with_refs,
        "total_refs":      total_refs,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-esco", action="store_true")
    p.add_argument("--skip-onet", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    stats = run(skip_esco=args.skip_esco, skip_onet=args.skip_onet, batch_size=args.batch_size)
    print("外部对齐完成：", stats)

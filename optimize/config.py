"""Central configuration for the entity extraction pipeline.

路径分为两类：
  READ-ONLY  — 指向原项目 data/，只读，绝不覆盖。
  WRITE      — 指向 optimize/pipeline_data/，管道所有产物写在这里，与原项目隔离。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final


REPO_ROOT:     Final[Path] = Path(__file__).resolve().parents[1]
OPTIMIZE_ROOT: Final[Path] = Path(__file__).resolve().parent
PIPELINE_DATA: Final[Path] = OPTIMIZE_ROOT / "pipeline_data"
PIPELINE_VERSION: Final[str] = "v1.0.0"


@dataclass(frozen=True)
class PathConfig:

    # ── READ-ONLY：原项目文件，绝不写入 ──────────────────────────────────
    # 编译后图谱（运行时推理使用）
    seeds_nodes:           Path = REPO_ROOT / "data" / "seeds" / "nodes.json"
    seeds_edges:           Path = REPO_ROOT / "data" / "seeds" / "edges.json"
    # 已编译词典
    dict_skill_aliases:    Path = REPO_ROOT / "data" / "dictionaries" / "skill_aliases.json"
    dict_pref_patterns:    Path = REPO_ROOT / "data" / "dictionaries" / "preference_patterns.json"
    dict_parsing_patterns: Path = REPO_ROOT / "data" / "dictionaries" / "parsing_patterns.json"
    # Source 层（build_graph.py 的输入，Step 8 在 output/ 中生成副本供审阅后手动合并）
    sources_skills:        Path = REPO_ROOT / "data" / "sources" / "skills.json"
    sources_aliases:       Path = REPO_ROOT / "data" / "sources" / "aliases.json"
    sources_profiles:      Path = REPO_ROOT / "data" / "sources" / "imported_profiles.json"
    # 原项目 canonical 文件（已有 term_lexicon / entity_links / relation_triples）
    term_lexicon:          Path = REPO_ROOT / "data" / "canonical" / "term_lexicon.json"
    entity_links:          Path = REPO_ROOT / "data" / "canonical" / "entity_links.json"
    # 原 staging（O*NET / roadmap 已规范化文档，只读参考）
    orig_staging_docs:     Path = REPO_ROOT / "data" / "staging" / "normalized_documents.json"

    # ── WRITE：管道产物，全部写入 optimize/pipeline_data/ ────────────────
    # Raw zone
    raw_root:     Path = PIPELINE_DATA / "raw"
    raw_fairCV:   Path = PIPELINE_DATA / "raw" / "fairCV"
    raw_jd:       Path = PIPELINE_DATA / "raw" / "jd"
    raw_esco:     Path = PIPELINE_DATA / "raw" / "external" / "esco"
    raw_onet:     Path = PIPELINE_DATA / "raw" / "external" / "onet"
    # Staging zone
    staging_root:     Path = PIPELINE_DATA / "staging"
    staging_mentions: Path = PIPELINE_DATA / "staging" / "mentions.jsonl"
    # Canonical zone（管道生成的候选实体、消歧日志等）
    canonical_root:   Path = PIPELINE_DATA / "canonical"
    disambig_log:     Path = PIPELINE_DATA / "canonical" / "disambiguation_log.jsonl"
    golden_set:       Path = PIPELINE_DATA / "canonical" / "golden_set.jsonl"
    cooccurrence:     Path = PIPELINE_DATA / "canonical" / "entity_cooccurrence_candidates.jsonl"
    # 数据目录（不放入原项目 docs/）
    data_catalog:     Path = PIPELINE_DATA / "data_catalog.md"
    # 最终输出（enriched 副本，供手动审阅后合并到原项目）
    output_dir:       Path = OPTIMIZE_ROOT / "output"
    output_skills:    Path = OPTIMIZE_ROOT / "output" / "skills_enriched.json"
    output_aliases:   Path = OPTIMIZE_ROOT / "output" / "aliases_enriched.json"
    output_profiles:  Path = OPTIMIZE_ROOT / "output" / "imported_profiles_new.json"
    # Pipeline 内部缓存
    embedding_cache:  Path = OPTIMIZE_ROOT / ".cache" / "embeddings"
    log_dir:          Path = OPTIMIZE_ROOT / ".logs"
    abbr_expansion:   Path = OPTIMIZE_ROOT / "ner" / "abbr_expansion.json"


@dataclass(frozen=True)
class EntityTypeConfig:
    evidence_types: tuple[str, ...] = (
        "skill", "tool", "language", "knowledge",
        "project", "interest", "soft_skill", "constraint",
    )
    id_prefix: dict[str, str] = field(default_factory=lambda: {
        "skill":      "skill_",
        "tool":       "tool_",
        "language":   "language_",
        "knowledge":  "knowledge_",
        "project":    "project_",
        "interest":   "interest_",
        "soft_skill": "soft_skill_",
        "constraint": "constraint_",
    })


@dataclass(frozen=True)
class NerConfig:
    spacy_zh_model: str = "zh_core_web_sm"
    spacy_en_model: str = "en_core_web_sm"
    rule_confidence: float = 0.93
    distant_supervision_confidence: float = 0.85
    llm_high_confidence_threshold: float = 0.80
    llm_min_confidence: float = 0.50
    cross_validated_confidence: float = 0.97
    llm_window_tokens: int = 800
    llm_overlap_tokens: int = 64


@dataclass(frozen=True)
class DisambigConfig:
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    auto_link_threshold: float = 0.88
    review_threshold: float = 0.72
    dbscan_eps: float = 0.25
    dbscan_min_samples: int = 3
    top_k_candidates: int = 3


@dataclass(frozen=True)
class ExternalAlignConfig:
    esco_skills_url: str = "https://esco.ec.europa.eu/en/use-esco/download"
    # 中英跨语言对齐场景下 0.75 更合适（纯英-英对齐时可调回 0.85）
    esco_auto_align_threshold: float = 0.75
    max_external_refs: int = 3


@dataclass(frozen=True)
class LLMConfig:
    api_base: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.0
    max_retries: int = 2
    request_interval: float = 0.5
    api_key_env: str = "DEEPSEEK_API_KEY"


@dataclass(frozen=True)
class CollectionConfig:
    fairCV_dataset_name: str = "OhMyKing/FairCV"
    fairCV_max_samples: int = 2000
    jd_request_interval: float = 2.5
    jd_max_per_keyword: int = 30
    jd_target_keywords: tuple[str, ...] = (
        "后端开发", "前端开发", "全栈工程师", "数据工程师", "数据分析",
        "机器学习", "算法工程师", "NLP工程师", "计算机视觉",
        "DevOps", "SRE", "安全工程师", "测试开发", "嵌入式开发",
        "Android开发", "iOS开发", "移动端开发", "Go后端", "Java后端",
        "Python后端",
    )


@dataclass(frozen=True)
class Config:
    paths:          PathConfig          = field(default_factory=PathConfig)
    entity_types:   EntityTypeConfig    = field(default_factory=EntityTypeConfig)
    ner:            NerConfig           = field(default_factory=NerConfig)
    disambig:       DisambigConfig      = field(default_factory=DisambigConfig)
    external_align: ExternalAlignConfig = field(default_factory=ExternalAlignConfig)
    llm:            LLMConfig           = field(default_factory=LLMConfig)
    collection:     CollectionConfig    = field(default_factory=CollectionConfig)
    pipeline_version: str               = PIPELINE_VERSION


cfg: Final[Config] = Config()

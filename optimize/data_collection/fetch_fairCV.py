"""Download and normalise the OhMyKing/FairCV resume dataset.

Owner: ztt

Dataset
-------
OhMyKing/FairCV  https://huggingface.co/datasets/OhMyKing/FairCV
~719 k simulated Chinese resumes for bias-detection research.
Each record has two fields:

  metadata  dict  position, skill_level, recruitment_type, gender, age,
                  marriage, hukou, political, disability, company_size, timestamp
  content   str   Full resume in Markdown, sections:
                    ### 教育背景 (incl. course list)
                    ### 技术技能
                    ### 项目经历
                    ### 自我评价

Entity extraction uses ``content``.
``metadata.position`` maps directly to the role layer.
``metadata.skill_level`` provides relative proficiency context.

Network note for mainland China
--------------------------------
HuggingFace streaming is often slow or blocked.  Three acquisition routes
are provided (in recommended order):

  Route A — local Parquet file (fastest):
    1. Download one or more Parquet shards manually from the browser:
       https://hf-mirror.com/datasets/OhMyKing/FairCV/tree/main/data
    2. Run:  python -m optimize.data_collection.fetch_fairCV
               --parquet-path /path/to/train-00000-of-00015.parquet

  Route B — huggingface-cli with mirror (bulk download):
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    huggingface-cli download OhMyKing/FairCV --repo-type dataset
                             --local-dir data/raw/fairCV_hf_cache
    Then:
    python -m optimize.data_collection.fetch_fairCV
           --local-dir data/raw/fairCV_hf_cache

  Route C — streaming (requires stable HuggingFace access):
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    python -m optimize.data_collection.fetch_fairCV

Output
------
Each record is saved as ``data/raw/fairCV/<doc_id>.json``.

Usage
-----
    python -m optimize.data_collection.fetch_fairCV --parquet-path train-00000.parquet
    python -m optimize.data_collection.fetch_fairCV --local-dir data/raw/fairCV_hf_cache
    python -m optimize.data_collection.fetch_fairCV --max-samples 500
    python -m optimize.data_collection.fetch_fairCV --positions "后端开发工程师,数据工程师"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterator

from optimize.config import REPO_ROOT, cfg
from optimize.data_collection.catalog import CatalogEntry, catalog
from optimize.utils.file_utils import ensure_dir, save_raw_doc
from optimize.utils.hash_utils import dict_sha256
from optimize.utils.logging_utils import get_pipeline_logger

logger = get_pipeline_logger("data_collection.fetch_fairCV")

_SNAPSHOT_DATE = date.today().isoformat()
_SOURCE_URL = f"https://huggingface.co/datasets/{cfg.collection.fairCV_dataset_name}"
_LICENSE = "Academic research only. See https://huggingface.co/datasets/OhMyKing/FairCV"


def _build_raw_doc(record: dict[str, Any], idx: int) -> dict[str, Any]:
    """Wrap a single FairCV record in the project raw-zone schema."""
    metadata: dict[str, Any] = record.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}

    content: dict[str, Any] = {
        "position":         str(metadata.get("position", "")),
        "skill_level":      str(metadata.get("skill_level", "")),
        "recruitment_type": str(metadata.get("recruitment_type", "")),
        "gender":           str(metadata.get("gender", "")),
        "age":              str(metadata.get("age", "")),
        "disability":       str(metadata.get("disability", "")),
        "resume_text":      str(record.get("content", "")).strip(),
    }
    doc_id = f"fairCV_{idx:06d}"
    payload: dict[str, Any] = {
        "doc_id":        doc_id,
        "source_name":   "fairCV",
        "source_url":    _SOURCE_URL,
        "snapshot_time": f"{_SNAPSHOT_DATE}T00:00:00Z",
        "language":      "zh",
        "license_note":  _LICENSE,
        "doc_type":      "resume",
        "content":       content,
    }
    payload["sha256"] = dict_sha256(content)
    return payload


def _iter_parquet(parquet_path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a locally downloaded Parquet shard."""
    try:
        import pyarrow.parquet as pq  # type: ignore[import]
    except ImportError:
        logger.error("pyarrow not installed.  Run: pip install pyarrow")
        sys.exit(1)

    logger.info("Reading Parquet file: %s", parquet_path)
    table = pq.read_table(parquet_path)
    for batch in table.to_batches():
        batch_dict = batch.to_pydict()
        n = len(next(iter(batch_dict.values())))
        for i in range(n):
            yield {col: batch_dict[col][i] for col in batch_dict}


def _iter_json_streaming(json_path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a large JSON array file using streaming parse (no full load)."""
    try:
        import ijson  # type: ignore[import]
    except ImportError:
        logger.error("ijson not installed.  Run: pip install ijson")
        sys.exit(1)

    logger.info("Streaming JSON: %s (%.0f MB)", json_path, json_path.stat().st_size / 1_048_576)
    with json_path.open("rb") as f:
        yield from ijson.items(f, "item")


def _iter_local_dir(local_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a directory of files downloaded via huggingface-cli.

    Parquet files are preferred; falls back to JSON with streaming parse so
    large files (e.g. the 6 GB resumes.json) are not loaded into memory.
    """
    parquet_files = sorted(local_dir.rglob("*.parquet"))
    json_files = sorted(
        p for p in local_dir.rglob("*.json")
        if ".cache" not in p.parts   # skip huggingface metadata files
    )

    if parquet_files:
        logger.info("Found %d Parquet file(s) in %s", len(parquet_files), local_dir)
        for p in parquet_files:
            yield from _iter_parquet(p)
    elif json_files:
        logger.info("Found %d JSON file(s) in %s", len(json_files), local_dir)
        for p in json_files:
            # Skip tiny metadata files; only stream files that look like record arrays
            if p.stat().st_size < 1024:
                continue
            yield from _iter_json_streaming(p)
    else:
        raise FileNotFoundError(f"No .parquet or .json files found in {local_dir}")


def _iter_streaming() -> Iterator[dict[str, Any]]:
    """Yield records via HuggingFace streaming API.

    Requires stable network access to HuggingFace (or hf-mirror.com via
    HF_ENDPOINT environment variable).
    """
    import os

    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        logger.error("Package 'datasets' not installed.  Run: pip install datasets")
        sys.exit(1)

    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    logger.info("Streaming from %s via %s …", cfg.collection.fairCV_dataset_name, hf_endpoint)
    if hf_endpoint == "https://huggingface.co":
        logger.warning(
            "HuggingFace direct access may be slow in mainland China. "
            "Consider: $env:HF_ENDPOINT='https://hf-mirror.com'"
        )

    ds = load_dataset(cfg.collection.fairCV_dataset_name, split="train", streaming=True)
    yield from ds


def fetch(
    max_samples: int | None = None,
    max_per_position: int | None = None,
    positions: list[str] | None = None,
    parquet_path: Path | None = None,
    local_dir: Path | None = None,
    output_dir: Path | None = None,
) -> int:
    """Download and save FairCV records to the raw zone.

    The dataset is sorted by position, so the first N records may all share
    the same job title.  Use ``max_per_position`` to cap records per title
    and get a diverse sample across all positions.

    Args:
        max_samples: Overall limit on saved records.
        max_per_position: Cap per distinct position (default 50).
                          Set to None to disable per-position capping.
        positions: If given, only save resumes for these job positions.
        parquet_path: Path to a locally downloaded Parquet shard.
        local_dir: Directory of files downloaded via huggingface-cli.
        output_dir: Override the default ``data/raw/fairCV`` directory.

    Returns:
        Number of records saved.
    """
    out_dir = output_dir or cfg.paths.raw_fairCV
    ensure_dir(out_dir)

    effective_max = max_samples if max_samples is not None else cfg.collection.fairCV_max_samples
    effective_per_pos = max_per_position if max_per_position is not None else 50
    pos_filter = set(positions) if positions else None

    if parquet_path is not None:
        raw_iter = _iter_parquet(parquet_path)
    elif local_dir is not None:
        raw_iter = _iter_local_dir(local_dir)
    else:
        raw_iter = _iter_streaming()

    saved = 0
    pos_counts: dict[str, int] = {}

    for idx, record in enumerate(raw_iter):
        if effective_max and saved >= effective_max:
            break

        meta = record.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        pos = str(meta.get("position", "unknown"))

        if pos_filter and pos not in pos_filter:
            continue

        if pos_counts.get(pos, 0) >= effective_per_pos:
            continue

        raw_doc = _build_raw_doc(record, idx)
        save_raw_doc(out_dir, raw_doc["doc_id"], raw_doc)
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
        saved += 1
        if saved % 200 == 0:
            logger.info("  saved %d records, %d distinct positions …", saved, len(pos_counts))

    logger.info(
        "Position distribution: %s",
        ", ".join(f"{p}×{n}" for p, n in sorted(pos_counts.items())),
    )

    catalog.upsert(CatalogEntry(
        source_id     = "fairCV_v1",
        source_name   = "OhMyKing/FairCV",
        source_type   = "fairCV",
        source_url    = _SOURCE_URL,
        license_note  = _LICENSE,
        snapshot_date = _SNAPSHOT_DATE,
        record_count  = saved,
        local_path    = str(out_dir.relative_to(REPO_ROOT)),
        description   = "Simulated Chinese resumes; position / skill_level / resume_text",
        tags          = ["resume", "zh", "fairCV"],
    ))

    logger.info("FairCV fetch complete: %d records saved to %s", saved, out_dir)
    return saved


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--max-samples", type=int, default=None,
                   help="Overall maximum records to save")
    p.add_argument("--max-per-position", type=int, default=50,
                   help="Max records per distinct position (default 50, prevents all-same-title samples)")
    p.add_argument("--positions", type=str, default=None,
                   help="Comma-separated job positions to filter")
    p.add_argument("--parquet-path", type=Path, default=None,
                   help="Path to a locally downloaded .parquet shard (Route A)")
    p.add_argument("--local-dir", type=Path, default=None,
                   help="Directory with files from huggingface-cli download (Route B)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override output directory")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    pos_list = [p.strip() for p in args.positions.split(",")] if args.positions else None
    n = fetch(
        max_samples=args.max_samples,
        max_per_position=args.max_per_position,
        positions=pos_list,
        parquet_path=args.parquet_path,
        local_dir=args.local_dir,
        output_dir=args.output_dir,
    )
    print(f"Done: {n} FairCV records saved.")

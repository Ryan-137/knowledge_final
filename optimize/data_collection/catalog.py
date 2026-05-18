"""Data catalog management: maintains docs/data_catalog.md."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from optimize.config import cfg
from optimize.utils.logging_utils import get_pipeline_logger

logger = get_pipeline_logger("data_collection.catalog")


@dataclass
class CatalogEntry:
    """Metadata record for a single data source."""

    source_id:     str
    source_name:   str
    source_type:   str
    source_url:    str
    license_note:  str
    snapshot_date: str
    record_count:  int
    local_path:    str
    description:   str = ""
    tags:          list[str] = field(default_factory=list)


class CatalogManager:
    """Reads and writes the Markdown data catalog at ``docs/data_catalog.md``."""

    def __init__(self, catalog_path: Optional[Path] = None) -> None:
        self._path = catalog_path or cfg.paths.data_catalog
        self._entries: dict[str, CatalogEntry] = {}
        if self._path.exists():
            self._load()

    def upsert(self, entry: CatalogEntry) -> None:
        """Insert or update an entry and flush to disk immediately."""
        self._entries[entry.source_id] = entry
        self._save()
        logger.info("catalog updated: %s (%d records)", entry.source_id, entry.record_count)

    def get(self, source_id: str) -> Optional[CatalogEntry]:
        return self._entries.get(source_id)

    def all_entries(self) -> list[CatalogEntry]:
        return sorted(self._entries.values(), key=lambda e: e.source_id)

    def _load(self) -> None:
        """Parse basic fields from the existing Markdown table."""
        text = self._path.read_text(encoding="utf-8")
        pattern = re.compile(
            r"\|\s*(?P<source_id>\S+)\s*"
            r"\|\s*(?P<source_name>[^|]+?)\s*"
            r"\|\s*(?P<source_type>\S+)\s*"
            r"\|\s*(?P<license_note>[^|]+?)\s*"
            r"\|\s*(?P<snapshot_date>\S+)\s*"
            r"\|\s*(?P<record_count>\d+)\s*"
            r"\|"
        )
        for m in pattern.finditer(text):
            sid = m.group("source_id")
            self._entries[sid] = CatalogEntry(
                source_id=sid,
                source_name=m.group("source_name").strip(),
                source_type=m.group("source_type").strip(),
                source_url="",
                license_note=m.group("license_note").strip(),
                snapshot_date=m.group("snapshot_date").strip(),
                record_count=int(m.group("record_count")),
                local_path="",
            )

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = [
            "# Data Catalog\n\n",
            f"> Auto-generated. Last updated: {date.today().isoformat()}\n\n",
            "| source_id | Name | Type | License | Snapshot | Records | Local path | Description |\n",
            "| --------- | ---- | ---- | ------- | -------- | ------- | ---------- | ----------- |\n",
        ]
        for entry in self.all_entries():
            lines.append(
                f"| {entry.source_id}"
                f" | {entry.source_name}"
                f" | {entry.source_type}"
                f" | {entry.license_note}"
                f" | {entry.snapshot_date}"
                f" | {entry.record_count}"
                f" | {entry.local_path}"
                f" | {entry.description}"
                " |\n"
            )
        self._path.write_text("".join(lines), encoding="utf-8")


catalog = CatalogManager()

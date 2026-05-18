"""File I/O helpers: JSON, JSONL, and raw-zone document storage."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Generator, Iterable


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    """Read and parse a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Write *payload* to *path* as pretty-printed JSON using an atomic rename.

    The atomic rename guarantees the original file stays intact if the
    process is interrupted mid-write.
    """
    ensure_dir(path.parent)
    text = json.dumps(payload, ensure_ascii=False, indent=indent) + "\n"
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=path.stem + "_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return all records as a list.

    Empty lines and lines starting with ``#`` are silently skipped.
    """
    results: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: JSON parse error — {e}") from e
    return results


def iter_jsonl(path: Path) -> Generator[dict[str, Any], None, None]:
    """Lazily yield records from a JSONL file (suitable for large files)."""
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: JSON parse error — {e}") from e


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write *records* to a JSONL file (overwrite if exists).

    Returns the number of records written.
    """
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a single record to a JSONL file, creating it if necessary."""
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl_batch(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Append multiple records to a JSONL file.

    Returns the number of records appended.
    """
    ensure_dir(path.parent)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def save_raw_doc(raw_dir: Path, doc_id: str, payload: dict[str, Any]) -> Path:
    """Save a raw document to ``<raw_dir>/<doc_id>.json``.

    Skips writing if the file already exists with the same sha256, avoiding
    unnecessary disk writes on repeated runs.
    """
    from optimize.utils.hash_utils import dict_sha256

    target = raw_dir / f"{doc_id}.json"
    if target.exists():
        existing = read_json(target)
        if existing.get("sha256") == payload.get("sha256"):
            return target
    write_json(target, payload)
    return target

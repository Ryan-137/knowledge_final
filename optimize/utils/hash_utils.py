"""SHA-256 helpers for content fingerprinting and deduplication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's raw bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    """Return the SHA-256 hex digest of *text* encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dict_sha256(obj: Any) -> str:
    """Return the SHA-256 hex digest of *obj* serialised as canonical JSON.

    ``sort_keys=True`` ensures identical dicts always produce the same hash
    regardless of insertion order.
    """
    canonical = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return text_sha256(canonical)

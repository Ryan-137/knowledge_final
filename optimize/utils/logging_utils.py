"""Logging factory for the entity extraction pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    *,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Return a logger that writes to stderr and optionally a file.

    Args:
        name: Module name, typically ``__name__``.
        level: Minimum level for console output.
        log_file: If given, also write to this file in UTF-8.
        file_level: Minimum level for file output.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_pipeline_logger(module_name: str) -> logging.Logger:
    """Return a logger with file output for a pipeline module.

    Log files are written to ``optimize/.logs/<module_name>.log``.
    """
    from optimize.config import cfg

    log_file = cfg.paths.log_dir / f"{module_name}.log"
    return get_logger(
        name=f"optimize.{module_name}",
        log_file=log_file,
    )

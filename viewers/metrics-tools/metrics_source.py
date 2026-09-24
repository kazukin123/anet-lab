"""Shared raw/gzip MetricsLogger source selection helpers."""

from __future__ import annotations

import gzip
from pathlib import Path


RAW_NAME = "metrics.jsonl"
GZIP_NAME = "metrics.jsonl.gz"


def resolve_run_metrics(run_dir: str | Path) -> Path | None:
    run_path = Path(run_dir)
    raw_path = run_path / RAW_NAME
    if raw_path.is_file():
        return raw_path
    gzip_path = run_path / GZIP_NAME
    if gzip_path.is_file():
        return gzip_path
    return None


def resolve_metrics_path(path: str | Path) -> Path:
    metrics_path = Path(path)
    if metrics_path.is_dir():
        resolved = resolve_run_metrics(metrics_path)
        if resolved is not None:
            return resolved
    elif metrics_path.is_file():
        return metrics_path
    elif metrics_path.name == RAW_NAME:
        gzip_path = metrics_path.with_name(GZIP_NAME)
        if gzip_path.is_file():
            return gzip_path
    raise FileNotFoundError(f"Metrics file not found: {metrics_path}")


def logical_metrics_path(path: str | Path) -> Path:
    metrics_path = Path(path)
    return metrics_path.parent / RAW_NAME


def open_metrics_binary(path: str | Path):
    metrics_path = Path(path)
    if metrics_path.name == GZIP_NAME:
        return gzip.open(metrics_path, "rb")
    return metrics_path.open("rb")


def open_metrics_text(path: str | Path, encoding: str = "utf-8"):
    metrics_path = Path(path)
    if metrics_path.name == GZIP_NAME:
        return gzip.open(metrics_path, "rt", encoding=encoding)
    return metrics_path.open("r", encoding=encoding)

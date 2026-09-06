#!/usr/bin/env python3
"""AI-facing run inspection CLI. Reads run artifacts without modifying them."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import sys
import tempfile
from array import array
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from metrics_source import GZIP_NAME, RAW_NAME, open_metrics_binary, resolve_run_metrics


SCHEMA_VERSION = 2

CACHE_NAME = "metrics_cache.db"
RAW_KIND = "jsonl"
GZIP_KIND = "jsonl.gz"

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACES_ROOT = REPO_ROOT / "apps" / "runner" / "workspaces"

CONFIG_DIR_NAME = "config"
CONFIG_DATA_NAME = "config_data.txt"
CONFIG_REL_PATH = Path(CONFIG_DIR_NAME) / CONFIG_DATA_NAME
RESOLUTION_JSON_REL_PATH = Path("json") / "config_resolution.json"
LEGACY_RESOLUTION_REL_PATH = Path("config") / "config_resolution.json"
RESOLUTION_SCHEMA_VERSION = 1

SERIES_MAX_POINTS = 128
SERIES_BUCKETS = 42

METRICS_DEFS_TAG = "metrics.scalar.defs"
LEGACY_METRICS_DEFS_TAG = "metrics.defs"
METRIC_KEY_PREFIX = "metrics.scalar.["
STEP_AXIS_NAMES = frozenset(
    ["train_step", "learn_step", "episode_step", "exp_step", "update_step", "sim_step"]
)
EVENT_NAMES = frozenset(["train", "learn", "episode_end", "session_end"])
TARGET_TOKENS = {
    "agent": "agent",
    "env": "env",
    "exp": "exp",
    "batch_experience": "exp",
    "update_result": "update_result",
    "batch_update_result": "update_result",
    "result": "update_result",
    "runner": "runner",
    "action": "action_info",
    "action_info": "action_info",
}
UNKNOWN = "unknown"
TRAIN_RUNNER = "train"

# JSONL の step は Metrics Viewer と同じく MAX_SAFE_INTEGER までを有効とする。
MAX_SAFE_STEP = 9007199254740991

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2


class UsageError(Exception):
    """引数、range構文、Run解決の失敗。終了値2に対応する。"""


class RuntimeFailure(Exception):
    """source read、query、出力書込み等の実行時失敗。終了値1に対応する。"""


class SourceError(Exception):
    """Metricsマスタの構造違反。Run単位のsource errorとして扱う。"""


# ---------------------------------------------------------------------------
# Run 解決と列挙
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedRun:
    input: str | None
    run_dir: Path
    run_name: str
    workspace: str | None


def _workspace_of(run_dir: Path) -> str | None:
    # workspace 配下は <workspaces_root>/<ws>/runs/<run> の形だけを認める。
    parent = run_dir.parent
    if parent.name != "runs":
        return None
    workspace_dir = parent.parent
    try:
        same_root = workspace_dir.parent == WORKSPACES_ROOT.resolve()
    except OSError:
        return None
    return workspace_dir.name if same_root else None


def resolve_run(value: str) -> ResolvedRun:
    # 1. cwd 基準または絶対 path として既存 directory へ解決できればそれを使う。
    candidate = Path(value)
    if candidate.is_dir():
        run_dir = candidate.resolve()
        return ResolvedRun(value, run_dir, run_dir.name, _workspace_of(run_dir))

    # 2. directory へ解決できなければ現行 workspace 直下の Run 名として完全一致で探す。
    #    独自 shorthand を認めないため、区切りを含む入力は名前探索の対象にしない。
    matches: list[Path] = []
    if value not in ("", ".", "..") and "/" not in value and "\\" not in value:
        if WORKSPACES_ROOT.is_dir():
            for workspace_dir in sorted(WORKSPACES_ROOT.iterdir()):
                run_dir = workspace_dir / "runs" / value
                if run_dir.is_dir():
                    matches.append(run_dir.resolve())

    if not matches:
        raise UsageError(f"Run not found: {value} (searched {WORKSPACES_ROOT})")
    if len(matches) > 1:
        candidates = ", ".join(str(path) for path in matches)
        raise UsageError(f"Ambiguous run name: {value} (candidates: {candidates})")

    run_dir = matches[0]
    return ResolvedRun(value, run_dir, run_dir.name, _workspace_of(run_dir))


def list_runs(workspace: str | None) -> list[ResolvedRun]:
    """workspace 配下の Run を列挙する。Metrics マスタの有無で絞らない。"""
    if not WORKSPACES_ROOT.is_dir():
        raise UsageError(f"workspaces root does not exist: {WORKSPACES_ROOT}")

    workspace_dirs = sorted(item for item in WORKSPACES_ROOT.iterdir() if item.is_dir())
    if workspace is not None:
        workspace_dirs = [item for item in workspace_dirs if item.name == workspace]
        if not workspace_dirs:
            available = ", ".join(item.name for item in sorted(WORKSPACES_ROOT.iterdir()))
            raise UsageError(f"workspace not found: {workspace} (available: {available})")

    resolved: list[ResolvedRun] = []
    for workspace_dir in workspace_dirs:
        runs_dir = workspace_dir / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in sorted(runs_dir.iterdir(), key=lambda item: item.name.casefold()):
            if not run_dir.is_dir():
                continue
            target = run_dir.resolve()
            resolved.append(ResolvedRun(None, target, target.name, workspace_dir.name))
    return resolved


# ---------------------------------------------------------------------------
# 実効 config
# ---------------------------------------------------------------------------


def _parse_config_text(text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        if not separator:
            continue
        entries.append((key.strip(), value.strip()))
    return entries


def read_config_entries(config_path: Path) -> list[tuple[str, str]]:
    """config_data.txt を flat key の出現順で読む。値は型変換しない。"""
    try:
        text = config_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise RuntimeFailure(f"Failed to read config: {config_path}: {exc}") from exc
    return _parse_config_text(text)


def read_effective_keys(run_dir: Path) -> set[str]:
    """config/<module>.txt が持つ key 集合。各 module が実際に読んだ設定の dump。"""
    config_dir = run_dir / CONFIG_DIR_NAME
    if not config_dir.is_dir():
        return set()
    keys: set[str] = set()
    for path in sorted(config_dir.glob("*.txt")):
        if path.name == CONFIG_DATA_NAME:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError:
            continue
        keys.update(key for key, _ in _parse_config_text(text))
    return keys


def compile_selector(selector: str) -> re.Pattern:
    """`*` と `?` だけを glob meta とする。

    実効 config の key と metric tag はどちらも `[tag]` 記法が偏在するため、
    `fnmatch` の character class 解釈をやめて `[` `]` をリテラル文字として照合する。
    """
    pattern = []
    for char in selector:
        if char == "*":
            pattern.append(".*")
        elif char == "?":
            pattern.append(".")
        else:
            pattern.append(re.escape(char))
    return re.compile("".join(pattern) + r"\Z")


def has_glob_meta(selector: str) -> bool:
    return "*" in selector or "?" in selector


def select_config_values(entries, selectors, effective_keys, effective_only):
    """selector 順、その中では config file 出現順で値を返す。同じ key は1回だけ。"""
    by_key: dict[str, str] = {}
    for key, value in entries:
        by_key.setdefault(key, value)

    selector_results = []
    values = []
    seen: set[str] = set()
    for selector in selectors:
        pattern = compile_selector(selector)
        matched = [key for key in by_key if pattern.match(key)]
        if effective_only:
            matched = [key for key in matched if key in effective_keys]
        selector_results.append(
            {
                "selector": selector,
                "status": "ok" if matched else "missing",
                "matched": len(matched),
            }
        )
        for key in matched:
            if key in seen:
                continue
            seen.add(key)
            # 判定できないものを false と言わない。module dump を出さない領域があるため。
            values.append(
                {
                    "key": key,
                    "value": by_key[key],
                    "effective": True if key in effective_keys else None,
                }
            )
    return selector_results, values


def build_config_diff(run_names, run_entries, selectors, effective_keys_list, effective_only):
    """値または存在有無が Run 間で異なる key だけを返す。Run が1件なら空。"""
    if len(run_entries) < 2:
        return []

    maps = [dict(entries) for entries in run_entries]
    union: list[str] = []
    seen: set[str] = set()
    for entry_map in maps:
        for key in entry_map:
            if key not in seen:
                seen.add(key)
                union.append(key)

    if selectors:
        patterns = [compile_selector(selector) for selector in selectors]
        union = [key for key in union if any(pattern.match(key) for pattern in patterns)]
    if effective_only:
        union = [
            key for key in union if any(key in keys for keys in effective_keys_list)
        ]

    diff = []
    for key in union:
        presence = [(key in entry_map, entry_map.get(key)) for entry_map in maps]
        if len(set(presence)) <= 1:
            continue
        diff.append(
            {
                "key": key,
                "runs": [
                    {"run": name, "present": present, "value": value if present else None}
                    for name, (present, value) in zip(run_names, presence)
                ],
            }
        )
    return diff


# ---------------------------------------------------------------------------
# metrics 定義（step 座標系）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDef:
    step_axis: str = UNKNOWN
    runner: str = UNKNOWN
    event: str | None = None
    target: str | None = None
    source_key: str | None = None
    ema_alpha: float | None = None
    interval: int | None = None
    scope: str | None = None
    eval_name: str | None = None
    eval_episodes: int | None = None
    num_envs: int | None = None
    clip: float | None = None

    def space(self) -> tuple[str, str]:
        """step 座標系。軸名だけでは同一性が決まらないので Runner との組で識別する。"""
        return (self.runner, self.step_axis)

    def is_known(self) -> bool:
        return self.step_axis != UNKNOWN and self.runner != UNKNOWN

    def to_json(self) -> dict:
        return {
            "step_axis": self.step_axis,
            "runner": self.runner,
            "event": self.event,
            "target": self.target,
            "source_key": self.source_key,
            "ema_alpha": self.ema_alpha,
            "interval": self.interval,
            "scope": self.scope,
            "eval_name": self.eval_name,
            "eval_episodes": self.eval_episodes,
            "num_envs": self.num_envs,
            "clip": self.clip,
        }


def _runner_of(runner_scope: str, event: str) -> str:
    """eval scope かつ @train のときだけ eval runner 自身の counts が載る。"""
    if runner_scope != TRAIN_RUNNER and event == "train":
        return runner_scope
    return TRAIN_RUNNER


def metric_def_from_definition(definition: str) -> MetricDef:
    """metrics.scalar 定義の token 列から解決済み定義を復元する。token 順序は自由。"""
    axis: str | None = None
    event: str | None = None
    runner_scope = TRAIN_RUNNER
    scope = "train"
    target: str | None = None
    source_key: str | None = None
    ema_alpha: float | None = None
    interval: int | None = None
    clip: float | None = None

    for token in definition.split():
        if token.startswith("$"):
            name = token[1:]
            if name in STEP_AXIS_NAMES:
                axis = name
            elif name in TARGET_TOKENS:
                target = TARGET_TOKENS[name]
            elif name == TRAIN_RUNNER:
                runner_scope = TRAIN_RUNNER
                scope = "train"
            elif name.startswith("eval.[") and name.endswith("]"):
                runner_scope = name[len("eval.[") : -1] or UNKNOWN
                scope = "eval"
            elif name == "ema" and ema_alpha is None:
                ema_alpha = 0.01
            continue
        if token.startswith("@"):
            event = token[1:]
            continue

        head, separator, tail = token.partition(":")
        if not separator or not tail:
            source_key = token
            continue
        if head in ("step", "step_axis") and tail in STEP_AXIS_NAMES:
            axis = tail
        elif head == "event":
            event = tail
        elif head == "target" and tail in TARGET_TOKENS:
            target = TARGET_TOKENS[tail]
        elif head == "key":
            source_key = tail
        elif head == "ema_alpha":
            ema_alpha = _safe_float(tail)
        elif head == "interval":
            interval = _safe_int(tail)
        elif head == "clip":
            clip = _safe_float(tail)

    # 購読先は token から復元するが、実際の eval 条件は config から推測しない。
    eval_name = runner_scope if scope == "eval" and runner_scope != UNKNOWN else None
    resolved_event = event if event in EVENT_NAMES else ("train" if event is None else UNKNOWN)
    if resolved_event == UNKNOWN:
        return MetricDef(source_key=source_key, ema_alpha=ema_alpha, interval=interval,
                         scope=scope, eval_name=eval_name, clip=clip)

    # 明示指定を最優先し、無い場合だけ event 既定へ落とす。
    if axis is None:
        axis = "train_step" if resolved_event == "train" else "exp_step"

    return MetricDef(
        step_axis=axis,
        runner=_runner_of(runner_scope, resolved_event),
        event=resolved_event,
        target=target,
        source_key=source_key,
        ema_alpha=ema_alpha,
        interval=interval,
        scope=scope,
        eval_name=eval_name,
        clip=clip,
    )


def _safe_float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _safe_int(text: str) -> int | None:
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def metric_defs_from_config(config_entries) -> dict[str, MetricDef]:
    """解決済みの metrics.scalar.[<tag>] だけを読む。未適用の生キーは対象外。"""
    defs: dict[str, MetricDef] = {}
    for key, value in config_entries:
        if not key.startswith(METRIC_KEY_PREFIX) or not key.endswith("]"):
            continue
        tag = key[len(METRIC_KEY_PREFIX) : -1]
        defs[tag] = metric_def_from_definition(value)
    return defs


def metric_defs_from_record(data) -> dict[str, MetricDef]:
    """metrics.scalar.defs レコードの data を MetricDef へ変換する。"""
    defs: dict[str, MetricDef] = {}
    if not isinstance(data, dict):
        return defs
    for tag, entry in data.items():
        if not isinstance(entry, dict):
            continue
        defs[str(tag)] = MetricDef(
            step_axis=str(entry.get("step_axis") or UNKNOWN),
            runner=str(entry.get("runner") or UNKNOWN),
            event=entry.get("event"),
            target=entry.get("target"),
            source_key=entry.get("source_key"),
            ema_alpha=entry.get("ema_alpha"),
            interval=entry.get("interval"),
            scope=entry.get("scope"),
            eval_name=entry.get("eval_name"),
            eval_episodes=entry.get("eval_episodes"),
            num_envs=entry.get("num_envs"),
            clip=entry.get("clip"),
        )
    return defs


# ---------------------------------------------------------------------------
# artifact inspection
# ---------------------------------------------------------------------------


def _mtime_iso(stat_result) -> str:
    return datetime.fromtimestamp(stat_result.st_mtime).astimezone().isoformat(timespec="seconds")


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeFailure(f"Failed to hash {path}: {exc}") from exc
    return digest.hexdigest()


def inspect_artifacts(run_dir: Path, master_path: Path | None) -> dict:
    """Run directory 直下だけを見る。再帰走査はしない。"""
    config_path = run_dir / CONFIG_REL_PATH
    config_info: dict = {"path": str(config_path), "exists": False,
                         "size": None, "mtime": None, "sha256": None}
    if config_path.is_file():
        stat_result = config_path.stat()
        config_info.update(
            exists=True,
            size=stat_result.st_size,
            mtime=_mtime_iso(stat_result),
            sha256=_sha256_of(config_path),
        )

    raw_path = run_dir / RAW_NAME
    gzip_path = run_dir / GZIP_NAME
    master_info: dict = {
        "raw_exists": raw_path.is_file(),
        "gzip_exists": gzip_path.is_file(),
        "path": None,
        "kind": None,
        "size": None,
        "mtime": None,
    }
    if master_path is not None:
        stat_result = master_path.stat()
        master_info.update(
            path=str(master_path),
            kind=GZIP_KIND if master_path.name == GZIP_NAME else RAW_KIND,
            size=stat_result.st_size,
            mtime=_mtime_iso(stat_result),
        )

    cache_path = run_dir / CACHE_NAME
    cache_info: dict = {"path": str(cache_path), "exists": False, "size": None, "mtime": None,
                        "status": "absent", "reason": None, "source_meta": None}
    if cache_path.is_file():
        stat_result = cache_path.stat()
        cache_info.update(exists=True, size=stat_result.st_size, mtime=_mtime_iso(stat_result))

    # log と snapshot は名前で拾い、directory は辿らない。
    # agent_close.anet の有無と mtime は Run の完了・中断判定の材料になる。
    files = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_file() or child.suffix not in (".log", ".anet"):
            continue
        stat_result = child.stat()
        files.append(
            {
                "name": child.name,
                "path": str(child),
                "size": stat_result.st_size,
                "mtime": _mtime_iso(stat_result),
            }
        )

    return {"config": config_info, "master": master_info, "cache": cache_info, "files": files}


# ---------------------------------------------------------------------------
# Metrics キャッシュ
# ---------------------------------------------------------------------------


CACHE_APPLICATION_ID = 0x414E4554
CACHE_SCHEMA_VERSION = 1
FINGERPRINT_BYTES = 64 * 1024

CACHE_REQUIRED_COLUMNS = {
    "tags": ("id", "key", "type", "status"),
    "scalars": ("tag_id", "ordinal", "step", "value"),
    "scalars_lod": ("tag_id", "level", "bucket"),
    "tag_stats": ("tag_id", "count", "mean"),
    "json_lines": ("ordinal", "type", "json"),
    "source_meta": ("k", "v"),
}
CACHE_REQUIRED_META = (
    "generation",
    "state",
    "source_kind",
    "source_size",
    "source_mtime",
    "source_head_sha256",
    "source_commit_tail_sha256",
    "committed_offset",
)


@dataclass
class CacheStatus:
    status: str
    reason: str | None = None
    source_meta: dict | None = None

    def is_current(self) -> bool:
        return self.status == "current"


def open_cache_readonly(cache_path: Path) -> sqlite3.Connection:
    """read-only URI で開く。作成・migration・修復・更新は一切行わない。"""
    connection = sqlite3.connect(
        f"{cache_path.resolve().as_uri()}?mode=ro", uri=True, isolation_level=None
    )
    connection.execute("PRAGMA query_only = ON")
    return connection


def _file_sha256_range(path: Path, start: int, length: int) -> str:
    digest = hashlib.sha256()
    if length <= 0:
        return digest.hexdigest()
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining > 0:
            chunk = handle.read(min(1 << 20, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def validate_cache(cache_path: Path, master_path: Path | None, master_stat) -> CacheStatus:
    """PRD の eligibility 6条件を順に判定する。cache contract は独自に再定義しない。"""
    if not cache_path.is_file():
        return CacheStatus("absent", "cache file does not exist")

    try:
        connection = open_cache_readonly(cache_path)
    except sqlite3.Error as exc:
        return CacheStatus("invalid", f"database_open_failed: {exc}")

    try:
        # 1. file identity
        if connection.execute("PRAGMA application_id").fetchone()[0] != CACHE_APPLICATION_ID:
            return CacheStatus("invalid", "application_id_mismatch")
        if connection.execute("PRAGMA user_version").fetchone()[0] != CACHE_SCHEMA_VERSION:
            return CacheStatus("invalid", "schema_version_mismatch")

        # 2. 必須 table と column
        for table, columns in CACHE_REQUIRED_COLUMNS.items():
            present = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            if not present:
                return CacheStatus("invalid", f"required_table_missing: {table}")
            missing = [name for name in columns if name not in present]
            if missing:
                return CacheStatus("invalid", f"required_column_missing: {table}.{missing[0]}")

        # 3. source_meta の必須 key と型変換
        meta = dict(connection.execute("SELECT k, v FROM source_meta"))
        missing_keys = [key for key in CACHE_REQUIRED_META if key not in meta]
        if missing_keys:
            return CacheStatus(
                "invalid", f"source_metadata_invalid: missing {', '.join(missing_keys)}", meta
            )
        try:
            stored_size = int(meta["source_size"])
            stored_mtime = int(meta["source_mtime"])
            committed_offset = int(meta["committed_offset"])
        except (TypeError, ValueError):
            return CacheStatus("invalid", "source_metadata_invalid: non-numeric value", meta)

        # 4. state
        state = meta["state"]
        if state == "error":
            return CacheStatus(
                "error", f"cache ingest failed: {meta.get('error_code', 'unknown')}", meta
            )
        if state != "ready":
            return CacheStatus("partial", f"cache ingest state is {state}", meta)

        # 5. 選択済み Metrics マスタとの同一性
        if master_path is None:
            return CacheStatus("stale", "metrics master is absent", meta)
        master_size = master_stat[0]
        # Metrics Viewer は lastModifiedTime().toMillis() を保存するのでミリ秒へ揃える。
        master_mtime_ms = master_stat[1] // 1_000_000
        expected_kind = GZIP_KIND if master_path.name == GZIP_NAME else RAW_KIND
        if meta["source_kind"] != expected_kind:
            return CacheStatus("stale", "source_kind_changed", meta)
        if stored_size != master_size:
            return CacheStatus("stale", "source_size_changed", meta)
        if stored_mtime != master_mtime_ms:
            return CacheStatus("stale", "source_mtime_changed", meta)

        head_length = min(master_size, stored_size, FINGERPRINT_BYTES)
        if _file_sha256_range(master_path, 0, head_length) != meta["source_head_sha256"]:
            return CacheStatus("stale", "source_head_changed", meta)
        tail_end = max(0, min(committed_offset, master_size))
        tail_start = max(0, tail_end - FINGERPRINT_BYTES)
        tail = _file_sha256_range(master_path, tail_start, tail_end - tail_start)
        if tail != meta["source_commit_tail_sha256"]:
            return CacheStatus("stale", "committed_source_tail_changed", meta)

        # 6. commit 済み範囲がマスタ全体に追随しているか
        if committed_offset != stored_size or stored_size != master_size:
            return CacheStatus(
                "partial",
                f"committed_offset {committed_offset} does not cover source size {master_size}",
                meta,
            )

        return CacheStatus("current", None, meta)
    except sqlite3.Error as exc:
        return CacheStatus("invalid", f"cache query failed: {exc}")
    except OSError as exc:
        raise RuntimeFailure(f"Failed to fingerprint metrics master: {master_path}: {exc}") from exc
    finally:
        connection.close()


@dataclass
class TagSeries:
    steps: array = field(default_factory=lambda: array("q"))
    values: array = field(default_factory=lambda: array("d"))
    excluded: int = 0
    quarantined: bool = False
    present: bool = False


@dataclass
class CacheInventory:
    tags: dict = field(default_factory=dict)
    defs: dict = field(default_factory=dict)


def read_cache_inventory(cache_path: Path, want_observed: bool) -> CacheInventory:
    """1 read transaction で tags / tag_stats / metrics.scalar.defs を読む。"""
    try:
        connection = open_cache_readonly(cache_path)
    except sqlite3.Error as exc:
        raise RuntimeFailure(f"Failed to open metrics cache: {cache_path}: {exc}") from exc

    inventory = CacheInventory()
    try:
        connection.execute("BEGIN")
        for tag_id, key, status in connection.execute("SELECT id, key, status FROM tags"):
            inventory.tags[key] = {"tag_id": tag_id, "status": status, "observed": None}
        if want_observed:
            for tag_id, count, min_step, max_step in connection.execute(
                "SELECT tag_id, count, min_step, max_step FROM tag_stats"
            ):
                for entry in inventory.tags.values():
                    if entry["tag_id"] == tag_id:
                        entry["observed"] = {
                            "count": count,
                            "min_step": min_step,
                            "max_step": max_step,
                        }
                        break
        row = connection.execute(
            "SELECT json FROM json_lines WHERE tag = ? ORDER BY ordinal DESC LIMIT 1",
            (METRICS_DEFS_TAG,),
        ).fetchone()
        if row is None:
            row = connection.execute(
                "SELECT json FROM json_lines WHERE tag = ? ORDER BY ordinal DESC LIMIT 1",
                (LEGACY_METRICS_DEFS_TAG,),
            ).fetchone()
        if row is not None:
            try:
                inventory.defs = metric_defs_from_record(json.loads(row[0]).get("data"))
            except ValueError:
                inventory.defs = {}
        connection.execute("ROLLBACK")
    except sqlite3.Error as exc:
        raise RuntimeFailure(f"Metrics cache query failed: {cache_path}: {exc}") from exc
    finally:
        connection.close()
    return inventory


def read_cache_series(cache_path: Path, tags: list[str], known: dict) -> dict:
    """L0 scalars から選択 tag の全点を読む。統計は LOD から復元しない。"""
    series = {tag: TagSeries() for tag in tags}
    wanted = [tag for tag in tags if tag in known]
    if not wanted:
        return series

    try:
        connection = open_cache_readonly(cache_path)
    except sqlite3.Error as exc:
        raise RuntimeFailure(f"Failed to open metrics cache: {cache_path}: {exc}") from exc

    try:
        connection.execute("BEGIN")
        for tag in wanted:
            entry = series[tag]
            entry.present = True
            for step, value in connection.execute(
                "SELECT step, value FROM scalars WHERE tag_id = ? ORDER BY ordinal",
                (known[tag]["tag_id"],),
            ):
                entry.steps.append(step)
                entry.values.append(value)
            # cache 経路でも tags.status == error は quarantined として扱う。
            entry.quarantined = known[tag]["status"] == "error"
        connection.execute("ROLLBACK")
    except sqlite3.Error as exc:
        raise RuntimeFailure(f"Metrics cache query failed: {cache_path}: {exc}") from exc
    finally:
        connection.close()
    return series


# ---------------------------------------------------------------------------
# Metrics マスタ走査
# ---------------------------------------------------------------------------


def _float32_representable(value: float) -> bool:
    # Metrics Viewer の (float) キャストと同じ丸めで範囲外を判定する。
    try:
        struct.pack("<f", value)
    except (OverflowError, ValueError):
        return False
    return True


def _numeric_value(raw) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if not math.isfinite(value) or not _float32_representable(value):
        return None
    return value


def _source_snapshot(path: Path) -> tuple[int, int]:
    try:
        stat_result = path.stat()
    except OSError as exc:
        raise RuntimeFailure(f"Failed to stat metrics master: {path}: {exc}") from exc
    return stat_result.st_size, stat_result.st_mtime_ns


class MasterLineReader:
    """raw は開始時 snapshot size を上限に読み、未終端の末尾行を取り込まない。"""

    def __init__(self, handle, byte_limit: int | None):
        self._handle = handle
        self._remaining = byte_limit
        self.trailing_bytes = 0

    def lines(self):
        buffer = b""
        while True:
            if self._remaining is not None and self._remaining <= 0:
                break
            size = 1 << 20
            if self._remaining is not None:
                size = min(size, self._remaining)
            chunk = self._handle.read(size)
            if not chunk:
                break
            if self._remaining is not None:
                self._remaining -= len(chunk)
            buffer += chunk
            start = 0
            while True:
                index = buffer.find(b"\n", start)
                if index < 0:
                    break
                yield buffer[start:index]
                start = index + 1
            buffer = buffer[start:]
        self.trailing_bytes = len(buffer)


@dataclass
class MasterScan:
    series: dict = field(default_factory=dict)
    defs: dict = field(default_factory=dict)
    observed: dict = field(default_factory=dict)
    trailing_bytes: int = 0


def _parse_record(line: bytes):
    if not line.strip():
        raise SourceError("empty line in metrics master")
    try:
        record = json.loads(line)
    except ValueError as exc:
        raise SourceError(f"invalid JSON line: {exc}") from exc
    if not isinstance(record, dict) or "type" not in record:
        raise SourceError("invalid record: missing type")
    return record


def _scalar_fields(record: dict):
    tag = record.get("tag")
    if not isinstance(tag, str) or "value" not in record:
        raise SourceError("invalid scalar record: missing tag or value")
    step = record.get("step")
    if isinstance(step, bool) or not isinstance(step, int):
        raise SourceError(f"invalid step for tag {tag}")
    if step < 0 or step > MAX_SAFE_STEP:
        raise SourceError(f"invalid step for tag {tag}: {step}")
    return tag, step, record["value"]


def scan_master(master_path: Path, tags: list[str] | None, byte_limit: int | None) -> MasterScan:
    """必要な全 tag を 1 pass で読む。tags が None なら inventory だけを集める。"""
    scan = MasterScan()
    has_scalar_defs = False
    wanted = set(tags) if tags is not None else set()
    if tags is not None:
        scan.series = {tag: TagSeries() for tag in tags}

    try:
        handle = open_metrics_binary(master_path)
    except OSError as exc:
        raise RuntimeFailure(f"Failed to open metrics master: {master_path}: {exc}") from exc

    is_gzip = master_path.name == GZIP_NAME
    with handle:
        reader = MasterLineReader(handle, byte_limit)
        try:
            for line in reader.lines():
                record = _parse_record(line)
                if record["type"] != "scalar":
                    if record.get("tag") == METRICS_DEFS_TAG:
                        scan.defs = metric_defs_from_record(record.get("data"))
                        has_scalar_defs = True
                    elif record.get("tag") == LEGACY_METRICS_DEFS_TAG and not has_scalar_defs:
                        scan.defs = metric_defs_from_record(record.get("data"))
                    continue

                tag, step, raw_value = _scalar_fields(record)
                value = _numeric_value(raw_value)

                # tag inventory は選択に関わらず全 tag について集める。
                observed = scan.observed.get(tag)
                if observed is None:
                    observed = {"count": 0, "min_step": None, "max_step": None}
                    scan.observed[tag] = observed
                if value is not None:
                    observed["count"] += 1
                    if observed["min_step"] is None:
                        observed["min_step"] = step
                    observed["max_step"] = step

                if tag not in wanted:
                    continue
                entry = scan.series[tag]
                entry.present = True
                if entry.quarantined:
                    continue
                if value is None:
                    entry.excluded += 1
                    continue
                if entry.steps and step < entry.steps[-1]:
                    # step 逆行の tag は隔離し、逆行前の有効 prefix だけを公開する。
                    entry.quarantined = True
                    continue
                entry.steps.append(step)
                entry.values.append(value)
        except (EOFError, OSError) as exc:
            raise SourceError(f"failed to read metrics master: {exc}") from exc

    # gzip は immutable source なので、未終端行を追記待ちにせず source error とする。
    if is_gzip and reader.trailing_bytes:
        raise SourceError("gzip ended with an unterminated JSON line")

    scan.trailing_bytes = reader.trailing_bytes
    return scan


# ---------------------------------------------------------------------------
# range
# ---------------------------------------------------------------------------


RANGE_ALL_LABEL = "all"
RANGE_MODES = ("all", "common")
_ABSOLUTE_RE = re.compile(r"\A(-?)(\d+)([kKmMgG]?)\Z")
_PERCENT_RE = re.compile(r"\A(-?)(\d+(?:\.\d+)?|\.\d+)%\Z")
_SUFFIX_SCALE = {"": 1, "k": 1_000, "m": 1_000_000, "g": 1_000_000_000}


@dataclass(frozen=True)
class Endpoint:
    """range の端点。空指定、絶対値、百分率、末尾相対を表す。"""

    kind: str  # "open" | "absolute" | "percent"
    value: object = None
    negative: bool = False

    def needs_space(self) -> bool:
        return self.kind == "percent" or (self.kind == "absolute" and self.negative)


@dataclass(frozen=True)
class RangeSpec:
    label: str
    kind: str  # "all" | "absolute" | "relative" | "common"
    start: Endpoint | None = None
    end: Endpoint | None = None

    def needs_space(self) -> bool:
        if self.kind in ("common", "all"):
            return self.kind == "common"
        return bool(
            (self.start and self.start.needs_space())
            or (self.end and self.end.needs_space())
            or (self.start and self.start.kind == "open")
            or (self.end and self.end.kind == "open")
        )


def _parse_endpoint(text: str, spec: str, side: str) -> Endpoint:
    if not text:
        return Endpoint("open")

    percent = _PERCENT_RE.match(text)
    if percent:
        value = Fraction(percent.group(2))
        if value > 100:
            raise UsageError(f"invalid range: {spec} ({side} percentage must be within 0..100)")
        return Endpoint("percent", value, percent.group(1) == "-")

    absolute = _ABSOLUTE_RE.match(text)
    if absolute:
        value = int(absolute.group(2)) * _SUFFIX_SCALE[absolute.group(3).lower()]
        if value > MAX_SAFE_STEP:
            raise UsageError(f"invalid range: {spec} ({side} endpoint exceeds {MAX_SAFE_STEP})")
        return Endpoint("absolute", value, absolute.group(1) == "-")

    raise UsageError(
        f"invalid range: {spec} ({side} endpoint must be an integer with an optional "
        f"K/M/G suffix, or a percentage; an optional leading '-' means relative to the "
        f"last observed step)"
    )


def parse_range(text: str) -> RangeSpec:
    left, separator, right = text.partition(":")
    if not separator:
        raise UsageError(f"invalid range: {text} (expected START:END)")
    left = left.strip()
    right = right.strip()
    if not left and not right:
        raise UsageError(f"invalid range: {text} (at least one endpoint is required)")

    start = _parse_endpoint(left, text, "start")
    end = _parse_endpoint(right, text, "end")

    kinds = {point.kind for point in (start, end) if point.kind != "open"}
    if len(kinds) > 1:
        raise UsageError(
            f"invalid range: {text} (absolute and percentage endpoints must not be mixed)"
        )

    relative = start.needs_space() or end.needs_space() or start.kind == "open" or end.kind == "open"
    kind = "relative" if relative else "absolute"

    # 静的に比較できる場合だけ parse 時点で逆転を弾く。
    if (
        kind == "absolute"
        and not start.negative
        and not end.negative
        and start.value > end.value
    ):
        raise UsageError(f"invalid range: {text} (START must not exceed END)")

    return RangeSpec(label=text, kind=kind, start=start, end=end)


def parse_range_mode(text: str) -> RangeSpec:
    if text not in RANGE_MODES:
        raise UsageError(f"invalid range mode: {text} (expected one of {', '.join(RANGE_MODES)})")
    if text == "all":
        return RangeSpec(label=RANGE_ALL_LABEL, kind="all")
    return RangeSpec(label="common", kind="common")


def _resolve_endpoint(point: Endpoint, max_step: int, default: int) -> int:
    if point.kind == "open":
        return default
    if point.kind == "absolute":
        return max_step - point.value if point.negative else point.value
    ratio = (100 - point.value) if point.negative else point.value
    scaled = Fraction(max_step) * ratio / 100
    return math.floor(scaled) if default else math.ceil(scaled)


def resolve_range(spec: RangeSpec, space, spaces: dict, tag: str, run_name: str):
    """相対指定は Run × step 座標系ごとに absolute bounds へ解決する。"""
    if spec.kind == "all":
        return None, None

    if spec.needs_space() and space[0] == UNKNOWN:
        raise RuntimeFailure(
            f"cannot resolve range {spec.label} for tag {tag} in run {run_name}: "
            f"step coordinate space is unknown"
        )

    info = spaces.get(space)
    max_step = info["max_step"] if info else 0

    if spec.kind == "common":
        if info is None or info["common_low"] is None:
            return 0, 0
        return info["common_low"], info["common_high"]

    # 末尾相対が原点を下回る場合だけ 0 へ切り上げる。
    # 上端は clamp しない（絶対指定が観測範囲外なら空として扱うべきで、末尾1点へ化けさせない）。
    lower = max(0, _resolve_endpoint(spec.start, max_step, 0))
    upper = max(0, _resolve_endpoint(spec.end, max_step, max_step))
    return lower, upper


# ---------------------------------------------------------------------------
# 統計と間引き系列
# ---------------------------------------------------------------------------


def range_bounds(entry: TagSeries, lower: int | None, upper: int | None) -> tuple[int, int]:
    """step は非減少なので二分探索で range の序数範囲を求める。"""
    start = 0 if lower is None else bisect.bisect_left(entry.steps, lower)
    end = len(entry.steps) if upper is None else bisect.bisect_right(entry.steps, upper)
    return start, max(start, end)


EMPTY_STATS = {
    "status": "empty",
    "count": 0,
    "mean": None,
    "population_std": None,
    "min": None,
    "max": None,
    "first": None,
    "first_step": None,
    "last": None,
    "last_step": None,
    "min_step": None,
    "max_step": None,
}


def compute_stats(entry: TagSeries, start: int, end: int) -> dict:
    count = end - start
    if count <= 0:
        return dict(EMPTY_STATS)

    # 全点 list を統計専用に複製せず、保持済み配列を online accumulator で 1 走査する。
    values = entry.values
    mean = 0.0
    m2 = 0.0
    seen = 0
    minimum = values[start]
    maximum = values[start]
    for index in range(start, end):
        value = values[index]
        if value < minimum:
            minimum = value
        if value > maximum:
            maximum = value
        seen += 1
        delta = value - mean
        mean += delta / seen
        m2 += delta * (value - mean)

    return {
        "status": "ok",
        "count": count,
        "mean": mean,
        "population_std": math.sqrt(m2 / count),
        "min": minimum,
        "max": maximum,
        "first": values[start],
        "first_step": entry.steps[start],
        "last": values[end - 1],
        "last_step": entry.steps[end - 1],
        "min_step": entry.steps[start],
        "max_step": entry.steps[end - 1],
    }


def build_series(entry: TagSeries, start: int, end: int) -> list:
    count = end - start
    if count <= 0:
        return []
    if count <= SERIES_MAX_POINTS:
        return [[entry.steps[i], entry.values[i]] for i in range(start, end)]

    # 序数を等分し、各 bucket の最小値点・最大値点・末尾点を採る。
    values = entry.values
    chosen: set[int] = {start, end - 1}
    for bucket in range(SERIES_BUCKETS):
        low = start + (bucket * count) // SERIES_BUCKETS
        high = start + ((bucket + 1) * count) // SERIES_BUCKETS
        if high <= low:
            continue
        min_index = low
        max_index = low
        for index in range(low + 1, high):
            # 同値候補では序数が小さい点を残す。
            if values[index] < values[min_index]:
                min_index = index
            if values[index] > values[max_index]:
                max_index = index
        chosen.add(min_index)
        chosen.add(max_index)
        chosen.add(high - 1)

    return [[entry.steps[i], entry.values[i]] for i in sorted(chosen)]


# ---------------------------------------------------------------------------
# 実行本体
# ---------------------------------------------------------------------------


def dedupe(values: list[str]) -> list[str]:
    """最初の出現位置を残して重複を除去する。"""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def run_envelope(subcommand: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "subcommand": subcommand,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "runs": [],
        "warnings": [],
    }


def run_node(resolved: ResolvedRun) -> dict:
    return {
        "input": resolved.input,
        "run_name": resolved.run_name,
        "workspace": resolved.workspace,
        "run_dir": str(resolved.run_dir),
    }


@dataclass
class RunContext:
    """1 Run 分の共通入力。artifact、cache 判定、metrics 定義をまとめて持つ。"""

    resolved: ResolvedRun
    master_path: Path | None
    snapshot: tuple | None
    artifacts: dict
    cache_status: CacheStatus
    config_entries: list = field(default_factory=list)
    defs: dict = field(default_factory=dict)
    def_source: str | None = None
    cache_tags: dict = field(default_factory=dict)
    observed: dict = field(default_factory=dict)
    selected_source: str | None = None
    warnings: list = field(default_factory=list)

    def metric_def(self, tag: str) -> MetricDef:
        return self.defs.get(tag, MetricDef())

    def source_node(self) -> dict:
        return {
            "selected": self.selected_source,
            "master_path": self.artifacts["master"]["path"],
            "cache_path": self.artifacts["cache"]["path"]
            if self.artifacts["cache"]["exists"]
            else None,
            "cache_status": self.cache_status.status,
            "cache_reason": self.cache_status.reason,
            "provisional": False,
            "source_changed_during_read": False,
        }


def open_run(resolved: ResolvedRun, want_config: bool) -> RunContext:
    """Metrics マスタを open せずに、artifact と cache 判定まで進める。"""
    master_path = resolve_run_metrics(resolved.run_dir)
    artifacts = inspect_artifacts(resolved.run_dir, master_path)
    snapshot = _source_snapshot(master_path) if master_path is not None else None
    cache_status = validate_cache(resolved.run_dir / CACHE_NAME, master_path, snapshot)
    artifacts["cache"].update(
        status=cache_status.status,
        reason=cache_status.reason,
        source_meta=cache_status.source_meta,
    )

    config_path = resolved.run_dir / CONFIG_REL_PATH
    config_entries = (
        read_config_entries(config_path) if want_config and config_path.is_file() else []
    )
    return RunContext(
        resolved=resolved,
        master_path=master_path,
        snapshot=snapshot,
        artifacts=artifacts,
        cache_status=cache_status,
        config_entries=config_entries,
    )


def apply_config_fallback(context: RunContext) -> None:
    """scalar 定義レコードが無い Run は解決済み config から導出する（互換経路）。"""
    context.defs = metric_defs_from_config(context.config_entries)
    context.def_source = "config_derived"
    context.warnings.append(
        "scalar metric definitions record is absent; metric definitions were derived from "
        "config/config_data.txt (def_source=config_derived)"
    )


def load_cache_inventory(context: RunContext, want_observed: bool) -> None:
    inventory = read_cache_inventory(context.resolved.run_dir / CACHE_NAME, want_observed)
    context.cache_tags = inventory.tags
    if want_observed:
        context.observed = {
            tag: entry["observed"] for tag, entry in inventory.tags.items() if entry["observed"]
        }
    if inventory.defs:
        context.defs = inventory.defs
        context.def_source = "metrics_defs"


def load_definitions_cheap(context: RunContext) -> None:
    """Metrics マスタを開かずに定義だけ揃える。glob 展開の候補集合にも使う。"""
    if context.cache_status.is_current():
        context.selected_source = "cache"
        load_cache_inventory(context, want_observed=True)
        if not context.defs:
            apply_config_fallback(context)
        return
    apply_config_fallback(context)


def load_definitions(context: RunContext, want_observed: bool) -> None:
    """cache が current ならそこから、そうでなければ必要に応じて master を 1 pass 読む。"""
    if context.cache_status.is_current():
        context.selected_source = "cache"
        load_cache_inventory(context, want_observed)
        if not context.defs:
            apply_config_fallback(context)
        return

    if want_observed and context.master_path is not None:
        context.selected_source = "master"
        byte_limit = (
            None if context.master_path.name == GZIP_NAME else context.snapshot[0]
        )
        try:
            scan = scan_master(context.master_path, None, byte_limit)
        except SourceError as exc:
            context.warnings.append(f"source_error: {exc}")
            apply_config_fallback(context)
            return
        context.observed = scan.observed
        if scan.defs:
            context.defs = scan.defs
            context.def_source = "metrics_defs"
        else:
            apply_config_fallback(context)
        return

    apply_config_fallback(context)


def command_runs(args) -> tuple[dict, int]:
    resolved_runs = (
        [resolve_run(value) for value in args.runs]
        if args.runs
        else list_runs(args.workspace)
    )

    result = run_envelope("runs")
    for resolved in resolved_runs:
        context = open_run(resolved, want_config=False)
        node = run_node(resolved)
        node["artifacts"] = context.artifacts
        node["warnings"] = context.warnings
        result["runs"].append(node)
    return result, EXIT_OK


def expand_metric_selectors(context: RunContext, selectors: list[str]) -> list[str]:
    """glob は既知 tag 集合へ展開する。完全一致は既知集合が無くても通す。"""
    known = sorted(set(context.defs) | set(context.cache_tags) | set(context.observed))
    tags: list[str] = []
    for selector in selectors:
        if not has_glob_meta(selector):
            tags.append(selector)
            continue
        pattern = compile_selector(selector)
        tags.extend(tag for tag in known if pattern.match(tag))
    return dedupe(tags)


def load_metric_series(context: RunContext, tags: list[str], warnings: list) -> dict:
    """cache が current ならそこから、そうでなければ master を 1 pass だけ読む。"""
    if context.cache_status.is_current():
        context.selected_source = "cache"
        return read_cache_series(context.resolved.run_dir / CACHE_NAME, tags, context.cache_tags)

    if context.master_path is None:
        context.selected_source = None
        return {}

    context.selected_source = "master"
    if context.cache_status.status != "absent":
        warnings.append(
            f"{context.resolved.run_name}: metrics cache not usable "
            f"({context.cache_status.status}: {context.cache_status.reason}), "
            f"falling back to the metrics master"
        )

    byte_limit = None if context.master_path.name == GZIP_NAME else context.snapshot[0]
    scan = scan_master(context.master_path, tags, byte_limit)
    context.observed = scan.observed
    if scan.defs:
        context.defs = scan.defs
        context.def_source = "metrics_defs"
        context.warnings = [
            item for item in context.warnings if "scalar metric definitions record is absent" not in item
        ]
    if scan.trailing_bytes:
        context.artifacts["master"]["provisional"] = True
        warnings.append(
            f"{context.resolved.run_name}: dropped an unterminated trailing line "
            f"({scan.trailing_bytes} bytes)"
        )
    return scan.series


def collect_spaces(tags, defs, series) -> dict:
    """選択 tag を step 座標系ごとに束ね、観測 step の範囲を取る。"""
    spaces: dict = {}
    for tag in tags:
        entry = series.get(tag)
        if entry is None or not entry.steps:
            continue
        space = defs.get(tag, MetricDef()).space()
        low, high = entry.steps[0], entry.steps[-1]
        current = spaces.get(space)
        spaces[space] = (
            (min(current[0], low), max(current[1], high)) if current else (low, high)
        )
    return spaces


def intersect_spaces(per_run_spaces: list) -> dict:
    common: dict = {}
    for spaces in per_run_spaces:
        for space, (low, high) in spaces.items():
            current = common.get(space)
            common[space] = (
                (max(current[0], low), min(current[1], high)) if current else (low, high)
            )
    return common


def build_range_specs(args) -> list:
    specs = [parse_range(text) for text in dedupe(args.range)]
    specs.extend(parse_range_mode(text) for text in dedupe(args.range_mode))
    return specs or [RangeSpec(label=RANGE_ALL_LABEL, kind="all")]


def build_comparison(runs: list, range_specs: list, stat: str) -> list:
    """行=tag、列=Run の比較データ。md はこれを転記するだけにする。"""
    tag_order: list[str] = []
    for run in runs:
        for metric in run["metrics"]:
            if metric["tag"] not in tag_order:
                tag_order.append(metric["tag"])

    comparison = []
    for index, spec in enumerate(range_specs):
        rows = []
        for tag in tag_order:
            values = []
            for run in runs:
                metric = next((item for item in run["metrics"] if item["tag"] == tag), None)
                entry = (
                    metric["ranges"][index]
                    if metric and index < len(metric["ranges"])
                    else None
                )
                if metric is None:
                    status, value = "missing", None
                elif metric["status"] != "ok":
                    status, value = metric["status"], None
                elif entry is None or entry["status"] != "ok":
                    status, value = (entry["status"] if entry else "missing"), None
                else:
                    status, value = "ok", entry.get(stat)
                values.append({"run": run["run_name"], "value": value, "status": status})

            row = {"tag": tag, "values": values}
            numbers = [item["value"] for item in values if isinstance(item["value"], (int, float))]
            if len(values) == 2:
                base, other = values[0]["value"], values[1]["value"]
                delta = other - base if base is not None and other is not None else None
                row["delta"] = delta
                row["delta_ratio"] = delta / base if delta is not None and base else None
            elif len(values) >= 3 and len(numbers) >= 2:
                mean = sum(numbers) / len(numbers)
                variance = sum((item - mean) ** 2 for item in numbers) / len(numbers)
                row["mean"] = mean
                row["population_std"] = math.sqrt(variance)
                row["range"] = max(numbers) - min(numbers)
            rows.append(row)
        comparison.append({"range": spec.label, "stat": stat, "rows": rows})
    return comparison


def command_metrics(args) -> tuple[dict, int]:
    range_specs = build_range_specs(args)
    selectors = dedupe(args.metric)

    result = run_envelope("metrics")
    result["ranges"] = [{"label": spec.label, "kind": spec.kind} for spec in range_specs]
    result["comparison"] = []

    loaded = []
    for value in args.runs:
        context = open_run(resolve_run(value), want_config=True)
        load_definitions_cheap(context)
        if not expand_metric_selectors(context, selectors) and any(
            has_glob_meta(item) for item in selectors
        ):
            # 既知 tag が無いまま glob を解けないので、inventory のためだけに 1 pass 追加する。
            if context.master_path is not None:
                result["warnings"].append(
                    f"{context.resolved.run_name}: no known tag list was available, "
                    f"scanning the metrics master once to expand the metric globs"
                )
                load_definitions(context, want_observed=True)
        tags = expand_metric_selectors(context, selectors)

        source_error: str | None = None
        try:
            series = load_metric_series(context, tags, result["warnings"])
        except SourceError as exc:
            source_error = str(exc)
            series = {}
            context.warnings.append(f"source_error: {exc}")
        loaded.append((context, tags, series, source_error))

    per_run_spaces = [
        collect_spaces(tags, context.defs, series) for context, tags, series, _ in loaded
    ]
    common = intersect_spaces(per_run_spaces)

    for (context, tags, series, source_error), spaces in zip(loaded, per_run_spaces):
        space_info = {
            space: {
                "min_step": low,
                "max_step": high,
                "common_low": common[space][0],
                "common_high": common[space][1],
            }
            for space, (low, high) in spaces.items()
        }

        node = run_node(context.resolved)
        node["def_source"] = context.def_source
        node["metrics_source"] = context.source_node()
        node["metrics"] = []
        if context.snapshot is not None and context.master_path is not None:
            if _source_snapshot(context.master_path) != context.snapshot:
                node["metrics_source"]["source_changed_during_read"] = True
                node["metrics_source"]["provisional"] = True
                result["warnings"].append(
                    f"{context.resolved.run_name}: metrics master changed during read"
                )
        if context.artifacts["master"].get("provisional"):
            node["metrics_source"]["provisional"] = True

        for tag in tags:
            definition = context.metric_def(tag)
            entry = series.get(tag, TagSeries())
            if source_error is not None:
                status = "source_error"
            elif context.selected_source is None:
                status = "source_missing"
            elif not entry.present:
                status = "missing"
            elif entry.quarantined:
                status = "quarantined"
            else:
                status = "ok"

            observed = context.observed.get(tag)
            if observed is None and entry.steps:
                observed = {
                    "count": len(entry.steps),
                    "min_step": entry.steps[0],
                    "max_step": entry.steps[-1],
                }

            metric_node = {
                "tag": tag,
                "step_axis": definition.step_axis,
                "runner": definition.runner,
                "def_source": context.def_source,
                "source_key": definition.source_key,
                "status": status,
                "excluded": entry.excluded,
                "observed": observed,
                "ranges": [],
            }
            for spec in range_specs:
                lower, upper = resolve_range(
                    spec, definition.space(), space_info, tag, context.resolved.run_name
                )
                start, end = range_bounds(entry, lower, upper)
                if lower is not None and upper is not None and lower > upper:
                    start = end = 0
                    result["warnings"].append(
                        f"{context.resolved.run_name}/{tag}: range {spec.label} resolved to an "
                        f"empty span ({lower} > {upper})"
                    )
                range_node = {"label": spec.label, "kind": spec.kind, "start": lower, "end": upper}
                range_node.update(compute_stats(entry, start, end))
                if args.series:
                    range_node["series"] = build_series(entry, start, end)
                metric_node["ranges"].append(range_node)
            node["metrics"].append(metric_node)

        node["warnings"] = context.warnings
        result["runs"].append(node)

    result["comparison"] = build_comparison(result["runs"], range_specs, args.stat)

    exit_code = EXIT_OK
    if selectors and not any(
        metric["status"] in ("ok", "quarantined")
        for run in result["runs"]
        for metric in run["metrics"]
    ):
        exit_code = EXIT_RUNTIME
    return result, exit_code


def command_config(args) -> tuple[dict, int]:
    selectors = dedupe(args.config_key)
    result = run_envelope("config")
    result["config_diff"] = []

    contexts = [open_run(resolve_run(value), want_config=True) for value in args.runs]
    for context in contexts:
        effective_keys = read_effective_keys(context.resolved.run_dir)
        context.observed = effective_keys  # 突合用に保持する
        selector_results, values = select_config_values(
            context.config_entries, selectors, effective_keys, args.effective_only
        )
        node = run_node(context.resolved)
        node["config"] = {"selectors": selector_results, "values": values}
        node["warnings"] = context.warnings
        result["runs"].append(node)

    if args.diff:
        result["config_diff"] = build_config_diff(
            [run["run_name"] for run in result["runs"]],
            [context.config_entries for context in contexts],
            selectors,
            [context.observed for context in contexts],
            args.effective_only,
        )

    exit_code = EXIT_OK
    if selectors and not any(
        item["status"] == "ok" for run in result["runs"] for item in run["config"]["selectors"]
    ):
        exit_code = EXIT_RUNTIME
    return result, exit_code


def read_resolution_document(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeFailure(f"Failed to read resolution artifact: {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise RuntimeFailure(f"Invalid resolution artifact: {path}: expected object")
    return document


def validate_resolution_payload(path: Path, payload: dict) -> None:
    selections = payload.get("selections")
    references = payload.get("references")
    if not isinstance(selections, list):
        raise RuntimeFailure(f"Invalid resolution payload: {path}: expected selections array")
    if not isinstance(references, list):
        raise RuntimeFailure(f"Invalid resolution payload: {path}: expected references array")
    for selection in selections:
        if not isinstance(selection, dict) or not isinstance(selection.get("chain"), list):
            raise RuntimeFailure(
                f"Invalid resolution payload: {path}: expected selection object with chain array"
            )
        if not isinstance(selection.get("key"), str):
            raise RuntimeFailure(
                f"Invalid resolution payload: {path}: expected selection key string"
            )
        if not all(isinstance(item, dict) for item in selection["chain"]):
            raise RuntimeFailure(
                f"Invalid resolution payload: {path}: expected selection chain objects"
            )
    if not all(isinstance(item, dict) for item in references):
        raise RuntimeFailure(
            f"Invalid resolution payload: {path}: expected reference objects"
        )


def read_resolution_artifact(run_dir: Path) -> tuple[Path | None, dict | None]:
    path = run_dir / RESOLUTION_JSON_REL_PATH
    if path.is_file():
        document = read_resolution_document(path)
        if document.get("type") != "json" or document.get("tag") != "config_resolution":
            raise RuntimeFailure(
                f"Invalid resolution envelope: {path}: expected type=json tag=config_resolution"
            )
        payload = document.get("data")
        if not isinstance(payload, dict):
            raise RuntimeFailure(f"Invalid resolution envelope: {path}: expected object data")
        validate_resolution_payload(path, payload)
        return RESOLUTION_JSON_REL_PATH, payload

    path = run_dir / LEGACY_RESOLUTION_REL_PATH
    if path.is_file():
        payload = read_resolution_document(path)
        validate_resolution_payload(path, payload)
        return LEGACY_RESOLUTION_REL_PATH, payload

    return None, None


def command_resolution(args) -> tuple[dict, int]:
    result = run_envelope("resolution")
    exit_code = EXIT_OK
    for value in args.runs:
        resolved = resolve_run(value)
        try:
            source_path, payload = read_resolution_artifact(resolved.run_dir)
        except RuntimeFailure as exc:
            source_path = (
                RESOLUTION_JSON_REL_PATH
                if (resolved.run_dir / RESOLUTION_JSON_REL_PATH).is_file()
                else LEGACY_RESOLUTION_REL_PATH
            )
            node = run_node(resolved)
            node["resolution"] = {
                "status": "source_error",
                "source": str(source_path).replace("\\", "/"),
                "schema_version": None,
                "trunk": None,
                "selections": [],
                "references": [],
            }
            node["warnings"] = [str(exc)]
            result["runs"].append(node)
            exit_code = EXIT_RUNTIME
            continue

        if payload is None:
            node = run_node(resolved)
            node["resolution"] = {
                "status": "missing",
                "source": None,
                "schema_version": None,
                "trunk": None,
                "selections": [],
                "references": [],
            }
            node["warnings"] = []
            result["runs"].append(node)
            continue

        selections = payload.get("selections", [])
        schema_version = payload.get("schema_version")
        warnings = []
        if schema_version != RESOLUTION_SCHEMA_VERSION:
            warnings.append(f"unsupported resolution schema_version={schema_version}")
        node = run_node(resolved)
        node["resolution"] = {
            "status": "ok",
            "source": str(source_path).replace("\\", "/"),
            "schema_version": schema_version,
            "trunk": selections[0]
            if selections and selections[0].get("key") == "run.$"
            else None,
            "selections": selections,
            "references": payload.get("references", []),
        }
        node["warnings"] = warnings
        result["runs"].append(node)
    return result, exit_code


def command_tags(args) -> tuple[dict, int]:
    want_observed = not args.no_observed
    result = run_envelope("tags")
    for value in args.runs:
        context = open_run(resolve_run(value), want_config=True)
        load_definitions(context, want_observed)

        node = run_node(context.resolved)
        node["def_source"] = context.def_source
        node["metrics_source"] = context.source_node()
        node["tags"] = []
        for tag in sorted(set(context.defs) | set(context.observed)):
            definition = context.metric_def(tag)
            cache_entry = context.cache_tags.get(tag)
            entry = definition.to_json()
            entry["tag"] = tag
            entry["status"] = (
                "quarantined" if cache_entry and cache_entry["status"] == "error" else "ok"
            )
            entry["observed"] = context.observed.get(tag) if want_observed else None
            node["tags"].append(entry)
        node["warnings"] = context.warnings
        result["runs"].append(node)
    return result, EXIT_OK


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------


def render_json(result: dict) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _compact_number(value) -> str:
    """短く書くための最短往復表現。整数値の末尾 .0 だけ落とす。"""
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _compact_number(value)
    return str(value)


def _table(lines: list, header: list, rows: list) -> None:
    if not rows:
        lines.append("(none)")
        lines.append("")
        return
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(_cell(cell) for cell in row) + " |")
    lines.append("")


def _md_header(result: dict, title: str) -> list:
    return [
        f"# {title}",
        "",
        f"- schema_version: {result['schema_version']}",
        f"- generated_at: {result['generated_at']}",
        "",
    ]


def _md_warnings(result: dict) -> list:
    lines = ["## Warnings", ""]
    warnings = list(result["warnings"])
    for run in result["runs"]:
        warnings.extend(f"{run['run_name']}: {item}" for item in run.get("warnings", []))
    lines.extend([f"- {item}" for item in warnings] or ["(none)"])
    lines.append("")
    return lines


def render_runs_markdown(result: dict) -> str:
    lines = _md_header(result, "Runs")
    _table(
        lines,
        ["run", "workspace", "master", "kind", "size", "cache", "cache_reason", "run_dir"],
        [
            [
                run["run_name"],
                run["workspace"],
                run["artifacts"]["master"]["path"],
                run["artifacts"]["master"]["kind"],
                run["artifacts"]["master"]["size"],
                run["artifacts"]["cache"]["status"],
                run["artifacts"]["cache"]["reason"],
                run["run_dir"],
            ]
            for run in result["runs"]
        ],
    )
    lines.append("## Files")
    lines.append("")
    _table(
        lines,
        ["run", "name", "size", "mtime"],
        [
            [run["run_name"], item["name"], item["size"], item["mtime"]]
            for run in result["runs"]
            for item in run["artifacts"]["files"]
        ],
    )
    lines.extend(_md_warnings(result))
    return "\n".join(lines) + "\n"


def render_tags_markdown(result: dict) -> str:
    lines = _md_header(result, "Metric tags")
    for run in result["runs"]:
        lines.append(f"## {run['run_name']} (def_source: {_cell(run['def_source'])})")
        lines.append("")
        _table(
            lines,
            ["tag", "step_axis", "runner", "event", "target", "source_key",
             "ema_alpha", "interval", "clip", "scope", "eval_name", "eval_episodes", "num_envs",
             "status", "count", "min_step", "max_step"],
            [
                [
                    item["tag"], item["step_axis"], item["runner"], item["event"],
                    item["target"], item["source_key"], item["ema_alpha"], item["interval"],
                    item["clip"], item["scope"], item["eval_name"], item["eval_episodes"], item["num_envs"],
                    item["status"],
                    (item["observed"] or {}).get("count"),
                    (item["observed"] or {}).get("min_step"),
                    (item["observed"] or {}).get("max_step"),
                ]
                for item in run["tags"]
            ],
        )
    lines.extend(_md_warnings(result))
    return "\n".join(lines) + "\n"


def render_config_markdown(result: dict) -> str:
    lines = _md_header(result, "Effective config")
    _table(
        lines,
        ["run", "selector", "status", "matched"],
        [
            [run["run_name"], item["selector"], item["status"], item["matched"]]
            for run in result["runs"]
            for item in run["config"]["selectors"]
        ],
    )
    _table(
        lines,
        ["run", "key", "value", "effective"],
        [
            [run["run_name"], item["key"], item["value"], item["effective"]]
            for run in result["runs"]
            for item in run["config"]["values"]
        ],
    )
    lines.append("## Diff")
    lines.append("")
    _table(
        lines,
        ["key"] + [run["run_name"] for run in result["runs"]],
        [
            [item["key"]]
            + [entry["value"] if entry["present"] else "(absent)" for entry in item["runs"]]
            for item in result["config_diff"]
        ],
    )
    lines.extend(_md_warnings(result))
    return "\n".join(lines) + "\n"


def render_resolution_markdown(result: dict) -> str:
    lines = _md_header(result, "Config resolution")
    for run in result["runs"]:
        resolution = run["resolution"]
        lines.append(f"## {run['run_name']}")
        lines.append("")
        lines.append(f"- status: {_cell(resolution['status'])}")
        lines.append(f"- source: {_cell(resolution['source'])}")
        lines.append(f"- resolution_schema_version: {_cell(resolution['schema_version'])}")
        trunk = resolution["trunk"]
        if trunk is not None:
            trunk_terms = " > ".join(
                f"{_cell(item.get('term'))} => {_cell(item.get('resolved'))}"
                for item in trunk.get("chain", [])
            )
            lines.append(f"- trunk: {trunk_terms or '(empty)'}")
        lines.append("")

        lines.append("### Selections")
        lines.append("")
        _table(
            lines,
            ["key", "term", "resolved"],
            [
                [selection.get("key"), item.get("term"), item.get("resolved")]
                for selection in resolution["selections"]
                for item in selection.get("chain", [])
            ],
        )

        lines.append("### References")
        lines.append("")
        _table(
            lines,
            ["source", "target", "value"],
            [
                [item.get("source"), item.get("target"), item.get("value")]
                for item in resolution["references"]
            ],
        )

    lines.extend(_md_warnings(result))
    return "\n".join(lines) + "\n"


def render_metrics_markdown(result: dict) -> str:
    lines = _md_header(result, "Metrics")
    lines.append(f"- ranges: {', '.join(item['label'] for item in result['ranges'])}")
    lines.append("")

    _table(
        lines,
        ["run", "def_source", "selected", "cache", "provisional"],
        [
            [
                run["run_name"], run["def_source"], run["metrics_source"]["selected"],
                run["metrics_source"]["cache_status"], run["metrics_source"]["provisional"],
            ]
            for run in result["runs"]
        ],
    )

    # 3. rangeごとの比較表（行=tag、列=Run）
    run_names = [run["run_name"] for run in result["runs"]]
    for block in result["comparison"]:
        lines.append(f"## Comparison {block['range']} (stat: {block['stat']})")
        lines.append("")
        extra = []
        if len(run_names) == 2:
            extra = ["delta", "delta_ratio"]
        elif len(run_names) >= 3:
            extra = ["mean", "population_std", "range"]
        rows = []
        for row in block["rows"]:
            cells = [row["tag"]]
            for item in row["values"]:
                cells.append(item["value"] if item["status"] == "ok" else item["status"])
            cells.extend(row.get(name) for name in extra)
            rows.append(cells)
        _table(lines, ["tag"] + run_names + extra, rows)

    # 4. 詳細表
    lines.append("## Detail")
    lines.append("")
    _table(
        lines,
        ["run", "tag", "step_axis", "runner", "source_key", "status", "range", "range_status",
         "start", "end", "count", "mean", "population_std", "min", "max", "first", "last",
         "min_step", "max_step", "observed_min", "observed_max", "observed_count", "excluded"],
        [
            [
                run["run_name"], metric["tag"], metric["step_axis"], metric["runner"],
                metric["source_key"], metric["status"], entry["label"], entry["status"],
                entry["start"], entry["end"], entry["count"], entry["mean"],
                entry["population_std"], entry["min"], entry["max"], entry["first"],
                entry["last"], entry["min_step"], entry["max_step"],
                (metric["observed"] or {}).get("min_step"),
                (metric["observed"] or {}).get("max_step"),
                (metric["observed"] or {}).get("count"),
                metric["excluded"],
            ]
            for run in result["runs"]
            for metric in run["metrics"]
            for entry in metric["ranges"]
        ],
    )

    # 5. series は --series 指定時だけ
    series_lines = [
        f"- {run['run_name']} / {metric['tag']} / {entry['label']}: "
        + ", ".join(f"{step}:{_compact_number(value)}" for step, value in entry["series"])
        for run in result["runs"]
        for metric in run["metrics"]
        for entry in metric["ranges"]
        if entry.get("series")
    ]
    if series_lines:
        lines.append("## Series")
        lines.append("")
        lines.extend(series_lines)
        lines.append("")

    # 6. warning。missing と empty もここへ出す。
    lines.append("## Warnings")
    lines.append("")
    warnings = list(result["warnings"])
    for run in result["runs"]:
        warnings.extend(f"{run['run_name']}: {item}" for item in run.get("warnings", []))
        for metric in run["metrics"]:
            if metric["status"] != "ok":
                warnings.append(f"{run['run_name']}: {metric['tag']} is {metric['status']}")
            for entry in metric["ranges"]:
                if entry["status"] != "ok":
                    warnings.append(
                        f"{run['run_name']}: {metric['tag']} has no point in {entry['label']}"
                    )
    lines.extend([f"- {item}" for item in warnings] or ["(none)"])
    lines.append("")
    return "\n".join(lines) + "\n"


MARKDOWN_RENDERERS = {
    "runs": render_runs_markdown,
    "tags": render_tags_markdown,
    "config": render_config_markdown,
    "resolution": render_resolution_markdown,
    "metrics": render_metrics_markdown,
}


def render_output(result: dict, output_format: str) -> str:
    if output_format != "md":
        return render_json(result)
    renderer = MARKDOWN_RENDERERS.get(result["subcommand"])
    return renderer(result)


def write_output(text: str, target: Path | None, stdout) -> None:
    """途中失敗で既存 file を壊さないよう、同じ directory の一時 file 経由で置換する。"""
    if target is None:
        stdout.write(text)
        return

    temporary: Path | None = None
    try:
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=target.name + ".",
            suffix=".tmp",
            delete=False,
        )
        temporary = Path(handle.name)
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise RuntimeFailure(f"Failed to write output: {target}: {exc}") from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


DESCRIPTION = "Inspect and extract anet-lab run artifacts without modifying them."

RUN_HELP = (
    "run name, or an existing relative/absolute directory path. "
    "A bare name is looked up as apps/runner/workspaces/*/runs/<RUN>; "
    "legacy layouts under apps/runner/runs_* must be given as a path."
)

EXIT_HELP = (
    "exit codes: 0 = ok (per-run gaps are reported as missing), "
    "1 = runtime failure or the requested selectors matched nothing in any run, "
    "2 = argument/range syntax error, run not found, or ambiguous run name."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inspect_run.py", description=DESCRIPTION, epilog=EXIT_HELP)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    runs = subparsers.add_parser(
        "runs",
        help="list runs and report artifact, metrics master and cache state",
        description="List runs and report artifact, metrics master and cache state.",
        epilog=(
            "examples:\n"
            "  inspect_run.py runs\n"
            "  inspect_run.py runs --workspace dm-iqn\n"
            "  inspect_run.py runs run_a run_b --format md"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    runs.add_argument("runs", metavar="RUN", nargs="*", help=RUN_HELP)
    runs.add_argument("--workspace", metavar="WS", help="limit the listing to this workspace")
    _add_common_options(runs)

    tags = subparsers.add_parser(
        "tags",
        help="list metric tags with their resolved definition and observed step range",
        description=(
            "List metric tags with their resolved definition and observed step range. "
            "The step coordinate space is the (runner, step_axis) pair: the same axis name "
            "on a different runner is a different coordinate space."
        ),
        epilog=(
            "examples:\n"
            "  inspect_run.py tags run_a\n"
            "  inspect_run.py tags run_a --no-observed\n"
            "  inspect_run.py tags run_a run_b --format md"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    tags.add_argument("runs", metavar="RUN", nargs="+", help=RUN_HELP)
    tags.add_argument(
        "--no-observed",
        action="store_true",
        help="report declared definitions only and never scan the metrics master",
    )
    _add_common_options(tags)

    config = subparsers.add_parser(
        "config",
        help="extract effective config keys and diff them between runs",
        description=(
            "Extract effective config keys and diff them between runs. "
            "config_data.txt also carries merge-source and unselected definition namespaces; "
            "keys confirmed by a config/<module>.txt dump are marked effective=true, and keys "
            "that cannot be confirmed are left null rather than called false."
        ),
        epilog=(
            "examples:\n"
            "  inspect_run.py config run_a --config-key 'agent.*'\n"
            "  inspect_run.py config run_a --config-key 'train.eval.[eval1].run_mode'\n"
            "  inspect_run.py config run_a run_b --diff\n"
            "  inspect_run.py config run_a --config-key '*tau*' --effective-only"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    config.add_argument("runs", metavar="RUN", nargs="+", help=RUN_HELP)
    config.add_argument(
        "--config-key",
        metavar="KEY_OR_GLOB",
        action="append",
        default=[],
        help="effective config key or case-sensitive glob (repeatable). "
        "Only * and ? are glob meta characters; [ and ] are literal.",
    )
    config.add_argument(
        "--diff", action="store_true", help="report config keys that differ between runs"
    )
    config.add_argument(
        "--effective-only",
        action="store_true",
        help="keep only keys confirmed by a config/<module>.txt dump",
    )
    _add_common_options(config)

    resolution = subparsers.add_parser(
        "resolution",
        help="show config selection and value-reference resolution",
        description="Show config selections, named trunk and value references for each Run.",
        epilog=(
            "examples:\n"
            "  inspect_run.py resolution run_a\n"
            "  inspect_run.py resolution run_a run_b --format md"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    resolution.add_argument("runs", metavar="RUN", nargs="+", help=RUN_HELP)
    _add_common_options(resolution)

    metrics = subparsers.add_parser(
        "metrics",
        help="extract scalar metrics, aggregate them over ranges and compare runs",
        description=(
            "Extract scalar metrics, aggregate them over ranges and compare runs. "
            "Ranges are inclusive at both ends, so overlapping ranges count a boundary "
            "point twice. Relative ranges resolve per step coordinate space, which is the "
            "(runner, step_axis) pair."
        ),
        epilog=(
            "range forms (--range, repeatable):\n"
            "  10M:20M    absolute steps, K/M/G suffix allowed, both ends inclusive\n"
            "  10%%:20%%    percentage of the last observed step in that coordinate space\n"
            "  :20M       open start (0)          10M:   open end (last observed step)\n"
            "  -4M:       last 4M steps           -10%%:  last 10 percent\n"
            "\n"
            "examples:\n"
            "  inspect_run.py metrics run_a --metric '42_env/*' --range -4M:\n"
            "  inspect_run.py metrics run_a run_b --metric t/a --range-mode common\n"
            "  inspect_run.py metrics run_a --metric t/a --series --format md"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    metrics.add_argument("runs", metavar="RUN", nargs="+", help=RUN_HELP)
    metrics.add_argument(
        "--metric",
        metavar="TAG_OR_GLOB",
        action="append",
        default=[],
        help="scalar tag or glob (repeatable). Only * and ? are glob meta characters.",
    )
    metrics.add_argument(
        "--range",
        metavar="SPEC",
        action="append",
        default=[],
        help="aggregation range START:END (repeatable). See the range forms below.",
    )
    metrics.add_argument(
        "--range-mode",
        metavar="MODE",
        action="append",
        default=[],
        help="derived range (repeatable): all = every observed point, "
        "common = intersection across runs within the same coordinate space",
    )
    metrics.add_argument(
        "--stat",
        choices=("mean", "last", "first", "min", "max", "count", "population_std"),
        default="mean",
        help="statistic used in the comparison table (default: mean)",
    )
    metrics.add_argument(
        "--series", action="store_true", help="include the downsampled series (max 128 points)"
    )
    _add_common_options(metrics)

    return parser


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format", choices=("json", "md"), default="json", help="output format (default: json)"
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the result to a file instead of stdout (parent directory must exist)",
    )


COMMANDS = {
    "runs": command_runs,
    "tags": command_tags,
    "config": command_config,
    "resolution": command_resolution,
    "metrics": command_metrics,
}


def main(argv=None, stdout=None, stderr=None) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    with redirect_stdout(out), redirect_stderr(err):
        parser = build_parser()
        try:
            args = parser.parse_args(sys.argv[1:] if argv is None else argv)
        except SystemExit as exc:
            return EXIT_USAGE if exc.code else EXIT_OK

        try:
            target = Path(args.output).resolve() if args.output else None
            if target is not None and not target.parent.is_dir():
                raise UsageError(f"output parent directory does not exist: {target.parent}")

            result, exit_code = COMMANDS[args.subcommand](args)
            write_output(render_output(result, args.format), target, out)
        except UsageError as exc:
            print(f"error: {exc}", file=err)
            return EXIT_USAGE
        except RuntimeFailure as exc:
            print(f"error: {exc}", file=err)
            return EXIT_RUNTIME

        for warning in result["warnings"]:
            print(f"warning: {warning}", file=err)
        for run in result["runs"]:
            for warning in run.get("warnings", []):
                print(f"warning: {run['run_name']}: {warning}", file=err)
        return exit_code


if __name__ == "__main__":
    sys.exit(main())

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


SCHEMA_VERSION = 1

CACHE_NAME = "metrics_cache.db"
RAW_KIND = "jsonl"
GZIP_KIND = "jsonl.gz"

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACES_ROOT = REPO_ROOT / "apps" / "runner" / "workspaces"

CONFIG_REL_PATH = Path("config") / "config_data.txt"

SERIES_MAX_POINTS = 128
SERIES_BUCKETS = 42

METRIC_KEY_PREFIX = "metrics.scalar.["
STEP_AXIS_NAMES = frozenset(
    ["train_step", "learn_step", "episode_step", "exp_step", "update_step", "sim_step"]
)
STEP_AXIS_UNKNOWN = "unknown"

# JSONL の step は Metrics Viewer と同じく MAX_SAFE_INTEGER までを有効とする。
MAX_SAFE_STEP = 9007199254740991

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2


class UsageError(Exception):
    """引数、profile契約、Run解決の失敗。終了値2に対応する。"""


class RuntimeFailure(Exception):
    """source read、query、出力書込み等の実行時失敗。終了値1に対応する。"""


class SourceError(Exception):
    """Metricsマスタの構造違反。Run単位のsource errorとして扱う。"""


# ---------------------------------------------------------------------------
# Run 解決
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedRun:
    input: str
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
        return ResolvedRun(
            input=value,
            run_dir=run_dir,
            run_name=run_dir.name,
            workspace=_workspace_of(run_dir),
        )

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
    return ResolvedRun(
        input=value,
        run_dir=run_dir,
        run_name=run_dir.name,
        workspace=_workspace_of(run_dir),
    )


# ---------------------------------------------------------------------------
# 実効 config
# ---------------------------------------------------------------------------


def read_config_entries(config_path: Path) -> list[tuple[str, str]]:
    """config_data.txt を flat key の出現順で読む。値は型変換しない。"""
    try:
        text = config_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise RuntimeFailure(f"Failed to read config: {config_path}: {exc}") from exc

    entries: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        if not separator:
            continue
        entries.append((key.strip(), value.strip()))
    return entries


def compile_selector(selector: str) -> re.Pattern:
    """`*` と `?` だけを glob meta とする。

    実効 config の key は `train.eval.[eval1].run_mode` のように `[tag]` 記法が偏在するため、
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


def select_config_values(entries: list[tuple[str, str]], selectors: list[str]):
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
            values.append({"key": key, "value": by_key[key]})
    return selector_results, values


def build_config_diff(run_names: list[str], run_entries: list[list], selectors: list[str]) -> list:
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
# Run 解析プロファイル
# ---------------------------------------------------------------------------


PROFILE_FIELDS = ("version", "name", "metrics", "config_keys", "windows")


@dataclass(frozen=True)
class Profile:
    path: Path
    name: str
    metrics: list
    config_keys: list
    windows: list


def _profile_string_array(payload: dict, field_name: str) -> list:
    value = payload[field_name]
    if not isinstance(value, list):
        raise UsageError(f"invalid profile: {field_name} must be an array of strings")
    for item in value:
        if not isinstance(item, str) or not item:
            raise UsageError(
                f"invalid profile: {field_name} must contain non-empty strings, got {item!r}"
            )
    return list(value)


def load_profile(path: Path) -> Profile:
    """schema v1 だけを受理する。include、継承、built-in profile 名は設けない。"""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise UsageError(f"cannot read profile: {path}: {exc}") from exc
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise UsageError(f"invalid profile JSON: {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise UsageError(f"invalid profile: {path} must contain a JSON object")

    missing = [name for name in PROFILE_FIELDS if name not in payload]
    if missing:
        raise UsageError(f"invalid profile: missing required fields: {', '.join(missing)}")
    unknown = [name for name in payload if name not in PROFILE_FIELDS]
    if unknown:
        raise UsageError(f"invalid profile: unknown fields: {', '.join(sorted(unknown))}")

    version = payload["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise UsageError(f"invalid profile: version must be the integer 1, got {version!r}")

    name = payload["name"]
    if not isinstance(name, str) or not name.strip():
        raise UsageError("invalid profile: name must be a non-blank string")

    metrics = _profile_string_array(payload, "metrics")
    config_keys = _profile_string_array(payload, "config_keys")
    windows = _profile_string_array(payload, "windows")
    if not metrics and not config_keys:
        raise UsageError("invalid profile: at least one of metrics or config_keys must be non-empty")

    return Profile(
        path=path.resolve(),
        name=name,
        metrics=metrics,
        config_keys=config_keys,
        windows=windows,
    )


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
    cache_info: dict = {"path": str(cache_path), "exists": False, "size": None, "mtime": None}
    if cache_path.is_file():
        stat_result = cache_path.stat()
        cache_info.update(exists=True, size=stat_result.st_size, mtime=_mtime_iso(stat_result))

    # log と snapshot は名前で拾い、directory は辿らない。
    files = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_file():
            continue
        if child.suffix not in (".log", ".anet"):
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
# step 軸
# ---------------------------------------------------------------------------


def _step_axis_from_definition(definition: str) -> str:
    """metrics.scalar 定義の token 列から step 軸を決める。token 順序は自由。"""
    axis: str | None = None
    event: str | None = None
    for token in definition.split():
        if token.startswith("$"):
            name = token[1:]
            if name in STEP_AXIS_NAMES:
                axis = name
            continue
        if token.startswith("@"):
            event = token[1:]
            continue
        head, separator, tail = token.partition(":")
        if not separator or not tail:
            continue
        if head in ("step", "step_axis") and tail in STEP_AXIS_NAMES:
            axis = tail
        elif head == "event":
            event = tail

    # 明示指定を最優先し、無い場合だけ event 既定へ落とす。
    if axis is not None:
        return axis
    if event is None or event == "train":
        return "train_step"
    if event in ("learn", "episode_end"):
        return "exp_step"
    return STEP_AXIS_UNKNOWN


def resolve_step_axes(config_entries: list[tuple[str, str]]) -> dict[str, str]:
    """解決済みの metrics.scalar.[<tag>] だけを読む。未適用の生キーは対象外。"""
    axes: dict[str, str] = {}
    for key, value in config_entries:
        if not key.startswith(METRIC_KEY_PREFIX) or not key.endswith("]"):
            continue
        tag = key[len(METRIC_KEY_PREFIX) : -1]
        axes[tag] = _step_axis_from_definition(value)
    return axes


# ---------------------------------------------------------------------------
# Metrics キャッシュ
# ---------------------------------------------------------------------------


CACHE_APPLICATION_ID = 0x414E4554
CACHE_SCHEMA_VERSION = 1
FINGERPRINT_BYTES = 64 * 1024

# Metrics Viewer schema v1 の必須 table と、そこで読む必須 column。
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
                return CacheStatus(
                    "invalid", f"required_column_missing: {table}.{missing[0]}"
                )

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
            code = meta.get("error_code", "unknown")
            return CacheStatus("error", f"cache ingest failed: {code}", meta)
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


def read_cache_series(cache_path: Path, tags: list[str]) -> dict:
    """1 read transaction で tags と L0 scalars だけを読む。統計は LOD から復元しない。"""
    series = {tag: TagSeries() for tag in tags}
    try:
        connection = open_cache_readonly(cache_path)
    except sqlite3.Error as exc:
        raise RuntimeFailure(f"Failed to open metrics cache: {cache_path}: {exc}") from exc

    try:
        connection.execute("BEGIN")
        known = {
            key: (tag_id, status)
            for tag_id, key, status in connection.execute("SELECT id, key, status FROM tags")
        }
        for tag in tags:
            found = known.get(tag)
            if found is None:
                continue
            tag_id, status = found
            entry = series[tag]
            for step, value in connection.execute(
                "SELECT step, value FROM scalars WHERE tag_id = ? ORDER BY ordinal", (tag_id,)
            ):
                entry.steps.append(step)
                entry.values.append(value)
            # cache 経路でも tags.status == error は quarantined として扱う。
            entry.quarantined = status == "error"
        connection.execute("ROLLBACK")
    except sqlite3.Error as exc:
        raise RuntimeFailure(f"Metrics cache query failed: {cache_path}: {exc}") from exc
    finally:
        connection.close()
    return series


# ---------------------------------------------------------------------------
# Metrics マスタ走査
# ---------------------------------------------------------------------------


@dataclass
class TagSeries:
    steps: array = field(default_factory=lambda: array("q"))
    values: array = field(default_factory=lambda: array("d"))
    excluded: int = 0
    quarantined: bool = False


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
    if not math.isfinite(value):
        return None
    if not _float32_representable(value):
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


def _parse_scalar_record(line: bytes, wanted: set[str]):
    """1行を解析し、選択tagのscalarなら (tag, step, value) を返す。"""
    if not line.strip():
        raise SourceError("empty line in metrics master")
    try:
        record = json.loads(line)
    except ValueError as exc:
        raise SourceError(f"invalid JSON line: {exc}") from exc
    if not isinstance(record, dict) or "type" not in record:
        raise SourceError("invalid record: missing type")
    if record["type"] != "scalar":
        return None

    tag = record.get("tag")
    if not isinstance(tag, str) or "value" not in record:
        raise SourceError("invalid scalar record: missing tag or value")
    step = record.get("step")
    if isinstance(step, bool) or not isinstance(step, int):
        raise SourceError(f"invalid step for tag {tag}")
    if step < 0 or step > MAX_SAFE_STEP:
        raise SourceError(f"invalid step for tag {tag}: {step}")
    if tag not in wanted:
        return None
    return tag, step, record["value"]


def read_master_series(master_path: Path, tags: list[str], byte_limit: int | None):
    """必要な全 tag を 1 pass で読む。tag ごと・window ごとに開き直さない。"""
    series = {tag: TagSeries() for tag in tags}
    wanted = set(tags)
    try:
        handle = open_metrics_binary(master_path)
    except OSError as exc:
        raise RuntimeFailure(f"Failed to open metrics master: {master_path}: {exc}") from exc

    is_gzip = master_path.name == GZIP_NAME
    with handle:
        reader = MasterLineReader(handle, byte_limit)
        try:
            for line in reader.lines():
                parsed = _parse_scalar_record(line, wanted)
                if parsed is None:
                    continue
                tag, step, raw_value = parsed
                entry = series[tag]
                if entry.quarantined:
                    continue
                value = _numeric_value(raw_value)
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
            # gzip の破損は immutable source の構造違反として Run 単位の source error にする。
            raise SourceError(f"failed to read metrics master: {exc}") from exc

    # gzip は immutable source なので、未終端行を追記待ちにせず source error とする。
    if is_gzip and reader.trailing_bytes:
        raise SourceError("gzip ended with an unterminated JSON line")

    return series, reader.trailing_bytes


# ---------------------------------------------------------------------------
# window
# ---------------------------------------------------------------------------


WINDOW_ALL_LABEL = "all"
_ABSOLUTE_RE = re.compile(r"\A(\d+)([kKmMgG]?)\Z")
_PERCENT_RE = re.compile(r"\A(\d+(?:\.\d+)?|\.\d+)%\Z")
_SUFFIX_SCALE = {"": 1, "k": 1_000, "m": 1_000_000, "g": 1_000_000_000}


@dataclass(frozen=True)
class WindowSpec:
    label: str
    kind: str
    start: object = None
    end: object = None


def _parse_absolute_endpoint(text: str, window: str) -> int:
    matched = _ABSOLUTE_RE.match(text)
    if not matched:
        raise UsageError(
            f"invalid window: {window} (absolute endpoint must be a non-negative integer "
            f"with an optional K/M/G suffix, got {text})"
        )
    value = int(matched.group(1)) * _SUFFIX_SCALE[matched.group(2).lower()]
    if value > MAX_SAFE_STEP:
        raise UsageError(f"invalid window: {window} (endpoint {text} exceeds {MAX_SAFE_STEP})")
    return value


def _parse_percent_endpoint(text: str, window: str) -> Fraction:
    matched = _PERCENT_RE.match(text)
    if not matched:
        raise UsageError(
            f"invalid window: {window} (percentage endpoint must be a decimal in 0..100, "
            f"got {text})"
        )
    value = Fraction(matched.group(1))
    if value < 0 or value > 100:
        raise UsageError(f"invalid window: {window} (percentage endpoint {text} is out of 0..100)")
    return value


def parse_window(text: str) -> WindowSpec:
    left, separator, right = text.partition(":")
    if not separator or not left.strip() or not right.strip():
        raise UsageError(f"invalid window: {text} (expected START:END)")
    left = left.strip()
    right = right.strip()

    left_percent = left.endswith("%")
    right_percent = right.endswith("%")
    if left_percent != right_percent:
        raise UsageError(
            f"invalid window: {text} (absolute and percentage endpoints must not be mixed)"
        )

    if left_percent:
        start = _parse_percent_endpoint(left, text)
        end = _parse_percent_endpoint(right, text)
        kind = "percentage"
    else:
        start = _parse_absolute_endpoint(left, text)
        end = _parse_absolute_endpoint(right, text)
        kind = "absolute"

    if start > end:
        raise UsageError(f"invalid window: {text} (START must not exceed END)")
    return WindowSpec(label=text, kind=kind, start=start, end=end)


def resolve_window_bounds(spec: WindowSpec, axis: str, axis_max: dict, tag: str, run_name: str):
    """percentage は Run × step 軸の到達 step を基準に absolute bounds へ解決する。"""
    if spec.kind == WINDOW_ALL_LABEL:
        return None, None
    if spec.kind == "absolute":
        return spec.start, spec.end
    if axis == STEP_AXIS_UNKNOWN:
        raise RuntimeFailure(
            f"cannot apply percentage window {spec.label} to tag {tag} in run {run_name}: "
            f"step axis is unknown"
        )
    max_step = axis_max.get(axis, 0)
    lower = math.ceil(max_step * spec.start / 100)
    upper = math.floor(max_step * spec.end / 100)
    return lower, upper


# ---------------------------------------------------------------------------
# 統計と間引き系列
# ---------------------------------------------------------------------------


def window_bounds(entry: TagSeries, lower: int | None, upper: int | None) -> tuple[int, int]:
    """step は非減少なので二分探索で window の序数範囲を求める。"""
    start = 0 if lower is None else bisect.bisect_left(entry.steps, lower)
    end = len(entry.steps) if upper is None else bisect.bisect_right(entry.steps, upper)
    return start, max(start, end)


def compute_stats(entry: TagSeries, start: int, end: int) -> dict:
    count = end - start
    if count <= 0:
        return {
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


@dataclass
class Selection:
    """1実行分の抽出条件。profile と CLI option を統合した結果を持つ。"""

    metrics: list[str] = field(default_factory=list)
    config_keys: list[str] = field(default_factory=list)
    diff_config: bool = False
    windows: list[WindowSpec] = field(
        default_factory=lambda: [WindowSpec(label=WINDOW_ALL_LABEL, kind=WINDOW_ALL_LABEL)]
    )

    def wants_config(self) -> bool:
        return bool(self.metrics or self.config_keys or self.diff_config)


def inspect_run(resolved: ResolvedRun, selection: Selection, warnings: list[str]):
    metrics = selection.metrics
    config_path = resolved.run_dir / CONFIG_REL_PATH
    # 軽量 inspection では config の値 dump を行わない。
    config_entries = (
        read_config_entries(config_path)
        if selection.wants_config() and config_path.is_file()
        else []
    )
    step_axes = resolve_step_axes(config_entries)

    selector_results, selected_values = select_config_values(config_entries, selection.config_keys)

    # source 選択は is_file 判定だけなので、metric 未要求でも master を open しない。
    master_path = resolve_run_metrics(resolved.run_dir)
    artifacts = inspect_artifacts(resolved.run_dir, master_path)

    # cache 判定は開始時 snapshot と比較する。master が無い Run でも状態だけは記録する。
    start_snapshot = _source_snapshot(master_path) if master_path is not None else None
    cache_status = validate_cache(
        resolved.run_dir / CACHE_NAME, master_path, start_snapshot
    )
    artifacts["cache"]["status"] = cache_status.status
    artifacts["cache"]["reason"] = cache_status.reason
    artifacts["cache"]["source_meta"] = cache_status.source_meta

    run_result = {
        "input": resolved.input,
        "run_name": resolved.run_name,
        "workspace": resolved.workspace,
        "run_dir": str(resolved.run_dir),
        "artifacts": artifacts,
        "config": {"selectors": selector_results, "values": selected_values},
        "metrics_source": {
            "selected": None,
            "master_path": artifacts["master"]["path"],
            "cache_path": artifacts["cache"]["path"] if artifacts["cache"]["exists"] else None,
            "cache_status": cache_status.status,
            "cache_reason": cache_status.reason,
            "provisional": False,
            "source_changed_during_read": False,
        },
        "metrics": [],
        "warnings": [],
    }

    if not metrics:
        return run_result, config_entries

    if cache_status.is_current():
        # 完全に current な cache だけを read-only 利用し、master は走査しない。
        run_result["metrics_source"]["selected"] = "cache"
        series = read_cache_series(resolved.run_dir / CACHE_NAME, metrics)
        _append_metric_results(run_result, metrics, series, step_axes, selection, resolved)
        return run_result, config_entries

    if cache_status.status != "absent":
        warnings.append(
            f"{resolved.run_name}: metrics cache not usable ({cache_status.status}: "
            f"{cache_status.reason}), falling back to the metrics master"
        )

    if master_path is None:
        run_result["metrics"] = [
            {"tag": tag, "step_axis": step_axes.get(tag, STEP_AXIS_UNKNOWN),
             "status": "source_missing", "excluded": 0, "windows": []}
            for tag in metrics
        ]
        return run_result, config_entries

    run_result["metrics_source"]["selected"] = "master"

    # 実行中 Run の追記を同じ結果へ混ぜないよう、開始時 size を raw の読み取り上限にする。
    byte_limit = None if master_path.name == GZIP_NAME else start_snapshot[0]

    try:
        series, trailing = read_master_series(master_path, metrics, byte_limit)
    except SourceError as exc:
        run_result["warnings"].append(f"source_error: {exc}")
        run_result["metrics"] = [
            {"tag": tag, "step_axis": step_axes.get(tag, STEP_AXIS_UNKNOWN),
             "status": "source_error", "excluded": 0, "windows": []}
            for tag in metrics
        ]
        return run_result, config_entries

    if trailing:
        run_result["metrics_source"]["provisional"] = True
        warnings.append(
            f"{resolved.run_name}: dropped an unterminated trailing line ({trailing} bytes)"
        )

    # 読み取り後に source が動いていたら暫定であることを示す。自動 retry はしない。
    if _source_snapshot(master_path) != start_snapshot:
        run_result["metrics_source"]["source_changed_during_read"] = True
        run_result["metrics_source"]["provisional"] = True
        warnings.append(
            f"{resolved.run_name}: metrics master changed during read ({master_path})"
        )

    _append_metric_results(run_result, metrics, series, step_axes, selection, resolved)
    return run_result, config_entries


def _append_metric_results(run_result, metrics, series, step_axes, selection, resolved) -> None:
    """cache 経路と master 経路で status、統計、系列の意味を揃える。"""
    # percentage window の 100% は、同じ step 軸へ解決された選択 tag 群の最大到達 step とする。
    axis_max: dict[str, int] = {}
    for tag in metrics:
        entry = series[tag]
        if not entry.steps:
            continue
        axis = step_axes.get(tag, STEP_AXIS_UNKNOWN)
        axis_max[axis] = max(axis_max.get(axis, 0), entry.steps[-1])

    for tag in metrics:
        entry = series[tag]
        axis = step_axes.get(tag, STEP_AXIS_UNKNOWN)
        if entry.quarantined:
            status = "quarantined"
        elif not entry.steps and entry.excluded == 0:
            status = "missing"
        else:
            status = "ok"

        windows = []
        for spec in selection.windows:
            lower, upper = resolve_window_bounds(spec, axis, axis_max, tag, resolved.run_name)
            start, end = window_bounds(entry, lower, upper)
            window = {"label": spec.label, "kind": spec.kind, "start": lower, "end": upper}
            window.update(compute_stats(entry, start, end))
            window["series"] = build_series(entry, start, end)
            windows.append(window)

        run_result["metrics"].append(
            {
                "tag": tag,
                "step_axis": axis,
                "status": status,
                "excluded": entry.excluded,
                "windows": windows,
            }
        )


def build_result(run_inputs: list[str], selection: Selection) -> tuple[dict, int]:
    warnings: list[str] = []
    resolved_runs = [resolve_run(value) for value in run_inputs]

    runs = []
    run_entries = []
    for resolved in resolved_runs:
        run_result, config_entries = inspect_run(resolved, selection, warnings)
        runs.append(run_result)
        run_entries.append(config_entries)

    config_diff = []
    if selection.diff_config:
        config_diff = build_config_diff(
            [run["run_name"] for run in runs], run_entries, selection.config_keys
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "profile": {"path": None, "name": None},
        "windows": [{"label": spec.label, "kind": spec.kind} for spec in selection.windows],
        "runs": runs,
        "config_diff": config_diff,
        "warnings": warnings,
    }

    # 明示した metric / config selector が全 Run で1件も成立しなければ、result を出して失敗にする。
    # quarantined は有効 prefix を返せているので成立扱いとする。
    exit_code = EXIT_OK
    if selection.metrics and not any(
        metric["status"] in ("ok", "quarantined") for run in runs for metric in run["metrics"]
    ):
        exit_code = EXIT_RUNTIME
    if selection.config_keys and not any(
        item["status"] == "ok" for run in runs for item in run["config"]["selectors"]
    ):
        exit_code = EXIT_RUNTIME
    return result, exit_code


def build_selection(profile: Profile | None, args) -> Selection:
    """profile を土台に CLI option を重ねる。array は末尾追加、window は全置換。"""
    profile_metrics = profile.metrics if profile else []
    profile_config_keys = profile.config_keys if profile else []
    profile_windows = profile.windows if profile else []

    window_texts = dedupe(args.window) if args.window else dedupe(profile_windows)
    windows = (
        [parse_window(text) for text in window_texts]
        if window_texts
        else [WindowSpec(label=WINDOW_ALL_LABEL, kind=WINDOW_ALL_LABEL)]
    )

    return Selection(
        metrics=dedupe(profile_metrics + args.metric),
        config_keys=dedupe(profile_config_keys + args.config_key),
        diff_config=args.diff_config,
        windows=windows,
    )


def render_json(result: dict) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _compact_number(value) -> str:
    """系列を短く書くための最短往復表現。整数値の末尾 .0 だけ落とす。"""
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


def render_markdown(result: dict) -> str:
    """JSON と同じ result model から生成する。Markdown 固有の解析判断は足さない。"""
    lines: list[str] = ["# Run inspection", ""]

    # 1. 実行条件と profile
    lines.append(f"- schema_version: {result['schema_version']}")
    lines.append(f"- generated_at: {result['generated_at']}")
    lines.append(f"- profile: {_cell(result['profile']['name'])} ({_cell(result['profile']['path'])})")
    lines.append(f"- windows: {', '.join(item['label'] for item in result['windows'])}")
    lines.append("")

    # 2. Run ごとの artifact・source・cache 状態
    lines.append("## Artifacts and sources")
    lines.append("")
    _table(
        lines,
        ["run", "workspace", "run_dir", "master", "kind", "cache", "selected", "provisional"],
        [
            [
                run["run_name"],
                run["workspace"],
                run["run_dir"],
                run["artifacts"]["master"]["path"],
                run["artifacts"]["master"]["kind"],
                run["artifacts"]["cache"].get("status"),
                run["metrics_source"]["selected"],
                run["metrics_source"]["provisional"],
            ]
            for run in result["runs"]
        ],
    )

    # 3. config selector 結果と config diff
    lines.append("## Config")
    lines.append("")
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
        ["run", "key", "value"],
        [
            [run["run_name"], item["key"], item["value"]]
            for run in result["runs"]
            for item in run["config"]["values"]
        ],
    )
    lines.append("### Config diff")
    lines.append("")
    diff_runs = [run["run_name"] for run in result["runs"]]
    _table(
        lines,
        ["key"] + diff_runs,
        [
            [item["key"]]
            + [entry["value"] if entry["present"] else "(absent)" for entry in item["runs"]]
            for item in result["config_diff"]
        ],
    )

    # 4. Run x tag x window の統計 table
    lines.append("## Metrics")
    lines.append("")
    _table(
        lines,
        ["run", "tag", "step_axis", "status", "window", "start", "end", "count", "mean",
         "population_std", "min", "max", "first", "last", "min_step", "max_step", "excluded"],
        [
            [
                run["run_name"], metric["tag"], metric["step_axis"], metric["status"],
                window["label"], window["start"], window["end"], window["count"],
                window["mean"], window["population_std"], window["min"], window["max"],
                window["first"], window["last"], window["min_step"], window["max_step"],
                metric["excluded"],
            ]
            for run in result["runs"]
            for metric in run["metrics"]
            for window in metric["windows"]
        ],
    )

    # 5. tag ごとの間引き series
    lines.append("## Series")
    lines.append("")
    series_lines = [
        f"- {run['run_name']} / {metric['tag']} / {window['label']}: "
        + ", ".join(f"{step}:{_compact_number(value)}" for step, value in window["series"])
        for run in result["runs"]
        for metric in run["metrics"]
        for window in metric["windows"]
        if window["series"]
    ]
    lines.extend(series_lines or ["(none)"])
    lines.append("")

    # 6. warning 一覧
    lines.append("## Warnings")
    lines.append("")
    warnings = list(result["warnings"])
    for run in result["runs"]:
        warnings.extend(f"{run['run_name']}: {item}" for item in run["warnings"])
    lines.extend([f"- {item}" for item in warnings] or ["(none)"])
    lines.append("")

    return "\n".join(lines) + "\n"


def write_output(text: str, target: Path | None, stdout) -> None:
    """途中失敗で既存 file を壊さないよう、同じ directory の一時 file 経由で置換する。"""
    if target is None:
        stdout.write(text)
        return

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
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeFailure(f"Failed to write output: {target}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inspect_run.py",
        description="Inspect and extract anet-lab run artifacts without modifying them.",
    )
    parser.add_argument("runs", metavar="RUN", nargs="+", help="run name or existing directory path")
    parser.add_argument(
        "--metric", metavar="TAG", action="append", default=[], help="scalar tag to extract"
    )
    parser.add_argument(
        "--config-key",
        metavar="KEY_OR_GLOB",
        action="append",
        default=[],
        help="effective config key or case-sensitive glob",
    )
    parser.add_argument(
        "--diff-config",
        action="store_true",
        help="report effective config keys that differ between runs",
    )
    parser.add_argument(
        "--window",
        metavar="RANGE",
        action="append",
        default=[],
        help="metric aggregation window, START:END or START%%:END%%",
    )
    parser.add_argument("--profile", metavar="PATH", help="run analysis profile JSON")
    parser.add_argument(
        "--format", choices=("json", "md"), default="json", help="output format (default: json)"
    )
    parser.add_argument("--output", metavar="PATH", help="write the result to a file")
    return parser


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

            profile = load_profile(Path(args.profile)) if args.profile else None
            selection = build_selection(profile, args)
            result, exit_code = build_result(args.runs, selection)
            result["profile"] = {
                "path": str(profile.path) if profile else None,
                "name": profile.name if profile else None,
            }

            text = render_markdown(result) if args.format == "md" else render_json(result)
            write_output(text, target, out)
        except UsageError as exc:
            print(f"error: {exc}", file=err)
            return EXIT_USAGE
        except RuntimeFailure as exc:
            print(f"error: {exc}", file=err)
            return EXIT_RUNTIME

        for warning in result["warnings"]:
            print(f"warning: {warning}", file=err)
        return exit_code


if __name__ == "__main__":
    sys.exit(main())

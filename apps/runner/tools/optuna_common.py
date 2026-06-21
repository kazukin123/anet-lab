#!/usr/bin/env python3
"""Optuna harness の ENV 非依存共通部品。

各 ENV 向け entrypoint はこの module を import し、探索空間や config 生成だけを
個別 script 側に置く。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable

OPTUNA_ARTIFACT_FILENAMES = (
    "manifest.json",
    "multiseed_summary.json",
    "multiseed_summary.csv",
    "seed_runs.json",
)
HARNESS_LOG_MAX_BYTES = 5 * 1024 * 1024
TRIAL_NAME_PATTERN = re.compile(r"^t(\d{5})$")

_ACTIVE_RUNNER_LOCK = threading.Lock()
_ACTIVE_RUNNERS: set[subprocess.Popen] = set()
_INTERRUPTING = threading.Event()


def clear_interrupt_flag() -> None:
    """新しい harness 実行を開始する前に中断フラグを戻す。"""
    _INTERRUPTING.clear()


def set_interrupt_flag() -> None:
    """中断経路に入ったことを runner 管理側へ知らせる。"""
    _INTERRUPTING.set()


class JapaneseArgumentParser(argparse.ArgumentParser):
    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "使い方:", 1)

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "使い方:", 1)


class JapaneseHelpFormatter(argparse.RawTextHelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = super()._get_help_string(action)
        if isinstance(action, argparse._HelpAction):
            return help_text

        notes: list[str] = []
        if self._is_required(action):
            notes.append("必須")
        if self._has_visible_default(action):
            notes.append(f"初期値: {self._format_default(action.default)}".replace("%", "%%"))

        if not notes:
            return help_text
        note_text = f"（{'、'.join(notes)}）"
        if not help_text:
            return note_text
        return f"{help_text}{note_text}"

    @staticmethod
    def _is_required(action: argparse.Action) -> bool:
        if action.__class__.__name__ == "_ChoicesPseudoAction":
            return False
        if getattr(action, "required", False):
            return True
        if action.option_strings:
            return False
        return action.nargs not in (argparse.OPTIONAL, argparse.ZERO_OR_MORE)

    @staticmethod
    def _has_visible_default(action: argparse.Action) -> bool:
        default = action.default
        return default is not None and default is not argparse.SUPPRESS

    @staticmethod
    def _format_default(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)


class TrialExecutionError(Exception):
    pass


class TrialPrunedError(TrialExecutionError):
    pass


class TrialFailedError(TrialExecutionError):
    pass


@dataclass(frozen=True)
class TrialContext:
    study_name: str
    trial_number: int
    trial_name: str
    budget_name: str
    cost_budget: float
    cost_tf: float
    token_count: int
    run_name: str
    runs_dir: str
    artifact_dir: str
    run_dir: str
    config_path: str


@dataclass(frozen=True)
class TrialExecutionResult:
    ctx: TrialContext
    summary: dict

    @property
    def score(self) -> float:
        return float(self.summary["score"])


@dataclass(frozen=True)
class ScoreWindow:
    start: int
    end: int
    raw_start: int | str
    raw_end: int | str | None
    exp_exit_step: int


@dataclass(frozen=True)
class DuplicateParamsInfo:
    policy: str
    duplicate_count_before: int
    duplicate_index: int
    duplicate_params_max_runs: int
    duplicate_seed_stride: int
    base_seeds: list[int]
    effective_seeds: list[int]
    duplicate_matched_trials: list[int]
    pruned_by_duplicate: bool
    prune_reason: str | None


@dataclass(frozen=True)
class OptunaArtifactContext:
    base_path: Path
    artifact_store: object
    upload_artifact: object
    get_all_artifact_meta: object | None = None
    download_artifact: object | None = None


class HarnessLogger:
    def __init__(self, path: Path, max_bytes: int = HARNESS_LOG_MAX_BYTES):
        self.path = path
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    def log(self, level: str, event: str, *, console: bool = False, **fields: object) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        suffix = " ".join(f"{key}={format_log_value(value)}" for key, value in fields.items())
        line = f"{timestamp} [{level.upper()}] {event}"
        if suffix:
            line = f"{line} {suffix}"
        self._write_lines([line])
        if console:
            print(line, file=sys.stderr if level.upper() in ("WARN", "ERROR") else sys.stdout, flush=True)

    def exception(self, event: str, exc: BaseException, *, console: bool = False, **fields: object) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        suffix = " ".join(f"{key}={format_log_value(value)}" for key, value in fields.items())
        header = f"{timestamp} [ERROR] {event} error={format_log_value(str(exc))}"
        if suffix:
            header = f"{header} {suffix}"
        lines = [header, traceback.format_exc().rstrip()]
        self._write_lines(lines)
        if console:
            print(header, file=sys.stderr, flush=True)

    def _write_lines(self, lines: list[str]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed()
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                for line in lines:
                    stream.write(line)
                    stream.write("\n")

    def _rotate_if_needed(self) -> None:
        if not self.path.exists() or self.path.stat().st_size < self.max_bytes:
            return
        oldest = self.path.with_name(f"{self.path.name}.2")
        middle = self.path.with_name(f"{self.path.name}.1")
        if oldest.exists():
            oldest.unlink()
        if middle.exists():
            shutil.move(str(middle), str(oldest))
        shutil.move(str(self.path), str(middle))


def format_log_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Path):
        return json.dumps(str(value), ensure_ascii=False)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(str(value), ensure_ascii=False)


@dataclass(frozen=True)
class MetricsSpec:
    """環境ごとの metrics tag 定義を共通 summarizer に渡すための仕様。"""

    primary_tags: tuple[str, ...]
    supplemental_tags: tuple[str, ...] = ()
    late_score_windows: tuple[tuple[str, object, object], ...] = ()



def runner_root(repo_root: Path) -> Path:
    """runner project root を repo root から解決する。"""
    return repo_root / "apps" / "runner"


def default_main_config(repo_root: Path) -> str:
    """生成 config が include する既定 main config 名を返す。"""
    del repo_root
    return "_main.txt"


def resolve_runner_relative_path(repo_root: Path, path_text: str) -> Path:
    """runner root 基準の相対 path を絶対 path に解決する。"""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return runner_root(repo_root) / path


def parse_window_raw_value(raw_value: object, option_name: str) -> int | str:
    if isinstance(raw_value, int):
        return raw_value
    if raw_value is None:
        raise ValueError(f"Invalid {option_name}: value is required.")

    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"Invalid {option_name}: value must not be empty.")

    if value.endswith("%"):
        percent_text = value[:-1]
        if not percent_text or "%" in percent_text:
            raise ValueError(f"Invalid {option_name}: percent form must be like 80%. value={raw_value}")
        try:
            percent = float(percent_text)
        except ValueError as exc:
            raise ValueError(f"Invalid {option_name}: percent value must be numeric. value={raw_value}") from exc
        if percent < 0.0 or percent > 100.0:
            raise ValueError(f"Invalid {option_name}: percent must be between 0 and 100. value={raw_value}")
        return value

    if "%" in value:
        raise ValueError(f"Invalid {option_name}: percent marker is only allowed at the end. value={raw_value}")

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {option_name}: value must be step integer or percent. value={raw_value}") from exc


def resolve_window_value(raw_value: int | str, exp_exit_step: int) -> int:
    if isinstance(raw_value, str):
        percent = float(raw_value[:-1])
        return round(exp_exit_step * percent / 100.0)
    return exp_exit_step + raw_value if raw_value < 0 else raw_value


def resolve_score_window(args: argparse.Namespace) -> ScoreWindow:
    raw_start = parse_window_raw_value(args.window_start, "--window-start")
    raw_end = getattr(args, "window_end", None)
    exp_exit_step = int(args.exp_exit_step)
    raw_end_value = None if raw_end is None else parse_window_raw_value(raw_end, "--window-end")
    window_start = resolve_window_value(raw_start, exp_exit_step)
    window_end = exp_exit_step if raw_end_value is None else resolve_window_value(raw_end_value, exp_exit_step)

    if window_start < 0:
        raise ValueError(f"Invalid window_start: resolved value must be >= 0. value={window_start}")
    if window_end < 0:
        raise ValueError(f"Invalid window_end: resolved value must be >= 0. value={window_end}")
    if window_start > window_end:
        raise ValueError(f"Invalid score window: window_start={window_start} > window_end={window_end}")

    return ScoreWindow(
        start=window_start,
        end=window_end,
        raw_start=raw_start,
        raw_end=raw_end_value,
        exp_exit_step=exp_exit_step,
    )


def resolve_score_window_from_raw(raw_start: object, raw_end: object, exp_exit_step: int, label: str) -> ScoreWindow:
    raw_start_value = parse_window_raw_value(raw_start, f"{label}.window-start")
    raw_end_value = parse_window_raw_value(raw_end, f"{label}.window-end")
    window_start = resolve_window_value(raw_start_value, exp_exit_step)
    window_end = resolve_window_value(raw_end_value, exp_exit_step)
    if window_start < 0:
        raise ValueError(f"Invalid {label} window_start: resolved value must be >= 0. value={window_start}")
    if window_end < 0:
        raise ValueError(f"Invalid {label} window_end: resolved value must be >= 0. value={window_end}")
    if window_start > window_end:
        raise ValueError(f"Invalid {label} score window: window_start={window_start} > window_end={window_end}")
    return ScoreWindow(
        start=window_start,
        end=window_end,
        raw_start=raw_start_value,
        raw_end=raw_end_value,
        exp_exit_step=exp_exit_step,
    )



def harness_logger(args: argparse.Namespace) -> HarnessLogger | None:
    return getattr(args, "_harness_logger", None)


def log_harness(args: argparse.Namespace, level: str, event: str, *, console: bool = False, **fields: object) -> None:
    logger = harness_logger(args)
    if logger is not None:
        logger.log(level, event, console=console, **fields)
    elif console:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        suffix = " ".join(f"{key}={format_log_value(value)}" for key, value in fields.items())
        line = f"{timestamp} [{level.upper()}] {event}"
        if suffix:
            line = f"{line} {suffix}"
        print(line, file=sys.stderr if level.upper() in ("WARN", "ERROR") else sys.stdout, flush=True)


def log_harness_exception(
    args: argparse.Namespace,
    event: str,
    exc: BaseException,
    *,
    console: bool = False,
    **fields: object,
) -> None:
    logger = harness_logger(args)
    if logger is not None:
        logger.exception(event, exc, console=console, **fields)
    elif console:
        print(f"[ERROR] {event}: {exc}", file=sys.stderr, flush=True)



def storage_url_from_text(repo_root: Path, storage_text: str) -> str:
    sqlite_prefix = "sqlite:///"
    if storage_text.startswith(sqlite_prefix):
        storage_path = Path(storage_text[len(sqlite_prefix):])
    else:
        storage_path = Path(storage_text)

    if not storage_path.is_absolute():
        storage_path = runner_root(repo_root) / storage_path
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    return f"{sqlite_prefix}{storage_path.as_posix()}"


def storage_url_from_arg(args: argparse.Namespace) -> str:
    return storage_url_from_text(Path(args.repo_root).resolve(), str(args.storage))


def create_optuna_storage(
    optuna,
    storage_url: str,
    storage_timeout_sec: float,
    heartbeat_interval_sec: int | None = None,
    heartbeat_grace_period_sec: int | None = None,
):
    if storage_url.startswith("sqlite:///"):
        heartbeat_kwargs = {}
        if heartbeat_interval_sec is not None and heartbeat_interval_sec > 0:
            heartbeat_kwargs["heartbeat_interval"] = heartbeat_interval_sec
            heartbeat_kwargs["grace_period"] = heartbeat_grace_period_sec
        return optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={
                "connect_args": {
                    "timeout": storage_timeout_sec,
                },
            },
            **heartbeat_kwargs,
        )
    return storage_url


def resolve_optuna_artifact_dir(args: argparse.Namespace) -> Path:
    return resolve_runner_relative_path(Path(args.repo_root).resolve(), str(args.optuna_artifact_dir))


def resolve_artifact_dir_from_text(args: argparse.Namespace, path_text: str) -> Path:
    return resolve_runner_relative_path(Path(args.repo_root).resolve(), path_text)


def create_optuna_artifact_context(args: argparse.Namespace) -> OptunaArtifactContext | None:
    try:
        from optuna.artifacts import FileSystemArtifactStore
        from optuna.artifacts import upload_artifact
    except Exception as e:
        print(f"[WARN] Optuna artifacts are unavailable: {e}", file=sys.stderr)
        return None

    try:
        base_path = resolve_optuna_artifact_dir(args)
        base_path.mkdir(parents=True, exist_ok=True)
        return OptunaArtifactContext(
            base_path=base_path,
            artifact_store=FileSystemArtifactStore(base_path=str(base_path)),
            upload_artifact=upload_artifact,
        )
    except Exception as e:
        print(f"[WARN] Failed to initialize Optuna artifact store: {e}", file=sys.stderr)
        return None


def create_required_filesystem_artifact_context(optuna, base_path: Path, label: str) -> OptunaArtifactContext:
    try:
        from optuna.artifacts import FileSystemArtifactStore
        from optuna.artifacts import download_artifact
        from optuna.artifacts import get_all_artifact_meta
        from optuna.artifacts import upload_artifact
    except Exception as e:
        raise TrialExecutionError(f"Optuna artifacts are required for {label}: {e}") from e

    try:
        base_path.mkdir(parents=True, exist_ok=True)
        return OptunaArtifactContext(
            base_path=base_path,
            artifact_store=FileSystemArtifactStore(base_path=str(base_path)),
            upload_artifact=upload_artifact,
            get_all_artifact_meta=get_all_artifact_meta,
            download_artifact=download_artifact,
        )
    except Exception as e:
        raise TrialExecutionError(f"Failed to initialize {label} artifact store: path={base_path} error={e}") from e


def register_optuna_trial_artifacts(
    optuna_trial,
    artifact_context: OptunaArtifactContext | None,
    artifact_dir: Path,
) -> None:
    if optuna_trial is None or artifact_context is None:
        return

    for filename in OPTUNA_ARTIFACT_FILENAMES:
        path = artifact_dir / filename
        if not path.is_file():
            continue
        try:
            artifact_context.upload_artifact(
                artifact_store=artifact_context.artifact_store,
                file_path=str(path),
                study_or_trial=optuna_trial,
            )
        except Exception as e:
            print(f"[WARN] Failed to upload Optuna artifact: path={path} error={e}", file=sys.stderr)


class ArtifactStore:
    """Optuna Dashboard 用 artifact store の初期化と trial artifact 登録を担当する。"""

    @staticmethod
    def create_optional_context(args: argparse.Namespace) -> OptunaArtifactContext | None:
        """run-study 用に artifact store を初期化する。失敗しても study 実行は継続する。"""
        return create_optuna_artifact_context(args)

    @staticmethod
    def create_required_context(optuna, base_path: Path, label: str) -> OptunaArtifactContext:
        """summarize-study など artifact 必須処理用に store を初期化する。"""
        return create_required_filesystem_artifact_context(optuna, base_path, label)

    @staticmethod
    def register_trial_artifacts(optuna_trial, artifact_context: OptunaArtifactContext | None, artifact_dir: Path) -> None:
        """生成済み artifact ファイルを Optuna trial metadata として登録する。"""
        register_optuna_trial_artifacts(optuna_trial, artifact_context, artifact_dir)



def parse_seed_list(seed_text: str) -> list[int]:
    seeds: list[int] = []
    seen: set[int] = set()
    for raw_part in seed_text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            seed = int(part)
        except ValueError as e:
            raise ValueError(f"Invalid --seeds: seed must be an integer. value={part}") from e
        if seed < 0:
            raise ValueError(f"Invalid --seeds: seed must be >= 0. value={seed}")
        if seed in seen:
            raise ValueError(f"Invalid --seeds: duplicate seed is not allowed. value={seed}")
        seeds.append(seed)
        seen.add(seed)
    if not seeds:
        raise ValueError("Invalid --seeds: at least one seed is required.")
    return seeds


def append_cli_arg(parts: list[str], option_name: str, value: object) -> None:
    if value is None:
        return
    parts.extend([option_name, str(value)])


def quote_windows_cli_arg(value: object) -> str:
    text = str(value)
    if text == "":
        return '""'
    if not any(ch.isspace() or ch in '"&|<>^()%!' for ch in text):
        return text

    result = '"'
    backslashes = 0
    for ch in text:
        if ch == "\\":
            backslashes += 1
            continue
        if ch == '"':
            result += "\\" * (backslashes * 2 + 1)
            result += '"'
            backslashes = 0
            continue
        result += "\\" * backslashes
        result += ch
        backslashes = 0
    result += "\\" * (backslashes * 2)
    result += '"'
    return result


def format_cli_args(parts: list[str]) -> str:
    return " ".join(quote_windows_cli_arg(part) for part in parts)



def set_study_user_attrs(study, attrs: dict) -> None:
    for key, value in attrs.items():
        study.set_user_attr(key, value)


def seed_trial_name(trial_name: str, seed: int) -> str:
    return f"{trial_name}_s{seed}"


def args_with_seed(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    seed_args = argparse.Namespace(**vars(args))
    seed_args.seed = seed
    return seed_args


def trial_state_name(trial) -> str:
    return str(getattr(getattr(trial, "state", None), "name", getattr(trial, "state", "")))


def study_trials(study) -> list:
    get_trials = getattr(study, "get_trials", None)
    if get_trials is not None:
        return list(get_trials(deepcopy=False))
    return list(getattr(study, "trials", []))


def running_trials(study) -> list:
    return [trial for trial in study_trials(study) if trial_state_name(trial) == "RUNNING"]


def set_trial_failure_attrs(trial, exc: BaseException, traceback_text: str | None = None) -> None:
    if trial is None:
        return
    try:
        trial.set_user_attr("failure_type", type(exc).__name__)
        trial.set_user_attr("failure_message", str(exc))
        if traceback_text:
            trial.set_user_attr("failure_traceback", traceback_text[-12000:])
    except Exception:
        pass


def mark_trial_failed(study, optuna, trial) -> None:
    fail_state = optuna.trial.TrialState.FAIL
    try:
        try:
            study.tell(trial.number, state=fail_state, skip_if_finished=True)
        except TypeError:
            study.tell(trial.number, state=fail_state)
        return
    except Exception as tell_error:
        storage = getattr(study, "_storage", None)
        trial_id = getattr(trial, "_trial_id", None)
        if storage is not None and trial_id is not None and hasattr(storage, "set_trial_state_values"):
            storage.set_trial_state_values(trial_id, fail_state, None)
            return
        raise tell_error


def cleanup_running_trials(study, optuna, dry_run: bool = False) -> dict:
    targets = running_trials(study)
    target_numbers = [int(trial.number) for trial in targets]
    cleaned: list[int] = []
    errors: list[dict] = []

    if not dry_run:
        for trial in targets:
            try:
                mark_trial_failed(study, optuna, trial)
                cleaned.append(int(trial.number))
            except Exception as e:
                errors.append({
                    "trial_number": int(trial.number),
                    "error": str(e),
                })

        try:
            set_study_user_attrs(study, {
                "cleaned_running_trials": cleaned,
                "cleaned_running_trials_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
        except Exception as e:
            errors.append({
                "trial_number": None,
                "error": f"failed to save cleanup study attrs: {e}",
            })

    return {
        "dry_run": dry_run,
        "target_running_trials": target_numbers,
        "cleaned_running_trials": cleaned,
        "errors": errors,
    }


def heartbeat_enabled(args: argparse.Namespace) -> bool:
    return int(getattr(args, "heartbeat_interval_sec", 0) or 0) > 0


def fail_stale_trials_if_enabled(args: argparse.Namespace, optuna, study, phase: str) -> dict:
    if not heartbeat_enabled(args):
        return {"enabled": False, "phase": phase}

    before = [int(trial.number) for trial in running_trials(study)]
    try:
        fail_stale_trials = getattr(optuna.storages, "fail_stale_trials", None)
        if fail_stale_trials is None:
            log_harness(args, "WARN", "stale-trials-unsupported", console=True, phase=phase)
            return {
                "enabled": True,
                "phase": phase,
                "before": before,
                "after": before,
                "error": "optuna.storages.fail_stale_trials is unavailable",
            }
        fail_stale_trials(study)
    except Exception as e:
        log_harness_exception(args, "stale-trials-check-failed", e, console=True, phase=phase)
        return {
            "enabled": True,
            "phase": phase,
            "before": before,
            "after": [int(trial.number) for trial in running_trials(study)],
            "error": str(e),
        }

    after = [int(trial.number) for trial in running_trials(study)]
    log_harness(args, "INFO", "stale-trials-checked", phase=phase, before=before, after=after)
    if after:
        log_harness(args, "WARN", "running-trials-remain", console=True, phase=phase, trials=after)
    return {
        "enabled": True,
        "phase": phase,
        "before": before,
        "after": after,
    }




def shifted_seeds(seeds: list[int], duplicate_index: int, stride: int) -> list[int]:
    offset = duplicate_index * stride
    return [seed + offset for seed in seeds]



def apply_duplicate_info_to_args(args: argparse.Namespace, duplicate_info: DuplicateParamsInfo) -> argparse.Namespace:
    info_args = argparse.Namespace(**vars(args))
    info_args.duplicate_count_before = duplicate_info.duplicate_count_before
    info_args.duplicate_index = duplicate_info.duplicate_index
    info_args.base_seeds = duplicate_info.base_seeds
    info_args.effective_seeds = duplicate_info.effective_seeds
    info_args.duplicate_matched_trials = duplicate_info.duplicate_matched_trials
    return info_args


def aggregate_score_stats(scores: list[float], mode: str) -> dict[str, float | int | None]:
    if not scores:
        return {
            "score_mean": None,
            "score_std": None,
            "score_median": None,
            "score_min": None,
            "score_max": None,
            "score_range": None,
            "score_mean_minus_std": None,
            "aggregate_score": None,
        }
    score_mean = float(mean(scores))
    score_std = float(pstdev(scores)) if len(scores) > 1 else 0.0
    score_min = float(min(scores))
    score_max = float(max(scores))
    stats = {
        "score_mean": score_mean,
        "score_std": score_std,
        "score_median": float(median(scores)),
        "score_min": score_min,
        "score_max": score_max,
        "score_range": score_max - score_min,
        "score_mean_minus_std": score_mean - score_std,
        "aggregate_score": None,
    }
    if mode == "mean":
        stats["aggregate_score"] = stats["score_mean"]
    elif mode == "median":
        stats["aggregate_score"] = stats["score_median"]
    elif mode == "mean-minus-std":
        stats["aggregate_score"] = stats["score_mean_minus_std"]
    elif mode == "min":
        stats["aggregate_score"] = stats["score_min"]
    else:
        raise ValueError(f"Invalid --score-aggregate: unknown mode. value={mode}")
    return stats


def completed_run_values(seed_runs: list[dict], field: str) -> list[float]:
    values: list[float] = []
    for run in seed_runs:
        if run.get("status") != "complete":
            continue
        value = run.get(field)
        if value is None:
            continue
        values.append(float(value))
    return values



def scalar_records(metrics_path: Path) -> Iterable[dict]:
    with metrics_path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "scalar":
                yield record


def summarize_metrics_window(
    metrics_path: Path,
    window: ScoreWindow,
    spec: MetricsSpec,
) -> tuple[dict[str, dict[str, float | int]], float | None]:
    # 指定 exp_step window 内の scalar record だけを集計する。
    values: dict[str, list[tuple[int, float]]] = {}
    for record in scalar_records(metrics_path):
        tag = record.get("tag")
        step = record.get("step")
        value = record.get("value")
        if tag is None or step is None or value is None:
            continue
        try:
            step_int = int(step)
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if step_int < window.start or step_int > window.end:
            continue
        values.setdefault(str(tag), []).append((step_int, value_float))

    tag_summary: dict[str, dict[str, float | int]] = {}
    for tag, points in values.items():
        point_values = [value for _, value in points if math.isfinite(value)]
        if not point_values:
            continue
        tag_summary[tag] = {
            "count": len(point_values),
            "mean": mean(point_values),
            "last": point_values[-1],
            "min_step": min(step for step, _ in points),
            "max_step": max(step for step, _ in points),
        }

    primary_means = [
        float(tag_summary[tag]["mean"])
        for tag in spec.primary_tags
        if tag in tag_summary
    ]
    score = mean(primary_means) if len(primary_means) == len(spec.primary_tags) else None
    return tag_summary, score


def summarize_metrics(metrics_path: Path, window: ScoreWindow, spec: MetricsSpec) -> dict:
    tag_summary, score = summarize_metrics_window(metrics_path, window, spec)
    analysis_windows: dict[str, dict[str, object]] = {}
    for name, raw_start, raw_end in spec.late_score_windows:
        late_window = resolve_score_window_from_raw(raw_start, raw_end, window.exp_exit_step, name)
        _, late_score = summarize_metrics_window(metrics_path, late_window, spec)
        analysis_windows[name] = {
            "window_start": late_window.start,
            "window_end": late_window.end,
            "window_start_raw": late_window.raw_start,
            "window_end_raw": late_window.raw_end,
            "exp_exit_step": late_window.exp_exit_step,
            "score": late_score,
        }

    score_60_80 = analysis_windows.get("score_60_80", {}).get("score")
    score_80_100 = analysis_windows.get("score_80_100", {}).get("score")
    late_slope = None
    if score_60_80 is not None and score_80_100 is not None:
        late_slope = float(score_80_100) - float(score_60_80)

    return {
        "metrics_path": str(metrics_path),
        "window_start": window.start,
        "window_end": window.end,
        "window_start_raw": window.raw_start,
        "window_end_raw": window.raw_end,
        "exp_exit_step": window.exp_exit_step,
        "score": score,
        "score_60_80": score_60_80,
        "score_80_100": score_80_100,
        "late_slope": late_slope,
        "analysis_windows": analysis_windows,
        "tags": tag_summary,
    }


def write_summary_files(summary: dict, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with (artifact_dir / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tag", "count", "mean", "last", "min_step", "max_step"])
        for tag, data in sorted(summary["tags"].items()):
            writer.writerow([
                tag,
                data.get("count"),
                data.get("mean"),
                data.get("last"),
                data.get("min_step"),
                data.get("max_step"),
            ])



class MetricsSummarizer:
    """metrics.jsonl の採点 window 集計と single-seed summary 保存を担当する。"""

    @staticmethod
    def summarize(metrics_path: Path, window: ScoreWindow, spec: MetricsSpec) -> dict:
        """primary score と late window 補助指標をまとめて算出する。"""
        return summarize_metrics(metrics_path, window, spec)

    @staticmethod
    def write_single_seed_summary(summary: dict, artifact_dir: Path) -> None:
        """seed run 1件分の metrics_summary.json/csv を保存する。"""
        write_summary_files(summary, artifact_dir)


def register_active_runner(proc: subprocess.Popen) -> None:
    with _ACTIVE_RUNNER_LOCK:
        _ACTIVE_RUNNERS.add(proc)


def unregister_active_runner(proc: subprocess.Popen) -> None:
    with _ACTIVE_RUNNER_LOCK:
        _ACTIVE_RUNNERS.discard(proc)


def terminate_active_runners() -> list[int]:
    _INTERRUPTING.set()
    with _ACTIVE_RUNNER_LOCK:
        procs = list(_ACTIVE_RUNNERS)

    terminated: list[int] = []
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
            terminated.append(proc.pid)

    deadline = time.monotonic() + 10.0
    for proc in procs:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            if proc.poll() is None:
                proc.kill()
                terminated.append(proc.pid)
    return terminated


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_runner_process_file(
    artifact_dir: Path,
    data: dict,
) -> None:
    (artifact_dir / "process.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def runner_process_data(
    status: str,
    command: list[str],
    ctx: TrialContext,
    start_monotonic: float,
    *,
    runner_pid: int | None = None,
    returncode: int | None = None,
    started_at: str,
    finished_at: str | None = None,
    interrupted: bool = False,
    timed_out: bool = False,
    error: str | None = None,
) -> dict:
    data = {
        "status": status,
        "command": command,
        "config_path": ctx.config_path,
        "harness_pid": os.getpid(),
        "runner_pid": runner_pid,
        "started_at": started_at,
        "finished_at": finished_at,
        "returncode": returncode,
        "elapsed_sec": time.monotonic() - start_monotonic,
        "interrupted": interrupted,
        "timed_out": timed_out,
    }
    if error is not None:
        data["error"] = error
    return data


def run_runner(args: argparse.Namespace, ctx: TrialContext) -> int:
    # runner 本体には Optuna を入れず、生成済み config path だけを渡す。
    artifact_dir = Path(ctx.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    runner_exe = Path(args.runner_exe)
    if not runner_exe.is_absolute():
        # Python harness は repo root から呼ぶことが多いため、相対 runner path は repo root 基準で解決する。
        runner_exe = Path(args.repo_root).resolve() / runner_exe
    command = [
        str(runner_exe),
        "--config",
        ctx.config_path,
    ]
    start = time.monotonic()
    started_at = utc_timestamp()
    stdout_path = artifact_dir / "stdout.log"
    stderr_path = artifact_dir / "stderr.log"
    proc: subprocess.Popen | None = None
    stdout_stream = None
    stderr_stream = None
    try:
        stdout_stream = stdout_path.open("w", encoding="utf-8", newline="\n")
        stderr_stream = stderr_path.open("w", encoding="utf-8", newline="\n")
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(runner_root(Path(args.repo_root).resolve())),
                text=True,
                stdout=stdout_stream,
                stderr=stderr_stream,
            )
        except OSError as e:
            stderr_stream.write(str(e))
            stderr_stream.write("\n")
            stderr_stream.flush()
            write_runner_process_file(
                artifact_dir,
                runner_process_data(
                    "failed",
                    command,
                    ctx,
                    start,
                    started_at=started_at,
                    finished_at=utc_timestamp(),
                    error=str(e),
                ),
            )
            log_harness(args, "ERROR", "runner-start-failed", console=True, run=ctx.run_name, error=str(e))
            raise TrialFailedError(f"runner failed to start: {e}") from e

        register_active_runner(proc)
        write_runner_process_file(
            artifact_dir,
            runner_process_data(
                "running",
                command,
                ctx,
                start,
                runner_pid=proc.pid,
                started_at=started_at,
            ),
        )
        log_harness(args, "INFO", "runner-start", console=True, run=ctx.run_name, pid=proc.pid, config=ctx.config_path)

        try:
            proc.wait(timeout=args.timeout_sec)
        except subprocess.TimeoutExpired as e:
            proc.kill()
            proc.wait()
            write_runner_process_file(
                artifact_dir,
                runner_process_data(
                    "timed_out",
                    command,
                    ctx,
                    start,
                    runner_pid=proc.pid,
                    returncode=proc.returncode,
                    started_at=started_at,
                    finished_at=utc_timestamp(),
                    timed_out=True,
                ),
            )
            log_harness(args, "ERROR", "runner-timeout", console=True, run=ctx.run_name, pid=proc.pid, returncode=proc.returncode)
            raise TrialFailedError(f"runner timed out: timeout_sec={args.timeout_sec}") from e

        interrupted = _INTERRUPTING.is_set()
        status = "complete" if proc.returncode == 0 and not interrupted else "failed"
        write_runner_process_file(
            artifact_dir,
            runner_process_data(
                status,
                command,
                ctx,
                start,
                runner_pid=proc.pid,
                returncode=proc.returncode,
                started_at=started_at,
                finished_at=utc_timestamp(),
                interrupted=interrupted,
            ),
        )
        log_harness(args, "INFO" if proc.returncode == 0 else "ERROR", "runner-exit", console=True, run=ctx.run_name, pid=proc.pid, returncode=proc.returncode)
        if interrupted:
            raise KeyboardInterrupt()
        return int(proc.returncode)
    except KeyboardInterrupt:
        _INTERRUPTING.set()
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if proc is not None:
            write_runner_process_file(
                artifact_dir,
                runner_process_data(
                    "interrupted",
                    command,
                    ctx,
                    start,
                    runner_pid=proc.pid,
                    returncode=proc.returncode,
                    started_at=started_at,
                    finished_at=utc_timestamp(),
                    interrupted=True,
                ),
            )
            log_harness(args, "WARN", "runner-interrupted", console=True, run=ctx.run_name, pid=proc.pid, returncode=proc.returncode)
        raise
    finally:
        if proc is not None:
            unregister_active_runner(proc)
        if stdout_stream is not None:
            stdout_stream.close()
        if stderr_stream is not None:
            stderr_stream.close()


class RunnerProcessManager:
    """runner 子プロセスの起動、終了記録、active process cleanup を担当する。"""

    @staticmethod
    def run(args: argparse.Namespace, ctx: TrialContext) -> int:
        """runner を 1 回起動し、process.json と stdout/stderr artifact を更新する。"""
        return run_runner(args, ctx)

    @staticmethod
    def terminate_active() -> list[int]:
        """中断時に現在の harness が起動した runner を停止する。"""
        return terminate_active_runners()


class OptunaHarnessRuntime:
    """ENV 非依存の Optuna 実行骨格を domain callback で動かす runtime。"""

    def __init__(
        self,
        *,
        domain,
        harness_name: str,
        script_path: str | Path,
        budget_presets: dict[str, float],
        metrics_spec: MetricsSpec,
        summary_objective_names: tuple[str, ...] = ("mean", "range", "std"),
        summary_objective_directions: tuple[str, ...] = ("maximize", "minimize", "minimize"),
        group_summary_artifact_filename: str = "group_summary.json",
    ):
        self.domain = domain
        self.harness_name = harness_name
        self.script_path = Path(script_path)
        self.budget_presets = dict(budget_presets)
        self.metrics_spec = metrics_spec
        self.summary_objective_names = tuple(summary_objective_names)
        self.summary_objective_directions = tuple(summary_objective_directions)
        self.group_summary_artifact_filename = group_summary_artifact_filename

    def repo_root_from_script(self) -> Path:
        """entrypoint の配置から repo root を解決する。"""
        return self.script_path.resolve().parents[3]

    def params_to_dict(self, params) -> dict:
        """domain params を JSON/Optuna attrs 用の dict に変換する。"""
        converter = getattr(self.domain, "params_to_dict", None)
        if converter is not None:
            return dict(converter(params))
        if is_dataclass(params):
            return asdict(params)
        return dict(params)

    def resolve_budget(self, args: argparse.Namespace) -> tuple[str, float]:
        """CLI の budget preset と明示 cost_budget から実効 cost_budget を返す。"""
        if args.cost_budget is not None:
            return args.budget, float(args.cost_budget)
        return args.budget, float(self.budget_presets[args.budget])

    @staticmethod
    def validate_name_part(name: str, option_name: str) -> None:
        """study/trial 名に path separator が混ざる誤用を防ぐ。"""
        if "/" in name or "\\" in name:
            raise ValueError(f"Invalid {option_name}: path separator is not allowed. value={name}")

    @staticmethod
    def trial_number_from_name(trial_name: str) -> int | None:
        """tNNNNN 形式の trial 名から番号を復元する。"""
        match = TRIAL_NAME_PATTERN.fullmatch(trial_name)
        return int(match.group(1)) if match else None

    def scan_existing_trial_numbers(self, args: argparse.Namespace, budget_name: str) -> list[int]:
        """既存 run folder から同一 study の最大 trial 番号を探す。"""
        repo_root = Path(args.repo_root).resolve()
        placeholders = {
            "study": args.study_name,
            "budget": budget_name,
            "trial": "",
            "run": "",
        }
        root = resolve_runner_relative_path(repo_root, args.runs_dir.format(**placeholders))
        run_name_pattern = re.compile(rf"^{re.escape(args.study_name)}_t(\d{{5}})$")
        numbers: list[int] = []
        if not root.is_dir():
            return numbers
        for child in root.iterdir():
            if not child.is_dir():
                continue
            match = run_name_pattern.fullmatch(child.name)
            if match:
                numbers.append(int(match.group(1)))
        return numbers

    def next_trial_number(self, args: argparse.Namespace, budget_name: str) -> int:
        """既存出力から次に使う tNNNNN 番号を決める。"""
        existing = self.scan_existing_trial_numbers(args, budget_name)
        return max(existing) + 1 if existing else 0

    def resolve_trial_identity(
        self,
        args: argparse.Namespace,
        budget_name: str,
        trial_number: int | None,
        trial_name_override: str | None,
    ) -> tuple[int, str]:
        """明示 trial 名/番号と既存 run folder から trial identity を確定する。"""
        trial_name = trial_name_override or getattr(args, "trial_name", None)
        if trial_number is not None:
            return trial_number, trial_name or f"t{trial_number:05d}"

        if trial_name:
            number_from_name = self.trial_number_from_name(trial_name)
            if number_from_name is not None:
                return number_from_name, trial_name
            return self.next_trial_number(args, budget_name), trial_name

        number = self.next_trial_number(args, budget_name)
        return number, f"t{number:05d}"

    def make_trial_context(
        self,
        args: argparse.Namespace,
        params,
        trial_number: int | None,
        trial_name_override: str | None = None,
    ) -> TrialContext:
        """trial identity と run/trial artifact path を確定する。"""
        repo_root = Path(args.repo_root).resolve()
        run_root = runner_root(repo_root)
        budget_name, budget_value = self.resolve_budget(args)
        cost = float(self.domain.cost_tf(params, args.cost_k))
        n = int(self.domain.context_token_count(params))
        self.validate_name_part(args.study_name, "--study-name")
        trial_number, trial_name = self.resolve_trial_identity(args, budget_name, trial_number, trial_name_override)
        self.validate_name_part(trial_name, "--trial-name")
        run_name = f"{args.study_name}_{trial_name}"
        runs_dir = args.runs_dir.format(study=args.study_name, budget=budget_name, trial=trial_name, run=run_name)
        run_dir = run_root / runs_dir / run_name
        artifact_dir = run_dir / "trial"
        config_path = artifact_dir / "config.txt"
        return TrialContext(
            study_name=args.study_name,
            trial_number=trial_number,
            trial_name=trial_name,
            budget_name=budget_name,
            cost_budget=budget_value,
            cost_tf=cost,
            token_count=n,
            run_name=run_name,
            runs_dir=runs_dir.replace("\\", "/"),
            artifact_dir=str(artifact_dir),
            run_dir=str(run_dir),
            config_path=str(config_path),
        )

    def resolve_runs_root(self, args: argparse.Namespace) -> Path:
        """harness log と Optuna run が置かれる runs root を解決する。"""
        repo_root = Path(args.repo_root).resolve()
        budget_name, _ = self.resolve_budget(args)
        runs_dir = args.runs_dir.format(study=args.study_name, budget=budget_name, trial="", run="")
        return resolve_runner_relative_path(repo_root, runs_dir)

    def attach_harness_logger(self, args: argparse.Namespace) -> HarnessLogger:
        """同一 process 内で共有する harness.log writer を args に保持する。"""
        existing = getattr(args, "_harness_logger", None)
        if existing is not None:
            return existing
        logger = HarnessLogger(self.resolve_runs_root(args) / "harness.log")
        setattr(args, "_harness_logger", logger)
        return logger

    def build_run_study_copy_args(self, args: argparse.Namespace, storage_url: str) -> str:
        """Dashboard から貼れる run-study 再開用 args を作る。"""
        timeout_sec = getattr(args, "timeout_sec", 0)
        if timeout_sec is None:
            timeout_sec = 0

        parts = ["run-study"]
        append_cli_arg(parts, "--study-name", args.study_name)
        append_cli_arg(parts, "--storage", storage_url)
        append_cli_arg(parts, "--runs-dir", args.runs_dir)
        append_cli_arg(parts, "--budget", args.budget)
        if args.cost_budget is not None:
            append_cli_arg(parts, "--cost-budget", args.cost_budget)
        append_cli_arg(parts, "--cost-k", args.cost_k)
        append_cli_arg(parts, "--base-config", args.base_config)
        append_cli_arg(parts, "--extra-config", args.extra_config)
        append_cli_arg(parts, "--runner-exe", args.runner_exe)
        append_cli_arg(parts, "--timeout-sec", timeout_sec)
        append_cli_arg(parts, "--exp-exit-step", args.exp_exit_step)
        append_cli_arg(parts, "--window-start", args.window_start)
        append_cli_arg(parts, "--window-end", args.window_end)
        if hasattr(args, "nhead"):
            append_cli_arg(parts, "--nhead", args.nhead)
        fixed_param_cli_args = getattr(self.domain, "fixed_param_cli_args", None)
        if fixed_param_cli_args is not None:
            for option_name, value in fixed_param_cli_args(args):
                append_cli_arg(parts, option_name, value)
        append_cli_arg(parts, "--seeds", args.seeds)
        append_cli_arg(parts, "--score-aggregate", args.score_aggregate)
        if args.sampler_seed is not None:
            append_cli_arg(parts, "--sampler-seed", args.sampler_seed)
        append_cli_arg(parts, "--n-startup-trials", args.n_startup_trials)
        if args.constant_liar:
            parts.append("--constant-liar")
        append_cli_arg(parts, "--duplicate-params-policy", args.duplicate_params_policy)
        append_cli_arg(parts, "--duplicate-params-max-runs", args.duplicate_params_max_runs)
        append_cli_arg(parts, "--duplicate-seed-stride", args.duplicate_seed_stride)
        append_cli_arg(parts, "--heartbeat-interval-sec", args.heartbeat_interval_sec)
        append_cli_arg(parts, "--heartbeat-grace-period-sec", args.heartbeat_grace_period_sec)
        append_cli_arg(parts, "--storage-timeout-sec", args.storage_timeout_sec)
        append_cli_arg(parts, "--optuna-artifact-dir", args.optuna_artifact_dir)
        append_cli_arg(parts, "--n-trials", args.n_trials)
        append_cli_arg(parts, "--n-jobs", args.n_jobs)
        if args.study_note is not None:
            append_cli_arg(parts, "--study-note", args.study_note)
        return format_cli_args(parts)

    def build_study_user_attrs(self, args: argparse.Namespace, storage_url: str) -> dict:
        """Study Dashboard 用に最後の run-study 起動条件を flat attrs として保存する。"""
        budget_name, cost_budget = self.resolve_budget(args)
        window = resolve_score_window(args)
        seeds = parse_seed_list(args.seeds)
        storage_timeout_sec = getattr(args, "storage_timeout_sec", 120.0)
        attrs = {
            "00_last_run_study_args": self.build_run_study_copy_args(args, storage_url),
            "last_launch_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "last_harness": self.harness_name,
            "last_command": "run-study",
            "last_study_name": args.study_name,
            "last_storage": storage_url,
            "last_storage_timeout_sec": storage_timeout_sec,
            "last_heartbeat_interval_sec": getattr(args, "heartbeat_interval_sec", None),
            "last_heartbeat_grace_period_sec": getattr(args, "heartbeat_grace_period_sec", None),
            "last_runs_dir": args.runs_dir,
            "last_budget": budget_name,
            "last_cost_budget": cost_budget,
            "last_cost_k": args.cost_k,
            "last_exp_exit_step": args.exp_exit_step,
            "last_window_start_raw": window.raw_start,
            "last_window_end_raw": window.raw_end,
            "last_window_start": window.start,
            "last_window_end": window.end,
            "last_seeds": seeds,
            "last_seed_count": len(seeds),
            "last_score_aggregate": args.score_aggregate,
            "last_sampler_seed": args.sampler_seed,
            "last_n_startup_trials": args.n_startup_trials,
            "last_constant_liar": args.constant_liar,
            "last_duplicate_params_policy": args.duplicate_params_policy,
            "last_duplicate_params_max_runs": args.duplicate_params_max_runs,
            "last_duplicate_seed_stride": args.duplicate_seed_stride,
            "last_n_trials": args.n_trials,
            "last_n_jobs": args.n_jobs,
            "last_timeout_sec": args.timeout_sec,
            "last_runner_exe": args.runner_exe,
            "last_base_config": args.base_config,
            "last_extra_config": args.extra_config,
        }
        if args.study_note is not None:
            attrs["note"] = args.study_note
        return attrs

    def duplicate_relevant_trial_numbers(self, study, params, current_trial_number: int) -> list[int]:
        """同一 params の COMPLETE/RUNNING trial を duplicate 判定対象として集める。"""
        if study is None:
            return []

        current_key = self.domain.params_key(params)
        matched: list[int] = []
        for trial in getattr(study, "trials", []):
            if getattr(trial, "number", None) == current_trial_number:
                continue
            if trial_state_name(trial) not in ("COMPLETE", "RUNNING"):
                continue
            if self.domain.params_key_from_mapping(getattr(trial, "params", {})) == current_key:
                matched.append(int(trial.number))
        return matched

    def resolve_duplicate_params_info(
        self,
        args: argparse.Namespace,
        params,
        study,
        trial_number: int,
        base_seeds: list[int],
    ) -> DuplicateParamsInfo:
        """duplicate policy と既存 trial から、この trial の effective seeds を決める。"""
        policy = args.duplicate_params_policy
        matched_trials = self.duplicate_relevant_trial_numbers(study, params, trial_number)
        duplicate_count = len(matched_trials)
        max_runs = int(args.duplicate_params_max_runs)
        seed_stride = int(args.duplicate_seed_stride)

        if policy == "allow":
            duplicate_index = 0
            effective_seeds = list(base_seeds)
            prune_reason = None
        elif policy == "prune" and duplicate_count > 0:
            duplicate_index = duplicate_count
            effective_seeds = list(base_seeds)
            prune_reason = f"duplicate params: matched_trials={matched_trials}"
        else:
            duplicate_index = duplicate_count
            effective_seeds = shifted_seeds(base_seeds, duplicate_index, seed_stride)
            if policy == "reseed" and max_runs > 0 and duplicate_count >= max_runs:
                prune_reason = (
                    "duplicate params max runs exceeded: "
                    f"duplicate_count_before={duplicate_count} max_runs={max_runs}"
                )
            else:
                prune_reason = None

        return DuplicateParamsInfo(
            policy=policy,
            duplicate_count_before=duplicate_count,
            duplicate_index=duplicate_index,
            duplicate_params_max_runs=max_runs,
            duplicate_seed_stride=seed_stride,
            base_seeds=list(base_seeds),
            effective_seeds=effective_seeds,
            duplicate_matched_trials=matched_trials,
            pruned_by_duplicate=prune_reason is not None,
            prune_reason=prune_reason,
        )

    def make_manifest(self, args: argparse.Namespace, params, ctx: TrialContext, extra: dict | None = None) -> dict:
        """trial 再現に必要な domain params と harness 条件を manifest にまとめる。"""
        manifest = {
            "params": self.params_to_dict(params),
            "context": asdict(ctx),
            "base_config": args.base_config,
            "extra_config": args.extra_config,
            "cost_k": args.cost_k,
            "exp_exit_step": args.exp_exit_step,
            "seed": getattr(args, "seed", None),
            "sampler_seed": getattr(args, "sampler_seed", None),
            "n_startup_trials": getattr(args, "n_startup_trials", None),
            "constant_liar": getattr(args, "constant_liar", None),
            "duplicate_params_policy": getattr(args, "duplicate_params_policy", None),
            "duplicate_count_before": getattr(args, "duplicate_count_before", None),
            "duplicate_index": getattr(args, "duplicate_index", None),
            "duplicate_params_max_runs": getattr(args, "duplicate_params_max_runs", None),
            "duplicate_seed_stride": getattr(args, "duplicate_seed_stride", None),
            "base_seeds": getattr(args, "base_seeds", None),
            "effective_seeds": getattr(args, "effective_seeds", None),
            "duplicate_matched_trials": getattr(args, "duplicate_matched_trials", None),
            "primary_tags": list(self.metrics_spec.primary_tags),
            "supplemental_tags": list(self.metrics_spec.supplemental_tags),
        }
        if extra:
            manifest.update(extra)
        return manifest

    @staticmethod
    def write_manifest_file(manifest: dict, artifact_dir: Path) -> None:
        """manifest.json を LF/UTF-8 で保存する。"""
        (artifact_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def write_trial_files(self, args: argparse.Namespace, params, ctx: TrialContext) -> None:
        """runner 起動前に config と manifest を保存する。"""
        artifact_dir = Path(ctx.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        config_text = self.domain.render_config(params, ctx, args)
        Path(ctx.config_path).write_text(config_text, encoding="utf-8", newline="\n")
        self.write_manifest_file(self.make_manifest(args, params, ctx), artifact_dir)

    def write_representative_trial_files(
        self,
        args: argparse.Namespace,
        params,
        ctx: TrialContext,
        seeds: list[int],
        duplicate_info: DuplicateParamsInfo,
    ) -> None:
        """multi-seed 代表 trial の manifest を保存する。"""
        artifact_dir = Path(ctx.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.make_manifest(args, params, ctx, {
            "representative_trial": True,
            "seeds": seeds,
            "score_aggregate": args.score_aggregate,
            "duplicate_params_policy": duplicate_info.policy,
            "duplicate_count_before": duplicate_info.duplicate_count_before,
            "duplicate_index": duplicate_info.duplicate_index,
            "duplicate_params_max_runs": duplicate_info.duplicate_params_max_runs,
            "duplicate_seed_stride": duplicate_info.duplicate_seed_stride,
            "base_seeds": duplicate_info.base_seeds,
            "effective_seeds": duplicate_info.effective_seeds,
            "duplicate_matched_trials": duplicate_info.duplicate_matched_trials,
        })
        self.write_manifest_file(manifest, artifact_dir)

    @staticmethod
    def build_seed_runs_document(summary: dict) -> dict:
        """Dashboard artifact 用の seed_runs.json を summary から切り出す。"""
        return {
            "params": summary["params"],
            "context": summary["context"],
            "score": {
                "aggregate": summary["score_aggregate"],
                "value": summary["score"],
                "mean": summary["score_mean"],
                "std": summary["score_std"],
                "median": summary["score_median"],
                "min": summary["score_min"],
                "max": summary["score_max"],
                "range": summary["score_range"],
                "mean_minus_std": summary["score_mean_minus_std"],
            },
            "late_window_score": {
                "score_60_80_mean": summary["score_60_80_mean"],
                "score_80_100_mean": summary["score_80_100_mean"],
                "late_slope_mean": summary["late_slope_mean"],
                "late_slope_std": summary["late_slope_std"],
            },
            "seed_count": summary["seed_count"],
            "seed_success_count": summary["seed_success_count"],
            "seed_failure_count": summary["seed_failure_count"],
            "seeds": summary["seeds"],
            "runs": summary["runs"],
            "error": summary.get("error"),
        }

    def write_multiseed_summary_files(self, summary: dict, artifact_dir: Path) -> None:
        """multi-seed aggregate と seed 別 record を JSON/CSV で保存する。"""
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "multiseed_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (artifact_dir / "seed_runs.json").write_text(
            json.dumps(self.build_seed_runs_document(summary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        with (artifact_dir / "multiseed_summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "kind",
                "seed",
                "status",
                "score",
                "score_range",
                "score_60_80",
                "score_80_100",
                "late_slope",
                "late_slope_std",
                "run_name",
                "metrics_summary_path",
                "error",
            ])
            aggregate_status = "complete" if summary["score"] is not None else "failed"
            writer.writerow([
                "aggregate",
                "",
                aggregate_status,
                summary["score"],
                summary["score_range"],
                summary["score_60_80_mean"],
                summary["score_80_100_mean"],
                summary["late_slope_mean"],
                summary["late_slope_std"],
                summary["context"]["run_name"],
                "",
                summary.get("error"),
            ])
            for run in summary["runs"]:
                writer.writerow([
                    "seed",
                    run.get("seed"),
                    run.get("status"),
                    run.get("score"),
                    "",
                    run.get("score_60_80"),
                    run.get("score_80_100"),
                    run.get("late_slope"),
                    "",
                    run.get("run_name"),
                    run.get("metrics_summary_path"),
                    run.get("error"),
                ])

    @staticmethod
    def make_seed_run_record(
        seed: int,
        ctx: TrialContext,
        status: str,
        score: float | None = None,
        summary: dict | None = None,
        error: str | None = None,
    ) -> dict:
        """multi-seed artifact に残す seed run 1件分の record を作る。"""
        return {
            "seed": seed,
            "status": status,
            "score": score,
            "score_60_80": None if summary is None else summary.get("score_60_80"),
            "score_80_100": None if summary is None else summary.get("score_80_100"),
            "late_slope": None if summary is None else summary.get("late_slope"),
            "trial_name": ctx.trial_name,
            "run_name": ctx.run_name,
            "run_dir": ctx.run_dir,
            "artifact_dir": ctx.artifact_dir,
            "config_path": ctx.config_path,
            "metrics_path": str(Path(ctx.run_dir) / "metrics.jsonl"),
            "metrics_summary_path": str(Path(ctx.artifact_dir) / "metrics_summary.json"),
            "error": error,
        }

    def build_multiseed_summary(
        self,
        args: argparse.Namespace,
        params,
        ctx: TrialContext,
        seeds: list[int],
        seed_runs: list[dict],
        duplicate_info: DuplicateParamsInfo,
        error: str | None = None,
    ) -> dict:
        """seed run 群を Optuna objective 用の aggregate summary に集約する。"""
        scores = [
            float(run["score"])
            for run in seed_runs
            if run.get("status") == "complete" and run.get("score") is not None
        ]
        stats = aggregate_score_stats(scores, args.score_aggregate)
        score_60_80_stats = aggregate_score_stats(completed_run_values(seed_runs, "score_60_80"), "mean")
        score_80_100_stats = aggregate_score_stats(completed_run_values(seed_runs, "score_80_100"), "mean")
        late_slope_stats = aggregate_score_stats(completed_run_values(seed_runs, "late_slope"), "mean")
        seed_scores = {
            str(run["seed"]): run.get("score")
            for run in seed_runs
        }
        seed_run_names = [
            str(run["run_name"])
            for run in seed_runs
            if run.get("run_name") is not None
        ]
        seed_failure_count = sum(1 for run in seed_runs if run.get("status") != "complete")
        return {
            "params": self.params_to_dict(params),
            "context": asdict(ctx),
            "score": stats["aggregate_score"],
            "score_aggregate": args.score_aggregate,
            "score_mean": stats["score_mean"],
            "score_std": stats["score_std"],
            "score_median": stats["score_median"],
            "score_min": stats["score_min"],
            "score_max": stats["score_max"],
            "score_range": stats["score_range"],
            "score_mean_minus_std": stats["score_mean_minus_std"],
            "score_60_80_mean": score_60_80_stats["score_mean"],
            "score_80_100_mean": score_80_100_stats["score_mean"],
            "late_slope_mean": late_slope_stats["score_mean"],
            "late_slope_std": late_slope_stats["score_std"],
            "seed_count": len(seeds),
            "seed_success_count": len(scores),
            "seed_failure_count": seed_failure_count,
            "seeds": seeds,
            "duplicate_params_policy": duplicate_info.policy,
            "duplicate_count_before": duplicate_info.duplicate_count_before,
            "duplicate_index": duplicate_info.duplicate_index,
            "duplicate_params_max_runs": duplicate_info.duplicate_params_max_runs,
            "duplicate_seed_stride": duplicate_info.duplicate_seed_stride,
            "base_seeds": duplicate_info.base_seeds,
            "effective_seeds": duplicate_info.effective_seeds,
            "duplicate_matched_trials": duplicate_info.duplicate_matched_trials,
            "seed_scores": seed_scores,
            "seed_run_names": seed_run_names,
            "runs": seed_runs,
            "error": error,
        }

    def command_dry_run(self, args: argparse.Namespace) -> int:
        """trial config と manifest を生成し、runner は起動しない。"""
        params = self.domain.params_from_args(args)
        ctx = self.make_trial_context(args, params, args.trial_number)
        score_window = resolve_score_window(args)
        try:
            self.write_trial_files(args, params, ctx)
        except ValueError as e:
            raise TrialExecutionError(str(e)) from e
        result = {
            "params": self.params_to_dict(params),
            "context": asdict(ctx),
            "score_window": asdict(score_window),
            "pruned_by_cost": ctx.cost_tf > ctx.cost_budget,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if ctx.cost_tf <= ctx.cost_budget else 2

    def command_summarize(self, args: argparse.Namespace) -> int:
        """既存 metrics.jsonl を domain の MetricsSpec で採点する。"""
        summary = MetricsSummarizer.summarize(Path(args.metrics_jsonl), resolve_score_window(args), self.metrics_spec)
        if args.output_dir:
            MetricsSummarizer.write_single_seed_summary(summary, Path(args.output_dir))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["score"] is not None else 2

    @staticmethod
    def optuna_study_exists(optuna, study_name: str, storage) -> bool:
        """Optuna version 差を吸収して study の存在を調べる。"""
        get_all_study_names = getattr(optuna.study, "get_all_study_names", None)
        if get_all_study_names is None:
            get_all_study_names = getattr(optuna, "get_all_study_names")
        return study_name in get_all_study_names(storage=storage)

    @staticmethod
    def delete_optuna_study(optuna, study_name: str, storage) -> None:
        """Optuna version 差を吸収して study を削除する。"""
        delete_study = getattr(optuna.study, "delete_study", None)
        if delete_study is None:
            delete_study = getattr(optuna, "delete_study")
        delete_study(study_name=study_name, storage=storage)

    @staticmethod
    def load_trial_json_artifact(
        artifact_context: OptunaArtifactContext,
        storage,
        trial,
        filename: str,
        *,
        required: bool,
    ) -> dict | None:
        """Optuna artifact store から trial JSON artifact を読む。"""
        if artifact_context.get_all_artifact_meta is None or artifact_context.download_artifact is None:
            raise TrialExecutionError("Optuna artifact download API is unavailable.")

        artifact_metas = artifact_context.get_all_artifact_meta(trial, storage=storage)
        matches = [
            meta
            for meta in artifact_metas
            if Path(str(getattr(meta, "filename", ""))).name == filename
        ]
        if not matches:
            if not required:
                return None
            raise TrialExecutionError(
                f"Required source artifact is missing: trial={trial.number} filename={filename}"
            )

        artifact_meta = matches[-1]
        with tempfile.TemporaryDirectory(prefix="anet-optuna-artifact-") as temp_dir:
            download_path = Path(temp_dir) / filename
            artifact_context.download_artifact(
                artifact_store=artifact_context.artifact_store,
                artifact_id=artifact_meta.artifact_id,
                file_path=str(download_path),
            )
            return json.loads(download_path.read_text(encoding="utf-8-sig"))

    @staticmethod
    def upload_json_trial_artifact(
        artifact_context: OptunaArtifactContext,
        storage,
        trial,
        filename: str,
        payload: dict,
    ) -> None:
        """dict を一時 JSON として作り、Optuna artifact に登録する。"""
        with tempfile.TemporaryDirectory(prefix="anet-optuna-summary-") as temp_dir:
            artifact_path = Path(temp_dir) / filename
            artifact_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            artifact_context.upload_artifact(
                artifact_store=artifact_context.artifact_store,
                file_path=str(artifact_path),
                study_or_trial=trial,
                storage=storage,
            )

    @staticmethod
    def source_trial_score(trial) -> float:
        """source COMPLETE trial の objective score を取得する。"""
        value = getattr(trial, "value", None)
        if value is None:
            value = getattr(trial, "user_attrs", {}).get("score")
        if value is None:
            raise TrialExecutionError(f"Source COMPLETE trial has no objective score: trial={trial.number}")
        return float(value)

    @staticmethod
    def source_score_context(trial, seed_runs_doc: dict, manifest_doc: dict | None) -> dict:
        """summary group の score 前提比較用 context を作る。"""
        user_attrs = getattr(trial, "user_attrs", {})
        score_stats = seed_runs_doc.get("score_stats") or {}
        return {
            "trial_number": int(trial.number),
            "score_aggregate": user_attrs.get("score_aggregate") or score_stats.get("aggregate"),
            "seed_count": user_attrs.get("seed_count") or seed_runs_doc.get("seed_count"),
            "cost_budget": user_attrs.get("cost_budget"),
            "exp_exit_step": None if manifest_doc is None else manifest_doc.get("exp_exit_step"),
            "primary_tags": None if manifest_doc is None else manifest_doc.get("primary_tags"),
            "base_config": None if manifest_doc is None else manifest_doc.get("base_config"),
            "extra_config": None if manifest_doc is None else manifest_doc.get("extra_config"),
        }

    @staticmethod
    def score_context_fingerprint(context: dict) -> str:
        """trial number を除いた score 前提の比較 key を作る。"""
        comparable = {
            key: value
            for key, value in context.items()
            if key != "trial_number"
        }
        return json.dumps(comparable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def increment_count(counts: dict[str, int], key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    def collect_source_trial_groups(self, source_study) -> tuple[dict, dict, dict[str, int], list[dict]]:
        """source study の trial を domain params key 単位に集約する。"""
        complete_groups: dict[tuple[tuple[str, object], ...], dict] = {}
        state_counts_by_key: dict[tuple[tuple[str, object], ...], dict[str, int]] = {}
        global_state_counts: dict[str, int] = {}
        skipped_trials: list[dict] = []

        for trial in study_trials(source_study):
            state_name = trial_state_name(trial)
            self.increment_count(global_state_counts, state_name)
            try:
                params = self.domain.params_from_mapping(
                    getattr(trial, "params", {}),
                    source=f"trial={trial.number} params",
                )
                self.domain.validate_params_in_search_space(params)
            except ValueError as e:
                skipped_trials.append({
                    "trial_number": int(getattr(trial, "number", -1)),
                    "state": state_name,
                    "error": str(e),
                })
                continue

            key = self.domain.params_key(params)
            state_counts = state_counts_by_key.setdefault(key, {})
            self.increment_count(state_counts, state_name)
            if state_name != "COMPLETE":
                continue

            group = complete_groups.setdefault(key, {
                "params": params,
                "trials": [],
            })
            group["trials"].append(trial)

        return complete_groups, state_counts_by_key, global_state_counts, skipped_trials

    def build_summary_group_document(
        self,
        group_id: str,
        params,
        source_trials: list,
        source_state_counts: dict[str, int],
        source_artifact_context: OptunaArtifactContext,
        source_storage,
    ) -> tuple[dict, dict, list[float]]:
        """同一 params group から summary study の trial document/attrs/objectives を作る。"""
        seed_scores: list[float] = []
        seed_score_60_80: list[float] = []
        seed_score_80_100: list[float] = []
        seed_late_slopes: list[float] = []
        seed_records: list[dict] = []
        source_trial_scores: list[float] = []
        source_trial_records: list[dict] = []
        source_contexts: list[dict] = []

        for trial in sorted(source_trials, key=lambda item: int(item.number)):
            seed_runs_doc = self.load_trial_json_artifact(
                source_artifact_context,
                source_storage,
                trial,
                "seed_runs.json",
                required=True,
            )
            manifest_doc = self.load_trial_json_artifact(
                source_artifact_context,
                source_storage,
                trial,
                "manifest.json",
                required=False,
            )
            runs = seed_runs_doc.get("runs")
            if not isinstance(runs, list):
                raise TrialExecutionError(f"Invalid seed_runs.json: missing runs list. trial={trial.number}")

            trial_seed_scores: list[float] = []
            trial_score_60_80: list[float] = []
            trial_score_80_100: list[float] = []
            trial_late_slopes: list[float] = []
            for run in runs:
                if run.get("status") != "complete":
                    continue
                if run.get("score") is None:
                    raise TrialExecutionError(
                        f"Invalid seed_runs.json: complete seed run has no score. "
                        f"trial={trial.number} seed={run.get('seed')}"
                    )
                score = float(run["score"])
                score_60_80 = None if run.get("score_60_80") is None else float(run["score_60_80"])
                score_80_100 = None if run.get("score_80_100") is None else float(run["score_80_100"])
                late_slope = None if run.get("late_slope") is None else float(run["late_slope"])
                trial_seed_scores.append(score)
                seed_scores.append(score)
                if score_60_80 is not None:
                    trial_score_60_80.append(score_60_80)
                    seed_score_60_80.append(score_60_80)
                if score_80_100 is not None:
                    trial_score_80_100.append(score_80_100)
                    seed_score_80_100.append(score_80_100)
                if late_slope is not None:
                    trial_late_slopes.append(late_slope)
                    seed_late_slopes.append(late_slope)
                seed_records.append({
                    "source_trial_number": int(trial.number),
                    "duplicate_index": getattr(trial, "user_attrs", {}).get("duplicate_index"),
                    "seed": run.get("seed"),
                    "score": score,
                    "score_60_80": score_60_80,
                    "score_80_100": score_80_100,
                    "late_slope": late_slope,
                    "run_name": run.get("run_name"),
                    "trial_name": run.get("trial_name"),
                    "run_dir": run.get("run_dir"),
                    "artifact_dir": run.get("artifact_dir"),
                    "metrics_path": run.get("metrics_path"),
                    "metrics_summary_path": run.get("metrics_summary_path"),
                })

            if not trial_seed_scores:
                raise TrialExecutionError(f"Source COMPLETE trial has no complete seed scores: trial={trial.number}")

            trial_score = self.source_trial_score(trial)
            trial_score_60_80_stats = aggregate_score_stats(trial_score_60_80, "mean")
            trial_score_80_100_stats = aggregate_score_stats(trial_score_80_100, "mean")
            trial_late_slope_stats = aggregate_score_stats(trial_late_slopes, "mean")
            source_trial_scores.append(trial_score)
            context = self.source_score_context(trial, seed_runs_doc, manifest_doc)
            source_contexts.append(context)
            source_trial_records.append({
                "trial_number": int(trial.number),
                "trial_value": trial_score,
                "seed_count": len(trial_seed_scores),
                "seed_score_mean": float(mean(trial_seed_scores)),
                "seed_score_range": float(max(trial_seed_scores) - min(trial_seed_scores)),
                "seed_score_std": float(pstdev(trial_seed_scores)) if len(trial_seed_scores) > 1 else 0.0,
                "seed_score_60_80_mean": trial_score_60_80_stats["score_mean"],
                "seed_score_80_100_mean": trial_score_80_100_stats["score_mean"],
                "seed_late_slope_mean": trial_late_slope_stats["score_mean"],
                "seed_late_slope_std": trial_late_slope_stats["score_std"],
                "duplicate_index": getattr(trial, "user_attrs", {}).get("duplicate_index"),
                "effective_seeds": getattr(trial, "user_attrs", {}).get("effective_seeds"),
                "seed_run_names": [
                    run.get("run_name")
                    for run in runs
                    if run.get("status") == "complete" and run.get("run_name") is not None
                ],
                "score_context": context,
            })

        group_stats = aggregate_score_stats(seed_scores, "mean")
        group_score_60_80_stats = aggregate_score_stats(seed_score_60_80, "mean")
        group_score_80_100_stats = aggregate_score_stats(seed_score_80_100, "mean")
        group_late_slope_stats = aggregate_score_stats(seed_late_slopes, "mean")
        source_trial_stats = aggregate_score_stats(source_trial_scores, "mean")
        source_context_mixed = len({self.score_context_fingerprint(context) for context in source_contexts}) > 1
        params_dict = self.params_to_dict(params)

        user_attrs = {
            "group_id": group_id,
            "params": params_dict,
            "source_trial_numbers": [record["trial_number"] for record in source_trial_records],
            "source_trial_count": len(source_trial_records),
            "source_seed_count": len(seed_scores),
            "group_score_mean": group_stats["score_mean"],
            "group_score_range": group_stats["score_range"],
            "group_score_std": group_stats["score_std"],
            "group_score_min": group_stats["score_min"],
            "group_score_max": group_stats["score_max"],
            "group_score_median": group_stats["score_median"],
            "group_score_mean_minus_std": group_stats["score_mean_minus_std"],
            "group_score_60_80_mean": group_score_60_80_stats["score_mean"],
            "group_score_80_100_mean": group_score_80_100_stats["score_mean"],
            "group_late_slope_mean": group_late_slope_stats["score_mean"],
            "group_late_slope_std": group_late_slope_stats["score_std"],
            "source_trial_score_mean": source_trial_stats["score_mean"],
            "source_trial_score_range": source_trial_stats["score_range"],
            "source_trial_score_std": source_trial_stats["score_std"],
            "source_context_mixed": source_context_mixed,
            "source_state_counts": dict(sorted(source_state_counts.items())),
        }
        document = {
            "group_id": group_id,
            "params": params_dict,
            "objectives": {
                "names": list(self.summary_objective_names),
                "directions": list(self.summary_objective_directions),
                "values": [
                    group_stats["score_mean"],
                    group_stats["score_range"],
                    group_stats["score_std"],
                ],
            },
            "group_score_stats": {
                "mean": group_stats["score_mean"],
                "range": group_stats["score_range"],
                "std": group_stats["score_std"],
                "min": group_stats["score_min"],
                "max": group_stats["score_max"],
                "median": group_stats["score_median"],
                "mean_minus_std": group_stats["score_mean_minus_std"],
            },
            "group_late_window_score_stats": {
                "score_60_80": {
                    "mean": group_score_60_80_stats["score_mean"],
                    "range": group_score_60_80_stats["score_range"],
                    "std": group_score_60_80_stats["score_std"],
                    "min": group_score_60_80_stats["score_min"],
                    "max": group_score_60_80_stats["score_max"],
                    "median": group_score_60_80_stats["score_median"],
                },
                "score_80_100": {
                    "mean": group_score_80_100_stats["score_mean"],
                    "range": group_score_80_100_stats["score_range"],
                    "std": group_score_80_100_stats["score_std"],
                    "min": group_score_80_100_stats["score_min"],
                    "max": group_score_80_100_stats["score_max"],
                    "median": group_score_80_100_stats["score_median"],
                },
                "late_slope": {
                    "mean": group_late_slope_stats["score_mean"],
                    "range": group_late_slope_stats["score_range"],
                    "std": group_late_slope_stats["score_std"],
                    "min": group_late_slope_stats["score_min"],
                    "max": group_late_slope_stats["score_max"],
                    "median": group_late_slope_stats["score_median"],
                },
            },
            "source_trial_score_stats": {
                "mean": source_trial_stats["score_mean"],
                "range": source_trial_stats["score_range"],
                "std": source_trial_stats["score_std"],
                "min": source_trial_stats["score_min"],
                "max": source_trial_stats["score_max"],
                "median": source_trial_stats["score_median"],
                "mean_minus_std": source_trial_stats["score_mean_minus_std"],
            },
            "source_state_counts": dict(sorted(source_state_counts.items())),
            "source_context_mixed": source_context_mixed,
            "source_score_contexts": source_contexts,
            "source_trials": source_trial_records,
            "seed_scores": seed_records,
        }
        return document, user_attrs, [
            float(group_stats["score_mean"]),
            float(group_stats["score_range"]),
            float(group_stats["score_std"]),
        ]

    def command_summarize_study(self, args: argparse.Namespace) -> int:
        """source study を同一 params group の summary study に変換する。"""
        if args.storage_timeout_sec < 0:
            print("storage-timeout-sec must be >= 0.", file=sys.stderr)
            return 2
        if args.target_study_name is None:
            args.target_study_name = f"{args.source_study_name}_summary"
        self.validate_name_part(args.source_study_name, "--source-study-name")
        self.validate_name_part(args.target_study_name, "--target-study-name")

        try:
            import optuna
        except ImportError:
            print("Optuna is required for summarize-study. Install optuna in the Python environment.", file=sys.stderr)
            return 2

        repo_root = Path(args.repo_root).resolve()
        source_storage_url = storage_url_from_text(repo_root, args.source_storage)
        target_storage_url = storage_url_from_text(repo_root, args.target_storage or args.source_storage)
        if source_storage_url == target_storage_url and args.source_study_name == args.target_study_name:
            print(
                "target study must not be the same as source study in the same storage.",
                file=sys.stderr,
            )
            return 2

        source_storage = create_optuna_storage(optuna, source_storage_url, args.storage_timeout_sec)
        source_artifact_context = ArtifactStore.create_required_context(
            optuna,
            resolve_artifact_dir_from_text(args, args.source_artifact_dir),
            "source",
        )

        source_study = optuna.load_study(
            study_name=args.source_study_name,
            storage=source_storage,
        )
        complete_groups, state_counts_by_key, global_state_counts, skipped_trials = (
            self.collect_source_trial_groups(source_study)
        )
        if not complete_groups:
            print("source study has no COMPLETE TrialParams groups to summarize.", file=sys.stderr)
            return 1

        summary_groups: list[dict] = []
        mixed_context_group_count = 0
        source_complete_trial_count = 0
        source_seed_count = 0

        for index, (key, group) in enumerate(sorted(complete_groups.items(), key=lambda item: item[0])):
            del key
            params = group["params"]
            source_trials = group["trials"]
            source_state_counts = state_counts_by_key.get(self.domain.params_key(params), {})
            group_id = f"g{index:05d}"
            document, user_attrs, objective_values = self.build_summary_group_document(
                group_id,
                params,
                source_trials,
                source_state_counts,
                source_artifact_context,
                source_storage,
            )
            if user_attrs["source_context_mixed"]:
                mixed_context_group_count += 1
                print(
                    "[WARN] source score context is mixed: "
                    f"group_id={group_id} source_trials={user_attrs['source_trial_numbers']}",
                    file=sys.stderr,
                )

            summary_groups.append({
                "params": params,
                "source_trials": source_trials,
                "document": document,
                "user_attrs": user_attrs,
                "objective_values": objective_values,
            })
            source_complete_trial_count += len(source_trials)
            source_seed_count += int(user_attrs["source_seed_count"])

        target_storage = (
            source_storage
            if target_storage_url == source_storage_url
            else create_optuna_storage(optuna, target_storage_url, args.storage_timeout_sec)
        )
        target_artifact_context = ArtifactStore.create_required_context(
            optuna,
            resolve_artifact_dir_from_text(args, args.target_artifact_dir),
            "target",
        )

        if self.optuna_study_exists(optuna, args.target_study_name, target_storage):
            if not args.overwrite_target_study:
                print(
                    "target study already exists. "
                    "Use --overwrite-target-study to delete and recreate it. "
                    f"target_study_name={args.target_study_name}",
                    file=sys.stderr,
                )
                return 2
            self.delete_optuna_study(optuna, args.target_study_name, target_storage)

        target_study = optuna.create_study(
            study_name=args.target_study_name,
            storage=target_storage,
            directions=list(self.summary_objective_directions),
        )
        try:
            target_study.set_metric_names(list(self.summary_objective_names))
        except Exception as e:
            print(f"[WARN] Failed to set target study metric names: {e}", file=sys.stderr)

        target_distributions = self.domain.param_distributions(optuna)
        created_trial_numbers: list[int] = []

        for summary_group in summary_groups:
            params = summary_group["params"]
            params_dict = self.params_to_dict(params)
            document = summary_group["document"]
            user_attrs = summary_group["user_attrs"]
            objective_values = summary_group["objective_values"]
            target_study.enqueue_trial(params_dict, user_attrs=user_attrs, skip_if_exists=False)
            target_trial = target_study.ask(fixed_distributions=target_distributions)
            if target_trial.params != params_dict:
                raise TrialExecutionError(
                    "Target summary trial params did not match the requested group params: "
                    f"group_id={user_attrs['group_id']} requested={params_dict} actual={target_trial.params}"
                )
            try:
                document["target_trial_number"] = int(target_trial.number)
                self.upload_json_trial_artifact(
                    target_artifact_context,
                    target_storage,
                    target_trial,
                    self.group_summary_artifact_filename,
                    document,
                )
                target_study.tell(target_trial, objective_values)
            except Exception:
                try:
                    target_study.tell(target_trial, state=optuna.trial.TrialState.FAIL, skip_if_finished=True)
                except Exception:
                    pass
                raise

            created_trial_numbers.append(int(target_trial.number))

        summary_attrs = {
            "summary_created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "summary_harness": self.harness_name,
            "summary_command": "summarize-study",
            "source_study_name": args.source_study_name,
            "source_storage": source_storage_url,
            "source_artifact_dir": str(source_artifact_context.base_path),
            "target_study_name": args.target_study_name,
            "target_storage": target_storage_url,
            "target_artifact_dir": str(target_artifact_context.base_path),
            "summary_objective_names": list(self.summary_objective_names),
            "summary_objective_directions": list(self.summary_objective_directions),
            "summary_group_count": len(summary_groups),
            "summary_source_complete_trial_count": source_complete_trial_count,
            "summary_source_seed_count": source_seed_count,
            "summary_source_state_counts": dict(sorted(global_state_counts.items())),
            "summary_mixed_context_group_count": mixed_context_group_count,
            "summary_skipped_paramless_trial_count": len(skipped_trials),
        }
        if skipped_trials:
            summary_attrs["summary_skipped_paramless_trials"] = skipped_trials
        set_study_user_attrs(target_study, summary_attrs)

        result = {
            "source_study_name": args.source_study_name,
            "target_study_name": args.target_study_name,
            "source_storage": source_storage_url,
            "target_storage": target_storage_url,
            "group_count": len(summary_groups),
            "source_complete_trial_count": source_complete_trial_count,
            "source_seed_count": source_seed_count,
            "mixed_context_group_count": mixed_context_group_count,
            "created_target_trial_numbers": created_trial_numbers,
            "skipped_paramless_trials": skipped_trials,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    def command_cleanup_running(self, args: argparse.Namespace) -> int:
        """Study に残った RUNNING trial を FAIL に寄せる。"""
        if args.storage_timeout_sec < 0:
            print("storage-timeout-sec must be >= 0.", file=sys.stderr)
            return 2
        try:
            import optuna
        except ImportError:
            print("Optuna is required for cleanup-running. Install optuna in the Python environment.", file=sys.stderr)
            return 2

        storage_url = storage_url_from_arg(args)
        storage = create_optuna_storage(optuna, storage_url, args.storage_timeout_sec)
        study = optuna.load_study(
            study_name=args.study_name,
            storage=storage,
        )
        result = cleanup_running_trials(study, optuna, dry_run=args.dry_run)
        result["study_name"] = args.study_name
        result["storage"] = storage_url
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if not result["errors"] else 2

    def command_run_trial(self, args: argparse.Namespace) -> int:
        """CLI 固定 params を runner で 1 回だけ評価する。"""
        clear_interrupt_flag()
        self.attach_harness_logger(args)
        params = self.domain.params_from_args(args)
        try:
            result = self.execute_single(args, params, args.trial_number, getattr(args, "trial_name", None))
        except TrialPrunedError as e:
            print(str(e), file=sys.stderr)
            return 2
        except TrialExecutionError as e:
            log_harness(args, "ERROR", "trial-failed", console=True, study=args.study_name, error=str(e))
            print(str(e), file=sys.stderr)
            return 2
        print(json.dumps(
            {
                "params": self.params_to_dict(params),
                "context": asdict(result.ctx),
                "score": result.summary["score"],
                "metrics_summary_path": str(Path(result.ctx.artifact_dir) / "metrics_summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    def set_optuna_trial_attrs(self, trial, params, ctx: TrialContext) -> None:
        """Optuna trial に domain params と run path を保存する。"""
        trial.set_user_attr("params", self.params_to_dict(params))
        trial.set_user_attr("cost_tf", ctx.cost_tf)
        trial.set_user_attr("cost_budget", ctx.cost_budget)
        trial.set_user_attr("token_count", ctx.token_count)
        trial.set_user_attr("trial_name", ctx.trial_name)
        trial.set_user_attr("run_name", ctx.run_name)
        trial.set_user_attr("run_dir", ctx.run_dir)
        trial.set_user_attr("artifact_dir", ctx.artifact_dir)
        trial.set_user_attr("config_path", ctx.config_path)

    @staticmethod
    def set_multiseed_trial_attrs(trial, summary: dict) -> None:
        """Dashboard で見る最小 multi-seed 統計を Trial User Attributes に保存する。"""
        trial.set_user_attr("score", summary["score"])
        trial.set_user_attr("score_aggregate", summary["score_aggregate"])
        trial.set_user_attr("score_mean", summary["score_mean"])
        trial.set_user_attr("score_std", summary["score_std"])
        trial.set_user_attr("score_min", summary["score_min"])
        trial.set_user_attr("score_max", summary["score_max"])
        trial.set_user_attr("score_range", summary["score_range"])
        trial.set_user_attr("score_60_80_mean", summary["score_60_80_mean"])
        trial.set_user_attr("score_80_100_mean", summary["score_80_100_mean"])
        trial.set_user_attr("late_slope_mean", summary["late_slope_mean"])
        trial.set_user_attr("late_slope_std", summary["late_slope_std"])
        trial.set_user_attr("seed_count", summary["seed_count"])
        trial.set_user_attr("seed_success_count", summary["seed_success_count"])
        trial.set_user_attr("seed_failure_count", summary["seed_failure_count"])
        trial.set_user_attr("duplicate_params_policy", summary["duplicate_params_policy"])
        trial.set_user_attr("duplicate_count_before", summary["duplicate_count_before"])
        trial.set_user_attr("duplicate_index", summary["duplicate_index"])
        trial.set_user_attr("duplicate_params_max_runs", summary["duplicate_params_max_runs"])
        trial.set_user_attr("duplicate_seed_stride", summary["duplicate_seed_stride"])
        trial.set_user_attr("base_seeds", summary["base_seeds"])
        trial.set_user_attr("effective_seeds", summary["effective_seeds"])
        trial.set_user_attr("duplicate_matched_trials", summary["duplicate_matched_trials"])

    def execute_trial(
        self,
        args: argparse.Namespace,
        params,
        trial_number: int | None,
        trial_name: str | None = None,
        optuna_trial=None,
    ) -> TrialExecutionResult:
        """single seed の runner 実行から metrics summary 生成までを担当する。"""
        ctx = self.make_trial_context(args, params, trial_number, trial_name)
        try:
            score_window = resolve_score_window(args)
        except ValueError as e:
            raise TrialExecutionError(str(e)) from e
        if optuna_trial is not None:
            self.set_optuna_trial_attrs(optuna_trial, params, ctx)

        if not getattr(args, "_suppress_trial_start_log", False):
            log_harness(
                args,
                "INFO",
                "trial-start",
                console=True,
                study=ctx.study_name,
                trial=ctx.trial_number,
                run=ctx.run_name,
                cost_tf=ctx.cost_tf,
                cost_budget=ctx.cost_budget,
            )

        if ctx.cost_tf > ctx.cost_budget:
            log_harness(
                args,
                "WARN",
                "trial-pruned",
                console=True,
                study=ctx.study_name,
                trial=ctx.trial_number,
                run=ctx.run_name,
                reason=f"cost_tf={ctx.cost_tf} > cost_budget={ctx.cost_budget}",
            )
            raise TrialPrunedError(f"cost_tf={ctx.cost_tf} > cost_budget={ctx.cost_budget}")

        self.write_trial_files(args, params, ctx)
        returncode = RunnerProcessManager.run(args, ctx)
        if optuna_trial is not None:
            optuna_trial.set_user_attr("returncode", returncode)
        if returncode != 0:
            raise TrialFailedError(f"runner failed: returncode={returncode}")

        metrics_path = Path(ctx.run_dir) / "metrics.jsonl"
        if not metrics_path.exists():
            raise TrialFailedError(f"metrics.jsonl not found: {metrics_path}")

        summary = MetricsSummarizer.summarize(metrics_path, score_window, self.metrics_spec)
        MetricsSummarizer.write_single_seed_summary(summary, Path(ctx.artifact_dir))
        if optuna_trial is not None:
            for tag, data in summary["tags"].items():
                optuna_trial.set_user_attr(f"metric:{tag}:mean", data["mean"])
                optuna_trial.set_user_attr(f"metric:{tag}:last", data["last"])
        if summary["score"] is None:
            raise TrialFailedError("primary score is unavailable in the selected window")
        if optuna_trial is not None:
            optuna_trial.set_user_attr("score", summary["score"])
        if not getattr(args, "_suppress_trial_start_log", False):
            log_harness(
                args,
                "INFO",
                "trial-complete",
                console=True,
                study=ctx.study_name,
                trial=ctx.trial_number,
                run=ctx.run_name,
                score=summary["score"],
            )
        return TrialExecutionResult(ctx=ctx, summary=summary)

    def execute_study_trial(
        self,
        args: argparse.Namespace,
        params,
        trial_number: int,
        study=None,
        optuna_trial=None,
        optuna_artifact_context: OptunaArtifactContext | None = None,
    ) -> TrialExecutionResult:
        """run-study の 1 Optuna trial を multi-seed aggregate として評価する。"""
        base_seeds = parse_seed_list(args.seeds)
        duplicate_info = self.resolve_duplicate_params_info(args, params, study, trial_number, base_seeds)
        seeds = duplicate_info.effective_seeds
        study_args = apply_duplicate_info_to_args(args, duplicate_info)
        ctx = self.make_trial_context(args, params, trial_number)
        try:
            resolve_score_window(args)
        except ValueError as e:
            raise TrialExecutionError(str(e)) from e
        if optuna_trial is not None:
            self.set_optuna_trial_attrs(optuna_trial, params, ctx)

        log_harness(
            args,
            "INFO",
            "trial-start",
            console=True,
            study=ctx.study_name,
            trial=trial_number,
            run=ctx.run_name,
            cost_tf=ctx.cost_tf,
            cost_budget=ctx.cost_budget,
            duplicate_count=duplicate_info.duplicate_count_before,
            seeds=seeds,
        )

        self.write_representative_trial_files(study_args, params, ctx, seeds, duplicate_info)
        seed_runs: list[dict] = []
        if duplicate_info.pruned_by_duplicate:
            error = duplicate_info.prune_reason or "duplicate params"
            summary = self.build_multiseed_summary(study_args, params, ctx, seeds, seed_runs, duplicate_info, error)
            self.write_multiseed_summary_files(summary, Path(ctx.artifact_dir))
            if optuna_trial is not None:
                self.set_multiseed_trial_attrs(optuna_trial, summary)
                ArtifactStore.register_trial_artifacts(optuna_trial, optuna_artifact_context, Path(ctx.artifact_dir))
            log_harness(
                args,
                "WARN",
                "trial-pruned",
                console=True,
                study=ctx.study_name,
                trial=trial_number,
                run=ctx.run_name,
                reason=error,
            )
            raise TrialPrunedError(error)

        if ctx.cost_tf > ctx.cost_budget:
            error = f"cost_tf={ctx.cost_tf} > cost_budget={ctx.cost_budget}"
            summary = self.build_multiseed_summary(study_args, params, ctx, seeds, seed_runs, duplicate_info, error)
            self.write_multiseed_summary_files(summary, Path(ctx.artifact_dir))
            if optuna_trial is not None:
                self.set_multiseed_trial_attrs(optuna_trial, summary)
                ArtifactStore.register_trial_artifacts(optuna_trial, optuna_artifact_context, Path(ctx.artifact_dir))
            log_harness(
                args,
                "WARN",
                "trial-pruned",
                console=True,
                study=ctx.study_name,
                trial=trial_number,
                run=ctx.run_name,
                reason=error,
            )
            raise TrialPrunedError(error)

        for seed in seeds:
            seed_trial = seed_trial_name(ctx.trial_name, seed)
            seed_args = args_with_seed(study_args, seed)
            seed_args._suppress_trial_start_log = True
            seed_ctx = self.make_trial_context(seed_args, params, trial_number, seed_trial)
            log_harness(
                args,
                "INFO",
                "seed-start",
                console=True,
                study=ctx.study_name,
                trial=trial_number,
                seed=seed,
                run=seed_ctx.run_name,
            )
            try:
                result = self.execute_single(seed_args, params, trial_number, seed_trial)
            except (TrialPrunedError, TrialFailedError, ValueError) as e:
                seed_runs.append(self.make_seed_run_record(seed, seed_ctx, "failed", error=str(e)))
                summary = self.build_multiseed_summary(study_args, params, ctx, seeds, seed_runs, duplicate_info, str(e))
                self.write_multiseed_summary_files(summary, Path(ctx.artifact_dir))
                if optuna_trial is not None:
                    self.set_multiseed_trial_attrs(optuna_trial, summary)
                    ArtifactStore.register_trial_artifacts(optuna_trial, optuna_artifact_context, Path(ctx.artifact_dir))
                log_harness(
                    args,
                    "ERROR",
                    "seed-failed",
                    console=True,
                    study=ctx.study_name,
                    trial=trial_number,
                    seed=seed,
                    run=seed_ctx.run_name,
                    error=str(e),
                )
                if isinstance(e, (TrialPrunedError, TrialFailedError)):
                    raise
                raise TrialFailedError(str(e)) from e
            seed_runs.append(self.make_seed_run_record(seed, result.ctx, "complete", score=result.score, summary=result.summary))
            log_harness(
                args,
                "INFO",
                "seed-complete",
                console=True,
                study=ctx.study_name,
                trial=trial_number,
                seed=seed,
                run=result.ctx.run_name,
                score=result.score,
            )

        summary = self.build_multiseed_summary(study_args, params, ctx, seeds, seed_runs, duplicate_info)
        self.write_multiseed_summary_files(summary, Path(ctx.artifact_dir))
        if optuna_trial is not None:
            self.set_multiseed_trial_attrs(optuna_trial, summary)
            ArtifactStore.register_trial_artifacts(optuna_trial, optuna_artifact_context, Path(ctx.artifact_dir))
        if summary["score"] is None:
            raise TrialFailedError("aggregate score is unavailable")
        log_harness(
            args,
            "INFO",
            "trial-complete",
            console=True,
            study=ctx.study_name,
            trial=trial_number,
            run=ctx.run_name,
            score=summary["score"],
        )
        return TrialExecutionResult(ctx=ctx, summary=summary)

    def execute_single(
        self,
        args: argparse.Namespace,
        params,
        trial_number: int | None,
        trial_name: str | None = None,
        optuna_trial=None,
    ) -> TrialExecutionResult:
        """run-trial と seed run の 1 runner 実行を行う。"""
        return self.execute_trial(args, params, trial_number, trial_name, optuna_trial)

    def execute_multi_seed(
        self,
        args: argparse.Namespace,
        params,
        trial_number: int,
        study=None,
        optuna_trial=None,
        optuna_artifact_context: OptunaArtifactContext | None = None,
    ) -> TrialExecutionResult:
        """run-study の 1 Optuna trial を multi-seed aggregate として実行する。"""
        return self.execute_study_trial(args, params, trial_number, study, optuna_trial, optuna_artifact_context)

    def objective(
        self,
        trial,
        args: argparse.Namespace,
        study,
        optuna_artifact_context: OptunaArtifactContext | None,
    ) -> float:
        """Optuna objective。prune/fail の意味づけをここで Optuna state に接続する。"""
        import optuna

        params = self.domain.suggest_params(trial, args)
        try:
            return self.execute_multi_seed(
                args,
                params,
                trial.number,
                study=study,
                optuna_trial=trial,
                optuna_artifact_context=optuna_artifact_context,
            ).score
        except TrialPrunedError as e:
            raise optuna.TrialPruned(str(e)) from e
        except (KeyboardInterrupt, SystemExit):
            raise
        except TrialFailedError as e:
            traceback_text = traceback.format_exc()
            set_trial_failure_attrs(trial, e, traceback_text)
            log_harness_exception(args, "trial-failed", e, console=True, study=args.study_name, trial=trial.number)
            raise
        except Exception as e:
            traceback_text = traceback.format_exc()
            set_trial_failure_attrs(trial, e, traceback_text)
            log_harness_exception(args, "trial-unexpected-error", e, console=True, study=args.study_name, trial=trial.number)
            raise TrialFailedError(str(e)) from e

    def command_run_study(self, args: argparse.Namespace) -> int:
        """Optuna study 全体を実行し、trial ごとに runner を別 process で起動する。"""
        clear_interrupt_flag()
        self.attach_harness_logger(args)
        resolve_score_window(args)
        parse_seed_list(args.seeds)
        if args.duplicate_params_max_runs < 0:
            print("duplicate-params-max-runs must be >= 0.", file=sys.stderr)
            return 2
        if args.duplicate_seed_stride < 0:
            print("duplicate-seed-stride must be >= 0.", file=sys.stderr)
            return 2
        if args.storage_timeout_sec < 0:
            print("storage-timeout-sec must be >= 0.", file=sys.stderr)
            return 2
        if args.heartbeat_interval_sec < 0:
            print("heartbeat-interval-sec must be >= 0.", file=sys.stderr)
            return 2
        if args.heartbeat_interval_sec > 0 and args.heartbeat_grace_period_sec <= 0:
            print("heartbeat-grace-period-sec must be > 0 when heartbeat is enabled.", file=sys.stderr)
            return 2
        try:
            import optuna
        except ImportError:
            print("Optuna is required for run-study. Install optuna in the Python environment.", file=sys.stderr)
            return 2

        log_harness(
            args,
            "INFO",
            "study-start",
            console=True,
            study=args.study_name,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            heartbeat_interval_sec=args.heartbeat_interval_sec,
            heartbeat_grace_period_sec=args.heartbeat_grace_period_sec,
        )
        sampler = optuna.samplers.TPESampler(
            seed=args.sampler_seed,
            n_startup_trials=args.n_startup_trials,
            constant_liar=args.constant_liar,
        )
        fixed_params_for_sampler = getattr(self.domain, "fixed_params_for_sampler", None)
        if fixed_params_for_sampler is not None:
            fixed_params = fixed_params_for_sampler(args)
            if fixed_params:
                sampler = optuna.samplers.PartialFixedSampler(fixed_params, sampler)
        storage_url = storage_url_from_arg(args)
        storage = create_optuna_storage(
            optuna,
            storage_url,
            args.storage_timeout_sec,
            args.heartbeat_interval_sec,
            args.heartbeat_grace_period_sec,
        )
        optuna_artifact_context = ArtifactStore.create_optional_context(args)
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            sampler=sampler,
            direction="maximize",
            load_if_exists=True,
        )
        set_study_user_attrs(study, self.build_study_user_attrs(args, storage_url))
        fail_stale_trials_if_enabled(args, optuna, study, "before-optimize")
        try:
            study.optimize(
                lambda trial: self.objective(trial, args, study, optuna_artifact_context),
                n_trials=args.n_trials,
                n_jobs=args.n_jobs,
                catch=(TrialFailedError,),
            )
        except KeyboardInterrupt:
            set_interrupt_flag()
            terminated_pids = RunnerProcessManager.terminate_active()
            try:
                study.stop()
            except Exception:
                pass
            cleanup_result = cleanup_running_trials(study, optuna, dry_run=False)
            log_harness(
                args,
                "WARN",
                "study-interrupted",
                console=True,
                study=args.study_name,
                terminated_runner_pids=terminated_pids,
                cleanup=cleanup_result,
            )
            print(json.dumps(
                {
                    "interrupted": True,
                    "terminated_runner_pids": terminated_pids,
                    "cleanup": cleanup_result,
                },
                ensure_ascii=False,
                indent=2,
            ), file=sys.stderr)
            return 130
        except Exception as e:
            terminated_pids = RunnerProcessManager.terminate_active()
            cleanup_result = cleanup_running_trials(study, optuna, dry_run=False)
            log_harness_exception(
                args,
                "study-failed",
                e,
                console=True,
                study=args.study_name,
                terminated_runner_pids=terminated_pids,
                cleanup=cleanup_result,
            )
            print(json.dumps(
                {
                    "failed": True,
                    "error": str(e),
                    "terminated_runner_pids": terminated_pids,
                    "cleanup": cleanup_result,
                },
                ensure_ascii=False,
                indent=2,
            ), file=sys.stderr)
            return 1
        fail_stale_trials_if_enabled(args, optuna, study, "after-optimize")
        complete_trials = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if complete_trials:
            print(f"best_value={study.best_value} best_trial={study.best_trial.number}")
        else:
            print("best_value unavailable: no completed trials")
        log_harness(args, "INFO", "study-complete", console=True, study=args.study_name, complete_trials=len(complete_trials))
        return 0



def add_runner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runner-exe",
        default="apps/runner/bin/Release/AnetRLRunner.exe",
        help="AnetRLRunner.exe のパス。相対時は repo root 基準。未指定時は Release runner。",
    )
    parser.add_argument("--timeout-sec", type=int, default=0, help="runner 1 trial の timeout 秒。0 は timeout なし。")


def add_score_window_args(parser: argparse.ArgumentParser, primary_score: bool = True) -> None:
    description_prefix = "primary score 集計 window" if primary_score else "集計 window"
    parser.add_argument(
        "--window-start",
        default="80%",
        help=f"{description_prefix} の開始。絶対 step、負数相対 step、exp_exit_step 比率の %% 指定を使える。",
    )
    parser.add_argument(
        "--window-end",
        default="100%",
        help=f"{description_prefix} の終了。絶対 step、負数相対 step、exp_exit_step 比率の %% 指定を使える。",
    )


def localize_parser(parser: argparse.ArgumentParser, positionals_title: str = "引数") -> None:
    parser.add_argument("-h", "--help", action="help", help="このヘルプを表示して終了する。")
    parser._positionals.title = positionals_title
    parser._optionals.title = "オプション"

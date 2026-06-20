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
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
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

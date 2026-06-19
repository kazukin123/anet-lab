#!/usr/bin/env python3
"""DropMerge NN 構成探索用の Optuna harness。

Optuna は C++ runner の外側に置き、このスクリプトが trial ごとの main config
生成、AnetRLRunner の ``--config`` 起動、``metrics.jsonl`` の採点を担当する。
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable


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


PRIMARY_TAGS = (
    "21_eval/03_target_reward_ema",
    "21_eval/04_policy_reward_ema",
)

# primary score には入れず、trial の解釈や後段分析のために保存する補助指標。
SUPPLEMENTAL_TAGS = (
    "51_eval1/61_ep_maxrank_mean_ema",
    "52_eval2/61_ep_maxrank_mean_ema",
    "51_eval1/62_ep_frct_mean_ema",
    "52_eval2/62_ep_frct_mean_ema",
    "51_eval1/83_tr_blk_mean_ema",
    "52_eval2/83_tr_blk_mean_ema",
    "51_eval1/84_tr_timeout_max_ema",
    "52_eval2/84_tr_timeout_mean_ema",
    "51_eval1/85_tr_nolg_mean_ema",
    "52_eval2/85_tr_nolg_mean_ema",
    "90_perf/12_exp_step_per_sec",
    "90_perf/22_exp_step_per_sec_ema",
    "90_perf/90_elapse_hour",
)

BUDGETS = {
    "small": 35_000_000.0,
    "medium": 70_000_000.0,
}

TRIAL_NAME_PATTERN = re.compile(r"^t(\d{5})$")
SCORE_AGGREGATES = ("mean", "median", "mean-minus-std", "min")
DUPLICATE_PARAMS_POLICIES = ("allow", "prune", "reseed")
TRIAL_PARAM_CHOICES = {
    "cnn_channels": [48, 64],
    "res_blocks": [2, 4],
    "token_mode": ["current", "stronger"],
    "d_model": [96, 128, 192],
    "transformer_layers": [2, 4],
    "ff_mult": [2, 4],
    "trunk_width": [1024, 2048],
    "head_width": [512, 1024],
}
SUMMARY_OBJECTIVE_NAMES = ("mean", "range", "std")
SUMMARY_OBJECTIVE_DIRECTIONS = ("maximize", "minimize", "minimize")
GROUP_SUMMARY_ARTIFACT_FILENAME = "group_summary.json"
LATE_SCORE_WINDOWS = (
    ("score_60_80", "60%", "80%"),
    ("score_80_100", "80%", "100%"),
)
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


@dataclass(frozen=True)
class TrialParams:
    cnn_channels: int
    res_blocks: int
    token_mode: str
    d_model: int
    transformer_layers: int
    ff_mult: int
    trunk_width: int
    head_width: int


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


def repo_root_from_script() -> Path:
    # apps/runner/tools/dropmerge_optuna.py から repo root へ戻る。
    return Path(__file__).resolve().parents[3]


def runner_root(repo_root: Path) -> Path:
    return repo_root / "apps" / "runner"


def default_main_config(repo_root: Path) -> str:
    return "_main.txt"


def config_include_line(path_text: str) -> str:
    path = Path(path_text)
    if path.is_absolute():
        return f"$include \"{path.resolve().as_posix()}\""
    normalized = path_text.replace("\\", "/")
    return f"$include <{normalized}>"


def conv_out(size: int, stride: int = 2, kernel: int = 3, padding: int = 1, dilation: int = 1) -> int:
    return math.floor((size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)


def token_dims(token_mode: str, grid_width: int = 58, grid_height: int = 46) -> tuple[int, int]:
    # DropMerge G5846 の既定 grid から、trial の stride 構成に応じた token 解像度を概算する。
    width = conv_out(grid_width)
    height = conv_out(grid_height)
    if token_mode == "hr":
        return width, height
    width = conv_out(width)
    height = conv_out(height)
    if token_mode == "current":
        return width, height
    if token_mode == "stronger":
        return conv_out(width), conv_out(height)
    raise ValueError(f"unknown token_mode: {token_mode}")


def token_count(token_mode: str) -> int:
    width, height = token_dims(token_mode)
    return width * height


def cost_tf(params: TrialParams, k: float) -> float:
    # 実時間そのものではなく、Transformer の N/M/L 支配項を見るための事前 proxy。
    n = token_count(params.token_mode)
    m = params.d_model
    l = params.transformer_layers
    return float(l * ((n * n * m) + (k * n * m * m)))


def suggest_params(trial) -> TrialParams:
    return TrialParams(
        cnn_channels=trial.suggest_categorical("cnn_channels", TRIAL_PARAM_CHOICES["cnn_channels"]),
        res_blocks=trial.suggest_categorical("res_blocks", TRIAL_PARAM_CHOICES["res_blocks"]),
        token_mode=trial.suggest_categorical("token_mode", TRIAL_PARAM_CHOICES["token_mode"]),
        d_model=trial.suggest_categorical("d_model", TRIAL_PARAM_CHOICES["d_model"]),
        transformer_layers=trial.suggest_categorical("transformer_layers", TRIAL_PARAM_CHOICES["transformer_layers"]),
        ff_mult=trial.suggest_categorical("ff_mult", TRIAL_PARAM_CHOICES["ff_mult"]),
        trunk_width=trial.suggest_categorical("trunk_width", TRIAL_PARAM_CHOICES["trunk_width"]),
        head_width=trial.suggest_categorical("head_width", TRIAL_PARAM_CHOICES["head_width"]),
    )


def trial_params_from_mapping(mapping: dict, *, source: str = "params") -> TrialParams:
    missing = [
        name
        for name in TrialParams.__dataclass_fields__
        if mapping.get(name) is None
    ]
    if missing:
        raise ValueError(f"Invalid {source}: missing TrialParams fields. missing={missing}")

    try:
        return TrialParams(
            cnn_channels=int(mapping["cnn_channels"]),
            res_blocks=int(mapping["res_blocks"]),
            token_mode=str(mapping["token_mode"]),
            d_model=int(mapping["d_model"]),
            transformer_layers=int(mapping["transformer_layers"]),
            ff_mult=int(mapping["ff_mult"]),
            trunk_width=int(mapping["trunk_width"]),
            head_width=int(mapping["head_width"]),
        )
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid {source}: TrialParams fields have invalid values. params={mapping}") from e


def trial_param_distributions(optuna) -> dict:
    return {
        name: optuna.distributions.CategoricalDistribution(choices)
        for name, choices in TRIAL_PARAM_CHOICES.items()
    }


def validate_trial_params_in_search_space(params: TrialParams) -> None:
    values = asdict(params)
    invalid = {
        name: value
        for name, value in values.items()
        if value not in TRIAL_PARAM_CHOICES[name]
    }
    if invalid:
        raise ValueError(f"Source trial params are outside current Optuna search space: {invalid}")


def params_from_args(args: argparse.Namespace) -> TrialParams:
    return TrialParams(
        cnn_channels=args.cnn_channels,
        res_blocks=args.res_blocks,
        token_mode=args.token_mode,
        d_model=args.d_model,
        transformer_layers=args.transformer_layers,
        ff_mult=args.ff_mult,
        trunk_width=args.trunk_width,
        head_width=args.head_width,
    )


def resolve_budget(args: argparse.Namespace) -> tuple[str, float]:
    if args.cost_budget is not None:
        return args.budget, float(args.cost_budget)
    return args.budget, BUDGETS[args.budget]


def validate_name_part(name: str, option_name: str) -> None:
    if "/" in name or "\\" in name:
        raise ValueError(f"Invalid {option_name}: path separator is not allowed. value={name}")


def trial_number_from_name(trial_name: str) -> int | None:
    match = TRIAL_NAME_PATTERN.fullmatch(trial_name)
    return int(match.group(1)) if match else None


def resolve_runner_relative_path(repo_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return runner_root(repo_root) / path


def scan_existing_trial_numbers(args: argparse.Namespace, budget_name: str) -> list[int]:
    repo_root = Path(args.repo_root).resolve()
    # run_name が <study>_tNNNNN で揃っている前提で、Optuna run root 直下を見る。
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


def next_trial_number(args: argparse.Namespace, budget_name: str) -> int:
    existing = scan_existing_trial_numbers(args, budget_name)
    return max(existing) + 1 if existing else 0


def resolve_trial_identity(
    args: argparse.Namespace,
    budget_name: str,
    trial_number: int | None,
    trial_name_override: str | None,
) -> tuple[int, str]:
    trial_name = trial_name_override or getattr(args, "trial_name", None)
    if trial_number is not None:
        return trial_number, trial_name or f"t{trial_number:05d}"

    if trial_name:
        number_from_name = trial_number_from_name(trial_name)
        if number_from_name is not None:
            return number_from_name, trial_name
        return next_trial_number(args, budget_name), trial_name

    number = next_trial_number(args, budget_name)
    return number, f"t{number:05d}"


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


def make_trial_context(
    args: argparse.Namespace,
    params: TrialParams,
    trial_number: int | None,
    trial_name_override: str | None = None,
) -> TrialContext:
    repo_root = Path(args.repo_root).resolve()
    run_root = runner_root(repo_root)
    budget_name, budget_value = resolve_budget(args)
    cost = cost_tf(params, args.cost_k)
    n = token_count(params.token_mode)
    validate_name_part(args.study_name, "--study-name")
    trial_number, trial_name = resolve_trial_identity(args, budget_name, trial_number, trial_name_override)
    validate_name_part(trial_name, "--trial-name")
    run_name = f"{args.study_name}_{trial_name}"
    # viewer と手動掃除の単位を揃えるため、Optuna run は runs_optuna 直下へフラットに集める。
    runs_dir = args.runs_dir.format(study=args.study_name, budget=budget_name, trial=trial_name, run=run_name)
    run_dir = run_root / runs_dir / run_name
    # runner 自身が run root に config.txt/stdout.log/stderr.log を作るため、harness artifact は隔離する。
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


def resolve_runs_root(args: argparse.Namespace) -> Path:
    repo_root = Path(args.repo_root).resolve()
    budget_name, _ = resolve_budget(args)
    runs_dir = args.runs_dir.format(study=args.study_name, budget=budget_name, trial="", run="")
    return resolve_runner_relative_path(repo_root, runs_dir)


def attach_harness_logger(args: argparse.Namespace) -> HarnessLogger:
    existing = getattr(args, "_harness_logger", None)
    if existing is not None:
        return existing
    logger = HarnessLogger(resolve_runs_root(args) / "harness.log")
    setattr(args, "_harness_logger", logger)
    return logger


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


def build_run_study_copy_args(args: argparse.Namespace, storage_url: str) -> str:
    # Dashboard からそのまま貼れるよう、run-study の再開に必要な実効値を明示する。
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
    append_cli_arg(parts, "--nhead", args.nhead)
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


def build_study_user_attrs(args: argparse.Namespace, storage_url: str) -> dict:
    budget_name, cost_budget = resolve_budget(args)
    window = resolve_score_window(args)
    seeds = parse_seed_list(args.seeds)
    storage_timeout_sec = getattr(args, "storage_timeout_sec", 120.0)
    attrs = {
        "00_last_run_study_args": build_run_study_copy_args(args, storage_url),
        "last_launch_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_harness": "dropmerge_optuna",
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


def set_study_user_attrs(study, attrs: dict) -> None:
    for key, value in attrs.items():
        study.set_user_attr(key, value)


def seed_trial_name(trial_name: str, seed: int) -> str:
    return f"{trial_name}_s{seed}"


def args_with_seed(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    seed_args = argparse.Namespace(**vars(args))
    seed_args.seed = seed
    return seed_args


def trial_params_key(params: TrialParams) -> tuple[tuple[str, object], ...]:
    return tuple(asdict(params).items())


def trial_params_key_from_mapping(mapping: dict) -> tuple[tuple[str, object], ...]:
    return tuple((name, mapping.get(name)) for name in TrialParams.__dataclass_fields__)


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


def duplicate_relevant_trial_numbers(study, params: TrialParams, current_trial_number: int) -> list[int]:
    if study is None:
        return []

    current_key = trial_params_key(params)
    matched: list[int] = []
    for trial in getattr(study, "trials", []):
        if getattr(trial, "number", None) == current_trial_number:
            continue
        if trial_state_name(trial) not in ("COMPLETE", "RUNNING"):
            continue
        if trial_params_key_from_mapping(getattr(trial, "params", {})) == current_key:
            matched.append(int(trial.number))
    return matched


def shifted_seeds(seeds: list[int], duplicate_index: int, stride: int) -> list[int]:
    offset = duplicate_index * stride
    return [seed + offset for seed in seeds]


def resolve_duplicate_params_info(
    args: argparse.Namespace,
    params: TrialParams,
    study,
    trial_number: int,
    base_seeds: list[int],
) -> DuplicateParamsInfo:
    policy = args.duplicate_params_policy
    matched_trials = duplicate_relevant_trial_numbers(study, params, trial_number)
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


def generated_structure(params: TrialParams) -> str:
    # Flatten family 固定。token_mode に応じて ConvDown の回数だけを変える。
    blocks = [
        "OptConvInit",
        f"OptResBlock(*{params.res_blocks})",
    ]
    if params.token_mode in ("current", "stronger"):
        blocks.extend(["OptConvDown", "SiLU"])
    if params.token_mode == "stronger":
        blocks.extend(["OptConvDown2", "SiLU"])
    blocks.extend([
        "OptViTProj",
        "SiLU",
        "PosEmbed2D",
        "OptTransEnc",
        "Flatten",
        "OptLinear",
        "SiLU",
    ])
    return " > ".join(blocks)


def render_config(repo_root: Path, params: TrialParams, ctx: TrialContext, args: argparse.Namespace) -> str:
    ff_dim = params.d_model * params.ff_mult
    nhead = args.nhead
    if params.d_model % nhead != 0:
        raise ValueError(f"d_model must be divisible by nhead: d_model={params.d_model} nhead={nhead}")

    lines = [
        "# Generated by apps/runner/tools/dropmerge_optuna.py",
        f"# study={ctx.study_name} budget={ctx.budget_name} trial={ctx.trial_number}",
        config_include_line(args.base_config),
        config_include_line(args.extra_config),
        "",
        "app.$ = app.batchrun > P",
        f"app.run_name = {ctx.run_name}",
        f"app.runs_dir = {ctx.runs_dir}",
        f"app.batchrun.exp_exit_step = {args.exp_exit_step}",
        f"train.seed = {args.seed}",
        "",
        "net.block.[OptConvInit].type = Conv2d",
        f"net.block.[OptConvInit].conv.out_channels = {params.cnn_channels}",
        "net.block.[OptConvInit].conv.kernel_size = 3",
        "net.block.[OptConvInit].conv.padding = 1",
        "net.block.[OptConvInit].conv.stride = 2",
        "net.block.[OptConvInit].init.mode = 2",
        "",
        "net.block.[OptResBlock].type = ResBlock",
        f"net.block.[OptResBlock].res.channels = {params.cnn_channels}",
        "net.block.[OptResBlock].res.kernel_size = 3",
        "net.block.[OptResBlock].res.activation = silu",
        "net.block.[OptResBlock].res.activation_mode = pre",
        "",
        "net.block.[OptConvDown].type = Conv2d",
        f"net.block.[OptConvDown].conv.out_channels = {params.cnn_channels}",
        "net.block.[OptConvDown].conv.kernel_size = 3",
        "net.block.[OptConvDown].conv.stride = 2",
        "net.block.[OptConvDown].conv.padding = 1",
        "",
        "net.block.[OptConvDown2].type = Conv2d",
        f"net.block.[OptConvDown2].conv.out_channels = {params.cnn_channels}",
        "net.block.[OptConvDown2].conv.kernel_size = 3",
        "net.block.[OptConvDown2].conv.stride = 2",
        "net.block.[OptConvDown2].conv.padding = 1",
        "",
        "net.block.[OptViTProj].type = Conv2d",
        f"net.block.[OptViTProj].conv.out_channels = {params.d_model}",
        "net.block.[OptViTProj].conv.kernel_size = 1",
        "net.block.[OptViTProj].conv.stride = 1",
        "net.block.[OptViTProj].conv.padding = 0",
        "",
        "net.block.[OptTransEnc].type = TransformerEncoder",
        f"net.block.[OptTransEnc].tf.d_model = {params.d_model}",
        f"net.block.[OptTransEnc].tf.nhead = {nhead}",
        f"net.block.[OptTransEnc].tf.num_layers = {params.transformer_layers}",
        f"net.block.[OptTransEnc].tf.dim_feedforward = {ff_dim}",
        "net.block.[OptTransEnc].tf.norm_first = true",
        "net.block.[OptTransEnc].tf.use_sdpa = true",
        "net.block.[OptTransEnc].tf.activation = gelu",
        "",
        "net.block.[OptLinear].type = Linear",
        f"net.block.[OptLinear].linear.out_features = {params.trunk_width}",
        "net.block.[OptLinear].linear.bias = true",
        "net.block.[OptLinear].init.mode = 3",
        "net.block.[OptLinear].init.manual_gain = 1.0",
        "",
        "net.block.[OptHeadFC].type = Linear",
        f"net.block.[OptHeadFC].linear.out_features = {params.head_width}",
        "net.block.[OptHeadFC].init.mode = 1",
        "",
        "net.branch.[main_feature].$ = net.branch.OptunaDropMerge",
        "net.branch.OptunaDropMerge.bind = grid, vector_feature",
        f"net.branch.OptunaDropMerge.structure = {generated_structure(params)}",
        "net.branch.[value_stream].structure = OptHeadFC > SiLU",
        "net.branch.[adv_stream].structure = OptHeadFC > SiLU",
        "",
    ]
    return "\n".join(lines)


def make_manifest(args: argparse.Namespace, params: TrialParams, ctx: TrialContext, extra: dict | None = None) -> dict:
    manifest = {
        "params": asdict(params),
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
        "primary_tags": list(PRIMARY_TAGS),
        "supplemental_tags": list(SUPPLEMENTAL_TAGS),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest_file(manifest: dict, artifact_dir: Path) -> None:
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_trial_files(args: argparse.Namespace, params: TrialParams, ctx: TrialContext) -> None:
    # trial 再現に必要な config と manifest を runner 起動前に必ず残す。
    repo_root = Path(args.repo_root).resolve()
    artifact_dir = Path(ctx.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config_text = render_config(repo_root, params, ctx, args)
    Path(ctx.config_path).write_text(config_text, encoding="utf-8", newline="\n")
    write_manifest_file(make_manifest(args, params, ctx), artifact_dir)


def write_representative_trial_files(
    args: argparse.Namespace,
    params: TrialParams,
    ctx: TrialContext,
    seeds: list[int],
    duplicate_info: DuplicateParamsInfo,
) -> None:
    # multi-seed trial の代表 artifact。実 runner run ではないため config.txt は作らない。
    artifact_dir = Path(ctx.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(args, params, ctx, {
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
    write_manifest_file(manifest, artifact_dir)


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


def summarize_metrics_window(metrics_path: Path, window: ScoreWindow) -> tuple[dict[str, dict[str, float | int]], float | None]:
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
        for tag in PRIMARY_TAGS
        if tag in tag_summary
    ]
    score = mean(primary_means) if len(primary_means) == len(PRIMARY_TAGS) else None
    return tag_summary, score


def summarize_metrics(metrics_path: Path, window: ScoreWindow) -> dict:
    tag_summary, score = summarize_metrics_window(metrics_path, window)
    analysis_windows: dict[str, dict[str, object]] = {}
    for name, raw_start, raw_end in LATE_SCORE_WINDOWS:
        late_window = resolve_score_window_from_raw(raw_start, raw_end, window.exp_exit_step, name)
        _, late_score = summarize_metrics_window(metrics_path, late_window)
        analysis_windows[name] = {
            "window_start": late_window.start,
            "window_end": late_window.end,
            "window_start_raw": late_window.raw_start,
            "window_end_raw": late_window.raw_end,
            "exp_exit_step": late_window.exp_exit_step,
            "score": late_score,
        }

    score_60_80 = analysis_windows["score_60_80"]["score"]
    score_80_100 = analysis_windows["score_80_100"]["score"]
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


def write_multiseed_summary_files(summary: dict, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "multiseed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (artifact_dir / "seed_runs.json").write_text(
        json.dumps(build_seed_runs_document(summary), ensure_ascii=False, indent=2) + "\n",
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


def build_seed_runs_document(summary: dict) -> dict:
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


def make_seed_run_record(
    seed: int,
    ctx: TrialContext,
    status: str,
    score: float | None = None,
    summary: dict | None = None,
    error: str | None = None,
) -> dict:
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
    args: argparse.Namespace,
    params: TrialParams,
    ctx: TrialContext,
    seeds: list[int],
    seed_runs: list[dict],
    duplicate_info: DuplicateParamsInfo,
    error: str | None = None,
) -> dict:
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
        "params": asdict(params),
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


def command_dry_run(args: argparse.Namespace) -> int:
    params = params_from_args(args)
    ctx = make_trial_context(args, params, args.trial_number)
    score_window = resolve_score_window(args)
    try:
        write_trial_files(args, params, ctx)
    except ValueError as e:
        raise TrialExecutionError(str(e)) from e
    result = {
        "params": asdict(params),
        "context": asdict(ctx),
        "score_window": asdict(score_window),
        "pruned_by_cost": ctx.cost_tf > ctx.cost_budget,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ctx.cost_tf <= ctx.cost_budget else 2


def command_summarize(args: argparse.Namespace) -> int:
    summary = summarize_metrics(Path(args.metrics_jsonl), resolve_score_window(args))
    if args.output_dir:
        write_summary_files(summary, Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["score"] is not None else 2


def optuna_study_exists(optuna, study_name: str, storage) -> bool:
    get_all_study_names = getattr(optuna.study, "get_all_study_names", None)
    if get_all_study_names is None:
        get_all_study_names = getattr(optuna, "get_all_study_names")
    return study_name in get_all_study_names(storage=storage)


def delete_optuna_study(optuna, study_name: str, storage) -> None:
    delete_study = getattr(optuna.study, "delete_study", None)
    if delete_study is None:
        delete_study = getattr(optuna, "delete_study")
    delete_study(study_name=study_name, storage=storage)


def load_trial_json_artifact(
    artifact_context: OptunaArtifactContext,
    storage,
    trial,
    filename: str,
    *,
    required: bool,
) -> dict | None:
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


def upload_json_trial_artifact(
    artifact_context: OptunaArtifactContext,
    storage,
    trial,
    filename: str,
    payload: dict,
) -> None:
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


def source_trial_score(trial) -> float:
    value = getattr(trial, "value", None)
    if value is None:
        value = getattr(trial, "user_attrs", {}).get("score")
    if value is None:
        raise TrialExecutionError(f"Source COMPLETE trial has no objective score: trial={trial.number}")
    return float(value)


def source_score_context(trial, seed_runs_doc: dict, manifest_doc: dict | None) -> dict:
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


def score_context_fingerprint(context: dict) -> str:
    comparable = {
        key: value
        for key, value in context.items()
        if key != "trial_number"
    }
    return json.dumps(comparable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def collect_source_trial_groups(source_study) -> tuple[dict, dict, dict[str, int], list[dict]]:
    complete_groups: dict[tuple[tuple[str, object], ...], dict] = {}
    state_counts_by_key: dict[tuple[tuple[str, object], ...], dict[str, int]] = {}
    global_state_counts: dict[str, int] = {}
    skipped_trials: list[dict] = []

    for trial in study_trials(source_study):
        state_name = trial_state_name(trial)
        increment_count(global_state_counts, state_name)
        try:
            params = trial_params_from_mapping(getattr(trial, "params", {}), source=f"trial={trial.number} params")
            validate_trial_params_in_search_space(params)
        except ValueError as e:
            skipped_trials.append({
                "trial_number": int(getattr(trial, "number", -1)),
                "state": state_name,
                "error": str(e),
            })
            continue

        key = trial_params_key(params)
        state_counts = state_counts_by_key.setdefault(key, {})
        increment_count(state_counts, state_name)
        if state_name != "COMPLETE":
            continue

        group = complete_groups.setdefault(key, {
            "params": params,
            "trials": [],
        })
        group["trials"].append(trial)

    return complete_groups, state_counts_by_key, global_state_counts, skipped_trials


def build_summary_group_document(
    group_id: str,
    params: TrialParams,
    source_trials: list,
    source_state_counts: dict[str, int],
    source_artifact_context: OptunaArtifactContext,
    source_storage,
) -> tuple[dict, dict, list[float]]:
    seed_scores: list[float] = []
    seed_score_60_80: list[float] = []
    seed_score_80_100: list[float] = []
    seed_late_slopes: list[float] = []
    seed_records: list[dict] = []
    source_trial_scores: list[float] = []
    source_trial_records: list[dict] = []
    source_contexts: list[dict] = []

    for trial in sorted(source_trials, key=lambda item: int(item.number)):
        seed_runs_doc = load_trial_json_artifact(
            source_artifact_context,
            source_storage,
            trial,
            "seed_runs.json",
            required=True,
        )
        manifest_doc = load_trial_json_artifact(
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

        trial_score = source_trial_score(trial)
        trial_score_60_80_stats = aggregate_score_stats(trial_score_60_80, "mean")
        trial_score_80_100_stats = aggregate_score_stats(trial_score_80_100, "mean")
        trial_late_slope_stats = aggregate_score_stats(trial_late_slopes, "mean")
        source_trial_scores.append(trial_score)
        context = source_score_context(trial, seed_runs_doc, manifest_doc)
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
    source_context_mixed = len({score_context_fingerprint(context) for context in source_contexts}) > 1

    user_attrs = {
        "group_id": group_id,
        "params": asdict(params),
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
        "params": asdict(params),
        "objectives": {
            "names": list(SUMMARY_OBJECTIVE_NAMES),
            "directions": list(SUMMARY_OBJECTIVE_DIRECTIONS),
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


def command_summarize_study(args: argparse.Namespace) -> int:
    if args.storage_timeout_sec < 0:
        print("storage-timeout-sec must be >= 0.", file=sys.stderr)
        return 2
    if args.target_study_name is None:
        args.target_study_name = f"{args.source_study_name}_summary"
    validate_name_part(args.source_study_name, "--source-study-name")
    validate_name_part(args.target_study_name, "--target-study-name")

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
    source_artifact_context = create_required_filesystem_artifact_context(
        optuna,
        resolve_artifact_dir_from_text(args, args.source_artifact_dir),
        "source",
    )

    source_study = optuna.load_study(
        study_name=args.source_study_name,
        storage=source_storage,
    )
    complete_groups, state_counts_by_key, global_state_counts, skipped_trials = collect_source_trial_groups(source_study)
    if not complete_groups:
        print("source study has no COMPLETE TrialParams groups to summarize.", file=sys.stderr)
        return 1

    summary_groups: list[dict] = []
    mixed_context_group_count = 0
    source_complete_trial_count = 0
    source_seed_count = 0

    for index, (key, group) in enumerate(sorted(complete_groups.items(), key=lambda item: item[0])):
        params = group["params"]
        source_trials = group["trials"]
        source_state_counts = state_counts_by_key.get(key, {})
        group_id = f"g{index:05d}"
        document, user_attrs, objective_values = build_summary_group_document(
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
    target_artifact_context = create_required_filesystem_artifact_context(
        optuna,
        resolve_artifact_dir_from_text(args, args.target_artifact_dir),
        "target",
    )

    if optuna_study_exists(optuna, args.target_study_name, target_storage):
        if not args.overwrite_target_study:
            print(
                "target study already exists. "
                "Use --overwrite-target-study to delete and recreate it. "
                f"target_study_name={args.target_study_name}",
                file=sys.stderr,
            )
            return 2
        delete_optuna_study(optuna, args.target_study_name, target_storage)

    target_study = optuna.create_study(
        study_name=args.target_study_name,
        storage=target_storage,
        directions=list(SUMMARY_OBJECTIVE_DIRECTIONS),
    )
    try:
        target_study.set_metric_names(list(SUMMARY_OBJECTIVE_NAMES))
    except Exception as e:
        print(f"[WARN] Failed to set target study metric names: {e}", file=sys.stderr)

    target_distributions = trial_param_distributions(optuna)
    created_trial_numbers: list[int] = []

    for summary_group in summary_groups:
        params = summary_group["params"]
        document = summary_group["document"]
        user_attrs = summary_group["user_attrs"]
        objective_values = summary_group["objective_values"]
        target_study.enqueue_trial(asdict(params), user_attrs=user_attrs, skip_if_exists=False)
        target_trial = target_study.ask(fixed_distributions=target_distributions)
        if target_trial.params != asdict(params):
            raise TrialExecutionError(
                "Target summary trial params did not match the requested group params: "
                f"group_id={user_attrs['group_id']} requested={asdict(params)} actual={target_trial.params}"
            )
        try:
            document["target_trial_number"] = int(target_trial.number)
            upload_json_trial_artifact(
                target_artifact_context,
                target_storage,
                target_trial,
                GROUP_SUMMARY_ARTIFACT_FILENAME,
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
        "summary_harness": "dropmerge_optuna",
        "summary_command": "summarize-study",
        "source_study_name": args.source_study_name,
        "source_storage": source_storage_url,
        "source_artifact_dir": str(source_artifact_context.base_path),
        "target_study_name": args.target_study_name,
        "target_storage": target_storage_url,
        "target_artifact_dir": str(target_artifact_context.base_path),
        "summary_objective_names": list(SUMMARY_OBJECTIVE_NAMES),
        "summary_objective_directions": list(SUMMARY_OBJECTIVE_DIRECTIONS),
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


def command_cleanup_running(args: argparse.Namespace) -> int:
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


def command_run_trial(args: argparse.Namespace) -> int:
    _INTERRUPTING.clear()
    attach_harness_logger(args)
    params = params_from_args(args)
    try:
        result = execute_trial(args, params, args.trial_number, getattr(args, "trial_name", None))
    except TrialPrunedError as e:
        print(str(e), file=sys.stderr)
        return 2
    except TrialExecutionError as e:
        log_harness(args, "ERROR", "trial-failed", console=True, study=args.study_name, error=str(e))
        print(str(e), file=sys.stderr)
        return 2
    print(json.dumps(
        {
            "params": asdict(params),
            "context": asdict(result.ctx),
            "score": result.summary["score"],
            "metrics_summary_path": str(Path(result.ctx.artifact_dir) / "metrics_summary.json"),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


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


def set_optuna_trial_attrs(trial, params: TrialParams, ctx: TrialContext) -> None:
    trial.set_user_attr("params", asdict(params))
    trial.set_user_attr("cost_tf", ctx.cost_tf)
    trial.set_user_attr("cost_budget", ctx.cost_budget)
    trial.set_user_attr("token_count", ctx.token_count)
    trial.set_user_attr("trial_name", ctx.trial_name)
    trial.set_user_attr("run_name", ctx.run_name)
    trial.set_user_attr("run_dir", ctx.run_dir)
    trial.set_user_attr("artifact_dir", ctx.artifact_dir)
    trial.set_user_attr("config_path", ctx.config_path)


def set_multiseed_trial_attrs(trial, summary: dict) -> None:
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
    args: argparse.Namespace,
    params: TrialParams,
    trial_number: int | None,
    trial_name: str | None = None,
    optuna_trial=None,
) -> TrialExecutionResult:
    ctx = make_trial_context(args, params, trial_number, trial_name)
    try:
        score_window = resolve_score_window(args)
    except ValueError as e:
        raise TrialExecutionError(str(e)) from e
    if optuna_trial is not None:
        set_optuna_trial_attrs(optuna_trial, params, ctx)

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

    write_trial_files(args, params, ctx)
    returncode = run_runner(args, ctx)
    if optuna_trial is not None:
        optuna_trial.set_user_attr("returncode", returncode)
    if returncode != 0:
        raise TrialFailedError(f"runner failed: returncode={returncode}")

    metrics_path = Path(ctx.run_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        raise TrialFailedError(f"metrics.jsonl not found: {metrics_path}")

    summary = summarize_metrics(metrics_path, score_window)
    write_summary_files(summary, Path(ctx.artifact_dir))
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
    args: argparse.Namespace,
    params: TrialParams,
    trial_number: int,
    study=None,
    optuna_trial=None,
    optuna_artifact_context: OptunaArtifactContext | None = None,
) -> TrialExecutionResult:
    base_seeds = parse_seed_list(args.seeds)
    duplicate_info = resolve_duplicate_params_info(args, params, study, trial_number, base_seeds)
    seeds = duplicate_info.effective_seeds
    study_args = apply_duplicate_info_to_args(args, duplicate_info)
    ctx = make_trial_context(args, params, trial_number)
    try:
        resolve_score_window(args)
    except ValueError as e:
        raise TrialExecutionError(str(e)) from e
    if optuna_trial is not None:
        set_optuna_trial_attrs(optuna_trial, params, ctx)

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

    write_representative_trial_files(study_args, params, ctx, seeds, duplicate_info)
    seed_runs: list[dict] = []
    if duplicate_info.pruned_by_duplicate:
        error = duplicate_info.prune_reason or "duplicate params"
        summary = build_multiseed_summary(study_args, params, ctx, seeds, seed_runs, duplicate_info, error)
        write_multiseed_summary_files(summary, Path(ctx.artifact_dir))
        if optuna_trial is not None:
            set_multiseed_trial_attrs(optuna_trial, summary)
            register_optuna_trial_artifacts(optuna_trial, optuna_artifact_context, Path(ctx.artifact_dir))
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
        summary = build_multiseed_summary(study_args, params, ctx, seeds, seed_runs, duplicate_info, error)
        write_multiseed_summary_files(summary, Path(ctx.artifact_dir))
        if optuna_trial is not None:
            set_multiseed_trial_attrs(optuna_trial, summary)
            register_optuna_trial_artifacts(optuna_trial, optuna_artifact_context, Path(ctx.artifact_dir))
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
        seed_ctx = make_trial_context(seed_args, params, trial_number, seed_trial)
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
            result = execute_trial(seed_args, params, trial_number, seed_trial)
        except (TrialPrunedError, TrialFailedError, ValueError) as e:
            # 失敗 seed が混じった aggregate は比較対象として偏るため、原因別の trial state へ送る。
            seed_runs.append(make_seed_run_record(seed, seed_ctx, "failed", error=str(e)))
            summary = build_multiseed_summary(study_args, params, ctx, seeds, seed_runs, duplicate_info, str(e))
            write_multiseed_summary_files(summary, Path(ctx.artifact_dir))
            if optuna_trial is not None:
                set_multiseed_trial_attrs(optuna_trial, summary)
                register_optuna_trial_artifacts(optuna_trial, optuna_artifact_context, Path(ctx.artifact_dir))
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
        seed_runs.append(make_seed_run_record(seed, result.ctx, "complete", score=result.score, summary=result.summary))
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

    summary = build_multiseed_summary(study_args, params, ctx, seeds, seed_runs, duplicate_info)
    write_multiseed_summary_files(summary, Path(ctx.artifact_dir))
    if optuna_trial is not None:
        set_multiseed_trial_attrs(optuna_trial, summary)
        register_optuna_trial_artifacts(optuna_trial, optuna_artifact_context, Path(ctx.artifact_dir))
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


def objective(
    trial,
    args: argparse.Namespace,
    study,
    optuna_artifact_context: OptunaArtifactContext | None,
) -> float:
    import optuna

    params = suggest_params(trial)
    try:
        return execute_study_trial(
            args,
            params,
            trial.number,
            study=study,
            optuna_trial=trial,
            optuna_artifact_context=optuna_artifact_context,
        ).score
    except TrialPrunedError as e:
        # cost / duplicate は探索制約として PRUNED、runner / metrics / score 欠落は FAIL として扱う。
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


def command_run_study(args: argparse.Namespace) -> int:
    _INTERRUPTING.clear()
    attach_harness_logger(args)
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
    # Optuna 既定も TPE だが、探索系列の再現性と初期ランダム件数を CLI から明示制御する。
    sampler = optuna.samplers.TPESampler(
        seed=args.sampler_seed,
        n_startup_trials=args.n_startup_trials,
        constant_liar=args.constant_liar,
    )
    storage_url = storage_url_from_arg(args)
    storage = create_optuna_storage(
        optuna,
        storage_url,
        args.storage_timeout_sec,
        args.heartbeat_interval_sec,
        args.heartbeat_grace_period_sec,
    )
    optuna_artifact_context = create_optuna_artifact_context(args)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )
    set_study_user_attrs(study, build_study_user_attrs(args, storage_url))
    fail_stale_trials_if_enabled(args, optuna, study, "before-optimize")
    try:
        study.optimize(
            lambda trial: objective(trial, args, study, optuna_artifact_context),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            catch=(TrialFailedError,),
        )
    except KeyboardInterrupt:
        _INTERRUPTING.set()
        terminated_pids = terminate_active_runners()
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
        terminated_pids = terminate_active_runners()
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


def add_common_args(parser: argparse.ArgumentParser, include_seed: bool = True) -> None:
    repo_root = repo_root_from_script()
    parser.add_argument("--repo-root", default=str(repo_root), help="anet-lab のリポジトリルート。")
    parser.add_argument(
        "--base-config",
        default=str(default_main_config(repo_root)),
        help="trial config が最初に $include する基準 main config。",
    )
    parser.add_argument(
        "--extra-config",
        default="DropMerge_optuna.txt",
        help="生成 trial config が base config の後に $include する Optuna 専用 config。",
    )
    parser.add_argument("--study-name", default="dropmergeOptuna", help="Optuna study 名。")
    parser.add_argument("--budget", choices=sorted(BUDGETS), default="small", help="使う cost_budget プリセット。")
    parser.add_argument("--cost-budget", type=float, help="cost_budget を明示指定する。指定時は --budget の値を上書きする。")
    parser.add_argument("--cost-k", type=float, default=4.0, help="cost_tf の N*M^2 項に掛ける係数。")
    parser.add_argument(
        "--runs-dir",
        default="runs_optuna",
        help="runner の app.runs_dir に渡す出力先。runner project root 基準。",
    )
    parser.add_argument("--exp-exit-step", type=int, default=1_000_000, help="proxy trial の app.batchrun.exp_exit_step。%% window の基準 step。")
    if include_seed:
        parser.add_argument("--seed", type=int, default=12345, help="train.seed に使う seed。")
    parser.add_argument("--nhead", type=int, default=8, help="Transformer の attention head 数。")


def add_trial_identity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trial-name", help="trial 名。未指定時は既存出力から t00000 形式で自動採番する。")


def add_param_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trial-number", type=int, help="trial_name 未指定時に使う trial 番号。未指定時は既存出力から自動採番する。")
    parser.add_argument("--cnn-channels", type=int, choices=[48, 64], default=64, help="CNN/ResBlock の channel 数 C。")
    parser.add_argument("--res-blocks", type=int, choices=[2, 4], default=4, help="ResBlock の繰り返し数 D。")
    parser.add_argument("--token-mode", choices=["current", "stronger"], default="current", help="token 解像度 N のモード。")
    parser.add_argument("--d-model", type=int, choices=[96, 128, 192], default=96, help="Transformer の d_model M。")
    parser.add_argument("--transformer-layers", type=int, choices=[2, 4], default=2, help="Transformer 層数 L。")
    parser.add_argument("--ff-mult", type=int, choices=[2, 4], default=2, help="dim_feedforward = d_model * ff_mult。")
    parser.add_argument("--trunk-width", type=int, choices=[1024, 2048], default=1024, help="Flatten 後 trunk Linear 幅 H。")
    parser.add_argument("--head-width", type=int, choices=[512, 1024], default=512, help="value/adv stream の head Linear 幅。")


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


def build_parser() -> argparse.ArgumentParser:
    usage = (
        "%(prog)s <command> [options]\n\n"
        "例:\n"
        "  %(prog)s dry-run --budget small\n"
        "  %(prog)s run-trial --study-name dropmergeSmall --trial-name t00001\n"
        "  %(prog)s summarize apps/runner/runs_optuna/dropmergeSmall_t00001/metrics.jsonl --window-start 80%% --window-end 100%%\n"
        "  %(prog)s summarize-study --source-study-name dropmergeSmall\n"
        "  %(prog)s cleanup-running --study-name dropmergeSmall --dry-run\n"
        "  %(prog)s run-study --budget small --n-trials 10 --seeds 12345,23456"
    )
    parser = JapaneseArgumentParser(
        usage=usage,
        description=(
            "DropMerge の NN 構成探索用 harness。\n"
            "trial config を生成し、runner を --config で起動し、metrics.jsonl を採点します。"
        ),
        formatter_class=JapaneseHelpFormatter,
        add_help=False,
    )
    localize_parser(parser, "コマンド")
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="command",
        title="コマンド",
        description="利用可能なコマンド",
        parser_class=JapaneseArgumentParser,
    )

    dry_run = subparsers.add_parser(
        "dry-run",
        help="trial config と manifest だけ生成する",
        description="指定パラメータで trial config / manifest を生成し、cost_budget 判定を表示します。",
        formatter_class=JapaneseHelpFormatter,
        add_help=False,
    )
    localize_parser(dry_run)
    add_common_args(dry_run)
    add_trial_identity_args(dry_run)
    add_param_args(dry_run)
    add_score_window_args(dry_run)
    dry_run.set_defaults(func=command_dry_run)

    summarize = subparsers.add_parser(
        "summarize",
        help="metrics.jsonl を採点する",
        description="既存の metrics.jsonl から固定 step window の score と補助指標を抽出します。",
        formatter_class=JapaneseHelpFormatter,
        add_help=False,
    )
    localize_parser(summarize)
    summarize.add_argument("metrics_jsonl", help="採点対象の metrics.jsonl。")
    summarize.add_argument("--exp-exit-step", type=int, default=1_000_000, help="負数 window と %% window の基準 step。")
    add_score_window_args(summarize, primary_score=False)
    summarize.add_argument("--output-dir", help="metrics_summary.json/csv の出力先。")
    summarize.set_defaults(func=command_summarize)

    summarize_study = subparsers.add_parser(
        "summarize-study",
        help="同一 params group を Dashboard 用 Study に集約する",
        description=(
            "既存 Optuna study の同一 TrialParams を group 化し、"
            "mean/range/std の multi-objective summary study を生成します。"
        ),
        formatter_class=JapaneseHelpFormatter,
        add_help=False,
    )
    localize_parser(summarize_study)
    repo_root = repo_root_from_script()
    summarize_study.add_argument("--repo-root", default=str(repo_root), help="anet-lab のリポジトリルート。")
    summarize_study.add_argument("--source-study-name", required=True, help="集約元の Optuna study 名。")
    summarize_study.add_argument(
        "--target-study-name",
        help="集約先の Optuna study 名。未指定時は <source-study-name>_summary。",
    )
    summarize_study.add_argument(
        "--source-storage",
        default="sqlite:///runs_optuna/optuna.db",
        help="集約元 Optuna SQLite DB の URL またはパス。相対時は runner project root 基準。",
    )
    summarize_study.add_argument(
        "--target-storage",
        help="集約先 Optuna SQLite DB の URL またはパス。未指定時は source-storage と同じ。",
    )
    summarize_study.add_argument(
        "--source-artifact-dir",
        default="runs_optuna/artifacts",
        help="集約元 Optuna Dashboard artifact store。相対時は runner project root 基準。",
    )
    summarize_study.add_argument(
        "--target-artifact-dir",
        default="runs_optuna/artifacts",
        help="集約先 Optuna Dashboard artifact store。相対時は runner project root 基準。",
    )
    summarize_study.add_argument(
        "--storage-timeout-sec",
        type=float,
        default=120.0,
        help="SQLite storage の lock 待ち timeout 秒。",
    )
    summarize_study.add_argument(
        "--overwrite-target-study",
        action="store_true",
        default=False,
        help="既存 target study を削除して作り直す。同一 storage の source study 自体は指定できない。",
    )
    summarize_study.set_defaults(func=command_summarize_study)

    cleanup = subparsers.add_parser(
        "cleanup-running",
        help="Study に残った RUNNING trial を FAIL にする",
        description="中断などで RUNNING のまま残った Optuna trial を指定 study 内で FAIL に変更します。",
        formatter_class=JapaneseHelpFormatter,
        add_help=False,
    )
    localize_parser(cleanup)
    repo_root = repo_root_from_script()
    cleanup.add_argument("--repo-root", default=str(repo_root), help="anet-lab のリポジトリルート。")
    cleanup.add_argument("--study-name", default="dropmergeOptuna", help="Optuna study 名。")
    cleanup.add_argument(
        "--storage",
        default="sqlite:///runs_optuna/optuna.db",
        help="Optuna SQLite DB の URL またはパス。相対時は runner project root 基準。",
    )
    cleanup.add_argument(
        "--storage-timeout-sec",
        type=float,
        default=120.0,
        help="SQLite storage の lock 待ち timeout 秒。",
    )
    cleanup.add_argument("--dry-run", action="store_true", default=False, help="対象 RUNNING trial を表示するだけで DB を変更しない。")
    cleanup.set_defaults(func=command_cleanup_running)

    run_trial = subparsers.add_parser(
        "run-trial",
        help="固定 params の trial を 1 件実行する",
        description="CLI で指定した NN 構成を runner で 1 件実行し、metrics.jsonl を採点します。",
        formatter_class=JapaneseHelpFormatter,
        add_help=False,
    )
    localize_parser(run_trial)
    add_common_args(run_trial)
    add_trial_identity_args(run_trial)
    add_param_args(run_trial)
    add_runner_args(run_trial)
    add_score_window_args(run_trial)
    run_trial.set_defaults(func=command_run_trial)

    run_study = subparsers.add_parser(
        "run-study",
        help="Optuna study を実行する",
        description="Optuna study を作成/再開し、Optuna が生成した trial を順次 runner で実行します。",
        formatter_class=JapaneseHelpFormatter,
        add_help=False,
    )
    localize_parser(run_study)
    add_common_args(run_study, include_seed=False)
    add_runner_args(run_study)
    add_score_window_args(run_study)
    run_study.add_argument(
        "--storage",
        default="sqlite:///runs_optuna/optuna.db",
        help="Optuna SQLite DB の URL またはパス。相対時は runner project root 基準。",
    )
    run_study.add_argument(
        "--storage-timeout-sec",
        type=float,
        default=120.0,
        help="SQLite storage の lock 待ち timeout 秒。",
    )
    run_study.add_argument(
        "--heartbeat-interval-sec",
        type=int,
        default=60,
        help="Optuna RDBStorage heartbeat interval 秒。0 は heartbeat を無効化する。",
    )
    run_study.add_argument(
        "--heartbeat-grace-period-sec",
        type=int,
        default=600,
        help="heartbeat が途絶えた RUNNING trial を stale とみなす猶予秒。",
    )
    run_study.add_argument(
        "--optuna-artifact-dir",
        default="runs_optuna/artifacts",
        help="Optuna Dashboard 用 artifact store の base path。相対時は runner project root 基準。",
    )
    run_study.add_argument("--n-trials", type=int, default=10, help="この実行で追加する trial 数。")
    run_study.add_argument("--n-jobs", type=int, default=1, help="Optuna の並列 worker 数。")
    run_study.add_argument("--study-note", help="Study User Attributes の note に保存する任意メモ。未指定時は既存 note を変更しない。")
    run_study.add_argument("--seeds", default="12345", help="同一 params を評価する train.seed の comma-separated list。")
    run_study.add_argument(
        "--score-aggregate",
        choices=SCORE_AGGREGATES,
        default="mean",
        help="multi-seed score を Optuna trial value に集約する方法。",
    )
    run_study.add_argument("--sampler-seed", type=int, help="Optuna sampler の乱数 seed。未指定時は Optuna 既定。")
    run_study.add_argument(
        "--constant-liar",
        action="store_true",
        default=False,
        help="TPESampler の constant_liar を有効にし、RUNNING trial 近傍の再提案を避けやすくする。",
    )
    run_study.add_argument(
        "--duplicate-params-policy",
        choices=DUPLICATE_PARAMS_POLICIES,
        default="reseed",
        help="同一 NN params が再提案されたときの扱い。",
    )
    run_study.add_argument(
        "--duplicate-params-max-runs",
        type=int,
        default=3,
        help="同一 NN params を実行する最大回数。0 は制限なし。",
    )
    run_study.add_argument(
        "--duplicate-seed-stride",
        type=int,
        default=100_000,
        help="duplicate params を reseed するときに duplicate_index ごとに seed へ足す値。",
    )
    run_study.add_argument(
        "--n-startup-trials",
        type=int,
        default=10,
        help="TPE に切り替える前に random sampling する完了 trial 数。",
    )
    run_study.set_defaults(func=command_run_study)

    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "timeout_sec", 0) == 0:
        args.timeout_sec = None
    try:
        if hasattr(args, "study_name"):
            validate_name_part(args.study_name, "--study-name")
        if getattr(args, "trial_name", None):
            validate_name_part(args.trial_name, "--trial-name")
        return args.func(args)
    except TrialExecutionError as e:
        print(str(e), file=sys.stderr)
        return 2
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

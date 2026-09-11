#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mlflow_bridge.py — Windows完全対応版
──────────────────────────────
C++ MetricsLogger が出力する JSONL ログを読み取り、
MLflow Tracking Server（ローカルUI含む）へ転送するスクリプト。
──────────────────────────────
"""

import argparse
from dataclasses import dataclass
import json
import pathlib
import re
import time

import mlflow
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient

from metrics_source import (
    logical_metrics_path,
    open_metrics_binary,
    resolve_metrics_path,
    resolve_run_metrics,
)


DEFAULT_EXPERIMENT_ID = "0"
SOURCE_METRICS_TAG = "anet.bridge.source_metrics"
METRICS_OFFSET_TAG = "anet.bridge.metrics_offset"
BRIDGE_STATE_TAG = "anet.bridge.state"
READ_BATCH_SIZE = 1000
STATUS_INTERVAL_SECONDS = 10.0
LATEST_BATCHES_PER_HISTORY_BATCH = 10


@dataclass
class MonitoredRun:
    metrics_path: pathlib.Path
    run_id: str
    offset: int
    gzip_stream: object | None = None


@dataclass
class PollScheduleState:
    latest_batches: int = 0
    history_cursor: int = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Bridge JSONL logs to MLflow")
    parser.add_argument("--logdir", default="runs", help="MetricsLogger 出力フォルダ（省略時: runs）")
    parser.add_argument(
        "--tracking-db",
        default=str(pathlib.Path("runs", "mlflow.db")),
        help="MLflow SQLite database path (default: runs/mlflow.db)",
    )
    parser.add_argument("--run-name", default=None, help="MLflow上のRun名")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="監視間隔[秒]")
    parser.add_argument("--once", action="store_true", help="一度だけ変換して終了")
    return parser.parse_args()


def tracking_uri_from_path(tracking_db):
    db_path = pathlib.Path(tracking_db).absolute()
    return db_path, f"sqlite:///{db_path.as_posix()}"


def load_config_params(filepath):
    config_path = pathlib.Path(filepath)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config data file not found: {config_path}")

    params = {}
    source_keys = {}

    # ConfigData の出力形式である key = value を行単位で変換する
    with config_path.open("r", encoding="utf-8") as config_file:
        for line_number, raw_line in enumerate(config_file, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue

            separator = line.find(" = ")
            if separator <= 0:
                raise ValueError(
                    f"Invalid config_data line: {config_path}:{line_number}: "
                    "expected 'key = value'"
                )

            key = line[:separator]
            value = line[separator + 3 :]
            normalized_key = key.replace("[", "").replace("]", "")
            normalized_key = re.sub(r"[^\w.\- /]", "_", normalized_key)
            if not normalized_key.strip():
                raise ValueError(
                    f"Invalid config_data key: {config_path}:{line_number}: key='{key}'"
                )

            if normalized_key in source_keys:
                previous_key = source_keys[normalized_key]
                if previous_key != key:
                    raise ValueError(
                        f"Config_data keys collide after MLflow normalization: "
                        f"{config_path}:{line_number}: keys='{previous_key}', '{key}', "
                        f"normalized_key='{normalized_key}'"
                    )
                raise ValueError(
                    f"Duplicate config_data key: {config_path}:{line_number}: key='{key}'"
                )

            source_keys[normalized_key] = key
            params[normalized_key] = value

    return params


def find_run_metrics(logdir):
    log_root = pathlib.Path(logdir)
    metrics_paths = []
    for run_dir in log_root.glob("run_*"):
        if not run_dir.is_dir():
            continue
        metrics_path = resolve_run_metrics(run_dir)
        if metrics_path is not None:
            metrics_paths.append(metrics_path)
    return sorted(metrics_paths)


def source_metrics_key(metrics_path):
    return logical_metrics_path(metrics_path).resolve().as_posix()


def _read_jsonl_stream(metrics_file, start_offset, max_lines):
    entries = []
    next_offset = start_offset
    if metrics_file.tell() != start_offset:
        metrics_file.seek(start_offset)

    for _ in range(max_lines):
        line_start = metrics_file.tell()
        raw_line = metrics_file.readline()
        if not raw_line:
            break

        has_newline = raw_line.endswith(b"\n")
        try:
            entry = json.loads(raw_line.decode("utf-8").strip())
        except (UnicodeDecodeError, json.JSONDecodeError):
            if not has_newline:
                # 書き込み途中の末尾行はoffsetを進めず、次回pollで再読込する
                metrics_file.seek(line_start)
                break
        else:
            entries.append(entry)

        next_offset = metrics_file.tell()
    return entries, next_offset


def read_jsonl_batch(filepath, start_offset=0, max_lines=READ_BATCH_SIZE):
    metrics_path = resolve_metrics_path(filepath)
    # offsetはraw/gzipとも展開後JSONL byte位置で統一する。
    with open_metrics_binary(metrics_path) as metrics_file:
        if metrics_path.name == "metrics.jsonl" and start_offset > metrics_path.stat().st_size:
            raise ValueError(
                f"Metrics file was truncated: path='{metrics_path}', "
                f"offset={start_offset}, size={metrics_path.stat().st_size}"
            )
        try:
            metrics_file.seek(start_offset)
        except (EOFError, OSError) as error:
            raise ValueError(
                f"Metrics file was truncated: path='{metrics_path}', offset={start_offset}"
            ) from error
        if metrics_file.tell() != start_offset:
            raise ValueError(
                f"Metrics file was truncated: path='{metrics_path}', offset={start_offset}"
            )
        return _read_jsonl_stream(metrics_file, start_offset, max_lines)


def refresh_monitored_source(monitored_run):
    current_path = resolve_run_metrics(monitored_run.metrics_path.parent)
    if current_path is None:
        raise FileNotFoundError(f"Metrics file not found: {monitored_run.metrics_path.parent}")
    if current_path == monitored_run.metrics_path:
        return
    if monitored_run.gzip_stream is not None:
        monitored_run.gzip_stream.close()
        monitored_run.gzip_stream = None
    monitored_run.metrics_path = current_path


def entry_to_metric(entry, timestamp_ms=None):
    tag = entry.get("tag") or entry.get("key") or entry.get("name")
    step = entry.get("step")
    if step is None:
        step = entry.get("global_step")
    value = entry.get("value")
    if value is None:
        value = entry.get("scalar")

    if tag is None or value is None or not isinstance(value, (int, float)):
        return None

    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    return Metric(
        key=tag,
        value=value,
        timestamp=timestamp_ms,
        step=0 if step is None else int(step),
    )


def load_existing_bridge_runs(client):
    existing_runs = {}
    page_token = None

    # source tag を持つ既存MLflow Runを読み、重複作成せず再利用する
    while True:
        page = client.search_runs(
            experiment_ids=[DEFAULT_EXPERIMENT_ID],
            max_results=1000,
            page_token=page_token,
        )
        for run in page:
            source_key = run.data.tags.get(SOURCE_METRICS_TAG)
            if source_key is None:
                continue
            if source_key in existing_runs:
                raise ValueError(
                    f"Duplicate MLflow runs for source metrics: source='{source_key}', "
                    f"run_ids='{existing_runs[source_key].info.run_id}', "
                    f"'{run.info.run_id}'"
                )
            existing_runs[source_key] = run

        page_token = page.token
        if not page_token:
            break

    return existing_runs


def register_run(client, metrics_path, existing_runs, run_name=None):
    metrics_path = pathlib.Path(metrics_path)
    source_key = source_metrics_key(metrics_path)
    config_data_path = metrics_path.parent / "config" / "config_data.txt"
    config_params = load_config_params(config_data_path)
    existing_run = existing_runs.get(source_key)

    # source tagでRunを一意に対応付け、再起動時は保存済みoffsetから再開する
    if existing_run is None:
        mlflow_run = client.create_run(
            experiment_id=DEFAULT_EXPERIMENT_ID,
            run_name=run_name or metrics_path.parent.name,
            tags={
                SOURCE_METRICS_TAG: source_key,
                METRICS_OFFSET_TAG: "0",
                BRIDGE_STATE_TAG: "monitoring",
            },
        )
        offset = 0
        existing_runs[source_key] = mlflow_run
        action = "Registered"
    else:
        mlflow_run = existing_run
        raw_offset = mlflow_run.data.tags.get(METRICS_OFFSET_TAG, "0")
        try:
            offset = int(raw_offset)
        except ValueError as error:
            raise ValueError(
                f"Invalid MLflow metrics offset: run_id='{mlflow_run.info.run_id}', "
                f"value='{raw_offset}'"
            ) from error
        if offset < 0:
            raise ValueError(
                f"Invalid MLflow metrics offset: run_id='{mlflow_run.info.run_id}', "
                f"value='{raw_offset}'"
            )
        client.update_run(mlflow_run.info.run_id, status="RUNNING")
        client.set_tag(mlflow_run.info.run_id, BRIDGE_STATE_TAG, "monitoring")
        action = "Resumed"

    # 同じ値のparameter再登録を許し、途中失敗した初回登録も次回起動で回復する
    client.log_batch(
        mlflow_run.info.run_id,
        params=[Param(key, value) for key, value in config_params.items()],
    )
    print(
        f"[INFO] {action}: {metrics_path} "
        f"(run_id={mlflow_run.info.run_id}, offset={offset}, "
        f"parameters={len(config_params)})"
    )
    return MonitoredRun(metrics_path=metrics_path, run_id=mlflow_run.info.run_id, offset=offset)


def poll_run(client, monitored_run):
    refresh_monitored_source(monitored_run)
    if monitored_run.metrics_path.name == "metrics.jsonl.gz":
        if monitored_run.gzip_stream is None:
            monitored_run.gzip_stream = open_metrics_binary(monitored_run.metrics_path)
            monitored_run.gzip_stream.seek(monitored_run.offset)
        entries, next_offset = _read_jsonl_stream(
            monitored_run.gzip_stream,
            monitored_run.offset,
            READ_BATCH_SIZE,
        )
    else:
        entries, next_offset = read_jsonl_batch(
            monitored_run.metrics_path,
            monitored_run.offset,
        )
    if next_offset == monitored_run.offset:
        return False

    # 1 batch分を記録してからoffsetを更新し、再起動時の取りこぼしを防ぐ
    timestamp_ms = int(time.time() * 1000)
    metrics = [
        metric
        for entry in entries
        if (metric := entry_to_metric(entry, timestamp_ms)) is not None
    ]
    if metrics:
        client.log_batch(monitored_run.run_id, metrics=metrics)

    client.set_tag(monitored_run.run_id, METRICS_OFFSET_TAG, str(next_offset))
    monitored_run.offset = next_offset
    return True


def poll_monitored_runs(client, monitored_runs, schedule_state):
    for monitored_run in monitored_runs.values():
        refresh_monitored_source(monitored_run)
    ordered_runs = sorted(
        monitored_runs.values(),
        key=lambda monitored_run: monitored_run.metrics_path.stat().st_mtime_ns,
        reverse=True,
    )

    # 最新更新Runを優先しつつ、一定batchごとに過去Runへ処理枠を渡す
    latest_run = ordered_runs[0]
    latest_progress = poll_run(client, latest_run)
    historical_runs = ordered_runs[1:]
    if latest_progress:
        progressed_historical_runs = []
        schedule_state.latest_batches += 1
        if (
            historical_runs
            and schedule_state.latest_batches >= LATEST_BATCHES_PER_HISTORY_BATCH
        ):
            history_index = schedule_state.history_cursor % len(historical_runs)
            historical_run = historical_runs[history_index]
            if poll_run(client, historical_run):
                progressed_historical_runs.append(historical_run)
            schedule_state.history_cursor += 1
            schedule_state.latest_batches = 0
        elif not historical_runs:
            schedule_state.latest_batches = 0
        return True, progressed_historical_runs

    # 最新Runへ追いついた後、過去Runを1 batchずつ処理する
    schedule_state.latest_batches = 0
    progressed_historical_runs = []
    for monitored_run in historical_runs:
        if poll_run(client, monitored_run):
            progressed_historical_runs.append(monitored_run)

    return False, progressed_historical_runs


def format_bridge_status(monitored_runs):
    latest_run = max(
        monitored_runs.values(),
        key=lambda monitored_run: monitored_run.metrics_path.stat().st_mtime_ns,
    )
    file_size = latest_run.metrics_path.stat().st_size
    lag = max(0, file_size - latest_run.offset)
    mebibyte = 1024 * 1024
    return (
        f"[INFO] Latest progress: runs={len(monitored_runs)}, "
        f"run={latest_run.metrics_path.parent.name}, "
        f"offset={latest_run.offset / mebibyte:.1f}/{file_size / mebibyte:.1f} MiB, "
        f"lag={lag / mebibyte:.1f} MiB"
    )


def format_historical_status(monitored_run):
    file_size = monitored_run.metrics_path.stat().st_size
    lag = max(0, file_size - monitored_run.offset)
    mebibyte = 1024 * 1024
    return (
        f"[INFO] Historical progress: run={monitored_run.metrics_path.parent.name}, "
        f"offset={monitored_run.offset / mebibyte:.1f}/{file_size / mebibyte:.1f} MiB, "
        f"lag={lag / mebibyte:.1f} MiB"
    )


def resolve_initial_metrics(logdir):
    log_path = pathlib.Path(logdir)
    if log_path.is_dir():
        metrics_paths = find_run_metrics(log_path)
        if metrics_paths:
            return metrics_paths, log_path

        direct_metrics = resolve_run_metrics(log_path)
        if direct_metrics is not None:
            return [direct_metrics], None

        raise FileNotFoundError(
            f"Direct run metrics were not found: {log_path / 'run_*' / 'metrics.jsonl(.gz)'}"
        )

    return [resolve_metrics_path(log_path)], None


def main():
    args = parse_args()
    initial_metrics, scan_root = resolve_initial_metrics(args.logdir)
    if scan_root is not None and args.run_name is not None:
        raise ValueError("--run-name can only be used when --logdir points to one metrics file")

    # launcher が選択した workspace のDBを、Windowsでも安定する絶対POSIX URIへ変換する。
    mlflow_db_path, tracking_uri = tracking_uri_from_path(args.tracking_db)
    mlflow_db_path.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[INFO] Tracking database: {mlflow_db_path}")
    if scan_root is not None:
        print(f"[INFO] Monitoring direct runs: {scan_root} ({len(initial_metrics)} found)")

    client = MlflowClient()
    existing_runs = load_existing_bridge_runs(client)
    monitored_runs = {}

    def add_metrics_path(metrics_path, run_name=None):
        source_key = source_metrics_key(metrics_path)
        if source_key in monitored_runs:
            monitored_run = monitored_runs[source_key]
            if monitored_run.metrics_path != metrics_path:
                if monitored_run.gzip_stream is not None:
                    monitored_run.gzip_stream.close()
                    monitored_run.gzip_stream = None
                monitored_run.metrics_path = metrics_path
            return
        monitored_runs[source_key] = register_run(
            client,
            metrics_path,
            existing_runs,
            run_name=run_name,
        )

    for metrics_path in initial_metrics:
        add_metrics_path(metrics_path, run_name=args.run_name)

    next_status_time = 0.0
    latest_progress_since_status = False
    historical_progress_since_status = {}
    poll_schedule_state = PollScheduleState()
    while True:
        # 監視中に作成されたruns直下のRunも追加する。再帰globは使わない。
        if scan_root is not None:
            for metrics_path in find_run_metrics(scan_root):
                add_metrics_path(metrics_path)

        latest_progress, progressed_historical_runs = poll_monitored_runs(
            client,
            monitored_runs,
            poll_schedule_state,
        )
        latest_progress_since_status = (
            latest_progress_since_status or latest_progress
        )
        for monitored_run in progressed_historical_runs:
            historical_progress_since_status[monitored_run.run_id] = monitored_run
        made_progress = latest_progress or bool(progressed_historical_runs)

        current_time = time.monotonic()
        if current_time >= next_status_time:
            if latest_progress_since_status:
                print(format_bridge_status(monitored_runs), flush=True)
            for monitored_run in historical_progress_since_status.values():
                print(format_historical_status(monitored_run), flush=True)
            latest_progress_since_status = False
            historical_progress_since_status.clear()
            next_status_time = current_time + STATUS_INTERVAL_SECONDS

        if args.once and not made_progress:
            for monitored_run in monitored_runs.values():
                client.set_tag(monitored_run.run_id, BRIDGE_STATE_TAG, "converted")
                client.set_terminated(monitored_run.run_id, status="FINISHED")
            return

        if not made_progress:
            time.sleep(args.poll_interval)

if __name__ == "__main__":
    main()

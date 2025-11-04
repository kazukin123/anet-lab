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
import json
import os
import time
import glob
import mlflow
import pathlib


def parse_args():
    parser = argparse.ArgumentParser(description="Bridge JSONL logs to MLflow")
    parser.add_argument("--logdir", default="logs", help="MetricsLogger 出力フォルダ（省略時: logs）")
    parser.add_argument("--run-name", default=None, help="MLflow上のRun名")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="監視間隔[秒]")
    parser.add_argument("--once", action="store_true", help="一度だけ変換して終了")
    return parser.parse_args()


def log_entry(entry):
    tag = entry.get("tag") or entry.get("key") or entry.get("name")
    step = entry.get("step") or entry.get("global_step")
    value = entry.get("value") or entry.get("scalar")

    if tag is None or value is None:
        return

    if isinstance(value, (int, float)):
        mlflow.log_metric(tag, value, step=step or 0)


def stream_jsonl(filepath, start_offset=0):
    with open(filepath, "r", encoding="utf-8") as f:
        f.seek(start_offset)
        while True:
            line = f.readline()
            if not line:
                yield None, f.tell()
                time.sleep(0.1)
                continue
            try:
                yield json.loads(line.strip()), f.tell()
            except json.JSONDecodeError:
                continue


def find_latest_run_dir(logdir):
    candidates = sorted(
        glob.glob(os.path.join(logdir, "run_*/metrics.jsonl")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"{logdir}/run_*/metrics.jsonl が見つかりません。")
    return candidates[0]


def main():
    args = parse_args()

    # 対象 JSONL ファイルの決定
    target_jsonl = None
    if os.path.isdir(args.logdir):
        try:
            target_jsonl = find_latest_run_dir(args.logdir)
        except FileNotFoundError:
            direct = os.path.join(args.logdir, "metrics.jsonl")
            if os.path.exists(direct):
                target_jsonl = direct
            else:
                raise
    else:
        target_jsonl = args.logdir  # ファイルパス指定時

    print(f"[INFO] Monitoring: {target_jsonl}")

    # --- Windows対応: mlruns を POSIX形式で指定 ---
    mlruns_path = pathlib.Path("mlruns").absolute().as_posix()
    mlflow.set_tracking_uri(f"file:///{mlruns_path}")

    # Run名を決定
    run_name = args.run_name or os.path.basename(os.path.dirname(target_jsonl))
    print(f"[INFO] Run name: {run_name}")

    with mlflow.start_run(run_name=run_name):
        offset = 0
        for entry, offset in stream_jsonl(target_jsonl, 0):
            if entry:
                log_entry(entry)

        if not args.once:
            while True:
                for entry, offset in stream_jsonl(target_jsonl, offset):
                    if entry:
                        log_entry(entry)
                time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()

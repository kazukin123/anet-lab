#!/usr/bin/env python3
"""
---------------------------------------
C++ MetricsLogger が出力する JSONL ログを監視し、
TensorBoard 用の tfevents ファイルへリアルタイム変換するブリッジ。

特徴:
  - 複数 run ディレクトリ自動検出 (runs/run_YYYYMMDD_HHMMSS/)
  - JSONL 形式: 1行1イベント (meta/scalar/vector/tensor)
  - 書き込み途中行は静かにスキップ（漏れなし）
  - 100行ごとに進捗ログを出力
  - 起動時に古い .tfevents を削除
  - filename_suffix=".bridge" により常に1ファイル固定
---------------------------------------
実行方法:
    python tb_bridge.py --runsdir runs
    tensorboard --runsdir runs
---------------------------------------
"""

import os
import time
import json
import glob
from dataclasses import dataclass

from metrics_source import open_metrics_binary, resolve_metrics_path, resolve_run_metrics


@dataclass
class RunState:
    writer: object
    source_path: object
    last_pos: int = 0
    line_count: int = 0
    gzip_stream: object | None = None


def tail_jsonl(file_path, last_pos):
    """新しい行リストと次のファイル位置を返す"""
    new_lines = []
    new_pos = last_pos
    try:
        metrics_path = resolve_metrics_path(file_path)
        with open_metrics_binary(metrics_path) as f:
            f.seek(last_pos)
            while line := f.readline():
                line_start = f.tell() - len(line)
                if not line.endswith(b"\n"):
                    # 書き込み途中の末尾行はoffsetを進めず、次回pollで再読込する。
                    f.seek(line_start)
                    break
                if line.strip():
                    new_lines.append(line.decode("utf-8").strip())
            new_pos = f.tell()
    except FileNotFoundError:
        pass
    return new_lines, new_pos


def tail_run_state(state):
    """rawは追従し、gzipは同じ展開streamを保持して差分を読む。"""
    if state.source_path.name != "metrics.jsonl.gz":
        return tail_jsonl(state.source_path, state.last_pos)
    if state.gzip_stream is None:
        state.gzip_stream = open_metrics_binary(state.source_path)
        state.gzip_stream.seek(state.last_pos)

    lines = []
    for raw_line in state.gzip_stream:
        if raw_line.strip():
            lines.append(raw_line.decode("utf-8").strip())
    return lines, state.gzip_stream.tell()


def process_json_line(writer, j):
    """1行分のJSONイベントをTensorBoardイベントに変換して出力"""
    t = j.get("type")

    if t == "meta":
        event = j.get("event", "")
        ts = j.get("timestamp", "")
        info = json.dumps(j, ensure_ascii=False, indent=2)
        writer.add_text(f"meta/{event}", info)
        print(f"[META] {ts} {event}")

    elif t == "scalar":
        tag = j.get("tag")
        step = j.get("step", 0)
        val = j.get("value", 0.0)
        writer.add_scalar(tag, val, step)

    elif t == "vector":
        tag = j.get("tag")
        step = j.get("step", 0)
        vals = j.get("values", [])
        for i, v in enumerate(vals):
            writer.add_scalar(f"{tag}/{i}", v, step)

    elif t == "tensor":
        tag = j.get("tag")
        step = j.get("step", 0)
        mean = j.get("mean", 0.0)
        std = j.get("std", 0.0)
        writer.add_scalar(f"{tag}/mean", mean, step)
        writer.add_scalar(f"{tag}/std", std, step)

    elif t == "json":
        tag = j.get("tag")
        ts = j.get("timestamp", "")
        data = json.dumps(j.get("data", {}), indent=2)
        writer.add_text(f"meta/{tag}", data)
        print(f"[JSON] {ts} {tag} {data}")

    else:
        # 未知タイプもtext出力
        writer.add_text("raw_event", json.dumps(j, ensure_ascii=False))
        print(f"Unknown type: {t} {j}")


def main(log_root="runs", poll_interval=1.0, log_interval=100, clean_events=True):
    from torch.utils.tensorboard import SummaryWriter

    print(f"📡 Watching '{log_root}' for JSONL runs...")

    run_states = {}

    while True:
        run_dirs = [d for d in glob.glob(os.path.join(log_root, "*")) if os.path.isdir(d)]

        for run_dir in run_dirs:
            metrics_path = resolve_run_metrics(run_dir)
            if metrics_path is None:
                continue

            if run_dir not in run_states:
                # 古いtfevents削除
                if clean_events:
                    for ev in glob.glob(os.path.join(run_dir, "events.out.tfevents.*")):
                        try:
                            os.remove(ev)
                            print(f"🧹 Cleared old TensorBoard file: {ev}")
                        except OSError:
                            pass

                print(f"🆕 New run detected: {run_dir}")
                # ファイル名固定化 (再起動時も追記扱い)
                writer = SummaryWriter(log_dir=run_dir, filename_suffix=".bridge")
                run_states[run_dir] = RunState(writer, metrics_path)

            state = run_states[run_dir]
            if state.source_path != metrics_path:
                if state.gzip_stream is not None:
                    state.gzip_stream.close()
                    state.gzip_stream = None
                state.source_path = metrics_path
            new_lines, new_pos = tail_run_state(state)

            if new_lines:
                for line in new_lines:
                    try:
                        j = json.loads(line)
                        process_json_line(state.writer, j)
                        state.line_count += 1
                        # ✅ 定期的に進捗ログを出力
                        if state.line_count % log_interval == 0:
                            print(f"[{time.strftime('%H:%M:%S')}] {run_dir}: processed {state.line_count} lines")
                    except json.JSONDecodeError:
                        # 書き込み途中行 → スキップ（再処理保証あり）
                        continue
                state.writer.flush()
            state.last_pos = new_pos

        time.sleep(poll_interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert MetricsLogger JSONL runs to TensorBoard events.")
    parser.add_argument("--runsdir", default="runs", help="Root directory where JSONL runs are stored")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--log-interval", type=int, default=1000, help="Print progress every N lines")
    parser.add_argument("--no-clean", action="store_true", help="Do not delete old TensorBoard event files")
    args = parser.parse_args()

    main(args.runsdir, args.interval, args.log_interval, clean_events=not args.no_clean)

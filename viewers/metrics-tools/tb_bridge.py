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
from torch.utils.tensorboard import SummaryWriter


def tail_jsonl(file_path, last_pos):
    """新しい行リストと次のファイル位置を返す"""
    new_lines = []
    new_pos = last_pos
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            f.seek(last_pos)
            for line in f:
                if line.strip():
                    new_lines.append(line.strip())
            new_pos = f.tell()
    except FileNotFoundError:
        pass
    return new_lines, new_pos


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
    print(f"📡 Watching '{log_root}' for JSONL runs...")

    run_states = {}  # { run_dir: (SummaryWriter, last_pos, line_count) }

    while True:
        run_dirs = [d for d in glob.glob(os.path.join(log_root, "*")) if os.path.isdir(d)]

        for run_dir in run_dirs:
            jsonl_path = os.path.join(run_dir, "metrics.jsonl")
            if not os.path.exists(jsonl_path):
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
                run_states[run_dir] = (writer, 0, 0)  # (writer, last_pos, line_count)

            writer, last_pos, line_count = run_states[run_dir]
            new_lines, new_pos = tail_jsonl(jsonl_path, last_pos)

            if new_lines:
                for line in new_lines:
                    try:
                        j = json.loads(line)
                        process_json_line(writer, j)
                        line_count += 1
                        # ✅ 定期的に進捗ログを出力
                        if line_count % log_interval == 0:
                            print(f"[{time.strftime('%H:%M:%S')}] {run_dir}: processed {line_count} lines")
                    except json.JSONDecodeError:
                        # 書き込み途中行 → スキップ（再処理保証あり）
                        continue
                writer.flush()
                run_states[run_dir] = (writer, new_pos, line_count)

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

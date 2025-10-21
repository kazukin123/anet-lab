# pip install tensorboard
# 使い方:
#   python tools/tb_bridge.py --input logs/train.jsonl --logdir tb_logs
# オプション:
#   --follow           ログを追尾して増分を取り込み(学習しながら可視化)
#   --reset            既存のlogdirを削除してから出力
#   --flush-interval   フラッシュ間隔(秒) 既定: 2

import argparse, json, os, time, shutil, datetime
from typing import Optional

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def make_scalar_summary(tag: str, value: float) -> Summary:
    return Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])


def make_text_summary(tag: str, text: str) -> Summary:
    # Text プラグイン用の 1要素 string Tensor を作る
    # 参考: TensorBoard の text サマリは string tensor を受け取る
    tensor = TensorProto(
        dtype=7,  # DT_STRING
        string_val=[text.encode("utf-8")],
        tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
    )
    meta = SummaryMetadata()  # plugin_name は省略でもTextプラグインで解釈される
    return Summary(value=[Summary.Value(tag=tag, tensor=tensor, metadata=meta)])


def is_numberlike(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def write_event(writer: EventFileWriter, summary: Summary, step: Optional[int] = None):
    ev = Event(wall_time=time.time(), summary=summary)
    if step is not None:
        ev.step = int(step)
    writer.add_event(ev)


def process_line(obj: dict, writer: EventFileWriter):
    typ = obj.get("type")

    if typ == "scalar":
        tag = obj.get("tag")
        step = obj.get("step", 0)
        val = obj.get("value", 0.0)
        if tag is not None:
            write_event(writer, make_scalar_summary(tag, float(val)), step)

    elif typ == "text":
        tag = obj.get("tag")
        step = obj.get("step", 0)
        val = obj.get("value", "")
        if tag is not None:
            write_event(writer, make_text_summary(f"{tag}", str(val)), step)

    elif typ == "config":
        key = obj.get("key")
        val = obj.get("value", "")
        if not key:
            return
        # 数値っぽければ Scalars、文字列は Text
        if is_numberlike(val):
            write_event(writer, make_scalar_summary(f"config/{key}", float(val)), 0)
        else:
            write_event(writer, make_text_summary(f"config/{key}", str(val)), 0)

    elif typ == "meta":
        event = obj.get("event")
        ts = obj.get("timestamp", "")
        if event == "start":
            # 文字系メタは Text に
            for k in ("timestamp", "torch_version", "device"):
                if obj.get(k) is not None:
                    write_event(writer, make_text_summary(f"meta/{k}", str(obj[k])), 0)
        elif event == "end":
            if ts:
                write_event(writer, make_text_summary("meta/end_timestamp", ts), 0)
            # 数値メタ（学習時間など）は Scalars へ
            if "train_time_sec" in obj:
                write_event(
                    writer, make_scalar_summary("meta/train_time_sec", float(obj["train_time_sec"])), 0
                )
    # それ以外のtypeは無視（将来拡張用）


def mkrunlogdirpath(logdir: str):
     now = datetime.datetime.now()
     runlogdir = logdir + '/' + now.strftime('%Y%m%d-%H%M%S')
     return runlogdir

def convert(input_path: str, logdir: str, follow: bool, flush_interval: float, reset: bool):
    runlogdir = mkrunlogdirpath(logdir)
    os.makedirs(runlogdir, exist_ok=True)
    print("runlogdir=" + runlogdir)
    writer = EventFileWriter(runlogdir)
    processed_bytes = 0

    try:
        line_count = 0
        with open(input_path, "rb") as f:
            # 既存ぶんを一気に処理
            for line in f:
                if not line.strip():
                    continue
                try:
                    line_count += 1
                    obj = json.loads(line.decode("utf-8"))
                    if ((line_count + 1) % 1000 == 1):
                        processed_bytes = f.tell()
                        print("Reading input_path=%s line_count=%d processed_bytes=%d" % (input_path, line_count, processed_bytes) )
                    process_line(obj, writer)
                except Exception:
                    # 壊れた行はスキップ
                    pass
            writer.flush()
            processed_bytes = f.tell()
            print("Loaded. input_path=%s  line_count=%d processed_bytes=%d" % (input_path, line_count, processed_bytes) )
        if not follow:
            return

        # 追尾モード: ファイル末尾から新規行を取り込む
        while True:
            print("Watching file. input_path=%s  processed_bytes=%d" % (input_path, processed_bytes) )
            time.sleep(flush_interval)
            size = os.path.getsize(input_path)
            if size < processed_bytes:
                # ローテーション・truncate された可能性 → 先頭から読み直し
                processed_bytes = 0
                print("Truncated.")
                if reset:
                    writer.close()
                    runlogdir = mkrunlogdirpath(logdir)
                    print("runlogdir=" + runlogdir)
                    writer = EventFileWriter(runlogdir)

            if size > processed_bytes:
                with open(input_path, "rb") as f:
                    f.seek(processed_bytes)
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            line_count += 1
                            obj = json.loads(line.decode("utf-8"))
                            process_line(obj, writer)
                            if ((line_count + 1) % 1000 == 1):
                                processed_bytes = f.tell()
                                print("Reading input_path=%s line_count=%d processed_bytes=%d" % (input_path, line_count, processed_bytes) )
                        except Exception:
                            pass
                    processed_bytes = f.tell()
                    writer.flush()
    finally:
        writer.flush()
        writer.close()




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../out/build/x64-Debug/cart2/train.jsonl", required=True, help="C++側が出力した JSONL (例: logs/train.jsonl)")
    ap.add_argument("--logdir", default="./tb_logs", help="TensorBoard の出力ディレクトリ (例: tb_logs)")
    ap.add_argument("--follow", action="store_true", help="ログを追尾して増分取り込み（学習と同時に可視化）")
    ap.add_argument("--reset", action="store_true", help="logdirを削除してから開始")
    ap.add_argument("--flush-interval", type=float, default=2.0, help="追尾時のフラッシュ間隔(秒)")
    args = ap.parse_args()

    if args.reset and os.path.isdir(args.logdir):
        shutil.rmtree(args.logdir)

    convert(args.input, args.logdir, args.follow, args.flush_interval, args.reset)
    print(f"Done. Events are in: {args.logdir}")


if __name__ == "__main__":
    main()

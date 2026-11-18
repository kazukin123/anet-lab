"""PRD170: PRD103 と同じ6構成・条件で、保存した実行体のメトリクスを比較する。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / ".scratch/prd170/validation"
CASES = ("GridMaze_off", "LunarLander_off", "Atari_off", "DropMerge_off", "GridMaze_on", "LunarLander_on")


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def execute(binary, label, case):
    destination = OUT / (label + "_" + case)
    destination.mkdir(parents=True, exist_ok=False)
    config = OUT / (case + ".txt")
    suffix = "p170_" + label + "_" + case
    arguments = [str(binary), "--config", str(config), "app.run_name=run_{t}_" + suffix]
    record = {"arguments": arguments, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}
    save(destination / "execution.json", record)
    print("START", label, case, flush=True)
    start = time.monotonic()
    with (destination / "launcher.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(arguments, cwd=REPO / "apps/runner", stdout=log,
                                stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    runs = list((OUT / "runs").glob("run_*_" + suffix))
    record.update(exit_code=result.returncode, seconds=time.monotonic() - start,
                  run=str(runs[0]) if len(runs) == 1 else None)
    save(destination / "execution.json", record)
    print("END", label, case, result.returncode, round(record["seconds"], 2), flush=True)
    if result.returncode or len(runs) != 1:
        raise RuntimeError("Run failed: " + case)
    # 履歴資材と同じ正規化を使い、性能・資源以外の tag / step / 値を全て比較する。
    path = REPO / "core/anet-core/testdata/prd084/compare_metrics.py"
    spec = importlib.util.spec_from_file_location("normalizer", path)
    normalizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalizer)
    normalizer.normalize(str(runs[0]), str(destination / "normalized.jsonl"), True)
    return record


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "compare"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--reference", default="before")
    parser.add_argument("--case", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    results = {}
    for case in args.case:
        if args.mode == "run":
            if args.binary is None:
                parser.error("--binary is required for run")
            results[case] = execute(args.binary.resolve(), args.label, case)
        else:
            before = (OUT / (args.reference + "_" + case) / "normalized.jsonl").read_bytes()
            after = (OUT / (args.label + "_" + case) / "normalized.jsonl").read_bytes()
            results[case] = {"exact_match": before == after, "points": after.count(b"\n"),
                             "sha256": hashlib.sha256(after).hexdigest()}
            print(case, results[case], flush=True)
    save(OUT / (args.label + "-" + args.mode + "-summary.json"), results)
    if args.mode == "compare" and not all(item["exact_match"] for item in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

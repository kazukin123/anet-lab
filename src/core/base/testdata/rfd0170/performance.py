"""RFD0170: 学習側 throughput を変更前→変更後の3巡で記録する（閾値なし）。"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[5]
BASE = REPO / ".scratch/rfd0170/validation"
TAG = "90_perf/12_exp_step_per_sec"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    destination = REPO / ".scratch/rfd0170/performance" / args.label
    destination.mkdir(parents=True, exist_ok=False)
    source = (BASE / "Atari_off.txt").read_text(encoding="utf-8")
    profile = "run.@rfd0098_validation : "
    lines = [source, profile + "app.exp_exit_step = 2048", profile + "app.batchrun.exp_exit_step = 2048",
             profile + "app.runs_dir = " + (destination / "runs").as_posix(),
             profile + "metrics.scalar.[" + TAG + "] = exp_step_per_sec @train $runner $exp_step interval:32"]
    config = destination / "run.txt"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    binaries = {"before": REPO / "apps/runner/bin/Debug/AnetRLRunner-rfd0170-before.exe",
                "after": REPO / "apps/runner/bin/Debug/AnetRLRunner-rfd0170-final.exe"}
    records = []
    for turn in range(1, 4):
        for group, binary in binaries.items():
            name = group + str(turn)
            suffix = "p170_perf_" + args.label + "_" + name
            command = [str(binary), "--config", str(config), "app.run_name=run_{t}_" + suffix]
            start = time.monotonic()
            record = {"group": group, "round": turn, "arguments": command,
                      "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
            print("START", name, flush=True)
            with (destination / (name + ".log")).open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=REPO / "apps/runner", stdout=log,
                                           stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                record["pid"] = process.pid
                save(destination / (name + "-execution.json"), record)
                code = process.wait()
            runs = list((destination / "runs").glob("run_*_" + suffix))
            record.update(exit_code=code, seconds=time.monotonic() - start,
                          run=str(runs[0]) if len(runs) == 1 else None)
            save(destination / (name + "-execution.json"), record)
            if code or len(runs) != 1:
                raise RuntimeError("Performance Run failed: " + name)
            report = destination / (name + "-metrics.json")
            with (destination / (name + "-inspect.log")).open("w", encoding="utf-8") as log:
                subprocess.run([sys.executable, str(REPO / "apps/tools/inspect_run.py"), "metrics",
                                str(runs[0]), "--metric", TAG, "--range", "1024:2048", "--stat", "mean",
                                "--format", "json", "--output", str(report)], cwd=REPO,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            metric = json.loads(report.read_text(encoding="utf-8"))["runs"][0]["metrics"][0]["ranges"][0]
            if metric["status"] != "ok" or not metric["count"]:
                raise RuntimeError("Missing throughput samples: " + name)
            record.update(mean_exp_steps_per_second=metric["mean"], samples=metric["count"])
            save(destination / (name + "-execution.json"), record)
            records.append(record)
            print("END", name, round(metric["mean"], 3), "exp/s;", metric["count"], "samples", flush=True)
    medians = {group: statistics.median(record["mean_exp_steps_per_second"] for record in records
                                      if record["group"] == group) for group in binaries}
    result = {"records": records, "median_exp_steps_per_second": medians,
              "after_before_ratio": medians["after"] / medians["before"], "acceptance_threshold": None}
    save(destination / "summary.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}), flush=True)


if __name__ == "__main__":
    main()

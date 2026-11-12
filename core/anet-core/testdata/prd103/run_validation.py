"""PRD103: 保存した入力による6構成の完全一致検証。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
OUT = REPO / ".scratch/prd103/validation"
CASES = (("GridMaze", False), ("LunarLander", False), ("Atari", False),
         ("DropMerge", False), ("GridMaze", True), ("LunarLander", True))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def execute(stage, label, env, enabled):
    # PRD098と同じ短縮条件を使い、実行体だけを独立して選べるようにする。
    previous = module(ROOT.parent / "prd098/run_validation.py", "prd098")
    previous.OUT = ROOT
    lines = previous.settings(env, before=True, enabled=enabled)
    lines = [line.replace((ROOT / "runs").as_posix(), (OUT / "runs").as_posix()) for line in lines]
    if env == "Atari":
        lines[-1] = lines[-1].replace(" > run.@prd098_validation", " > run.@btrnet > run.@prd098_validation")
    name = f"{label}_{env}_{'on' if enabled else 'off'}"
    destination = OUT / name
    if destination.exists():
        raise RuntimeError(f"Use a fresh --label; output already exists: {destination}")
    destination.mkdir(parents=True)
    config = destination / "run.txt"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    binary = ROOT / "baseline-bin/AnetRLRunner.exe" if stage == "before" else REPO / "apps/runner/bin/Debug/AnetRLRunner.exe"
    suffix = f"t103_{name}"
    args = [str(binary), "--config", str(config), "app.run_name=run_{t}_" + suffix]
    record = {"arguments": args, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "baseline_manifest": json.loads((ROOT / "baseline-manifest.json").read_text()), "stage": stage}
    save(destination / "execution.json", record)
    print("START", name, flush=True)
    started = time.monotonic()
    with (destination / "launcher.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(args, cwd=REPO / "apps/runner", stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW)
    runs = list((OUT / "runs").glob("run_*_" + suffix))
    record.update(exit_code=result.returncode, seconds=time.monotonic() - started,
        run=str(runs[0]) if len(runs) == 1 else None)
    save(destination / "execution.json", record)
    print("END", name, result.returncode, round(record["seconds"], 1), flush=True)
    if result.returncode or len(runs) != 1:
        raise RuntimeError(f"Run failed: {name}")
    normalizer = module(ROOT.parent / "prd084/compare_metrics.py", "normalize")
    normalizer.normalize(str(runs[0]), str(destination / "normalized.jsonl"), True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("before", "after", "compare"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--reference", default="before")
    parser.add_argument("--env", nargs="+", choices=sorted({env for env, _ in CASES}))
    args = parser.parse_args()
    errors = []
    for env, enabled in CASES:
        if args.env and env not in args.env:
            continue
        try:
            if args.stage != "compare":
                execute(args.stage, args.label, env, enabled)
                continue
            # 除外はPRD098と同じ性能・資源の3群だけ。点の値・tag・stepを完全一致で比較する。
            case = f"{env}_{'on' if enabled else 'off'}"
            before = (OUT / f"{args.reference}_{case}/normalized.jsonl").read_bytes()
            after = (OUT / f"{args.label}_{case}/normalized.jsonl").read_bytes()
            report = {"exact_match": before == after, "points": after.count(b"\n"),
                "sha256": hashlib.sha256(after).hexdigest()}
            save(OUT / f"{args.label}_{case}/comparison.json", report)
            print(case, report, flush=True)
            if before != after:
                raise RuntimeError(f"Metrics differ: {case}")
        except Exception as error:
            errors.append(str(error))
            print("FAIL", str(error), flush=True)
    save(OUT / f"{args.label}-{args.stage}-summary.json", {"errors": errors})
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()

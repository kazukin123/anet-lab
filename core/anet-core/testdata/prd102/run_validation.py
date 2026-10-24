"""PRD102: isolated public tests, frozen fixed-run comparison and config differences."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / ".scratch/prd102"
GRID_TEST = REPO / "core/envs/gridmaze1/bin/Debug/GridMazeEnv-test.exe"
CORE_TEST = REPO / "core/anet-core/bin/Debug/anet-core-test.exe"


def preserve_existing(path):
    # 再検証の記録も残し、以前の成功・失敗へ上書きしない。
    if not path.exists():
        return
    attempt = 1
    archive = path.with_name(path.stem + ".previous-" + str(attempt) + path.suffix)
    while archive.exists():
        attempt += 1
        archive = path.with_name(path.stem + ".previous-" + str(attempt) + path.suffix)
    path.rename(archive)


def save(path, value):
    preserve_existing(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def invoke(executable, tag, name):
    # 条件ごとの終了値を残し、失敗があっても後続条件は実行する。
    path = OUT / (name + ".log")
    preserve_existing(path)
    with path.open("w", encoding="utf-8") as stream:
        result = subprocess.run([str(executable), tag, "--reporter", "compact"], cwd=REPO,
                                stdout=stream, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW)
    print(name, "exit=" + str(result.returncode), flush=True)
    return {"name": name, "tag": tag, "exit_code": result.returncode, "log": str(path.relative_to(REPO))}


def tests():
    specs = [(CORE_TEST, "[prd102_optional]", "optional"),
             (GRID_TEST, "[gridmaze]", "gridmaze"),
             (GRID_TEST, "[prd100_log]", "fixed-log"),
             (GRID_TEST, "[prd102_process]", "process-1"),
             (GRID_TEST, "[prd102_process]", "process-2"),
             (GRID_TEST, "[prd102_concurrent]", "concurrent"),
             (GRID_TEST, "[prd102_late]", "late"),
             (GRID_TEST, "[prd102_settings]", "settings-errors"),
             (GRID_TEST, "[prd102_runner]", "runner")]
    rows = [invoke(executable, tag, name) for executable, tag, name in specs]
    save(OUT / "tests.json", rows)
    if any(row["exit_code"] for row in rows):
        raise RuntimeError("Public tests failed; inspect tests.json and individual logs")


def after():
    # 現行PRD100 driverの正規化・実行体隔離を再利用し、出力先だけPRD102へ分離する。
    spec = importlib.util.spec_from_file_location("prd100_current", REPO / "core/anet-core/testdata/prd100/run_validation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.execute("after")


def compare():
    before = (OUT / "before/normalized.jsonl").read_bytes()
    after_data = (OUT / "after/normalized.jsonl").read_bytes()
    result = {"exact_match": before == after_data, "before_points": before.count(b"\n"),
              "after_points": after_data.count(b"\n"), "before_sha256": hashlib.sha256(before).hexdigest(),
              "after_sha256": hashlib.sha256(after_data).hexdigest()}
    save(OUT / "comparison.json", result)
    print(json.dumps(result), flush=True)
    if not result["exact_match"] or not result["before_points"]:
        raise RuntimeError("Fixed-run scalar comparison failed")


def settings():
    for name in ("config-before.json", "config-after.json", "config-comparison.json"):
        preserve_existing(OUT / name)
    row = invoke(GRID_TEST, "[prd102_config_snapshot]", "config-snapshot")
    if row["exit_code"]:
        raise RuntimeError("Config snapshot test failed")
    before = json.loads((OUT / "config-before.json").read_text(encoding="utf-8"))
    after_data = json.loads((OUT / "config-after.json").read_text(encoding="utf-8"))
    rows = []
    for name, old in before.items():
        new = after_data[name]
        removed = sorted(old.keys() - new.keys())
        added = sorted(new.keys() - old.keys())
        changed = [key for key in old.keys() & new.keys() if old[key] != new[key]]
        if name in ("gridmaze-default", "gridmaze_muzero-default"):
            expected_removed = 2 if name == "gridmaze_muzero-default" else 1
            ok = (len(removed) == expected_removed and all(key.startswith("GridMazeEnv.") for key in removed)
                  and added == ["GridMazeEnv.seed_mode"] and new[added[0]] == "process" and not changed)
        else:
            ok = not removed and not added and not changed
        rows.append({"id": name, "before_keys": len(old), "after_keys": len(new),
                     "removed": removed, "added": added, "changed": changed, "pass": ok})
    save(OUT / "config-comparison.json", rows)
    print("Config inputs:", len(rows), "passed:", sum(row["pass"] for row in rows), flush=True)
    if not all(row["pass"] for row in rows):
        raise RuntimeError("Unexpected config differences; inspect config-comparison.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("tests", "after", "compare", "settings"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"tests": tests, "after": after, "compare": compare, "settings": settings}[args.stage]()


if __name__ == "__main__":
    main()

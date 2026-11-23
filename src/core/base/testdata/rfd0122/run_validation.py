"""RFD0122: 凍結した入力を既存の検証資材で前後比較する。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import importlib.util

REPO = Path(__file__).resolve().parents[5]
BIN = REPO / "apps/runner/bin/Debug"
ENVS = {"GridMaze": "gm", "LunarLander": "ll", "Atari": "at", "DropMerge": "dm"}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = load("rfd0122_common", REPO / "src/core/base/testdata/rfd0109/run_validation.py")


def compare(baseline, output):
    # 同じ凍結設定を前後の実行体へ渡し、過去のRFDの基準は使い回さない。
    output.mkdir(parents=True, exist_ok=False)
    driver = load("rfd0122_settings", REPO / "src/core/base/testdata/rfd0098/run_validation.py")
    normalizer = load("rfd0122_normalize", REPO / "src/core/base/testdata/rfd0084/compare_metrics.py")
    driver.OUT = output
    shutil.copytree(baseline / "config", output / "baseline-config")
    info = COMMON.execute([str(BIN / "AnetRLRunner-gui-test.exe"), "[rfd0109_config]"],
                          output / "capture", cwd=REPO,
                          env={**os.environ, "RFD0109_CONFIG_OUTPUT": str(output)})
    if info["exit_code"]:
        raise RuntimeError("Frozen configuration capture failed")
    reports = {}
    for env, short_name in ENVS.items():
        try:
            text = "\n".join(driver.settings(env, before=True)).replace("rfd0098", "rfd0122") + "\n"
            points = []
            for side, binary in (("b", baseline / "bin/AnetRLRunner.exe"),
                                 ("a", BIN / "AnetRLRunner.exe")):
                config = output / (side + "_" + short_name + ".txt")
                config.write_text(text, encoding="utf-8", newline="\n")
                suffix = "tmp_122_" + side + "_" + short_name
                folder = output / (side + "_" + short_name)
                info = COMMON.execute([str(binary), "--config", str(config),
                                       "app.run_name=run_{t}_" + suffix], folder)
                runs = list((output / "runs").glob("run_*_" + suffix))
                if info["exit_code"] or len(runs) != 1:
                    raise RuntimeError("Run failed: " + side + " " + env)
                normalized = folder / "normalized.jsonl"
                normalizer.normalize(runs[0], normalized, include_noisy=True)
                points.append(normalized.read_bytes())
                COMMON.save(folder / "inputs.json", {"run": str(runs[0]),
                                                    "config_sha256": COMMON.digest(config)})
            reports[env] = {"exact_match": points[0] == points[1],
                            "before_points": points[0].count(b"\n"), "after_points": points[1].count(b"\n"),
                            "before_sha256": hashlib.sha256(points[0]).hexdigest(),
                            "after_sha256": hashlib.sha256(points[1]).hexdigest()}
            COMMON.save(output / "comparison.json", reports)
            print("COMPARE", env, reports[env], flush=True)
        except Exception as error:
            # 失敗した環境のログを残し、独立した次の環境も確認する。
            reports[env] = {"exact_match": False, "error": str(error)}
            COMMON.save(output / "comparison.json", reports)
            print("FAILED", env, str(error), flush=True)
    if not all(item["exact_match"] for item in reports.values()):
        raise RuntimeError("Learning scalar mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.baseline.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

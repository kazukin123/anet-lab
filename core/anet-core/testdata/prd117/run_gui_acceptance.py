"""Runnerの起動テストを別プロセスで実行し、転送されたログを保存する。"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    output = root / ".scratch" / "prd117" / "gui-acceptance"
    executable = root / "apps/runner/bin/Debug/AnetRLRunner-gui-test.exe"
    cases = (
        ("logs", "[log_panel]"),
        ("conv2d", "[conv2d_config]"),
        ("startup", "[prd117_startup]"),
        ("mode_online", "[prd117_mode_online]"),
        ("mode_batchrun", "[prd117_mode_batchrun]"),
        ("centre_fallback", "[prd110_fallback]"),
        ("centre_redundant_show", "[prd110_redundant_show]"),
        ("save_zero", "[prd117_save_zero]"),
    )
    results = []
    for name, tag in cases:
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        report = destination / "result.xml"
        report.unlink(missing_ok=True)
        process = subprocess.Popen(
            [str(executable), tag, "--reporter", "junit", "--out", str(report)],
            cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        timed_out = False
        try:
            console, _ = process.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            console, _ = process.communicate()
        (destination / "console.log").write_bytes(console)
        run = Path(tempfile.gettempdir()) / f"anet-prd109-{process.pid}" / "startup"
        for filename in ("stdout.log", "stderr.log", "startup.log"):
            source = run / filename
            if source.is_file():
                shutil.copyfile(source, destination / filename)
        tests = []
        failures = []
        if report.is_file():
            tree = ET.parse(report)
            tests = tree.findall(".//testcase")
            failures = tree.findall(".//failure") + tree.findall(".//error")
        passed = process.returncode == 0 and not timed_out and bool(tests) and not failures
        results.append({"name": name, "tag": tag, "exit_code": process.returncode,
                        "timed_out": timed_out, "test_cases": len(tests),
                        "failures": len(failures), "passed": passed,
                        "artifacts": str(destination), "run": str(run)})
        print(f"{name}: exit={process.returncode}, cases={len(tests)}, failures={len(failures)}, passed={passed}", flush=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())

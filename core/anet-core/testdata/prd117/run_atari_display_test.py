"""Atariの表示通知をSDLのdummy driverと隔離した実行ファイルで検証する。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdl-dll", type=Path, default=root / ".scratch/prd117/sdl2/SDL2.dll")
    args = parser.parse_args()
    source = root / "core/envs/atari1/bin/Debug/AtariEnv-test.exe"
    output = root / ".scratch/prd117/atari-display"
    output.mkdir(parents=True, exist_ok=True)
    executable = output / source.name
    # ALEの安全なDLL探索はPATHを使わないため、検証用exeと同じdirectoryに置く。
    shutil.copyfile(source, executable)
    shutil.copyfile(args.sdl_dll, output / "SDL2.dll")
    environment = dict(os.environ)
    environment["SDL_VIDEODRIVER"] = "dummy"
    environment["PATH"] = str(source.parent) + os.pathsep + environment.get("PATH", "")
    report = output / "result.xml"
    report.unlink(missing_ok=True)
    with (output / "console.log").open("wb") as log:
        process = subprocess.run(
            [str(executable), "[prd117_display]", "--reporter", "junit", "--out", str(report)],
            cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=180,
        )
    cases = []
    failures = []
    if report.is_file():
        tree = ET.parse(report)
        cases = tree.findall(".//testcase")
        failures = tree.findall(".//failure") + tree.findall(".//error")
    passed = process.returncode == 0 and bool(cases) and not failures
    result = {"exit_code": process.returncode, "test_cases": len(cases), "failures": len(failures),
              "passed": passed, "executable_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "sdl_sha256": hashlib.sha256(args.sdl_dll.read_bytes()).hexdigest()}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

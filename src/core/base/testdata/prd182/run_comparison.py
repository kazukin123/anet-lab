import argparse
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
repo = Path.cwd()
validation = repo / ".scratch/prd182/comparison"
testdata = repo / "src/core/base/testdata"
if not testdata.is_dir():
    testdata = repo / "core/anet-core/testdata"
spec = importlib.util.spec_from_file_location("prd128_validation", testdata / "prd128/run_validation.py")
prd128 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prd128)
base = prd128.BASE
base.ROOT = validation
base.BASELINE = validation / "baseline"
base.INPUTS = validation / "inputs"
prd128.ROOT = validation

# ログのソース位置だけを正規化し、メッセージと順序を比較する。
source = re.compile(r"(?:(?:[A-Za-z]:)?[^\s\[\]\"'<>]*[/\\])?[\w.-]+\.(?:cpp|hpp):\d+\b")
original_records = prd128.artifact_records

def artifact_records(normalized):
    records = original_records(normalized)
    records["warnings"] = [source.sub("<SOURCE>", line) for line in records["warnings"]]
    return records

prd128.artifact_records = artifact_records
def verify_baseline():
    manifest_path = base.BASELINE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = base.BASELINE / relative
        if not path.is_file() or base.COMMON.digest(path) != expected:
            raise RuntimeError(f"Saved comparison input changed: {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare PRD182 behavior against the saved pre-layout runner.")
    parser.add_argument("mode", choices=("snapshot", "repeat", "compare"))
    parser.add_argument("--label", default="before-layout")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if args.mode != "snapshot":
        verify_baseline()
    if args.mode == "snapshot":
        debug = repo / "apps/runner/bin/Debug"
        binaries = ("AnetRLRunner.exe", "AnetRLRunner-gui-test.exe")
        for name in binaries:
            if not (debug / name).is_file():
                raise FileNotFoundError(debug / name)
        saved_debug = base.BASELINE / "Debug"
        saved_debug.mkdir(parents=True, exist_ok=False)
        for path in debug.iterdir():
            if path.name in binaries or path.suffix.lower() == ".dll":
                shutil.copy2(path, saved_debug / path.name)
        shutil.copytree(repo / "apps/runner/config", base.BASELINE / "config")
        manifest = {path.relative_to(base.BASELINE).as_posix(): base.COMMON.digest(path)
                    for path in base.BASELINE.rglob("*") if path.is_file()}
        base.COMMON.save(base.BASELINE / "manifest.json", manifest)
        prd128.prepare()
        print(f"Snapshot and {len(prd128.definitions())} input cases saved at {validation}", flush=True)
    elif args.mode == "repeat":
        base.repeat(args.label, tuple(prd128.definitions()), args.workers)
    else:
        base.compare(args.label, "before-layout", tuple(prd128.definitions()), args.workers)

if __name__ == "__main__":
    main()

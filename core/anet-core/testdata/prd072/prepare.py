"""固定commitの設定を準備し、版管理されたgoldenとの比較・明示採取を実行する。"""

import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import zipfile

from migrate import migrate_text
from expected import verify_expectations


REPO = Path(__file__).resolve().parents[4]
ROOT = REPO / ".scratch/prd072-differential/validation"
MANIFEST = Path(__file__).with_name("manifest.json")


def write_frozen(path, content):
    # 既存の採取条件を黙って書き換えず、同一内容の再準備だけを許す。
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Frozen input differs: {path}. Use a fresh checkout for another manifest.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def prepare(manifest):
    commit = manifest["commit"]
    archive = subprocess.check_output(
        ["git", "archive", "--format=zip", commit, "apps/runner/config"], cwd=REPO
    )
    with zipfile.ZipFile(io.BytesIO(archive)) as entries:
        for entry in entries.infolist():
            if not entry.is_dir():
                write_frozen(ROOT / "frozen" / entry.filename, entries.read(entry))
                content = entries.read(entry).decode("utf-8")
                migrated, _ = migrate_text(Path(entry.filename).name, content)
                write_frozen(ROOT / "migrated" / entry.filename, migrated.encode("utf-8"))
    write_frozen(ROOT / "manifest.json", MANIFEST.read_bytes())
    for tree, inputs in (("frozen", "inputs-original"), ("migrated", "inputs")):
        config = (ROOT / tree / "apps/runner/config").as_posix()
        for item in manifest["inputs"]:
            content = f'$include <{config}/_main.txt>\n$include <{config}/{item["env"]}.txt>\n'
            write_frozen(ROOT / inputs / (item["id"] + ".txt"), content.encode("utf-8"))
    print(f"Prepared {len(manifest['inputs'])} inputs from {commit}", flush=True)


def report_comparison(manifest):
    # 値と診断だけを比較し、生成された行順は比較対象にしない。
    rows = []
    for item in manifest["inputs"]:
        name = item["id"] + ".json"
        before = json.loads((MANIFEST.parent / "baseline" / name).read_text(encoding="utf-8"))
        after = json.loads((ROOT / "actual" / name).read_text(encoding="utf-8"))
        rows.append({"id": item["id"], "keys": len(after["values"]),
                     "value_differences": [key for key in sorted(before["values"].keys() | after["values"].keys())
                                           if before["values"].get(key) != after["values"].get(key)],
                     "old_selection_keys": [entry["key"] for entry in before["resolution"]["selections"]],
                     "new_selection_keys": [entry["key"] for entry in after["resolution"]["selections"]],
                     "overrides": after["overrides"]})
    (ROOT / "comparison_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "capture", "compare"), nargs="?", default="prepare")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    # captureは契約変更時の明示採取に限り、版管理されたgoldenを上書きしない。
    capture = args.mode == "capture"
    baseline = MANIFEST.parent / "baseline"
    if capture and any((baseline / (item["id"] + ".json")).exists() for item in manifest["inputs"]):
        raise RuntimeError(f"Baseline already exists: {baseline}")
    prepare(manifest)
    if args.mode == "prepare":
        return
    if not capture:
        verify_expectations(REPO, ROOT, manifest)
    executable = REPO / "core/anet-core/bin/Debug" / "anet-core-test.exe"
    environment = os.environ.copy()
    environment.pop("ANET_PRD072_CAPTURE", None)
    if capture:
        environment["ANET_PRD072_CAPTURE"] = "1"
    result = subprocess.run([str(executable), "[prd072-baseline]"], cwd=REPO, env=environment)
    if not capture and result.returncode == 0:
        report_comparison(manifest)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

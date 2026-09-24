"""現用の選択宣言からownerを求め、説明のない個別葉を検出する。"""

import base64
import json
from pathlib import Path
import re
import subprocess

from migrate import normalize, assignment_role, expected_operator

REPO = Path(__file__).resolve().parents[4]
ROOT = REPO / ".scratch/prd072-default-leaf"


def assignments(line):
    # コロン表記、Run、batや生成ツールの文字列にも同じ抽出を使う。
    for match in re.finditer(r"(?<![\w.@])([\w.@\[\]/-]+(?:\s*:\s*[\w.@\[\]/$-]+)?(?:\s*\.\s*\$)?)\s*(\?=|=)\s*([^\r\n]*)", line):
        key = normalize(match[1])
        run = key.startswith("run.@")
        if run:
            key = ".".join(key.split(".")[2:])
        value = re.split(r"[\"'`#]", match[3], maxsplit=1)[0].strip()
        yield key, match[2], value, run


def main():
    command = ["rg", "--json", "=", "apps", "core", "-g", "*.txt", "-g", "*.bat",
               "-g", "*.py", "-g", "*.ps1", "-g", "*.cpp", "-g", "*.hpp",
               "-g", "!**/bin/**", "-g", "!**/lib/**", "-g", "!**/runs*/**",
               "-g", "!**/workspaces/**", "-g", "!**/testdata/**"]
    result = subprocess.run(command, cwd=REPO, capture_output=True, encoding="utf-8")
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    entries, owners, sources = [], set(), set()
    file_owners = {}
    for raw in result.stdout.splitlines():
        data = json.loads(raw)
        if data["type"] != "match":
            continue
        data = data["data"]
        file = data["path"]["text"].replace("\\", "/")
        line = data["lines"].get("text")
        if line is None:
            line = base64.b64decode(data["lines"]["bytes"]).decode("cp932")
        if line.lstrip().startswith(("#", "//", "REM ")):
            continue
        for key, op, value, run in assignments(line):
            entry = dict(file=file, line=data["line_number"], key=key, operator=op, run=run)
            entries.append(entry)
            if key.endswith(".$") and "@" not in key and not file.endswith((".cpp", ".hpp", "_test.ps1", "_test.py")):
                owners.add(key[:-2])
                file_owners.setdefault(file, set()).add(key[:-2])
                for term in value.split(">"):
                    term = term.strip()
                    if re.fullmatch(r"[\w.\[\]-]+", term):
                        sources.add(term)
    report, errors = [], []
    for entry in entries:
        key = entry["key"]
        scope = owners
        if entry["file"].startswith("apps/runner/config/") and Path(entry["file"]).name not in ("common.txt", "agent.txt"):
            scope = file_owners.get(entry["file"], set()) | file_owners.get("apps/runner/config/agent.txt", set()) | file_owners.get("apps/runner/config/common.txt", set())
        # OptunaはDropMerge本体の後に追加設定をincludeする（dropmerge_optuna.py）。
        if entry["file"] == "apps/runner/config/DropMerge_optuna.txt":
            scope = scope | file_owners.get("apps/runner/config/DropMerge.txt", set())
        matches = [owner for owner in scope if key.startswith(owner + ".")]
        role = assignment_role(Path(entry["file"]).name, key) if entry["file"].startswith("apps/runner/config/") else None
        if role is not None:
            entry["owner"] = max(matches, key=len) if matches else key.split(".")[0]
            if entry["run"]:
                role = "explicit Run assignment"
            entry["reason"] = role
            entry["expected_operator"] = expected_operator(role)
            report.append(entry)
            if entry["operator"] != entry["expected_operator"]:
                errors.append(entry)
            continue
        if not matches or key.endswith(".$"):
            continue
        entry["owner"] = max(matches, key=len)
        reason = None
        if entry["file"].endswith((".cpp", ".hpp")):
            reason = "C++ diagnostic text or independent fixture"
        elif "@" in key:
            reason = "profile definition"
        elif entry["run"]:
            reason = "explicit Run assignment"
        elif any(key.startswith(source + ".") for source in sources):
            reason = "ordinary selection source"
        elif entry["operator"] == "?=":
            reason = "default leaf"
        elif entry["file"].endswith("_test.cpp") or entry["file"].endswith("_test.ps1"):
            reason = "independent test or explicit test override"
        elif entry["file"] == "apps/runner/tools/dropmerge_optuna.py" and key in ("app.run_name", "app.runs_dir"):
            reason = "explicit trial output location"
        elif entry["file"] == "apps/runner/tools/dropmerge_optuna.py" and key == "run.seed":
            reason = "explicit trial seed"
        elif entry["file"].endswith(".bat"):
            reason = "explicit CLI assignment"
        elif entry["file"] == "apps/runner/config/DropMerge_optuna.txt" and key in ("DropMergeEnv.seed_mode", "DropMergeEnv.global_seed"):
            reason = "intentional Optuna seed override"
        entry["reason"] = reason
        report.append(entry)
        if reason is None:
            errors.append(entry)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "default-leaf-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Assignments: {len(entries)}, audited: {len(report)}, errors: {len(errors)}")
    for entry in errors:
        print(f"{entry['file']}:{entry['line']}: {entry['key']}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

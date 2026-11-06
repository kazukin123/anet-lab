"""CHANGELOGの対応表から廃止キーを引き、現用契約の移行漏れを検出する。"""
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "core/anet-core/testdata/prd072"))
from profiled_migration import names, other


def main():
    # D13の履歴・en・CHANGELOGは対象外。C++のDTOメンバ.valueはキー記法と区別する。
    paths = []
    for root in ("core/anet-core/include", "core/anet-core/src", "core/anet-core/testdata/prd072", "core/anet-core/testdata/prd101",
                 "apps/runner/config", "apps/runner/src", "core/envs", "docs/wiki/categories", "docs/wiki/apps"):
        paths.extend(p for p in (REPO / root).rglob("*")
                     if p.is_file() and p.suffix in (".hpp", ".cpp", ".py", ".json", ".md", ".txt"))
    paths.extend((REPO / "apps").glob("*.bat"))
    paths.extend((REPO / "docs/design").glob("*.jp.md"))
    paths.append(REPO / "CONTEXT.md")
    errors = []
    for path in sorted(set(paths)):
        text = path.read_text(encoding="cp932" if path.suffix == ".bat" else "utf-8-sig")
        for line_no, line in enumerate(text.splitlines(), 1):
            literals = " ".join(re.findall(r'"(?:[^"\\]|\\.)*"', line))
            for key in (*names, *other):
                searchable = literals if key.endswith(".value") and path.suffix in (".hpp", ".cpp") else line
                if key in searchable:
                    errors.append({"path": path.relative_to(REPO).as_posix(), "line": line_no, "key": key})
    output = REPO / ".scratch/prd101/contract-scan.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in errors:
        print(f"{row['path']}:{row['line']}: {row['key']}")
    print(f"Files: {len(set(paths))}, errors: {len(errors)}")
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()

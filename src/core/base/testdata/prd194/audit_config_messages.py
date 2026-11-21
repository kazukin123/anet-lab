"""Collect candidate relative settings in non-test C++ diagnostics."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys


REPO = Path(__file__).resolve().parents[5]
ERROR_STARTS = ("ANET_SYSTEM_ERROR(", "ANET_CHECK_MSG(", "ANET_ASSERT_MSG(")
LOG_START = re.compile(r"\b(info|warn|error|fatal)(_once)?\((?:[^()]|\([^()]*\))*\)\s*(<<|$)")
CPP_STRING = re.compile(r'"((?:\\.|[^"\\])*)"')
FULL_KEY_LOOKUP = re.compile(r'\bGetFullKey\(\s*"(?:\\.|[^"\\])*"\s*\)')
HIERARCHICAL = re.compile(
    r"(?<![A-Za-z0-9_.\]}])([A-Za-z_][A-Za-z0-9_]*(?:\.(?:\[[^\]]*\]|\{\}|[A-Za-z_][A-Za-z0-9_]*))+)"
)
ASSIGNMENT = re.compile(r"(?<![A-Za-z0-9_.\]}])([a-z][a-z0-9_]{2,})=")
READ_FIELD = re.compile(r"ANET_READ_CONFIG\(\s*[A-Za-z_]+\s*,\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\)")
RELATIVE_ARG = re.compile(r'[(,]\s*"([a-z_][a-z0-9_]*(?:\.\[[^\]"]*\]|\.[a-z_][a-z0-9_]*)+)"\s*[,)]')
READ_CALLS = (
    "ANET_READ_CONFIG", "ReadConfig(", "ReadSubConfig(", ".Get(", ".Get<", ".Has(", ".Set(",
    ".Read(", "MakeSubConfigData(", "CreateReadScope(", "MakeReadScope(", "MakeDefaultConfigKey(",
    "MarkInventory(", "HasConfigValue(", "RecordConfigValue(", "GetSubConfigTags(",
    "MakeTaggedSubConfigKey(", "CopyValueFrom(", "GetFullKey(", "RecordFullKey(",
    "ReadInheritedSubConfig(", ".CopyFieldFrom(",
)
EXTENSIONS = set("txt cpp hpp json jsonl db gz mkv png py bat exe log md g e".split())
ROOTS = set("app run env agent net metrics backend image".split())
TEST_PATH_PARTS = ("_test.", "/testdata/", "/src_testing/", "/include/anet/test/")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO).decode("utf-8-sig")


def source_files(ref):
    if ref is None:
        paths = git("ls-files", "-z", "--", "src")
    else:
        paths = git("ls-tree", "-rz", "--name-only", ref, "--", "src")
    for path in sorted(set(paths.strip("\0").split("\0"))):
        if path.endswith((".cpp", ".hpp")) and not any(part in path for part in TEST_PATH_PARTS):
            yield path


def is_comment(line):
    return line.lstrip().startswith(("//", "/*", "*"))


def diagnostic_kind(line):
    if any(start in line for start in ERROR_STARTS):
        return "error"
    match = LOG_START.search(line)
    if match is None:
        return None
    return "error" if match.group(1) == "fatal" else match.group(1)


def has_relative_name(name):
    first = name.split(".", 1)[0]
    return (
        first not in ROOTS
        and not first[0].isupper()
        and name.rsplit(".", 1)[-1] not in EXTENSIONS
        and not name.startswith(("std.", "torch.", "e.g", "i.e"))
    )


def collect(path, source, fields):
    lines = source.splitlines()
    index = 0
    while index < len(lines):
        start = index
        line = lines[index]
        index += 1
        if is_comment(line):
            continue
        kind = diagnostic_kind(line)
        if kind is not None:
            # 診断に連なる文字列だけを集め、表示しない完全キー検索の引数を除く。
            while index < min(start + 15, len(lines)) and not lines[index - 1].rstrip().endswith(";"):
                index += 1
            statement = "\n".join(lines[start:index])
            message = "{}".join(CPP_STRING.findall(FULL_KEY_LOOKUP.sub("GetFullKey()", statement)))
            names = [name for name in HIERARCHICAL.findall(message) if has_relative_name(name)]
            names.extend(name for name in ASSIGNMENT.findall(message) if name in fields)
        elif not any(call in line for call in READ_CALLS):
            kind = "call"
            names = [name for name in RELATIVE_ARG.findall(line) if has_relative_name(name)]
        else:
            continue
        if names:
            yield path, start + 1, kind, names


def row_counts(rows):
    return Counter((path, kind, tuple(names)) for path, kind, names in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="Read sources from this Git revision")
    parser.add_argument("--expected", type=Path, help="Compare with this JSON Lines file")
    args = parser.parse_args()
    # 読み口の欄名を共有集合に集めてから、ファイルごとの診断と呼び出しを検査する。
    sources = {
        path: git("show", f"{args.ref}:{path}") if args.ref else (REPO / path).read_text(encoding="utf-8-sig")
        for path in source_files(args.ref)
    }
    fields = {
        match.rsplit(".", 1)[-1]
        for source in sources.values()
        for match in READ_FIELD.findall(source)
    }
    rows = []
    for path, source in sources.items():
        rows.extend(collect(path, source, fields))

    if args.expected is None:
        for path, line, kind, names in rows:
            print(json.dumps([path, line, kind, names]))
        return 0

    actual = row_counts((path, kind, names) for path, _, kind, names in rows)
    expected = row_counts(json.loads(line) for line in args.expected.read_text(encoding="utf-8").splitlines())
    for label, differences in (("unexpected", actual - expected), ("missing", expected - actual)):
        for row, count in sorted(differences.items()):
            print(f"{label} ({count}): {json.dumps(row)}")
    if actual != expected:
        return 1
    print(f"Matched {sum(actual.values())} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())

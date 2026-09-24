"""PRD061 P1 の解決値・typed 設定を、明示した改名対応だけで比較する。"""

import argparse
import json
from pathlib import Path


RENAMES = {
    "train.seed": "run.seed",
    "train.num_envs": "run.train.num_envs",
    "train.main_runner_type": "run.train.runner_type",
    "train.eval": "run.eval",
}


def rename(value):
    for old, new in RENAMES.items():
        value = value.replace(old, new)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    before_paths = sorted(args.before.glob("*.json"))
    if not before_paths:
        raise RuntimeError(f"No baseline captures: {args.before}")
    differences = []
    for path in before_paths:
        before = json.loads(path.read_text(encoding="utf-8"))
        after = json.loads((args.after / path.name).read_text(encoding="utf-8"))
        expected = {rename(key): rename(value) for key, value in before["values"].items()}
        for key in sorted(expected.keys() | after["values"].keys()):
            if expected.get(key) != after["values"].get(key):
                differences.append((path.stem, key, expected.get(key), after["values"].get(key)))
        if before["typed"] != after["typed"]:
            differences.append((path.stem, "typed", "baseline", "changed"))
    for difference in differences:
        print(difference)
    print(f"Compared {len(before_paths)} inputs; differences={len(differences)}")
    raise SystemExit(bool(differences))


if __name__ == "__main__":
    main()

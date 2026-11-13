"""PRD127の設定差分を、固定入力と解決済みの全キーで検査する。"""

import json
import os
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[4]
BASELINE = REPO / ".scratch/prd127/baseline-20261112"
OUTPUT = REPO / ".scratch/prd127/config-comparison"
BINARY = REPO / "core/anet-core/bin/Debug/anet-core-test.exe"
GOLDEN = json.loads(Path(__file__).with_name("trace_columns.json").read_text(encoding="utf-8"))


def resolve(root, name, chain, label):
    # 変更前後で同じ選択を使い、差分の許可はgoldenの3列だけにする。
    stem = OUTPUT / f"{label}-{name}"
    source, destination = stem.with_suffix(".txt"), stem.with_suffix(".json")
    source.write_text(f"$include <{root.as_posix()}/_main.txt>\n"
                      f"$include <{root.as_posix()}/{name}.txt>\n"
                      f"run.$ = {chain}\n", encoding="utf-8")
    environment = dict(os.environ, ANET_PRD117_RESOLVE_INPUT=str(source),
                       ANET_PRD117_RESOLVE_OUTPUT=str(destination))
    with stem.with_suffix(".log").open("w", encoding="utf-8") as log:
        subprocess.run([str(BINARY), "[prd117-resolve]"], cwd=REPO, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(destination.read_text(encoding="utf-8"))["config"]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in ("Atari", "GridMaze", "LunarLander"):
        # 単独と両方の順序を解決し、列の上書きと表示位置を確認する。
        for case, chain, selected in (
            ("plain", "", False),
            ("rnd", "run.@rnd", True),
            ("skip", "run.@skip", False),
            ("skip-rnd", "run.@skip > run.@rnd", True),
            ("rnd-skip", "run.@rnd > run.@skip", True),
        ):
            before = resolve(BASELINE / "baseline-config", name, chain, "before-" + case)
            after = resolve(REPO / "apps/runner/config", name, chain, "after-" + case)
            added = {key: after[key] for key in after.keys() - before.keys()}
            removed = sorted(before.keys() - after.keys())
            changed = {key: [before[key], after[key]] for key in before.keys() & after.keys()
                       if before[key] != after[key]}
            expected = GOLDEN if selected else {}
            if added != expected or removed or changed:
                raise AssertionError(dict(name=name, case=case, added=added,
                                          removed=removed, changed=changed))
            if not selected and GOLDEN.keys() & after.keys():
                raise AssertionError(f"intrinsic Trace columns leaked without RND: {name}")
            if selected:
                order = sorted(int(key.removeprefix("app.eval_trace_panel.columns.[").removesuffix("]"))
                               for key in after if key.startswith("app.eval_trace_panel.columns.["))
                if order.index(235) != order.index(230) + 1:
                    raise AssertionError(f"rnd_bonus does not immediately follow reward: {name}/{case}")
            reports.append(dict(name=name, case=case, selected=selected, added=added))
    # 実装時に採取した既定7環境の比較記録も再検査する。現在のenv選択は固定しない。
    for source in sorted((BASELINE / "resolved-before").glob("*.json")):
        before = json.loads(source.read_text(encoding="utf-8"))
        after = json.loads((BASELINE / "resolved-after" / source.name).read_text(encoding="utf-8"))
        if before != after:
            raise AssertionError(f"default resolution changed: {source.name}")
    (OUTPUT / "result.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print("PASS: 15 profile combinations preserve existing keys and column order; archived default snapshots match")


if __name__ == "__main__":
    main()

"""新しい既定設定を実際の短い Run へ通し、Actor の dump を確認する。"""
import json
import re
from pathlib import Path

from common import OUT, REPO, execute, load, properties, resolve, save


def inspect(run, atari=False):
    selected = properties(run / "config/config_data.txt")
    actors = sorted({match[1] for key in selected
                     if (match := re.fullmatch(r"DefaultDQNAgent\.actor\.\[([^]]+)\]\.network", key))})
    assert {"train", "eval", "eval_target"}.issubset(actors)
    if atari:
        assert "greedy" in actors
    for actor in actors:
        prefix = f"DefaultDQNAgent.actor.[{actor}].skip."
        assert selected[prefix + "nn_forward_mode"] == "shared", (run, actor)
        assert prefix + "nn_forward_key" not in selected, (run, actor)
    return {"run": str(run), "actors": actors, "mode": "shared", "skip_forward_keys": []}


def main():
    destination = OUT / "dump"
    destination.mkdir(parents=True, exist_ok=False)
    previous = load("prd098/run_validation.py", "prd106_dump_settings")
    previous.OUT = OUT
    lines = [line.replace((REPO / "apps/runner/config").as_posix(), (OUT / "current-config").as_posix())
             for line in previous.settings("GridMaze", before=False, enabled=False)]
    lines += ["run.@prd098_validation : app.exp_exit_step = 512",
              "run.@prd098_validation : app.batchrun.exp_exit_step = 512"]
    config = destination / "run.txt"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    resolve(config, destination)
    actual = execute(destination, config, "t106_current_dump_GridMaze", "Debug")
    report = {"GridMaze": inspect(actual)}
    # Atari は同じ現行設定で完了した G の評価 Run を使い、余分な評価を流さない。
    result = json.loads((OUT / "evaluation/new_spl_G_s1/result.json").read_text(encoding="utf-8"))
    report["Atari"] = inspect(Path(result["run"]), atari=True)
    save(destination / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""旧checkpointの形状と保存形式をCPUの短い公開Runner経路で確認する。"""
import json
import os
from pathlib import Path
import re

from common import OUT, REPO, execute, resolve, save


def main():
    manifest = json.loads((OUT / "experiments/recipes.json").read_text(encoding="utf-8"))
    results = []
    os.environ["PATH"] = str(REPO / "apps/runner/bin/Release") + os.pathsep + os.environ["PATH"]
    # 本評価の入力を複製し、deviceと全評価の休止だけを上書きする。
    for recipe in manifest["recipes"]:
        if recipe["stage"] != 0 or recipe["arm"] == "fallback":
            continue
        destination = OUT / "checkpoint-preflight" / recipe["arm"]
        destination.mkdir(parents=True, exist_ok=False)
        record = {"arm": recipe["arm"]}
        try:
            selected = json.loads((Path(recipe["directory"]) / "resolved.json").read_text(encoding="utf-8"))
            slots = sorted({m[1] for key in selected if (m := re.match(r"run\.eval_schedule\.\[([^]]+)\]\.", key))})
            lines = Path(recipe["config"]).read_text(encoding="utf-8").splitlines()
            lines += ["run.@prd104 : agent.device = cpu", "run.@prd104 : run.eval_device = cpu"]
            lines += [f"run.@prd104 : run.eval_schedule.[{slot}].interval = 0" for slot in slots]
            config = destination / "run.txt"
            config.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            actual = resolve(config, destination)
            assert actual["agent.device"] == actual["run.eval_device"] == "cpu"
            assert actual["DefaultDQNAgent.learner.enabled"] == "false"
            assert all(actual[f"run.eval_schedule.[{slot}].interval"] == "0" for slot in slots)
            run = execute(destination, config, "t104_checkpoint_cpu_" + recipe["arm"],
                          binary=OUT / "experiments/AnetRLRunner.exe")
            record.update(passed=True, run=str(run))
        except Exception as error:
            record.update(passed=False, error=str(error))
        results.append(record)
        save(destination / "result.json", record)
    save(OUT / "checkpoint-preflight/results.json", results)
    if len(results) != 4 or not all(result["passed"] for result in results):
        raise SystemExit("Some checkpoint preflights failed; see checkpoint-preflight/results.json")


if __name__ == "__main__":
    main()

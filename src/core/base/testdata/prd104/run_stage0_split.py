"""段0と同じ1Run内のfull/single評価を、追加指定のsplit checkpointへ適用する。"""
import json
import os
from pathlib import Path
import shutil

from common import OUT, digest, execute, resolve, save
from run_experiments import episode_summary, metric_values, wait_idle


def main():
    # 元3本の結果・多数決とは別枠で保存し、段1へ進むdriverを使わない。
    destination = OUT / "experiments/stage0_split"
    destination.mkdir(exist_ok=False)
    source_case = OUT / "experiments/stage0_fallback"
    config = destination / "run.txt"
    shutil.copy2(source_case / "run.txt", config)
    source = json.loads((source_case / "source.json").read_text(encoding="utf-8"))
    checkpoint = Path(source["checkpoint"])
    source["checkpoint_sha256"] = digest(checkpoint)
    source["role"] = "Additional split comparison; excluded from the original three-checkpoint vote"
    save(destination / "source.json", source)
    actual = resolve(config, destination, binary=OUT / "current-debug-bin/anet-core-test.exe")
    assert actual["DefaultDQNAgent.learner.enabled"] == "false"
    assert actual["DefaultDQNAgent.learner.skip.feature_gradient"] == "split"
    assert actual["DefaultDQNAgent.actor.[greedy].skip.max_length_mode"] == "full"
    assert actual["DefaultDQNAgent.actor.[greedy_single].skip.max_length_mode"] == "single"
    assert actual["run.eval.[greedy].eval_episodes"] == actual["run.eval.[greedy_single].eval_episodes"] == "200"
    assert actual["app.save_agent_on_close"] == "false"
    # 同じ凍結ビルドを使い、既存Runの終了を待ってから追加の1本だけを実行する。
    recipes = json.loads((OUT / "experiments/recipes.json").read_text(encoding="utf-8"))
    binary = OUT / "experiments/AnetRLRunner.exe"
    assert digest(binary) == recipes["runner_sha256"]
    assert digest(OUT / "current-release-bin/AnetRLRunner.exe") == recipes["runner_sha256"]
    os.environ["PATH"] = str(OUT / "current-release-bin") + os.pathsep + os.environ["PATH"]
    wait_idle()
    run = execute(destination, config, "t104_stage0_split", binary=binary)
    # 元と同じ指標と採用trace件数で読み、主判定へ混ぜず追加結果にする。
    full = metric_values(run, "53_evalg/10_game_score_mean")
    single = metric_values(run, "53_evalg/60_single_score_mean")
    assert len(full) == len(single) == 1
    baseline = json.loads((OUT / "experiments/stage0_off/result.json").read_text(encoding="utf-8"))["full_score"]
    denominator = baseline - full[0]
    record = {"stage": 0, "arm": "split", "additional": True, "passed": True, "run": str(run),
              "checkpoint_sha256": source["checkpoint_sha256"], "full_score": full[0], "single_score": single[0],
              "full_episodes": episode_summary(run, "53_evalg/episode"),
              "single_episodes": episode_summary(run, "53_evalg/single_episode"),
              "g_denominator": denominator,
              "g": (single[0] - full[0]) / denominator if denominator > 0 else None}
    save(destination / "result.json", record)
    print("ADDITIONAL SPLIT COMPLETED", flush=True)


if __name__ == "__main__":
    main()

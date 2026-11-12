"""PRD123: 保存した実行体による決定的な指標・checkpoint の比較。"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[4]
OUT = REPO / ".scratch/prd123/baseline-20261111"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("before", "after", "compare", "load"))
    parser.add_argument("--label", default="after-r1")
    parser.add_argument("--case", nargs="+", choices=("GridMaze", "LunarLander", "Atari", "DropMerge", "collect", "feedback"),
                        default=["GridMaze", "LunarLander", "Atari", "DropMerge", "collect", "feedback"])
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("prd098_validation", REPO / "core/anet-core/testdata/prd098/run_validation.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.OUT, driver.RUN_PREFIX = OUT, "prd123"
    original_settings = driver.settings
    current_case = ""

    def settings(*values, **keywords):
        lines = original_settings(*values, **keywords)
        if current_case in ("collect", "feedback"):
            # 最後の検証用 profile を保ち、RND の選択をその直前へ足す。
            chain, _, validation = lines[-1].rpartition(" > ")
            lines[-1] = chain + " > run.@rnd > " + validation
        return lines

    driver.settings = settings
    if args.stage in ("after", "load"):
        folder = OUT / (args.label + "-bin")
        folder.mkdir(exist_ok=True)
        source = driver.RUNNER
        snapshot = folder / source.name
        if snapshot.exists():
            if snapshot.read_bytes() != source.read_bytes():
                raise RuntimeError("Binary changed; choose a fresh label")
        else:
            shutil.copy2(source, snapshot)
            # DLL は読み取り専用の依存としてリンクし、実行体だけ複製する。
            for dll in source.parent.glob("*.dll"):
                (folder / dll.name).hardlink_to(dll)
        driver.RUNNER = snapshot
    errors = []
    for case in args.case:
        current_case = case
        try:
            if args.stage == "compare":
                driver.compare(args.label, case)
                definitions = []
                for side in ("before", args.label):
                    execution = json.loads((OUT / f"{side}_{case}/execution.json").read_text(encoding="utf-8"))
                    definitions.append(json.loads((Path(execution["run"]) / "json/metrics.scalar.defs.json").read_text(encoding="utf-8-sig")))
                if definitions[0] != definitions[1]:
                    raise RuntimeError("Metric definitions differ: " + case)
                print("PASS scalar checksum and definitions", case, flush=True)
                continue
            env = "LunarLander" if case in ("collect", "feedback") else case
            overrides = ["DefaultDQNAgent.intrinsic.enabled=" + ("true" if case in ("collect", "feedback") else "false")]
            if case in ("collect", "feedback"):
                overrides += ["app.exp_exit_step=100000", "app.batchrun.exp_exit_step=100000", "app.save_agent_on_close=true",
                              "DefaultDQNAgent.intrinsic.rnd.warmup_mode=collect_errors"]
            if case == "feedback":
                # Atari の run.@rnd_taiga の 8 値を同じ比較条件へ明示する。
                overrides += ["DefaultDQNAgent.intrinsic.observation=state", "DefaultDQNAgent.intrinsic.learn_ratio=1",
                    "DefaultDQNAgent.intrinsic.learn_batch_size=1", "DefaultDQNAgent.intrinsic.beta=0.00005",
                    "DefaultDQNAgent.intrinsic.rnd.multiplier_mode=unclipped",
                    "DefaultDQNAgent.intrinsic.rnd.statistics_mode=variance_feedback",
                    "DefaultDQNAgent.intrinsic.rnd.warmup_steps=0", "DefaultDQNAgent.intrinsic.rnd.adam_eps=1e-5"]
            if args.stage == "load":
                if case != "collect":
                    continue
                record = json.loads((OUT / "before_collect/execution.json").read_text(encoding="utf-8"))
                checkpoint = Path(record["run"]) / "agent_close.anet"
                overrides += ["DefaultDQNAgent.auto_load_file=" + checkpoint.as_posix(),
                              "app.exp_exit_step=16", "app.batchrun.exp_exit_step=16"]
            side = "before" if args.stage == "before" else args.label
            run = driver.execute(side + ("_load_" if args.stage == "load" else "_") + case,
                                 env, before=args.stage == "before", overrides=overrides)
            print("PASS", case, run, flush=True)
        except Exception as error:
            errors.append(case + ": " + str(error))
            print("ERROR", errors[-1], flush=True)
    # 条件別の並列確認でも集計を上書きしない。
    driver.save(OUT / (args.stage + "_" + args.label + "_" + "-".join(args.case) + "_summary.json"), {"errors": errors})
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

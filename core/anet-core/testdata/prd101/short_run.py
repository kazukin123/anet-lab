"""PRD101: 実Runの探索停止・学習率phase境界を検証する。"""
import argparse
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / ".scratch/prd101"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("epsilon", "learning_rate"))
    parser.add_argument("--label")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("prd098", REPO / "core/anet-core/testdata/prd098/run_validation.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.OUT = OUT
    if args.kind == "epsilon":
        root = "DefaultDQNAgent.actor.[train].policy.epsilon"
        overrides = ["DefaultDQNAgent.actor.[train].policy.policy_type=EpsilonGreedy",
                     "DefaultDQNAgent.actor.[train].policy.spatial.enabled=false",
                     root + ".type=phased", root + ".phases=decay off",
                     root + ".phase.[decay].type=exponential", root + ".phase.[decay].start=0.4",
                     root + ".phase.[decay].end=0", root + ".phase.[decay].time_constant=128",
                     root + ".phase.[decay].steps=512", root + ".phase.[off].value=0",
                     root + ".phase.[off].steps=4096"]
        key, source, event = "epsilon", "$actor", "@train"
    else:
        root = "DefaultDQNAgent.learner.learning_rate"
        overrides = [root + ".type=phased", root + ".phases=high low",
                     root + ".phase.[high].value=0.001", root + ".phase.[high].steps=512",
                     root + ".phase.[low].value=0.0001", root + ".phase.[low].steps=4096"]
        key, source, event = "learning_rate", "$agent", "@learn"
    overrides += ["app.exp_exit_step=2048", "app.batchrun.exp_exit_step=2048",
                  f"metrics.scalar.[32_agent_base/prd101_switch]={source} {key} {event} $exp_step interval:8"]
    name = args.label or ("short_" + args.kind + "_final")
    environment = "Atari" if args.kind == "epsilon" else "GridMaze"
    run = runner.execute(name, environment, overrides=overrides)
    points = [(r["step"], r["value"]) for r in runner.rows(run)
              if r.get("type") == "scalar" and r.get("tag") == "32_agent_base/prd101_switch"]
    before = [value for step, value in points if 256 <= step < 512]
    after = [value for step, value in points if step >= 520]
    assert before and after, points
    if args.kind == "epsilon":
        assert all(value > 0 for value in before) and all(value == 0 for value in after), points
    else:
        assert all(abs(value - 0.001) < 1e-9 for value in before), points
        assert all(abs(value - 0.0001) < 1e-10 for value in after), points
    runner.save(OUT / name / "assertions.json",
                {"passed": True, "environment": environment, "switch_step": 512,
                 "run": str(run), "before": before, "after": after})
    print(args.kind, "PASS", len(points))

if __name__ == "__main__":
    main()

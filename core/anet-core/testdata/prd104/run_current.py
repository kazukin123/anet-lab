"""PRD104の短縮smokeを実行し、実際のscalar点を検査する。"""
import argparse
import math

from common import OUT, execute, load, resolve, save


def smoke(label, arm):
    destination = OUT / "smoke" / f"{label}_{arm}"
    destination.mkdir(parents=True, exist_ok=False)
    previous = load("prd098/run_validation.py", "prd104_smoke_input")
    previous.OUT = OUT
    lines = previous.settings("LunarLander", enabled=True)
    prefix = "run.@prd098_validation : "
    if arm == "variable":
        leaves = {"DefaultDQNAgent.actor.[train].skip.max_length_mode": "variable",
                  "DefaultDQNAgent.actor.[train].skip.max_length.type": "linear",
                  "DefaultDQNAgent.actor.[train].skip.max_length.start": 4,
                  "DefaultDQNAgent.actor.[train].skip.max_length.end": 1,
                  "DefaultDQNAgent.actor.[train].skip.max_length.steps": 4352,
                  "metrics.scalar.[50_agent_skip/07_max_length]": "$actor skip_max_length @learn $exp_step"}
    else:
        leaves = {"DefaultDQNAgent.actor.[train].skip.max_length_mode": "single",
                  "DefaultDQNAgent.actor.[eval].skip.max_length_mode": "single",
                  "DefaultDQNAgent.learner.skip.target_window": arm}
    lines += [prefix + f"{key} = {value}" for key, value in leaves.items()]
    config = destination / "run.txt"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    resolve(config, destination)
    run = execute(destination, config, "t104_smoke_" + destination.name, "Debug")
    audit(destination, run, arm)


def audit(destination, run, arm):
    from common import rows
    values = {}
    for point in rows(run):
        if point.get("type") == "scalar":
            values.setdefault(point["tag"], []).append(point["value"])
    def series(suffix):
        result = values.get("50_agent_skip/" + suffix, [])
        assert result and all(math.isfinite(value) for value in result), suffix
        return result
    if arm == "variable":
        limits = series("07_max_length")
        assert limits[0] == 3 and limits[-1] == 1, limits
        assert all(a >= b for a, b in zip(limits, limits[1:])), limits
        assert {1, 2, 3}.issubset(limits), limits
    else:
        assert all(value == 1 for value in series("01_decision_rate"))
        maximums = series("14_kmax_mean")
        assert all(value >= (3.5 if arm == "episode_end" else 1) for value in maximums)
        if arm == "action_change":
            assert any(value > 1 for value in maximums)
        series("11_loss")
    save(destination / "audit.json", {"passed": True, "run": str(run), "arm": arm,
                                     "metrics": {k: {"count": len(v), "first": v[0], "last": v[-1]}
                                                 for k, v in values.items() if k.startswith("50_agent_skip/")}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--arm", nargs="+", choices=("action_change", "episode_end", "variable"),
                        default=["action_change", "episode_end", "variable"])
    args = parser.parse_args()
    failures = []
    for arm in args.arm:
        try:
            smoke(args.label, arm)
        except Exception as error:
            failures.append({"arm": arm, "error": str(error)})
    save(OUT / "smoke" / f"{args.label}-summary.json", {"failures": failures})
    if failures:
        raise SystemExit(str(failures))


if __name__ == "__main__":
    main()

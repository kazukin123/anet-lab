"""Exercise the Atari timing tool's pure parts with synthetic inputs; never launch Runner."""
import importlib.util
import sys

sys.dont_write_bytecode = True
from pathlib import Path
import tempfile

repo = Path(__file__).resolve().parents[5]
script = repo / "src/core/base/testdata/prd116/run_performance_atari.py"
spec = importlib.util.spec_from_file_location("atari_timing", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
passed = []


def check_failure(label, action, terms):
    try:
        action()
    except RuntimeError as error:
        assert all(term in str(error) for term in terms), (label, str(error), terms)
    else:
        raise AssertionError(label + ": expected failure")
    passed.append(label)


# 区間は最速の群で目標秒数を測れる長さを lane 単位に切り上げる。
pilots = [{"arm": "off", "measurement_seconds": 2.0}, {"arm": "on", "measurement_seconds": 3.0},
          {"arm": "taiga", "measurement_seconds": 10.0}]
window = module.choose_window(pilots)
assert window % module.LANES == 0, window
assert window / module.PILOT_WINDOW * 2.0 >= module.FASTEST_TARGET_SECONDS, window
assert window == 81920, window
passed.append("window from fastest pilot")
assert module.choose_window([{"measurement_seconds": 100.0}]) == 2 * module.PILOT_WINDOW
passed.append("window lower bound")

# 3 群の中央値・比・巡ごとの比を合成記録から求める。
seconds = {"off": 2.0, "on": 3.0, "taiga": 6.0}
records = [{"name": f"r{index + 1:02d}_{arm}", "arm": arm, "profiled": False, "window_experiences": 4096,
            "binary_sha256": "fixed", "measurement_seconds": seconds[arm]} for index, arm in enumerate(module.ORDER)]
summary = module.summarize(records)
assert summary["medians_seconds"] == seconds, summary
assert summary["ratios"] == {"on_off": 1.5, "taiga_off": 3.0}, summary
assert len(summary["cycle_ratios"]) == 3 and all(item == {"on_off": 1.5, "taiga_off": 3.0}
                                                  for item in summary["cycle_ratios"]), summary
assert summary["experiences_per_second"]["off"] == 2048.0, summary
passed.append("summary medians and ratios")
check_failure("summary rejects failed run", lambda: module.summarize(records[:-1] + [{"name": "x", "arm": "off",
              "error": "boom"}]), ["invalid", "boom"])
check_failure("summary rejects order", lambda: module.summarize([records[1], records[0]] + records[2:]), ["order"])
check_failure("summary rejects mixed binaries", lambda: module.summarize(records[:-1] + [dict(records[-1],
              binary_sha256="other")]), ["one executable SHA256"])

# 群ごとのチェーンと測定用の設定キーを、Atari.txt の最後の run.$ の後ろへ重ねる。
with tempfile.TemporaryDirectory(prefix="prd116-atari-perf-check-") as temporary:
    out = Path(temporary)
    (out / "config").mkdir()
    (out / "config/_main.txt").write_text("", encoding="utf-8")
    (out / "config/Atari.txt").write_text("#run.$ = run.@old\nrun.$ = run.@a>run.@b  # note\n", encoding="utf-8")
    text = module.config(out, "r03_taiga", "taiga", 4096, ["eval", "greedy"])
    lines = text.splitlines()
    assert lines[-1] == "run.$ = run.@a>run.@b > run.@rnd > run.@rnd_taiga > run.@prd116_aperf", lines[-1]
    for expected in ("run.@prd116_aperf : run.train.num_envs = 64", "run.@prd116_aperf : agent.device = cuda",
                     "run.@prd116_aperf : DefaultDQNAgent.intrinsic.enabled = true",
                     "run.@prd116_aperf : run.eval_schedule.[greedy].interval = 0",
                     "run.@prd116_aperf : app.run_name = run_{t}_prd116_ap_r03_taiga",
                     f"run.@prd116_aperf : app.exp_exit_step = {module.WARM + 4096 + 2 * module.LANES}"):
        assert expected in lines, (expected, lines)
    off = module.config(out, "r01_off", "off", 4096, []).splitlines()
    assert off[-1] == "run.$ = run.@a>run.@b > run.@prd116_aperf", off[-1]
    assert "run.@prd116_aperf : DefaultDQNAgent.intrinsic.enabled = false" in off
    passed.append("config chains per arm")

# 群の差は RND・群 56・Run 名だけを許し、それ以外は報告する。
left = {"app.run_name": "a", "DefaultDQNAgent.intrinsic.beta": "0.1", "metrics.scalar.[56_agent_in/01_bonus_mean]": "x",
        "run.train.num_envs": "64"}
right = {"app.run_name": "b", "DefaultDQNAgent.intrinsic.beta": "5e-05", "run.train.num_envs": "128"}
assert module.unexpected_differences(left, right) == {"run.train.num_envs": ["64", "128"]}
passed.append("config difference filter")
assert module.eval_tags_of({"run.eval.[eval].actor": "x", "run.eval_schedule.[greedy].interval": "1",
                            "run.train.num_envs": "64"}) == ["eval", "greedy"]
passed.append("eval tags")

print(f"{len(passed)} behaviors passed: " + ", ".join(passed))

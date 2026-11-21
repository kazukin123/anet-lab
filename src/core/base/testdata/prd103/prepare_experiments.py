"""PRD103 §7 の長時間実験の入力を生成・解決する。学習Runは開始しない。"""
import re

from run_current import resolve
from run_validation import ROOT, REPO, save

OUT = REPO / ".scratch/prd103/experiments"
CONFIG = REPO / "apps/runner/config"
SAMPLE_CHAIN = " > ".join("run.@" + name for name in (
    "v5_iqn_impala_x2", "a5", "a5_apex", "va_base", "hard500", "rr4", "munch", "rfit", "a5_metrics",
    "btrnet", "btrsn12", "cap2m", "eval2ch", "nz_learning", "pixel_area", "skip", "batch"))


def sample_efficiency():
    # 段1のBTR-aligned + default初期値 + online Munchausen + phased epsilonと揃える。
    leaves = {"run.train.num_envs": 64, "A2.learner.replay_capacity": 1048576,
        "A2.actor.[train].sync_interval": 1, "A2.actor.[train].policy.spatial.enabled": "false",
        "A2.actor.[train].policy.epsilon.type": "phased", "A2.actor.[train].policy.epsilon.phases": "decay off",
        "A2.actor.[train].policy.epsilon.phase.[decay].type": "exponential",
        "A2.actor.[train].policy.epsilon.phase.[decay].start": 1,
        "A2.actor.[train].policy.epsilon.phase.[decay].end": 0.01,
        "A2.actor.[train].policy.epsilon.phase.[decay].time_constant": 2000000,
        "A2.actor.[train].policy.epsilon.phase.[decay].steps": 25000000,
        "A2.actor.[train].policy.epsilon.phase.[off].type": "constant",
        "A2.actor.[train].policy.epsilon.phase.[off].value": 0,
        "A2.actor.[train].policy.epsilon.phase.[off].steps": 1,
        "net.block.[AtariHeadFC512].init.mode": "default", "net.block.[AtariHeadFC512].noisy.sigma0": 0.5,
        "A2.head.init.mode": "default", "A2.head.projection.noisy.sigma0": 0.5,
        "A2.learner.munchausen.log_policy_mode": "online"}
    for block in ("AtariImpalaConv32", "AtariImpalaConv64", "AtariIQNTauProj2304"):
        leaves[f"net.block.[{block}].init.mode"] = "default"
    for block in ("AtariImpalaRes32SN", "AtariImpalaRes64SN"):
        for init in ("init1", "init2"):
            leaves[f"net.block.[{block}].{init}.mode"] = "default"
    return leaves


def prepare(stage, env, arm, seed, budget, changes, condition="always"):
    name = f"stage{stage}_{env}_{arm}_s{seed}"
    destination = OUT / name
    destination.mkdir(parents=True, exist_ok=False)
    source = (CONFIG / (env + ".txt")).read_text(encoding="utf-8-sig")
    chain = re.findall(r"(?m)^run\.\$\s*=\s*([^#\n]+)", source)[-1].strip()
    if "run.@skip" not in chain:
        chain += " > run.@skip"
    if stage in (1, 3):
        chain = SAMPLE_CHAIN
    if arm in ("conv_shared", "conv_separate"):
        chain += " > run.@skip_" + arm
    leaves = {"app.$": "app.batchrun > P1", "app.show_error_dialog": "false",
        "app.train_auto_start": "true", "app.eval_panel.auto_start": "false",
        "app.save_agent_on_close": "false", "app.runs_dir": (OUT / "runs").as_posix(),
        "app.run_name": "run_{t}_t103_" + name, "run.seed": seed,
        "app.exp_exit_step": budget, "app.batchrun.exp_exit_step": budget}
    if env == "Atari":
        leaves["E1.game"] = "qbert"
    # envのskip用選択が持つMunchausen/replay-fit等を保持してNoisy診断だけを足す。
    metrics = re.findall(r"(?m)^run\.@skip\s*:\s*metrics\.scalar\.\$\s*=\s*([^#\n]+)", source)[-1].strip()
    leaves["metrics.scalar.$"] = metrics + " > metrics.scalar.@noisy"
    if stage in (1, 3):
        leaves.update(sample_efficiency())
    leaves.update(changes)
    config = destination / "run.txt"
    lines = [f"$include <{CONFIG.as_posix()}/_main.txt>", f"$include <{CONFIG.as_posix()}/{env}.txt>",
        f"$include <{ROOT.as_posix()}/arms.txt>"]
    lines += [f"run.@prd103_experiment : {key} = {value}" for key, value in leaves.items()]
    lines += ["run.$ = " + chain + " > run.@prd103_experiment"]
    # 診断追加前のRun選択を公開resolverで解決し、元の全scalar定義を保存する。
    reference_lines = [line for line in lines if not line.startswith("run.@prd103_experiment : metrics.scalar.$")]
    config.write_text("\n".join(reference_lines) + "\n", encoding="utf-8", newline="\n")
    reference = resolve(config, destination, "reference")
    config.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    resolved = resolve(config, destination, "resolved")
    for key, value in reference.items():
        if key.startswith("metrics.scalar.["):
            assert resolved.get(key) == value, (name, key, value, resolved.get(key))
    assert resolved["DefaultDQNAgent.skip.enabled"] == "true"
    assert resolved["app.batchrun.exp_exit_step"] == str(budget)
    assert resolved["run.seed"] == str(seed)
    assert resolved["run.eval.[greedy].actor"] == "greedy"
    if arm == "epsilon":
        assert resolved["DefaultDQNAgent.actor.[train].skip.policy.spatial.epsilon.first"] == "0.3"
        assert resolved["DefaultDQNAgent.actor.[train].skip.policy.spatial.epsilon.last"] == "0"
    return {"name": name, "condition": condition, "config": str(config),
        "command": f'& "{REPO / "apps/runner/bin/Release/AnetRLRunner.exe"}" --config "{config}"'}


def main():
    results, errors = [], []
    prefix = "DefaultDQNAgent.learner.skip."
    cases = [(1, "Atari", arm, 1, 50000000, changes, "段1の既存Runがあれば再実行不要") for arm, changes in (
        ("stop", {prefix + "feature_gradient": "stop"}),
        ("weight03", {prefix + "loss_weight": 0.3}),
        ("clip30", {"DefaultDQNAgent.learner.grad_clip_tau": 30}))]
    for seed in (1, 2, 3):
        cases += [(2, "LunarLander", arm, seed, 2000000, changes, "always") for arm, changes in (
            ("merge", {prefix + "feature_gradient": "merge"}),
            ("split", {prefix + "feature_gradient": "split"}),
            ("kmax", {prefix + "sample_weight": "kmax"}),
            ("epsilon", {"A2.actor.[train].skip.policy.policy_type": "EpsilonGreedy",
                "A2.actor.[train].skip.policy.spatial.epsilon.first": 0.3,
                "A2.actor.[train].skip.policy.spatial.epsilon.last": 0}))]
    cases += [(3, "Atari", "split", 1, 50000000, {prefix + "feature_gradient": "split"}, "always"),
        (3, "Atari", "split_kmax", 1, 50000000,
            {prefix + "feature_gradient": "split", prefix + "sample_weight": "kmax"}, "段2でkmaxが基準以上の場合だけ"),
        (3, "Atari", "merge", 2, 50000000, {prefix + "feature_gradient": "merge"}, "段1でseedのばらつきが候補の場合だけ")]
    for seed in (1, 2):
        cases += [(4, "Atari", arm, seed, 10000000, {}, "always") for arm in ("merge", "conv_separate", "conv_shared")]
    for case in cases:
        try:
            results.append(prepare(*case))
        except Exception as error:
            errors.append({"case": case, "error": str(error)})
    save(OUT / "recipes.json", {"recipes": results, "errors": errors, "executed_learning_runs": 0})
    print(f"Prepared {len(results)} recipes; errors={len(errors)}; learning Runs not started")
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()

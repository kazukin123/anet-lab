"""BTR 公開実装の出力を anet-lab の Run 形式 (metrics.jsonl) へ変換する。

BTR (arXiv:2411.03820) の実行結果を MetricsViewer と inspect_run.py で
anet-lab の Run と横並びに見るための変換器。再実行すると全て再生成する。

取り込む入力は 3 種類。

  Experiment.npy   train エピソードの (score, env_step)
  Evaluation.npy   eval 点ごとの全エピソードスコア (点数, エピソード数)
  *_<N>M.model     10M frames ごとの online network state_dict

軸の対応。BTR の env_steps は遷移数なので anet-lab の exp_step と同一単位。
checkpoint 由来の指標は anet-lab 側が learn_step 軸なので、BTR の更新則
(1 更新 / 64 遷移、warmup 200,000 遷移) で換算して合わせる。

    learn_step = (exp_step - 200,000) / 64

BTR の eval はオンライン網を読むので eval2 へ対応させる (eval1 = target 網に
相当するものが BTR に無い)。EMA は runner と同じゼロ初期化 + 逐次積 debias
(PRD 045 案A) で、alpha は train 0.001 / eval 0.1。

probe 指標は全 checkpoint へ同一の観測バッチを食わせて測る。観測は最新
checkpoint の方策で 1 度だけ収集し、out 配下へ残して再利用する。

使い方 (WSL 側の BTR venv から実行する):

    python btr_to_metrics.py --btr-run ~/BTR/BTR_Breakout200M_noisy0 \
        --out /mnt/c/.../runs/run_20260911-095808_btr_noisyoff_wsl \
        --btr-src ~/BTR --start 2026-09-11T09:58:08
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

TRAIN_EMA_ALPHA = 0.001
EVAL_EMA_ALPHA = 0.1
THRESHOLDS = [(432, "40", "41"), (600, "42", "43")]

FRAMES_PER_EVAL_POINT = 1_000_000
TRANSITIONS_PER_FRAME = 0.25
WARMUP_TRANSITIONS = 200_000
TRANSITIONS_PER_UPDATE = 64

SRANK_DELTA = 0.01
DORMANT_TAU = 0.025


def debiased_ema(values, alpha):
    """runner の EmaFilter と同じ式 (ゼロ初期化 + 逐次積で debias)。"""
    out, m = [], 0.0
    for i, value in enumerate(values, start=1):
        m += alpha * (value - m)
        out.append(m / (1.0 - (1.0 - alpha) ** i))
    return out


def to_learn_step(exp_step):
    return max(0, int((exp_step - WARMUP_TRANSITIONS) // TRANSITIONS_PER_UPDATE))


class Emitter:
    """defs と scalar 行を溜める。"""

    def __init__(self):
        self.defs = {}
        self.rows = []

    def add(self, tag, steps, values, *, step_axis, scope, event, source_key,
            alpha=None, target="env", eval_name=None, eval_episodes=None, num_envs=None):
        self.defs[tag] = {
            "clip": None, "ema_alpha": alpha,
            "eval_episodes": eval_episodes, "eval_name": eval_name,
            "event": event, "interval": 1, "num_envs": num_envs,
            "runner": "train", "scope": scope, "source_key": source_key,
            "step_axis": step_axis, "target": target,
        }
        series = debiased_ema(values, alpha) if alpha is not None else values
        self.rows.extend((int(s), tag, float(v)) for s, v in zip(steps, series))

    def add_pair(self, group, raw_num, ema_num, name, source_key, steps, values, alpha, **meta):
        """raw と EMA を同じ素材から 1 対で出す。"""
        self.add(f"{group}/{raw_num}_{name}", steps, values, source_key=source_key, alpha=None, **meta)
        self.add(f"{group}/{ema_num}_{name}_ema", steps, values, source_key=source_key, alpha=alpha, **meta)


def load_one(pattern, directory):
    hits = glob.glob(os.path.join(directory, pattern))
    if not hits:
        raise SystemExit(f"not found: {pattern} in {directory}")
    return np.load(hits[0], allow_pickle=True)


def emit_episodes(emitter, btr_run):
    """Experiment.npy: train エピソードのスコア。"""
    exp = load_one("*Experiment.npy", btr_run)
    order = np.argsort(exp[:, 1], kind="stable")
    steps, scores = exp[order, 1], exp[order, 0]
    meta = dict(step_axis="exp_step", scope="train", event="train")
    emitter.add_pair("42_env", "10", "11", "game_score_mean", "mean.game_score",
                     steps, scores, TRAIN_EMA_ALPHA, **meta)
    for threshold, raw_num, ema_num in THRESHOLDS:
        emitter.add_pair("42_env", raw_num, ema_num, f"game_score_ge{threshold}",
                         f"mean.game_score.ge.[{threshold}]", steps,
                         (scores >= threshold).astype(float), TRAIN_EMA_ALPHA, **meta)
    return len(exp)


def emit_evals(emitter, btr_run):
    """Evaluation.npy: eval 点ごとの平均・最大・閾値超え率。"""
    ev = load_one("*Evaluation.npy", btr_run)
    rows = [i for i in range(ev.shape[0]) if ev[i].any()]
    steps = [int((i + 1) * FRAMES_PER_EVAL_POINT * TRANSITIONS_PER_FRAME) for i in rows]
    meta = dict(step_axis="exp_step", scope="eval", event="session_end",
                eval_name="eval2", eval_episodes=int(ev.shape[1]), num_envs=10)
    emitter.add_pair("52_eval2", "10", "11", "game_score_mean", "mean.game_score",
                     steps, [float(ev[i].mean()) for i in rows], EVAL_EMA_ALPHA, **meta)
    emitter.add("52_eval2/16_game_score_max", steps, [float(ev[i].max()) for i in rows],
                source_key="max.game_score", **meta)
    for threshold, raw_num, ema_num in THRESHOLDS:
        emitter.add_pair("52_eval2", raw_num, ema_num, f"game_score_ge{threshold}",
                         f"mean.game_score.ge.[{threshold}]", steps,
                         [float(np.mean(ev[i] >= threshold)) for i in rows], EVAL_EMA_ALPHA, **meta)
    return len(rows)


def checkpoint_steps(btr_run):
    """<name>_<N>M.model の N は 1M frames 単位。exp_step へ直して昇順で返す。"""
    found = []
    for path in glob.glob(os.path.join(btr_run, "*_*M.model")):
        match = re.search(r"_(\d+)M\.model$", path)
        if match:
            frames = int(match.group(1)) * FRAMES_PER_EVAL_POINT
            found.append((int(frames * TRANSITIONS_PER_FRAME), path))
    return sorted(found)


def split_weight_norms(state_dict):
    """feature / readout の依存閉包で重みノルムを分ける (PRD 063 と同じ 2 群)。"""
    import torch

    feature_sq, readout_sq = 0.0, 0.0
    for key, value in state_dict.items():
        if not key.endswith(("weight", "original")) or value.dim() < 2:
            continue
        squared = float(torch.linalg.vector_norm(value.float())) ** 2
        if key.startswith("dueling") or "cos_embedding" in key:
            readout_sq += squared
        else:
            feature_sq += squared
    return feature_sq ** 0.5, readout_sq ** 0.5


def plasticity_from_features(features):
    """nn_impl.cpp の PlasticityMetric と同じ式。features は [samples, dim]。"""
    import torch

    singular = torch.linalg.svdvals(features)
    total = float(singular.sum())
    srank, srank_ratio = 0.0, 0.0
    if total > 0.0:
        cumulative = singular.cumsum(0)
        reached = cumulative >= total * (1.0 - SRANK_DELTA)
        srank = float(int(reached.to(torch.int64).argmax()) + 1)
        srank_ratio = srank / float(min(features.size(0), features.size(1)))
    unit = features.abs().mean(0)
    mean_unit = float(unit.mean())
    if mean_unit > 0.0:
        normalized = unit / mean_unit
        dormant = float((normalized <= DORMANT_TAU).float().mean())
        dead = float((normalized <= 0.0).float().mean())
    else:
        dormant, dead = 1.0, 1.0
    feature_norm = float(torch.linalg.vector_norm(features, 2, dim=1).mean())
    return dormant, dead, feature_norm, srank, srank_ratio


def build_network(args, device):
    sys.path.insert(0, os.path.expanduser(args.btr_src))
    from Agent import create_network

    return create_network(
        impala=True, iqn=True, input_dims=[args.framestack, 84, 84], n_actions=args.n_actions,
        spectral_norm=bool(args.spectral), device=device, noisy=bool(args.noisy),
        maxpool=bool(args.maxpool), model_size=args.model_size, maxpool_size=args.maxpool_size,
        linear_size=args.linear_size, num_tau=args.num_tau, dueling=True, ncos=args.ncos,
        non_factorised=False, arch="impala", layer_norm=False, activation="relu").to(device)


def collect_probe(args, network, device, cache_path):
    """全 checkpoint へ共通で食わせる観測バッチ。1 度収集して残す。"""
    import torch

    # BTR の forward が内部で float()/255 するので、観測は 0-255 の float のまま渡す。
    if os.path.exists(cache_path):
        return torch.from_numpy(np.load(cache_path)).float().to(device)

    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    sys.path.insert(0, os.path.expanduser(args.btr_src))
    from AtariPreprocessingCustom import AtariPreprocessingCustom

    lanes = 16
    env = gym.vector.AsyncVectorEnv([lambda: gym.wrappers.FrameStack(
        AtariPreprocessingCustom(
            gym.make("ALE/" + args.game + "-v5", frameskip=1, repeat_action_probability=0.25),
            life_information=False), args.framestack, lz4_compress=False)
        for _ in range(lanes)], context="fork")
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    collected = []
    # 方策の状態分布へ寄せたいので greedy で回し、eval と同じ eps 0.01 だけ混ぜる。
    while len(collected) * lanes < args.probe_batch:
        state = torch.tensor(np.asarray(obs), dtype=torch.float32, device=device)
        with torch.no_grad():
            action = torch.argmax(network.qvals(state, advantages_only=True), dim=1).cpu().numpy()
        explore = rng.random(lanes) < 0.01
        action[explore] = rng.integers(0, args.n_actions, size=int(explore.sum()))
        obs, _, _, _, _ = env.step(action)
        collected.append(np.asarray(obs))
    env.close()
    batch = np.concatenate(collected, axis=0)[:args.probe_batch]
    np.save(cache_path, batch)
    return torch.from_numpy(batch).float().to(device)


def emit_checkpoints(emitter, args, out_dir):
    """checkpoint 由来の重み空間 / probe 指標。learn_step 軸で出す。"""
    import torch

    entries = checkpoint_steps(args.btr_run)
    if not entries:
        return 0
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    network = build_network(args, device)
    network.eval()

    # probe は最新 checkpoint の方策で集めて、全点へ同じものを使う。
    network.load_state_dict(torch.load(entries[-1][1], map_location=device, weights_only=True))
    probe = collect_probe(args, network, device, os.path.join(out_dir, "probe_obs.npy"))

    steps, series = [], {k: [] for k in
                         ["wn_feature", "wn_readout", "dormant", "dead", "feature_norm",
                          "srank", "srank_ratio", "q_max", "q_gap"]}
    for exp_step, path in entries:
        state_dict = torch.load(path, map_location=device, weights_only=True)
        feature_norm_w, readout_norm_w = split_weight_norms(state_dict)
        network.load_state_dict(state_dict)
        with torch.no_grad():
            hidden = network.conv(probe / 255.0)
            if getattr(network, "maxpool", False):
                hidden = network.pool(hidden)
            features = hidden.view(probe.size(0), -1).float()
            q_values = network.qvals(probe, advantages_only=False).float()
        dormant, dead, feature_norm, srank, srank_ratio = plasticity_from_features(features)
        top2 = q_values.topk(2, dim=1).values
        steps.append(to_learn_step(exp_step))
        for key, value in [
                ("wn_feature", feature_norm_w), ("wn_readout", readout_norm_w),
                ("dormant", dormant), ("dead", dead), ("feature_norm", feature_norm),
                ("srank", srank), ("srank_ratio", srank_ratio),
                ("q_max", float(q_values.max(dim=1).values.mean())),
                ("q_gap", float((top2[:, 0] - top2[:, 1]).mean()))]:
            series[key].append(value)

    # 34_agent_plasticity は learn_step 軸、37_agent_qtd は exp_step 軸 (anet 側の定義に合わせる)。
    exp_steps = [step for step, _ in entries]
    plasticity_meta = dict(step_axis="learn_step", scope="train", event="learn", target="agent")
    qtd_meta = dict(step_axis="exp_step", scope="train", event="learn", target="agent")
    for tag, key, source_key in [
            ("34_agent_plasticity/61_weight_norm_feature", "wn_feature", "plasticity_weight_norm_feature"),
            ("34_agent_plasticity/62_weight_norm_readout", "wn_readout", "plasticity_weight_norm_readout"),
            ("34_agent_plasticity/41_probe_dormant_ratio", "dormant", "plasticity_probe_dormant_ratio"),
            ("34_agent_plasticity/42_probe_dead_ratio", "dead", "plasticity_probe_dead_ratio"),
            ("34_agent_plasticity/43_probe_feature_norm", "feature_norm", "plasticity_probe_feature_norm"),
            ("34_agent_plasticity/44_probe_srank", "srank", "plasticity_probe_srank"),
            ("34_agent_plasticity/45_probe_srank_ratio", "srank_ratio", "plasticity_probe_srank_ratio")]:
        emitter.add(tag, steps, series[key], source_key=source_key, **plasticity_meta)
    for tag, key, source_key in [
            ("37_agent_qtd/11_q_max_mean", "q_max", "q_max_mean"),
            ("37_agent_qtd/17_q_gap", "q_gap", "q_gap")]:
        emitter.add(tag, exp_steps, series[key], source_key=source_key, **qtd_meta)
    return len(entries)


def write_run(out_dir, emitter, start_iso):
    os.makedirs(os.path.join(out_dir, "json"), exist_ok=True)
    payload = {"data": emitter.defs, "tag": "metrics.scalar.defs", "type": "json"}
    with open(os.path.join(out_dir, "json", "metrics.scalar.defs.json"), "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=1)

    cache = os.path.join(out_dir, "metrics_cache.db")
    note = "no cache"
    if os.path.exists(cache):
        try:
            os.remove(cache)
            note = "cache removed"
        except OSError:
            note = "CACHE LOCKED - close the run in MetricsViewer, then delete metrics_cache.db"

    emitter.rows.sort(key=lambda row: row[0])
    with open(os.path.join(out_dir, "metrics.jsonl"), "w", encoding="utf-8") as fp:
        fp.write(json.dumps({"event": "start", "timestamp": start_iso, "type": "meta"}) + "\n")
        fp.write(json.dumps(payload) + "\n")
        for step, tag, value in emitter.rows:
            fp.write(json.dumps({"step": step, "tag": tag, "type": "scalar", "value": value}) + "\n")
    return note


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--btr-run", required=True, help="BTR の出力ディレクトリ")
    parser.add_argument("--out", required=True, help="生成先の Run ディレクトリ")
    parser.add_argument("--btr-src", default="~/BTR", help="BTR のソース (networks.py がある場所)")
    parser.add_argument("--start", default="1970-01-01T00:00:00", help="Run 開始時刻 (ISO8601)")
    parser.add_argument("--game", default="Breakout")
    parser.add_argument("--n-actions", type=int, default=4)
    parser.add_argument("--probe-batch", type=int, default=512, help="anet 側 plasticity.probe.batch_size と揃える")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-checkpoints", action="store_true", help="重み由来の指標を出さない")
    # BTR 側の構成。既定は BTR の CLI 既定値に一致する。
    parser.add_argument("--framestack", type=int, default=4)
    parser.add_argument("--model-size", type=float, default=2)
    parser.add_argument("--linear-size", type=int, default=512)
    parser.add_argument("--maxpool", type=int, default=1)
    parser.add_argument("--maxpool-size", type=int, default=6)
    parser.add_argument("--num-tau", type=int, default=8)
    parser.add_argument("--ncos", type=int, default=64)
    parser.add_argument("--spectral", type=int, default=1)
    parser.add_argument("--noisy", type=int, default=0)
    args = parser.parse_args()

    args.btr_run = os.path.expanduser(args.btr_run)
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    emitter = Emitter()
    episodes = emit_episodes(emitter, args.btr_run)
    eval_points = emit_evals(emitter, args.btr_run)
    checkpoints = 0 if args.skip_checkpoints else emit_checkpoints(emitter, args, out_dir)
    note = write_run(out_dir, emitter, args.start)

    print(f"episodes={episodes} eval_points={eval_points} checkpoints={checkpoints}")
    print(f"tags={len(emitter.defs)} rows={len(emitter.rows)}")
    print(note)


if __name__ == "__main__":
    main()

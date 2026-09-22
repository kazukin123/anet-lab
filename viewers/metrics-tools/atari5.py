#!/usr/bin/env python3
"""Atari-5 集計 CLI。5 本の Run を 1 つのベンチマーク結果として読む。

Atari-5 (Aitchison et al., ICML 2023) は ALE 57 ゲームの median human normalized
score を 5 ゲームだけで推定する部分集合である。対象は battle_zone / double_dunk /
name_this_game / phoenix / qbert の 5 本で、集約は単純中央値ではなく log 空間の
重み付き和で行う。Appendix E の係数を A5_WEIGHTS に置いた。

    s_log = sum(w_i * log10(1 + hns_i))      atari5 = 10 ** s_log - 1

このツールは Run の `*/12_hns57_mean` をそのまま読むので、正規化定数を二重に
持たない。参照エージェント (REFERENCES) の生スコアを正規化するときだけ HNS57 を
使い、Run 側の値と突き合わせて食い違えば警告する。

inspect_run.py と同じ Run 解決・cache 経路・出力エンベロープを使う。export を除いて
read-only で、実行中の Run へ当てても artifact を変更しない。--format の既定だけ
inspect_run.py と違って md にしてある。出力が固定 5 行の報告表で、人間がそのまま読む
前提のため。機械可読が要るときは --format json を渡す。

    atari5.py score   RUN...   ゲーム別スコアと Atari-5 集約
    atari5.py curve   RUN...   予算・実時間に対する Atari-5 の推移
    atari5.py compare RUN...   公表エージェントとの突き合わせ
    atari5.py refs             参照テーブルの一覧と出典
    atari5.py export  RUN...   MetricsViewer で開ける疑似 Run を書き出す（唯一の書き込み）

Run は名前でも path でも渡せる。ゲームの対応付けは Run 名ではなく実効 config の
AtariEnv.game で行うので、命名規約に依存しない。
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from inspect_run import (
    EXIT_OK,
    EXIT_USAGE,
    EXIT_RUNTIME,
    RuntimeFailure,
    SourceError,
    UsageError,
    WORKSPACES_ROOT,
    _md_header,
    _md_warnings,
    _resolve_endpoint,
    load_definitions_cheap,
    load_metric_series,
    open_run,
    parse_range,
    render_json,
    resolve_run,
    run_envelope,
    run_node,
    write_output,
)


GAMES = ("battle_zone", "double_dunk", "name_this_game", "phoenix", "qbert")

# Atari-5 回帰の係数。Aitchison et al. 2023, Appendix E。
# 切片は 0（random 方策が 0 点になるよう論文側で無効化している）。重み和は 1 にならない。
A5_WEIGHTS = {
    "battle_zone": 0.3820,
    "double_dunk": 0.0679,
    "name_this_game": 0.3108,
    "phoenix": 0.1241,
    "qbert": 0.0805,
}

# 57 ゲーム表の (random, human)。core/envs/atari1/src/AtariEnv.cpp の 57 ゲーム表と
# 同じ値でなければならない。参照エージェントの生スコアを Run と同じ尺度へ載せるためだけに使う。
# Run 側の hns57 と突き合わせて乖離を検出するので、片方が動けば警告で気づける。
# 注: Atari-5 の公開データセットは double_dunk の random を -18.55 とする。ここは Run 側へ
# 揃えてあるため、参照値の double_dunk は論文表記より約 2.3% 低く出る。
HNS57 = {
    "battle_zone": (2360.0, 37187.5),
    "double_dunk": (-18.6, -16.4),
    "name_this_game": (2292.3, 8049.0),
    "phoenix": (761.4, 7242.6),
    "qbert": (163.9, 13455.0),
}

# 評価チャネル。prefix は metric tag の前半で、config の eval 定義名とは別物。
CHANNELS = {
    "train": ("42_env", "raw train episodes (actor on the epsilon schedule)"),
    "eval1": ("51_eval1", "eval on the target network"),
    "eval2": ("52_eval2", "eval on the online network"),
    "evalg": ("53_evalg", "eval on the greedy distribution (epsilon = 0)"),
}
DEFAULT_CHANNELS = ("eval2", "evalg")

SCORE_SUFFIX = "10_game_score_mean"
HNS_SUFFIX = "12_hns57_mean"
LEN_SUFFIX = "20_game_len_mean"

PERF_ELAPSE_TAG = "90_perf/90_elapse_hour"
PERF_RATE_TAG = "90_perf/12_exp_step_per_sec"

CONFIG_GAME = "AtariEnv.game"
CONFIG_KEYS = {
    "game": CONFIG_GAME,
    "replay_ratio": "DefaultDQNAgent.learner.replay_ratio",
    "replay_capacity": "DefaultDQNAgent.learner.replay_capacity",
    "num_envs": "run.train.num_envs",
    "replay_batch_size": "DefaultDQNAgent.learner.replay_batch_size",
    "frame_skip": "AtariEnv.frame_skip",
}

# 公表エージェントの 5 ゲーム生スコア。
# source=atari5-dataset は Atari-5 論文の公開データセット (github.com/maitchison/Atari-5,
# dataset.csv) の Score 列。source=btr-paper は arXiv:2411.03820 Table A2。
# 追加するときは生スコアで書く。正規化はこのツールが HNS57 で行う。
REFERENCES = {
    "btr": {
        "label": "BTR",
        "year": 2024,
        "budget": "200M frames",
        "source": "btr-paper",
        "note": "best eval during training, 100 episodes x 3 seeds, online network; sticky actions on; replay ratio 4",
        "scores": {
            "battle_zone": 168340.0,
            "double_dunk": 23.0,
            "name_this_game": 27917.0,
            "phoenix": 427481.0,
            "qbert": 42927.0,
        },
    },
    "dqn": {
        "label": "DQN (noop)",
        "year": 2015,
        "budget": "200M frames",
        "source": "atari5-dataset",
        "note": "no sticky actions",
        "scores": {
            "battle_zone": 29900.0,
            "double_dunk": -6.6,
            "name_this_game": 8207.8,
            "phoenix": 8485.2,
            "qbert": 13117.3,
        },
    },
    "c51": {
        "label": "C51",
        "year": 2017,
        "budget": "200M frames",
        "source": "atari5-dataset",
        "note": "no sticky actions",
        "scores": {
            "battle_zone": 28742.0,
            "double_dunk": 2.5,
            "name_this_game": 12542.0,
            "phoenix": 17490.0,
            "qbert": 23784.0,
        },
    },
    "qrdqn": {
        "label": "QR-DQN-1",
        "year": 2017,
        "budget": "200M frames",
        "source": "atari5-dataset",
        "note": "no sticky actions; qbert 572,510 sits in the game-bug exploit range",
        "scores": {
            "battle_zone": 39268.0,
            "double_dunk": 21.9,
            "name_this_game": 21890.0,
            "phoenix": 16585.0,
            "qbert": 572510.0,
        },
    },
    "rainbow": {
        "label": "Rainbow (noop)",
        "year": 2018,
        "budget": "200M frames",
        "source": "atari5-dataset",
        "note": "no sticky actions",
        "scores": {
            "battle_zone": 62010.0,
            "double_dunk": -0.3,
            "name_this_game": 13136.0,
            "phoenix": 108528.6,
            "qbert": 33817.5,
        },
    },
    "iqn": {
        "label": "IQN",
        "year": 2018,
        "budget": "200M frames",
        "source": "atari5-dataset",
        "note": "no sticky actions; the direct ancestor of this codebase",
        "scores": {
            "battle_zone": 42244.0,
            "double_dunk": 5.6,
            "name_this_game": 22682.0,
            "phoenix": 56599.0,
            "qbert": 25750.0,
        },
    },
    "impala": {
        "label": "IMPALA (deep)",
        "year": 2018,
        "budget": "200M frames",
        "source": "atari5-dataset",
        "note": "qbert 351,200 sits in the game-bug exploit range",
        "scores": {
            "battle_zone": 20885.0,
            "double_dunk": -0.33,
            "name_this_game": 21537.2,
            "phoenix": 210996.45,
            "qbert": 351200.12,
        },
    },
    "laser": {
        "label": "LASER Shared (200M)",
        "year": 2019,
        "budget": "200M frames",
        "source": "atari5-dataset",
        "note": "near the top at the same budget",
        "scores": {
            "battle_zone": 131880.0,
            "double_dunk": 23.5,
            "name_this_game": 27910.7,
            "phoenix": 628711.6,
            "qbert": 24600.8,
        },
    },
    "gdi": {
        "label": "GDI-H3 (200M)",
        "year": 2021,
        "budget": "200M frames",
        "source": "atari5-dataset",
        "note": "the top at the same budget",
        "scores": {
            "battle_zone": 824360.0,
            "double_dunk": 24.0,
            "name_this_game": 36296.0,
            "phoenix": 959580.0,
            "qbert": 28657.0,
        },
    },
    "dreamerv2": {
        "label": "DreamerV2",
        "year": 2020,
        "budget": "200M frames",
        "source": "atari5-dataset",
        "note": "model-based",
        "scores": {
            "battle_zone": 40325.0,
            "double_dunk": 17.0,
            "name_this_game": 14649.0,
            "phoenix": 49375.0,
            "qbert": 94688.0,
        },
    },
    "apex": {
        "label": "Ape-X",
        "year": 2018,
        "budget": "orders of magnitude more (360 distributed actors)",
        "source": "atari5-dataset",
        "note": "the ancestor of this distributed setup",
        "scores": {
            "battle_zone": 98895.0,
            "double_dunk": 23.5,
            "name_this_game": 25783.3,
            "phoenix": 224491.1,
            "qbert": 302391.3,
        },
    },
    "r2d2": {
        "label": "R2D2",
        "year": 2019,
        "budget": "orders of magnitude more",
        "source": "atari5-dataset",
        "note": "recurrent",
        "scores": {
            "battle_zone": 751880.0,
            "double_dunk": 23.7,
            "name_this_game": 58182.7,
            "phoenix": 864020.0,
            "qbert": 408850.0,
        },
    },
    "agent57": {
        "label": "Agent57",
        "year": 2020,
        "budget": "orders of magnitude more",
        "source": "atari5-dataset",
        "note": "above human on all 57 games",
        "scores": {
            "battle_zone": 934134.88,
            "double_dunk": 23.93,
            "name_this_game": 54386.77,
            "phoenix": 908264.15,
            "qbert": 580328.14,
        },
    },
    "muzero": {
        "label": "MuZero",
        "year": 2019,
        "budget": "orders of magnitude more",
        "source": "atari5-dataset",
        "note": "model-based with search",
        "scores": {
            "battle_zone": 848623.0,
            "double_dunk": 23.94,
            "name_this_game": 157177.85,
            "phoenix": 955137.84,
            "qbert": 72276.0,
        },
    },
}

SOURCE_LABELS = {
    "atari5-dataset": "Atari-5 dataset (github.com/maitchison/Atari-5, dataset.csv)",
    "btr-paper": "Beyond The Rainbow (arXiv:2411.03820) Table A2",
}

DEFAULT_RANGE = "-10M:"

# 5 ゲーム揃っていないときの問題文の頭。Run 不足は load_runs が 1 度警告するので、
# 集約側からの同じ指摘は落とす。窓が空いただけのときは落とさない。
INCOMPLETE_SET = "Atari-5 needs all five games."

# Run の hns57 と HNS57 定数からの再計算がこの割合を超えて食い違えば警告する。
HNS_CHECK_TOLERANCE = 0.005

RIGHT = "---:"


# ---------------------------------------------------------------------------
# Atari-5 集約
# ---------------------------------------------------------------------------


def normalize(game: str, raw: float) -> float:
    """生スコアを 57 ゲーム表の human normalized score [%] へ直す。"""
    random_score, human_score = HNS57[game]
    return 100.0 * (raw - random_score) / (human_score - random_score)


def atari5_score(hns_by_game: dict):
    """5 ゲームの hns57[%] から Atari-5 の予測値を出す。

    log10(1 + hns) が定義できないゲームが 1 つでもあれば None を返す。項を黙って
    落とすと重み和が変わって過大評価になるため、算出しないほうを選ぶ。
    """
    missing = [game for game in GAMES if hns_by_game.get(game) is None]
    if missing:
        return None, [f"{INCOMPLETE_SET} missing: {', '.join(missing)}"]

    problems = []
    total = 0.0
    for game in GAMES:
        value = hns_by_game[game]
        if 1.0 + value <= 0.0:
            problems.append(
                f"{game}: hns57 = {value:.1f}% makes log10(1 + hns) undefined, "
                f"so Atari-5 is not computed"
            )
            continue
        total += A5_WEIGHTS[game] * math.log10(1.0 + value)
    if problems:
        return None, problems
    return 10.0**total - 1.0, []


def drop_incomplete(problems, complete_set: bool):
    """Run が足りないだけの指摘を落とす。集約 3 種 x チャネル分だけ重複するため。"""
    if complete_set:
        return problems
    return [item for item in problems if not item.startswith(INCOMPLETE_SET)]


def aggregate(hns_by_game: dict):
    """Atari-5 と、比較用の素朴な集約をまとめて返す。"""
    known = sorted(
        value for value in (hns_by_game.get(game) for game in GAMES) if value is not None
    )
    score, problems = atari5_score(hns_by_game)
    node = {
        "atari5": score,
        "median": known[len(known) // 2] if known else None,
        "mean": sum(known) / len(known) if known else None,
        # 中央 3 本の平均。BTR の IQM は seed x game 上の IQM なので直接は対応しない。
        "trimmed_mean3": sum(known[1:-1]) / 3.0 if len(known) == len(GAMES) else None,
        "n_games": len(known),
    }
    return node, problems


# ---------------------------------------------------------------------------
# Run 読み込み
# ---------------------------------------------------------------------------


@dataclass
class GameRun:
    game: str
    run_name: str
    node: dict
    series: dict
    config: dict
    max_step: int
    warnings: list = field(default_factory=list)

    def entry(self, tag: str):
        found = self.series.get(tag)
        if found is None or not getattr(found, "present", False) or not len(found.steps):
            return None
        return found


def channel_tags(channels) -> list:
    tags = []
    for name in channels:
        prefix = CHANNELS[name][0]
        tags.append(f"{prefix}/{SCORE_SUFFIX}")
        tags.append(f"{prefix}/{HNS_SUFFIX}")
        tags.append(f"{prefix}/{LEN_SUFFIX}")
    tags.append(PERF_ELAPSE_TAG)
    tags.append(PERF_RATE_TAG)
    return tags


def config_map(entries) -> dict:
    found = {}
    wanted = set(CONFIG_KEYS.values())
    for key, value in entries:
        if key in wanted:
            found[key] = value
    return {name: found.get(key) for name, key in CONFIG_KEYS.items()}


def _as_int(text):
    if text is None:
        return None
    try:
        return int(text.replace(",", "").strip())
    except ValueError:
        return None


def _as_float(text):
    if text is None:
        return None
    try:
        return float(text.replace(",", "").strip())
    except ValueError:
        return None


def load_runs(values, warnings) -> list:
    loaded = []
    seen = {}
    for value in values:
        resolved = resolve_run(value)
        context = open_run(resolved, want_config=True)
        load_definitions_cheap(context)
        run_warnings = []
        try:
            series = load_metric_series(context, channel_tags(CHANNELS), run_warnings)
        except SourceError as exc:
            raise RuntimeFailure(f"{resolved.run_name}: {exc}") from exc

        config = config_map(context.config_entries)
        game = config["game"]
        if game is None:
            raise UsageError(
                f"{resolved.run_name}: {CONFIG_GAME} is absent from the effective config, "
                f"so the game cannot be determined"
            )
        if game not in GAMES:
            raise UsageError(
                f"{resolved.run_name}: {game} is not an Atari-5 game (expected one of {', '.join(GAMES)})"
            )
        if game in seen:
            raise UsageError(f"{game} is covered by both {seen[game]} and {resolved.run_name}")
        seen[game] = resolved.run_name

        max_step = 0
        for entry in series.values():
            if getattr(entry, "present", False) and len(entry.steps):
                max_step = max(max_step, entry.steps[-1])

        node = run_node(resolved)
        node["game"] = game
        node["warnings"] = run_warnings + list(context.warnings)
        loaded.append(
            GameRun(
                game=game,
                run_name=resolved.run_name,
                node=node,
                series=series,
                config=config,
                max_step=max_step,
                warnings=node["warnings"],
            )
        )

    missing = [game for game in GAMES if game not in seen]
    if missing:
        warnings.append(f"the Atari-5 set is incomplete. missing: {', '.join(missing)}")
    return sorted(loaded, key=lambda item: GAMES.index(item.game))


# ---------------------------------------------------------------------------
# 区間統計
# ---------------------------------------------------------------------------


def resolve_bounds(spec, max_step: int):
    """相対 range を Run ごとの absolute bounds へ解決する。上端はその Run の到達 step。"""
    if spec is None:
        return 0, max_step
    lower = max(0, _resolve_endpoint(spec.start, max_step, 0))
    upper = max(0, _resolve_endpoint(spec.end, max_step, max_step))
    return lower, upper


def slice_values(entry, lower: int, upper: int) -> list:
    if entry is None:
        return []
    begin = bisect.bisect_left(entry.steps, lower)
    end = bisect.bisect_right(entry.steps, upper)
    return list(entry.values[begin:end])


def stats_of(entry, lower: int, upper: int) -> dict:
    window = slice_values(entry, lower, upper)
    everything = list(entry.values) if entry is not None else []
    return {
        "mean": sum(window) / len(window) if window else None,
        "n": len(window),
        "last": everything[-1] if everything else None,
        "max": max(everything) if everything else None,
        "n_all": len(everything),
    }


def check_hns(run: GameRun, prefix: str, warnings: list) -> None:
    """Run が出した hns57 と、HNS57 定数からの再計算を突き合わせる。

    定数がコード側と乖離すると参照エージェントだけ静かにずれるので、Run 1 点で毎回照合する。
    """
    score = run.entry(f"{prefix}/{SCORE_SUFFIX}")
    hns = run.entry(f"{prefix}/{HNS_SUFFIX}")
    if score is None or hns is None:
        return
    reported = hns.values[-1]
    expected = normalize(run.game, score.values[-1])
    if abs(expected - reported) / max(abs(reported), 1.0) > HNS_CHECK_TOLERANCE:
        warnings.append(
            f"{run.run_name}: the HNS57 constants disagree with the run "
            f"(run reports {reported:.2f}%, HNS57 gives {expected:.2f}%). "
            f"Check them against the 57-game table in core/envs/atari1/src/AtariEnv.cpp"
        )


# ---------------------------------------------------------------------------
# 出力の書式
# ---------------------------------------------------------------------------


def fmt_score(value) -> str:
    if value is None:
        return "-"
    return f"{value:,.1f}" if abs(value) < 1000 else f"{value:,.0f}"


def fmt_pct(value) -> str:
    return "-" if value is None else f"{value:,.1f}"


def fmt_ratio(value) -> str:
    return "-" if value is None else f"{value * 100:,.1f}%"


def table(headers, rows, align=None) -> list:
    align = align or ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")
    return lines


def render_tail(result: dict) -> list:
    lines = ["## Notes", ""]
    lines.extend(f"- {item}" for item in result["notes"])
    lines.append("")
    lines.extend(_md_warnings(result))
    return lines


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def command_score(args):
    spec = parse_range(args.range)
    result = run_envelope("atari5-score")
    result["range"] = args.range
    result["channels"] = list(args.channel)
    result["notes"] = []

    runs = load_runs(args.runs, result["warnings"])
    result["runs"] = [run.node for run in runs]

    setup = []
    for run in runs:
        capacity = _as_int(run.config["replay_capacity"])
        num_envs = _as_int(run.config["num_envs"])
        frame_skip = _as_int(run.config["frame_skip"])
        lane_window = capacity // num_envs if capacity and num_envs else None
        lower, upper = resolve_bounds(spec, run.max_step)
        length = stats_of(run.entry(f"42_env/{LEN_SUFFIX}"), lower, upper)["mean"]
        elapse = run.entry(PERF_ELAPSE_TAG)
        rate = run.entry(PERF_RATE_TAG)
        hours = elapse.values[-1] if elapse is not None else None
        setup.append(
            {
                "game": run.game,
                "run_name": run.run_name,
                "exp_step": run.max_step,
                "frames": run.max_step * frame_skip if frame_skip else None,
                "elapse_hour": hours,
                # 実効 throughput。eval による train 停止を含むので壁時計の見積りはこちらを使う。
                # peak は停止を挟まない区間の上限で、両者の差がそのまま eval の重さになる。
                "exp_step_per_sec": run.max_step / (hours * 3600.0) if hours else None,
                "exp_step_per_sec_peak": max(rate.values) if rate is not None else None,
                "replay_ratio": _as_float(run.config["replay_ratio"]),
                "lane_window": lane_window,
                "episode_len_mean": length,
                "w_over_l": lane_window / length if lane_window and length else None,
            }
        )
    result["setup"] = setup

    complete = len(runs) == len(GAMES)
    detail = {}
    for name in args.channel:
        prefix = CHANNELS[name][0]
        games = {}
        for run in runs:
            check_hns(run, prefix, result["warnings"])
            lower, upper = resolve_bounds(spec, run.max_step)
            games[run.game] = {
                "run_name": run.run_name,
                "range": [lower, upper],
                "score": stats_of(run.entry(f"{prefix}/{SCORE_SUFFIX}"), lower, upper),
                "hns57": stats_of(run.entry(f"{prefix}/{HNS_SUFFIX}"), lower, upper),
            }
        node = {"prefix": prefix, "description": CHANNELS[name][1], "games": games}
        for stat in ("mean", "last", "max"):
            summary, problems = aggregate({game: games[game]["hns57"][stat] for game in games})
            node[stat] = summary
            result["warnings"].extend(
                f"{name}/{stat}: {item}" for item in drop_incomplete(problems, complete)
            )
        detail[name] = node
    result["channels_detail"] = detail

    result["notes"].append(
        "mean is the --range window mean; last and max are the final and peak points "
        "over the whole run. max skews high on channels with few episodes per session."
    )
    result["notes"].append(
        "Atari-5 is a weighted sum in log space, not the plain median. Both are shown."
    )
    return result, EXIT_OK


def render_score(result: dict) -> str:
    lines = _md_header(result, "Atari-5 score")
    lines.append(f"- range: `{result['range']}`")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    lines.extend(
        table(
            [
                "game", "run", "exp_step", "frames", "hours",
                "exp/s", "peak/s", "RR", "lane", "L", "W/L",
            ],
            [
                [
                    item["game"],
                    item["run_name"],
                    f"{item['exp_step']:,}",
                    f"{item['frames']:,}" if item["frames"] else "-",
                    f"{item['elapse_hour']:.2f}" if item["elapse_hour"] else "-",
                    f"{item['exp_step_per_sec']:,.0f}" if item["exp_step_per_sec"] else "-",
                    f"{item['exp_step_per_sec_peak']:,.0f}"
                    if item["exp_step_per_sec_peak"]
                    else "-",
                    f"{item['replay_ratio']:g}" if item["replay_ratio"] else "-",
                    f"{item['lane_window']:,}" if item["lane_window"] else "-",
                    f"{item['episode_len_mean']:,.0f}" if item["episode_len_mean"] else "-",
                    f"{item['w_over_l']:.1f}" if item["w_over_l"] else "-",
                ]
                for item in result["setup"]
            ],
            ["---", "---"] + [RIGHT] * 9,
        )
    )

    for name, node in result["channels_detail"].items():
        lines.append(f"## {name}: {node['description']}")
        lines.append("")
        rows = []
        for game in GAMES:
            entry = node["games"].get(game)
            if entry is None:
                rows.append([game] + ["-"] * 6)
                continue
            rows.append(
                [
                    game,
                    fmt_score(entry["score"]["mean"]),
                    fmt_pct(entry["hns57"]["mean"]),
                    fmt_score(entry["score"]["last"]),
                    fmt_pct(entry["hns57"]["last"]),
                    fmt_score(entry["score"]["max"]),
                    f"{entry['score']['n']}/{entry['score']['n_all']}",
                ]
            )
        lines.extend(
            table(
                ["game", "win score", "win hns%", "last score", "last hns%", "max score", "n win/all"],
                rows,
                ["---", RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT],
            )
        )
        lines.extend(
            table(
                ["stat", "Atari-5", "median", "mean", "mid3 mean"],
                [
                    [
                        label,
                        fmt_pct(node[stat]["atari5"]),
                        fmt_pct(node[stat]["median"]),
                        fmt_pct(node[stat]["mean"]),
                        fmt_pct(node[stat]["trimmed_mean3"]),
                    ]
                    for stat, label in (("mean", "window"), ("last", "last"), ("max", "max"))
                ],
                ["---", RIGHT, RIGHT, RIGHT, RIGHT],
            )
        )

    lines.extend(render_tail(result))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# curve
# ---------------------------------------------------------------------------


def command_curve(args):
    result = run_envelope("atari5-curve")
    result["channel"] = args.channel
    result["bins"] = args.bins
    result["notes"] = []

    if args.bins < 1:
        raise UsageError("--bins must be 1 or more")

    runs = load_runs(args.runs, result["warnings"])
    result["runs"] = [run.node for run in runs]
    if not runs:
        raise UsageError("no runs were given")
    prefix = CHANNELS[args.channel][0]

    # 窓の上端は Run 間で最も短いものへ揃える。揃えないと最終窓だけ予算が食い違う。
    common_max = min(run.max_step for run in runs)
    spread = max(run.max_step for run in runs) - common_max
    if common_max and spread / common_max > 0.02:
        result["warnings"].append(
            f"the runs differ by {spread:,} in reached exp_step; truncated at the common "
            f"upper bound {common_max:,}"
        )
    result["common_max_step"] = common_max

    elapse = {}
    for run in runs:
        entry = run.entry(PERF_ELAPSE_TAG)
        elapse[run.game] = (list(entry.steps), list(entry.values)) if entry is not None else None

    width = common_max / args.bins
    points = []
    for index in range(args.bins):
        lower = int(round(width * index))
        upper = int(round(width * (index + 1)))
        hns = {}
        for run in runs:
            values = slice_values(run.entry(f"{prefix}/{HNS_SUFFIX}"), lower + 1, upper)
            hns[run.game] = sum(values) / len(values) if values else None
        summary, problems = aggregate(hns)
        hours = []
        for run in runs:
            pair = elapse.get(run.game)
            if pair is None:
                continue
            position = bisect.bisect_right(pair[0], upper) - 1
            if position >= 0:
                hours.append(pair[1][position])
        points.append(
            {
                "exp_step_from": lower,
                "exp_step_to": upper,
                "elapse_hour_mean": sum(hours) / len(hours) if hours else None,
                "hns57": hns,
                **summary,
            }
        )
        result["warnings"].extend(
            f"{lower:,}:{upper:,}: {item}"
            for item in drop_incomplete(problems, len(runs) == len(GAMES))
        )
    result["points"] = points
    result["notes"].append(
        "elapse_hour is the run mean at each window upper bound. It is measured "
        "wall-clock and includes the train stalls caused by eval."
    )
    result["notes"].append(
        "Early windows can hold a game far below random, which leaves Atari-5 empty."
    )
    return result, EXIT_OK


def render_curve(result: dict) -> str:
    lines = _md_header(result, "Atari-5 curve")
    lines.append(f"- channel: {result['channel']} / bins: {result['bins']}")
    lines.append(f"- common_max_step: {result['common_max_step']:,}")
    lines.append("")
    lines.extend(
        table(
            ["exp_step<=", "hours"] + list(GAMES) + ["Atari-5", "median"],
            [
                [
                    f"{point['exp_step_to'] / 1e6:,.1f}M",
                    f"{point['elapse_hour_mean']:.2f}" if point["elapse_hour_mean"] else "-",
                ]
                + [fmt_pct(point["hns57"].get(game)) for game in GAMES]
                + [fmt_pct(point["atari5"]), fmt_pct(point["median"])]
                for point in result["points"]
            ],
            ["---", RIGHT] + [RIGHT] * len(GAMES) + [RIGHT, RIGHT],
        )
    )
    lines.extend(render_tail(result))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def command_compare(args):
    for key in args.ref:
        if key not in REFERENCES:
            raise UsageError(
                f"unknown reference: {key} (available: {', '.join(sorted(REFERENCES))})"
            )

    spec = parse_range(args.range)
    result = run_envelope("atari5-compare")
    result["channel"] = args.channel
    result["stat"] = args.stat
    result["range"] = args.range
    result["notes"] = []

    runs = load_runs(args.runs, result["warnings"])
    result["runs"] = [run.node for run in runs]
    prefix = CHANNELS[args.channel][0]
    key = "mean" if args.stat == "window" else args.stat

    ours = {}
    ours_raw = {}
    for run in runs:
        check_hns(run, prefix, result["warnings"])
        lower, upper = resolve_bounds(spec, run.max_step)
        ours[run.game] = stats_of(run.entry(f"{prefix}/{HNS_SUFFIX}"), lower, upper)[key]
        ours_raw[run.game] = stats_of(run.entry(f"{prefix}/{SCORE_SUFFIX}"), lower, upper)[key]
    summary, problems = aggregate(ours)
    result["warnings"].extend(drop_incomplete(problems, len(runs) == len(GAMES)))
    result["ours"] = {"hns57": ours, "score": ours_raw, **summary}

    references = []
    for key_name in args.ref:
        spec_ref = REFERENCES[key_name]
        hns = {game: normalize(game, value) for game, value in spec_ref["scores"].items()}
        ref_summary, ref_problems = aggregate(hns)
        result["warnings"].extend(f"{key_name}: {item}" for item in ref_problems)
        references.append(
            {
                "key": key_name,
                "label": spec_ref["label"],
                "year": spec_ref["year"],
                "budget": spec_ref["budget"],
                "source": SOURCE_LABELS[spec_ref["source"]],
                "note": spec_ref["note"],
                "score": spec_ref["scores"],
                "hns57": hns,
                **ref_summary,
                "ratio": {
                    game: (ours[game] / hns[game])
                    if ours.get(game) is not None and hns.get(game)
                    else None
                    for game in GAMES
                },
                "atari5_ratio": (summary["atari5"] / ref_summary["atari5"])
                if summary["atari5"] and ref_summary["atari5"]
                else None,
            }
        )
    result["references"] = references

    result["notes"].append(
        "Published values are aggregated differently per source. BTR reports the best "
        "eval during training over 100 episodes x 3 seeds, so --stat max lines up best."
    )
    result["notes"].append(
        "The protocols do not always match. Against published values measured without "
        "sticky actions, these runs are on the harder side of the mismatch."
    )
    return result, EXIT_OK


def render_compare(result: dict) -> str:
    lines = _md_header(result, "Atari-5 compare")
    lines.append(
        f"- channel: {result['channel']} / stat: {result['stat']} / range: `{result['range']}`"
    )
    lines.append("")
    ours = result["ours"]
    for reference in result["references"]:
        lines.append(f"## vs {reference['label']} ({reference['year']}, {reference['budget']})")
        lines.append("")
        rows = [
            [
                game,
                fmt_score(reference["score"].get(game)),
                fmt_pct(reference["hns57"].get(game)),
                fmt_score(ours["score"].get(game)),
                fmt_pct(ours["hns57"].get(game)),
                fmt_ratio(reference["ratio"].get(game)),
            ]
            for game in GAMES
        ]
        rows.append(
            [
                "**Atari-5**",
                "",
                fmt_pct(reference["atari5"]),
                "",
                fmt_pct(ours["atari5"]),
                fmt_ratio(reference["atari5_ratio"]),
            ]
        )
        lines.extend(
            table(
                ["game", "ref score", "ref hns%", "ours score", "ours hns%", "ours/ref"],
                rows,
                ["---", RIGHT, RIGHT, RIGHT, RIGHT, RIGHT],
            )
        )
        lines.append(f"- source: {reference['source']}")
        lines.append(f"- protocol: {reference['note']}")
        lines.append("")
    lines.extend(render_tail(result))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# refs
# ---------------------------------------------------------------------------


def command_refs(args):
    result = run_envelope("atari5-refs")
    result["notes"] = []
    entries = []
    for key in REFERENCES:
        spec = REFERENCES[key]
        hns = {game: normalize(game, value) for game, value in spec["scores"].items()}
        summary, problems = aggregate(hns)
        result["warnings"].extend(f"{key}: {item}" for item in problems)
        entries.append(
            {
                "key": key,
                "label": spec["label"],
                "year": spec["year"],
                "budget": spec["budget"],
                "source": SOURCE_LABELS[spec["source"]],
                "note": spec["note"],
                "score": spec["scores"],
                "hns57": hns,
                **summary,
            }
        )
    entries.sort(key=lambda item: item["atari5"] if item["atari5"] is not None else 0.0, reverse=True)
    result["references"] = entries
    result["weights"] = A5_WEIGHTS
    result["hns57_constants"] = {game: list(HNS57[game]) for game in GAMES}
    result["notes"].append(
        "The HNS57 constants follow the 57-game table in core/envs/atari1/src/AtariEnv.cpp."
    )
    result["notes"].append(
        "Add new references as raw scores. This tool does the normalization."
    )
    return result, EXIT_OK


def render_refs(result: dict) -> str:
    lines = _md_header(result, "Atari-5 references")
    lines.extend(
        table(
            ["key", "label", "year", "budget", "Atari-5", "median"] + list(GAMES),
            [
                [
                    item["key"],
                    item["label"],
                    str(item["year"]),
                    item["budget"],
                    fmt_pct(item["atari5"]),
                    fmt_pct(item["median"]),
                ]
                + [fmt_score(item["score"].get(game)) for game in GAMES]
                for item in result["references"]
            ],
            ["---", "---", RIGHT, "---", RIGHT, RIGHT] + [RIGHT] * len(GAMES),
        )
    )
    lines.append("## Regression weights and normalization constants")
    lines.append("")
    lines.extend(
        table(
            ["game", "weight", "random", "human"],
            [
                [
                    game,
                    f"{A5_WEIGHTS[game]:.4f}",
                    f"{result['hns57_constants'][game][0]:,.1f}",
                    f"{result['hns57_constants'][game][1]:,.1f}",
                ]
                for game in GAMES
            ],
            ["---", RIGHT, RIGHT, RIGHT],
        )
    )
    lines.append("## Protocols")
    lines.append("")
    for item in result["references"]:
        lines.append(f"- **{item['label']}**: {item['note']} ({item['source']})")
    lines.append("")
    lines.extend(render_tail(result))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

# 疑似 Run の tag 設計。
#   既存 tag 名で出したものは、MetricsViewer で実 Run と同じグラフに重なる。
#   Atari-5 集約を 52_eval2/12_hns57_mean へ載せると、5 ゲームの hns57 曲線と
#   集約が 1 枚に並ぶ。
#   60_atari5/* は既存側に対応物が無い集約専用で、集約どうし・ゲーム横並び・
#   参照エージェントの水平線を置く。
EXPORT_GROUP = "60_atari5"

# チャネルごとの metrics 定義。eval_name は config の eval 定義名に合わせる。
CHANNEL_DEF = {
    "train": {"scope": "train", "event": "train", "eval_name": None},
    "eval1": {"scope": "eval", "event": "session_end", "eval_name": "eval_target"},
    "eval2": {"scope": "eval", "event": "session_end", "eval_name": "eval"},
    "evalg": {"scope": "eval", "event": "session_end", "eval_name": "greedy_dist"},
}


def metric_def(source_key, channel=None, step_axis="exp_step", target="env"):
    node = {
        "clip": None,
        "ema_alpha": None,
        "eval_episodes": None,
        "eval_name": None,
        "event": "train",
        "interval": 1,
        "num_envs": None,
        "runner": "train",
        "scope": "train",
        "source_key": source_key,
        "step_axis": step_axis,
        "target": target,
    }
    if channel is not None:
        node.update(CHANNEL_DEF[channel])
    return node


def asof_join(runs, tag):
    """全 Run が観測済みの step だけを残して値を揃える。

    grid が一致していればそのまま交差集合になる。ずれている Run があるときは直近値を
    持ち越すが、いずれかの Run の観測末尾を超えた step は捨てる。外挿はしない。
    """
    entries = {run.game: run.entry(tag) for run in runs}
    if not entries or any(entry is None for entry in entries.values()):
        return []
    lower = max(entry.steps[0] for entry in entries.values())
    upper = min(entry.steps[-1] for entry in entries.values())
    if lower > upper:
        return []
    steps = sorted({s for entry in entries.values() for s in entry.steps if lower <= s <= upper})
    joined = []
    for step in steps:
        row = {}
        for game, entry in entries.items():
            row[game] = entry.values[bisect.bisect_right(entry.steps, step) - 1]
        joined.append((step, row))
    return joined


class Emitter:
    """defs と scalar 行を溜める。btr_to_metrics.py と同じ出力契約。"""

    def __init__(self):
        self.defs = {}
        self.rows = []

    def add(self, tag, points, definition):
        if not points:
            return False
        self.defs[tag] = definition
        self.rows.extend((int(step), tag, float(value)) for step, value in points)
        return True


def command_export(args):
    for key in args.ref:
        if key not in REFERENCES:
            raise UsageError(
                f"unknown reference: {key} (available: {', '.join(sorted(REFERENCES))})"
            )

    result = run_envelope("atari5-export")
    result["channel"] = args.channel
    result["notes"] = []

    runs = load_runs(args.runs, result["warnings"])
    result["runs"] = [run.node for run in runs]
    if len(runs) != len(GAMES):
        raise UsageError(
            f"export needs all five games; got {', '.join(run.game for run in runs)}"
        )

    workspaces = {run.node["workspace"] for run in runs}
    if len(workspaces) != 1 or None in workspaces:
        raise UsageError(
            "the source runs must share one workspace; pass --out to choose the destination"
        )
    workspace = workspaces.pop()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (
        Path(args.out).resolve()
        if args.out
        else WORKSPACES_ROOT / workspace / "runs" / f"run_{stamp}_{args.name}"
    )
    if out_dir.exists() and not args.force:
        raise UsageError(f"destination already exists: {out_dir} (pass --force to overwrite)")

    emitter = Emitter()
    channels_written = []

    # 1. 既存 tag 名。実 Run と同じグラフに重なる。
    for index, name in enumerate(("train", "eval2", "evalg")):
        prefix = CHANNELS[name][0]
        joined = asof_join(runs, f"{prefix}/{HNS_SUFFIX}")
        points = []
        for step, row in joined:
            score, _ = atari5_score(row)
            if score is not None:
                points.append((step, score))
        if emitter.add(
            f"{prefix}/{HNS_SUFFIX}", points, metric_def("atari5.hns57", channel=name)
        ):
            channels_written.append({"channel": name, "points": len(points), "joined": len(joined)})
        emitter.add(
            f"{EXPORT_GROUP}/1{index}_atari5_{name}",
            points,
            metric_def("atari5.hns57", channel=name),
        )

    # 2. 主チャネルの集約と、ゲーム別 hns57 を 1 Run へまとめる。
    prefix = CHANNELS[args.channel][0]
    joined = asof_join(runs, f"{prefix}/{HNS_SUFFIX}")
    stats = {"median": [], "mean": [], "trimmed_mean3": []}
    per_game = {game: [] for game in GAMES}
    for step, row in joined:
        node, _ = aggregate(row)
        for key in stats:
            if node[key] is not None:
                stats[key].append((step, node[key]))
        for game in GAMES:
            if row.get(game) is not None:
                per_game[game].append((step, row[game]))
    for number, key in enumerate(("median", "mean", "trimmed_mean3"), start=20):
        emitter.add(
            f"{EXPORT_GROUP}/{number}_{key}",
            stats[key],
            metric_def(f"atari5.{key}", channel=args.channel),
        )
    for number, game in enumerate(GAMES, start=30):
        emitter.add(
            f"{EXPORT_GROUP}/{number}_{game}_hns57",
            per_game[game],
            metric_def("mean.hns57", channel=args.channel),
        )

    # 3. 5 本平均の経過時間。実時間軸で読むために置く。
    elapsed = asof_join(runs, PERF_ELAPSE_TAG)
    emitter.add(
        PERF_ELAPSE_TAG,
        [(step, sum(row.values()) / len(row)) for step, row in elapsed],
        metric_def("elapse_hour", target="runner"),
    )

    # 4. 参照エージェントは両端 2 点だけ置いて水平線にする。
    span = [point[0] for point in joined[:1] + joined[-1:]]
    references = []
    for key in args.ref:
        spec = REFERENCES[key]
        value, problems = atari5_score(
            {game: normalize(game, raw) for game, raw in spec["scores"].items()}
        )
        result["warnings"].extend(f"{key}: {item}" for item in problems)
        if value is None or not span:
            continue
        emitter.add(
            f"{EXPORT_GROUP}/90_ref_{key}",
            [(step, value) for step in span],
            metric_def(f"atari5.reference.{key}", channel=args.channel),
        )
        references.append({"key": key, "label": spec["label"], "atari5": value})
    result["references"] = references

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "json").mkdir(exist_ok=True)
    payload = {"data": emitter.defs, "tag": "metrics.scalar.defs", "type": "json"}
    (out_dir / "json" / "metrics.scalar.defs.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )

    cache = out_dir / "metrics_cache.db"
    if cache.exists():
        try:
            cache.unlink()
        except OSError:
            result["warnings"].append(
                "metrics_cache.db is locked; close the run in MetricsViewer and delete it"
            )

    emitter.rows.sort(key=lambda row: (row[0], row[1]))
    lines = [json.dumps({"event": "start", "timestamp": datetime.now().isoformat(timespec="seconds"), "type": "meta"})]
    lines.append(json.dumps(payload, ensure_ascii=False))
    lines.extend(
        json.dumps({"step": step, "tag": tag, "type": "scalar", "value": value})
        for step, tag, value in emitter.rows
    )
    (out_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    result["out_dir"] = str(out_dir)
    result["run_name"] = out_dir.name
    result["workspace"] = workspace
    result["tags"] = sorted(emitter.defs)
    result["rows"] = len(emitter.rows)
    result["channels"] = channels_written
    result["notes"].append(
        "Tags that already exist in real runs carry the Atari-5 aggregate, so the pseudo "
        "run overlays on the same graph as the five source runs."
    )
    result["notes"].append(
        f"{EXPORT_GROUP}/* holds aggregate-only series: the three channels side by side, "
        "the plain median and mean, every game on one axis, and reference agents as flat lines."
    )
    result["notes"].append(
        "This pseudo run has no config dump, so atari5.py cannot read it back. It is for "
        "MetricsViewer and inspect_run.py."
    )
    # 集約と平均は交換できない。log 空間の重み付き和なので Jensen の分だけずれる。
    result["notes"].append(
        "The series aggregate per step and are then plotted, so a window mean taken in the "
        "viewer is a mean of Atari-5 values. The score subcommand instead averages each game "
        "over the window and aggregates once. The two differ; neither is wrong."
    )
    result["notes"].append(
        "A channel stops at the earliest last step among the five games, so a run that lost "
        "an eval session truncates the joined series for every game."
    )
    return result, EXIT_OK


def render_export(result: dict) -> str:
    lines = _md_header(result, "Atari-5 export")
    lines.append(f"- run_name: `{result['run_name']}`")
    lines.append(f"- workspace: {result['workspace']}")
    lines.append(f"- out_dir: `{result['out_dir']}`")
    lines.append(f"- rows: {result['rows']:,}")
    lines.append("")
    lines.extend(
        table(
            ["channel", "joined steps", "atari5 points"],
            [
                [item["channel"], f"{item['joined']:,}", f"{item['points']:,}"]
                for item in result["channels"]
            ],
            ["---", RIGHT, RIGHT],
        )
    )
    if result["references"]:
        lines.extend(
            table(
                ["reference", "Atari-5"],
                [[item["label"], fmt_pct(item["atari5"])] for item in result["references"]],
                ["---", RIGHT],
            )
        )
    lines.append("## Tags")
    lines.append("")
    lines.extend(f"- `{tag}`" for tag in result["tags"])
    lines.append("")
    lines.extend(render_tail(result))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


COMMANDS = {
    "score": command_score,
    "curve": command_curve,
    "compare": command_compare,
    "refs": command_refs,
    "export": command_export,
}

RENDERERS = {
    "atari5-score": render_score,
    "atari5-curve": render_curve,
    "atari5-compare": render_compare,
    "atari5-refs": render_refs,
    "atari5-export": render_export,
}


def _add_common(parser) -> None:
    parser.add_argument(
        "--format", choices=("md", "json"), default="md", help="output format (default: md)"
    )
    parser.add_argument(
        "--output", metavar="PATH", help="write the result to a file instead of stdout"
    )


def _add_runs(parser) -> None:
    parser.add_argument(
        "runs",
        nargs="+",
        metavar="RUN",
        help="run name, or an existing relative/absolute directory path. The game comes from "
        f"the effective {CONFIG_GAME}, not from the run name.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atari5.py",
        description="Aggregate five Atari-5 runs into benchmark statistics. Read-only.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  atari5.py score RUN_BZ RUN_DD RUN_NTG RUN_PHX RUN_QB\n"
            "  atari5.py score RUN... --range -5M: --channel evalg\n"
            "  atari5.py curve RUN... --bins 10\n"
            "  atari5.py compare RUN... --ref btr --ref iqn --stat max\n"
            "  atari5.py refs\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    score = subparsers.add_parser("score", help="per-game scores and the Atari-5 aggregate")
    _add_runs(score)
    score.add_argument(
        "--range",
        default=DEFAULT_RANGE,
        help=f"window for the mean, in inspect_run.py range syntax (default: {DEFAULT_RANGE})",
    )
    score.add_argument(
        "--channel",
        action="append",
        choices=tuple(CHANNELS),
        help=f"evaluation channel, repeatable (default: {', '.join(DEFAULT_CHANNELS)})",
    )
    _add_common(score)

    curve = subparsers.add_parser("curve", help="Atari-5 against budget and wall-clock")
    _add_runs(curve)
    curve.add_argument("--bins", type=int, default=10, help="number of windows (default: 10)")
    curve.add_argument(
        "--channel", choices=tuple(CHANNELS), default="eval2", help="evaluation channel"
    )
    _add_common(curve)

    compare = subparsers.add_parser("compare", help="compare against published agents")
    _add_runs(compare)
    compare.add_argument(
        "--ref",
        action="append",
        metavar="KEY",
        help="reference agent key, repeatable (default: btr). See the refs subcommand.",
    )
    compare.add_argument(
        "--channel", choices=tuple(CHANNELS), default="eval2", help="evaluation channel"
    )
    compare.add_argument(
        "--stat",
        choices=("max", "last", "window"),
        default="max",
        help="statistic to compare (default: max, matching how most papers report)",
    )
    compare.add_argument("--range", default=DEFAULT_RANGE, help="window used by --stat window")
    _add_common(compare)

    refs = subparsers.add_parser("refs", help="list the built-in reference table")
    _add_common(refs)

    export = subparsers.add_parser(
        "export", help="write a pseudo run that MetricsViewer can open (writes files)"
    )
    _add_runs(export)
    export.add_argument(
        "--out",
        metavar="DIR",
        help="destination run directory (default: <workspace>/runs/run_<stamp>_<name>)",
    )
    export.add_argument("--name", default="atari5", help="run name suffix (default: atari5)")
    export.add_argument(
        "--channel",
        choices=tuple(CHANNELS),
        default="eval2",
        help=f"channel feeding the {EXPORT_GROUP}/2x and /3x series (default: eval2)",
    )
    export.add_argument(
        "--ref",
        action="append",
        metavar="KEY",
        help="reference agent to draw as a flat line, repeatable (default: btr)",
    )
    export.add_argument("--force", action="store_true", help="overwrite an existing destination")
    _add_common(export)
    return parser


def main(argv=None, stdout=None, stderr=None) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    parser = build_parser()
    try:
        args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else EXIT_OK

    if args.subcommand == "score" and not args.channel:
        args.channel = list(DEFAULT_CHANNELS)
    if args.subcommand in ("compare", "export") and not args.ref:
        args.ref = ["btr"]

    try:
        target = Path(args.output).resolve() if args.output else None
        if target is not None and not target.parent.is_dir():
            raise UsageError(f"output parent directory does not exist: {target.parent}")
        result, exit_code = COMMANDS[args.subcommand](args)
        text = (
            render_json(result)
            if args.format == "json"
            else RENDERERS[result["subcommand"]](result) + "\n"
        )
        write_output(text, target, out)
    except UsageError as exc:
        print(f"error: {exc}", file=err)
        return EXIT_USAGE
    except RuntimeFailure as exc:
        print(f"error: {exc}", file=err)
        return EXIT_RUNTIME

    for warning in result["warnings"]:
        print(f"warning: {warning}", file=err)
    for run in result["runs"]:
        for warning in run.get("warnings", []):
            print(f"warning: {run['run_name']}: {warning}", file=err)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

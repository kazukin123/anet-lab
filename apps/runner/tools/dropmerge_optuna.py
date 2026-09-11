#!/usr/bin/env python3
"""DropMerge NN 構成探索用の Optuna harness。

Optuna は C++ runner の外側に置き、このスクリプトが trial ごとの main config
生成、AnetRLRunner の ``--config`` 起動、``metrics.jsonl`` の採点を担当する。
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from optuna_common import (
    JapaneseArgumentParser,
    JapaneseHelpFormatter,
    MetricsSpec,
    OptunaHarnessRuntime,
    TrialExecutionError,
    add_runner_args,
    add_score_window_args,
    default_main_config,
    localize_parser,
    resolve_workspace_paths,
    storage_path_from_text,
    storage_url_from_text,
)


@dataclass(frozen=True)
class TrialParams:
    """DropMerge NN 構成探索で Optuna が直接扱うパラメータ。"""

    cnn_channels: int
    res_blocks: int
    token_mode: str
    d_model: int
    transformer_layers: int
    ff_mult: int
    trunk_width: int
    head_width: int


class DropMergeDomain:
    """DropMerge 固有の探索空間、cost proxy、trial config 生成を担当する。"""

    HELP_SUMMARY = (
        "DropMerge domain: NN 構成探索を対象にし、Flatten 固定の branch config を生成します。"
    )
    DRY_RUN_DESCRIPTION = (
        "DropMerge domain params を指定して trial config / manifest を生成し、"
        "cost_budget 判定を表示します。"
    )
    RUN_TRIAL_DESCRIPTION = (
        "CLI で指定した DropMerge NN 構成を Optuna trial として multi-seed 実行し、"
        "metrics.jsonl を採点して DB / artifact に登録します。"
    )
    RUN_STUDY_DESCRIPTION = (
        "Optuna study を作成/再開し、Optuna が生成した DropMerge NN 構成 trial を順次 runner で実行します。"
    )
    SUMMARIZE_STUDY_DESCRIPTION = (
        "既存 Optuna study の同一 DropMerge TrialParams を group 化し、"
        "mean/range/std の multi-objective summary study を生成します。"
    )

    PRIMARY_TAGS = (
        "21_eval/03_target_reward_ema",
        "21_eval/04_policy_reward_ema",
    )

    # primary score には入れず、trial の解釈や後段分析のために保存する補助指標。
    SUPPLEMENTAL_TAGS = (
        "51_eval1/61_ep_maxrank_mean_ema",
        "52_eval2/61_ep_maxrank_mean_ema",
        "51_eval1/62_ep_frct_mean_ema",
        "52_eval2/62_ep_frct_mean_ema",
        "51_eval1/83_tr_blk_mean_ema",
        "52_eval2/83_tr_blk_mean_ema",
        "51_eval1/84_tr_timeout_max_ema",
        "52_eval2/84_tr_timeout_mean_ema",
        "51_eval1/85_tr_nolg_mean_ema",
        "52_eval2/85_tr_nolg_mean_ema",
        "90_perf/12_exp_step_per_sec",
        "90_perf/22_exp_step_per_sec_ema",
        "90_perf/90_elapse_hour",
    )

    TRIAL_PARAM_CHOICES = {
        "cnn_channels": [48, 64],
        "res_blocks": [2, 4],
        "token_mode": ["current", "stronger"],
        "d_model": [96, 128, 192],
        "transformer_layers": [2, 4],
        "ff_mult": [2, 4],
        "trunk_width": [1024, 2048],
        "head_width": [512, 1024],
    }

    LATE_SCORE_WINDOWS = (
        ("score_60_80", "60%", "80%"),
        ("score_80_100", "80%", "100%"),
    )

    @classmethod
    def add_param_args(cls, parser, *, default_values: bool = True) -> None:
        """DropMerge 固有の NN params 引数を CLI parser に追加する。"""
        default_note = "" if default_values else "未指定時は全候補を探索し、指定時はその値に固定する。"
        parser.add_argument(
            "--cnn-channels",
            type=int,
            choices=cls.TRIAL_PARAM_CHOICES["cnn_channels"],
            default=64 if default_values else None,
            help=f"CNN/ResBlock の channel 数 C。{default_note}",
        )
        parser.add_argument(
            "--res-blocks",
            type=int,
            choices=cls.TRIAL_PARAM_CHOICES["res_blocks"],
            default=4 if default_values else None,
            help=f"ResBlock の繰り返し数 D。{default_note}",
        )
        parser.add_argument(
            "--token-mode",
            choices=cls.TRIAL_PARAM_CHOICES["token_mode"],
            default="current" if default_values else None,
            help=f"token 解像度 N のモード。{default_note}",
        )
        parser.add_argument(
            "--d-model",
            type=int,
            choices=cls.TRIAL_PARAM_CHOICES["d_model"],
            default=96 if default_values else None,
            help=f"Transformer の d_model M。{default_note}",
        )
        parser.add_argument(
            "--transformer-layers",
            type=int,
            choices=cls.TRIAL_PARAM_CHOICES["transformer_layers"],
            default=2 if default_values else None,
            help=f"Transformer 層数 L。{default_note}",
        )
        parser.add_argument(
            "--ff-mult",
            type=int,
            choices=cls.TRIAL_PARAM_CHOICES["ff_mult"],
            default=2 if default_values else None,
            help=f"dim_feedforward = d_model * ff_mult。{default_note}",
        )
        parser.add_argument(
            "--trunk-width",
            type=int,
            choices=cls.TRIAL_PARAM_CHOICES["trunk_width"],
            default=1024 if default_values else None,
            help=f"Flatten 後 trunk Linear 幅 H。{default_note}",
        )
        parser.add_argument(
            "--head-width",
            type=int,
            choices=cls.TRIAL_PARAM_CHOICES["head_width"],
            default=512 if default_values else None,
            help=f"value/adv stream の head Linear 幅。{default_note}",
        )

    @classmethod
    def suggest_params(cls, trial, args=None) -> TrialParams:
        """Optuna trial から DropMerge NN 構成を suggest する。"""
        del args
        return TrialParams(
            cnn_channels=trial.suggest_categorical("cnn_channels", cls.TRIAL_PARAM_CHOICES["cnn_channels"]),
            res_blocks=trial.suggest_categorical("res_blocks", cls.TRIAL_PARAM_CHOICES["res_blocks"]),
            token_mode=trial.suggest_categorical("token_mode", cls.TRIAL_PARAM_CHOICES["token_mode"]),
            d_model=trial.suggest_categorical("d_model", cls.TRIAL_PARAM_CHOICES["d_model"]),
            transformer_layers=trial.suggest_categorical(
                "transformer_layers",
                cls.TRIAL_PARAM_CHOICES["transformer_layers"],
            ),
            ff_mult=trial.suggest_categorical("ff_mult", cls.TRIAL_PARAM_CHOICES["ff_mult"]),
            trunk_width=trial.suggest_categorical("trunk_width", cls.TRIAL_PARAM_CHOICES["trunk_width"]),
            head_width=trial.suggest_categorical("head_width", cls.TRIAL_PARAM_CHOICES["head_width"]),
        )

    @classmethod
    def fixed_params_from_args(cls, args) -> dict[str, object]:
        """run-study で固定指定された探索 params だけを取り出す。"""
        result: dict[str, object] = {}
        for name in TrialParams.__dataclass_fields__:
            value = getattr(args, name, None)
            if value is not None:
                result[name] = value
        return result

    @classmethod
    def fixed_param_cli_args(cls, args) -> list[tuple[str, object]]:
        """run-study 再開用 args に含める固定探索 params だけを返す。"""
        result: list[tuple[str, object]] = []
        for name, value in cls.fixed_params_from_args(args).items():
            result.append((f"--{name.replace('_', '-')}", value))
        return result

    @classmethod
    def fixed_params_for_sampler(cls, args) -> dict[str, object]:
        """Optuna PartialFixedSampler へ渡す固定探索 params を返す。"""
        return cls.fixed_params_from_args(args)

    @classmethod
    def search_space_from_args(cls, args) -> dict[str, list[object]]:
        """run-study grid mode 用に固定指定を反映した探索空間を返す。"""
        fixed_params = cls.fixed_params_from_args(args)
        return {
            name: [fixed_params[name]] if name in fixed_params else list(choices)
            for name, choices in cls.TRIAL_PARAM_CHOICES.items()
        }

    @classmethod
    def params_from_mapping(cls, mapping: dict, *, source: str = "params") -> TrialParams:
        """Optuna params/user input の dict を型付き TrialParams に変換する。"""
        missing = [
            name
            for name in TrialParams.__dataclass_fields__
            if mapping.get(name) is None
        ]
        if missing:
            raise ValueError(f"Invalid {source}: missing TrialParams fields. missing={missing}")

        try:
            return TrialParams(
                cnn_channels=int(mapping["cnn_channels"]),
                res_blocks=int(mapping["res_blocks"]),
                token_mode=str(mapping["token_mode"]),
                d_model=int(mapping["d_model"]),
                transformer_layers=int(mapping["transformer_layers"]),
                ff_mult=int(mapping["ff_mult"]),
                trunk_width=int(mapping["trunk_width"]),
                head_width=int(mapping["head_width"]),
            )
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid {source}: TrialParams fields have invalid values. params={mapping}") from e

    @classmethod
    def params_from_args(cls, args) -> TrialParams:
        """run-trial/dry-run の固定 CLI 引数を TrialParams に変換する。"""
        return TrialParams(
            cnn_channels=args.cnn_channels,
            res_blocks=args.res_blocks,
            token_mode=args.token_mode,
            d_model=args.d_model,
            transformer_layers=args.transformer_layers,
            ff_mult=args.ff_mult,
            trunk_width=args.trunk_width,
            head_width=args.head_width,
        )

    @staticmethod
    def params_to_dict(params: TrialParams) -> dict:
        """common runtime が JSON/Optuna attrs へ保存する params dict を作る。"""
        return asdict(params)

    @classmethod
    def param_distributions(cls, optuna) -> dict:
        """summary study の fixed distribution 作成に使う探索分布を返す。"""
        return {
            name: optuna.distributions.CategoricalDistribution(choices)
            for name, choices in cls.TRIAL_PARAM_CHOICES.items()
        }

    @classmethod
    def validate_params_in_search_space(cls, params: TrialParams) -> None:
        """既存 study の params が現在の DropMerge 探索空間内か確認する。"""
        values = asdict(params)
        invalid = {
            name: value
            for name, value in values.items()
            if value not in cls.TRIAL_PARAM_CHOICES[name]
        }
        if invalid:
            raise ValueError(f"Source trial params are outside current Optuna search space: {invalid}")

    @staticmethod
    def params_key(params: TrialParams) -> tuple[tuple[str, object], ...]:
        """同一 NN 構成を判定するための安定 key を作る。"""
        return tuple(asdict(params).items())

    @staticmethod
    def params_key_from_mapping(mapping: dict) -> tuple[tuple[str, object], ...]:
        """Optuna trial.params から同一 NN 構成判定用 key を作る。"""
        return tuple((name, mapping.get(name)) for name in TrialParams.__dataclass_fields__)

    @staticmethod
    def conv_out(size: int, stride: int = 2, kernel: int = 3, padding: int = 1, dilation: int = 1) -> int:
        """DropMerge の stride 2 convolution 後の token 解像度を概算する。"""
        return math.floor((size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)

    @classmethod
    def token_dims(cls, token_mode: str, grid_width: int = 58, grid_height: int = 46) -> tuple[int, int]:
        """DropMerge G5846 grid から token_mode 別の token 幅/高さを求める。"""
        width = cls.conv_out(grid_width)
        height = cls.conv_out(grid_height)
        if token_mode == "hr":
            return width, height
        width = cls.conv_out(width)
        height = cls.conv_out(height)
        if token_mode == "current":
            return width, height
        if token_mode == "stronger":
            return cls.conv_out(width), cls.conv_out(height)
        raise ValueError(f"unknown token_mode: {token_mode}")

    @classmethod
    def token_count(cls, token_mode: str) -> int:
        """Transformer cost proxy で使う token 数 N を返す。"""
        width, height = cls.token_dims(token_mode)
        return width * height

    @classmethod
    def context_token_count(cls, params: TrialParams) -> int:
        """TrialContext に保存する token 数を TrialParams から求める。"""
        return cls.token_count(params.token_mode)

    @classmethod
    def cost_tf(cls, params: TrialParams, k: float) -> float:
        """Transformer の N/M/L 支配項を見るための事前 cost proxy。"""
        n = cls.token_count(params.token_mode)
        m = params.d_model
        l = params.transformer_layers
        return float(l * ((n * n * m) + (k * n * m * m)))

    @staticmethod
    def config_include_line(path_text: str) -> str:
        """runner config DSL の include 行を生成する。"""
        path = Path(path_text)
        if path.is_absolute():
            return f"$include \"{path.resolve().as_posix()}\""
        normalized = path_text.replace("\\", "/")
        return f"$include <{normalized}>"

    @staticmethod
    def generated_structure(params: TrialParams) -> str:
        """DropMerge Optuna branch の block 接続を TrialParams から生成する。"""
        blocks = [
            "OptConvInit",
            f"OptResBlock(*{params.res_blocks})",
        ]
        if params.token_mode in ("current", "stronger"):
            blocks.extend(["OptConvDown", "SiLU"])
        if params.token_mode == "stronger":
            blocks.extend(["OptConvDown2", "SiLU"])
        blocks.extend([
            "OptViTProj",
            "SiLU",
            "PosEmbed2D",
            "OptTransEnc",
            "Flatten",
            "OptLinear",
            "SiLU",
        ])
        return " > ".join(blocks)

    @classmethod
    def render_config(cls, params: TrialParams, ctx, args) -> str:
        """DropMerge trial 用 config を生成する唯一の入口。"""
        ff_dim = params.d_model * params.ff_mult
        nhead = args.nhead
        if params.d_model % nhead != 0:
            raise ValueError(f"d_model must be divisible by nhead: d_model={params.d_model} nhead={nhead}")

        lines = [
            "# Generated by apps/runner/tools/dropmerge_optuna.py",
            f"# study={ctx.study_name} budget={ctx.budget_name} trial={ctx.trial_number}",
            cls.config_include_line(args.base_config),
            cls.config_include_line(args.workspace_config),
            cls.config_include_line(args.extra_config),
            "",
            "app.$ = app.batchrun > P",
            f"app.run_name = {ctx.run_name}",
            f"app.runs_dir = {ctx.runs_dir}",
            f"app.batchrun.exp_exit_step = {args.exp_exit_step}",
            f"train.seed = {args.seed}",
            "",
            "net.block.[OptConvInit].type = Conv2d",
            f"net.block.[OptConvInit].conv.out_channels = {params.cnn_channels}",
            "net.block.[OptConvInit].conv.kernel_size = 3",
            "net.block.[OptConvInit].conv.padding = 1",
            "net.block.[OptConvInit].conv.stride = 2",
            "net.block.[OptConvInit].init.mode = he",
            "",
            "net.block.[OptResBlock].type = ResBlock",
            f"net.block.[OptResBlock].res.channels = {params.cnn_channels}",
            "net.block.[OptResBlock].res.kernel_size = 3",
            "net.block.[OptResBlock].res.activation = silu",
            "net.block.[OptResBlock].res.activation_mode = pre",
            "",
            "net.block.[OptConvDown].type = Conv2d",
            f"net.block.[OptConvDown].conv.out_channels = {params.cnn_channels}",
            "net.block.[OptConvDown].conv.kernel_size = 3",
            "net.block.[OptConvDown].conv.stride = 2",
            "net.block.[OptConvDown].conv.padding = 1",
            "",
            "net.block.[OptConvDown2].type = Conv2d",
            f"net.block.[OptConvDown2].conv.out_channels = {params.cnn_channels}",
            "net.block.[OptConvDown2].conv.kernel_size = 3",
            "net.block.[OptConvDown2].conv.stride = 2",
            "net.block.[OptConvDown2].conv.padding = 1",
            "",
            "net.block.[OptViTProj].type = Conv2d",
            f"net.block.[OptViTProj].conv.out_channels = {params.d_model}",
            "net.block.[OptViTProj].conv.kernel_size = 1",
            "net.block.[OptViTProj].conv.stride = 1",
            "net.block.[OptViTProj].conv.padding = 0",
            "",
            "net.block.[OptTransEnc].type = TransformerEncoder",
            f"net.block.[OptTransEnc].tf.d_model = {params.d_model}",
            f"net.block.[OptTransEnc].tf.nhead = {nhead}",
            f"net.block.[OptTransEnc].tf.num_layers = {params.transformer_layers}",
            f"net.block.[OptTransEnc].tf.dim_feedforward = {ff_dim}",
            "net.block.[OptTransEnc].tf.norm_first = true",
            "net.block.[OptTransEnc].tf.use_sdpa = true",
            "net.block.[OptTransEnc].tf.activation = gelu",
            "",
            "net.block.[OptLinear].type = Linear",
            f"net.block.[OptLinear].linear.out_features = {params.trunk_width}",
            "net.block.[OptLinear].linear.bias = true",
            "net.block.[OptLinear].init.mode = orthogonal",
            "net.block.[OptLinear].init.manual_gain = 1.0",
            "",
            "net.block.[OptHeadFC].type = Linear",
            f"net.block.[OptHeadFC].linear.out_features = {params.head_width}",
            "net.block.[OptHeadFC].init.mode = xavier",
            "",
            "net.branch.[main_feature].$ = net.branch.OptunaDropMerge",
            "net.branch.OptunaDropMerge.bind = grid, vector_feature",
            f"net.branch.OptunaDropMerge.structure = {cls.generated_structure(params)}",
            "net.branch.[value_stream].structure = OptHeadFC > SiLU",
            "net.branch.[adv_stream].structure = OptHeadFC > SiLU",
            "",
        ]
        return "\n".join(lines)


# DropMerge 固有の探索定義はこの entrypoint 内に集約する。
DOMAIN = DropMergeDomain
PRIMARY_TAGS = DOMAIN.PRIMARY_TAGS
SUPPLEMENTAL_TAGS = DOMAIN.SUPPLEMENTAL_TAGS
TRIAL_PARAM_CHOICES = DOMAIN.TRIAL_PARAM_CHOICES
LATE_SCORE_WINDOWS = DOMAIN.LATE_SCORE_WINDOWS
METRICS_SPEC = MetricsSpec(PRIMARY_TAGS, SUPPLEMENTAL_TAGS, LATE_SCORE_WINDOWS)

BUDGETS = {
    "small": 35_000_000.0,
    "medium": 70_000_000.0,
}

SCORE_AGGREGATES = ("mean", "median", "mean-minus-std", "min")
DUPLICATE_PARAMS_POLICIES = ("allow", "prune", "reseed")
SUMMARY_OBJECTIVE_NAMES = ("mean", "range", "std")
SUMMARY_OBJECTIVE_DIRECTIONS = ("maximize", "minimize", "minimize")
GROUP_SUMMARY_ARTIFACT_FILENAME = "group_summary.json"

RUNTIME = OptunaHarnessRuntime(
    domain=DOMAIN,
    harness_name="dropmerge_optuna",
    script_path=__file__,
    budget_presets=BUDGETS,
    metrics_spec=METRICS_SPEC,
    summary_objective_names=SUMMARY_OBJECTIVE_NAMES,
    summary_objective_directions=SUMMARY_OBJECTIVE_DIRECTIONS,
    group_summary_artifact_filename=GROUP_SUMMARY_ARTIFACT_FILENAME,
)


def repo_root_from_script() -> Path:
    return RUNTIME.repo_root_from_script()


def add_common_args(parser: argparse.ArgumentParser, include_seed: bool = True) -> None:
    repo_root = repo_root_from_script()
    parser.add_argument("--repo-root", default=str(repo_root), help="anet-lab のリポジトリルート。")
    parser.add_argument(
        "--workspace",
        default="_default",
        help="探索入出力を束ねる workspace path。相対時は runner/workspaces 基準。",
    )
    parser.add_argument(
        "--base-config",
        default=str(default_main_config(repo_root)),
        help="trial config が最初に $include する基準 main config。",
    )
    parser.add_argument(
        "--extra-config",
        default="DropMerge_optuna.txt",
        help="生成 trial config が base config の後に $include する Optuna 専用 config。",
    )
    parser.add_argument("--study-name", required=True, help="Optuna study 名。")
    parser.add_argument("--budget", choices=sorted(BUDGETS), default="small", help="使う cost_budget プリセット。")
    parser.add_argument("--cost-budget", type=float, help="cost_budget を明示指定する。指定時は --budget の値を上書きする。")
    parser.add_argument("--cost-k", type=float, default=4.0, help="cost_tf の N*M^2 項に掛ける係数。")
    parser.add_argument("--exp-exit-step", type=int, default=1_000_000, help="proxy trial の app.batchrun.exp_exit_step。%% window の基準 step。")
    if include_seed:
        parser.add_argument("--seed", type=int, default=12345, help="train.seed に使う seed。")
    parser.add_argument("--nhead", type=int, default=8, help="Transformer の attention head 数。")


def add_trial_identity_args(parser: argparse.ArgumentParser, *, include_trial_number: bool = True) -> None:
    trial_name_help = (
        "trial 名。未指定時は既存出力から t00000 形式で自動採番する。"
        if include_trial_number
        else "trial 名。未指定時は Optuna trial number から tNNNNN 形式で決める。"
    )
    parser.add_argument("--trial-name", help=trial_name_help)
    if include_trial_number:
        parser.add_argument("--trial-number", type=int, help="trial_name 未指定時に使う trial 番号。未指定時は既存出力から自動採番する。")


def add_optuna_storage_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--storage",
        help="Optuna SQLite DB の URL またはパス。省略時は workspace/optuna/optuna.db。",
    )
    parser.add_argument(
        "--storage-timeout-sec",
        type=float,
        default=120.0,
        help="SQLite storage の lock 待ち timeout 秒。",
    )


def add_optuna_artifact_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--optuna-artifact-dir",
        help="Optuna Dashboard 用 artifact store。省略時は workspace/optuna/artifacts。",
    )




def build_parser() -> argparse.ArgumentParser:
    usage = (
        "%(prog)s <command> [options]\n\n"
        "例:\n"
        "  %(prog)s dry-run --study-name dropmergeSmall --budget small\n"
        "  %(prog)s run-trial --study-name dropmergeSmall --trial-name t00001\n"
        "  %(prog)s summarize apps/runner/workspaces/dm_opt/runs/dropmergeSmall_t00001/metrics.jsonl --window-start 80%% --window-end 100%%\n"
        "  %(prog)s summarize-study --source-study-name dropmergeSmall\n"
        "  %(prog)s cleanup-running --study-name dropmergeSmall --dry-run\n"
        "  %(prog)s run-study --study-name dropmergeSmall --budget small --n-trials 10 --seeds 12345,23456"
    )
    parser = JapaneseArgumentParser(
        usage=usage,
        description=(
            "Optuna harness: trial config 生成、runner --config 起動、metrics.jsonl 採点を担当します。\n"
            f"{DOMAIN.HELP_SUMMARY}"
        ),
        formatter_class=JapaneseHelpFormatter,
        add_help=False,
    )
    localize_parser(parser, "コマンド")
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="command",
        title="コマンド",
        description="利用可能なコマンド",
        parser_class=JapaneseArgumentParser,
    )

    dry_run = subparsers.add_parser(
        "dry-run",
        help="trial config と manifest だけ生成する",
        description=DOMAIN.DRY_RUN_DESCRIPTION,
        formatter_class=JapaneseHelpFormatter,
        add_help=False,
    )
    localize_parser(dry_run)
    add_common_args(dry_run)
    add_trial_identity_args(dry_run)
    DOMAIN.add_param_args(dry_run)
    add_score_window_args(dry_run)
    dry_run.set_defaults(func=RUNTIME.command_dry_run)

    summarize = subparsers.add_parser(
        "summarize",
        help="metrics.jsonl を採点する",
        description="既存の metrics.jsonl から固定 step window の score と補助指標を抽出します。",
        formatter_class=JapaneseHelpFormatter,
        add_help=False,
    )
    localize_parser(summarize)
    summarize.add_argument("metrics_jsonl", help="採点対象の metrics.jsonl。")
    summarize.add_argument("--exp-exit-step", type=int, default=1_000_000, help="負数 window と %% window の基準 step。")
    add_score_window_args(summarize, primary_score=False)
    summarize.add_argument("--output-dir", help="metrics_summary.json/csv の出力先。")
    summarize.set_defaults(func=RUNTIME.command_summarize)

    summarize_study = subparsers.add_parser(
        "summarize-study",
        help="同一 params group を Dashboard 用 Study に集約する",
        description=DOMAIN.SUMMARIZE_STUDY_DESCRIPTION,
        formatter_class=JapaneseHelpFormatter,
        add_help=False,
    )
    localize_parser(summarize_study)
    repo_root = repo_root_from_script()
    summarize_study.add_argument("--repo-root", default=str(repo_root), help="anet-lab のリポジトリルート。")
    summarize_study.add_argument("--workspace", default="_default", help="source 省略時に使う workspace path。")
    summarize_study.add_argument("--source-study-name", required=True, help="集約元の Optuna study 名。")
    summarize_study.add_argument(
        "--target-study-name",
        help="集約先の Optuna study 名。未指定時は <source-study-name>_summary。",
    )
    summarize_study.add_argument(
        "--source-storage",
        help="集約元 Optuna SQLite DB。省略時は workspace/optuna/optuna.db。",
    )
    summarize_study.add_argument(
        "--target-storage",
        help="集約先 Optuna SQLite DB の URL またはパス。未指定時は source-storage と同じ。",
    )
    summarize_study.add_argument(
        "--source-artifact-dir",
        help="集約元 artifact store。省略時は workspace/optuna/artifacts。",
    )
    summarize_study.add_argument(
        "--target-artifact-dir",
        help="集約先 artifact store。省略時は source-artifact-dir を継承。",
    )
    summarize_study.add_argument(
        "--storage-timeout-sec",
        type=float,
        default=120.0,
        help="SQLite storage の lock 待ち timeout 秒。",
    )
    summarize_study.add_argument(
        "--overwrite-target-study",
        action="store_true",
        default=False,
        help="既存 target study を削除して作り直す。同一 storage の source study 自体は指定できない。",
    )
    summarize_study.set_defaults(func=RUNTIME.command_summarize_study)

    cleanup = subparsers.add_parser(
        "cleanup-running",
        help="Study に残った RUNNING trial を FAIL にする",
        description="中断などで RUNNING のまま残った Optuna trial を指定 study 内で FAIL に変更します。",
        formatter_class=JapaneseHelpFormatter,
        add_help=False,
    )
    localize_parser(cleanup)
    repo_root = repo_root_from_script()
    cleanup.add_argument("--repo-root", default=str(repo_root), help="anet-lab のリポジトリルート。")
    cleanup.add_argument("--workspace", default="_default", help="storage 省略時に使う workspace path。")
    cleanup.add_argument("--study-name", required=True, help="Optuna study 名。")
    add_optuna_storage_args(cleanup)
    cleanup.add_argument("--dry-run", action="store_true", default=False, help="対象 RUNNING trial を表示するだけで DB を変更しない。")
    cleanup.set_defaults(func=RUNTIME.command_cleanup_running)

    run_trial = subparsers.add_parser(
        "run-trial",
        help="固定 params の Optuna trial を 1 件実行する",
        description=DOMAIN.RUN_TRIAL_DESCRIPTION,
        formatter_class=JapaneseHelpFormatter,
        add_help=False,
    )
    localize_parser(run_trial)
    add_common_args(run_trial, include_seed=False)
    add_trial_identity_args(run_trial, include_trial_number=False)
    DOMAIN.add_param_args(run_trial)
    add_runner_args(run_trial)
    add_score_window_args(run_trial)
    add_optuna_storage_args(run_trial)
    add_optuna_artifact_args(run_trial)
    run_trial.add_argument("--seeds", default="12345", help="同一 params を評価する train.seed の comma-separated list。")
    run_trial.add_argument(
        "--score-aggregate",
        choices=SCORE_AGGREGATES,
        default="mean",
        help="multi-seed score を Optuna trial value に集約する方法。",
    )
    run_trial.set_defaults(func=RUNTIME.command_run_trial)

    run_study = subparsers.add_parser(
        "run-study",
        help="Optuna study を実行する",
        description=DOMAIN.RUN_STUDY_DESCRIPTION,
        formatter_class=JapaneseHelpFormatter,
        add_help=False,
    )
    localize_parser(run_study)
    add_common_args(run_study, include_seed=False)
    DOMAIN.add_param_args(run_study, default_values=False)
    add_runner_args(run_study)
    add_score_window_args(run_study)
    add_optuna_storage_args(run_study)
    run_study.add_argument(
        "--heartbeat-interval-sec",
        type=int,
        default=60,
        help="Optuna RDBStorage heartbeat interval 秒。0 は heartbeat を無効化する。",
    )
    run_study.add_argument(
        "--heartbeat-grace-period-sec",
        type=int,
        default=600,
        help="heartbeat が途絶えた RUNNING trial を stale とみなす猶予秒。",
    )
    add_optuna_artifact_args(run_study)
    run_study.add_argument(
        "--n-trials",
        type=int,
        help="この実行で追加する trial 数。tpe 未指定時は 10、grid 未指定時は未実行 combo 全件。",
    )
    run_study.add_argument("--n-jobs", type=int, default=1, help="Optuna の並列 worker 数。")
    run_study.add_argument("--study-note", help="Study User Attributes の note に保存する任意メモ。未指定時は既存 note を変更しない。")
    run_study.add_argument("--seeds", default="12345", help="同一 params を評価する train.seed の comma-separated list。")
    run_study.add_argument(
        "--score-aggregate",
        choices=SCORE_AGGREGATES,
        default="mean",
        help="multi-seed score を Optuna trial value に集約する方法。",
    )
    run_study.add_argument(
        "--search-mode",
        choices=("tpe", "grid"),
        default="tpe",
        help="探索方法。tpe は従来の TPE、grid は固定指定を反映した全組み合わせ探索。",
    )
    run_study.add_argument("--sampler-seed", type=int, help="TPE sampler の乱数 seed。grid では combo 列挙順の shuffle seed。未指定時は Optuna/通常順。")
    run_study.add_argument(
        "--constant-liar",
        action="store_true",
        default=False,
        help="TPESampler の constant_liar を有効にし、RUNNING trial 近傍の再提案を避けやすくする。",
    )
    run_study.add_argument(
        "--duplicate-params-policy",
        choices=DUPLICATE_PARAMS_POLICIES,
        default="reseed",
        help="同一 NN params が再提案されたときの扱い。",
    )
    run_study.add_argument(
        "--duplicate-params-max-runs",
        type=int,
        default=3,
        help="同一 NN params を実行する最大回数。0 は制限なし。",
    )
    run_study.add_argument(
        "--duplicate-seed-stride",
        type=int,
        default=100_000,
        help="duplicate params を reseed するときに duplicate_index ごとに seed へ足す値。",
    )
    run_study.add_argument(
        "--n-startup-trials",
        type=int,
        default=10,
        help="TPE に切り替える前に random sampling する完了 trial 数。",
    )
    run_study.set_defaults(func=RUNTIME.command_run_study)

    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "timeout_sec", 0) == 0:
        args.timeout_sec = None
    try:
        if hasattr(args, "study_name"):
            RUNTIME.validate_name_part(args.study_name, "--study-name")
        if getattr(args, "trial_name", None):
            RUNTIME.validate_name_part(args.trial_name, "--trial-name")
        if args.command == "cleanup-running":
            repo_root = Path(args.repo_root).resolve()
            if args.storage is None:
                workspace = resolve_workspace_paths(repo_root, str(args.workspace), require_config=False)
                args.storage = str(workspace.optuna_dir / "optuna.db")
            storage_path = storage_path_from_text(repo_root, str(args.storage))
            if not storage_path.is_file():
                raise TrialExecutionError(f"Storage file does not exist or is not a regular file: path={storage_path}")
            args.storage = storage_url_from_text(repo_root, str(args.storage))
        return args.func(args)
    except TrialExecutionError as e:
        print(str(e), file=sys.stderr)
        return 2
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

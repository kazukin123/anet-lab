# DropMerge Optuna 利用ガイド

## 概要

DropMerge の NN 構成探索は、runner 本体ではなく Python harness
`apps/runner/tools/dropmerge_optuna.py` から実行する。

runner 側は `--config <path>` で trial 用 main config を受け取り、harness 側が trial ごとに
config 生成、runner 起動、`metrics.jsonl` 集計、summary 保存を行う。

runner の `--config` は、指定 path が作業ディレクトリ基準で見つからない場合、
runner project root の `config/` 配下を基準に解決する。
trial config 内の相対 `$include` は、まず include 元ファイルのディレクトリ基準で解決し、
見つからない場合だけ runner project root の `config/` 配下を試す。
Optuna harness が生成する trial config は、`--base-config` で指定した main config と
`--extra-config` で指定した Optuna 専用 config を順に `$include` し、その後ろに trial override を書く。
既定では `$include <_main.txt>` の直後に `$include <DropMerge_optuna.txt>` を置く。

v1 の対象は DropMerge 専用、`Flatten` 固定、探索対象は NN 構成のみ。
学習系ハイパラは baseline 固定とする。

用語の対応:

- study: Optuna が管理する探索単位。`study_name`、storage、trial 履歴を持つ。
- trial: study 内の 1 params 候補。`run-study` では Optuna が params を suggest し、multi-seed aggregate を trial value にする。
- seed run: `run-study` の 1 seed 分の runner 実行出力。`train.seed` だけを seed ごとに変える。
- run: runner の 1 実行出力。`run-trial` では `1 trial = 1 run`、`run-study` では `1 trial = 複数 seed run`。
- multiseed summary: `run-study` の Optuna objective。seed run の score を集約した JSON/CSV。
- trial artifact: harness が残す再現用ファイル。config、manifest、stdout/stderr、summary など。

プロセス構造:

- `run-study` は study 全体を 1 つの Python プロセスで実行する。
- trial ごとに別プロセス化されるのは runner だけ。
- `run-study` が `run-trial` を子プロセス起動する構成ではない。

## 出力レイアウト

既定の出力先は `apps/runner/runs_optuna/`。

```text
apps/runner/runs_optuna/
  optuna.db
  artifacts/                            # Optuna Dashboard 用 artifact store。内部構造には依存しない。
  <study_name>_<trial_name>/          # run-study の代表フォルダ。metrics.jsonl は持たない。
    trial/
      manifest.json
      multiseed_summary.json
      multiseed_summary.csv
      seed_runs.json
  <study_name>_<trial_name>_s<seed>/  # seed run。metrics-viewer の run になる。
    metrics.jsonl
    config.txt
    stdout.log
    stderr.log
    trial/
      config.txt
      manifest.json
      process.json
      metrics_summary.json
      metrics_summary.csv
      stdout.log
      stderr.log
```

`trial_name` は `t00000` 形式を基本にする。
`run-study` では Optuna trial number から生成し、`dry-run` / `run-trial` では既存の同一 study 出力から最大番号+1で自動採番する。
`run-study` の代表 `run_name` は `<study_name>_<trial_name>`。
seed run の `run_name` は `<study_name>_<trial_name>_s<seed>`。
`run-trial` は単発実行なので `<study_name>_<trial_name>` のまま。

例:

```text
apps/runner/runs_optuna/dropmergeSmall_t00001/trial/multiseed_summary.json
apps/runner/runs_optuna/dropmergeSmall_t00001_s12345/metrics.jsonl
apps/runner/runs_optuna/dropmergeSmall_t00001_s12345/trial/manifest.json
```

trial artifact には、主に次のファイルを保存する。

- `config.txt`: runner に渡した trial config。
- `manifest.json`: study / trial / run 名、params、cost、出力 path など。
- `stdout.log`, `stderr.log`: runner process の標準出力と標準エラー。
- `process.json`: runner の exit code、duration、timeout / interrupt 有無など。
- `metrics_summary.json`, `metrics_summary.csv`: seed run の score と補助指標の集計結果。JSON には raw window、実効 window、`exp_exit_step` も残す。
- `multiseed_summary.json`, `multiseed_summary.csv`: `run-study` の代表フォルダに置く seed 集約結果。
- `seed_runs.json`: `run-study` の代表フォルダに置く seed run 別の詳細。seed、run 名、status、score、path、error などを残す。

`runs_optuna/<run_name>/trial` は人間と harness が直接読む artifact 置き場である。
一方、`runs_optuna/artifacts` は Optuna の `FileSystemArtifactStore` が管理する Dashboard 用 artifact store であり、階層やファイル名に依存しない。
Dashboard で `Show Artifacts` を有効にするには、`run-study` が `upload_artifact()` で trial に artifact metadata を登録し、optuna-dashboard を `--artifact-dir artifacts` 付きで起動する必要がある。

metrics-viewer で Optuna run を横断表示する場合は、viewer の runs dir に
`apps/runner/runs_optuna` を指定する。
metrics-viewer は直下フォルダに `metrics.jsonl` があるものだけを run として認識するため、`run-study` の代表フォルダは表示されず、seed run だけが表示される。

`<run_name>` フォルダを削除しても `optuna.db` 内の trial レコードは消えない。
DB には params、score、user_attrs、保存済み path が残るため、削除後は `run_dir` や `artifact_dir` が stale path になる。

## コマンド概要

### 共通引数

`dry-run`、`run-trial`、`run-study` は次の共通引数を持つ。

- `--repo-root`: `anet-lab` のリポジトリルート。path 解決の基準。
- `--base-config`: trial config が最初に `$include` する main config。既定は `_main.txt`。
- `--extra-config`: base config の後に `$include` する Optuna 専用 config。既定は `DropMerge_optuna.txt`。
- `--study-name`: study 名。`run_name` の prefix になる。
- `--budget`: `small` / `medium` の `cost_budget` preset。
- `--cost-budget`: preset を使わず `cost_budget` を直接指定する。
- `--cost-k`: `cost_tf` の `N*M^2` 項に掛ける係数。
- `--runs-dir`: runner の `app.runs_dir` に渡す出力先。既定は `runs_optuna`。
- `--exp-exit-step`: proxy trial の `app.batchrun.exp_exit_step`。負数相対 window と `%` window の基準にもなる。
- `--nhead`: Transformer の attention head 数。

`dry-run` と `run-trial` は trial 1 件を固定 params で扱うため、
`--trial-name`、`--trial-number`、`--seed`、固定 NN params を持つ。
`run-study` は Optuna が trial を生成するため、これらの trial 固有引数を持たない。

`dry-run` / `run-trial` で `--trial-name` と `--trial-number` をどちらも省略した場合、
`runs_optuna` 直下の `<study_name>_tNNNNN` を見て次番号を決める。
既存番号の穴埋めはせず、最大番号+1を使う。

### `dry-run`

trial config と manifest を生成し、`cost_tf` と prune 判定を表示する。
runner は起動しない。

主な用途:

- config 生成確認
- `run_name` / `artifact_dir` / `run_dir` の確認
- `cost_budget` 超過判定の確認

主な引数:

- `--study-name`: study 名。`run_name` の prefix になる。
- `--trial-name`: trial 名。未指定時は既存出力から `tNNNNN` 形式で自動採番する。
- `--trial-number`: 明示した番号を `trial_number` に使う。未指定時は自動採番または `--trial-name tNNNNN` から復元する。
- `--budget`: `small` / `medium` の `cost_budget` preset。
- `--cost-budget`: preset を使わず直接 `cost_budget` を指定する。
- `--seed`: `train.seed` に使う seed。
- `--cnn-channels`, `--res-blocks`, `--token-mode`, `--d-model`, `--transformer-layers`, `--ff-mult`, `--trunk-width`, `--head-width`: 固定 NN params。

### `run-trial`

CLI で固定指定した NN params を 1 件だけ runner で実行し、`metrics.jsonl` を採点する。
Optuna DB には登録しない。

主な用途:

- smoke run
- 手動候補の再評価
- `run-study` 前の runner 起動確認

追加の主な引数:

- `--runner-exe`: `AnetRLRunner.exe` の path。相対 path は repo root 基準。既定は `apps/runner/bin/Release/AnetRLRunner.exe`。
- `--timeout-sec`: runner 1 trial の timeout 秒。`0` は timeout なし。
- `--seed`: `train.seed` に使う seed。
- `--window-start`, `--window-end`: primary score を集計する `exp_step` window。絶対 step、負数相対 step、`80%` のような `exp_exit_step` 比率を指定できる。既定は `80%` から `100%`。

`cost_tf > cost_budget` の場合、runner は起動せず exit code `2` で終了する。

### `run-study`

Optuna study を作成または再開し、Optuna が suggest した params で trial を複数実行する。
1 Optuna trial は 1 NN params 候補を表し、`--seeds` で指定した seed ごとに runner を逐次起動する。
Optuna の trial value には seed 別 score ではなく、`multiseed_summary.json` の aggregate score を返す。

`run-study` は study 全体のコマンドなので、`--trial-name` と固定 NN params は持たない。
trial 名は Optuna trial number から `t00000`, `t00001`, ... と自動生成される。

主な引数:

- `--study-name`: Optuna study 名。
- `--storage`: Optuna SQLite DB URL または path。既定は `sqlite:///runs_optuna/optuna.db`。相対時は runner project root 基準。
- `--storage-timeout-sec`: SQLite storage の lock 待ち timeout 秒。既定は `120.0`。
- `--optuna-artifact-dir`: Optuna Dashboard 用 artifact store の base path。既定は `runs_optuna/artifacts`。相対時は runner project root 基準。
- `--n-trials`: この実行で追加する trial 数。
- `--n-jobs`: Optuna の並列 worker 数。
- `--study-note`: Study User Attributes の `note` に保存する任意メモ。未指定時は既存 `note` を変更しない。
- `--sampler-seed`: Optuna sampler の乱数 seed。未指定時は Optuna 既定で、探索候補列は固定されない。
- `--n-startup-trials`: TPE に切り替える前に random sampling する完了 trial 数。既定は `10`。
- `--constant-liar`: `TPESampler` の `constant_liar` を有効にする。RUNNING trial 近傍の再提案を避ける補助策であり、完了済み duplicate の完全禁止ではない。
- `--seeds`: 同一 params を評価する `train.seed` の comma-separated list。既定は `12345`。
- `--score-aggregate`: seed 別 score を trial value に集約する方法。`mean` / `median` / `mean-minus-std` / `min`。既定は `mean`。
- `--duplicate-params-policy`: 同一 NN params が再提案されたときの扱い。`allow` / `prune` / `reseed`。既定は `reseed`。
- `--duplicate-params-max-runs`: 同一 NN params を実行する最大回数。既定は `3`。`0` は制限なし。
- `--duplicate-seed-stride`: `reseed` 時に `duplicate_index` ごとに seed へ足す値。既定は `100000`。
- `--runner-exe`: `AnetRLRunner.exe` の path。既定は `apps/runner/bin/Release/AnetRLRunner.exe`。
- `--budget`, `--cost-budget`, `--cost-k`: cost 制約。
- `--exp-exit-step`, `--nhead`: trial config に入る共通設定。
- backend deterministic や DropMergeEnv の `seed_mode` / `global_seed` は、生成 config ではなく `--extra-config` 側で管理する。
- `--window-start`, `--window-end`: primary score 集計 window。絶対 step、負数相対 step、`80%` のような `exp_exit_step` 比率を指定できる。既定は `80%` から `100%`。

`cost_tf > cost_budget` と duplicate max 超過は study 内では `PRUNED` として扱う。
runner failure、metrics missing、primary score unavailable、seed run の一部失敗は `FAIL` として扱い、TPE の通常学習材料に入れない。
Trial User Attributes の multi-seed 統計は `score_mean`、`score_std`、`score_min`、`score_max`、`score_range`、`seed_count`、`seed_success_count`、`seed_failure_count` に絞る。
seed run 別の詳細は `seed_runs.json` と Optuna artifact を参照する。
Study User Attributes には `last_*` として最後の `run-study` 起動条件を保存する。これは Dashboard 用のメモであり、study 全体の固定契約ではない。
optuna-dashboard では aggregate score が trial value として表示される。

duplicate 判定は同一 study の `COMPLETE` / `RUNNING` trial の NN params だけを見る。`PRUNED` / `FAIL` trial は数えない。
`allow` は現状互換で同じ seed list のまま実行する。
`prune` は duplicate があれば runner 起動前に prune する。
`reseed` は `duplicate_index = duplicate_count_before` とし、`effective_seed = base_seed + duplicate_index * duplicate_seed_stride` で seed list をずらす。
`duplicate-params-max-runs` は total run count の上限で、`3` なら初回、2周目、3周目まで許可し、4周目を prune する。

### `summarize`

既存の `metrics.jsonl` から score と補助指標を抽出する。

主な引数:

- `metrics_jsonl`: 採点対象。
- `--exp-exit-step`: 負数相対 window と `%` window の基準 step。
- `--window-start`, `--window-end`: 集計 window。絶対 step、負数相対 step、`80%` のような `exp_exit_step` 比率を指定できる。既定は `80%` から `100%`。
- `--output-dir`: `metrics_summary.json` / `metrics_summary.csv` の出力先。

primary score は次の 2 tag の window mean を同等に扱う。

```text
21_eval/03_target_reward_ema
21_eval/04_policy_reward_ema
```

### `cleanup-running`

中断や強制終了で Optuna DB に残った `RUNNING` trial を `FAIL` に変更する。
artifact は削除せず、DB state だけを直す。

主な引数:

- `--study-name`: cleanup 対象の Optuna study 名。
- `--storage`: Optuna SQLite DB URL または path。
- `--dry-run`: 対象 trial number を表示するだけで DB を変更しない。

`RUNNING` trial が残ると、`--constant-liar` や duplicate 判定で「まだ実行中の trial」として扱われる。
`Ctrl+C` 後に dashboard で `RUNNING` が残っている場合は、再開前に `cleanup-running --dry-run` で対象を確認し、問題なければ本実行する。

## Optuna Attributes 一覧

Optuna Dashboard では Study User Attributes と Trial User Attributes が見える。
ただし attr は多くなりやすいため、Study attrs は「最後に起動した条件のメモ」、Trial attrs は「個々の trial の実体」として読む。
同じ study 内に異なる前提条件の trial が混ざることは許容するため、比較や後処理では Study attrs ではなく Trial attrs、`manifest.json`、`multiseed_summary.json` を正とする。

### Study User Attributes

`run-study` 起動時に、次の `last_*` を毎回上書きする。
これは dashboard で直近の起動条件を見るためのメモであり、study 全体の固定契約ではない。

| Attribute | 更新契機 | 意味 |
| --- | --- | --- |
| `last_launch_at` | `run-study` | 最後に `run-study` を起動した UTC 時刻。 |
| `last_harness` | `run-study` | harness 名。現在は `dropmerge_optuna`。 |
| `last_command` | `run-study` | 起動コマンド種別。現在は `run-study`。 |
| `last_study_name` | `run-study` | 起動時に指定した study 名。 |
| `last_storage` | `run-study` | 起動時に使った Optuna storage URL。 |
| `last_storage_timeout_sec` | `run-study` | SQLite storage lock 待ち timeout 秒。 |
| `last_runs_dir` | `run-study` | runner 出力先として使った `runs_dir`。 |
| `last_budget` | `run-study` | `small` / `medium` などの budget preset 名。 |
| `last_cost_budget` | `run-study` | 実効 `cost_budget`。 |
| `last_cost_k` | `run-study` | `cost_tf` の `N*M^2` 項に掛けた係数。 |
| `last_exp_exit_step` | `run-study` | trial config に入れた `app.batchrun.exp_exit_step`。 |
| `last_window_start_raw` | `run-study` | CLI 指定の score window start。`80%` や `-200000` などの未解決値。 |
| `last_window_end_raw` | `run-study` | CLI 指定の score window end。`100%` などの未解決値。 |
| `last_window_start` | `run-study` | `exp_exit_step` 基準で解決した score window start step。 |
| `last_window_end` | `run-study` | `exp_exit_step` 基準で解決した score window end step。 |
| `last_seeds` | `run-study` | 起動時の base seed list。duplicate reseed 前の値。 |
| `last_seed_count` | `run-study` | 起動時の base seed 数。 |
| `last_score_aggregate` | `run-study` | seed 別 score から trial value を作る集約方法。 |
| `last_sampler_seed` | `run-study` | Optuna sampler seed。未指定時は `null`。 |
| `last_n_startup_trials` | `run-study` | TPE 前に random sampling する完了 trial 数。 |
| `last_constant_liar` | `run-study` | `TPESampler(constant_liar=...)` の指定値。 |
| `last_duplicate_params_policy` | `run-study` | duplicate params の扱い。`allow` / `prune` / `reseed`。 |
| `last_duplicate_params_max_runs` | `run-study` | 同一 NN params を実行する最大回数。`0` は制限なし。 |
| `last_duplicate_seed_stride` | `run-study` | `reseed` 時に duplicate index ごとに seed へ足す値。 |
| `last_n_trials` | `run-study` | この起動で追加しようとした trial 数。 |
| `last_n_jobs` | `run-study` | Optuna worker 並列数。 |
| `last_timeout_sec` | `run-study` | runner 1 run の timeout 秒。 |
| `last_runner_exe` | `run-study` | 起動時に使った runner executable path。 |
| `last_base_config` | `run-study` | 生成 config が最初に `$include` した base config。 |
| `last_extra_config` | `run-study` | 生成 config が base config の次に `$include` した Optuna 専用 config。 |
| `note` | `run-study --study-note` | 人間向けメモ。未指定時は既存値を変更しない。空文字指定時は空文字で上書きする。 |
| `cleaned_running_trials` | `cleanup-running` | `FAIL` に変更した `RUNNING` trial number の一覧。 |
| `cleaned_running_trials_at` | `cleanup-running` | cleanup 実行 UTC 時刻。 |

### Trial User Attributes

`run-study` の各 Optuna trial には、候補 params、cost、出力 path、multi-seed 集約、duplicate 判定の情報を保存する。

| Attribute | 更新契機 | 意味 |
| --- | --- | --- |
| `params` | trial 開始時 | Optuna が suggest した NN params 一式。 |
| `cost_tf` | trial 開始時 | `L * (N^2 * M + k * N * M^2)` で計算した Transformer cost proxy。 |
| `cost_budget` | trial 開始時 | この trial に適用した `cost_budget`。 |
| `token_count` | trial 開始時 | `token_mode` から推定した token 数 `N`。 |
| `trial_name` | trial 開始時 | `t00000` などの trial 名。 |
| `run_name` | trial 開始時 | 代表 run 名。`run-study` では `<study_name>_<trial_name>`。 |
| `run_dir` | trial 開始時 | 代表 run folder path。 |
| `artifact_dir` | trial 開始時 | harness artifact folder path。 |
| `config_path` | trial 開始時 | runner に渡した generated config path。 |
| `score` | aggregate 確定時 | Optuna trial value と同じ aggregate score。prune 時は `null` になり得る。 |
| `score_aggregate` | aggregate 確定時 | `score` の集約方法。`mean` / `median` / `mean-minus-std` / `min`。 |
| `score_mean` | aggregate 確定時 | seed 別 score の平均。 |
| `score_std` | aggregate 確定時 | seed 別 score の population standard deviation。 |
| `score_min` | aggregate 確定時 | seed 別 score の最小値。 |
| `score_max` | aggregate 確定時 | seed 別 score の最大値。 |
| `score_range` | aggregate 確定時 | `score_max - score_min`。score が無い場合は `null`、1 seed の場合は `0.0`。 |
| `seed_count` | aggregate 確定時 | 実行予定だった effective seed 数。 |
| `seed_success_count` | aggregate 確定時 | 完了した seed run 数。 |
| `seed_failure_count` | aggregate 確定時 | 失敗した seed run 数。 |
| `duplicate_params_policy` | aggregate 確定時 | この trial に適用した duplicate policy。 |
| `duplicate_count_before` | aggregate 確定時 | この trial 開始前に見つかった同一 NN params の `COMPLETE` / `RUNNING` trial 数。 |
| `duplicate_index` | aggregate 確定時 | 同一 NN params の実行 index。`0` が初回。 |
| `duplicate_params_max_runs` | aggregate 確定時 | 同一 NN params の最大実行回数。`0` は制限なし。 |
| `duplicate_seed_stride` | aggregate 確定時 | `reseed` 時の seed offset 幅。 |
| `base_seeds` | aggregate 確定時 | CLI の `--seeds` で指定した元 seed list。 |
| `effective_seeds` | aggregate 確定時 | 実際に使った seed list。`reseed` では duplicate index に応じてずれる。 |
| `duplicate_matched_trials` | aggregate 確定時 | duplicate 判定で一致した過去 trial number。 |

`run-study` の trial value は `score` と同じ aggregate score である。
`score_mean` などは seed 別 score から計算した補助値で、どれを trial value に使うかは `score_aggregate` で分かる。
`score_median`、`score_mean_minus_std`、seed 別 score / run 名などの詳細は Trial User Attributes には入れず、`multiseed_summary.json` と `seed_runs.json` を正として読む。

`run-trial` に Optuna trial を渡す内部経路では、単発 run の補助 attr として次も保存する。
通常の `run-trial` CLI は Optuna DB に登録しないため、主に実装上の共通経路用である。

| Attribute | 更新契機 | 意味 |
| --- | --- | --- |
| `returncode` | runner 終了時 | runner process の exit code。 |
| `metric:<tag>:mean` | metrics 集計時 | 指定 window 内の scalar tag 平均。 |
| `metric:<tag>:last` | metrics 集計時 | 指定 window 内の scalar tag 最終値。 |

### 同じ params の見分け方

Dashboard 上で同じ NN params の再評価を見分けたい場合は、まず `params` または `trials_dataframe()` が作る `params_*` 列で grouping する。
そのうえで、duplicate 系 attr を読む。

- `duplicate_index`: 同一 params の何回目の実行か。`0` が初回、`1` が 2 回目。
- `duplicate_count_before`: この trial の開始前に、同一 params の `COMPLETE` / `RUNNING` trial がいくつあったか。
- `duplicate_matched_trials`: duplicate 判定で一致した過去 trial number。
- `base_seeds`: CLI の `--seeds` で指定した元 seed。
- `effective_seeds`: 実際に使った seed。`--duplicate-params-policy reseed` では `duplicate_index * duplicate_seed_stride` だけずれる。
- `score_std` / `score_range`: seed 違いのばらつき。

seed ごとの score、run folder 名、path、error は Trial User Attributes ではなく、代表フォルダの `trial/seed_runs.json` または Dashboard の Artifacts から確認する。

CSV で見る場合は、Optuna の `trials_dataframe()` を使うと `params_*` と `user_attrs_*` の列が生成される。

```python
import optuna

study = optuna.load_study(
    study_name="dropmergeSmall",
    storage="sqlite:///runs_optuna/optuna.db",
)

df = study.trials_dataframe()
df.to_csv("optuna_trials.csv", index=False)
```

同じ params の再評価だけを追いたい場合は、`params_*` 列を同一キーとして group 化し、`user_attrs_duplicate_index`、`user_attrs_effective_seeds`、`user_attrs_score` を並べると分かりやすい。

## Score 算出基準

score は `metrics.jsonl` の scalar record から次の手順で算出する。

1. `type=scalar` の record だけを読む。
2. `window_start <= step <= window_end` の record だけを対象にする。window は両端を含む。
3. `--window-start` / `--window-end` は、絶対 step、負数相対 step、`80%` のような `exp_exit_step` 比率を受け付ける。
4. `%` 指定は `round(exp_exit_step * percent / 100)` に解決し、負数指定は `exp_exit_step + value` に解決する。
5. tag ごとに finite な `value` だけを算術平均し、`count`、`mean`、`last`、`min_step`、`max_step` を出す。
6. primary 2 tag の `mean` が両方そろった場合だけ、次式で score を出す。

```text
score = mean(
  mean(21_eval/03_target_reward_ema),
  mean(21_eval/04_policy_reward_ema)
)
```

どちらかの primary tag が window 内に存在しない、または finite な値を持たない場合、`score` は `null` になる。
`run-trial` では非 0 終了、`run-study` では `FAIL` として扱う。

補助 tag、duration、step/sec、終端理由、`max_rank`、`fruit_count` は summary には保存するが、v1 の score には入れない。
また、primary tag の `last` ではなく window 内の `mean` を使うため、短い window ではログ間隔や評価回数の影響を受けやすい。

`run-study` では、上記の単発 score を seed ごとに計算したうえで、`--score-aggregate` に従って trial value を決める。

```text
mean           = average(seed score list)
median         = median(seed score list)
mean-minus-std = average(seed score list) - population_stddev(seed score list)
min            = min(seed score list)
```

seed の一部が失敗した場合、その Optuna trial は aggregate score を採用せず `FAIL` にする。

## 利用手順

### 1. runner をビルドする

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Release --target AnetRLRunner'
```

### 2. ヘルプを確認する

```powershell
python apps\runner\tools\dropmerge_optuna.py --help
python apps\runner\tools\dropmerge_optuna.py dry-run --help
python apps\runner\tools\dropmerge_optuna.py run-trial --help
python apps\runner\tools\dropmerge_optuna.py run-study --help
```

### 3. dry-run で config 生成を確認する

```powershell
python apps\runner\tools\dropmerge_optuna.py dry-run --study-name dropmergeSmall --trial-name t00001 --budget small
```

確認点:

- `run_name` が `dropmergeSmall_t00001`。
- `runs_dir` が `runs_optuna`。
- `run_dir` が `apps/runner/runs_optuna/dropmergeSmall_t00001`。
- `artifact_dir` が `apps/runner/runs_optuna/dropmergeSmall_t00001/trial`。
- `pruned_by_cost` が期待どおり。

`--trial-name` を省略すると、同一 `study_name` の既存 `<study_name>_tNNNNN` 出力から次番号が採番される。

### 4. run-trial で smoke run する

短い step で runner 起動、metrics 生成、summary 生成まで確認する。

```powershell
python apps\runner\tools\dropmerge_optuna.py run-trial --study-name dropmergeSmoke --trial-name t00000 --budget small --exp-exit-step 2000 --window-start 0 --window-end 100% --timeout-sec 600
```

確認点:

- `apps/runner/runs_optuna/dropmergeSmoke_t00000/metrics.jsonl` がある。
- `apps/runner/runs_optuna/dropmergeSmoke_t00000/trial/manifest.json` がある。
- `apps/runner/runs_optuna/dropmergeSmoke_t00000/trial/metrics_summary.json` の `score` が `null` ではない。

cost prune だけ確認する場合:

```powershell
python apps\runner\tools\dropmerge_optuna.py run-trial --study-name dropmergeSmoke --trial-name tCost --cost-budget 1
```

この場合 runner は起動せず、exit code `2` になる。

### 5. run-study で探索する

small study:

```powershell
python apps\runner\tools\dropmerge_optuna.py run-study --study-name dropmergeSmall --budget small --n-trials 20 --n-jobs 1 --seeds 12345,23456,34567 --score-aggregate mean-minus-std --duplicate-params-policy reseed --duplicate-params-max-runs 3 --exp-exit-step 1000000
```

medium study:

```powershell
python apps\runner\tools\dropmerge_optuna.py run-study --study-name dropmergeMedium --budget medium --n-trials 20 --n-jobs 1 --seeds 12345,23456,34567 --score-aggregate mean-minus-std --exp-exit-step 1000000
```

`--n-jobs > 1` は同一 GPU 上で runner が並列起動する。duration や step/sec は干渉を受けるため、score には使わず補助指標として読む。
multi-seed は 1 Optuna trial の内部で seed を逐次実行する。`--n-jobs > 1` の場合は、複数 params 候補が並列に進む。
SQLite storage で `--n-jobs > 1` を使うと、Optuna の trial / user attrs 書き込みが競合し `database is locked` になることがある。
既定では `--storage-timeout-sec 120.0` を設定して短い lock 競合を待つ。
それでも再発する場合は `--n-jobs 1` にするか、SQLite ではなく PostgreSQL / MySQL などの RDB storage を使う。

同じ `--study-name` と `--storage` で再実行すると study は再開され、trial が追加される。
同じ study を再現性重視で回す場合は、`--sampler-seed` を固定し、`--n-startup-trials` も明示しておく。
Debug runner を使う場合は、従来どおり `--runner-exe apps/runner/bin/Debug/AnetRLRunner.exe` で上書きする。

既定では `exp_exit_step` の 80% から 100% までを採点する。
終了直前の固定 step 幅だけを採点したい場合は、例えば `--window-start -200000` とし、`--window-end` は省略する。
この場合、`exp_exit_step - 200000` から `exp_exit_step` までを集計する。

seed 固定・決定論設定で duplicate params が完全に無駄になる場合は、次のように prune する。

```powershell
python apps\runner\tools\dropmerge_optuna.py run-study --study-name dropmergeFixed --budget small --n-trials 20 --seeds 1,2,3 --duplicate-params-policy prune
```

有望 params を何度も seed を変えて再評価したい場合は、既定の `reseed` を使う。無制限に再評価を許す場合は `--duplicate-params-max-runs 0` を指定する。

探索を中断するときは、まず `Ctrl+C` を 1 回だけ押す。
通常は harness が実行中 runner を止め、同一 study の `RUNNING` trial を `FAIL` に変更する。
`Ctrl+C` 連打、ターミナルごと終了、OS kill では cleanup が走らない場合がある。
その場合は次の手順で後処理する。

```powershell
python apps\runner\tools\dropmerge_optuna.py cleanup-running --study-name dropmergeSmall --dry-run
python apps\runner\tools\dropmerge_optuna.py cleanup-running --study-name dropmergeSmall
```

### 6. metrics-viewer で見る

metrics-viewer の runs dir に次を指定する。

```text
apps/runner/runs_optuna
```

viewer 側では、この直下の `<run_name>/metrics.jsonl` が run として扱われる。
`run-study` の代表フォルダ `<study_name>_<trial_name>` は `metrics.jsonl` を持たないため表示対象外になり、`<study_name>_<trial_name>_s<seed>` の seed run だけが表示される。
study をまたいだ比較をしたい場合も、`dropmergeSmall_t00000`、`dropmergeMedium_t00000` のように同じ viewer root で横断表示する。

### 7. 既存 metrics を再集計する

```powershell
python apps\runner\tools\dropmerge_optuna.py summarize apps/runner/runs_optuna/dropmergeSmall_t00000/metrics.jsonl --window-start 80% --window-end 100% --output-dir apps/runner/runs_optuna/dropmergeSmall_t00000/trial
```

相対 window で再集計する例:

```powershell
python apps\runner\tools\dropmerge_optuna.py summarize apps/runner/runs_optuna/dropmergeSmall_t00000/metrics.jsonl --exp-exit-step 1000000 --window-start -200000 --output-dir apps/runner/runs_optuna/dropmergeSmall_t00000/trial
```

## 探索対象と制約

探索対象:

- `C`: `--cnn-channels`。CNN/ResBlock の channel 数。既定探索では `48` / `64`。
- `D`: `--res-blocks`。ResBlock の繰り返し数。既定探索では `2` / `4`。
- `N`: `--token-mode` から決まる token 数。DropMerge G5846 の grid を前提に、stride 2 convolution の回数から概算する。
  - `current`: `ConvInit` + `ConvDown`。
  - `stronger`: `ConvInit` + `ConvDown` + `ConvDown2`。
- `M`: `--d-model`。Transformer の hidden width。既定探索では `96` / `128` / `192`。
- `L`: `--transformer-layers`。Transformer 層数。既定探索では `2` / `4`。
- `ff_mult`: `--ff-mult`。`dim_feedforward = d_model * ff_mult`。既定探索では `2` / `4`。
- `H`: `--trunk-width`, `--head-width`。Flatten 後 trunk と value/adv stream の Linear 幅。既定探索では trunk `1024` / `2048`、head `512` / `1024`。

`P=Flatten` は固定し、`GAP1D` / `CLS` / pooling family は v1 では探索しない。
学習系ハイパラ、環境設定、報酬スケーラ、TBO、PER、UQE、replay ratio、batch size は baseline または `DropMerge_optuna.txt` 側で固定する。

探索対象のうち、実行前 prune に使う主要制約は Transformer cost proxy の `cost_tf` である。
これはパラメータ数そのものではなく、実時間や GPU 負荷に効きやすい `N` / `M` / `L` の傾向を見るための粗い proxy として扱う。

cost proxy:

```text
cost_tf = L * (N^2 * M + k * N * M^2)
```

- `N^2 * M`: self-attention の token-token 相互作用を表す項。token 数 `N` が増えると二乗で効く。
- `N * M^2`: Q/K/V/O projection、FFN、channel mixing 系をまとめて見る項。`d_model` `M` が増えると二乗で効く。
- `k`: `N * M^2` 項に掛ける経験的な重み。既定は `--cost-k 4.0`。

`cost_tf > cost_budget` の trial は runner 起動前に prune する。
`d_model % nhead != 0` のように Transformer 設定として不正な組み合わせも、config 生成前に失敗させる。

`ff_mult` と `H` は探索対象だが、v1 の `cost_tf` には直接入れない。
そのため `ff_mult=4` や trunk/head 幅の増加による実時間差は、score summary の duration / step/sec など補助指標で後から見る。
必要になった場合は、将来 `k_proj` / `k_ff` のように projection 系と FFN 系を分けた proxy へ拡張する。

既定 budget:

- `small`: `35,000,000`
- `medium`: `70,000,000`

## 運用メモ

- `run-study` は探索用、`run-trial` は固定候補の smoke / 再評価用。
- `run-trial` は Optuna DB に登録しない。
- `run-study` では trial 名を明示指定しない。Optuna trial number から自動生成し、同一 trial 内の seed run は `_s<seed>` suffix を付ける。
- `run-study` は `TPESampler` を明示的に使う。最初の `--n-startup-trials` 件は random sampling、その後は過去の完了 trial に基づいて候補を寄せる。
- `--constant-liar` は RUNNING trial 近傍の再提案を避ける補助策。完了済み duplicate params の扱いは `--duplicate-params-policy` で制御する。
- `run-study` の探索単位は seed run ではなく multi-seed aggregate。DB や optuna-dashboard で見る trial value は aggregate score。
- 中断時はまず `Ctrl+C` を 1 回だけ押す。`RUNNING` が残った場合は `cleanup-running --dry-run` で確認してから cleanup する。
- Study User Attributes の `last_*` は最後の `run-study` 起動条件を表す。異なる前提の trial が同一 study に混ざることは許容し、各 trial の正確な条件は Trial User Attributes、manifest、summary を正として読む。
- 既定では duplicate params は `reseed` され、最大 3 回まで seed を変えて再評価される。
- 手動の `dry-run` / `run-trial` では、`--trial-name` 未指定時に既存出力から最大番号+1で採番する。並列手動実行で衝突が困る場合は `--trial-name` か `--trial-number` を明示する。
- `study_name` と `trial_name` に path separator は使えない。
- 既存の `runs/` とは混ぜない。Optuna 関連は `apps/runner/runs_optuna/` 配下へ集約する。
- 既存 `runs_optina/` 生成物の migration はしない。

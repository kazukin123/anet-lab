# DropMerge Optuna 対応 仕様書

## 目的

DropMerge の ResNet + ViT 系 NN 構成について、個人環境で回せる範囲の Optuna 探索を導入する。

汎用ハイパラ探索基盤ではなく、DropMerge の NN 構成だけを対象にする。学習系ハイパラを固定し、`cost_budget` が小さい proxy 探索を複数 budget で回して、`N/M/L/C/D/H` の効き方を読むことを目的にする。

## 背景

現行 DropMerge は `SuikaHybrid_Flatten_128_pre2` 系の `ConvInit64 -> ResBlock64p(*4) -> ConvDown64 -> ViT_Proj_128 -> TransEnc_128 -> Flatten -> Linear_2048` を中心に調整している。

パラメータ数では `Flatten -> Linear_2048` が大きいが、探索の実時間・GPU負荷に効く支配項は Transformer の token 数 `N`、`d_model` `M`、層数 `L` である。したがって v1 ではパラメータ数を主制約にせず、次の proxy を `cost_budget` として使う。

```text
cost_tf = L * (N^2 * M + k * N * M^2)
```

`cost_budget` は実時間そのものではなく、Transformer 計算量の事前 proxy である。trial の duration や step/sec は同一 GPU 並列の干渉を受けるため、v1 の primary score には入れない。

## 対象範囲

- 対象 env は DropMerge のみ。
- 対象 network family は `Flatten` 固定の ResNet + ViT hybrid。
- 探索対象は NN 構成のみ。
  - `C`: CNN channel 数。
  - `D`: ResBlock 数。
  - `N`: token 解像度。stride 構成から算出する。
  - `M`: `d_model`。
  - `L`: Transformer 層数。
  - `ff_mult`: Transformer FFN の倍率。
  - `H`: trunk/head FC 幅。
- 学習系ハイパラ、報酬スケーラ、TBO、PER、UQE、replay ratio、batch size は baseline 固定。
- proxy 探索で backend deterministic や DropMergeEnv seed 固定を使う場合は、生成 config ではなく Optuna 専用 config 側で指定する。

## 非対象範囲

- Optuna を runner 本体へ組み込まない。
- v1 では汎用 env / 汎用 metric 探索にはしない。
- `GAP1D` / `CLS` / pooling family は探索しない。
- 実行中 metrics tail による早期停止はしない。
- primary score に duration、step/sec、終端理由ペナルティは入れない。
- ADR は作らない。外部 Optuna harness と `--config` 差し替えが長期方針として固定された場合に改めて検討する。

## Runner 設定差し替え

runner は既定で `apps/runner/config/_main.txt` を読む。Optuna trial では main config 自体を差し替えるため、runner に `--config <path>` を追加する。

trial main config は次の形にする。

```text
$include <_main.txt>
$include <DropMerge_optuna.txt>

app.$ = app.batchrun > P
app.run_name = dropmergeSmall_t00001
app.runs_dir = runs_optuna
app.batchrun.exp_exit_step = 1000000

train.seed = 12345

net.block.[Opt...]
net.branch.[main_feature].$ = net.branch.OptunaDropMerge
```

base config と Optuna 専用 config の `$include` 後に trial override を置くことで、既存の設定ファイル、AutoMerge、CLI `key=value` override の仕組みを維持する。Optuna 専用 config は既定で `DropMerge_optuna.txt` とし、通常 run からは読ませない。backend deterministic や DropMergeEnv の `seed_mode` / `global_seed` は、必要に応じて Optuna 専用 config 側で管理する。

## 生成物

生成物は既定で `apps/runner/runs_optuna/` 配下へ保存する。通常の手動 run が入る `runs/` とは混ぜない。metrics-viewer で全 Optuna run を横断表示できるよう、runner 出力は `runs_optuna` 直下へフラットに集める。

```text
apps/runner/runs_optuna/
  optuna.db
  <study_name>_<trial_name>/
    trial/
      manifest.json
      multiseed_summary.json
      multiseed_summary.csv
  <study_name>_<trial_name>_s<seed>/
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

- trial config: runner に渡した main config。`<run_name>/trial/config.txt` に置く。
- manifest JSON: trial params、`cost_tf`、`cost_budget`、seed、run dir。
- metrics summary JSON/CSV: seed run の `metrics.jsonl` から抽出した score と補助指標。
- multiseed summary JSON/CSV: `run-study` の Optuna objective になる seed 集約結果。
- stdout/stderr/process JSON: runner 実行結果。
- runner run dir: runner が出す `metrics.jsonl`、config dump、dot、画像など。

`trial_name` は `t00000` 形式を基本とする。`run-study` では Optuna trial number から生成し、`dry-run` / `run-trial` では同一 `study_name` の既存 `<study_name>_tNNNNN` 出力を見て最大番号+1で自動採番する。`run-study` の代表 `run_name` は `<study_name>_<trial_name>`、seed run の `run_name` は `<study_name>_<trial_name>_s<seed>` とする。`study_name` / `trial_name` に path separator が含まれる場合は実行前エラーにする。

## Study / Trial / Run

- study: Optuna が管理する探索単位。`study_name`、storage、trial 履歴を持つ。
- trial: study 内の 1 params 候補。Optuna が NN params を suggest し、multi-seed aggregate を trial value にする。
- seed run: `run-study` の 1 seed 分の runner 実行出力。`train.seed` だけを seed ごとに変える。
- run: runner の 1 実行出力。`run-trial` では `1 trial = 1 run`、`run-study` では `1 trial = 複数 seed run` とする。
- multiseed summary: `run-study` の Optuna objective。seed run の score を集約した JSON/CSV。
- trial artifact: harness が残す再現用ファイル。runner 出力とは分け、`runs_optuna/<run_name>/trial` に置く。

`run-study` は study 全体を 1 つの Python プロセスで実行し、内部で trial ごとに runner プロセスを起動する。`run-trial` は固定 params の trial を 1 件だけ runner で実行する。`run-study` が `run-trial` を子プロセス起動する構成にはしない。

`run-study` では 1 Optuna trial の内部で `--seeds` の seed run を逐次実行する。Optuna の探索単位は seed run ではなく params 候補であり、trial value には `multiseed_summary` の aggregate score を返す。代表フォルダ `<study_name>_<trial_name>` は `metrics.jsonl` を持たず、metrics-viewer では seed run だけを表示対象にする。

`run-study` は Optuna の `TPESampler` を明示的に使う。`--sampler-seed` で sampler の乱数 seed を固定でき、`--n-startup-trials` で TPE に切り替える前の random sampling 完了 trial 数を指定できる。`--sampler-seed` 未指定時は探索候補列を固定しない。
`--constant-liar` は `TPESampler` の `constant_liar` を有効にし、RUNNING trial 近傍の再提案を避ける補助策として使う。完了済み duplicate params の扱いは harness 側の duplicate policy で制御する。

Study User Attributes には `last_*` として最後の `run-study` 起動条件を保存する。`--study-note` が指定された場合だけ `note` も保存する。Study 内で異なる前提の trial が混ざることは許容し、Study attrs は Dashboard 用のメモとして扱う。各 trial の正確な条件は Trial User Attributes、manifest、summary を正とする。

`Ctrl+C` による通常中断では、harness は実行中 runner を停止し、同一 study に残った `RUNNING` trial を `FAIL` に変更する。強制終了などで cleanup が走らなかった場合に備え、`cleanup-running` subcommand で既存 `RUNNING` trial を確認・掃除できるようにする。

## 評価

primary score は `eval1` と `eval2` の reward EMA を同等に扱う固定 `exp_step` window 平均とする。`--window-start` / `--window-end` は絶対 step、負数相対 step、`80%` のような `exp_exit_step` 比率を受け付ける。既定は `80%` から `100%` までとし、`%` 指定は `round(exp_exit_step * percent / 100)` に解決する。

既定 tag:

```text
21_eval/03_target_reward_ema
21_eval/04_policy_reward_ema
```

補助指標として以下を保存する。

- eval1/eval2 の `ep_maxrank_mean_ema`。
- eval1/eval2 の `ep_end_fruit_count`。
- eval1/eval2 の `term_reason_*`。
- `exp_step_per_sec` / `exp_step_per_sec_ema` / `elapse_hour`。
- runner return code と elapsed time。

補助指標は分析用であり、v1 の Optuna objective には含めない。

`run-study` の aggregate は `--score-aggregate` で選ぶ。既定は `mean`。選択肢は `mean`、`median`、`mean-minus-std`、`min` とし、`mean-minus-std` の std は population stddev を使う。seed run の一部が失敗した trial は、偏った aggregate を採用せず `FAIL` として扱う。

同一 NN params が再提案された場合は、既定で `reseed` する。`duplicate_count_before` を `duplicate_index` とし、`effective_seed = base_seed + duplicate_index * duplicate_seed_stride` で seed list をずらす。既定の `duplicate_params_max_runs` は `3` とし、`0` は制限なしを表す。seed 固定・決定論設定で duplicate params が無駄になる場合は `prune` policy を使う。

## 探索プロトコル

- `small` と `medium` の 2 study に分ける。
- 各 study は同じ objective を使うが、`cost_budget` を変える。
- `cost_tf > cost_budget` と duplicate max 超過は実行前に prune する。runner failure、metrics missing、primary score unavailable は `FAIL` として扱う。
- `run-study` は既定では単一 seed `12345` だが、乱数影響を見る study では `--seeds 12345,23456,...` のように複数 seed を指定し、aggregate score で比較する。
- 有望 params は duplicate policy の既定 `reseed` により最大 3 回まで seed を変えて再評価する。より広く探索したい場合は `prune`、再評価を制限しない場合は `--duplicate-params-max-runs 0` を使う。
- 同一 GPU 並列は許可する。ただし duration や step/sec は補助記録に留める。
- SQLite storage で同一 process 内 `--n-jobs > 1` を使う場合は、Optuna の DB 書き込み lock 競合が起き得る。既定では SQLite lock 待ち timeout を伸ばすが、再発する場合は `--n-jobs 1` または SQLite 以外の RDB storage を使う。
- `small` / `medium` の結果から、最良 trial そのものではなく、`N/M/L/C/D/H` の感度と Pareto 傾向を見る。
- target budget の長時間評価は v1 の自動探索から外し、上位候補を手動再評価する。

## 成功条件

- `--config <path>` 未指定時に従来どおり `_main.txt` が読まれる。
- `--config <path>` 指定時に trial main config が読まれる。
- dry-run で trial config と manifest を生成できる。
- `cost_budget` 超過 trial が runner 実行前に prune される。
- `metrics.jsonl` から指定 window の score と補助指標を抽出できる。
- 正常完了 trial、実行失敗 trial、事前 prune trial を区別して記録できる。

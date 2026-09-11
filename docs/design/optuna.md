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
Optuna harness が生成する trial config は、`--base-config` で指定した main config、選択 workspace の
`config/_main.txt`、`--extra-config` で指定した domain 専用 config を順に `$include` し、その後ろに trial override を書く。

この文書では、runner 起動、Optuna storage、artifact、multi-seed、summary study などを
「Harness 共通仕様」として扱い、DropMerge の探索対象、generated config、score tag、cost proxy を
「DropMerge Domain 仕様」として分けて説明する。

DropMerge domain v1 は `Flatten` 固定、探索対象は NN 構成のみ。
既定では `$include <_main.txt>`、workspace config、`$include <DropMerge_optuna.txt>` の順に置く。
学習系ハイパラは baseline または `DropMerge_optuna.txt` 側で固定する。

用語の対応:

- study: Optuna が管理する探索単位。`study_name`、storage、trial 履歴を持つ。
- trial: study 内の 1 params 候補。`run-study` では Optuna が params を suggest し、`run-trial` では CLI 固定 params を使う。どちらも multi-seed aggregate を trial value にする。
- seed run: 1 seed 分の runner 実行出力。`train.seed` だけを seed ごとに変える。
- run: runner の 1 実行出力。multi-seed trial では 1 trial が複数 seed run を持つ。
- multiseed summary: Optuna objective。seed run の score を集約した JSON/CSV。
- trial artifact: harness が残す再現用ファイル。config、manifest、stdout/stderr、summary など。
- params group: 同一 `TrialParams` 8 項目を持つ source trial 群。`summarize-study` では 1 params group を summary study の 1 trial にする。
- summary study: source study を Dashboard 閲覧用に再構成した multi-objective study。objective は group seed score 分布の mean / range / std。

プロセス構造:

- `run-study` は study 全体を 1 つの Python プロセスで実行する。
- trial ごとに別プロセス化されるのは runner だけ。
- `run-study` が `run-trial` を子プロセス起動する構成ではない。

## Harness 共通仕様: 出力レイアウト

`--workspace` は既定 `_default` で、相対値は `apps/runner/workspaces/` 基準、絶対 path も使用できる。
出力は workspace 内の `runs/` と `optuna/` に分離する。

```text
apps/runner/workspaces/<workspace>/
  config/_main.txt                      # Env選択を含むworkspace config。
  optuna/
    optuna.db
    harness.log                         # harness共通デバッグログ。harness.log.1/.2へrotate。
    artifacts/                          # Optuna Dashboard用artifact store。
  runs/
    <study_name>_<trial_name>/          # multi-seed trialの代表フォルダ。metrics.jsonlは持たない。
      trial/
        manifest.json
        multiseed_summary.json
        multiseed_summary.csv
        seed_runs.json
    <study_name>_<trial_name>_s<seed>/  # seed run。Metrics ViewerのRunになる。
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
`run-study` / `run-trial` では Optuna trial number から生成し、`dry-run` では既存の同一 study 出力から最大番号+1で自動採番する。
代表 `run_name` は `<study_name>_<trial_name>`。
seed run の `run_name` は `<study_name>_<trial_name>_s<seed>`。

例:

```text
apps/runner/workspaces/dm_opt/runs/dropmergeSmall_t00001/trial/multiseed_summary.json
apps/runner/workspaces/dm_opt/runs/dropmergeSmall_t00001_s12345/metrics.jsonl
apps/runner/workspaces/dm_opt/runs/dropmergeSmall_t00001_s12345/trial/manifest.json
```

trial artifact には、主に次のファイルを保存する。

- `config.txt`: runner に渡した trial config。
- `manifest.json`: study / trial / run 名、params、cost、出力 path など。
- `stdout.log`, `stderr.log`: runner process の標準出力と標準エラー。
- `process.json`: runner process の開始/終了状態。開始直後に `status="running"` で作成し、終了時に `complete` / `failed` / `timed_out` / `interrupted` へ更新する。
- `metrics_summary.json`, `metrics_summary.csv`: seed run の score と補助指標の集計結果。JSON には raw window、実効 window、`exp_exit_step`、late window 指標も残す。
- `multiseed_summary.json`, `multiseed_summary.csv`: 代表フォルダに置く seed 集約結果。seed 別 score に加えて late window 指標の平均とばらつきも保存する。
- `seed_runs.json`: 代表フォルダに置く seed run 別の詳細。seed、run 名、status、score、late window 指標、path、error などを残す。

`<workspace>/optuna/harness.log` は study/trial artifact ではなく、harness 自体の診断用ログである。
`trial-start`、`seed-start`、`runner-start pid=...`、`runner-exit returncode=...`、`trial-complete`、`trial-pruned`、`trial-failed` などの進行ログを 1 行 text で追記する。
既定では `harness.log` が 5 MiB 以上になった次の書き込み前に `harness.log.1` へ rotate し、既存 `.1` は `.2` へ送る。
`.2` より古いログは保持しない。

`<workspace>/runs/<run_name>/trial` は人間と harness が直接読む artifact 置き場である。
一方、`<workspace>/optuna/artifacts` は Optuna の `FileSystemArtifactStore` が管理する Dashboard 用 artifact store であり、階層やファイル名に依存しない。
Dashboard で `Show Artifacts` を有効にするには、`run-study` / `run-trial` が `upload_artifact()` で trial に artifact metadata を登録し、optuna-dashboard を `--artifact-dir artifacts` 付きで起動する必要がある。
`summarize-study` が作る summary study では、target trial の詳細を `group_summary.json` として同じ artifact store に登録する。

Metrics Viewer では画面上部の workspace selector から Optuna workspace を選択する。
metrics-viewer は直下フォルダに `metrics.jsonl` があるものだけを run として認識するため、multi-seed 代表フォルダは表示されず、seed run だけが表示される。

`<run_name>` フォルダを削除しても `optuna.db` 内の trial レコードは消えない。
DB には params、score、user_attrs、保存済み path が残るため、削除後は `run_dir` や `artifact_dir` が stale path になる。

### workspace preflight と出力境界

run 系 command は、既存 workspace directory と `config/_main.txt` を要求する。
`--storage` と `--optuna-artifact-dir` の override は、path component 単位で解決後の `<workspace>/optuna/` 配下に含まれる場合だけ許可する。
`<workspace>/runs/`、SQLite DB の親 directory、artifact store は、引数・workspace・出力 path・既存 target type の検証がすべて成功した後にだけ生成する。
検証失敗時は Optuna 接続、runner 起動、config/manifest/log 出力を行わない。

`cleanup-running` は `--storage` を明示した場合、その storage だけで完結し、workspace を要求しない。
`summarize-study` は source storage/artifact を明示した場合、それらだけで source を解決する。
target storage/artifact を省略した場合は source と同じ場所を継承するため、別 workspace への暗黙出力は行わない。

## Harness 共通仕様: コマンド概要

### 共通引数

`dry-run`、`run-trial`、`run-study` は次の共通引数を持つ。
Harness 共通の引数と DropMerge domain 固有の引数が同じ command に並ぶが、CLI 名と意味は従来どおりである。

Harness 共通:

- `--repo-root`: `anet-lab` のリポジトリルート。path 解決の基準。
- `--workspace`: Run、storage、artifact を束ねる workspace path。既定は `_default`。
- `--base-config`: trial config が最初に `$include` する main config。既定は `_main.txt`。
- `--study-name`: study 名。`run_name` の prefix になり、`dry-run` / `run-trial` / `run-study` / `cleanup-running` では必須。
- `--exp-exit-step`: proxy trial の `app.batchrun.exp_exit_step`。負数相対 window と `%` window の基準にもなる。

DropMerge domain 固有:

- `--extra-config`: base config の後に `$include` する DropMerge Optuna 専用 config。既定は `DropMerge_optuna.txt`。
- `--budget`: `small` / `medium` の `cost_budget` preset。
- `--cost-budget`: preset を使わず `cost_budget` を直接指定する。
- `--cost-k`: `cost_tf` の `N*M^2` 項に掛ける係数。
- `--nhead`: Transformer の attention head 数。

`dry-run` と `run-trial` は trial 1 件を固定 params で扱うため、固定 NN params を持つ。
`dry-run` は `--trial-name`、`--trial-number`、`--seed` を持つ。
`run-trial` は Optuna DB に trial 登録するため、`--trial-name` は任意の run 名 override として扱い、trial number は Optuna が割り当てる。seed は `--seeds` で指定する。
`run-study` は Optuna が trial を生成するため、`--trial-name`、`--trial-number`、`--seed` は持たない。
ただし固定 NN params は探索空間の制限として指定できる。
未指定の NN params は既定探索候補のまま残り、指定した NN params だけその値に固定される。

`dry-run` で `--trial-name` と `--trial-number` をどちらも省略した場合、
選択 workspace の `runs/` 直下にある `<study_name>_tNNNNN` を見て次番号を決める。
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

CLI で固定指定した NN params を 1 Optuna trial として multi-seed 実行し、`metrics.jsonl` を採点する。
`run-study` の 1 trial と同じく、DB state、Trial User Attributes、Optuna artifact、`multiseed_summary.*`、`seed_runs.json` を残す。

主な用途:

- smoke run
- 手動候補の再評価
- `run-study` 前の runner 起動確認

追加の主な引数:

- `--runner-exe`: `AnetRLRunner.exe` の path。相対 path は repo root 基準。既定は `apps/runner/bin/Release/AnetRLRunner.exe`。
- `--timeout-sec`: runner 1 trial の timeout 秒。`0` は timeout なし。
- `--seeds`: 同一 params を評価する `train.seed` の comma-separated list。既定は `12345`。
- `--score-aggregate`: seed 別 score を trial value に集約する方法。
- `--storage`, `--storage-timeout-sec`, `--optuna-artifact-dir`: DB 登録と Dashboard artifact 用の設定。
- `--window-start`, `--window-end`: primary score を集計する `exp_step` window。絶対 step、負数相対 step、`80%` のような `exp_exit_step` 比率を指定できる。既定は `80%` から `100%`。

`cost_tf > cost_budget` の場合、runner は起動せず Optuna trial は `PRUNED`、CLI は exit code `2` で終了する。
runner failure、metrics missing、primary score unavailable は `FAIL` として登録し、CLI は exit code `2` で終了する。

### `run-study`

Optuna study を作成または再開し、Optuna が suggest した params で trial を複数実行する。
1 Optuna trial は 1 NN params 候補を表し、`--seeds` で指定した seed ごとに runner を逐次起動する。
Optuna の trial value には seed 別 score ではなく、`multiseed_summary.json` の aggregate score を返す。

`run-study` は study 全体のコマンドなので、`--trial-name` は持たない。
trial 名は Optuna trial number から `t00000`, `t00001`, ... と自動生成される。
固定 NN params は trial 固有値ではなく、探索空間の制限として指定できる。
例えば `--token-mode stronger --d-model 128` を指定すると、`token_mode` と `d_model` はその値だけを候補にし、他の NN params は通常どおり探索する。
内部的には Optuna の `PartialFixedSampler` で固定するため、固定 params 指定時は Optuna の experimental warning が表示される場合がある。
`--search-mode grid` では固定指定を反映した grid search space を作り、未実行 combo だけを事前 enqueue して実行する。
grid の処理済み判定は `params + seed batch` 単位で行うため、同じ NN params でも `--seeds 1,2,3` と `--seeds 4,5,6` は別 combo として扱う。
grid mode は同一 params の duplicate PRUNED を作らないが、`cost_tf > cost_budget` の combo は `PRUNED` として残す。

主な引数:

- `--study-name`: Optuna study 名。
- `--storage`: Optuna SQLite DB URL または path。既定は `<workspace>/optuna/optuna.db`。run系のoverrideはworkspaceの`optuna/`配下だけを許可する。
- `--storage-timeout-sec`: SQLite storage の lock 待ち timeout 秒。既定は `120.0`。
- `--heartbeat-interval-sec`: Optuna RDBStorage heartbeat interval 秒。既定は `60`。`0` で heartbeat を無効にする。
- `--heartbeat-grace-period-sec`: heartbeat が途絶えた `RUNNING` trial を stale とみなす猶予秒。既定は `600`。
- `--optuna-artifact-dir`: Optuna Dashboard 用 artifact store。既定は `<workspace>/optuna/artifacts`。overrideもworkspaceの`optuna/`配下だけを許可する。
- `--n-trials`: この実行で追加する trial 数。`tpe` 未指定時は `10`、`grid` 未指定時は未実行 combo 全件。
- `--n-jobs`: Optuna の並列 worker 数。
- `--study-note`: Study User Attributes の `note` に保存する任意メモ。未指定時は既存 `note` を変更しない。
- `--search-mode`: 探索方法。`tpe` は従来の TPE、`grid` は固定指定を反映した全組み合わせ探索。既定は `tpe`。
- `--sampler-seed`: TPE sampler の乱数 seed。grid では combo 列挙順の shuffle seed。未指定時は TPE は Optuna 既定、grid は通常順。
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
- `--cnn-channels`, `--res-blocks`, `--token-mode`, `--d-model`, `--transformer-layers`, `--ff-mult`, `--trunk-width`, `--head-width`: 指定した param だけ探索候補を 1 値に制限する。未指定 param は既定探索候補を使う。
- backend deterministic や DropMergeEnv の `seed_mode` / `global_seed` は、生成 config ではなく `--extra-config` 側で管理する。
- `--window-start`, `--window-end`: primary score 集計 window。絶対 step、負数相対 step、`80%` のような `exp_exit_step` 比率を指定できる。既定は `80%` から `100%`。

`cost_tf > cost_budget` と duplicate max 超過は study 内では `PRUNED` として扱う。
runner failure、metrics missing、primary score unavailable、seed run の一部失敗は `FAIL` として扱い、TPE の通常学習材料に入れない。
Trial User Attributes の multi-seed 統計は `score_mean`、`score_std`、`score_min`、`score_max`、`score_range`、late window 指標、`seed_count`、`seed_success_count`、`seed_failure_count` に絞る。
seed run 別の詳細は `seed_runs.json` と Optuna artifact を参照する。
Study User Attributes には `last_*` として最後の `run-study` 起動条件を保存する。これは Dashboard 用のメモであり、study 全体の固定契約ではない。
optuna-dashboard では aggregate score が trial value として表示される。

`search-mode=tpe` では、`--n-startup-trials` 件までは random sampling し、その後 TPE が過去 trial から候補を寄せる。
固定 params は `PartialFixedSampler` で強制する。

`search-mode=grid` では、DropMerge の探索候補と CLI 固定 params から harness 側で grid search space を作る。
Optuna の `GridSampler` ではなく、harness が combo を列挙して `enqueue_trial()` へ積む。
grid identity は `params + seed batch` なので、同じ params でも seed batch が違えば別 combo として再評価できる。
既存 `COMPLETE` / cost 超過 `PRUNED` / `RUNNING` / `WAITING` combo は処理済みとして再 enqueue しない。
duplicate max 超過など cost 超過ではない `PRUNED` combo は処理済みにしない。
既存 `FAIL` combo は事故扱いとして未実行に戻し、再 enqueue 対象にする。
`--sampler-seed` は grid combo の列挙順 shuffle seed として使う。
`--n-startup-trials` と `--constant-liar` は TPE 用 option であり、grid mode の探索ロジックには影響しない。

heartbeat 有効時は `run-study` 開始時と終了時に `optuna.storages.fail_stale_trials(study)` を呼び、親 Python が OS kill や crash で落ちた後に残った stale `RUNNING` trial を `FAIL` へ寄せる。
ただし、プロセスが死んだ瞬間に即時 `FAIL` へ変わるわけではない。
次回 `run-study` 起動時の stale check、または `cleanup-running` で復旧する前提で読む。

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
- `--workspace`: `--storage` 省略時にstorageを導出するworkspace。明示storageだけで完結する場合はworkspace configを参照しない。
- `--storage`: Optuna SQLite DB URL または path。省略時は`<workspace>/optuna/optuna.db`。
- `--dry-run`: 対象 trial number を表示するだけで DB を変更しない。

`RUNNING` trial が残ると、`--constant-liar` や duplicate 判定で「まだ実行中の trial」として扱われる。
`Ctrl+C` 後に dashboard で `RUNNING` が残っている場合は、再開前に `cleanup-running --dry-run` で対象を確認し、問題なければ本実行する。

### `summarize-study`

既存 source study の同一 params group をまとめ、Dashboard 閲覧用の summary study を生成する。
source study は変更しない。
target study は multi-objective study で、1 params group を 1 target trial として保存する。

主な引数:

- `--source-study-name`: 集約元の Optuna study 名。必須。
- `--target-study-name`: 集約先の Optuna study 名。未指定時は `<source-study-name>_summary`。
- `--workspace`: source省略時にstorage/artifactを導出するworkspace。
- `--source-storage`: 集約元 Optuna SQLite DB URL または path。省略時は`<workspace>/optuna/optuna.db`。
- `--target-storage`: 集約先 Optuna SQLite DB URL または path。未指定時は source と同じ。
- `--source-artifact-dir`: 集約元 Dashboard artifact store。省略時は`<workspace>/optuna/artifacts`。
- `--target-artifact-dir`: 集約先 Dashboard artifact store。未指定時は source と同じ。
- `--overwrite-target-study`: 既存 target study を削除して作り直す。未指定時に target study が存在する場合はエラー。

group key は `TrialParams` の 8 項目完全一致である。
`cnn_channels`、`res_blocks`、`token_mode`、`d_model`、`transformer_layers`、`ff_mult`、`trunk_width`、`head_width` がすべて同じ source trial だけを 1 group にする。

objective 集約対象は source の `COMPLETE` trial だけである。
`FAIL` / `PRUNED` / `RUNNING` は objective に混ぜず、summary trial の `source_state_counts` にだけ残す。
seed score 分布は source trial に紐づく Dashboard artifact の `seed_runs.json` を正とする。
`seed_runs.json` が欠けている古い study は、誤った分布で代用せず、対象 source trial number を出してエラーにする。

summary study の objective は次の 3 つである。

| Objective | Direction | 意味 |
| --- | --- | --- |
| `mean` | maximize | group 内の全 complete seed score の平均。主指標として読む。 |
| `range` | minimize | group 内の全 complete seed score の `max - min`。小さいほど seed ばらつきが狭い。 |
| `std` | minimize | group 内の全 complete seed score の population standard deviation。 |

Dashboard 上では multi-objective study として表示されるため、single-objective の `best_trial` / `best_value` ではなく、Pareto front と objective columns を読む。
運用上は `mean` を主指標にし、`range` / `std` で上振れ候補や不安定候補を見分ける。

同じ params group 内で source trial の採点条件が混ざっていても、`summarize-study` は停止しない。
その場合は WARN を出し、target trial attrs の `source_context_mixed=true` と `group_summary.json` の `source_score_contexts` に根拠を残す。

## Harness 共通仕様: Optuna Attributes 一覧

Optuna Dashboard では Study User Attributes と Trial User Attributes が見える。
ただし attr は多くなりやすいため、Study attrs は「最後に起動した条件のメモ」、Trial attrs は「個々の trial の実体」として読む。
同じ study 内に異なる前提条件の trial が混ざることは許容するため、比較や後処理では Study attrs ではなく Trial attrs、`manifest.json`、`multiseed_summary.json` を正とする。

### Study User Attributes

`run-study` 起動時に、次の `00_*` / `last_*` を毎回上書きする。
これは dashboard で直近の起動条件を見るためのメモであり、study 全体の固定契約ではない。
Dashboard の `00_last_run_study_args` は、次回再開時に次のように貼り付けて使う。

```powershell
python apps\runner\tools\dropmerge_optuna.py <00_last_run_study_args の値>
```

| Attribute | 更新契機 | 意味 |
| --- | --- | --- |
| `00_last_run_study_args` | `run-study` | Dashboard 上で上に出しやすい copy-paste 用 args。`run-study ...` から始まり、Python executable と script path は含めない。 |
| `last_launch_at` | `run-study` | 最後に `run-study` を起動した UTC 時刻。 |
| `last_harness` | `run-study` | harness 名。現在は `dropmerge_optuna`。 |
| `last_command` | `run-study` | 起動コマンド種別。現在は `run-study`。 |
| `last_study_name` | `run-study` | 起動時に指定した study 名。 |
| `last_storage` | `run-study` | 起動時に使った Optuna storage URL。 |
| `last_storage_timeout_sec` | `run-study` | SQLite storage lock 待ち timeout 秒。 |
| `last_heartbeat_interval_sec` | `run-study` | RDBStorage heartbeat interval 秒。`0` は無効。 |
| `last_heartbeat_grace_period_sec` | `run-study` | stale `RUNNING` trial とみなす heartbeat 猶予秒。 |
| `last_workspace` | `run-study` | 起動時に採用したworkspace入力。 |
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
| `last_search_mode` | `run-study` | `tpe` / `grid` の探索 mode。 |
| `last_sampler_seed` | `run-study` | TPE sampler seed。grid では combo 列挙順の shuffle seed。未指定時は `null`。 |
| `last_n_startup_trials` | `run-study` | TPE 前に random sampling する完了 trial 数。 |
| `last_constant_liar` | `run-study` | `TPESampler(constant_liar=...)` の指定値。 |
| `last_fixed_params` | `run-study` | CLI で固定指定した NN params。未指定 param は含まれない。 |
| `last_duplicate_params_policy` | `run-study` | duplicate params の扱い。`allow` / `prune` / `reseed`。 |
| `last_duplicate_params_max_runs` | `run-study` | 同一 NN params を実行する最大回数。`0` は制限なし。 |
| `last_duplicate_seed_stride` | `run-study` | `reseed` 時に duplicate index ごとに seed へ足す値。 |
| `last_n_trials` | `run-study` | この起動で追加しようとした trial 数。 |
| `last_n_jobs` | `run-study` | Optuna worker 並列数。 |
| `last_timeout_sec` | `run-study` | runner 1 run の timeout 秒。 |
| `last_runner_exe` | `run-study` | 起動時に使った runner executable path。 |
| `last_base_config` | `run-study` | 生成 config が最初に `$include` した base config。 |
| `last_extra_config` | `run-study` | 生成 config が base config の次に `$include` した Optuna 専用 config。 |
| `last_grid_seed_batch_key` | `run-study --search-mode grid` | grid identity に含めた seed batch key。例: `1,2,3`。 |
| `last_grid_seeds` | `run-study --search-mode grid` | grid identity に含めた seed list。 |
| `last_grid_total_count` | `run-study --search-mode grid` | 固定指定と seed batch を反映した grid combo 総数。 |
| `last_grid_already_handled_count` | `run-study --search-mode grid` | 起動時点で `COMPLETE` / cost 超過 `PRUNED` / `RUNNING` / `WAITING` として処理済みだった combo 数。 |
| `last_grid_missing_count` | `run-study --search-mode grid` | 起動時点で未実行扱いだった combo 数。`FAIL` のみ、または cost 超過ではない `PRUNED` の combo はここに戻る。 |
| `last_grid_waiting_count` | `run-study --search-mode grid` | 起動時点で既に `WAITING` だった grid combo 数。 |
| `last_grid_scheduled_count` | `run-study --search-mode grid` | 今回新たに enqueue した combo 数。 |
| `last_grid_optimize_n_trials` | `run-study --search-mode grid` | 今回 `study.optimize()` に渡した実効 trial 数。既存 `WAITING` と新規 enqueue の合計。 |
| `last_grid_cost_over_budget_count` | `run-study --search-mode grid` | grid combo 全体のうち `cost_tf > cost_budget` になる候補数。 |
| `note` | `run-study --study-note` | 人間向けメモ。未指定時は既存値を変更しない。空文字指定時は空文字で上書きする。 |
| `cleaned_running_trials` | `cleanup-running` | `FAIL` に変更した `RUNNING` trial number の一覧。 |
| `cleaned_running_trials_at` | `cleanup-running` | cleanup 実行 UTC 時刻。 |

### Trial User Attributes

`run-study` / `run-trial` の各 Optuna trial には、候補 params、cost、出力 path、multi-seed 集約、duplicate 判定の情報を保存する。

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
| `score_60_80_mean` | aggregate 確定時 | seed 別 `score_60_80` の平均。primary window とは別の補助分析値。 |
| `score_80_100_mean` | aggregate 確定時 | seed 別 `score_80_100` の平均。 |
| `late_slope_mean` | aggregate 確定時 | seed 別 `late_slope = score_80_100 - score_60_80` の平均。後半で伸びているかを見る補助値。 |
| `late_slope_std` | aggregate 確定時 | seed 別 `late_slope` の population standard deviation。 |
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
| `grid_seed_batch_key` | grid trial enqueue / aggregate 確定時 | grid identity に含めた seed batch key。grid mode 以外では通常未設定。 |
| `grid_seeds` | grid trial enqueue / aggregate 確定時 | grid identity に含めた seed list。grid mode 以外では通常未設定。 |

`run-study` の trial value は `score` と同じ aggregate score である。
`score_mean` などは seed 別 score から計算した補助値で、どれを trial value に使うかは `score_aggregate` で分かる。
`score_median`、`score_mean_minus_std`、seed 別 score / run 名などの詳細は Trial User Attributes には入れず、`multiseed_summary.json` と `seed_runs.json` を正として読む。
late window 指標の seed 別値も `seed_runs.json` を正とする。

単発 seed run を直接 Optuna trial に渡す内部経路では、補助 attr として次も保存できる。
通常の CLI では multi-seed trial の summary と artifact を正として読む。

| Attribute | 更新契機 | 意味 |
| --- | --- | --- |
| `returncode` | runner 終了時 | runner process の exit code。 |
| `metric:<tag>:mean` | metrics 集計時 | 指定 window 内の scalar tag 平均。 |
| `metric:<tag>:last` | metrics 集計時 | 指定 window 内の scalar tag 最終値。 |

### Summary Study / Trial Attributes

`summarize-study` が作る summary study では、Study User Attributes に生成条件と件数を保存する。

| Attribute | 意味 |
| --- | --- |
| `summary_created_at` | summary study を生成した UTC 時刻。 |
| `summary_harness` | 生成した harness 名。現在は `dropmerge_optuna`。 |
| `summary_command` | 生成コマンド。現在は `summarize-study`。 |
| `source_study_name` / `source_storage` | 集約元 study 名と storage URL。 |
| `source_artifact_dir` | 集約元 Dashboard artifact store。 |
| `target_study_name` / `target_storage` | 集約先 study 名と storage URL。 |
| `target_artifact_dir` | 集約先 Dashboard artifact store。 |
| `summary_objective_names` | `["mean", "range", "std"]`。 |
| `summary_objective_directions` | `["maximize", "minimize", "minimize"]`。 |
| `summary_group_count` | 生成した target trial 数。 |
| `summary_source_complete_trial_count` | objective 集約に使った source `COMPLETE` trial 数。 |
| `summary_source_seed_count` | objective 集約に使った complete seed score 数。 |
| `summary_source_state_counts` | source study 全体の trial state 件数。 |
| `summary_mixed_context_group_count` | 採点条件混在を検出した params group 数。 |

summary trial の Trial User Attributes は、Dashboard table で読むための要約に絞る。

| Attribute | 意味 |
| --- | --- |
| `group_id` | `g00000` 形式の summary group id。 |
| `params` | group key になった `TrialParams` 一式。 |
| `source_trial_numbers` | この group に含めた source `COMPLETE` trial number。 |
| `source_trial_count` | この group に含めた source `COMPLETE` trial 数。 |
| `source_seed_count` | この group に含めた complete seed score 数。 |
| `group_score_mean` / `group_score_range` / `group_score_std` | target objective と同じ値。 |
| `group_score_min` / `group_score_max` / `group_score_median` / `group_score_mean_minus_std` | group seed score 分布の補助統計。 |
| `group_score_60_80_mean` / `group_score_80_100_mean` / `group_late_slope_mean` / `group_late_slope_std` | group seed score から計算した late window 補助統計。summary objective には使わない。 |
| `source_trial_score_mean` / `source_trial_score_range` / `source_trial_score_std` | source trial aggregate value の分布。seed score 分布とは別に読む。 |
| `source_context_mixed` | source trial の採点条件が group 内で混在している場合に `true`。 |
| `source_state_counts` | 同じ params group に属する source trial state 件数。objective に使わない state も含む。 |

seed 別 score、source trial 別 score、effective seeds、run name、path、採点条件の詳細は Trial User Attributes ではなく、target trial artifact の `group_summary.json` を正として読む。

### 同じ params の見分け方

Dashboard 上で同じ NN params の再評価を見分けたい場合は、まず `params` または `trials_dataframe()` が作る `params_*` 列で grouping する。
そのうえで、duplicate 系 attr を読む。

- `duplicate_index`: 同一 params の何回目の実行か。`0` が初回、`1` が 2 回目。
- `duplicate_count_before`: この trial の開始前に、同一 params の `COMPLETE` / `RUNNING` trial がいくつあったか。
- `duplicate_matched_trials`: duplicate 判定で一致した過去 trial number。
- `base_seeds`: CLI の `--seeds` で指定した元 seed。
- `effective_seeds`: 実際に使った seed。`--duplicate-params-policy reseed` では `duplicate_index * duplicate_seed_stride` だけずれる。
- `score_std` / `score_range`: seed 違いのばらつき。
- `late_slope_mean`: 後半 80%〜100% が 60%〜80% より伸びているか。正なら後半も伸びている候補、0 以下なら頭打ちまたは失速候補として暫定的に読む。

seed ごとの score、late window 指標、run folder 名、path、error は Trial User Attributes ではなく、代表フォルダの `trial/seed_runs.json` または Dashboard の Artifacts から確認する。

同じ params group を Dashboard 上で直接比較したい場合は、`summarize-study` で summary study を作る。
summary study では 1 params group が 1 trial になり、`mean` / `range` / `std` の multi-objective として表示される。

CSV で見る場合は、Optuna の `trials_dataframe()` を使うと `params_*` と `user_attrs_*` の列が生成される。

```python
import optuna

study = optuna.load_study(
    study_name="dropmergeSmall",
    storage="sqlite:///apps/runner/workspaces/dm_opt/optuna/optuna.db",
)

df = study.trials_dataframe()
df.to_csv("optuna_trials.csv", index=False)
```

同じ params の再評価だけを追いたい場合は、`params_*` 列を同一キーとして group 化し、`user_attrs_duplicate_index`、`user_attrs_effective_seeds`、`user_attrs_score` を並べると分かりやすい。

## DropMerge Domain 仕様: Score 算出基準

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

late window 指標は、primary score と同じ 2 tag を固定 window で再集計した補助値である。

```text
score_60_80  = score over [round(exp_exit_step * 0.60), round(exp_exit_step * 0.80)]
score_80_100 = score over [round(exp_exit_step * 0.80), exp_exit_step]
late_slope   = score_80_100 - score_60_80
```

これらは trial value には使わない。
`late_slope > 0` は後半で伸びている候補、`late_slope <= 0` は頭打ちまたは失速候補として読む。
ただし、seed ぶれや評価 window 内のイベント密度に影響されるため、最終性能の代替ではなく分析用の補助指標として扱う。

補助 tag、duration、step/sec、終端理由、`max_rank`、`fruit_count` は summary には保存するが、v1 の score には入れない。
また、primary tag の `last` ではなく window 内の `mean` を使うため、短い window ではログ間隔や評価回数の影響を受けやすい。

`run-trial` / `run-study` では、上記の単発 score を seed ごとに計算したうえで、`--score-aggregate` に従って trial value を決める。

```text
mean           = average(seed score list)
median         = median(seed score list)
mean-minus-std = average(seed score list) - population_stddev(seed score list)
min            = min(seed score list)
```

seed の一部が失敗した場合、その Optuna trial は aggregate score を採用せず `FAIL` にする。

## Harness 共通仕様: 利用手順

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
python apps\runner\tools\dropmerge_optuna.py dry-run --workspace dm_opt --study-name dropmergeSmall --trial-name t00001 --budget small
```

確認点:

- `run_name` が `dropmergeSmall_t00001`。
- `runs_dir` が `workspaces/dm_opt/runs`。
- `run_dir` が `apps/runner/workspaces/dm_opt/runs/dropmergeSmall_t00001`。
- `artifact_dir` が `apps/runner/workspaces/dm_opt/runs/dropmergeSmall_t00001/trial`。
- `pruned_by_cost` が期待どおり。

`--trial-name` を省略すると、同一 `study_name` の既存 `<study_name>_tNNNNN` 出力から次番号が採番される。

### 4. run-trial で smoke run する

短い step で runner 起動、metrics 生成、summary 生成まで確認する。

```powershell
python apps\runner\tools\dropmerge_optuna.py run-trial --workspace dm_opt --study-name dropmergeSmoke --trial-name t00000 --budget small --exp-exit-step 2000 --window-start 0 --window-end 100% --timeout-sec 600
```

確認点:

- `apps/runner/workspaces/dm_opt/runs/dropmergeSmoke_t00000_s12345/metrics.jsonl` がある。
- `apps/runner/workspaces/dm_opt/runs/dropmergeSmoke_t00000/trial/manifest.json` がある。
- `apps/runner/workspaces/dm_opt/runs/dropmergeSmoke_t00000/trial/multiseed_summary.json` の `score` が `null` ではない。
- Optuna DB の `dropmergeSmoke` study に `COMPLETE` trial が登録される。

cost prune だけ確認する場合:

```powershell
python apps\runner\tools\dropmerge_optuna.py run-trial --workspace dm_opt --study-name dropmergeSmoke --trial-name tCost --cost-budget 1
```

この場合 runner は起動せず、exit code `2` になる。
Optuna DB には `PRUNED` trial として残る。

### 5. run-study で探索する

small study:

```powershell
python apps\runner\tools\dropmerge_optuna.py run-study --workspace dm_opt --study-name dropmergeSmall --budget small --n-trials 20 --n-jobs 1 --seeds 12345,23456,34567 --score-aggregate mean-minus-std --duplicate-params-policy reseed --duplicate-params-max-runs 3 --exp-exit-step 1000000
```

medium study:

```powershell
python apps\runner\tools\dropmerge_optuna.py run-study --workspace dm_opt --study-name dropmergeMedium --budget medium --n-trials 20 --n-jobs 1 --seeds 12345,23456,34567 --score-aggregate mean-minus-std --exp-exit-step 1000000
```

一部の NN params を固定して探索する場合:

```powershell
python apps\runner\tools\dropmerge_optuna.py run-study --workspace dm_opt --study-name dropmergeStrong128 --budget medium --n-trials 20 --seeds 12345,23456,34567 --token-mode stronger --d-model 128
```

この例では `token_mode=stronger` と `d_model=128` だけを固定し、`cnn_channels`、`res_blocks`、`transformer_layers`、`ff_mult`、`trunk_width`、`head_width` は通常どおり探索する。

固定済み params の残り組み合わせを重複なしで総当たりする場合:

```powershell
python apps\runner\tools\dropmerge_optuna.py run-study --workspace dm_opt --study-name dropmergeGridD96 --budget medium --search-mode grid --n-jobs 2 --seeds 1,2,3 --cnn-channels 64 --token-mode current --d-model 96 --trunk-width 2048 --head-width 1024
```

この例では残りが `res_blocks x transformer_layers x ff_mult = 8` combo になる。
`--n-trials` を省略すると未実行 combo を全件 schedule する。
`--n-trials 3` のように指定すると、未実行 combo のうち最大 3 件だけを schedule する。
grid の再開判定は `params + seed batch` 単位なので、同じ study に `--seeds 4,5,6` で再実行すると、`--seeds 1,2,3` で完了済みの params も別 combo として実行できる。
同じ seed batch の既存 `COMPLETE` / cost 超過 `PRUNED` / `RUNNING` / `WAITING` combo は再実行せず、既存 `FAIL` combo は再実行対象に戻す。
cost 超過 combo は `PRUNED` として残るため、どの組み合わせが cost 制約外だったかを後から確認できる。

`--n-jobs > 1` は同一 GPU 上で runner が並列起動する。duration や step/sec は干渉を受けるため、score には使わず補助指標として読む。
multi-seed は 1 Optuna trial の内部で seed を逐次実行する。`--n-jobs > 1` の場合は、複数 params 候補が並列に進む。
SQLite storage で `--n-jobs > 1` を使うと、Optuna の trial / user attrs 書き込みが競合し `database is locked` になることがある。
既定では `--storage-timeout-sec 120.0` を設定して短い lock 競合を待つ。
それでも再発する場合は `--n-jobs 1` にする。PH3 の harness CLI は SQLite storage だけを受け付け、非 SQLite URL は fail-fast する。

同じ `--study-name` と `--storage` で再実行すると study は再開され、trial が追加される。
同じ study を再現性重視で回す場合は、`--sampler-seed` を固定し、`--n-startup-trials` も明示しておく。
Debug runner を使う場合は、従来どおり `--runner-exe apps/runner/bin/Debug/AnetRLRunner.exe` で上書きする。

既定では `exp_exit_step` の 80% から 100% までを採点する。
終了直前の固定 step 幅だけを採点したい場合は、例えば `--window-start -200000` とし、`--window-end` は省略する。
この場合、`exp_exit_step - 200000` から `exp_exit_step` までを集計する。

seed 固定・決定論設定で duplicate params が完全に無駄になる場合は、次のように prune する。

```powershell
python apps\runner\tools\dropmerge_optuna.py run-study --workspace dm_opt --study-name dropmergeFixed --budget small --n-trials 20 --seeds 1,2,3 --duplicate-params-policy prune
```

有望 params を何度も seed を変えて再評価したい場合は、既定の `reseed` を使う。無制限に再評価を許す場合は `--duplicate-params-max-runs 0` を指定する。

探索を中断するときは、まず `Ctrl+C` を 1 回だけ押す。
通常は harness が実行中 runner を止め、同一 study の `RUNNING` trial を `FAIL` に変更する。
`Ctrl+C` 連打、ターミナルごと終了、OS kill では cleanup が走らない場合がある。
その場合でも heartbeat 有効な `run-study` では、次回起動時に stale `RUNNING` を `FAIL` へ寄せる。
既存の heartbeat 無し `RUNNING` や、すぐ手動確認したい場合は次の手順で後処理する。

```powershell
python apps\runner\tools\dropmerge_optuna.py cleanup-running --workspace dm_opt --study-name dropmergeSmall --dry-run
python apps\runner\tools\dropmerge_optuna.py cleanup-running --workspace dm_opt --study-name dropmergeSmall
```

実行中のどの段階で止まったかは、まず `apps/runner/workspaces/dm_opt/optuna/harness.log` を見る。
seed run が runner 起動まで進んでいれば、`apps/runner/workspaces/dm_opt/runs/<run_name>/trial/process.json` が `status="running"` で先に作られる。
親 Python が落ちた場合は `process.json` が `running` のまま残ることがあり、その場合は `runner_pid`、`started_at`、`command`、`config_path` が調査の手掛かりになる。

### 6. summary study を作る

同じ params の reseed 結果を 1 trial として Dashboard で見る場合は、source study から summary study を作る。

```powershell
python apps\runner\tools\dropmerge_optuna.py summarize-study --workspace dm_opt --source-study-name dropmergeSmall
```

既定では `dropmergeSmall_summary` が同じ `apps/runner/workspaces/dm_opt/optuna/optuna.db` 内に作られる。
optuna-dashboard の study list で source study と summary study を切り替えて見る。

target study が既にある場合は、誤って手作業メモや既存 summary を消さないように停止する。
作り直す場合だけ明示する。

```powershell
python apps\runner\tools\dropmerge_optuna.py summarize-study --workspace dm_opt --source-study-name dropmergeSmall --overwrite-target-study
```

### 7. metrics-viewer で見る

Metrics Viewer の workspace selector で `dm_opt` を選択する。
viewer 側では、選択 workspace の `runs/` 直下にある `<run_name>/metrics.jsonl` が run として扱われる。
`run-study` の代表フォルダ `<study_name>_<trial_name>` は `metrics.jsonl` を持たないため表示対象外になり、`<study_name>_<trial_name>_s<seed>` の seed run だけが表示される。
study をまたいだ比較をしたい場合も、`dropmergeSmall_t00000`、`dropmergeMedium_t00000` のように同じ viewer root で横断表示する。
summary study は Dashboard 閲覧用であり、metrics-viewer の run にはならない。

### 8. 既存 metrics を再集計する

```powershell
python apps\runner\tools\dropmerge_optuna.py summarize apps/runner/workspaces/dm_opt/runs/dropmergeSmall_t00000/metrics.jsonl --window-start 80% --window-end 100% --output-dir apps/runner/workspaces/dm_opt/runs/dropmergeSmall_t00000/trial
```

相対 window で再集計する例:

```powershell
python apps\runner\tools\dropmerge_optuna.py summarize apps/runner/workspaces/dm_opt/runs/dropmergeSmall_t00000/metrics.jsonl --exp-exit-step 1000000 --window-start -200000 --output-dir apps/runner/workspaces/dm_opt/runs/dropmergeSmall_t00000/trial
```

## DropMerge Domain 仕様: 探索対象と制約

この節は `dropmerge_optuna.py` 内の `DropMergeDomain` が持つ仕様に対応する。
共通 harness 部品は `optuna_common.py` に分離している。
DropMerge domain は、固定 NN params 引数、Optuna suggest の探索空間、`TrialParams` group key、
token 数計算、`cost_tf`、primary/supplemental score tag、late window、generated config をまとめて定義する。

generated config は次の順で構成する。

1. `$include <_main.txt>` または `--base-config`。
2. 選択 workspace の `config/_main.txt`。
3. `$include <DropMerge_optuna.txt>` または `--extra-config`。
4. trial 固有 override。`app.run_name`、`app.runs_dir`、`app.batchrun.exp_exit_step`、`train.seed`、DropMerge NN block / branch 設定を書く。

DropMerge domain v1 の generated branch は `net.branch.OptunaDropMerge` で、入力 bind は `grid, vector_feature`。
`P=Flatten` は固定し、`token_mode` に応じて `ConvDown` / `ConvDown2` の有無だけを変える。

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

`run-study` では上記 option を個別に指定すると、その param だけ探索候補が指定値 1 つに制限される。
未指定 param は既定探索候補を使う。
固定指定した param も Optuna trial params には通常どおり保存されるため、duplicate 判定や summary-study の group key は従来と同じ 8 params 全体で扱う。

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

## Harness 共通仕様: 運用メモ

- `run-study` は探索用、`run-trial` は固定候補の smoke / 再評価用。
- `run-trial` も Optuna DB に登録し、`run-study` の 1 trial と同じ multi-seed aggregate として扱う。
- `run-study` では trial 名を明示指定しない。`run-trial` では任意で `--trial-name` を指定できる。未指定時は Optuna trial number から自動生成し、同一 trial 内の seed run は `_s<seed>` suffix を付ける。
- `run-study --search-mode tpe` は `TPESampler` を使う。最初の `--n-startup-trials` 件は random sampling、その後は過去の完了 trial に基づいて候補を寄せる。
- `run-study --search-mode grid` は固定指定と seed batch を反映した grid combo を harness 側で列挙し、未実行 combo だけを実行する。duplicate params による prune は作らず、cost 超過は `PRUNED` として残す。
- `--constant-liar` は RUNNING trial 近傍の再提案を避ける補助策。完了済み duplicate params の扱いは `--duplicate-params-policy` で制御する。
- `run-study` / `run-trial` の探索・評価単位は seed run ではなく multi-seed aggregate。DB や optuna-dashboard で見る trial value は aggregate score。
- `summarize-study` の target study は閲覧専用。source study は変更せず、同一 params group の全 complete seed score 分布を multi-objective trial にする。
- 中断時はまず `Ctrl+C` を 1 回だけ押す。`RUNNING` が残った場合は `cleanup-running --dry-run` で確認してから cleanup する。
- Study User Attributes の `last_*` は最後の `run-study` 起動条件を表す。異なる前提の trial が同一 study に混ざることは許容し、各 trial の正確な条件は Trial User Attributes、manifest、summary を正として読む。
- 既定では duplicate params は `reseed` され、最大 3 回まで seed を変えて再評価される。
- 手動の `dry-run` では、`--trial-name` 未指定時に既存出力から最大番号+1で採番する。`run-trial` は Optuna trial number を使う。
- `study_name` と `trial_name` に path separator は使えない。
- Run、storage、artifact、harness log は同じ workspace に束ねる。Run は `<workspace>/runs/`、Optuna 管理物は `<workspace>/optuna/` に置く。
- PH3 は旧 `apps/runner/runs_optuna/` からの自動 migration を行わない。

### 旧 `runs_optuna` からの手動移行

1. 対象の `run-study`、`run-trial`、Optuna Dashboard を停止する。
2. seed run と代表 trial folder を対象 workspace の `runs/` へ移す。
3. `optuna.db`、`harness.log*`、`artifacts/` を同じ workspace の `optuna/` へ移す。SQLite の `-wal` / `-shm` が存在する場合は DB と一緒に移す。
4. 旧 `00_last_run_study_args` はそのまま再利用しない。`--workspace` を追加し、廃止された `--runs-dir` を削除し、必要なら `--storage` / `--optuna-artifact-dir` override を新しい workspace 内 path に更新する。
5. `apps\23_optuna_dashboard.bat <workspace_path>` で DB と artifact store を確認してから探索を再開する。

履歴 artifact や過去の実験記録は当時の path を保持してよい。現行 harness は旧 path の alias や自動変換を持たない。

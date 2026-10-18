# DropMerge Optuna 対応 実装計画

## 実装方針

Optuna は runner 本体に入れず、Python harness から runner を起動する。runner 側の変更は main config file path を差し替える `--config <path>` だけに留める。

v1 の実装単位:

- runner CLI: `--config <path>`。
- Python harness: `apps/runner/tools/dropmerge_optuna.py`。
- metrics 抽出: 同 harness の `summarize` subcommand。
- dry-run: 同 harness の `dry-run` subcommand。
- 固定 trial 実行: 同 harness の `run-trial` subcommand。
- study 実行: 同 harness の `run-study` subcommand。

## Runner 改修

`apps/runner/src/RunnerApp.cpp` の CLI 定義に `--config` / `-c` を追加する。

- 未指定時: 従来どおり `GetProjectRootDir() / "config" / "_main.txt"` を読む。
- 指定時: 指定ファイルを main config として読む。
- 相対パス: まず process current directory 基準で存在確認し、見つからなければ `GetProjectRootDir() / "config"` 基準で解決する。
- 存在しない path: `ANET_SYSTEM_ERROR` で落とす。

`ConfigManager` の挙動は変えない。trial main config 側で `$include` を使う。

## Python Harness

`apps/runner/tools/dropmerge_optuna.py` を追加する。

subcommand:

- `dry-run`: trial config と manifest を生成し、`cost_tf` と prune 判定を表示する。
- `summarize`: 既存 `metrics.jsonl` から score と補助指標を JSON/CSV に出す。
- `run-trial`: CLI で固定指定した params を 1 件 runner で実行し、metrics 抽出を行う。Optuna DB には登録しない。
- `run-study`: Optuna study を作成/再開し、Optuna が生成した trial を同じ Python プロセス内で順次実行する。1 trial は 1 params 候補であり、必要に応じて複数 seed run を集約する。
- `cleanup-running`: 中断や強制終了で Optuna DB に残った `RUNNING` trial を `FAIL` に変更する。

`run-study` だけ Optuna import を要求する。`dry-run`、`run-trial`、`summarize` は標準ライブラリのみで動かす。

`run-study` は `run-trial` を子プロセス起動しない。`objective()` から共通 trial 実行関数を呼び、trial ごとに別プロセス化するのは runner だけにする。
`run-study` は `TPESampler` を明示的に作成し、`--sampler-seed` と `--n-startup-trials` で探索候補列と初期 random sampling 件数を制御する。
`--constant-liar` は `TPESampler(constant_liar=True)` に接続し、RUNNING trial 近傍の再提案を避ける補助策として使う。
`run-study` は `--seeds` で指定された `train.seed` ごとに runner を逐次実行し、`--score-aggregate` で aggregate score を作る。Optuna の trial value は seed 別 run ではなく aggregate score にする。
`run-study` は同一 NN params が再提案されたときの duplicate policy を持つ。既定は `reseed`、最大実行回数は `3`、seed offset stride は `100000` とする。`--duplicate-params-max-runs 0` は制限なしを表す。
`run-study` は Study User Attributes に最後の起動条件を `last_*` として保存する。`--study-note` が指定された場合だけ `note` を保存し、未指定時は既存 `note` を変更しない。異なる前提の trial が同一 study に混ざることは許容し、Study attrs は Dashboard 用のメモとして扱う。
SQLite storage では `--n-jobs > 1` の trial attrs 書き込みで `database is locked` が起き得るため、`run-study` は SQLite 用に `--storage-timeout-sec` を持つ。既定は `120.0` 秒とし、Optuna の `RDBStorage(engine_kwargs={"connect_args": {"timeout": ...}})` に渡す。再発する場合は `--n-jobs 1` または SQLite 以外の RDB storage を使う。
`run-study` は `KeyboardInterrupt` を捕捉し、active runner process を terminate/kill してから同一 study の `RUNNING` trial を `FAIL` に cleanup する。`cleanup-running` は同じ cleanup を手動実行するための subcommand とし、`--dry-run` では対象 trial number だけを表示する。
`run-trial` / `run-study` の `--runner-exe` は任意とし、未指定時は repo root 基準の `apps/runner/bin/Release/AnetRLRunner.exe` を使う。Debug runner を使う場合だけ明示指定する。

## Trial Config 生成

trial config は以下を行う。

- base `_main.txt` を `$include <_main.txt>` で読み、その直後に Optuna 専用 config を `$include <DropMerge_optuna.txt>` で読む。
- Optuna 専用 config は `--extra-config` で差し替え可能にする。
- `app.$ = app.batchrun > P` を設定する。
- `app.run_name` と `app.runs_dir` を trial 固有にする。
- `trial_name` は `t{trial_number:05d}` とする。`dry-run` / `run-trial` で `--trial-number` と `--trial-name` が未指定の場合は、`runs_optuna` 直下の既存 `<study_name>_tNNNNN` を見て最大番号+1で自動採番する。
- `run_name` は `<study_name>_<trial_name>` とする。
- `app.runs_dir` の既定は `runs_optuna` とし、通常の `runs/` と混ぜない。
- trial artifact の既定は `runs_optuna/<run_name>/trial` に置く。
- `app.batchrun.exp_exit_step` を proxy 探索用に上書きする。
- `train.seed` を設定する。`run-study` では seed run ごとに `--seeds` の値を 1 つずつ入れる。
- `backend.deterministic_algorithms`、`backend.deterministic_warn_only`、DropMergeEnv の `seed_mode` / `global_seed` は生成 config では設定せず、必要に応じて Optuna 専用 config 側で管理する。
- `OptConvInit`、`OptResBlock`、`OptConvDown`、`OptViTProj`、`OptTransEnc`、`OptLinear`、`OptHeadFC` を生成する。
- `net.branch.[main_feature].$ = net.branch.OptunaDropMerge` で main feature を差し替える。
- value/adv stream は `OptHeadFC > SiLU` に差し替える。

`token_mode`:

- `current`: `ConvInit` と `ConvDown` の 2 回 stride 2。
- `stronger`: さらに `ConvDown2` を追加。

`Flatten` は固定する。

## Study / Trial / Run

- study: Optuna が管理する探索単位。`study_name`、storage、trial 履歴を持つ。
- trial: study 内の 1 params 候補。`run-study` では Optuna が params を suggest し、multi-seed aggregate を trial value にする。
- seed run: `run-study` の 1 seed 分の runner 実行出力。`train.seed` だけを seed ごとに変える。
- run: runner の 1 実行出力。`run-trial` では `1 trial = 1 run`、`run-study` では `1 trial = 複数 seed run` とする。
- multiseed summary: `run-study` の Optuna objective。seed run の score を集約した JSON/CSV。
- trial artifact: harness が残す config、manifest、stdout/stderr、summary。

`run-study` の CLI には `--trial-name` と固定 NN params を出さない。`run-trial` と `dry-run` だけがそれらを持つ。

`run-study` の代表 `run_name` は `<study_name>_<trial_name>` とし、代表フォルダ `runs_optuna/<run_name>/trial/` に `multiseed_summary.json` と `multiseed_summary.csv` を置く。seed run は `<study_name>_<trial_name>_s<seed>` とし、通常どおり `metrics.jsonl` と `trial/metrics_summary.*` を持つ。中断された seed run の `trial/process.json` には `interrupted=true` を残す。

duplicate 判定は同一 study の既存 `COMPLETE` / `RUNNING` trial の NN params だけを比較する。`PRUNED` / `FAIL` trial は count しない。`allow` は現状互換、`prune` は重複時に runner 起動前 prune、`reseed` は `duplicate_index = duplicate_count_before` として seed list をずらす。

## Cost / Prune

`cost_tf` は trial params から実行前に算出する。

```text
cost_tf = L * (N^2 * M + k * N * M^2)
```

- `N`: token_mode と grid size から畳み込み出力サイズを計算する。
- `M`: `d_model`。
- `L`: Transformer 層数。
- `k`: 既定 `4.0`。

`cost_tf > cost_budget` と duplicate max 超過は runner を起動せず prune にする。runner failure、metrics missing、primary score unavailable は `FAIL` として扱い、study 全体は継続する。`d_model % nhead != 0` は設定生成前にエラーにする。

## Metrics 抽出

`metrics.jsonl` の `type=scalar` record を読む。

primary tags:

```text
21_eval/03_target_reward_ema
21_eval/04_policy_reward_ema
```

指定 `window_start <= step <= window_end` の record を集計し、各 primary tag の mean を取り、その平均を score にする。`window_start` / `window_end` は絶対 step、負数相対 step、`80%` のような `exp_exit_step` 比率を受け付ける。既定は `80%` から `100%` までとし、`%` 指定は `round(exp_exit_step * percent / 100)` に解決する。どちらかの primary tag が欠ける場合、その trial は score 不成立として扱う。

補助 tag は JSON/CSV に出すが Optuna objective には入れない。

`run-study` では seed run ごとの score を集め、`mean`、`median`、`mean-minus-std`、`min` のいずれかで集約する。`mean-minus-std` の std は population stddev とする。Trial User Attributes には aggregate 指標と seed 別 evidence を保存する。
Study User Attributes には最後の `run-study` 起動条件を `last_*` として保存する。Trial User Attributes、代表 manifest、`multiseed_summary.json` には duplicate policy、duplicate count、duplicate index、base/effective seeds、matched trial numbers を保存する。

## Metrics Viewer 連携

metrics-viewer で Optuna run を横断表示する場合は、`metricsviewer.runs-dir=apps/runner/runs_optuna` を指定する。

viewer 側の `LoadingThread` は `RunScanner.resolveRunDir(runId).resolve("metrics.jsonl")` を読む。`runs/<runId>/metrics.jsonl` の固定 path にはしない。
`RunScanner.listRunId()` は直下ディレクトリであっても `metrics.jsonl` が無いものを除外する。これにより `run-study` の代表フォルダは viewer の run list に出ず、seed run だけが表示される。

## 検証手順

1. `anet-core-test` で `ConfigManager` の `$include` 後 override と AutoMerge の回帰テストを確認する。
2. `dropmerge_optuna.py dry-run --study-name dropmergeSmall --trial-name t00001` を実行し、`run_name=dropmergeSmall_t00001`、`runs_dir=runs_optuna`、`artifact_dir=apps/runner/runs_optuna/dropmergeSmall_t00001/trial` になることを確認する。
3. `dropmerge_optuna.py dry-run --study-name dropmergeSmall` を同じ出力 root で 2 回実行し、`t00000`、`t00001` の順に自動採番されることを確認する。
4. `dropmerge_optuna.py summarize --exp-exit-step 2000` を小さい sample `metrics.jsonl` に対して実行し、既定の実効 window が `[1600, 2000]` になることを確認する。
5. `dropmerge_optuna.py summarize --exp-exit-step 2000 --window-start 12.5% --window-end 87.5%` を小さい sample `metrics.jsonl` に対して実行し、実効 window が `[250, 1750]` になることを確認する。
6. `dropmerge_optuna.py run-trial` は `cost_budget` 超過時に runner を起動せず exit code `2` を返すことを確認する。
7. `dropmerge_optuna.py run-study --help` に `--trial-name` と固定 NN params が出ないことを確認する。
8. `dropmerge_optuna.py run-trial --help` に `--trial-name` と固定 NN params が出ることを確認する。
9. `dropmerge_optuna.py run-study --help` に `--sampler-seed` と `--n-startup-trials` が出ることを確認する。
10. `dropmerge_optuna.py run-study --help` に `--seeds` と `--score-aggregate` が出ることを確認する。
11. `dropmerge_optuna.py run-study --help` に `--constant-liar`、duplicate policy option、`--study-note`、`--storage-timeout-sec` が出ることを確認する。
12. `dropmerge_optuna.py --help` に `cleanup-running` が出ること、`cleanup-running --help` に `--dry-run` が出ることを確認する。
13. SQLite storage の場合に `create_optuna_storage()` が `RDBStorage` へ `connect_args.timeout` を渡すことを確認する。
14. cleanup helper が `RUNNING` trial だけを対象にし、`--dry-run` 相当では state mutation しないことを確認する。
15. Study attrs helper が `80%` window の raw/resolved、seed list、`last_*`、指定時の `note` を生成し、未指定時は `note` を含めないことを確認する。
16. cost prune smoke でも Study User Attributes が保存されることを確認する。
17. duplicate helper が `COMPLETE` / `RUNNING` を count し、`PRUNED` / `FAIL` を count しないことを確認する。
18. `reseed` で base seeds `1,2,3`、stride `100000`、duplicate index `2` が `200001,200002,200003` になることを確認する。
19. `policy=prune`、`policy=reseed --duplicate-params-max-runs 3`、`policy=reseed --duplicate-params-max-runs 0`、`policy=allow` の挙動を fake study で確認する。
20. aggregate helper が `mean`、`median`、`mean-minus-std`、`min` を計算できることを確認する。
21. `dropmerge_optuna.py summarize` を小さい sample `metrics.jsonl` に対して実行し、score が計算されることを確認する。
22. runner Release build を実行し、`--config <path>` がコンパイル上問題ないことを確認する。
23. metrics-viewer の unit test で custom runs-dir の `runA/metrics.jsonl` を読めることを確認する。
24. metrics-viewer の unit test で `metrics.jsonl` が無い代表フォルダが `RunScanner.listRunId()` に出ないことを確認する。
25. 実際の Optuna smoke は別途、短い `exp_exit_step` で prune trial、FAIL trial、正常 trial を 1 件ずつ回す。

## 注意点

- 同一 GPU 並列では duration と step/sec が干渉するため、score には使わない。
- `app.runs_dir` は runner project root からの相対 path として扱われ、既定は `runs_optuna` にする。
- Optuna storage は既定で `sqlite:///runs_optuna/optuna.db` とし、実効 DB は `apps/runner/runs_optuna/optuna.db` に置く。
- `--config` path は再現コマンドに残す。環境変数での暗黙差し替えは使わない。
- 生成物は repo の実験出力として扱い、通常のコード差分には含めない。

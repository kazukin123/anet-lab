# PRD100 GridMaze検証

## Debugと公開経路テスト

リポジトリルートでMSVCを初期化し、Debug全体をビルドする。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```

```bash
core\envs\gridmaze1\bin\Debug\GridMazeEnv-test.exe "[gridmaze]"
```

構築・Reset・auxとStepを通じて到達可能性、seedの決定性と復元、既定25セル、上限、寸法・座標・候補なし、seed未指定・seed_mode=episodeでの自動seed50回を確認する。指定seedはmode指定なしで固定される。穴終端の既存テストはseed58の `.OG / ...` を使う。ログ専用タグ `[prd100_log]` は新しいプロセスで実行し、共有記録の既存テストとの干渉を避ける。

```bash
core\envs\gridmaze1\bin\Debug\GridMazeEnv-test.exe "[prd100_log]"
```

## 短いRun

`run_validation.py` は `.scratch/prd100/` 専用。変更前の現行ソースからDebugをビルドした後、Runner/DLLを `baseline-bin/`、設定一式を `baseline-config/` へ保存してからbeforeを実行する。既存の成功・失敗記録へ上書きしない。再試行では以前のstageディレクトリを別名へ退避する。過去Runは変更しない。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd100/run_validation.py before
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd100/run_validation.py after
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd100/run_validation.py compare
```

条件はrun seed1、迷路seed6545537233705677626、8 lane、CPU/FP32、決定論backend、torch thread1、batch16、warmup256、replay capacity4096、replay ratio1、8704 experience、skip無効、評価停止。primaryの既存NN設定を保持する。全scalarをtag・step・値で正規化し完全一致を要求する。Noisy指標を含み、90_perf/・91_resource/・92_resource/の時間・資源だけを除外する。Noisyの11定義を明示選択し、noisy-coverage.jsonへ定義と実出力tagを保存する。既定GridMazeはNoisyが無効のため値が出ない場合も、除外したものとして扱わない。normalized.summary.jsonに点数・tagごとの件数・除外・SHA256、comparison.jsonに比較結果を残す。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd100/run_validation.py log-default
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd100/run_validation.py log-retry
```

ログRunは学習256 laneと既定評価用Envを構築し、256 experienceで終了する。warmupを100000にして学習更新を行わない。既定seedは生成通知0件、seed2は設定seed・retries・actual_seedを持つ通知1件を要求する。ログと終了コードはstage配下へ保存する。実行体はstageごとに固定し、DLLは不変なbaseline-binからhard linkする。新しい依存DLLを追加する変更にはこの検証方法を流用しない。

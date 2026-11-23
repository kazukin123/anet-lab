# anet-lab Archify Atlas

現行 checkout を根拠に、anet-lab の全体像から実行時詳細までを 5 枚の Archify マップで表した索引です。
5 図は同じ canonical term（Run、RunManager、TrainRunner、RunnerThread、Agent、Actor、Learner、BatchEnv、
NetworkModel、ReplayBuffer、Notifier、Observer、MetricsLogger、Run フォルダ、Metricsマスタ、Metrics Viewer）を使います。

## 推奨閲覧順

`表示` は GitHub Pages 上のレンダリング結果、`html` と `json` はリポジトリ内の実ファイルです。

| 順序 | 図 | 何が分かるか | 表示 | ファイル |
|---:|---|---|---|---|
| 1 | システム構成 | 主要コンポーネント、主経路、thread・device・ファイルの境界、外部依存 | [表示](https://kazukin123.github.io/anet-lab/docs/archify/anet_lab_00_system_architecture.html) | [html](anet_lab_00_system_architecture.html) / [json](anet_lab_00_system_architecture.archify.json) |
| 2 | Run 実行ワークフロー | workspace 選択から構築、反復、評価、終了、成果物の分析まで | [表示](https://kazukin123.github.io/anet-lab/docs/archify/anet_lab_10_run_workflow.html) | [html](anet_lab_10_run_workflow.html) / [json](anet_lab_10_run_workflow.archify.json) |
| 3 | 学習ステップの呼び出し順序 | 1 step の行動、Env の Step、学習の更新、通知の順序 | [表示](https://kazukin123.github.io/anet-lab/docs/archify/anet_lab_20_training_step_sequence.html) | [html](anet_lab_20_training_step_sequence.html) / [json](anet_lab_20_training_step_sequence.archify.json) |
| 4 | 実験データの流れ | 設定、State、Experience、minibatch、Event、metrics、設定の写し、成果物の流れ | [表示](https://kazukin123.github.io/anet-lab/docs/archify/anet_lab_30_experiment_dataflow.html) | [html](anet_lab_30_experiment_dataflow.html) / [json](anet_lab_30_experiment_dataflow.archify.json) |
| 5 | runtime の状態遷移 | 設定の解決、構築、実行、一時停止、終了処理、失敗の遷移 | [表示](https://kazukin123.github.io/anet-lab/docs/archify/anet_lab_40_runtime_lifecycle.html) | [html](anet_lab_40_runtime_lifecycle.html) / [json](anet_lab_40_runtime_lifecycle.archify.json) |

## 生成根拠の Git revision

`0f4fd0f9718bddfd0e8f0ea5a9248920b0da209b` の作業ツリーを根拠としています。`src/`、`apps/runner/config/`、ビルド定義に未コミットの変更はありません。
前回（`ad6ce58f`）からの変更のうち、図に効くものを反映しました。Metrics Viewer の Run 比較（設定表・範囲統計・タブ）、Run フォルダへの設定の写しと Viewer の cache、Notifier の並行な登録と解除、内的報酬と価値の分解、設定の誤りを完全キーで示す fail-fast、DQN のファイル名の短縮です。

## Evidence gap

- 学習シーケンスは `SerialTrainRunner` と `DefaultDQNAgent` の代表経路です。`PipelineTrainRunner` は差分だけをカードで示し、Rainbow、MuZero、ImageCls の内部は同じ深さで追っていません。
- RND の内的報酬と、外的・内的の 2 Head による価値の分解（`decomposed`）は、`DefaultDQNAgent::UpdateFromBatch` の報酬の渡し方だけをカードで示しています。RND の predictor の学習、rollout の補助学習、Head ごとの勾配の扱いは追っていません。
- Metrics Viewer の Run 比較は、REST の入口（`config.json`・`range-stats.json`）と読み手（`RunConfigService`・`RangeStatsQuery`）を確かめました。ブラウザ側の描画（`metrics-viewer.js`）と、タグセットの保存（`metricsviewer.tag-sets-file`）は図に入れていません。
- 重みの初期化（fan_in に応じた分散の指定）とテストのタグの改名は、図の粒度に現れないため描いていません。
- libtorch、wxWidgets、Box2D、Tracy、Catch2、ALE、Spring Boot、sqlite-jdbc は CMake と `pom.xml` の宣言から確かめました。C++ の build と test、実 Run、Metrics Viewer の起動は行っていません。
- Optuna harness、metrics の圧縮、GUI パネル個々の描画、EvalPanel（鏡写しインスタンス）の駆動は図の範囲外です。
- TensorBoard・MLflow bridge と `inspect_run.py` は、読むファイル（`metrics.jsonl`、`metrics.jsonl.gz`、`metrics_cache.db`）だけを確かめました。外部サービスへの送信は確かめていません。
- 例外で main loop を抜けたあとの終了順は、`OnExceptionInMainLoop` が false を返したときの wxWidgets の既定動作に依存します。wxWidgets 本体は読んでいません。

## 主なコード根拠

- 起動と終了: [RunnerApp.cpp](../../src/apps/runner/src/RunnerApp.cpp) の `OnInit`・`CompleteRunInitialization`・`OnRun`・`OnExit`、[RunnerFrame.cpp](../../src/apps/runner/src/RunnerFrame.cpp) の `OnClose`。
- workspace と設定: [app_util.cpp](../../src/core/base/src/app_util.cpp) の `GetDataRootDir`・`WorkspaceService::Resolve`・`CreateWorkspaceConfigManager`。
- 構築と反復: [trainer.cpp](../../src/core/base/src/trainer.cpp) の `RunManager::RunManager`・`SerialTrainRunner::DoStep`・`PipelineTrainRunner::DoStep`・`RunnerThread::ProcessStep`、[thread.cpp](../../src/core/base/src/thread.cpp) の `ThreadBase::ThreadMain`。
- Agent の系統: [dqn.cpp](../../src/algo/dqn/src/dqn.cpp) の `InitDqn`、[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) の `CreateActor`・`UpdateFromBatch`、[dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) の `Actor::MakeAction`、[dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) の `Learner::UpdateFromBatch`・`SetupReplayBuffer`。
- 評価と通知: [rl.hpp](../../src/core/base/include/anet/rl.hpp) の `Notifier`（登録・解除・通知を同時に呼べる契約）、[observers.cpp](../../src/core/base/src/observers.cpp) の `EpisodeEvalObserver::OnLearn`・`Shutdown`、[rl.cpp](../../src/core/base/src/rl.cpp) の `Notifier::Shutdown`。
- Metrics の書き手: [metrics_logger.hpp](../../src/core/base/include/anet/metrics_logger.hpp) の `LogScalar`・`LogTrace`、[metrics_logger.cpp](../../src/core/base/src/metrics_logger.cpp) の `Log(tag, ConfigData)`（`config/<tag>.txt`）。読み手: [MetricsIngestor.java](../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsIngestor.java)、[RunConfigService.java](../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/RunConfigService.java)、[RangeStatsQuery.java](../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/RangeStatsQuery.java)、[metrics_source.py](../../apps/tools/metrics_source.py)。cache の置き場所: [MetricsCacheDatabase.java](../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/infra/MetricsCacheDatabase.java)、扱い: [ADR0015](../adr/0015-metrics-cache-disposable-derivative.md)。

## doc/code drift

図は実装に合わせ、原文は変更していません。

- [全体概要](../design/010_framework_overview.jp.md) §4.1 の本文と §6.11 は基盤を `anet-core` と呼びます。CMake の target は `anet-core-base`（テスト用の口は `anet-core-testing`）で、Agent の系統と Env は `anet-algo-*`・`anet-env-*` に分かれています。
- 同 §4.2 のコードマップは、`apps/runner/` に画面パネル、`apps/metrics-viewer/` に Java/Spring 製 Metrics Viewer があると書きます。ソースは `src/apps/runner/src` と `src/apps/metrics-viewer` にあり、`apps/` には実行体・設定・jar・launcher が残ります（[ソース配置](../design/011_source_layout.jp.md)と[アプリケーションとツール](../design/160_applications_and_tools.jp.md) §1.3 はこの配置です）。
- 同 §4.2 と §5.2 の Env の一覧に Atari がありません。`src/env/atari` は実在し、`ANET_ENABLE_ATARI` と `ALE_ROOT` で build に入ります（[環境](../design/120_environments.jp.md)には記載があります）。
- 同 §5.3 は Metrics Viewer を「scalar メトリクスの比較」とだけ書きます。現行の Viewer は Config タブで Run 間の設定の差を表にし、tag の範囲の統計も出します（[Metrics Viewer](../design/210_metrics_viewer.jp.md)には記載があります）。
- 同 §6.7.1 は起動時の登録を「具象 Env を repository へ登録」とだけ書きます。実装は `InitDqn`・`InitMuZero`・`InitImageCls` で Agent の系統も登録し、`RunManager` を背景の構築 thread で作ります。

## 言語について

説明文は日本語で、型名、関数名、設定キー、ファイル名は原表記です。
日本語は Viewer UI の対応 locale ではないため `meta.locale` を省略しています。
固定 Viewer UI と `<html lang>` は英語です。

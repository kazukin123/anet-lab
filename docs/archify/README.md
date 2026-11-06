# anet-lab Archify Atlas

現行 checkout を根拠に、anet-lab の全体像から実行時詳細までを 5 枚の Archify マップで表した索引です。
すべての図は同じ canonical term（Run、RunManager、Runner、Agent、Actor、Learner、BatchEnv、
NetworkModel、ReplayBuffer、Notifier、Observer、MetricsLogger、Run 成果物）を使っています。

## 推奨閲覧順

`表示` は GitHub Pages 上のレンダリング結果、`html` と `json` はリポジトリ内の実ファイルです。

| 順序 | 図 | 何が分かるか | 表示 | ファイル |
|---:|---|---|---|---|
| 1 | システム構成 | 主要コンポーネント、主経路、スレッド境界、外部依存 | [表示](https://kazukin123.github.io/anet-lab/docs/archify/anet_lab_00_system_architecture.html) | [html](anet_lab_00_system_architecture.html) / [json](anet_lab_00_system_architecture.archify.json) |
| 2 | Run 実行ワークフロー | workspace 選択から成果物分析までの工程 | [表示](https://kazukin123.github.io/anet-lab/docs/archify/anet_lab_10_run_workflow.html) | [html](anet_lab_10_run_workflow.html) / [json](anet_lab_10_run_workflow.archify.json) |
| 3 | 学習ステップの呼び出し順序 | 1 step の呼び出しと通知の順序 | [表示](https://kazukin123.github.io/anet-lab/docs/archify/anet_lab_20_training_step_sequence.html) | [html](anet_lab_20_training_step_sequence.html) / [json](anet_lab_20_training_step_sequence.archify.json) |
| 4 | 実験データの流れ | 設定、Tensor、経験、metrics、成果物の流れ | [表示](https://kazukin123.github.io/anet-lab/docs/archify/anet_lab_30_experiment_dataflow.html) | [html](anet_lab_30_experiment_dataflow.html) / [json](anet_lab_30_experiment_dataflow.archify.json) |
| 5 | runtime の状態遷移 | 構築、実行、保留、完了、失敗の遷移 | [表示](https://kazukin123.github.io/anet-lab/docs/archify/anet_lab_40_runtime_lifecycle.html) | [html](anet_lab_40_runtime_lifecycle.html) / [json](anet_lab_40_runtime_lifecycle.archify.json) |

## 生成根拠の Git revision

`c07622e58077ed6cc21bb7194aec4b1f9281b4b0` の作業ツリーを根拠としています。未コミットの Actor カタログ移行も含みます。
ActorRequest による生成、学習側 counts、評価セッション、背景評価の例外回収を現行実装へ合わせました。

## Evidence gap

- 学習シーケンスは `SerialTrainRunner` と `DefaultDQNAgent` の代表経路です。Rainbow、MuZero、ImageCls の内部経路を同じ深さでは追跡していません。
- Atari/ALE と外部依存は CMake 宣言を根拠にしています。C++ の build/test や実 Run は実行していません。
- Optuna harness と metrics 圧縮の内部工程、GUI パネル個々の描画処理は図の範囲外です。
- 背景評価の終了時 drain は正常な次セッション起動時とは別です。`EpisodeEvalObserver` destructor の wait と `RunnerApp::OnExit` のログ終了順序を、セッション完了保証としては描いていません。
- bridge はファイル境界の消費者として示しています。今回の更新では外部サービスへの送信や bridge の実行検証は行っていません。

## 主なコード根拠

- 起動と終了: [RunnerApp.cpp](../../apps/runner/src/RunnerApp.cpp) の `OnInit`、`OnRun`、`OnExit`、`OnExceptionInMainLoop`。
- Actor 生成と実行: [trainer.cpp](../../core/anet-core/src/trainer.cpp) の `RunManager`、`SerialTrainRunner::DoStep`、`EvalRunner::RunSession`。
- Actor カタログ: [default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp) の `CreateActor`、[rl.hpp](../../core/anet-core/include/anet/rl.hpp) の `ActorRequest`。
- 行動と学習: [dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp) の `Actor::MakeAction`、`Learner::UpdateFromBatch`。
- 評価と通知: [observers.cpp](../../core/anet-core/src/observers.cpp) の `EpisodeEvalObserver`、[util.hpp](../../core/anet-core/include/anet/util.hpp) の `IntervalGate`。
- Metrics の書き手: [metrics_logger.cpp](../../core/anet-core/src/metrics_logger.cpp)。読み手: [MetricsSource.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/infra/MetricsSource.java)。cache の扱い: [ADR0015](../adr/0015-metrics-cache-disposable-derivative.md)。

## doc/code drift

[全体概要](../design/010_framework_overview.jp.md) §6.7.3 は評価を「学習更新数が interval に達したとき」と説明しています。
実装の `EpisodeEvalObserver::OnLearn` が使う `IntervalGate::ShouldFire` は初回呼び出しで発火し、以後は bucket 境界で発火します。
図ではこの実装に合わせ、原文は変更していません。

## 言語について

説明文は日本語で、型名、関数名、設定キー、ファイル名は原表記です。
日本語は Viewer UI の対応 locale ではないため `meta.locale` を省略しています。
固定 Viewer UI と `<html lang>` は英語です。

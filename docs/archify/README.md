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

`8755992cd26f751b79a35700ec38e600af29d581`（branch `main`）の作業ツリーを調査対象としています。
未コミットの作業ツリー変更も現行 checkout の一部として扱っています。

## Evidence gap

- 学習ステップの順序図は `DefaultDQNAgent` 系の `SerialTrainRunner` を代表として描いています。`RainbowAgent`、
  `MuZeroAgent`、`ImageClsAgent` の更新経路は同じ深さまで追跡しておらず、図に反映していません。
- Atari/ALE モジュールは `ANET_ENABLE_ATARI` と外部 `ALE_ROOT` 参照によるオプショナルビルドです。この checkout では
  `ALE_ROOT` が設定されておらず、`CMakeLists.txt` の宣言だけを根拠にしています。実ビルド構成は未確認です。
- `viewers/metrics-tools/` の `tb_bridge.py`、`mlflow_bridge.py`、`inspect_run.py` は、入出力と冒頭の記述から
  「Run ディレクトリだけを読む」ことを確認しています。実行して出力形式まで検証してはいません。
- Optuna harness（`apps/runner/tools/dropmerge_optuna.py`）と metrics 圧縮ツールは、5 図の主経路に含めていません。
- GUI パネル（`QValuePanel`、`HeatMapPanel`、`Conv2dPanel` など）の内部構成は、システム構成図の
  `AnetRLRunner` と Observer 購読という粒度までしか調べていません。
- C++ の build と test はこの Atlas 作成作業に含めておらず、実行していません。

## doc/code drift

- 評価の起動条件: [`docs/design/010_framework_overview.jp.md`](../design/010_framework_overview.jp.md) 6.7.3 は
  「学習更新数が interval に達したとき」と記述します。実装 `EpisodeEvalObserver::OnLearn`
  （`core/anet-core/src/observers.cpp:552-587`）は `anet::IntervalGate::ShouldFire`
  （`core/anet-core/include/anet/util.hpp:236-270`）を使い、**初回呼び出しは step 値によらず必ず発火**し、
  以後は bucket を跨いだ最初の呼び出しだけで発火して catch-up しません（ADR 0028）。
  図は実装側の挙動に合わせています。
- Metrics Viewer の配置: 設計資料のコードマップは `apps/metrics-viewer/` だけを挙げており、実装もそこにあります。
  一方で作業ツリーには空の `viewers/metrics-viewer/` が残っています。図は実装のある `apps/metrics-viewer/` を採用しました。

## 言語について

authored content（タイトル、ノード名、関係ラベル、カード）は日本語です。型名、関数名、設定キー、
ファイル名、プロトコル、製品名は原表記のまま残しています。

日本語は Archify の Viewer UI がサポートする locale ではないため `meta.locale` を設定していません。
その結果、各 HTML の**固定 Viewer UI（Light / Dark、Classic、Present、Export、Legend、PATH / MAP / LENS など）と
`<html lang>` は英語**になります。

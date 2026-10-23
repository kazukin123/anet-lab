# 開発プロセス・開発ツール・テスト基盤（311）

## 概要・責務

- CLIはGUIによる標準開発を補い、AIエージェントのビルド・検証とCI相当の再現、問題の診断に使う（出典: [開発環境構築ガイド](../../design/040_development_environment.jp.md)、§1.1・§2.4）。
- Optuna harnessの共通runtimeは探索、実行、集計、成果物を扱い、RunnerProcessManagerが子processの起動・timeout・中断・終了を管理する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§2.3・§3.3・§7.3）。
- wikiの取り込み・点検は正本のまとめと機械的な一覧を保ち、台帳buildは派生一覧と地図の台帳区画を更新する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§7.4）。
- reports/toolsの統計ツールと、各CI workflowの個別の概要・責務は、設計文書に記述なし。

## 成立の経緯

- 2026-08-10: Run 分類は物理 workspace フォルダで行い、workspace を自己完結の箱とする（出典: [ADR 0021](../../adr/0021-run-classification-by-workspace-folder.md)）。
- 2026-08-12: MLflow DB は選択 workspace の runs 配下に置く（出典: [ADR 0022](../../adr/0022-mlflow-database-lives-with-workspace-runs.md)）。
- 2026-08-12: Workspace 機構(Run 分類の物理フォルダ化と Runner/Viewer/optuna 統合)（出典: [PRD046](../../memo/done/046_workspace_10prd.md)、題・目的）。
- 2026-08-15: AI向けRun解析CLI `inspect_run.py`（出典: [PRD054](../../memo/done/054_inspect_run_10prd.md)、題・目的）。
- 2026-08-17: interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化（出典: [PRD053](../../memo/done/053_interval_gate_perf_ema_10prd.md)、題・目的）。
- 2026-08-17: 解析用メタデータは Runner が出力し、解析側は設定から再導出しない（出典: [ADR 0029](../../adr/0029-analysis-metadata-emitted-by-runner.md)）。
- 2026-08-21: DropMerge探索を起動するOptuna harnessを追加した（出典: [PRD016](../../memo/done/016_dropmerge_optuna_10prd.md)、題・目的）。
- 2026-08-21: 043: GitHub Actions リリースパッケージング(実行用 bin パッケージ)（出典: [PRD043](../../memo/done/043_release_packaging_10prd.md)、題・目的）。
- 2026-09-05: メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す（出典: [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md)、題・目的）。
- 2026-09-05: 個体メトリクスは trace チャネルで出し、評価セッションは採用 episode ごとの EpisodeEndEvent と SessionEndEvent を分けて発火する（出典: [ADR 0037](../../adr/0037-metrics-trace-channel-and-session-end-event.md)）。
- 2026-09-19: Actor 設定カタログと eval スロットごとの方策 / network 指定（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、題・目的）。
- 2026-09-22: 終了時の評価セッション排水（出典: [PRD076](../../memo/done/076_eval_session_drain_on_exit_10prd.md)、題・目的）。
- 2026-10-18: categories.jsonの対応を直し、apps/runner/srcのファイルを319 GUI共通基盤から110 RunnerAppへ分類するようにした（出典: [PRD095](../../memo/095_llm_wiki_10prd.md)、§3「実装を分ける語」）。

## 選ばなかった形

- inspect_runによるcacheの生成・更新・修復・migration・削除 — 不採用: 読み取り専用の抽出ツールから分ける（出典: [PRD054](../../memo/done/054_inspect_run_10prd.md)、§10）。
- inspect_runによるmetricの意味解釈・採否判断・Run ranking — 不採用: 分析の判断をツールへ統合しない（出典: [PRD054](../../memo/done/054_inspect_run_10prd.md)、§10・§11）。
- inspect_runのraw全点の無制限出力・plot画像生成・interactive viewer — 不採用: 分析用の材料抽出と分ける（出典: [PRD054](../../memo/done/054_inspect_run_10prd.md)、§10）。
- 監査の指摘の本文や対処方針を地図にも要約する — 不採用: 台帳の派生一覧と同じ正本の要約が2か所になる（出典: [PRD095](../../memo/095_llm_wiki_10prd.md)、§1の非ゴール・§6）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- workspaceの箱を移動・削除・退避しても、Run・cache・Optuna履歴・基底configが一体で移動する自己完結性を保つ（出典: [PRD046](../../memo/done/046_workspace_10prd.md)、§1の不変条件）。
- inspect_runのstreaming・1 pass・128点上限をfocused testで確かめる（出典: [PRD054](../../memo/done/054_inspect_run_10prd.md)、§9・§10）。
- inspect_runで128点以下は全点を保持し、間引く場合も先頭・末尾と決定性を保つ（出典: [PRD054](../../memo/done/054_inspect_run_10prd.md)、§9）。
- 台帳の分類をstatsと共有する変更をした場合は、statsのカテゴリ集計が変わらないことを確かめる（出典: [PRD095](../../memo/095_llm_wiki_10prd.md)、§4.7の台帳のツール・受入15）。

## 関係

- [ANET フレームワーク全体概要](../../design/010_framework_overview.jp.md)。
- [ANET ドキュメント](../../design/README.jp.md)。
- 設計: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md) §2.3・§3.3・§7.3・§8。
- ユーザーガイド: [開発環境構築ガイド](../../design/040_development_environment.jp.md) §6・§7、[Run分析ガイド](../../design/030_user_guide_analysis.jp.md) §1。
- 含まれるアプリケーション: [RunnerApp](../apps/110_runner_app.md)。

## ファイル

点検が作り直す。手で直さない。

- [push_review.py](../../../.github/scripts/push_review.py)
- [gemini_ai_review.yml](../../../.github/workflows/gemini_ai_review.yml)
- [pages.yml](../../../.github/workflows/pages.yml)
- [release.yml](../../../.github/workflows/release.yml)
- [windows-ci.yml](../../../.github/workflows/windows-ci.yml)
- [batchrun_fatal_error_handling_test.ps1](../../../apps/runner/tools/batchrun_fatal_error_handling_test.ps1)
- [compress_workspace_metrics.py](../../../apps/runner/tools/compress_workspace_metrics.py) / [compress_workspace_metrics_test.py](../../../apps/runner/tools/compress_workspace_metrics_test.py)
- [dropmerge_optuna.py](../../../apps/runner/tools/dropmerge_optuna.py)
- [optuna_common.py](../../../apps/runner/tools/optuna_common.py)
- [optuna_metrics_gzip_test.py](../../../apps/runner/tools/optuna_metrics_gzip_test.py)
- [optuna_workspace_test.py](../../../apps/runner/tools/optuna_workspace_test.py)
- [resolve_workspace.bat](../../../apps/runner/tools/resolve_workspace.bat) / [resolve_workspace_test.ps1](../../../apps/runner/tools/resolve_workspace_test.ps1)
- [symbolize.py](../../../apps/runner/tools/symbolize.py)
- [watch_log.ps1](../../../apps/runner/tools/watch_log.ps1)
- [categories.json](../../../reports/tools/categories.json)
- [registry.py](../../../reports/tools/registry.py)
- [stats.py](../../../reports/tools/stats.py)
- [wiki_support.py](../../../reports/tools/wiki_support.py)
- [atari5.py](../../../viewers/metrics-tools/atari5.py) / [atari5_test.py](../../../viewers/metrics-tools/atari5_test.py)
- [btr_to_metrics.py](../../../viewers/metrics-tools/btr_to_metrics.py)
- [inspect_run.py](../../../viewers/metrics-tools/inspect_run.py) / [inspect_run_test.py](../../../viewers/metrics-tools/inspect_run_test.py)
- [metrics_source.py](../../../viewers/metrics-tools/metrics_source.py) / [metrics_source_test.py](../../../viewers/metrics-tools/metrics_source_test.py)
- [metrics_viewer.py](../../../viewers/metrics-tools/metrics_viewer.py)
- [mlflow_bridge.py](../../../viewers/metrics-tools/mlflow_bridge.py) / [mlflow_bridge_test.py](../../../viewers/metrics-tools/mlflow_bridge_test.py)
- [requirements.txt](../../../viewers/metrics-tools/requirements.txt)
- [start_mlflow.bat](../../../viewers/metrics-tools/start_mlflow.bat)
- [start_tb.bat](../../../viewers/metrics-tools/start_tb.bat)
- [tb_bridge.py](../../../viewers/metrics-tools/tb_bridge.py) / [tb_bridge_test.py](../../../viewers/metrics-tools/tb_bridge_test.py)
- [reload.lua](../../../viewers/metrics-tools/vlc/lua/extensions/reload.lua)

## PRD の一覧

点検が作り直す。手で直さない。

- [PRD046](../../memo/done/046_workspace_10prd.md) — Workspace 機構(Run 分類の物理フォルダ化と Runner/Viewer/optuna 統合)
- [PRD053](../../memo/done/053_interval_gate_perf_ema_10prd.md) — PRD 053: interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化
- [PRD054](../../memo/done/054_inspect_run_10prd.md) — PRD 054: AI向けRun解析CLI `inspect_run.py`
- [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md) — メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す
- [PRD076](../../memo/done/076_eval_session_drain_on_exit_10prd.md) — PRD 076: 終了時の評価セッション排水
- [PRD094](../../memo/094_evalpanel_step_history_10prd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD
- [PRD095](../../memo/095_llm_wiki_10prd.md) — LLM-Wiki 型のまとめの層（PRD の上流で、知見を題材と機構ごとに、実装の地図をアプリケーションと機能カテゴリごとにまとめる）PRD
- [PRD097](../../memo/097_env_auto_reset_after_notification_10prd.md) — 遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD

## 台帳の指摘

台帳のツールが作り直す。手で直さない。

<!-- registry:begin -->
なし

[台帳の一覧](../../../reports/registry/findings.md)
<!-- registry:end -->

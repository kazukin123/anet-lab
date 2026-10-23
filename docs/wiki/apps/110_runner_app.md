# RunnerApp（110）

## 概要・責務

- RunnerAppはconfig、RunManager、RunnerThread、GUI、loggingのlifecycleを統括し、RunnerThreadが学習RunnerをUI threadから分離する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§2.1・§5.1）。
- RunnerFrameは画面と操作を、TrainPanelは学習表示を、EvalPanelは専用EvalRunnerとmodel同期を受け持つ。QValuePanelは行動候補を可視化する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§2.1・§3.1）。
- apps/runner/toolsのOptuna harnessはRunner子processとRun artifactの境界を扱い、workspace共通基盤と補助launcherを接続する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§2.3・§3.3・§6）。

- EvalTracePanelはEvaluation Traceの行・表示履歴を保持し、EvalPanel・QValuePanel・Conv2dを選択した遷移へ追従させる（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§3.4）。

## 成立の経緯

- 2026-07-18: ImageCls batch-native env の seam：EnvRepository を single/batch variant 化＋Factory/Builder 再レイヤ（出典: [ADR 0009](../../adr/0009-imagecls-batch-env-seam.md)）。
- 2026-07-20: ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、題・目的）。
- 2026-08-10: Run 分類は物理 workspace フォルダで行い、workspace を自己完結の箱とする（出典: [ADR 0021](../../adr/0021-run-classification-by-workspace-folder.md)）。
- 2026-08-12: MLflow DB は選択 workspace の runs 配下に置く（出典: [ADR 0022](../../adr/0022-mlflow-database-lives-with-workspace-runs.md)）。
- 2026-08-12: Workspace 機構(Run 分類の物理フォルダ化と Runner/Viewer/optuna 統合)（出典: [PRD046](../../memo/done/046_workspace_10prd.md)、題・目的）。
- 2026-08-14: configured eval の定義と定期駆動を別 namespace に分離し、生成は schedule エントリが駆動する（出典: [ADR 0027](../../adr/0027-eval-definition-schedule-separation.md)）。
- 2026-08-19: PRD 055: Runner ツールバー(対象軸 4 バー構成)（出典: [PRD055](../../memo/done/055_runner_toolbar_10prd.md)、題・目的）。
- 2026-09-08: Actor 設定は名前付きカタログ参照とし、Agent インタフェースから RunMode を外す（出典: [ADR 0038](../../adr/0038-actor-config-catalog-without-runmode.md)）。
- 2026-09-19: Actor 設定カタログと eval スロットごとの方策 / network 指定（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、題・目的）。
- 2026-09-22: 終了時の背景評価セッションは完走を既定とし、設定または上限時間による協調キャンセルで抜けられるようにして、部分集計は設けない（出典: [ADR 0045](../../adr/0045-eval-session-drain-on-exit-wait-or-discard.md)）。
- 2026-09-22: 終了時の評価セッション排水（出典: [PRD076](../../memo/done/076_eval_session_drain_on_exit_10prd.md)、題・目的）。
- 2026-09-24: device 指定の auto 対応とキー体系統一（出典: [PRD070](../../memo/070_device_spec_auto_10prd.md)、題・目的）。
- 2026-10-22: Evaluation Traceの表示・操作を、内容に応じた列幅・システム色・再生／一時停止トグルを持つアイコン操作・Trace中の操作バー赤枠、低速倍率の小数表示、操作バーと行動列までを守る最小幅、行選択に依存しない列見出し配色、truncated列の短い見出しtrunc、action直後のreward列へ調整する仕様を定めた（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、D10〜D13・§8のUI受入）。

## 選ばなかった形

- workspace選択のCancelで `_default` へ進む案 — 不採用: 選んでいないworkspaceの箱が作られる副作用がある（出典: [PRD046](../../memo/done/046_workspace_10prd.md)、§2.1）。
- Runner GUIで実行中にworkspaceを切り替える — 範囲外: 現状は再起動で切り替え、後続PRDへ分ける（出典: [PRD046](../../memo/done/046_workspace_10prd.md)、§7）。
- `RunManager` が `EpisodeEvalObserver` を直接保持して排水する案 — 不採用: 背景スレッドを持つobserver一般の終了の口を優先した（出典: [PRD076](../../memo/done/076_eval_session_drain_on_exit_10prd.md)、Complexity audit・9）。
- `RunnerApp::DrainBackgroundObservers()` で例外を捕捉する案 — 不採用: 既存の再送出契約とfatal経路でexit code 1へ到達でき、捕捉層を足さない（出典: [PRD076](../../memo/done/076_eval_session_drain_on_exit_10prd.md)、Complexity audit・11）。
- Evaluation Traceに学習側の優先度を表示する — 不採用: EvalPanelは1つのnetworkのforwardしか持たず、replayも無い（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§6）。
- Evaluation TraceでAgentの入力を表示する — 先送り: 画像Tensorの部分表示は読みにくい。再開条件: 必要と分かったとき、選択行の入力を文字列でログへ出す形から起こす（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§6）。
- Evaluation Traceの列を折れ線で表示する — 先送り: まず表と再生で足りるかを見る。再開条件: 近似優先度の推移を一目で見たくなったとき（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§6）。
- Evaluation Traceで条件に合う行を自動で一時停止する — 先送り: まず再生で場面を探す。再開条件: その手間が大きいと分かったとき（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§6）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- ImageCls batch入力の変更前後で、trainのmax_steps episode、episode_start、Conv2d表示のcadenceを維持する（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、§1.3）。
- close時のTrain停止、checkpoint保存、log/metricsのflush、EvalPanel detach、AUI破棄の順序を維持する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§7.1）。
- Train panelのFPSが0でも学習は続け、実行時のFPS選択をconfig dumpへ書き戻さない（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§7.1）。
- Actorを追加しても既存Actorの行動列を変えず、同一checkpoint・同一seedで全スロットの結果がEvalPanel起動中でも一致することを確かめる（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、§7・§8の受入9）。
- メトリクス配線の整理前後で、同seed・決定的モードの短いRunのmetrics checksum、scalar/traceの定義、WARN行を一致させる（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、D3・§8の受入1）。
- ActorのQヒントのヘルパー切り出し前後で、actor_approxの同seed・決定的モードの短いRunのmetrics checksumを一致させる（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§8の受入3）。
- EvalRunnerと学習側のカウンタを数える時機はそろえず、既存のmetricsのstepをずらさない（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、D9）。

## 関係

- 設計: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md) §2.1・§3.1・§5.1・§7.1・§8。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md) §2・§3。
- 含む機能カテゴリ: [311 開発ツール](../categories/311_dev_tools.md)。

## ファイル

点検が作り直す。手で直さない。

- [Conv2dPanel.cpp](../../../apps/runner/src/Conv2dPanel.cpp) / [Conv2dPanel.hpp](../../../apps/runner/src/Conv2dPanel.hpp)
- [ErrorDialog.cpp](../../../apps/runner/src/ErrorDialog.cpp) / [ErrorDialog.hpp](../../../apps/runner/src/ErrorDialog.hpp)
- [EvalPanel.cpp](../../../apps/runner/src/EvalPanel.cpp) / [EvalPanel.hpp](../../../apps/runner/src/EvalPanel.hpp)
- [EvalTracePanel.cpp](../../../apps/runner/src/EvalTracePanel.cpp) / [EvalTracePanel.hpp](../../../apps/runner/src/EvalTracePanel.hpp) / [EvalTracePanel_test.cpp](../../../apps/runner/src/EvalTracePanel_test.cpp)
- [HeatMapPanel.cpp](../../../apps/runner/src/HeatMapPanel.cpp) / [HeatMapPanel.hpp](../../../apps/runner/src/HeatMapPanel.hpp)
- [LogPanel.cpp](../../../apps/runner/src/LogPanel.cpp) / [LogPanel.hpp](../../../apps/runner/src/LogPanel.hpp)
- [PlotPanel.cpp](../../../apps/runner/src/PlotPanel.cpp) / [PlotPanel.hpp](../../../apps/runner/src/PlotPanel.hpp)
- [QValuePanel.cpp](../../../apps/runner/src/QValuePanel.cpp) / [QValuePanel.hpp](../../../apps/runner/src/QValuePanel.hpp)
- [RunnerApp.cpp](../../../apps/runner/src/RunnerApp.cpp) / [RunnerApp.hpp](../../../apps/runner/src/RunnerApp.hpp)
- [RunnerFrame.cpp](../../../apps/runner/src/RunnerFrame.cpp) / [RunnerFrame.hpp](../../../apps/runner/src/RunnerFrame.hpp)
- [TrainPanel.cpp](../../../apps/runner/src/TrainPanel.cpp) / [TrainPanel.hpp](../../../apps/runner/src/TrainPanel.hpp)
- [WorkspaceDialog.cpp](../../../apps/runner/src/WorkspaceDialog.cpp) / [WorkspaceDialog.hpp](../../../apps/runner/src/WorkspaceDialog.hpp)
- [pch.hpp](../../../apps/runner/src/pch.hpp)
- [runner.ico](../../../apps/runner/src/runner.ico) / [runner.manifest](../../../apps/runner/src/runner.manifest) / [runner.rc](../../../apps/runner/src/runner.rc)

### 311 開発プロセス/開発ツール/テスト基盤

- [batchrun_fatal_error_handling_test.ps1](../../../apps/runner/tools/batchrun_fatal_error_handling_test.ps1)
- [compress_workspace_metrics.py](../../../apps/runner/tools/compress_workspace_metrics.py) / [compress_workspace_metrics_test.py](../../../apps/runner/tools/compress_workspace_metrics_test.py)
- [dropmerge_optuna.py](../../../apps/runner/tools/dropmerge_optuna.py)
- [optuna_common.py](../../../apps/runner/tools/optuna_common.py)
- [optuna_metrics_gzip_test.py](../../../apps/runner/tools/optuna_metrics_gzip_test.py)
- [optuna_workspace_test.py](../../../apps/runner/tools/optuna_workspace_test.py)
- [resolve_workspace.bat](../../../apps/runner/tools/resolve_workspace.bat) / [resolve_workspace_test.ps1](../../../apps/runner/tools/resolve_workspace_test.ps1)
- [symbolize.py](../../../apps/runner/tools/symbolize.py)
- [watch_log.ps1](../../../apps/runner/tools/watch_log.ps1)

### 機能カテゴリなし

- [CMakeLists.txt](../../../apps/runner/CMakeLists.txt)
- [Atari.txt](../../../apps/runner/config/Atari.txt)
- [CartPole.txt](../../../apps/runner/config/CartPole.txt)
- [DropMerge.txt](../../../apps/runner/config/DropMerge.txt)
- [DropMerge_optuna.txt](../../../apps/runner/config/DropMerge_optuna.txt)
- [GridMaze.txt](../../../apps/runner/config/GridMaze.txt)
- [GridMaze_muzero.txt](../../../apps/runner/config/GridMaze_muzero.txt)
- [ImageCls.txt](../../../apps/runner/config/ImageCls.txt)
- [LunarLander.txt](../../../apps/runner/config/LunarLander.txt)
- [LunarLander_cpu.txt](../../../apps/runner/config/LunarLander_cpu.txt)
- [_main.txt](../../../apps/runner/config/_main.txt)
- [_workspace_template.txt](../../../apps/runner/config/_workspace_template.txt)
- [agent.txt](../../../apps/runner/config/agent.txt)
- [common.txt](../../../apps/runner/config/common.txt)
- [metrics_image.txt](../../../apps/runner/config/metrics_image.txt)
- [metrics_scalar.txt](../../../apps/runner/config/metrics_scalar.txt)
- [nn.txt](../../../apps/runner/config/nn.txt)
- [nn_cnx.txt](../../../apps/runner/config/nn_cnx.txt)
- [_main.txt](../../../apps/runner/workspaces/Atari-01/config/_main.txt)
- [_main.txt](../../../apps/runner/workspaces/CartPole-01/config/_main.txt)
- [_main.txt](../../../apps/runner/workspaces/DropMerge-01/config/_main.txt)
- [_main.txt](../../../apps/runner/workspaces/GridMaze-01/config/_main.txt)
- [_main.txt](../../../apps/runner/workspaces/GridMaze_muzero-01/config/_main.txt)
- [_main.txt](../../../apps/runner/workspaces/ImageCls-01/config/_main.txt)
- [_main.txt](../../../apps/runner/workspaces/LunarLander-01/config/_main.txt)

## PRD の一覧

点検が作り直す。手で直さない。

- [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [PRD046](../../memo/done/046_workspace_10prd.md) — Workspace 機構(Run 分類の物理フォルダ化と Runner/Viewer/optuna 統合)
- [PRD052](../../memo/done/052_eval_schedule_separation_10prd.md) — PRD 052: eval 定義とスケジュールの namespace 分離
- [PRD053](../../memo/done/053_interval_gate_perf_ema_10prd.md) — PRD 053: interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化
- [PRD057](../../memo/done/057_native_stack_dim_handling_10prd.md) — Stack 次元のネイティブ対応(Reshape ハックを廃する根本対応)
- [PRD060](../../memo/done/060_eval_batch_episodes_10prd.md) — 評価セッション契約: eval_batch_size（lane 数）と eval_episodes（本数）で N 本平均を 1 点として記録する
- [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [PRD062](../../memo/done/062_plasticity_metrics_10prd.md) — 可塑性メトリクス（srank / dormant unit 率）PRD
- [PRD063](../../memo/done/063_plasticity_weight_norm_10prd.md) — 可塑性 weight norm（feature/readout 分割）PRD
- [PRD065](../../memo/done/065_nn_spectral_norm_10prd.md) — NN Spectral Normalization（weight_norm.mode）PRD
- [PRD066](../../memo/done/066_policy_churn_metrics_10prd.md) — Policy churn メトリクス（35_agent_churn）PRD
- [PRD067](../../memo/done/067_MunchausenRL_10prd.md) — Munchausen RL（M-DQN / M-QR / M-IQN）PRD
- [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md) — メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す
- [PRD070](../../memo/070_device_spec_auto_10prd.md) — PRD 070: device 指定の auto 対応とキー体系統一
- [PRD073](../../memo/done/073_replay_fit_metrics_10prd.md) — PRD 073: replay 抽選履歴別の TD・損失メトリクス
- [PRD075](../../memo/done/075_episode_steps_and_eval_session_log_10prd.md) — PRD 075: エピソード長の汎用メトリクス化と評価セッションの実行ログ
- [PRD076](../../memo/done/076_eval_session_drain_on_exit_10prd.md) — PRD 076: 終了時の評価セッション排水
- [PRD082](../../memo/082_atari_ram_progress_metrics_10prd.md) — Atari のゲーム固有イベントを RAM メトリクスとして取る — 番号付き汎用キーへゲーム別に紐づける PRD
- [PRD083](../../memo/083_action_context_retire_10prd.md) — ActionContext の廃止（Actor が RNG と frame stacker を直接持つ）
- [PRD084](../../memo/084_noisynet_10prd.md) — NoisyNet (Noisy Networks for Exploration) PRD
- [PRD092](../../memo/092_gradient_noise_scale_10prd.md) — 勾配ノイズの大きさ（gradient noise scale）の計測 PRD
- [PRD093](../../memo/093_attention_entropy_diagnostics_10prd.md) — TransformerEncoder の attention エントロピー診断 PRD
- [PRD094](../../memo/094_evalpanel_step_history_10prd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD
- [PRD097](../../memo/097_env_auto_reset_after_notification_10prd.md) — 遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD

## 台帳の指摘

台帳のツールが作り直す。手で直さない。

<!-- registry:begin -->
- F-0018 P2 refactor — QValuePanel::Updateが356行で、統計・Grid更新・HeatMap加工を1関数で行う
- F-0024 P2 drift — nn.txtのbodyのサンプル2つ（@ResNet・@MLP_RES）が、選ぶとエラーになる
- F-0022 P3 drift — metrics_scalar.txt冒頭のscalar記法のミニ仕様が古い（&step、@learn/@trainだけ、属性の欠落）
- F-0023 P3 drift — nn.txtの対応typeの一覧とinit.modeのコメントが実装とずれている

[台帳の一覧](../../../reports/registry/findings.md)
<!-- registry:end -->

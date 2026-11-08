# 320 Env 共通基盤

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機能カテゴリ](../index.md#機能カテゴリ)

## 概要・責務

- SingleDiscreteEnvとBatchEnv、spec・state・Reset・Step、factoryとBuilderの共通契約を扱う。（出典: [環境](../../design/120_environments.jp.md)、§1.1・§2・§3）。

## 成立の経緯

- 2026-07-18: ImageCls batch-native env の seam：EnvRepository を single/batch variant 化＋Factory/Builder 再レイヤ（出典: [ADR 0009](../../adr/0009-imagecls-batch-env-seam.md)、本文）。
- 2026-07-20: ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、題・目的）。
- 2026-08-21: Envインスタンスname PRD（出典: [PRD037](../../memo/done/037_env_instance_name_10prd.md)、題・目的）。
- 2026-08-21: Envログ name prefix 自動付与 PRD（出典: [PRD038](../../memo/done/038_env_log_auto_prefix_10prd.md)、題・目的）。
- 2026-09-03: 評価セッションの N 本集約は BatchEnv デコレータと開始時動的グラントで行う（出典: [ADR 0034](../../adr/0034-eval-session-aggregation-in-batchenv-decorator.md)、本文）。
- 2026-09-04: 評価セッション契約: eval_batch_size（lane 数）と eval_episodes（本数）で N 本平均を 1 点として記録する（出典: [PRD060](../../memo/done/060_eval_batch_episodes_10prd.md)、題・目的）。
- 2026-09-05: メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す（出典: [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md)、題・目的）。
- 2026-09-13: エピソード長は SHARED でも Step() 回数とし、episode return の総和とは非対称にする（出典: [ADR 0043](../../adr/0043-episode-steps-shared-scope-is-step-count.md)、本文）。
- 2026-09-17: PRD 075: エピソード長の汎用メトリクス化と評価セッションの実行ログ（出典: [PRD075](../../memo/done/075_episode_steps_and_eval_session_log_10prd.md)、題・目的）。
- 2026-09-19: PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、題・目的）。
- 2026-09-24: Envのdevice指定を文字列へ移し指定値と実効deviceを分けた（出典: [PRD070](../../memo/done/070_device_spec_auto_10prd.md)、確定したD1・D3・D9）。
- 2026-10-19: Evaluation Trace の列はメトリクスの定義と分けて番号付きで宣言し、遷移の時点に合う target だけを読む（出典: [ADR 0057](../../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)、本文）。
- 2026-10-20: BatchEnv の Step は終端の lane をリセットせず、Runner が通知のあとに終わったエピソードをリセットする（後にADR0060で取消）（出典: [ADR 0059](../../adr/0059-batchenv-step-does-not-reset-runner-resets-ended-episodes-after-notification.md)、本文）。
- 2026-10-20: 遷移の結果は Step の結果で運び、速報値か確定値かは読むイベントで分ける（出典: [ADR 0060](../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)、本文）。
- 2026-10-20: 遷移の結果をStepの結果へ移しBatchEnvの自動リセットをStep内に戻した（出典: [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md)、§4.1 D1・D2）。
- 2026-10-21: Env遷移の値をstep traceへ渡す読み口を接続した（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§4.2 D5・D9）。
- 2026-10-26: Envのframe_skipとは別の層として、Actorが行動を続けるスキップと判断の旗を共通の経験へ渡す境界を定めた（出典: [PRD098](../../memo/098_temporl_skip_policy_10prd.md)、§4 D15・§5.5）。

## 選ばなかった形

- episode開始通知とViewの開始表示 — 範囲外: 今回のゴールに効かず手動操作の時点ずれと同じ関心。再開条件: 948_qvaluepanel_manual_action_state_alignmentで手動の時点ずれを決めるとき（出典: [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md)、§5）。
- ResetEndedEpisodesの段・外から読む値の写し・予備ALE — 不採用: 段の案は非同期学習を5〜8%遅くし、写しは埋め合わせを残し、予備ALEはAtari固有になる（出典: [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md)、§5・§4.1 D1/D2）。
- 速報値と確定値を別の鍵名に分ける形 — 不採用: Step結果が2組の値を持つ作りになり、現用の設定キー名も移行する必要がある（出典: [ADR 0060](../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)、Considered Options）。
- 評価の終わりを driver 側（Runner）で sample 数を数えて決める形 — 不採用: 端数の padding は物理的に Source 側で必須なので処理が二重になり、汎用の EvalRunner が valid 件数という dataset の語彙を持つことになる（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、12.8 D5解決）。
- dormant tag の非生成を native ImageCls だけに限る形 — 不採用: 限るには class_id の比較か batch-native かの Env 種別の分岐が要り、どちらもフレームワークへの染み出しになる（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、12.7 D4解決）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- batch-native移行後も既存Runner・Observer・GUIの接続とEnv nameを維持する（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、1.3 維持する外部contract）。
- Actor設定の移行でRunMode enumとEnv側の用途を変えない（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、5.2 Agent インタフェース）。
- 読み口と配線の整理でmetrics出力・定義レコード・WARN・fail-fastの文言と時機を変えない。同seed・決定的モードの短Runでchecksum・defs・WARNを比較する（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§4.1 D3・§8 1）。
- Agentとreplayへ入る値、loss・Q、episode return・長さを保持する。Atari2構成と他Env・逐次/非同期の7構成で全点を比較し、実時間は交互3組、ばらつきが大きければ5組で性能低下を確認する（出典: [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md)、ゴールG3・§7 1/4）。
- 評価セッションの採用規則と@session_endの集約の読み口を変えない（出典: [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md)、§4.2 4・§4.3.4）。
- スキップを加えてもRunnerのstepとEnvのframe skipを変えない。無効時の4環境のmetrics checksumを着手前の基準と一致させる（出典: [Agentと学習](../../design/110_agents_and_learning.jp.md)、§6.1；[PRD098](../../memo/098_temporl_skip_policy_10prd.md)、§5.9・§7の段0）。

## 関係

- 設計: [環境](../../design/120_environments.jp.md)、§1.1・§2・§3。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [env.hpp](../../../core/anet-core/include/anet/env.hpp)
- [env.cpp](../../../core/anet-core/src/env.cpp) / [env_test.cpp](../../../core/anet-core/src/env_test.cpp)

## PRD の一覧

点検が作り直す。手で直さない。

- [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [PRD060](../../memo/done/060_eval_batch_episodes_10prd.md) — 評価セッション契約: eval_batch_size（lane 数）と eval_episodes（本数）で N 本平均を 1 点として記録する
- [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md) — メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す
- [PRD070](../../memo/done/070_device_spec_auto_10prd.md) — PRD 070: device 指定の auto 対応とキー体系統一
- [PRD075](../../memo/done/075_episode_steps_and_eval_session_log_10prd.md) — PRD 075: エピソード長の汎用メトリクス化と評価セッションの実行ログ
- [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD
- [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md) — 遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD
- [PRD098](../../memo/098_temporl_skip_policy_10prd.md) — TempoRL 対応（スキップ: 選んだ行動を続ける長さの学習）PRD

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| # | ID | 状態 | WSJF | 優先度 | 待ち | 種別 | 工程 | 領域 | 性質 | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [issue-20261103-785e68](../../management/issues/issue-20261103-785e68.md) | 未着手 | 2.0 | 中 | — | 成果物 | PRD | 315_nn, 318_rl_core, 321_agent_core, 324_dqn, 319_gui, 320_env_core, 314_tensor, 312_util, 110_runner_app, 322_env_individual, 325_muzero | 索引 | 未確認 | anet-core の外で実装する拡張点とコード配置の規約 暫定 PRD |
| 2 | [issue-20261103-182ca1](../../management/issues/issue-20261103-182ca1.md) | 未着手 | 1.4 | 中 | — | 成果物 | PRD | 318_rl_core, 324_dqn, 315_nn, 320_env_core | 索引 | 未確認 | 2026-07 のコアの詳細レビューで挙がった性能の改善候補 5 件 暫定 PRD |
| 3 | [issue-20261103-50644a](../../management/issues/issue-20261103-50644a.md) | 未着手 | 1.4 | 中 | — | 成果物 | PRD | 320_env_core, 317_metrics, 321_agent_core, 318_rl_core, 110_runner_app | 索引 | 未確認 | 評価 session の step trace を Run の記録に残す PRD |
| 4 | [issue-20261103-b4e5c9](../../management/issues/issue-20261103-b4e5c9.md) | 未着手 | 1.0 | 中 | — | 成果物 | PRD | 110_runner_app, 317_metrics, 318_rl_core, 321_agent_core, 312_util, 320_env_core, 314_tensor, 316_config, 311_dev_tools, 322_env_individual | 索引 | 未確認 | 設定 root の所有整合(`train.` / `app.run_name` 等の階層ずれ棚卸し) 暫定 PRD |
| 5 | [issue-20261103-c16aab](../../management/issues/issue-20261103-c16aab.md) | 未着手 | 1.0 | 中 | — | 成果物 | PRD | 320_env_core, 322_env_individual, 312_util, 120_metrics_viewer, 311_dev_tools, 318_rl_core | 索引 | 未確認 | スカラー閾値越え率の汎用化（Observation 層 / Viewer 層） 暫定 PRD |
| 6 | [issue-20261103-b13ed2](../../management/issues/issue-20261103-b13ed2.md) | 未着手 | 0.8 | 中 | — | 成果物 | PRD | 120_metrics_viewer, 311_dev_tools, 312_util, 316_config, 320_env_core, 313_profiling | 索引 | 未確認 | リリース物の組み立てを CMake（install・CPack・ワークフロー preset）へ移し、手元でも同じコマンドで作れるようにする PRD |

着手できる 6 / 待ち 0 / 保留 0 / 次版 0 / 解決 0
<!-- issues:end -->

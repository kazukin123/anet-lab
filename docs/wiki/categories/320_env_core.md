# 320 Env 共通基盤

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
- 2026-09-24: Envのdevice指定を文字列へ移し指定値と実効deviceを分けた（出典: [PRD070](../../memo/070_device_spec_auto_10prd.md)、確定したD1・D3・D9）。
- 2026-10-19: Evaluation Trace の列はメトリクスの定義と分けて番号付きで宣言し、遷移の時点に合う target だけを読む（出典: [ADR 0057](../../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)、本文）。
- 2026-10-20: BatchEnv の Step は終端の lane をリセットせず、Runner が通知のあとに終わったエピソードをリセットする（後にADR0060で取消）（出典: [ADR 0059](../../adr/0059-batchenv-step-does-not-reset-runner-resets-ended-episodes-after-notification.md)、本文）。
- 2026-10-20: 遷移の結果は Step の結果で運び、速報値か確定値かは読むイベントで分ける（出典: [ADR 0060](../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)、本文）。
- 2026-10-20: 遷移の結果をStepの結果へ移しBatchEnvの自動リセットをStep内に戻した（出典: [PRD097](../../memo/097_env_auto_reset_after_notification_10prd.md)、§4.1 D1・D2）。
- 2026-10-21: Env遷移の値をstep traceへ渡す読み口を接続した（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§4.2 D5・D9）。

## 選ばなかった形

- episode開始通知とViewの開始表示 — 範囲外: 今回のゴールに効かず手動操作の時点ずれと同じ関心。再開条件: 999_qvaluepanel_manual_action_state_alignmentで手動の時点ずれを決めるとき（出典: [PRD097](../../memo/097_env_auto_reset_after_notification_10prd.md)、§5）。
- ResetEndedEpisodesの段・外から読む値の写し・予備ALE — 不採用: 段の案は非同期学習を5〜8%遅くし、写しは埋め合わせを残し、予備ALEはAtari固有になる（出典: [PRD097](../../memo/097_env_auto_reset_after_notification_10prd.md)、§5・§4.1 D1/D2）。
- 速報値と確定値を別の鍵名に分ける形 — 不採用: Step結果が2組の値を持つ作りになり、現用の設定キー名も移行する必要がある（出典: [ADR 0060](../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)、Considered Options）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- batch-native移行後も既存Runner・Observer・GUIの接続とEnv nameを維持する（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、1.3 維持する外部contract）。
- Actor設定の移行でRunMode enumとEnv側の用途を変えない（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、5.2 Agent インタフェース）。
- 読み口と配線の整理でmetrics出力・定義レコード・WARN・fail-fastの文言と時機を変えない。同seed・決定的モードの短Runでchecksum・defs・WARNを比較する（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§4.1 D3・§8 1）。
- Agentとreplayへ入る値、loss・Q、episode return・長さを保持する。Atari2構成と他Env・逐次/非同期の7構成で全点を比較し、実時間は交互3組、ばらつきが大きければ5組で性能低下を確認する（出典: [PRD097](../../memo/097_env_auto_reset_after_notification_10prd.md)、ゴールG3・§7 1/4）。
- 評価セッションの採用規則と@session_endの集約の読み口を変えない（出典: [PRD097](../../memo/097_env_auto_reset_after_notification_10prd.md)、§4.2 4・§4.3.4）。

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
- [PRD070](../../memo/070_device_spec_auto_10prd.md) — PRD 070: device 指定の auto 対応とキー体系統一
- [PRD075](../../memo/done/075_episode_steps_and_eval_session_log_10prd.md) — PRD 075: エピソード長の汎用メトリクス化と評価セッションの実行ログ
- [PRD094](../../memo/094_evalpanel_step_history_10prd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD
- [PRD097](../../memo/097_env_auto_reset_after_notification_10prd.md) — 遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD

## 台帳の指摘

台帳のツールが作り直す。手で直さない。

<!-- registry:begin -->
なし

[台帳の一覧](../../../reports/registry/findings.md)
<!-- registry:end -->

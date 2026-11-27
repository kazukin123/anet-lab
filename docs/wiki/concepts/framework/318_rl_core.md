---
title: 318 強化学習 共通基盤
page_kind: フレームワーク
aliases:
  - 318_rl_core
  - Notifier
  - eval_schedule
  - 強制行動
  - ForcedActions
  - MakeActionOptions
description: ReplayBufferのExperience、N-step、PER、stack履歴、転送とprefetchの共通契約を引く。RunnerとObserverの実行・通知の所在もまとめる。
updated: 2026-11-27
---
# 318 強化学習 共通基盤

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [フレームワーク](../../index.md#フレームワーク)

## 概要・責務

- ReplayBufferのExperience、N-step、PER、stack履歴、転送とprefetchの共通契約を引く。RunnerとObserverの実行・通知の所在もまとめる。（出典: [ReplayBuffer](../../../design/150_replay_buffer.jp.md)、§1・§2）。
- ReplayBuffer は `GetSampleObsSpecs` で Sample の観測 spec を答える。複数枚の対象キーには stack 軸があり、1 枚の Sample は Frame の形を返す（出典: [ReplayBuffer](../../../design/150_replay_buffer.jp.md)、§2.2・§2.3）。
- TrainerとRunnerが学習・評価を実行し、Notifierがeventを配送する。値の収集と保存はObserverとMetricsLoggerへ分ける（出典: [可観測性](../../../design/140_observability.jp.md)、§2）。
- Actor初期priorityのhint行は共通層では列の意味を解釈せずに運び、schema検証と推定は注入されたInitialPriorityEstimatorが担う。DQNの5列schemaの正本はDQN系Agent設計である（出典: [ReplayBuffer](../../../design/150_replay_buffer.jp.md)、§2.4 PER）。
- ReplayBufferは可変長windowの抽出・転送・prefetchを担い、時間で変わるγでは保存したstep報酬のsnapshotから収益を確定する。窓の規則と利用するγは呼び出し側から渡す（出典: [ReplayBuffer](../../../design/150_replay_buffer.jp.md)、§2・§6.1.1）。
- Observerはコードで付けるなら基盤の外で書ける拡張点で、`anet/rl.hpp`・`anet/observers.hpp`の型で実装し、`Notifier::Attach`で付ける。ReplayBufferは共通ReplayBufferの実装内部で組み、差し替えはできない（出典: [ANETフレームワーク全体概要](../../../design/010_framework_overview.jp.md)、§4.3）。
- Notifierは、eventを`Notify()`を呼んだスレッドで登録順に配信し、登録・解除・通知・終了・一覧の操作を並行して呼べる。呼び出し開始時の一覧を保持してObserverをロックの外で呼び、`Detach()`と`Clear()`は進行中の通知を待たない。Observerが触る相手の寿命は登録する側が管理する（出典: [可観測性](../../../design/140_observability.jp.md)、§2・§5.1）。
- ReplayBufferは、Agentが複数の報酬成分を使うときだけ報酬を成分ごとに受け取り、成分ごとの割引率の列で収益を作る。成分の数は構築時に固定し、成分の意味は利用側のAgentが持つ。1成分のshapeと計算順は変えない（出典: [ReplayBuffer](../../../design/150_replay_buffer.jp.md)、§2.1・§2.2）。
- Runnerのstepは、呼ぶ側が渡す`MakeActionOptions`をそのstepのActorの行動選択へそのまま渡す。nn_traceは要求したstepだけ作り、Runner・Actor・Observer・Notifierに要求の登録を持たない（出典: [実行基盤と設定](../../../design/100_runtime_and_configuration.jp.md)、§2.2；[可観測性](../../../design/140_observability.jp.md)、§5.4）。
- 強制行動は、その回の`MakeActionOptions::forced_actions`（`ForcedActions::ByLane`・`ForcedActions::ByMask`）で渡す。EvalRunnerはResetの前に指定を検査し、遷移の`forced`をlaneごとに記録し、Actorが返した行動をEnvが実行する（出典: [Agentと学習](../../../design/110_agents_and_learning.jp.md)、§2.2）。
- eval scheduleの発火の列は、`EpisodeEvalObserver`がoffsetに届くまで問い合わせを止め、先頭の`warmup_count`回の発火を走らせない形で調整し、共通の`IntervalGate`は変えない（出典: [可観測性](../../../design/140_observability.jp.md)、§6.x step座標系）。
- ReplayBuffer は Push のときに、呼び出し側が渡す今の γ で成分ごとの n-step 収益を作って保存し、初期優先度の完成待ちにも同じ Push の収益を使う。学習に返す収益は、γ が定数の成分では保存値のままで、schedule の成分だけを抽選時に作り直す（出典: [ReplayBuffer](../../../design/150_replay_buffer.jp.md)、§2.2）。

## 成立の経緯

- 2026-05-21: EpisodeEvalObserver の Eval メトリクス強化 — 構造分析と修正方針（出典: [RFD0003](../../../memo/done/0003_eval_10rfd.md)、題・目的）。
- 2026-06-15: ReplayBuffer サンプリング + H2D を 1ステップ stale で先読みし GPU 学習と overlap する（AsyncLoader 方式）（出典: [ADR 0005](../../../adr/0005-sample-prefetch-stale-per.md)、本文）。
- 2026-07-01: VideoLogger 堅牢化 — ffmpeg 異常の fail-fast 検出 と コーデック `auto` 選択 RFD（出典: [RFD0025](../../../memo/done/0025_videolog_codec_auto_ffmpeg_guard_10rfd.md)、題・目的）。
- 2026-07-12: Replay優先度更新にはgeneration付きreplay item keyを使用する（出典: [ADR 0011](../../../adr/0011-generation-aware-replay-item-key.md)、本文）。
- 2026-07-13: Replay初期優先度ヒントを単一packed carrierとし、completionを専用coordinatorへ分離する（出典: [ADR 0012](../../../adr/0012-replay-initial-priority-hint-completion.md)、本文）。
- 2026-07-13: 追加 forward なしの近似 Actor PER 初期優先度 RFD（出典: [RFD0035](../../../memo/done/0035_approx_actor_priority_per_10rfd.md)、題・目的）。
- 2026-07-20: ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter（出典: [RFD0034](../../../memo/done/0034_imagecls_batch_input_10rfd.md)、題・目的）。
- 2026-08-13: RFD0050: ReplayBuffer ring折り返し時のframe stack sampleability（出典: [RFD0050](../../../memo/done/0050_replay_ring_stack_margin_10rfd.md)、題・目的）。
- 2026-08-13: ReplayBufferのsample可能範囲は、ring上書きで失われたstack履歴を必要とするtransitionを除外する（出典: [ADR 0024](../../../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md)、本文）。
- 2026-08-14: RFD0052: eval 定義とスケジュールの namespace 分離（出典: [RFD0052](../../../memo/done/0052_eval_schedule_separation_10rfd.md)、題・目的）。
- 2026-08-14: configured eval の定義と定期駆動を別 namespace に分離し、生成は schedule エントリが駆動する（出典: [ADR 0027](../../../adr/0027-eval-definition-schedule-separation.md)、本文）。
- 2026-08-17: RFD0053: interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化（出典: [RFD0053](../../../memo/done/0053_interval_gate_perf_ema_10rfd.md)、題・目的）。
- 2026-08-17: 飛び飛びのカウンタに対する interval を bucket-crossing で定義する（出典: [ADR 0028](../../../adr/0028-interval-fires-on-bucket-crossing.md)、本文）。
- 2026-08-21: Conv2d 可視化「実Forwardトレース化」実装指示書（出典: [RFD0006](../../../memo/done/0006_conv2d_visual_trace_10rfd.md)、題・目的）。
- 2026-08-21: Device 転送共通部品 `anet/transfer.hpp` と PrefetchingReplayBuffer 適用 RFD（出典: [RFD0020](../../../memo/done/0020_device_transfer_common_part_10rfd.md)、題・目的）。
- 2026-08-21: PrefetchingReplayBuffer リファクタ（Learner 埋め込み prefetch を decorator へ分離 + 再現性修正）仕様書（出典: [RFD0014](../../../memo/done/0014_async_loader_prefetch_10rfd.md)、題・目的）。
- 2026-08-21: ReplayBuffer PER priority 転送境界整理 RFD（出典: [RFD0021](../../../memo/done/0021_replay_per_priority_transfer_10rfd.md)、題・目的）。
- 2026-08-21: Sample+To プリフェッチ（ReplayBuffer サンプリング + H2D を GPU 学習と overlap）仕様書（出典: [RFD0013](../../../memo/done/0013_sample_prefetch_10rfd.md)、題・目的）。
- 2026-08-21: V2: 可視化層（metrics_image / HeatMap / 状態スイープ）の TensorDict 復旧 実装指示書（出典: [RFD0010](../../../memo/done/0010_image_metrics_v2_10rfd.md)、題・目的）。
- 2026-08-21: anet-lab ハンドオフ: ReplayBuffer の t=0 FrameStack で未初期化 grid を読む問題（出典: [RFD0007](../../../memo/done/0007_rb-grid-corruption_10rfd.md)、題・目的）。
- 2026-08-21: mixed targetを運ぶ共通学習結果の接続を更新した（出典: [RFD0030](../../../memo/done/0030_imagecls_mixup_cutmix_10rfd.md)、前提事実（実コード確認済み））。
- 2026-08-29: 可塑性 weight norm（feature/readout 分割）RFD（出典: [RFD0063](../../../memo/done/0063_plasticity_weight_norm_10rfd.md)、題・目的）。
- 2026-08-29: 可塑性メトリクス（srank / dormant unit 率）RFD（出典: [RFD0062](../../../memo/done/0062_plasticity_metrics_10rfd.md)、題・目的）。
- 2026-09-02: Policy churn メトリクス（35_agent_churn）RFD（出典: [RFD0066](../../../memo/done/0066_policy_churn_metrics_10rfd.md)、題・目的）。
- 2026-09-03: 評価セッションの N 本集約は BatchEnv デコレータと開始時動的グラントで行う（出典: [ADR 0034](../../../adr/0034-eval-session-aggregation-in-batchenv-decorator.md)、本文）。
- 2026-09-04: 評価セッション契約: eval_batch_size（lane 数）と eval_episodes（本数）で N 本平均を 1 点として記録する（出典: [RFD0060](../../../memo/done/0060_eval_batch_episodes_10rfd.md)、題・目的）。
- 2026-09-05: メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す（出典: [RFD0069](../../../memo/done/0069_metrics_trace_channel_10rfd.md)、題・目的）。
- 2026-09-05: 個体メトリクスは trace チャネルで出し、評価セッションは採用 episode ごとの EpisodeEndEvent と SessionEndEvent を分けて発火する（出典: [ADR 0037](../../../adr/0037-metrics-trace-channel-and-session-end-event.md)、本文）。
- 2026-09-13: エピソード長は SHARED でも Step() 回数とし、episode return の総和とは非対称にする（出典: [ADR 0043](../../../adr/0043-episode-steps-shared-scope-is-step-count.md)、本文）。
- 2026-09-17: RFD0075: エピソード長の汎用メトリクス化と評価セッションの実行ログ（出典: [RFD0075](../../../memo/done/0075_episode_steps_and_eval_session_log_10rfd.md)、題・目的）。
- 2026-09-19: RFD0061: Actor 設定カタログと eval スロットごとの方策 / network 指定（出典: [RFD0061](../../../memo/done/0061_eval_slot_policy_override_10rfd.md)、題・目的）。
- 2026-09-20: RFD0077: sampleable index 列の確保を持続バッファへ移す（出典: [RFD0077](../../../memo/done/0077_valid_index_scratch_reuse_10rfd.md)、題・目的）。
- 2026-09-20: ReplayBuffer の観測履歴・bootstrap stack 整合性修復 RFD（出典: [RFD0078](../../../memo/done/0078_replay_frame_history_integrity_10rfd.md)、題・目的）。
- 2026-09-20: ReplayBufferのframe履歴開始はPush時のepisode_startで確定し、n-step確定metadataから導出しない（ENDをslot単位に持たない部分だけ後にADR0063で改め、履歴開始の判断は維持）（出典: [ADR 0044](../../../adr/0044-replay-frame-history-start-from-episode-start-at-push.md)、本文と冒頭の後発決定の注記）。
- 2026-09-22: RFD0076: 終了時の評価セッション排水（出典: [RFD0076](../../../memo/done/0076_eval_session_drain_on_exit_10rfd.md)、題・目的）。
- 2026-09-22: 終了時の背景評価セッションは完走を既定とし、設定または上限時間による協調キャンセルで抜けられるようにして、部分集計は設けない（出典: [ADR 0045](../../../adr/0045-eval-session-drain-on-exit-wait-or-discard.md)、本文）。
- 2026-09-24: 評価Runnerが新しいdevice指定を解決する接続へ移行した（出典: [RFD0070](../../../memo/done/0070_device_spec_auto_10rfd.md)、確定したD1・D9）。
- 2026-10-10: NN実行設定の移行に合わせObserverとTrainerの接続・回帰を確認した（出典: [RFD0084](../../../memo/done/0084_noisynet_10rfd.md)、§12 受け入れ条件・§13 実装順と複雑度監査）。
- 2026-10-19: Evaluation Trace の列はメトリクスの定義と分けて番号付きで宣言し、遷移の時点に合う target だけを読む（出典: [ADR 0057](../../../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)、本文）。
- 2026-10-19: step trace の近似優先度は Agent が出す step 部品が計算し、Actor の設定キーで eval_panel にだけヒントを作らせることはしない（出典: [ADR 0058](../../../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)、本文）。
- 2026-10-20: BatchEnv の Step は終端の lane をリセットせず、Runner が通知のあとに終わったエピソードをリセットする（後にADR0060で取消）（出典: [ADR 0059](../../../adr/0059-batchenv-step-does-not-reset-runner-resets-ended-episodes-after-notification.md)、本文）。
- 2026-10-20: 遷移の結果は Step の結果で運び、速報値か確定値かは読むイベントで分ける（出典: [ADR 0060](../../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)、本文）。
- 2026-10-20: 遷移結果をStepResultと通知eventで運び終端laneの値を正した（出典: [RFD0097](../../../memo/done/0097_env_auto_reset_after_notification_10rfd.md)、題・§4.1）。
- 2026-10-21: step traceの値取得・通知とAgent診断Providerの共通接続を加えた（出典: [RFD0094](../../../memo/done/0094_evalpanel_step_history_10rfd.md)、§4.2 D5〜D9）。
- 2026-10-24: 時間で変わる割引率は、schedule のときだけ抽選時に保存済みの step 報酬から収益を作り直す（出典: [ADR 0065](../../../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)、本文）。
- 2026-10-24: ReplayBuffer は γ を自分で持たず、Push と Sample の入力として呼び出し側が渡す今の γ を使い、schedule の構成では抽選済みのstep報酬列と長さを保持し、prefetchを消費するSampleが返す時点のγで収益を作り直す。ring上書き後のstorageを読み直さない（出典: [RFD0101](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md)、§4.4 D20）。
- 2026-10-26: Replayに1 stepの記録と窓規則による可変長抽出を足した（出典: [RFD0098](../../../memo/0098_temporl_skip_policy_10rfd.md)、§5.1・§5.2）。
- 2026-10-28: 評価セッションのskip指標を学習側のstep座標で読む境界を整えた（出典: [RFD0103](../../../memo/done/0103_temporl_reference_alignment_10rfd.md)、§4 D8・§5.4）。
- 2026-10-29: EvalRunnerが最後のstepで行動を指定されたか（強制行動）を`$runner forced`として答える仕様を定めた（出典: [RFD0105](../../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、D3）。
- 2026-10-29: 判断の旗に依存しないaction_changeとepisode_endの窓規則を足した（出典: [RFD0104](../../../memo/done/0104_temporl_skip_auxiliary_only_10rfd.md)、§5.4）。
- 2026-11-03: action_infoからscalarを読む際の実行時型判別を整理する課題を記録した（出典: [RFD0178](../../../memo/done/0178_metrics_action_info_dynamic_cast_10rfd.md)、背景と問題・取り組むこと）。
- 2026-11-09: InitialPriorityCompleterの`@note`に`GetStats`を足し、ObserverFactoryの古いscalar記法のコメントを140 §6と`ParseMetricTokens`への参照の1行に置き換えた（出典: [RFD0119](../../../memo/0119_small_doc_comment_config_fixes_10rfd.md)、§4 D3・D9）。
- 2026-11-10: eval schedule は既定で学習直後に評価せず、発火の時点を offset でずらす（出典: [ADR 0070](../../../adr/0070-eval-schedule-skips-warmup-fires-by-default-and-shifts-with-offset.md)、本文）。
- 2026-11-10: RFD0158 で eval schedule の `offset` と `warmup_count` を追加した。列の調整は EpisodeEvalObserver が持ち、RunManager が Env 生成前の値の検証と配線を行う（出典: [RFD0158](../../../memo/0158_eval_schedule_first_fire_10rfd.md)、§4・§5.5；[Observability](../../../design/140_observability.jp.md)、eval schedule の説明）。
- 2026-11-11: WARN は利用者が直すべき状態に限り、読まれない設定キーは NoCare にせず検出する（出典: [ADR 0071](../../../adr/0071-warn-means-user-must-fix-and-unread-config-leaves-are-detected.md)、本文）。
- 2026-11-11: `DefaultReplayBuffer`のコンストラクタへpriority storeとcompleterの片方だけを渡すと止まることと、MuZeroのReplayの終端での価値ターゲットの打ち切りとunroll窓のmaskを、テストで固定した（出典: [RFD0122](../../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§1.1・§4 D2・D4）。
- 2026-11-12: Replay は報酬を成分ごとに記録し、成分ごとの割引率で収益を作る。成分の意味は持たない（出典: [ADR 0073](../../../adr/0073-replay-keeps-reward-components-and-builds-returns-per-component-discount.md)、本文）。
- 2026-11-12: metricsの宣言の未知のevent・step軸・target・属性を構築時に止め、宣言したscalarの値が取れない通知をtagごとにRunで1回にし、休止した評価tagを参照する通知をINFOにした。PERの抽選が埋まらないときの代替の通知もRunで1回にし、`run.train.runner_type`の未知の値を止めるようにした（出典: [RFD0117](../../../memo/0117_warn_policy_and_unread_config_10rfd.md)、§4 D7〜D9・§5.5）。
- 2026-11-14: ExperienceSamplesのフィールドの並びを1つにし、`To`をCPUとCUDAで1つの経路にして未定義のtensorをそのまま返し、効き目の無いstream guardを消した（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、§5 D7〜D9・§6.1）。
- 2026-11-15: 価値の分解のために、Replayが報酬を成分ごとに受け取り、n-stepの収益・抽選時のstepの報酬・初期優先度の見積もりの入力を成分ごとに作るようにした。1成分なら今と同じ値になる（出典: [RFD0121](../../../memo/0121_extrinsic_intrinsic_value_decomposition_10rfd.md)、§5 D8・§8.2）。
- 2026-11-15: `Agent::CreateStepDiagnosticsProvider`の引数を、laneの数・Evaluation Traceの列の鍵の一覧・RunnerのActorを持つ入力（`AgentStepDiagnosticsRequest`）に替えた。鍵の一覧は計算するかどうかだけを決め、鍵の認識は変えない（出典: [RFD0127](../../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md)、§4 D3・D9・§5.2・§6.1）。
- 2026-11-15: Actorに、同期に従うもの（`ActorSyncFollower`）をつなぐ口（`AttachSyncFollower`）を足した。Actorは、sourceから写し直すとき、sourceを守るロックの中でそれを呼び、相手を所有しない。基底の既定は、つながずに偽を返す（出典: [RFD0127](../../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md)、§3・§4 D4・§6.1）。
- 2026-11-15: step診断の部品（`AgentStepDiagnosticsProvider`）に、休ませる口（`Suspend`）と使い始める口（`Resume`。Actorの同期が要るかを返す）を足した。基底の既定は、何もせず、偽を返す（出典: [RFD0127](../../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md)、§4 D20・§5.4・§6.1）。
- 2026-11-17: 行動選択に添える入力をMakeActionOptionsにし、Runnerのstep（DoStep・DoUpdateFrame）とActorのMakeActionの引数で都度渡すようにした。nn_traceは要求したstepだけ作り、coreに要求の登録を持たない（出典: [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md)、§4 D2・D3・D4・D6；[ADR 0072](../../../adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md)）。
- 2026-11-19: Replay・Observer・Runnerの共通基盤のファイルを基盤のライブラリ（`src/core/base`、ターゲット`anet-core-base`）へ、MuZeroのReplayを系統のライブラリ（`src/algo/muzero`）へ移し、テストがほかのライブラリを使うときは公開ヘッダとテスト用の口を通す形にした（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.1 D3・D4・D12）。
- 2026-11-20: Notifierの`Attach`・`Detach`・`Notify`などを別々のスレッドから同時に呼んでもObserverの列が壊れないようにし、`Detach`は進行中の通知を待たず、Observerをロックの外で呼ぶようにした（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5 D1〜D5；[ADR 0082](../../../adr/0082-notifier-detach-does-not-wait-for-in-flight-notifications.md)）。
- 2026-11-20: 呼んでよいスレッドと時点を、Notifierと、キャッシュを持つ`BatchStepResult`・`ReplayInitialPriorityHint`・`BatchActionInfo`の宣言のコメントに書いた（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5 D7）。
- 2026-11-21: Observer・ReplayBuffer・RunManagerのエラー・WARN・INFOが示す設定キーを完全キーにし、評価の`scheduled`の行の各欄も完全キーで示すようにした（出典: [RFD0194](../../../memo/0194_config_key_full_names_in_messages_10rfd.md)、§5 D2・D25）。
- 2026-11-22: 強制行動は`MakeAction`の入力とし、Actorの出力を実行された行動にする（出典: [ADR 0083](../../../adr/0083-forced-actions-are-make-action-input-and-actor-output-is-executed-action.md)、本文）。
- 2026-11-23: 強制行動を`MakeActionOptions`の欄（`ForcedActions`）でActorへ渡す形にし、行動の引数を持つ`EvalRunner::DoStep`と、あとから差し替えるWithActionを消し、Runnerの「強制したか」の記録をlaneの印から作るようにした。Runnerに強制行動を渡す口は1つになり、負数が「強制なし」を表す決まりは無くなった（出典: [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md)、§6.2 D7〜D10・D12・D15・D16）。
- 2026-11-24: 重ねる枚数とキーの設定はAgent全体に置き、Sampleの束のspecはReplayBufferが答える（出典: [ADR 0095](../../../adr/0095-frame-stack-setting-is-agent-wide-and-replay-buffer-answers-sample-spec.md)、本文）。
- 2026-11-25: ReplayBufferが、Sampleで返すobsとnext_obsのspecを答える口（`GetSampleObsSpecs`）を、通常とプリフェッチの両方に足した。Sample用のForwarderはこのspecを受けて作る（出典: [RFD0200](../../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.2 D14・§6.9）。

## 選ばなかった形

- SampleIndicesをcallerに置いた内部plan/fetch分離 — 不採用: callerの待ちとReplayBuffer内部seamを増やす（出典: [RFD0014](../../../memo/done/0014_async_loader_prefetch_10rfd.md)、確定した設計判断）。
- Observerの周期判定をAgent内部のtriggerへ戻す形 — 不採用: roundより細かなevalは既存通知境界では実現できない（出典: [RFD0053](../../../memo/done/0053_interval_gate_perf_ema_10rfd.md)、5. スコープ外）。
- step traceをtraceチャネルへ記録する配線 — 範囲外: 鏡写しインスタンスを購読先にする配線と同じ関心である（出典: [RFD0094](../../../memo/done/0094_evalpanel_step_history_10rfd.md)、§6）。
- 学習側の厳密な優先度の列 — 不採用: EvalPanelには単一networkのforwardしかなくreplayも無いため学習側のonline/targetによる値は得られない（出典: [RFD0094](../../../memo/done/0094_evalpanel_step_history_10rfd.md)、§6）。
- Event/Observerの型の統一 — 範囲外: step traceとは別の関心。再開条件: 新しいEvent型を足す変更が出たとき（出典: [RFD0094](../../../memo/done/0094_evalpanel_step_history_10rfd.md)、§6）。
- episode開始通知とViewの開始表示 — 範囲外: 今回のゴールに効かず手動操作の時点ずれと同じ関心。再開条件: 0176_qvaluepanel_manual_action_state_alignmentで手動の時点ずれを決めるとき（出典: [RFD0097](../../../memo/done/0097_env_auto_reset_after_notification_10rfd.md)、§5）。
- ResetEndedEpisodesの段・外から読む値の写し・予備ALE — 不採用: 段の案は非同期学習を5〜8%遅くし、写しは埋め合わせを残し、予備ALEはAtari固有になる（出典: [RFD0097](../../../memo/done/0097_env_auto_reset_after_notification_10rfd.md)、§5・§4.1 D1/D2）。
- Actor設定キーでeval_panelだけにhintを生成させる形 — 不採用: 設定が増え、autoの語が今の条件を表すものになり、n-step窓の所有がAgent外へ出る（出典: [ADR 0058](../../../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)、本文）。
- 全targetを許し値の時点差は文書だけで説明する形 — 不採用: 終端行でリセット前後の値が混ざっても見た目で気付けない（出典: [ADR 0057](../../../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)、Considered Options）。
- 速報値と確定値を別の鍵名に分ける形 — 不採用: Step結果が2組の値を持つ作りになり、現用の設定キー名も移行する必要がある（出典: [ADR 0060](../../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)、Considered Options）。
- skip 用の replay を別に持つ（TempoRL 論文の形） — 不採用: PER・prefetch・ready・世代 key が 2 系統になり、obs の複製でメモリが 2 倍（出典: [RFD0098 TempoRL](../../../memo/0098_temporl_skip_policy_10rfd.md)、§4 D11、[ADR 0063](../../../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)）。
- Push 時に同一行動の区間を確定して slot に足す形 — 不採用: 同じ起点から短い sub-skip を取れない（出典: [RFD0098 TempoRL](../../../memo/0098_temporl_skip_policy_10rfd.md)、§4 D11、[ADR 0063](../../../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)）。
- 判断の境界を越えて同じ行動の連続を数える窓 — 不採用: 次の判断で同じ行動を選び直すかが次の状態に依存するため、確率的な env で長い窓に偏りが入る。窓は判断の旗で区切る（出典: [RFD0098 TempoRL](../../../memo/0098_temporl_skip_policy_10rfd.md)、§4 D13、[ADR 0063](../../../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)）。→ RFD0104 で、skip の出力を行動に使わない構成（補助学習のみモード。全 Actor がスキップ長を常に 1 にする）に限り採用。行動の変化で閉じる窓と、上限 J とエピソードの終わりだけで閉じる窓の 2 つ（出典: [RFD0104](../../../memo/done/0104_temporl_skip_auxiliary_only_10rfd.md)、§4 D3・D4）。
- k_max に比例する重みで論文の「接続ごとに等しい重み」に合わせる形 — 先送り: 一様な k で始める。再開条件: 短い k の偏りが skip の学習に効くと分かったら。後続の暫定 RFD では、全 k の一括学習（K）が全接続を等しく数える形でこの重みも扱う（出典: [RFD0098 TempoRL](../../../memo/0098_temporl_skip_policy_10rfd.md)、§8、[RFD0183 TempoRL の後続](../../../memo/0183_temporl_skip_followups_10rfd.md)、§4 D7）。
- eviction 統計（last_evicted 系）も sampleable の範囲を基準にする形 — 不採用（却下）: 追い出される slot は必ず history margin を過ぎてから追い出されるので、wrap 後に統計が恒常的に 0 件になり、一度も引かれずに追い出された割合の指標が死ぬ（出典: [ADR 0024](../../../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md)、Considered Options；[RFD0050](../../../memo/done/0050_replay_ring_stack_margin_10rfd.md)、0. 決定一覧 D7）。
- sampleable index を返す GetValidIndices1D を非 const にする形 — 不採用（採らない）: const な呼び出し元 4 箇所が、unique_ptr の operator-> が const を伝播しないことに依存して非 const を呼ぶ形になり、将来 propagate_const などで塞がれると通らなくなる（出典: [RFD0077](../../../memo/done/0077_valid_index_scratch_reuse_10rfd.md)、D4）。
- 初期優先度のヒントを TensorDict で任意の named tensor として運ぶ形 — 不採用（棄却）: 複数 value の CPU 化が複数の D2H になり、1 key へ pack すると単一 tensor の carrier と実質同じになる（出典: [ADR 0012](../../../adr/0012-replay-initial-priority-hint-completion.md)、Considered Options）。
- 転送の in-flight 数を blocking cap で抑える形 — 不採用（入れない）: worker の Acquire と learner の future.get の間でデッドロックを生む。in-flight は呼び側の 1-deep prefetch で back-pressure され有界になる（出典: [RFD0020](../../../memo/done/0020_device_transfer_common_part_10rfd.md)、Solution（概要）設計判断 5）。
- 評価セッションで group ごとに固定の quota を割り当てる形 — 不採用（却下）: 不偏だが、episode 長が不均一なほど最も遅い group の quota 合計に律速される（出典: [RFD0060](../../../memo/done/0060_eval_batch_episodes_10rfd.md)、11. 却下した旧解決候補）。
- partial Reset・masked Step（採用権を持たない group を Step の対象から外す） — 先送り: P2 には導入しない。再開条件: P3 の実測で必要性が出たら、別の契約として設計する（出典: [RFD0060](../../../memo/done/0060_eval_batch_episodes_10rfd.md)、4.3 評価セッションと採用規則）。
- 保存済みの収益は Push 時の γ、bootstrap は学習時の γ の 2 時点を契約にする形 — 不採用: 変化中に学習の target の中へ既知のずれを作る。Replay が step 報酬を保存しているので作り直す方が安い（出典: [RFD0101](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md)、§4.4 D20・§7、[ADR 0065](../../../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)、Considered Options）。
- γ の schedule で常に抽選時に収益を作り直す形 — 不採用: 定数でも計算が増え、足す順を揃えないと既存 Run との checksum が変わる。定数は保存値の経路に留める（出典: [ADR 0065](../../../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)、Considered Options；[RFD0101](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md)、§4.4 D20）。
- γ を今回 constant だけに限る形 — 不採用: 器としては γ だけ空く。schedule のときだけ作り直す形で両立できた（出典: [RFD0101](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md)、§7、[ADR 0065](../../../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)、Considered Options）。
- 強制したかを行動の情報（WithActionの結果。当時の名前で、今のコードに無い）で答える形 — 不採用: 基底・DQN系・MuZero系の行動の情報とテストの型を変え、「WithActionの結果＝強制」という意味を公開の型に付ける。強制するのはRunnerで、Runnerが答えれば1か所で済む（出典: [RFD0105](../../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、D3）。
- metrics_scalar.txtにscalar記法の要素ごとの語の一覧を写す形 — 不採用: 語を足すたびに`ParseMetricTokens`・140 §6・設定ファイルの3か所を直すことになり、知らない語はkeyとして黙って扱われるので、古い写しどおりに書くと値が記録されない。ObserverFactoryのコメントは参照の1行だけにする（出典: [RFD0119](../../../memo/0119_small_doc_comment_config_fixes_10rfd.md)、§4 D9・§7）。
- `interval = 0` で止めた評価 tag と、schedule の記述が無い tag で、参照の skip の通知のレベルを分ける形 — 不採用: ADR 0027 が 1 つにまとめた dormant をまた 2 つに分け、休止の出どころを運ぶ分岐が増える（出典: [RFD0117](../../../memo/0117_warn_policy_and_unread_config_10rfd.md)、D8・§8）。
- PER の抽選が埋まらないとき Run を止める形 — 不採用: 過去の Run のログ 616 本で 1 度も起きておらず、起きる条件が分からないまま止めると正当な極初期の状況で Run を落とす恐れがある。代替の是非は課題に起票した（出典: [RFD0117](../../../memo/0117_warn_policy_and_unread_config_10rfd.md)、D9・§8・§9 の 20）。
- Runnerが要求の数を持ち、nn_traceを読む側がRunnerに登録する形 — 不採用: stepの結果が引数に現れない数で決まり、入力の側に「事前の登録」という2つ目の渡し方ができる（出典: [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md)、§4.1；[ADR 0072](../../../adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md)、Considered Options）。
- Observerが「nn_traceが要る」と宣言し、RunnerがNotifierに聞く形 — 不採用: 全イベント・全Runnerが共有するObserverの仕組みにnn_traceの話が入り、Learnerの購読に近い形がActor側にも増える（出典: [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md)、§4.1；[ADR 0072](../../../adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md)、Considered Options）。
- 購読ヒントと同じ経路でnn_traceの要否を渡す形 — 不採用: 学習の前に1回だけ渡す仕組みで、Runの途中で開閉するpaneに追従できない（出典: [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md)、§2.5・§4.1）。
- MakeActionの3つ目の引数を真偽の値にする形、既定値を付けない形、引数2つの版をvirtualでない関数として残す形 — 不採用: 入力が増えるたびに基底の引数が変わる、テストの呼び出し約65か所に空の値を足すことになる、派生クラスで基底の2引数の版が名前ごと隠れる（出典: [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md)、§4 D2・D3・§4.1）。
- 公開のMakeActionをvirtualでない1本にして既定値をそこだけに書き、実装は別名のvirtualをoverrideする形 — 不採用: 既定値は空の構造体で食い違う余地がほとんど無く、Runnerのstepも同じ作りにそろえるとEvalRunnerで基底のDoStepが名前ごと隠れる（出典: [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md)、§4.1）。
- EvalRunnerの手動の行動の引数をMakeActionOptionsにまとめる形 — 範囲外: RFD0161の範囲（出典: [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md)、§3.2・§9）。→ RFD0161 で採用（出典: [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md)、§6.2 D8）。
- Learnerの購読ヒントを、開始前の宣言と呼び出しごとの要求に分けて見直す形 — 範囲外: 別の関心（出典: [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md)、§8の1）。
- Replayはstepの報酬だけを成分で持ち、収益はLearnerが抽選のあとに作る形 — 不採用（却下）: 成分が1つのときは今と同じ値を保つために定数の割引率でPush時に保存する経路を残す必要があり、成分があるときだけ別の経路にすると、収益を作る経路が2通りになる（出典: [ADR 0073](../../../adr/0073-replay-keeps-reward-components-and-builds-returns-per-component-discount.md)、Considered Options、[RFD0121](../../../memo/0121_extrinsic_intrinsic_value_decomposition_10rfd.md)、§5 D8）。
- 内的報酬という名前の2本目の報酬をReplayに足す形 — 不採用（却下）: 汎用のReplayに、特定のアルゴリズムの名前と意味が入る（出典: [ADR 0073](../../../adr/0073-replay-keeps-reward-components-and-builds-returns-per-component-discount.md)、Considered Options）。
- game overで切らない形のために、終端の観測からbootstrapする形 — 不採用: Replayはgame overの終端の観測を保存しておらず、次のエピソードの最初の観測からのbootstrapが、Replayを変えずに作れる（出典: [RFD0121](../../../memo/0121_extrinsic_intrinsic_value_decomposition_10rfd.md)、§4・§5 D7・§5.1）。
- 窓の収益を成分ごとに作る形 — 先送り: 窓の規則を使うのはskipだけで、価値の分解とskipの併用はERRORなので、今は使う構成が無い。成分が2つ以上のReplayに窓の規則を渡したらERRORにする。再開条件: 価値の分解とTempoRLの併用を作るとき（出典: [RFD0121](../../../memo/0121_extrinsic_intrinsic_value_decomposition_10rfd.md)、§5 D8・§5.1、[ADR 0073](../../../adr/0073-replay-keeps-reward-components-and-builds-returns-per-component-discount.md)、Consequences）。
- `ExperienceSamples::To`で、必須のフィールドが未定義なら止める形 — 不採用: どのフィールドが必須かは標本を使う側が決める（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、§5 D7・§5.1）。
- `ExperienceSamples::To`の経路ごとの未定義の扱いを、引数で切り替えて保つ形 — 不採用: 片方だけ直した跡と読める食い違いを引数として残す（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、§5 D7・§5.1）。
- `ExperienceSamples::To`のstream guardをCUDAのときだけ残す形 — 不採用: 今のdeviceの今のstreamを設定し直すだけで効き目が無い（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、§2.2・§5 D8・§5.1）。
- `Detach`が、そのObserverへの進行中の呼び出しが終わるまで待つ形 — 不採用: 待たれているObserverがGUIスレッドの持つロックやGUIの処理を待つと互いに待って固まり、Observerの中から自分を外すときの特例も要る（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5.1；[ADR 0082](../../../adr/0082-notifier-detach-does-not-wait-for-in-flight-notifications.md)、Considered Options）。
- 通知のたびに、ロックの中でObserverの一覧を写す形 — 不採用: Observerの数に比例する処理が通知のたびに増える（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5 D5・§5.1）。
- 守りの無いキャッシュ2つ（`GetPayloadCpu`・`GetAction(device)`）にも`GetEpisodeEndMask`と同じ守りを足す形 — 不採用: 同じ実体を同時に呼ぶ経路が無く、行動の1 stepごとに通る箇所へ使われない守りを足すことになる（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§1.2・§5 D7）。
- 強制行動のdeviceを、呼ぶ側のEvalRunnerで合わせる形 — 不採用: 汎用のEvalRunnerがDQN系の都合のためにdeviceをそろえる形になる（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5 D9・§5.1）。
- 学習スレッドからPanelかFrameを触る3つのObserver（TrainPanel、EvalPanelが学習側のstep数を受けるObserver、RunnerFrameの状態表示用のObserver）を、終了の順序に頼らない作りに変える形 — 不採用: 今は終了の順序で守られていて、今ある危険が無く、得が小さい（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§2）。
- eval scheduleの既定を学習直後に評価する今の挙動のままにし、要る所で止める回数を書く形 — 不採用: 初期の重みの点が、書き忘れたすべてのRunの系列の先頭に残る（出典: [ADR 0070](../../../adr/0070-eval-schedule-skips-warmup-fires-by-default-and-shifts-with-offset.md)、Considered Options）。
- offsetは格子だけをずらし、学習直後の1回はstep 0のままにする形 — 不採用: 2本の評価が学習直後に必ず重なる（出典: [ADR 0070](../../../adr/0070-eval-schedule-skips-warmup-fires-by-default-and-shifts-with-offset.md)、Considered Options）。
- offsetをinterval未満に限る形 — 不採用: 書いたとおりに動くので上限は要らず、開始の遅れもstepで書けなくなる（出典: [ADR 0070](../../../adr/0070-eval-schedule-skips-warmup-fires-by-default-and-shifts-with-offset.md)、Considered Options）。
- メトリクス側のintervalにもoffsetと先頭で止める回数を足す形 — 不採用: メトリクスの学習直後の点は安く、解析の邪魔になった例が無い（出典: [RFD0158](../../../memo/0158_eval_schedule_first_fire_10rfd.md)、§3.2）。
- 評価が1回も走らないままRunが終わったときにWARNかINFOで知らせる形 — 不採用: 最初の発火より短い配線・確認のRunで毎回出て、毎回出るWARNが本物を埋もれさせる（出典: [RFD0158](../../../memo/0158_eval_schedule_first_fire_10rfd.md)、§3.2）。
- 休止した評価tagを参照する通知をWARNのまま対象ごとに1回にする形 — 不採用: 普通のRunで印が付き続け、0件にするには休止のたびに参照している設定を消して回ることになる（出典: [ADR 0071](../../../adr/0071-warn-means-user-must-fix-and-unread-config-leaves-are-detected.md)、Considered Options）。
- priority storeとcompleterを1つの組にして、片方だけの状態を作れなくする形 — 不採用: `DefaultReplayBuffer`のコンストラクタとファクトリの形が変わり、題材はテストの空白だった（出典: [RFD0122](../../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§7・§4 D2）。
- MuZeroのReplayのtruncatedで補う値を直す形 — 範囲外: 学習ターゲットが変わる。課題に記録した（出典: [RFD0122](../../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§7・§4 D4・D15）。
- ObserverとReplayBufferの差し替え口の公開 — 範囲外（先送り）: 別の関心。再開条件: 基盤の外で差し替えると決めたとき（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.1 D25・§7）。
- 強制行動を、Actorの出力のあとで差し替える形（WithAction）のまま、作り直す値を足していく形 — 不採用（却下）: 行動から作る値が増えるたびに、差し替えの側も増える（出典: [ADR 0083](../../../adr/0083-forced-actions-are-make-action-input-and-actor-output-is-executed-action.md)、Considered Options）。
- 強制行動をlane数ぶんの印と行動だけで表す形 — 不採用（却下）: 指定のほとんどは1 laneで、呼ぶ側がlane数を調べて組み立てることになる（出典: [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md)、§6.2 D9・§6.6；[ADR 0083](../../../adr/0083-forced-actions-are-make-action-input-and-actor-output-is-executed-action.md)、Considered Options）。
- 強制行動を`MakeAction`の3つ目の引数にする形 — 不採用: `MakeAction`の追加の入力は、呼び出しごとに構造体で渡すと決まっている（出典: [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md)、§6.2 D7）。
- 人の操作を強制行動としてTrainのRunnerへ届ける配線 — 範囲外: 口はTrain側にも開いており、届ける配線は別の関心（RFD0160）（出典: [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md)、§2 のNG8・§6.2 D15）。
- ReplayBufferが受ける1枚のspecを、経験の受入れ用のForwarderの出力のspecにする形 — 不採用: 宣言を変える部品が、Sampleの流れにだけ2度効く（出典: [RFD0200](../../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.6、§5.2 D14）。
- Sampleの束のspecを、並びが依頼の種類や専用の口で答える形 — 不採用（却下）: 束を作るのはReplayBufferなのに、Forwarderの側が主になる（出典: [RFD0200](../../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.6；[ADR 0095](../../../adr/0095-frame-stack-setting-is-agent-wide-and-replay-buffer-answers-sample-spec.md)、Considered Options）。
- 抽選と優先度更新を内部の FIFO ticket だけで関連付ける — 不採用: 両者が必ず一対一で同順に届く暗黙の契約ができ、欠落や重複に弱い（出典: [ADR 0011](../../../adr/0011-generation-aware-replay-item-key.md)、Considered Options）。
- リング上書きによる frame 履歴の欠損を抽出時に padding する — 不採用: 実エピソードの開始と異なり、元の frame を失った区間に架空の遷移列を作る（出典: [ADR 0024](../../../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md)、Considered Options）。
- frame stack の開始境界を lane の直前 END だけから復元する — 不採用: 最初の lane 状態を仮定する必要があり、Actor が受け取る episode_start と食い違っても検出できない（出典: [ADR 0044](../../../adr/0044-replay-frame-history-start-from-episode-start-at-push.md)、Considered Options）。
- Mixup・CutMix の処理全体を GPU に限定する形 — 不採用。重い Tensor 演算は学習デバイスで行える一方、スカラーや矩形の計算まで移す実時間上の利益は小さく、可読性と再現性を優先した（出典: [RFD0030](../../../memo/done/0030_imagecls_mixup_cutmix_10rfd.md)、確定した設計判断）。
- 評価の開始だけを verbose にする形 — 不採用。既定の info ログでは評価が始まったか分からず、学習が評価を待つ間の診断ができない（出典: [RFD0075](../../../memo/done/0075_episode_steps_and_eval_session_log_10rfd.md)、比較履歴）。
- RunManager が EpisodeEvalObserver だけを直接保持し、終了時に待つ形 — 不採用。背景処理を持つほかの Observer にも使える API を優先した（出典: [RFD0076](../../../memo/done/0076_eval_session_drain_on_exit_10rfd.md)、Complexity audit）。
- 抽選範囲を余分に狭めて壊れた観測履歴を出さない形 — 不採用。エピソード由来の不足は補い、上書きで失った履歴は既存の sampleable 範囲で除外する条件を区別する必要がある（出典: [RFD0078](../../../memo/done/0078_replay_frame_history_integrity_10rfd.md)、1. 問題と目的（Problem Statement / Solution））。
- n-step の和に、次のエピソードの報酬まで足す形 — 範囲外: Replay の窓がエピソードをまたぐ契約の変更になる別の関心。内的を終端で切らない形の bootstrap は、窓を変えずに作れる（出典: [0121_extrinsic_intrinsic_value_decomposition_10rfd.md](../../../memo/0121_extrinsic_intrinsic_value_decomposition_10rfd.md)、§1.4）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../../reports/stats/trend.md)の「テスト充足」を見る。

- PushとExtractのstorage shared/exclusive lockによる排他を維持する（出典: [RFD0014](../../../memo/done/0014_async_loader_prefetch_10rfd.md)、1.2 ReplayBuffer）。
- scratchの再利用でもindexの昇順を変えず、wrap・stack・n-step・PERの既存回帰を通す（出典: [RFD0077](../../../memo/done/0077_valid_index_scratch_reuse_10rfd.md)、D5: 昇順契約を維持する）。
- 公開ReplayBuffer API、replay item keyの符号化、sampling分布、PER completionとeviction統計のready基準を変更しない（出典: [RFD0078](../../../memo/done/0078_replay_frame_history_integrity_10rfd.md)、6.2 モジュールの責務）。
- 既存のHead/SN FP32保護と用途別AMP精度を維持する。Noisy無効の既存チェーンは決定論backendでmetrics checksumを比較し、sample時の乱数列の割当変更を無効側へ持ち込まない。比較の相手は実装前に保存した基準の実行体・依存DLL・出力とし、開始時の状態を再現できないHEADからの後追い比較は基準にしない（出典: [RFD0084](../../../memo/done/0084_noisynet_10rfd.md)、§8.2・§12・§13；[RFD0060](../../../memo/done/0060_eval_batch_episodes_10rfd.md)、6. 受入条件 8）。
- 読み口と配線の整理でmetrics出力・定義レコード・WARN・fail-fastの文言と時機を変えない。同seed・決定的モードの短Runでchecksum・defs・WARNを比較する（出典: [RFD0094](../../../memo/done/0094_evalpanel_step_history_10rfd.md)、§4.1 D3・§8 1）。
- ヒント計算の共有化前後でactor_approxの同seed・決定的Runの学習metrics checksumを一致させ、既定のEvalPanel networkとRunnerカウンタの通知前後の時機を変えない（出典: [RFD0094](../../../memo/done/0094_evalpanel_step_history_10rfd.md)、§4.2 D6/D7・§8 3）。
- Agentとreplayへ入る値、loss・Q、episode return・長さを保持する。Atari2構成と他Env・逐次/非同期の7構成で全点を比較し、実時間は交互3組、ばらつきが大きければ5組で性能低下を確認する（出典: [RFD0097](../../../memo/done/0097_env_auto_reset_after_notification_10rfd.md)、ゴールG3・§7 1/4）。
- slot に 1 step の記録を足しても、sampling 分布・item key・PER の completion と eviction 統計の ready 基準・Push と Extract の lock・index の昇順を変えない。スキップが無効なら ready の遅れは max(1, n_step) のまま（出典: [RFD0098 TempoRL](../../../memo/0098_temporl_skip_policy_10rfd.md)、§5.1・§6 の 11）。
- EvalRunnerに強制の値を足す前後で、RFD0098の検証資材の段0の短縮条件のmetrics checksumを4環境とも一致させる（出典: [RFD0105](../../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、§7の受入6）。
- windowを使わない経路の乱数消費とsample列を保ち、窓を増やすときは通常転送・prefetch・probeも確かめる。定数γでは保存済み収益を使い、prefetch後のγの確定で上書き済みringを読み直さない（出典: [ReplayBuffer](../../../design/150_replay_buffer.jp.md)、§2・§6.1.1・§8）。
- RunnerFrameの状態表示用のtrain observerは、学習を停止し背景評価を排水してからDetachする。Detach前に始まった通知は解除後もそのObserverを呼びうるので、通知を終えてからFrameを破棄する（出典: [可観測性](../../../design/140_observability.jp.md)、§5.1・§5.2）。
- eval scheduleの発火の列を変えるときも共通の`IntervalGate`は変えず、その既存のテスト（`[interval_gate]`）を変えずに通す。offsetと先頭で止める回数は`EpisodeEvalObserver`が受け持ち、メトリクスのintervalは初回に発火するまま保つ（出典: [RFD0158](../../../memo/0158_eval_schedule_first_fire_10rfd.md)、§3.1・§7 A1；[ADR 0070](../../../adr/0070-eval-schedule-skips-warmup-fires-by-default-and-shifts-with-offset.md)、Consequences）。
- 小さなリファクタリングとテストの追加の前後で、RFD0098の検証資材の短いRun（4環境、seed 1、決定的モード）を流し、時間と資源のtagを除く全scalarの点を一致させる（出典: [RFD0122](../../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§4 D8・§6 A7）。
- 新しい設定が既定のときは、着手前に保存した実行体・設定を使い、スキップ無効と既存の有効構成のmetrics checksumを一致させる。古い設定と新しい実行体を組み合わせる比較を保つ（出典: [RFD0103](../../../memo/done/0103_temporl_reference_alignment_10rfd.md)、§4 D14・§5.7・§7の受入1）。
- 既定のfull・decisionの組み合わせではActorの行動・乱数消費とReplayの抽選列を変えず、RFD0103の検証資材でmetrics checksumを一致させる（出典: [RFD0104](../../../memo/done/0104_temporl_skip_auxiliary_only_10rfd.md)、§5.7・§7の受入1）。
- MakeActionOptionsを渡さないstepと行動選択は、nn_traceを作らないことのほかは今と同じ結果にし、決定的な設定の短いRunのmetrics checksumを変更の前後で一致させる（出典: [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md)、§5.2・§7のA2・A11）。
- 報酬の成分を足しても、終端、打ち切り、次観測、実際に進んだstep数、抽選の分布、item key、PERのcompletionとeviction統計のreadyの基準、PushとExtractのlock、indexの昇順を変えない。成分が1つのとき、記録・収益（窓の収益を含む）・標本・乱数の消費を今と同じに保ち、既存の断定を期待値を変えずに通す（出典: [RFD0121](../../../memo/0121_extrinsic_intrinsic_value_decomposition_10rfd.md)、§8.2・§9.2の2）。
- `ExperienceSamples`のフィールドの並びを変えるときは、CPUの経路（GridMaze・LunarLander）とCUDAの経路（Atari qbert・DropMerge）の両方の構成で、実装前に保存した実行体と全scalarの点を比べ、学習の速さを交互3巡以上で確かめる（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、§5 D16・D18、§6.4）。
- Notifierの排他を足しても、Observerを呼ぶ順（登録順）と、Observerが動くスレッド（`Notify`を呼んだスレッド）、例外の伝わり方を変えない。RunnerFrameの終了時の順序（排水のあとに`Detach`）も変えない（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5 D4・D6・§6.2）。
- 既定の設定のRunの結果を変えないことは、RFD0103の検証資材と同じ6構成・短縮条件で、実装の前後の実行体の、性能・資源以外の全scalarの一致で確かめる（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§7のA13）。
- ファイルの置き場やライブラリを移すときは、着手前に保存した実行体と、RFD0128 §6.4の前後比較の組（seed 1本、決定的なbackend）で、全scalarの点、設定の記録、ログのWARNとERRORの行の一致を確かめ、テストのケースの数を前後でそろえる（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.1 D24・§6 C3・S3-6）。
- 強制なしの`MakeAction`は、行動・aux・ヒント・乱数の消費のどれも変えない。強制していないlaneの行動・Q・ヒントも、強制の有無で変えない（出典: [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md)、§7.2・§6.2 D11）。
- ReplayBufferの重ね直しの方式と、1枚のときにstack軸を付けずに返す動きを変えない。Rainbowが同じReplayBufferを1枚で使う（出典: [RFD0200](../../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§2 のNG4；[RFD0161](../../../memo/0161_observation_pipeline_10rfd.md)、§2 のNG5）。
- 先読み有効どうしを同じ seed で動かしたときは抽選 index 列が一致する。先読み無効との比較では参照した優先度が1ステップ異なる（出典: [ADR 0005](../../../adr/0005-sample-prefetch-stale-per.md)、Consequences）。
- 上書き済み世代の優先度を受け取っても、木の値、初期優先度の由来、最大値を変えない。同じ呼び出しの有効なキーは適用する（出典: [ADR 0011](../../../adr/0011-generation-aware-replay-item-key.md)、Decision）。
- 先読み有効・無効では PER の優先度を使う時点が違う。同じフラグと同じ seed の反復で再現性を調べ、有無をまたぐ比較では loss・q_max・grad_norm の曲線で学習の非退行を見る（出典: [RFD0013](../../../memo/done/0013_sample_prefetch_10rfd.md)、§6）。
- 優先度の計算と転送を前倒ししても、ReplayBuffer の優先度を書き換える UpdatePriorities の呼び出しは Optimize の後に保つ（出典: [RFD0021](../../../memo/done/0021_replay_per_priority_transfer_10rfd.md)、Implementation Decisions）。
- Mixup 無効時は既存の loss と accuracy に bit 一致し、バッチが2件未満の場合も通常の cross entropy と一致することを確認する（出典: [RFD0030](../../../memo/done/0030_imagecls_mixup_cutmix_10rfd.md)、受け入れ基準）。
- action_info のメトリクスの読み取りを変えるときは、scalar を返す場合と返せない場合、未知キー、値の未成立、対象外イベントを公開の通知経路で比べ、step と NaN・nullopt・不正イベントの停止の意味を保つ（出典: [RFD0178](../../../memo/done/0178_metrics_action_info_dynamic_cast_10rfd.md)、「取り組むこと」「完了の目安と未決」）。

## 外の事例

- TorchRL v0.15.0 の ReplayBuffer は sample で Transform の順変換、add/extend で逆変換を適用する。保存時と取得時の表現を変えられる（出典: [Survey: TorchRL の設計と現状 ― Transforms、実装の参考、実験結果の比較元](../../../../reports/torchrl_survey_2026-11-20.md)、3. Transforms「3.5 Replay に付けた Transform が働く時点」）。
- Acme の n-step の adder は、報酬・環境の割引・追加の割引を同じ木の形にそろえて葉ごとに n-step の収益と割引の積を作り（成分ごとの割引率を表せる）、エピソードの終わりでは残りの短い遷移を書き出して、窓を次のエピソードへ延ばさない（出典: [value_decomposition_head_nstep_survey_2026-11-27.md](../../../../reports/value_decomposition_head_nstep_survey_2026-11-27.md)、§3.2）。
- Acme の系列の adder は、エピソードの終わりの扱いを 4 つ（WRITE、CONTINUE、ZERO_PAD、TRUNCATE）から選び、CONTINUE は最後の系列を次のエピソードの最初の step まで続けて書く。既定は ZERO_PAD で、Acme の R2D2 は TRUNCATE を指定する。CONTINUE を使う Acme の agent は確かめていない（出典: [value_decomposition_head_nstep_survey_2026-11-27.md](../../../../reports/value_decomposition_head_nstep_survey_2026-11-27.md)、§3.3）。
- deep_rl_zoo（個人の公開の再実装）の系列の切り出しは、エピソードの境をまたぐかを引数で選べ、既定はまたぐ。IMPALA の agent はまたぎ、R2D2・NGU・Agent57 の agent はまたがない。またがないとき、系列の長さに足りずに終わったエピソードは、同じエピソードの前の系列の末尾で埋め、前の系列が無ければ書き出さない（出典: [value_decomposition_head_nstep_survey_2026-11-27.md](../../../../reports/value_decomposition_head_nstep_survey_2026-11-27.md)、§3.3）。
- Dopamine の Replay は、n-step の和を窓の中の最初の終端で止め（その step までの報酬だけを割り引いて足す）、終端の印なしにエピソードが終わる窓を抽選の対象から外す（出典: [value_decomposition_head_nstep_survey_2026-11-27.md](../../../../reports/value_decomposition_head_nstep_survey_2026-11-27.md)、§3.2）。

## 関係

- 設計: [ReplayBuffer](../../../design/150_replay_buffer.jp.md)、§1・§2。強制行動は[Agentと学習](../../../design/110_agents_and_learning.jp.md)、§2.2。Sampleの束と観測の加工は[Forwarder と Perception](../../../design/135_forwarder_and_perception.jp.md)、§3。
- 拡張点: [ANETフレームワーク全体概要](../../../design/010_framework_overview.jp.md)、§4.3。
- 可変長windowとγ・検証手順: [ReplayBuffer](../../../design/150_replay_buffer.jp.md)、§6.1.1・§8。通知の並行と寿命は[可観測性](../../../design/140_observability.jp.md)、§5.1・§5.2。Evaluation Traceの採点と同期、nn_traceの要求は同、§5.3・§5.4。
- 置き場とテストの参照範囲: [ソース配置とライブラリの境界](../../../design/011_source_layout.jp.md)、「ライブラリとアプリ」「テスト」。
- ユーザーガイド: [Run実行ガイド](../../../design/020_user_guide_run.jp.md)、§6.4（評価の発火列の`offset`・`warmup_count`）。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [muzero_rb.cpp](../../../../src/algo/muzero/src/muzero_rb.cpp) / [muzero_rb.hpp](../../../../src/algo/muzero/src/muzero_rb.hpp)
- [observers.hpp](../../../../src/core/base/include/anet/observers.hpp)
- [replay_buffer.hpp](../../../../src/core/base/include/anet/replay_buffer.hpp)
- [replay_experience.hpp](../../../../src/core/base/include/anet/replay_experience.hpp)
- [rl.hpp](../../../../src/core/base/include/anet/rl.hpp)
- [serialize.hpp](../../../../src/core/base/include/anet/serialize.hpp)
- [trainer.hpp](../../../../src/core/base/include/anet/trainer.hpp)
- [episode_end_test.cpp](../../../../src/core/base/src/episode_end_test.cpp)
- [observers.cpp](../../../../src/core/base/src/observers.cpp) / [observers_test.cpp](../../../../src/core/base/src/observers_test.cpp)
- [replay_buffer_test.cpp](../../../../src/core/base/src/replay_buffer_test.cpp)
- [replay_buffer_impl.cpp](../../../../src/core/base/src/replay_buffer_impl.cpp) / [replay_buffer_impl.hpp](../../../../src/core/base/src/replay_buffer_impl.hpp)
- [replay_experience.cpp](../../../../src/core/base/src/replay_experience.cpp)
- [rl.cpp](../../../../src/core/base/src/rl.cpp)
- [serialize.cpp](../../../../src/core/base/src/serialize.cpp)
- [trainer.cpp](../../../../src/core/base/src/trainer.cpp) / [trainer_test.cpp](../../../../src/core/base/src/trainer_test.cpp)

## RFD の一覧

点検が作り直す。手で直さない。

- [RFD0034](../../../memo/done/0034_imagecls_batch_input_10rfd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [RFD0035](../../../memo/done/0035_approx_actor_priority_per_10rfd.md) — 追加 forward なしの近似 Actor PER 初期優先度 PRD
- [RFD0050](../../../memo/done/0050_replay_ring_stack_margin_10rfd.md) — PRD 050: ReplayBuffer ring折り返し時のframe stack sampleability
- [RFD0052](../../../memo/done/0052_eval_schedule_separation_10rfd.md) — PRD 052: eval 定義とスケジュールの namespace 分離
- [RFD0053](../../../memo/done/0053_interval_gate_perf_ema_10rfd.md) — PRD 053: interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化
- [RFD0060](../../../memo/done/0060_eval_batch_episodes_10rfd.md) — 評価セッション契約: eval_batch_size（lane 数）と eval_episodes（本数）で N 本平均を 1 点として記録する
- [RFD0061](../../../memo/done/0061_eval_slot_policy_override_10rfd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [RFD0062](../../../memo/done/0062_plasticity_metrics_10rfd.md) — 可塑性メトリクス（srank / dormant unit 率）PRD
- [RFD0063](../../../memo/done/0063_plasticity_weight_norm_10rfd.md) — 可塑性 weight norm（feature/readout 分割）PRD
- [RFD0066](../../../memo/done/0066_policy_churn_metrics_10rfd.md) — Policy churn メトリクス（35_agent_churn）PRD
- [RFD0069](../../../memo/done/0069_metrics_trace_channel_10rfd.md) — メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す
- [RFD0070](../../../memo/done/0070_device_spec_auto_10rfd.md) — PRD 070: device 指定の auto 対応とキー体系統一
- [RFD0075](../../../memo/done/0075_episode_steps_and_eval_session_log_10rfd.md) — PRD 075: エピソード長の汎用メトリクス化と評価セッションの実行ログ
- [RFD0076](../../../memo/done/0076_eval_session_drain_on_exit_10rfd.md) — PRD 076: 終了時の評価セッション排水
- [RFD0077](../../../memo/done/0077_valid_index_scratch_reuse_10rfd.md) — PRD 077: sampleable index 列の確保を持続バッファへ移す
- [RFD0078](../../../memo/done/0078_replay_frame_history_integrity_10rfd.md) — ReplayBuffer の観測履歴・bootstrap stack 整合性修復 PRD
- [RFD0084](../../../memo/done/0084_noisynet_10rfd.md) — NoisyNet (Noisy Networks for Exploration) PRD
- [RFD0094](../../../memo/done/0094_evalpanel_step_history_10rfd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD
- [RFD0097](../../../memo/done/0097_env_auto_reset_after_notification_10rfd.md) — 遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD
- [RFD0098](../../../memo/0098_temporl_skip_policy_10rfd.md) — TempoRL 対応（スキップ: 選んだ行動を続ける長さの学習）RFD
- [RFD0101](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md) — 探索と学習のパラメータを ProfiledValue で時間変化させる PRD
- [RFD0103](../../../memo/done/0103_temporl_reference_alignment_10rfd.md) — TempoRL 第二弾（著者の実装との差の洗い出しと是正）PRD
- [RFD0104](../../../memo/done/0104_temporl_skip_auxiliary_only_10rfd.md) — TempoRL の補助学習のみモード（毎 step 判断し、スキップの出力は補助タスクとしてだけ学ぶ）PRD
- [RFD0105](../../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md) — Evaluation Trace の矢印キーと強制した step の見分け、表の並べ替えと行のダンプ PRD
- [RFD0117](../../../memo/0117_warn_policy_and_unread_config_10rfd.md) — RFD0117: WARN の意味の線引きと、読まれない設定の検出 RFD
- [RFD0119](../../../memo/0119_small_doc_comment_config_fixes_10rfd.md) — RFD0119: 文書・コメント・設定の小さな食い違いをまとめて直す
- [RFD0121](../../../memo/0121_extrinsic_intrinsic_value_decomposition_10rfd.md) — 外的と内的の価値の分解（Q を外的報酬と内的報酬の 2 つの Head で学ぶ） RFD
- [RFD0122](../../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md) — RFD0122: 小さなリファクタリングとテストの空白をまとめて片づける
- [RFD0127](../../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md) — Evaluation Trace で RND の内的報酬を見る（統計を変えない採点と step 診断の鍵）RFD
- [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md) — RFD0128: DQN 系 Agent まわりのリファクタリング 5 件をまとめて片づける（価値の分解の前に） RFD
- [RFD0131](../../../memo/0131_rfd_9xx_renumber_by_creation_date_10rfd.md) — RFD0131: 9xx の番号の RFD 57 本を、起票の日の順に 900 未満の連番へ振り直す RFD
- [RFD0158](../../../memo/0158_eval_schedule_first_fire_10rfd.md) — RFD0158: eval スケジュールの発火の列 — 学習直後の評価を既定で止め、発火の時点をずらせるようにする
- [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md) — 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化 暫定 RFD
- [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md) — RFD0170: forward の名前付き出力を capture に一本化し、nn_trace を要求した step だけ作る RFD
- [RFD0182](../../../memo/0182_source_layout_10rfd.md) — ソース配置の見直し（置き場の規則、Agent の系統ごとのライブラリ、DQN のファイル分割、テストの参照範囲）RFD
- [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md) — RFD0192: 監査で見つかった不具合 5 件をまとめて直す（Notifier の同時実行、動画の画像の大きさ、強制行動の device、重複登録、QValuePanel の平滑化）
- [RFD0194](../../../memo/0194_config_key_full_names_in_messages_10rfd.md) — RFD0194: エラー・WARN・INFO が示す設定キーを完全キーにする（device の二重の fatal、指標名、用語の不整合 6 件を含む）
- [RFD0195](../../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md) — RFD0195: 文書の呼び名を RFD に改め、番号を 0 埋めの 4 桁にし、今使っている場所をすべて移す PRD
- [RFD0200](../../../memo/0200_forwarder_public_interface_rework_10rfd.md) — forwarder.hpp の公開の口の作り直し（作る側と作られるものを分け、spec を依頼ごとに渡す） RFD

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | RFD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261103-955e50](../../../management/issues/issue-20261103-955e50.md) | 調査中 | 1.2 | — | コード | RFD | 318_rl_core, 317_metrics, 110_runner_app, 311_dev_tools, 120_metrics_viewer | UI | 機能追加 | [RFD0081](../../../memo/0081_evalpanel_episode_trace_10rfd.md) | 残件有り | EvalPanel のゲームを trace へ出す — dormant タグの trace を鏡写しインスタンスへ結び付け、trace 行の時刻で振り返る |
| [issue-20261123-ff1a3f](../../../management/issues/issue-20261123-ff1a3f.md) | 未着手 | 6.0 | — | 文書 | 監査 | 318_rl_core | 学習基盤 | 不整合 | — | 未確認 | CONTEXT.md の target return が「ReplayBuffer に保持される値」とだけ書き、γ が schedule の成分は抽選時に作り直すとした ADR 0065・0073 を反映していない |
| [issue-20261123-003c85](../../../management/issues/issue-20261123-003c85.md) | 未着手 | 4.0 | — | コード | 監査 | 318_rl_core, 324_dqn | 学習基盤 | 不具合 | — | 残件有り | Replay は n_step が 0 以下でも構築でき、0 では 1 件も抽選できる状態にならないので、学習も評価も黙って始まらない |
| [issue-20261123-7671ff](../../../management/issues/issue-20261123-7671ff.md) | 未着手 | 4.0 | — | コード | 監査 | 318_rl_core, 320_env_core | 学習基盤 | 不具合 | — | 未確認 | BatchActionInfo::GetAction(device) が index の無い "cuda" をキャッシュの鍵として覚えるので、同じ実体へ同じ device で 2 回目に呼ぶと不一致で止まる |
| [issue-20261123-ff9a9a](../../../management/issues/issue-20261123-ff9a9a.md) | 未着手 | 4.0 | — | コード | 監査 | 318_rl_core | 学習基盤 | 不具合 | — | 未確認 | Replay の prefetch で、遅らせて投入した Push が例外になると、使い終わった future が列に残り、次の同期と終了時のデストラクタで落ちうる |
| [issue-20261123-3ebab4](../../../management/issues/issue-20261123-3ebab4.md) | 未着手 | 3.5 | — | コード | 監査 | 318_rl_core, 317_metrics | 学習基盤 | リファクタリング | — | 残件有り | Env の設定のファイル名の衝突検査（RunManager）が、MetricsLogger の private な置換規則と書き出し先の作り方を写して予測している |
| [issue-20261123-446e6f](../../../management/issues/issue-20261123-446e6f.md) | 未着手 | 3.5 | — | コード | 監査 | 318_rl_core, 317_metrics | 学習基盤 | 不具合 | — | 残件有り | メトリクスの宣言の数値の属性（interval・ema_alpha・clip ほか）を std::stoi・stof で読むので、後ろの文字が黙って捨てられ、数でない値は設定キーの無い例外になる |
| [issue-20261123-4df9cc](../../../management/issues/issue-20261123-4df9cc.md) | 未着手 | 3.5 | — | コード | 監査 | 318_rl_core, 110_runner_app | 学習基盤 | リファクタリング | — | 未確認 | core の RunManager が、アプリの EvalPanel があとで作る Runner の名前 "EvalPanel" を文字列で予約し、同じ名前を 2 つの層に書いている |
| [issue-20261123-ad7829](../../../management/issues/issue-20261123-ad7829.md) | 未着手 | 3.5 | — | コード | 監査 | 318_rl_core, 317_metrics | 学習基盤 | リファクタリング | — | 残件有り | Runner の counts と scalar の鍵に、一度も進まない sim_step と、reward・reward_ema と同じ値を返す別名 train_reward・train_reward_ema が残る |
| [issue-20261125-29c3b2](../../../management/issues/issue-20261125-29c3b2.md) | 未着手 | 3.5 | — | コード | 監査 | 317_metrics, 318_rl_core | 学習基盤 | 不整合 | — | 未確認 | metrics.graph の宣言の文法が設計140 に無く、実装は未知の @ や $ の語を黙って捨て、interval を両方 0 以下にすると黙って 1000 step ごとになる |
| [issue-20261103-1beaf9](../../../management/issues/issue-20261103-1beaf9.md) | 未着手 | 3.3 | — | コード | 監査 | 318_rl_core | 学習基盤 | 不具合 | — | 残件有り | PipelineTrainRunnerのイベントで、経験と学習結果が1 stepずれた組になる |
| [issue-20261111-7cc67b](../../../management/issues/issue-20261111-7cc67b.md) | 未着手 | 3.3 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | EvalRunner の 1 step が、多重定義 4 つ・-1 の特別な値・bool の引数で、パネルからの 1 step と評価 session の 1 step を切り替えている |
| [issue-20261125-16374e](../../../management/issues/issue-20261125-16374e.md) | 未着手 | 3.3 | — | コード | 監査 | 321_agent_core, 318_rl_core | 学習基盤 | 不具合 | — | 未確認 | checkpoint の読み込みが parameter の device と形を今の Network と突き合わせず、CUDA で保存したものを agent.device=cpu で読むと Network が CUDA に載ったままになる（最初の forward で落ちる見込み） |
| [issue-20261111-4a6b6f](../../../management/issues/issue-20261111-4a6b6f.md) | 未着手 | 3.0 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | RunManager のコンストラクタが 277 行あり、評価の設定の解釈・名前の予約・構築・ログ・配線が 1 つの関数に並んでいる |
| [issue-20261111-585e60](../../../management/issues/issue-20261111-585e60.md) | 未着手 | 3.0 | — | コード | 監査 | 318_rl_core, 321_agent_core, 315_nn, 110_runner_app | 基盤 | リファクタリング | — | 残件有り | 名前やコメントが実際の動きと合わない関数・クラス・設定項目がある（BatchState の IsDone、DefaultAgentFactory、NetworkGraphVizConfig の 2 項目、Runner の 4 関数） |
| [issue-20261111-ea3483](../../../management/issues/issue-20261111-ea3483.md) | 未着手 | 3.0 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | scalar メトリクスの Observer が、イベントの中身を 6〜9 個の引数にばらして 3 段持ち回り、同じ 8 引数のコンストラクタを 4 回書いている |
| [issue-20261123-6ffe02](../../../management/issues/issue-20261123-6ffe02.md) | 未着手 | 3.0 | — | 文書 | 監査 | 318_rl_core, 317_metrics | 学習基盤 | 調査 | — | 残件有り | scalar メトリクスの Observer の 8 引数のコンストラクタと Parsed の struct を、cfe994 と ea3483 の 2 件が別々の直し方で扱っている |
| [issue-20261123-fb50b6](../../../management/issues/issue-20261123-fb50b6.md) | 未着手 | 3.0 | — | 文書 | 監査 | 318_rl_core, 317_metrics | 学習基盤 | 不整合 | — | 残件有り | CONTEXT.md が定義の中で使う step の軸（exp step・train step など）・agent step・lane を、用語として持っていない |
| [issue-20261125-ae4c45](../../../management/issues/issue-20261125-ae4c45.md) | 未着手 | 3.0 | — | 文書 | 監査 | 317_metrics, 318_rl_core | 学習基盤 | 不整合 | — | 未確認 | 設計140 の章立てと図が今の形に合わない（節番号が「6.x」のまま 2 つ、env_step の節が関連文書の下、図に SessionEnd と trace が無く画像の Observer の種類と作り手が違い、コードマップに metrics・image が無い） |
| [issue-20261125-bdbcac](../../../management/issues/issue-20261125-bdbcac.md) | 未着手 | 3.0 | — | 文書 | 監査 | 317_metrics, 318_rl_core | 学習基盤 | 不整合 | — | 未確認 | 設計140 が本文で書く仕組みの ADR 8 本（評価セッションの集約・エピソード長・trace の行の時刻・forward の capture・遷移の結果・eval schedule・WARN の表・nn_trace の渡し方）を、どの設計文書もリンクしない |
| [issue-20261125-eb6135](../../../management/issues/issue-20261125-eb6135.md) | 未着手 | 3.0 | — | 文書 | 監査 | 317_metrics, 318_rl_core | 学習基盤 | 不整合 | — | 未確認 | scalar の宣言の正本である設計140 §6 に、target を省いたときの探索順（出荷設定の 57 件が使う）・属性の既定・ema_alpha だけで EMA になること・$action_info などの event の制限が無く、§7 に動画の設定キーが無い |
| [issue-20261103-9df5c5](../../../management/issues/issue-20261103-9df5c5.md) | 未着手 | 2.7 | — | コード | RFD | 318_rl_core, 110_runner_app, 314_tensor | 学習基盤 | 不具合 | [RFD0146](../../../memo/0146_notifier_scoped_detach_10rfd.md) | 残件有り | Notifier::AttachScoped の Detach 契約修復 — wrapper ハンドルを返す |
| [issue-20261107-16ffd8](../../../management/issues/issue-20261107-16ffd8.md) | 未着手 | 2.7 | — | コード | RFD | 318_rl_core, 322_env_individual, 110_runner_app | 学習基盤 | 不整合 | — | 残件有り | ERROR のログを出したあとも Run を止めずに続ける箇所（Env・Agent の作成失敗、DropMerge の果物の範囲外）が Fail-Fast 原則と合わない |
| [issue-20261111-645ecc](../../../management/issues/issue-20261111-645ecc.md) | 未着手 | 2.7 | — | コード | 監査 | 317_metrics, 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | メトリクス宣言の語（event・step・target）と enum の対応が 5 か所以上に分かれ、解決済みの定義も同じ値を文字列と enum の 2 形で持っている |
| [issue-20261111-fe23e0](../../../management/issues/issue-20261111-fe23e0.md) | 未着手 | 2.7 | — | コード | 監査 | 318_rl_core, 320_env_core | 学習基盤 | リファクタリング | — | 残件有り | BatchEnvSpec に「group の数」「group と lane の対応」を返す関数が無く、使う側が三項演算を 19 か所に書き、終端の mask を CPU へ移す処理も関数ごとに書いている |
| [issue-20261103-b2a0b1](../../../management/issues/issue-20261103-b2a0b1.md) | 未着手 | 2.5 | — | コード | 監査 | 318_rl_core | 学習基盤 | 不具合 | — | 残件有り | ActionSpec::GetShapeが連続行動で次元数ぶんの1を並べた形を返す |
| [issue-20261123-2709db](../../../management/issues/issue-20261123-2709db.md) | 未着手 | 2.5 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 未確認 | StateSpec::ValidateObservation は bool を返すが、失敗の経路はどれも例外を投げ、false を返すことがない |
| [issue-20261103-34d35d](../../../management/issues/issue-20261103-34d35d.md) | 未着手 | 2.3 | — | コード | RFD | 318_rl_core | 学習基盤 | 不具合 | [RFD0177](../../../memo/0177_episode_end_read_semantics_10rfd.md) | 残件有り | `@episode_end` で読む値を、どの target でもそのエピソードの値にそろえる |
| [issue-20261111-2c251b](../../../management/issues/issue-20261111-2c251b.md) | 未着手 | 2.3 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | Replay の窓の規則 3 クラスが同じ骨格の写しで、PER のスカラーのキーも一覧・条件式・分岐の 3〜4 か所に並んでいる |
| [issue-20261123-79bf13](../../../management/issues/issue-20261123-79bf13.md) | 未着手 | 2.3 | — | コード | 監査 | 318_rl_core, 317_metrics | 学習基盤 | リファクタリング | — | 残件有り | 観測のキー「base か base.subkey」の解き方が rl.cpp・Replay・probe.cpp・image.cpp の 4 か所にあり、subkey が空のときの扱いと dtype が既に食い違う |
| [issue-20261103-5addea](../../../management/issues/issue-20261103-5addea.md) | 未着手 | 2.2 | — | コード | RFD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [RFD0079](../../../memo/0079_cpp_internal_linkage_convention_10rfd.md) | 残件有り | RFD0079: .cpp の内部リンケージと namespace インデント規約 |
| [issue-20261123-c3b5ef](../../../management/issues/issue-20261123-c3b5ef.md) | 未着手 | 2.2 | — | コード | 監査 | 318_rl_core, 324_dqn | 学習基盤 | リファクタリング | — | 残件有り | Actor 近似の初期優先度で「n 先の hint を待って Estimate に渡す」手順が、Replay の内部と DQN の step 診断の 2 か所に写され、割引の計算はテストにも 3 つ目がある |
| [issue-20261103-78c643](../../../management/issues/issue-20261103-78c643.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | ObserverFactoryのコンストラクタが331行で、graphの節だけが共通の字句解析を使わずに独自に読む |
| [issue-20261103-785e68](../../../management/issues/issue-20261103-785e68.md) | 未着手 | 2.0 | — | コード | RFD | 315_nn, 318_rl_core, 321_agent_core, 324_dqn, 319_gui, 320_env_core, 314_tensor, 312_util, 110_runner_app, 322_env_individual, 325_muzero | 開発 | リファクタリング | [RFD0169](../../../memo/0169_outside_core_extension_layout_10rfd.md) | 残件無し | anet-core の外で実装する拡張点とコード配置の規約 |
| [issue-20261103-9bd71e](../../../management/issues/issue-20261103-9bd71e.md) | 未着手 | 2.0 | — | コード | RFD | 321_agent_core, 318_rl_core, 324_dqn, 317_metrics | 学習基盤 | 機能追加 | [RFD0142](../../../memo/0142_agent_prev_action_obs_10rfd.md) | 残件有り | 直前行動観測の共通部品（`prev_action` フレーム段） |
| [issue-20261111-07f4ed](../../../management/issues/issue-20261111-07f4ed.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | 「N エピソードに 1 回だけ記録する」状態が Conv2dVisualizationObserver と GraphVizObserver に少し違う形で書かれ、記録を始める条件と終える条件が食い違っている |
| [issue-20261111-54c7e1](../../../management/issues/issue-20261111-54c7e1.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core, 321_agent_core, 315_nn, 319_gui, 110_runner_app | 基盤 | リファクタリング | — | 残件有り | 呼ばれない関数・型・マクロと読まれないメンバ・届かない分岐が、core と Runner に合わせて 1,100 行以上残っている |
| [issue-20261112-983afe](../../../management/issues/issue-20261112-983afe.md) | 未着手 | 2.0 | — | コード | その他 | 324_dqn, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | — | 残件有り | KFMの学習がボスへ行き着くかが偶然に頼る（1階の廊下の位置が観測に無く、Qも内的報酬も左へ進む向きを持てない） |
| [issue-20261123-22d9bf](../../../management/issues/issue-20261123-22d9bf.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core, 110_runner_app, 322_env_individual | 学習基盤 | リファクタリング | — | 未確認 | TrainEvent の action_info が experience.action と同じ実体を重ねて持ち、読む側（Observer・パネル・View）が 2 通りに分かれている |
| [issue-20261123-68723d](../../../management/issues/issue-20261123-68723d.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core, 324_dqn | 学習基盤 | リファクタリング | — | 残件有り | 共通の Replay が Actor の info をすべて slot ごとに保存し、抽選のたびに切り出して学習 device へ送るが、本番で読む箇所が無い（入るのは step の旗にも保存する TempoRL の decision だけ） |
| [issue-20261123-cbe34a](../../../management/issues/issue-20261123-cbe34a.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core, 317_metrics, 110_runner_app | 学習基盤 | 不整合 | — | 残件有り | 共通の Runner が 1 フレームごとに MetricsLogger を flush し（学習スレッドでは毎 step）、設計 140 とアプリが書く「境界でだけ flush」と食い違う |
| [issue-20261124-11428b](../../../management/issues/issue-20261124-11428b.md) | 未着手 | 1.8 | — | コード | RFD | 324_dqn, 318_rl_core | アルゴリズム | 機能追加 | [RFD0201](../../../memo/0201_top_episode_memory_sil_lower_bound_10rfd.md) | 未確認 | まれに出た上位のエピソードを Replay と別に持ち続け、外的の Q を実際の収益で下から引き上げる仕組み（上位エピソードの記憶と SIL の下限損失）を DQN 系に足す |
| [issue-20261123-8cf911](../../../management/issues/issue-20261123-8cf911.md) | 未着手 | 1.7 | — | コード | 監査 | 318_rl_core, 321_agent_core | 学習基盤 | リファクタリング | — | 残件有り | Agent の拡張点 AgentFactory::CreateAgent が Notifier を受け取るが、本番の Agent 4 つはどれも使わず、使うのはテストの代役 1 つ（Observer の登録順を前にするため）だけ |
| [issue-20261103-0ef7b8](../../../management/issues/issue-20261103-0ef7b8.md) | 未着手 | 1.6 | — | コード | RFD | 317_metrics, 311_dev_tools, 316_config, 318_rl_core, 321_agent_core, 120_metrics_viewer, 110_runner_app | UI | 機能改善 | [RFD0108](../../../memo/0108_metrics_tag_numbering_10rfd.md) | 残件有り | メトリクスの tag の番号を全体で振り直す（群の番号と群の中の番号） |
| [issue-20261113-d38960](../../../management/issues/issue-20261113-d38960.md) | 未着手 | 1.6 | — | コード | 実装 | 110_runner_app, 324_dqn, 318_rl_core | UI | 性能改善 | — | 残件有り | EvalPanel のモデル同期が UI スレッドで学習ロックを待ち、画面の更新や操作応答を止め得る |
| [issue-20261103-75b8da](../../../management/issues/issue-20261103-75b8da.md) | 未着手 | 1.5 | — | コード | RFD | 321_agent_core, 318_rl_core | アルゴリズム | 機能改善 | [RFD0168](../../../memo/0168_nstep_target_correction_10rfd.md) | 残件有り | n-step の目標値の補正（窓の途中の Munchausen の項・方策オフの補正） |
| [issue-20261103-182ca1](../../../management/issues/issue-20261103-182ca1.md) | 未着手 | 1.4 | — | コード | RFD | 318_rl_core, 324_dqn, 315_nn, 320_env_core | 学習基盤 | 性能改善 | [RFD0180](../../../memo/0180_core_perf_candidates_202607_10rfd.md) | 残件有り | 2026-07 のコアの詳細レビューで挙がった性能の改善候補 5 件 |
| [issue-20261103-50644a](../../../management/issues/issue-20261103-50644a.md) | 未着手 | 1.4 | — | コード | RFD | 317_metrics, 320_env_core, 321_agent_core, 318_rl_core, 110_runner_app | UI | 機能追加 | [RFD0184](../../../memo/0184_eval_session_step_trace_10rfd.md) | 残件有り | 評価 session の step trace を Run の記録に残す |
| [issue-20261103-52bda0](../../../management/issues/issue-20261103-52bda0.md) | 未着手 | 1.4 | — | コード | RFD | 318_rl_core, 321_agent_core, 120_metrics_viewer, 311_dev_tools | UI | 機能改善 | [RFD0096](../../../memo/0096_metrics_step_axis_interval_10rfd.md) | 残件有り | メトリクスの横軸と間隔を数える単位を分ける |
| [issue-20261103-aa8701](../../../management/issues/issue-20261103-aa8701.md) | 未着手 | 1.4 | — | コード | RFD | 318_rl_core, 321_agent_core, 324_dqn, 320_env_core | 学習基盤 | 機能追加 | [RFD0143](../../../memo/0143_serialize_10rfd.md) | 残件有り | Run Serialization（Run 全体 Save / Load） |
| [issue-20261115-c89660](../../../management/issues/issue-20261115-c89660.md) | 未着手 | 1.4 | — | コード | その他 | 110_runner_app, 321_agent_core, 318_rl_core | UI | 機能追加 | — | 残件有り | Runner の Conv2dPanel と metrics.image の conv2d に、RND の target と predictor の畳み込みの出力が出ない |
| [issue-20261123-eac719](../../../management/issues/issue-20261123-eac719.md) | 未着手 | 1.4 | — | コード | 監査 | 318_rl_core, 317_metrics | 学習基盤 | リファクタリング | — | 残件有り | 画像の Observer 4 種は observers（318）にあり、それを作って Notifier へ付けるのは image（317）の ImageProviderFactory で、observers と image が互いを読んでいる |
| [issue-20261103-b68e49](../../../management/issues/issue-20261103-b68e49.md) | 未着手 | 1.3 | — | コード | RFD | 110_runner_app, 318_rl_core, 322_env_individual | UI | 機能追加 | [RFD0159](../../../memo/0159_input_mapping_joystick_10rfd.md) | 残件有り | 操作入力の Joystick 対応と操作マッピングの設定化（EvalPanel） |
| [issue-20261103-f39d10](../../../management/issues/issue-20261103-f39d10.md) | 未着手 | 1.3 | — | コード | RFD | 318_rl_core, 324_dqn | 学習基盤 | 不具合 | [RFD0141](../../../memo/0141_background_eval_snapshot_ordering_10rfd.md) | 残件有り | Background eval の network snapshot 順序保証 |
| [issue-20261103-07629c](../../../management/issues/issue-20261103-07629c.md) | 未着手 | 1.2 | — | コード | RFD | 110_runner_app, 318_rl_core, 319_gui | UI | 不具合 | [RFD0176](../../../memo/0176_qvaluepanel_manual_action_state_alignment_10rfd.md) | 残件有り | QValuePanel の Q の表示と、手動で選んだ行動が当たる状態のずれ |
| [issue-20261103-8c3c6d](../../../management/issues/issue-20261103-8c3c6d.md) | 未着手 | 1.2 | — | コード | RFD | 318_rl_core, 324_dqn, 321_agent_core, 322_env_individual | 学習基盤 | 機能追加 | [RFD0171](../../../memo/0171_replay_eviction_sample_count_10rfd.md) | 残件有り | replay の遷移が消えるまでの抽選回数と優先度を、PER の外の区分で測る |
| [issue-20261103-dbe666](../../../management/issues/issue-20261103-dbe666.md) | 未着手 | 1.2 | — | コード | RFD | 322_env_individual, 318_rl_core, 311_dev_tools | Env | 調査 | [RFD0163](../../../memo/0163_dropmerge_neet_diagnostics_10rfd.md) | 残件有り | DropMerge の NEET 診断（trace で足りるか確かめてから指標化） |
| [issue-20261110-d21e5f](../../../management/issues/issue-20261110-d21e5f.md) | 未着手 | 1.2 | — | コード | RFD | 318_rl_core, 325_muzero | 学習基盤 | 不具合 | — | 残件有り | MuZeroのReplayで、truncatedで終わるときに補う価値が、打ち切った次の状態ではなく最後のstep自身のroot_valueになっている |
| [issue-20261111-e90762](../../../management/issues/issue-20261111-e90762.md) | 未着手 | 1.2 | — | コード | RFD | 318_rl_core | 学習基盤 | 不整合 | — | 残件有り | PERの抽選が埋まらないときに残りを一様の抽選で埋める代替が、明示した要求を黙って代替しない規約と食い違う |
| [issue-20261103-51220d](../../../management/issues/issue-20261103-51220d.md) | 未着手 | 1.1 | — | コード | RFD | 318_rl_core, 321_agent_core, 322_env_individual, 317_metrics | 学習基盤 | 機能追加 | [RFD0144](../../../memo/0144_episode_forensics_10rfd.md) | 残件有り | Episode Forensics（特殊挙動エピソード自動捕捉・事後調査） |
| [issue-20261116-cc5942](../../../management/issues/issue-20261116-cc5942.md) | 未着手 | 1.1 | — | コード | その他 | 318_rl_core, 321_agent_core, 314_tensor, 324_dqn | 学習基盤 | 性能改善 | — | 残件有り | 同じデータを何度もdevice間で転送している箇所を洗い出して減らす（Replayのobsとnext_obsの重なるフレーム、RNDの採点とActorが別々に送る観測など） |
| [issue-20261111-d1d12b](../../../management/issues/issue-20261111-d1d12b.md) | 未着手 | 1.0 | — | コード | 監査 | 321_agent_core, 315_nn, 318_rl_core, 317_metrics, 316_config, 312_util, 313_profiling, 314_tensor, 322_env_individual, 110_runner_app | 学習基盤 | リファクタリング | — | 残件有り | 今は起こさないリファクタリング候補 16 群を暫定で 1 件にまとめた（手を付けるときに群ごとに分けて起票し直す） |
| [issue-20261103-cfe994](../../../management/issues/issue-20261103-cfe994.md) | 未着手 | 0.9 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | イベント種別ごとにObserver interface・RunnerScopedラッパ・Notifierメソッド・MetricsLog派生を複製している |
| [issue-20261103-c5c791](../../../management/issues/issue-20261103-c5c791.md) | 未着手 | 0.9 | — | コード | RFD | 322_env_individual, 318_rl_core | 学習基盤 | 調査 | [RFD0179](../../../memo/0179_rfd0097_performance_remeasurement_10rfd.md) | 残件有り | RFD0097: 判定不能4構成の性能再測定とDropMerge serial追加費用の調査 |
| [issue-20261103-f40ae8](../../../management/issues/issue-20261103-f40ae8.md) | 未着手 | 0.9 | — | コード | RFD | 317_metrics, 314_tensor, 318_rl_core | UI | 性能改善 | [RFD0056](../../../memo/0056_timehistogram_fine_grid_10rfd.md) | 残件有り | TimeHistogram fine-bin保持・描画時集約 |
| [issue-20261103-afe841](../../../management/issues/issue-20261103-afe841.md) | 未着手 | 0.8 | — | コード | RFD | 321_agent_core, 324_dqn, 318_rl_core | 学習基盤 | 機能追加 | [RFD0145](../../../memo/0145_staged_batch_replay_schedule_10rfd.md) | 残件有り | DefaultDQN Staged Batch / Replay Schedule |
| [issue-20261122-4c0af5](../../../management/issues/issue-20261122-4c0af5.md) | 未着手 | 0.8 | — | コード | RFD | 318_rl_core, 314_tensor, 321_agent_core | 学習基盤 | リファクタリング | — | 残件有り | 観測をためる仕組み（フレーム積層の履歴、Replay の観測の置き場、補助学習の rollout の置き場）が別々に実装されていて、共通の置き場の部品へ載せ替えていない |
| [issue-20261103-969184](../../../management/issues/issue-20261103-969184.md) | 未着手 | 0.6 | — | コード | RFD | 321_agent_core, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | [RFD0174](../../../memo/0174_llm_actor_10rfd.md) | 残件有り | LLM Actor（LLM の事前知識で行動する Actor。単独 Agent と DQN ハイブリッド） |
| [issue-20261119-497fce](../../../management/issues/issue-20261119-497fce.md) | 未着手 | 0.6 | — | コード | その他 | 324_dqn, 318_rl_core | アルゴリズム | 機能追加 | — | 残件有り | 内的の価値を非エピソードにしても、n-stepの和がエピソードの終わりで切れ、Burdaらのように内的の収益がエピソードをまたいで続かない |
| [issue-20261103-145ee2](../../../management/issues/issue-20261103-145ee2.md) | 未着手 | 0.5 | — | コード | RFD | 321_agent_core, 318_rl_core, 316_config | 学習基盤 | 機能追加 | [RFD0152](../../../memo/0152_metric_targeted_hyperparam_control_10rfd.md) | 残件有り | メトリクス目標値によるハイパラ自律調整 |
| [issue-20261103-ca1a85](../../../management/issues/issue-20261103-ca1a85.md) | 未着手 | 1.6 | [issue-20261103-1beaf9](../../../management/issues/issue-20261103-1beaf9.md) | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | SerialとPipelineのDoStepが初期化・経験の組み立て・通知・カウンタ更新を別々に持ち、既に食い違っている |
| [issue-20261103-124da5](../../../management/issues/issue-20261103-124da5.md) | 未着手 | 1.5 | [issue-20261103-3e1cf6](../../../management/issues/issue-20261103-3e1cf6.md) | コード | RFD | 321_agent_core, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | [RFD0167](../../../memo/0167_smdp_transition_duration_10rfd.md) | 残件有り | SMDP 対応（遷移ごとの経過時間を割引に通す） |
| [issue-20261103-736e39](../../../management/issues/issue-20261103-736e39.md) | 未着手 | 1.4 | [issue-20261103-9bd71e](../../../management/issues/issue-20261103-9bd71e.md), [issue-20261103-75b8da](../../../management/issues/issue-20261103-75b8da.md), [issue-20261107-949e8a](../../../management/issues/issue-20261107-949e8a.md) | コード | RFD | 324_dqn, 321_agent_core, 318_rl_core, 322_env_individual, 110_runner_app | アルゴリズム | 機能追加 | [RFD0173](../../../memo/0173_intrinsic_reward_novelty_10rfd.md) | 残件有り | 内的報酬（NGU・E3B のエピソード内の新しさと RND）と方策の組 |
| [issue-20261103-6661d2](../../../management/issues/issue-20261103-6661d2.md) | 未着手 | 1.3 | [issue-20261103-f39d10](../../../management/issues/issue-20261103-f39d10.md), [issue-20261103-2d66d2](../../../management/issues/issue-20261103-2d66d2.md) | コード | RFD | 318_rl_core, 324_dqn, 110_runner_app | 学習基盤 | 機能追加 | [RFD0136](../../../memo/0136_best_checkpoint_10rfd.md) | 残件有り | best checkpoint の自動保存（eval 自己ベスト更新時の重み保持） |
| [issue-20261103-578986](../../../management/issues/issue-20261103-578986.md) | 未着手 | 0.6 | [issue-20261103-b68e49](../../../management/issues/issue-20261103-b68e49.md) | コード | RFD | 318_rl_core, 319_gui, 110_runner_app | 学習基盤 | 機能追加 | [RFD0160](../../../memo/0160_input_target_train_forced_action_10rfd.md) | 残件有り | 操作入力の入力先切替と Train 側の強制行動（学習への反映） |

着手できる 70 / 待ち 5 / 保留 0 / 次版 0 / 解決 0
- 関連する課題: [issue-20261107-949e8a](../../../management/issues/issue-20261107-949e8a.md) — AtariのRAMメトリクスに値の種類数とビットを数える畳み方が無く、Montezumaの訪れた部屋の数や持ち物を記録できない
- 関連する課題: [issue-20261103-3e1cf6](../../../management/issues/issue-20261103-3e1cf6.md) — DropMerge の評価スコア（本家互換スコアほか複数の軸）
- 関連する課題: [issue-20261103-2d66d2](../../../management/issues/issue-20261103-2d66d2.md) — 999: train/eval mode 整理後の排他精査 RFD（草案）
<!-- issues:end -->

# 318 強化学習 共通基盤

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機能カテゴリ](../index.md#機能カテゴリ)

## 概要・責務

- ReplayBufferのExperience、N-step、PER、stack履歴、転送とprefetchの共通契約を引く。RunnerとObserverの実行・通知の所在もまとめる。（出典: [ReplayBuffer](../../design/150_replay_buffer.jp.md)、§1・§2）。
- TrainerとRunnerが学習・評価を実行し、Notifierがeventを配送する。値の収集と保存はObserverとMetricsLoggerへ分ける（出典: [可観測性](../../design/140_observability.jp.md)、§2）。
- Actor初期priorityのhint行は共通層では列の意味を解釈せずに運び、schema検証と推定は注入されたInitialPriorityEstimatorが担う。DQNの5列schemaの正本はDQN系Agent設計である（出典: [ReplayBuffer](../../design/150_replay_buffer.jp.md)、§2.4 PER）。
- ReplayBufferは可変長windowの抽出・転送・prefetchを担い、時間で変わるγでは保存したstep報酬のsnapshotから収益を確定する。窓の規則と利用するγは呼び出し側から渡す（出典: [ReplayBuffer](../../design/150_replay_buffer.jp.md)、§2・§6.1.1）。

## 成立の経緯

- 2026-05-21: EpisodeEvalObserver の Eval メトリクス強化 — 構造分析と修正方針（出典: [PRD003](../../memo/done/003_eval_10prd.md)、題・目的）。
- 2026-06-15: ReplayBuffer サンプリング + H2D を 1ステップ stale で先読みし GPU 学習と overlap する（AsyncLoader 方式）（出典: [ADR 0005](../../adr/0005-sample-prefetch-stale-per.md)、本文）。
- 2026-07-01: VideoLogger 堅牢化 — ffmpeg 異常の fail-fast 検出 と コーデック `auto` 選択 PRD（出典: [PRD025](../../memo/done/025_videolog_codec_auto_ffmpeg_guard_10prd.md)、題・目的）。
- 2026-07-12: Replay優先度更新にはgeneration付きreplay item keyを使用する（出典: [ADR 0011](../../adr/0011-generation-aware-replay-item-key.md)、本文）。
- 2026-07-13: Replay初期優先度ヒントを単一packed carrierとし、completionを専用coordinatorへ分離する（出典: [ADR 0012](../../adr/0012-replay-initial-priority-hint-completion.md)、本文）。
- 2026-07-13: 追加 forward なしの近似 Actor PER 初期優先度 PRD（出典: [PRD035](../../memo/done/035_approx_actor_priority_per_10prd.md)、題・目的）。
- 2026-07-20: ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、題・目的）。
- 2026-08-13: PRD 050: ReplayBuffer ring折り返し時のframe stack sampleability（出典: [PRD050](../../memo/done/050_replay_ring_stack_margin_10prd.md)、題・目的）。
- 2026-08-13: ReplayBufferのsample可能範囲は、ring上書きで失われたstack履歴を必要とするtransitionを除外する（出典: [ADR 0024](../../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md)、本文）。
- 2026-08-14: PRD 052: eval 定義とスケジュールの namespace 分離（出典: [PRD052](../../memo/done/052_eval_schedule_separation_10prd.md)、題・目的）。
- 2026-08-14: configured eval の定義と定期駆動を別 namespace に分離し、生成は schedule エントリが駆動する（出典: [ADR 0027](../../adr/0027-eval-definition-schedule-separation.md)、本文）。
- 2026-08-17: PRD 053: interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化（出典: [PRD053](../../memo/done/053_interval_gate_perf_ema_10prd.md)、題・目的）。
- 2026-08-17: 飛び飛びのカウンタに対する interval を bucket-crossing で定義する（出典: [ADR 0028](../../adr/0028-interval-fires-on-bucket-crossing.md)、本文）。
- 2026-08-21: Conv2d 可視化「実Forwardトレース化」実装指示書（出典: [PRD006](../../memo/done/006_conv2d_visual_trace_10prd.md)、題・目的）。
- 2026-08-21: Device 転送共通部品 `anet/transfer.hpp` と PrefetchingReplayBuffer 適用 PRD（出典: [PRD020](../../memo/done/020_device_transfer_common_part_10prd.md)、題・目的）。
- 2026-08-21: PrefetchingReplayBuffer リファクタ（Learner 埋め込み prefetch を decorator へ分離 + 再現性修正）仕様書（出典: [PRD014](../../memo/done/014_async_loader_prefetch_10prd.md)、題・目的）。
- 2026-08-21: ReplayBuffer PER priority 転送境界整理 PRD（出典: [PRD021](../../memo/done/021_replay_per_priority_transfer_10prd.md)、題・目的）。
- 2026-08-21: Sample+To プリフェッチ（ReplayBuffer サンプリング + H2D を GPU 学習と overlap）仕様書（出典: [PRD013](../../memo/done/013_sample_prefetch_10prd.md)、題・目的）。
- 2026-08-21: V2: 可視化層（metrics_image / HeatMap / 状態スイープ）の TensorDict 復旧 実装指示書（出典: [PRD010](../../memo/done/010_image_metrics_v2_10prd.md)、題・目的）。
- 2026-08-21: anet-lab ハンドオフ: ReplayBuffer の t=0 FrameStack で未初期化 grid を読む問題（出典: [PRD007](../../memo/done/007_rb-grid-corruption_10prd.md)、題・目的）。
- 2026-08-21: mixed targetを運ぶ共通学習結果の接続を更新した（出典: [PRD030](../../memo/done/030_imagecls_mixup_cutmix_10prd.md)、前提事実（実コード確認済み））。
- 2026-08-29: 可塑性 weight norm（feature/readout 分割）PRD（出典: [PRD063](../../memo/done/063_plasticity_weight_norm_10prd.md)、題・目的）。
- 2026-08-29: 可塑性メトリクス（srank / dormant unit 率）PRD（出典: [PRD062](../../memo/done/062_plasticity_metrics_10prd.md)、題・目的）。
- 2026-09-02: Policy churn メトリクス（35_agent_churn）PRD（出典: [PRD066](../../memo/done/066_policy_churn_metrics_10prd.md)、題・目的）。
- 2026-09-03: 評価セッションの N 本集約は BatchEnv デコレータと開始時動的グラントで行う（出典: [ADR 0034](../../adr/0034-eval-session-aggregation-in-batchenv-decorator.md)、本文）。
- 2026-09-04: 評価セッション契約: eval_batch_size（lane 数）と eval_episodes（本数）で N 本平均を 1 点として記録する（出典: [PRD060](../../memo/done/060_eval_batch_episodes_10prd.md)、題・目的）。
- 2026-09-05: メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す（出典: [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md)、題・目的）。
- 2026-09-05: 個体メトリクスは trace チャネルで出し、評価セッションは採用 episode ごとの EpisodeEndEvent と SessionEndEvent を分けて発火する（出典: [ADR 0037](../../adr/0037-metrics-trace-channel-and-session-end-event.md)、本文）。
- 2026-09-13: エピソード長は SHARED でも Step() 回数とし、episode return の総和とは非対称にする（出典: [ADR 0043](../../adr/0043-episode-steps-shared-scope-is-step-count.md)、本文）。
- 2026-09-17: PRD 075: エピソード長の汎用メトリクス化と評価セッションの実行ログ（出典: [PRD075](../../memo/done/075_episode_steps_and_eval_session_log_10prd.md)、題・目的）。
- 2026-09-19: PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、題・目的）。
- 2026-09-20: PRD 077: sampleable index 列の確保を持続バッファへ移す（出典: [PRD077](../../memo/done/077_valid_index_scratch_reuse_10prd.md)、題・目的）。
- 2026-09-20: ReplayBuffer の観測履歴・bootstrap stack 整合性修復 PRD（出典: [PRD078](../../memo/done/078_replay_frame_history_integrity_10prd.md)、題・目的）。
- 2026-09-20: ReplayBufferのframe履歴開始はPush時のepisode_startで確定し、n-step確定metadataから導出しない（ENDをslot単位に持たない部分だけ後にADR0063で改め、履歴開始の判断は維持）（出典: [ADR 0044](../../adr/0044-replay-frame-history-start-from-episode-start-at-push.md)、本文と冒頭の後発決定の注記）。
- 2026-09-22: PRD 076: 終了時の評価セッション排水（出典: [PRD076](../../memo/done/076_eval_session_drain_on_exit_10prd.md)、題・目的）。
- 2026-09-22: 終了時の背景評価セッションは完走を既定とし、設定または上限時間による協調キャンセルで抜けられるようにして、部分集計は設けない（出典: [ADR 0045](../../adr/0045-eval-session-drain-on-exit-wait-or-discard.md)、本文）。
- 2026-09-24: 評価Runnerが新しいdevice指定を解決する接続へ移行した（出典: [PRD070](../../memo/done/070_device_spec_auto_10prd.md)、確定したD1・D9）。
- 2026-10-10: NN実行設定の移行に合わせObserverとTrainerの接続・回帰を確認した（出典: [PRD084](../../memo/done/084_noisynet_10prd.md)、§12 受け入れ条件・§13 実装順と複雑度監査）。
- 2026-10-19: Evaluation Trace の列はメトリクスの定義と分けて番号付きで宣言し、遷移の時点に合う target だけを読む（出典: [ADR 0057](../../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)、本文）。
- 2026-10-19: step trace の近似優先度は Agent が出す step 部品が計算し、Actor の設定キーで eval_panel にだけヒントを作らせることはしない（出典: [ADR 0058](../../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)、本文）。
- 2026-10-20: BatchEnv の Step は終端の lane をリセットせず、Runner が通知のあとに終わったエピソードをリセットする（後にADR0060で取消）（出典: [ADR 0059](../../adr/0059-batchenv-step-does-not-reset-runner-resets-ended-episodes-after-notification.md)、本文）。
- 2026-10-20: 遷移の結果は Step の結果で運び、速報値か確定値かは読むイベントで分ける（出典: [ADR 0060](../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)、本文）。
- 2026-10-20: 遷移結果をStepResultと通知eventで運び終端laneの値を正した（出典: [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md)、題・§4.1）。
- 2026-10-21: step traceの値取得・通知とAgent診断Providerの共通接続を加えた（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§4.2 D5〜D9）。
- 2026-10-24: 時間で変わる割引率は、schedule のときだけ抽選時に保存済みの step 報酬から収益を作り直す（出典: [ADR 0065](../../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)、本文）。
- 2026-10-24: ReplayBuffer は γ を自分で持たず、Push と Sample の入力として呼び出し側が渡す今の γ を使い、schedule の構成では抽選済みのstep報酬列と長さを保持し、prefetchを消費するSampleが返す時点のγで収益を作り直す。ring上書き後のstorageを読み直さない（出典: [PRD101](../../memo/done/101_exploration_schedule_profiled_value_10prd.md)、§4.4 D20）。
- 2026-10-26: Replayに1 stepの記録と窓規則による可変長抽出を足した（出典: [PRD098](../../memo/098_temporl_skip_policy_10prd.md)、§5.1・§5.2）。
- 2026-10-28: 評価セッションのskip指標を学習側のstep座標で読む境界を整えた（出典: [PRD103](../../memo/done/103_temporl_reference_alignment_10prd.md)、§4 D8・§5.4）。
- 2026-10-29: EvalRunnerが最後のstepで行動を指定されたか（強制行動）を`$runner forced`として答える仕様を定めた（出典: [PRD105](../../memo/done/105_eval_trace_arrow_key_forced_action_10prd.md)、D3）。
- 2026-10-29: 判断の旗に依存しないaction_changeとepisode_endの窓規則を足した（出典: [PRD104](../../memo/done/104_temporl_skip_auxiliary_only_10prd.md)、§5.4）。
- 2026-11-03: action_infoからscalarを読む際の実行時型判別を整理する課題を記録した（出典: [PRD178](../../memo/done/178_metrics_action_info_dynamic_cast_10prd.md)、背景と問題・取り組むこと）。
- 2026-11-09: 文書・コメント・設定の小さな食い違いを束ねたPRDを再グリルで決め、InitialPriorityCompleterの`@note`に`GetStats`を足し、ObserverFactoryの古いscalar記法のコメントを140 §6と`ParseMetricTokens`への参照の1行に置き換えることにした（出典: [PRD119](../../memo/119_small_doc_comment_config_fixes_10prd.md)、§4 D3・D9）。
- 2026-11-10: PRD158 で eval schedule の `offset` と `warmup_count` を追加した。列の調整は EpisodeEvalObserver が持ち、RunManager が Env 生成前の値の検証と配線を行う（出典: [PRD158](../../memo/158_eval_schedule_first_fire_10prd.md)、§4・§5.5；[Observability](../../design/140_observability.jp.md)、eval schedule の説明）。
- 2026-11-12: 行動選択に添える入力をMakeActionOptionsにし、Runnerのstep（DoStep・DoUpdateFrame）とActorのMakeActionの引数で都度渡す仕様をグリルで決めた。nn_traceは要求したstepだけ作り、coreに要求の登録を持たない（出典: [PRD170](../../memo/170_nn_trace_capture_unification_10prd.md)、§4 D2・D3・D4・D6；[ADR 0072](../../adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md)）。
- 2026-11-12: `Agent::CreateStepDiagnosticsProvider`の引数を、laneの数・Evaluation Traceの列の鍵の一覧・RunnerのActorを持つ入力（`AgentStepDiagnosticsRequest`）に替える仕様をグリルで決めた。鍵の一覧は計算するかどうかだけを決め、鍵の認識は変えない（出典: [PRD127](../../memo/127_eval_trace_intrinsic_reward_columns_10prd.md)、§4 D3・D9・§5.2・§6.1）。
- 2026-11-12: Actorに、同期に従うもの（`ActorSyncFollower`）をつなぐ口（`AttachSyncFollower`）を足す仕様をグリルで決めた。Actorは、sourceから写し直すとき、sourceを守るロックの中でそれを呼び、相手を所有しない。基底の既定は、つながずに偽を返す（出典: [PRD127](../../memo/127_eval_trace_intrinsic_reward_columns_10prd.md)、§3・§4 D4・§6.1）。
- 2026-11-12: step診断の部品（`AgentStepDiagnosticsProvider`）に、休ませる口（`Suspend`）と使い始める口（`Resume`。Actorの同期が要るかを返す）を足す仕様をグリルで決めた。基底の既定は、何もせず、偽を返す（出典: [PRD127](../../memo/127_eval_trace_intrinsic_reward_columns_10prd.md)、§4 D20・§5.4・§6.1）。
- 2026-11-12: Replay は報酬を成分ごとに記録し、成分ごとの割引率で収益を作る。成分の意味は持たない（出典: [ADR 0073](../../adr/0073-replay-keeps-reward-components-and-builds-returns-per-component-discount.md)、本文）。
- 2026-11-12: 価値の分解のために、Replayが報酬を成分ごとに受け取り、n-stepの収益・抽選時のstepの報酬・初期優先度の見積もりの入力を成分ごとに作る契約を、グリルで決めた。1成分なら今と同じ値になる（出典: [PRD121](../../memo/121_extrinsic_intrinsic_value_decomposition_10prd.md)、§5 D8・§8.2）。
- 2026-11-12: ExperienceSamplesのフィールドの並びを1つにする直しを含む、リファクタリング5件の仕様を、暫定で起票した（出典: [PRD128](../../memo/128_dqn_agent_refactor_bundle_10prd.md)、冒頭の起票の行）。
- 2026-11-13: ExperienceSamplesのフィールドの並びを1つにし、`To`をCPUとCUDAで1つの経路にして未定義のtensorをそのまま返し、効き目の無いstream guardを消す仕様をグリルで決めた（出典: [PRD128](../../memo/128_dqn_agent_refactor_bundle_10prd.md)、§5 D7〜D9・§6.1）。
- 2026-11-19: Notifierの`Attach`・`Detach`・`Notify`などを別々のスレッドから同時に呼んでもObserverの列が壊れないようにし、`Detach`は進行中の通知を待たず、Observerをロックの外で呼ぶ仕様をグリルで決めた（出典: [PRD192](../../memo/192_concurrency_and_boundary_defect_fixes_10prd.md)、§5 D1〜D5；[ADR 0082](../../adr/0082-notifier-detach-does-not-wait-for-in-flight-notifications.md)）。
- 2026-11-19: 呼んでよいスレッドと時点を、Notifierと、キャッシュを持つ`BatchStepResult`・`ReplayInitialPriorityHint`・`BatchActionInfo`の宣言のコメントに書く仕様をグリルで決めた（出典: [PRD192](../../memo/192_concurrency_and_boundary_defect_fixes_10prd.md)、§5 D7）。

## 選ばなかった形

- SampleIndicesをcallerに置いた内部plan/fetch分離 — 不採用: callerの待ちとReplayBuffer内部seamを増やす（出典: [PRD014](../../memo/done/014_async_loader_prefetch_10prd.md)、確定した設計判断）。
- Observerの周期判定をAgent内部のtriggerへ戻す形 — 不採用: roundより細かなevalは既存通知境界では実現できない（出典: [PRD053](../../memo/done/053_interval_gate_perf_ema_10prd.md)、5. スコープ外）。
- step traceをtraceチャネルへ記録する配線 — 範囲外: 鏡写しインスタンスを購読先にする配線と同じ関心である（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§6）。
- 学習側の厳密な優先度の列 — 不採用: EvalPanelには単一networkのforwardしかなくreplayも無いため学習側のonline/targetによる値は得られない（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§6）。
- Event/Observerの型の統一 — 範囲外: step traceとは別の関心。再開条件: 新しいEvent型を足す変更が出たとき（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§6）。
- episode開始通知とViewの開始表示 — 範囲外: 今回のゴールに効かず手動操作の時点ずれと同じ関心。再開条件: 176_qvaluepanel_manual_action_state_alignmentで手動の時点ずれを決めるとき（出典: [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md)、§5）。
- ResetEndedEpisodesの段・外から読む値の写し・予備ALE — 不採用: 段の案は非同期学習を5〜8%遅くし、写しは埋め合わせを残し、予備ALEはAtari固有になる（出典: [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md)、§5・§4.1 D1/D2）。
- Actor設定キーでeval_panelだけにhintを生成させる形 — 不採用: 設定が増え、autoの語が今の条件を表すものになり、n-step窓の所有がAgent外へ出る（出典: [ADR 0058](../../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)、本文）。
- 全targetを許し値の時点差は文書だけで説明する形 — 不採用: 終端行でリセット前後の値が混ざっても見た目で気付けない（出典: [ADR 0057](../../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)、Considered Options）。
- 速報値と確定値を別の鍵名に分ける形 — 不採用: Step結果が2組の値を持つ作りになり、現用の設定キー名も移行する必要がある（出典: [ADR 0060](../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)、Considered Options）。
- skip 用の replay を別に持つ（TempoRL 論文の形） — 不採用: PER・prefetch・ready・世代 key が 2 系統になり、obs の複製でメモリが 2 倍（出典: [PRD098 TempoRL](../../memo/098_temporl_skip_policy_10prd.md)、§4 D11、[ADR 0063](../../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)）。
- Push 時に同一行動の区間を確定して slot に足す形 — 不採用: 同じ起点から短い sub-skip を取れない（出典: [PRD098 TempoRL](../../memo/098_temporl_skip_policy_10prd.md)、§4 D11、[ADR 0063](../../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)）。
- 判断の境界を越えて同じ行動の連続を数える窓 — 不採用: 次の判断で同じ行動を選び直すかが次の状態に依存するため、確率的な env で長い窓に偏りが入る。窓は判断の旗で区切る（出典: [PRD098 TempoRL](../../memo/098_temporl_skip_policy_10prd.md)、§4 D13、[ADR 0063](../../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)）。→ PRD104 で、skip の出力を行動に使わない構成（補助学習のみモード。全 Actor がスキップ長を常に 1 にする）に限り採用。行動の変化で閉じる窓と、上限 J とエピソードの終わりだけで閉じる窓の 2 つ（出典: [PRD104](../../memo/done/104_temporl_skip_auxiliary_only_10prd.md)、§4 D3・D4）。
- k_max に比例する重みで論文の「接続ごとに等しい重み」に合わせる形 — 先送り: 一様な k で始める。再開条件: 短い k の偏りが skip の学習に効くと分かったら。後続の暫定 PRD では、全 k の一括学習（K）が全接続を等しく数える形でこの重みも扱う（出典: [PRD098 TempoRL](../../memo/098_temporl_skip_policy_10prd.md)、§8、[PRD183 TempoRL の後続](../../memo/183_temporl_skip_followups_10prd.md)、§4 D7）。
- eviction 統計（last_evicted 系）も sampleable の範囲を基準にする形 — 不採用（却下）: 追い出される slot は必ず history margin を過ぎてから追い出されるので、wrap 後に統計が恒常的に 0 件になり、一度も引かれずに追い出された割合の指標が死ぬ（出典: [ADR 0024](../../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md)、Considered Options；[PRD050](../../memo/done/050_replay_ring_stack_margin_10prd.md)、0. 決定一覧 D7）。
- sampleable index を返す GetValidIndices1D を非 const にする形 — 不採用（採らない）: const な呼び出し元 4 箇所が、unique_ptr の operator-> が const を伝播しないことに依存して非 const を呼ぶ形になり、将来 propagate_const などで塞がれると通らなくなる（出典: [PRD077](../../memo/done/077_valid_index_scratch_reuse_10prd.md)、D4）。
- 初期優先度のヒントを TensorDict で任意の named tensor として運ぶ形 — 不採用（棄却）: 複数 value の CPU 化が複数の D2H になり、1 key へ pack すると単一 tensor の carrier と実質同じになる（出典: [ADR 0012](../../adr/0012-replay-initial-priority-hint-completion.md)、Considered Options）。
- 転送の in-flight 数を blocking cap で抑える形 — 不採用（入れない）: worker の Acquire と learner の future.get の間でデッドロックを生む。in-flight は呼び側の 1-deep prefetch で back-pressure され有界になる（出典: [PRD020](../../memo/done/020_device_transfer_common_part_10prd.md)、Solution（概要）設計判断 5）。
- 評価セッションで group ごとに固定の quota を割り当てる形 — 不採用（却下）: 不偏だが、episode 長が不均一なほど最も遅い group の quota 合計に律速される（出典: [PRD060](../../memo/done/060_eval_batch_episodes_10prd.md)、11. 却下した旧解決候補）。
- partial Reset・masked Step（採用権を持たない group を Step の対象から外す） — 先送り: P2 には導入しない。再開条件: P3 の実測で必要性が出たら、別の契約として設計する（出典: [PRD060](../../memo/done/060_eval_batch_episodes_10prd.md)、4.3 評価セッションと採用規則）。
- 保存済みの収益は Push 時の γ、bootstrap は学習時の γ の 2 時点を契約にする形 — 不採用: 変化中に学習の target の中へ既知のずれを作る。Replay が step 報酬を保存しているので作り直す方が安い（出典: [PRD101](../../memo/done/101_exploration_schedule_profiled_value_10prd.md)、§4.4 D20・§7、[ADR 0065](../../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)、Considered Options）。
- γ の schedule で常に抽選時に収益を作り直す形 — 不採用: 定数でも計算が増え、足す順を揃えないと既存 Run との checksum が変わる。定数は保存値の経路に留める（出典: [ADR 0065](../../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)、Considered Options）。
- γ を今回 constant だけに限る形 — 不採用: 器としては γ だけ空く。schedule のときだけ作り直す形で両立できた（出典: [PRD101](../../memo/done/101_exploration_schedule_profiled_value_10prd.md)、§7、[ADR 0065](../../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)、Considered Options）。
- 強制したかを行動の情報（`WithAction`の結果）で答える形 — 不採用: 基底・DQN系・MuZero系の行動の情報とテストの型を変え、「`WithAction`の結果＝強制」という意味を公開の型に付ける。強制するのはRunnerで、Runnerが答えれば1か所で済む（出典: [PRD105](../../memo/done/105_eval_trace_arrow_key_forced_action_10prd.md)、D3）。
- metrics_scalar.txtにscalar記法の要素ごとの語の一覧を写す形 — 不採用: 語を足すたびに`ParseMetricTokens`・140 §6・設定ファイルの3か所を直すことになり、知らない語はkeyとして黙って扱われるので、古い写しどおりに書くと値が記録されない。ObserverFactoryのコメントは参照の1行だけにする（出典: [PRD119](../../memo/119_small_doc_comment_config_fixes_10prd.md)、§4 D9・§7）。
- `interval = 0` で止めた評価 tag と、schedule の記述が無い tag で、参照の skip の通知のレベルを分ける形 — 不採用: ADR 0027 が 1 つにまとめた dormant をまた 2 つに分け、休止の出どころを運ぶ分岐が増える（出典: [PRD117](../../memo/117_warn_policy_and_unread_config_10prd.md)、D8・§8）。
- PER の抽選が埋まらないとき Run を止める形 — 不採用: 過去の Run のログ 616 本で 1 度も起きておらず、起きる条件が分からないまま止めると正当な極初期の状況で Run を落とす恐れがある。代替の是非は課題に起票した（出典: [PRD117](../../memo/117_warn_policy_and_unread_config_10prd.md)、D9・§8・§9 の 20）。
- Runnerが要求の数を持ち、nn_traceを読む側がRunnerに登録する形 — 不採用: stepの結果が引数に現れない数で決まり、入力の側に「事前の登録」という2つ目の渡し方ができる（出典: [PRD170](../../memo/170_nn_trace_capture_unification_10prd.md)、§4.1；[ADR 0072](../../adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md)、Considered Options）。
- Observerが「nn_traceが要る」と宣言し、RunnerがNotifierに聞く形 — 不採用: 全イベント・全Runnerが共有するObserverの仕組みにnn_traceの話が入り、Learnerの購読に近い形がActor側にも増える（出典: [PRD170](../../memo/170_nn_trace_capture_unification_10prd.md)、§4.1；[ADR 0072](../../adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md)、Considered Options）。
- 購読ヒントと同じ経路でnn_traceの要否を渡す形 — 不採用: 学習の前に1回だけ渡す仕組みで、Runの途中で開閉するpaneに追従できない（出典: [PRD170](../../memo/170_nn_trace_capture_unification_10prd.md)、§2.5・§4.1）。
- MakeActionの3つ目の引数を真偽の値にする形、既定値を付けない形、引数2つの版をvirtualでない関数として残す形 — 不採用: 入力が増えるたびに基底の引数が変わる、テストの呼び出し約65か所に空の値を足すことになる、派生クラスで基底の2引数の版が名前ごと隠れる（出典: [PRD170](../../memo/170_nn_trace_capture_unification_10prd.md)、§4 D2・D3・§4.1）。
- 公開のMakeActionをvirtualでない1本にして既定値をそこだけに書き、実装は別名のvirtualをoverrideする形 — 不採用: 既定値は空の構造体で食い違う余地がほとんど無く、Runnerのstepも同じ作りにそろえるとEvalRunnerで基底のDoStepが名前ごと隠れる（出典: [PRD170](../../memo/170_nn_trace_capture_unification_10prd.md)、§4.1）。
- EvalRunnerの手動の行動の引数をMakeActionOptionsにまとめる形 — 範囲外: PRD161の範囲（出典: [PRD170](../../memo/170_nn_trace_capture_unification_10prd.md)、§3.2・§9）。
- Learnerの購読ヒントを、開始前の宣言と呼び出しごとの要求に分けて見直す形 — 範囲外: 別の関心（出典: [PRD170](../../memo/170_nn_trace_capture_unification_10prd.md)、§8の1）。
- Replayはstepの報酬だけを成分で持ち、収益はLearnerが抽選のあとに作る形 — 不採用（却下）: 成分が1つのときは今と同じ値を保つために定数の割引率でPush時に保存する経路を残す必要があり、成分があるときだけ別の経路にすると、収益を作る経路が2通りになる（出典: [ADR 0073](../../adr/0073-replay-keeps-reward-components-and-builds-returns-per-component-discount.md)、Considered Options、[PRD121](../../memo/121_extrinsic_intrinsic_value_decomposition_10prd.md)、§5 D8）。
- 内的報酬という名前の2本目の報酬をReplayに足す形 — 不採用（却下）: 汎用のReplayに、特定のアルゴリズムの名前と意味が入る（出典: [ADR 0073](../../adr/0073-replay-keeps-reward-components-and-builds-returns-per-component-discount.md)、Considered Options）。
- game overで切らない形のために、終端の観測からbootstrapする形 — 不採用: Replayはgame overの終端の観測を保存しておらず、次のエピソードの最初の観測からのbootstrapが、Replayを変えずに作れる（出典: [PRD121](../../memo/121_extrinsic_intrinsic_value_decomposition_10prd.md)、§4・§5 D7・§5.1）。
- 窓の収益を成分ごとに作る形 — 先送り: 窓の規則を使うのはskipだけで、価値の分解とskipの併用はERRORなので、今は使う構成が無い。成分が2つ以上のReplayに窓の規則を渡したらERRORにする。再開条件: 価値の分解とTempoRLの併用を作るとき（出典: [PRD121](../../memo/121_extrinsic_intrinsic_value_decomposition_10prd.md)、§5 D8・§5.1、[ADR 0073](../../adr/0073-replay-keeps-reward-components-and-builds-returns-per-component-discount.md)、Consequences）。
- `ExperienceSamples::To`で、必須のフィールドが未定義なら止める形 — 不採用: どのフィールドが必須かは標本を使う側が決める（出典: [PRD128](../../memo/128_dqn_agent_refactor_bundle_10prd.md)、§5 D7・§5.1）。
- `ExperienceSamples::To`の経路ごとの未定義の扱いを、引数で切り替えて保つ形 — 不採用: 片方だけ直した跡と読める食い違いを引数として残す（出典: [PRD128](../../memo/128_dqn_agent_refactor_bundle_10prd.md)、§5 D7・§5.1）。
- `ExperienceSamples::To`のstream guardをCUDAのときだけ残す形 — 不採用: 今のdeviceの今のstreamを設定し直すだけで効き目が無い（出典: [PRD128](../../memo/128_dqn_agent_refactor_bundle_10prd.md)、§2.2・§5 D8・§5.1）。
- `Detach`が、そのObserverへの進行中の呼び出しが終わるまで待つ形 — 不採用: 待たれているObserverがGUIスレッドの持つロックやGUIの処理を待つと互いに待って固まり、Observerの中から自分を外すときの特例も要る（出典: [PRD192](../../memo/192_concurrency_and_boundary_defect_fixes_10prd.md)、§5.1；[ADR 0082](../../adr/0082-notifier-detach-does-not-wait-for-in-flight-notifications.md)、Considered Options）。
- 通知のたびに、ロックの中でObserverの一覧を写す形 — 不採用: Observerの数に比例する処理が通知のたびに増える（出典: [PRD192](../../memo/192_concurrency_and_boundary_defect_fixes_10prd.md)、§5 D5・§5.1）。
- 守りの無いキャッシュ2つ（`GetPayloadCpu`・`GetAction(device)`）にも`GetEpisodeEndMask`と同じ守りを足す形 — 不採用: 同じ実体を同時に呼ぶ経路が無く、行動の1 stepごとに通る箇所へ使われない守りを足すことになる（出典: [PRD192](../../memo/192_concurrency_and_boundary_defect_fixes_10prd.md)、§1.2・§5 D7）。
- 強制行動のdeviceを、呼ぶ側のEvalRunnerで合わせる形 — 不採用: 汎用のEvalRunnerがDQN系の都合のためにdeviceをそろえる形になる（出典: [PRD192](../../memo/192_concurrency_and_boundary_defect_fixes_10prd.md)、§5 D9・§5.1）。
- 学習スレッドからPanelかFrameを触る3つのObserver（TrainPanel、EvalPanelが学習側のstep数を受けるObserver、RunnerFrameの状態表示用のObserver）を、終了の順序に頼らない作りに変える形 — 不採用: 今は終了の順序で守られていて、今ある危険が無く、得が小さい（出典: [PRD192](../../memo/192_concurrency_and_boundary_defect_fixes_10prd.md)、§2）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- PushとExtractのstorage shared/exclusive lockによる排他を維持する（出典: [PRD014](../../memo/done/014_async_loader_prefetch_10prd.md)、1.2 ReplayBuffer）。
- scratchの再利用でもindexの昇順を変えず、wrap・stack・n-step・PERの既存回帰を通す（出典: [PRD077](../../memo/done/077_valid_index_scratch_reuse_10prd.md)、D5: 昇順契約を維持する）。
- 公開ReplayBuffer API、replay item keyの符号化、sampling分布、PER completionとeviction統計のready基準を変更しない（出典: [PRD078](../../memo/done/078_replay_frame_history_integrity_10prd.md)、6.2 モジュールの責務）。
- 既存のHead/SN FP32保護と用途別AMP精度を維持する。Noisy無効の既存チェーンは決定論backendでmetrics checksumを比較し、sample時の乱数列の割当変更を無効側へ持ち込まない。比較の相手は実装前に保存した基準の実行体・依存DLL・出力とし、開始時の状態を再現できないHEADからの後追い比較は基準にしない（出典: [PRD084](../../memo/done/084_noisynet_10prd.md)、§8.2・§12・§13；[PRD060](../../memo/done/060_eval_batch_episodes_10prd.md)、6. 受入条件 8）。
- 読み口と配線の整理でmetrics出力・定義レコード・WARN・fail-fastの文言と時機を変えない。同seed・決定的モードの短Runでchecksum・defs・WARNを比較する（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§4.1 D3・§8 1）。
- ヒント計算の共有化前後でactor_approxの同seed・決定的Runの学習metrics checksumを一致させ、既定のEvalPanel networkとRunnerカウンタの通知前後の時機を変えない（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§4.2 D6/D7・§8 3）。
- Agentとreplayへ入る値、loss・Q、episode return・長さを保持する。Atari2構成と他Env・逐次/非同期の7構成で全点を比較し、実時間は交互3組、ばらつきが大きければ5組で性能低下を確認する（出典: [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md)、ゴールG3・§7 1/4）。
- slot に 1 step の記録を足しても、sampling 分布・item key・PER の completion と eviction 統計の ready 基準・Push と Extract の lock・index の昇順を変えない。スキップが無効なら ready の遅れは max(1, n_step) のまま（出典: [PRD098 TempoRL](../../memo/098_temporl_skip_policy_10prd.md)、§5.1・§6 の 11）。
- EvalRunnerに強制の値を足す前後で、PRD098の検証資材の段0の短縮条件のmetrics checksumを4環境とも一致させる（出典: [PRD105](../../memo/done/105_eval_trace_arrow_key_forced_action_10prd.md)、§7の受入6）。
- windowを使わない経路の乱数消費とsample列を保ち、窓を増やすときは通常転送・prefetch・probeも確かめる。定数γでは保存済み収益を使い、prefetch後のγの確定で上書き済みringを読み直さない（出典: [ReplayBuffer](../../design/150_replay_buffer.jp.md)、§2・§6.1.1・§8）。
- train observerを外すのは学習停止と背景評価の排水の後とし、同じNotifierを走査中のworkerとDetachを競合させない（出典: [可観測性](../../design/140_observability.jp.md)、§5.2）。
- 新しい設定が既定のときは、着手前に保存した実行体・設定を使い、スキップ無効と既存の有効構成のmetrics checksumを一致させる。古い設定と新しい実行体を組み合わせる比較を保つ（出典: [PRD103](../../memo/done/103_temporl_reference_alignment_10prd.md)、§4 D14・§5.7・§7の受入1）。
- 既定のfull・decisionの組み合わせではActorの行動・乱数消費とReplayの抽選列を変えず、PRD103の検証資材でmetrics checksumを一致させる（出典: [PRD104](../../memo/done/104_temporl_skip_auxiliary_only_10prd.md)、§5.7・§7の受入1）。
- MakeActionOptionsを渡さないstepと行動選択は、nn_traceを作らないことのほかは今と同じ結果にし、決定的な設定の短いRunのmetrics checksumを変更の前後で一致させる（出典: [PRD170](../../memo/170_nn_trace_capture_unification_10prd.md)、§5.2・§7のA2・A11）。
- 報酬の成分を足しても、終端、打ち切り、次観測、実際に進んだstep数、抽選の分布、item key、PERのcompletionとeviction統計のreadyの基準、PushとExtractのlock、indexの昇順を変えない。成分が1つのとき、記録・収益（窓の収益を含む）・標本・乱数の消費を今と同じに保ち、既存の断定を期待値を変えずに通す（出典: [PRD121](../../memo/121_extrinsic_intrinsic_value_decomposition_10prd.md)、§8.2・§9.2の2）。
- `ExperienceSamples`のフィールドの並びを変えるときは、CPUの経路（GridMaze・LunarLander）とCUDAの経路（Atari qbert・DropMerge）の両方の構成で、実装前に保存した実行体と全scalarの点を比べ、学習の速さを交互3巡以上で確かめる（出典: [PRD128](../../memo/128_dqn_agent_refactor_bundle_10prd.md)、§5 D16・D18、§6.4）。
- Notifierの排他を足しても、Observerを呼ぶ順（登録順）と、Observerが動くスレッド（`Notify`を呼んだスレッド）、例外の伝わり方を変えない。RunnerFrameの終了時の順序（排水のあとに`Detach`）も変えない（出典: [PRD192](../../memo/192_concurrency_and_boundary_defect_fixes_10prd.md)、§5 D4・D6・§6.2）。
- 既定の設定のRunの結果を変えないことは、PRD103の検証資材と同じ6構成・短縮条件で、実装の前後の実行体の、性能・資源以外の全scalarの一致で確かめる（出典: [PRD192](../../memo/192_concurrency_and_boundary_defect_fixes_10prd.md)、§7のA13）。

## 関係

- 設計: [ReplayBuffer](../../design/150_replay_buffer.jp.md)、§1・§2。
- 可変長windowとγ・検証手順: [ReplayBuffer](../../design/150_replay_buffer.jp.md)、§6.1.1・§8。通知の寿命は[可観測性](../../design/140_observability.jp.md)、§5.2。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [muzero_rb.cpp](../../../src/algo/muzero/src/muzero_rb.cpp) / [muzero_rb.hpp](../../../src/algo/muzero/src/muzero_rb.hpp)
- [observers.hpp](../../../src/core/base/include/anet/observers.hpp)
- [replay_buffer.hpp](../../../src/core/base/include/anet/replay_buffer.hpp)
- [replay_experience.hpp](../../../src/core/base/include/anet/replay_experience.hpp)
- [rl.hpp](../../../src/core/base/include/anet/rl.hpp)
- [serialize.hpp](../../../src/core/base/include/anet/serialize.hpp)
- [trainer.hpp](../../../src/core/base/include/anet/trainer.hpp)
- [episode_end_test.cpp](../../../src/core/base/src/episode_end_test.cpp)
- [observers.cpp](../../../src/core/base/src/observers.cpp) / [observers_test.cpp](../../../src/core/base/src/observers_test.cpp)
- [replay_buffer_test.cpp](../../../src/core/base/src/replay_buffer_test.cpp)
- [replay_buffer_impl.cpp](../../../src/core/base/src/replay_buffer_impl.cpp) / [replay_buffer_impl.hpp](../../../src/core/base/src/replay_buffer_impl.hpp)
- [replay_experience.cpp](../../../src/core/base/src/replay_experience.cpp)
- [rl.cpp](../../../src/core/base/src/rl.cpp)
- [serialize.cpp](../../../src/core/base/src/serialize.cpp)
- [trainer.cpp](../../../src/core/base/src/trainer.cpp) / [trainer_test.cpp](../../../src/core/base/src/trainer_test.cpp)

## PRD の一覧

点検が作り直す。手で直さない。

- [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [PRD035](../../memo/done/035_approx_actor_priority_per_10prd.md) — 追加 forward なしの近似 Actor PER 初期優先度 PRD
- [PRD050](../../memo/done/050_replay_ring_stack_margin_10prd.md) — PRD 050: ReplayBuffer ring折り返し時のframe stack sampleability
- [PRD052](../../memo/done/052_eval_schedule_separation_10prd.md) — PRD 052: eval 定義とスケジュールの namespace 分離
- [PRD053](../../memo/done/053_interval_gate_perf_ema_10prd.md) — PRD 053: interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化
- [PRD060](../../memo/done/060_eval_batch_episodes_10prd.md) — 評価セッション契約: eval_batch_size（lane 数）と eval_episodes（本数）で N 本平均を 1 点として記録する
- [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [PRD062](../../memo/done/062_plasticity_metrics_10prd.md) — 可塑性メトリクス（srank / dormant unit 率）PRD
- [PRD063](../../memo/done/063_plasticity_weight_norm_10prd.md) — 可塑性 weight norm（feature/readout 分割）PRD
- [PRD066](../../memo/done/066_policy_churn_metrics_10prd.md) — Policy churn メトリクス（35_agent_churn）PRD
- [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md) — メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す
- [PRD070](../../memo/done/070_device_spec_auto_10prd.md) — PRD 070: device 指定の auto 対応とキー体系統一
- [PRD075](../../memo/done/075_episode_steps_and_eval_session_log_10prd.md) — PRD 075: エピソード長の汎用メトリクス化と評価セッションの実行ログ
- [PRD076](../../memo/done/076_eval_session_drain_on_exit_10prd.md) — PRD 076: 終了時の評価セッション排水
- [PRD077](../../memo/done/077_valid_index_scratch_reuse_10prd.md) — PRD 077: sampleable index 列の確保を持続バッファへ移す
- [PRD078](../../memo/done/078_replay_frame_history_integrity_10prd.md) — ReplayBuffer の観測履歴・bootstrap stack 整合性修復 PRD
- [PRD084](../../memo/done/084_noisynet_10prd.md) — NoisyNet (Noisy Networks for Exploration) PRD
- [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD
- [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md) — 遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD
- [PRD098](../../memo/098_temporl_skip_policy_10prd.md) — TempoRL 対応（スキップ: 選んだ行動を続ける長さの学習）PRD
- [PRD101](../../memo/done/101_exploration_schedule_profiled_value_10prd.md) — 探索と学習のパラメータを ProfiledValue で時間変化させる PRD
- [PRD103](../../memo/done/103_temporl_reference_alignment_10prd.md) — TempoRL 第二弾（著者の実装との差の洗い出しと是正）PRD
- [PRD104](../../memo/done/104_temporl_skip_auxiliary_only_10prd.md) — TempoRL の補助学習のみモード（毎 step 判断し、スキップの出力は補助タスクとしてだけ学ぶ）PRD
- [PRD105](../../memo/done/105_eval_trace_arrow_key_forced_action_10prd.md) — Evaluation Trace の矢印キーと強制した step の見分け、表の並べ替えと行のダンプ PRD
- [PRD117](../../memo/117_warn_policy_and_unread_config_10prd.md) — PRD117: WARN の意味の線引きと、読まれない設定の検出 PRD
- [PRD119](../../memo/119_small_doc_comment_config_fixes_10prd.md) — PRD119: 文書・コメント・設定の小さな食い違いをまとめて直す
- [PRD121](../../memo/121_extrinsic_intrinsic_value_decomposition_10prd.md) — 外的と内的の価値の分解（Q を外的報酬と内的報酬の 2 つの Head で学ぶ） PRD
- [PRD122](../../memo/122_small_refactor_and_test_gap_fixes_10prd.md) — PRD122: 小さなリファクタリングとテストの空白をまとめて片づける
- [PRD127](../../memo/127_eval_trace_intrinsic_reward_columns_10prd.md) — Evaluation Trace で RND の内的報酬を見る（統計を変えない採点と step 診断の鍵）PRD
- [PRD128](../../memo/128_dqn_agent_refactor_bundle_10prd.md) — PRD128: DQN 系 Agent まわりのリファクタリング 5 件をまとめて片づける（価値の分解の前に） PRD
- [PRD131](../../memo/131_prd_9xx_renumber_by_creation_date_10prd.md) — PRD131: 9xx の番号の PRD 57 本を、起票の日の順に 900 未満の連番へ振り直す PRD
- [PRD158](../../memo/158_eval_schedule_first_fire_10prd.md) — PRD158: eval スケジュールの発火の列 — 学習直後の評価を既定で止め、発火の時点をずらせるようにする
- [PRD170](../../memo/170_nn_trace_capture_unification_10prd.md) — PRD170: forward の名前付き出力を capture に一本化し、nn_trace を要求した step だけ作る PRD
- [PRD182](../../memo/182_source_layout_10prd.md) — ソース配置の見直し（置き場の規則、Agent の系統ごとのライブラリ、DQN のファイル分割、テストの参照範囲）PRD
- [PRD192](../../memo/192_concurrency_and_boundary_defect_fixes_10prd.md) — PRD192: 監査で見つかった不具合 5 件をまとめて直す（Notifier の同時実行、動画の画像の大きさ、強制行動の device、重複登録、QValuePanel の平滑化）
- [PRD194](../../memo/194_config_key_full_names_in_messages_10prd.md) — PRD194: エラー・WARN・INFO が示す設定キーを完全キーにする（device の二重の fatal、指標名、用語の不整合 6 件を含む）

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | PRD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261103-955e50](../../management/issues/issue-20261103-955e50.md) | 調査中 | 1.2 | — | コード | PRD | 318_rl_core, 317_metrics, 110_runner_app, 311_dev_tools, 120_metrics_viewer | UI | 機能追加 | [PRD081](../../memo/081_evalpanel_episode_trace_10prd.md) | 残件有り | EvalPanel のゲームを trace へ出す — dormant タグの trace を鏡写しインスタンスへ結び付け、trace 行の時刻で振り返る |
| [issue-20261103-1beaf9](../../management/issues/issue-20261103-1beaf9.md) | 未着手 | 3.3 | — | コード | 監査 | 318_rl_core | 学習基盤 | 不具合 | — | 残件有り | PipelineTrainRunnerのイベントで、経験と学習結果が1 stepずれた組になる |
| [issue-20261111-7cc67b](../../management/issues/issue-20261111-7cc67b.md) | 未着手 | 3.3 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 未確認 | EvalRunner の 1 step が、多重定義 4 つ・-1 の特別な値・bool の引数で、パネルからの 1 step と評価 session の 1 step を切り替えている |
| [issue-20261111-4a6b6f](../../management/issues/issue-20261111-4a6b6f.md) | 未着手 | 3.0 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 未確認 | RunManager のコンストラクタが 277 行あり、評価の設定の解釈・名前の予約・構築・ログ・配線が 1 つの関数に並んでいる |
| [issue-20261111-585e60](../../management/issues/issue-20261111-585e60.md) | 未着手 | 3.0 | — | コード | 監査 | 318_rl_core, 321_agent_core, 315_nn, 110_runner_app | 基盤 | リファクタリング | — | 未確認 | 名前やコメントが実際の動きと合わない関数・クラス・設定項目がある（BatchState の IsDone、DefaultAgentFactory、NetworkGraphVizConfig の 2 項目、Runner の 4 関数） |
| [issue-20261111-ea3483](../../management/issues/issue-20261111-ea3483.md) | 未着手 | 3.0 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 未確認 | scalar メトリクスの Observer が、イベントの中身を 6〜9 個の引数にばらして 3 段持ち回り、同じ 8 引数のコンストラクタを 4 回書いている |
| [issue-20261103-9df5c5](../../management/issues/issue-20261103-9df5c5.md) | 未着手 | 2.7 | — | コード | PRD | 318_rl_core, 110_runner_app, 314_tensor | 学習基盤 | 不具合 | [PRD146](../../memo/146_notifier_scoped_detach_10prd.md) | 残件有り | Notifier::AttachScoped の Detach 契約修復 — wrapper ハンドルを返す |
| [issue-20261107-16ffd8](../../management/issues/issue-20261107-16ffd8.md) | 未着手 | 2.7 | — | コード | PRD | 318_rl_core, 322_env_individual, 110_runner_app | 学習基盤 | 不整合 | — | 未確認 | ERROR のログを出したあとも Run を止めずに続ける箇所（Env・Agent の作成失敗、DropMerge の果物の範囲外）が Fail-Fast 原則と合わない |
| [issue-20261111-645ecc](../../management/issues/issue-20261111-645ecc.md) | 未着手 | 2.7 | — | コード | 監査 | 317_metrics, 318_rl_core | 学習基盤 | リファクタリング | — | 未確認 | メトリクス宣言の語（event・step・target）と enum の対応が 5 か所以上に分かれ、解決済みの定義も同じ値を文字列と enum の 2 形で持っている |
| [issue-20261111-fe23e0](../../management/issues/issue-20261111-fe23e0.md) | 未着手 | 2.7 | — | コード | 監査 | 318_rl_core, 320_env_core | 学習基盤 | リファクタリング | — | 未確認 | BatchEnvSpec に「group の数」「group と lane の対応」を返す関数が無く、使う側が三項演算を 19 か所に書き、終端の mask を CPU へ移す処理も関数ごとに書いている |
| [issue-20261103-b2a0b1](../../management/issues/issue-20261103-b2a0b1.md) | 未着手 | 2.5 | — | コード | 監査 | 318_rl_core | 学習基盤 | 不具合 | — | 残件有り | ActionSpec::GetShapeが連続行動で次元数ぶんの1を並べた形を返す |
| [issue-20261103-34d35d](../../management/issues/issue-20261103-34d35d.md) | 未着手 | 2.3 | — | コード | PRD | 318_rl_core | 学習基盤 | 不具合 | [PRD177](../../memo/177_episode_end_read_semantics_10prd.md) | 残件有り | `@episode_end` で読む値を、どの target でもそのエピソードの値にそろえる |
| [issue-20261111-2c251b](../../management/issues/issue-20261111-2c251b.md) | 未着手 | 2.3 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 未確認 | Replay の窓の規則 3 クラスが同じ骨格の写しで、PER のスカラーのキーも一覧・条件式・分岐の 3〜4 か所に並んでいる |
| [issue-20261103-5addea](../../management/issues/issue-20261103-5addea.md) | 未着手 | 2.2 | — | コード | PRD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [PRD079](../../memo/079_cpp_internal_linkage_convention_10prd.md) | 残件有り | PRD 079: .cpp の内部リンケージと namespace インデント規約 |
| [issue-20261103-78c643](../../management/issues/issue-20261103-78c643.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | ObserverFactoryのコンストラクタが331行で、graphの節だけが共通の字句解析を使わずに独自に読む |
| [issue-20261103-785e68](../../management/issues/issue-20261103-785e68.md) | 未着手 | 2.0 | — | コード | PRD | 315_nn, 318_rl_core, 321_agent_core, 324_dqn, 319_gui, 320_env_core, 314_tensor, 312_util, 110_runner_app, 322_env_individual, 325_muzero | 開発 | リファクタリング | [PRD169](../../memo/169_outside_core_extension_layout_10prd.md) | 残件有り | anet-core の外で実装する拡張点とコード配置の規約 |
| [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md) | 未着手 | 2.0（引継ぎ） | — | 成果物 | PRD | 321_agent_core, 318_rl_core, 324_dqn, 314_tensor, 325_muzero | 学習基盤 | リファクタリング | [PRD161](../../memo/161_observation_pipeline_10prd.md) | 残件有り | 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化 |
| [issue-20261111-07f4ed](../../management/issues/issue-20261111-07f4ed.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 未確認 | 「N エピソードに 1 回だけ記録する」状態が Conv2dVisualizationObserver と GraphVizObserver に少し違う形で書かれ、記録を始める条件と終える条件が食い違っている |
| [issue-20261111-54c7e1](../../management/issues/issue-20261111-54c7e1.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core, 321_agent_core, 315_nn, 319_gui, 110_runner_app | 基盤 | リファクタリング | — | 未確認 | 呼ばれない関数・型・マクロと読まれないメンバ・届かない分岐が、core と Runner に合わせて 1,100 行以上残っている |
| [issue-20261112-983afe](../../management/issues/issue-20261112-983afe.md) | 未着手 | 2.0 | — | コード | その他 | 324_dqn, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | — | 残件有り | KFMの学習がボスへ行き着くかが偶然に頼る（1階の廊下の位置が観測に無く、Qも内的報酬も左へ進む向きを持てない） |
| [issue-20261103-0ef7b8](../../management/issues/issue-20261103-0ef7b8.md) | 未着手 | 1.6 | — | コード | PRD | 317_metrics, 311_dev_tools, 316_config, 318_rl_core, 321_agent_core, 120_metrics_viewer, 110_runner_app | UI | 機能改善 | [PRD108](../../memo/108_metrics_tag_numbering_10prd.md) | 残件有り | メトリクスの tag の番号を全体で振り直す（群の番号と群の中の番号） |
| [issue-20261113-d38960](../../management/issues/issue-20261113-d38960.md) | 未着手 | 1.6 | — | コード | 実装 | 110_runner_app, 324_dqn, 318_rl_core | UI | 性能改善 | — | 残件有り | EvalPanel のモデル同期が UI スレッドで学習ロックを待ち、画面の更新や操作応答を止め得る |
| [issue-20261103-75b8da](../../management/issues/issue-20261103-75b8da.md) | 未着手 | 1.5 | — | コード | PRD | 321_agent_core, 318_rl_core | アルゴリズム | 機能改善 | [PRD168](../../memo/168_nstep_target_correction_10prd.md) | 残件有り | n-step の目標値の補正（窓の途中の Munchausen の項・方策オフの補正） |
| [issue-20261103-182ca1](../../management/issues/issue-20261103-182ca1.md) | 未着手 | 1.4 | — | コード | PRD | 318_rl_core, 324_dqn, 315_nn, 320_env_core | 学習基盤 | 性能改善 | [PRD180](../../memo/180_core_perf_candidates_202607_10prd.md) | 残件有り | 2026-07 のコアの詳細レビューで挙がった性能の改善候補 5 件 |
| [issue-20261103-50644a](../../management/issues/issue-20261103-50644a.md) | 未着手 | 1.4 | — | コード | PRD | 317_metrics, 320_env_core, 321_agent_core, 318_rl_core, 110_runner_app | UI | 機能追加 | [PRD184](../../memo/184_eval_session_step_trace_10prd.md) | 残件有り | 評価 session の step trace を Run の記録に残す |
| [issue-20261103-52bda0](../../management/issues/issue-20261103-52bda0.md) | 未着手 | 1.4 | — | コード | PRD | 318_rl_core, 321_agent_core, 120_metrics_viewer, 311_dev_tools | UI | 機能改善 | [PRD096](../../memo/096_metrics_step_axis_interval_10prd.md) | 残件有り | メトリクスの横軸と間隔を数える単位を分ける |
| [issue-20261103-aa8701](../../management/issues/issue-20261103-aa8701.md) | 未着手 | 1.4 | — | コード | PRD | 318_rl_core, 321_agent_core, 324_dqn, 320_env_core | 学習基盤 | 機能追加 | [PRD143](../../memo/143_serialize_10prd.md) | 残件有り | Run Serialization（Run 全体 Save / Load） |
| [issue-20261115-c89660](../../management/issues/issue-20261115-c89660.md) | 未着手 | 1.4 | — | コード | その他 | 110_runner_app, 321_agent_core, 318_rl_core | UI | 機能追加 | — | 未確認 | Runner の Conv2dPanel と metrics.image の conv2d に、RND の target と predictor の畳み込みの出力が出ない |
| [issue-20261103-f39d10](../../management/issues/issue-20261103-f39d10.md) | 未着手 | 1.3 | — | コード | PRD | 318_rl_core, 324_dqn | 学習基盤 | 不具合 | [PRD141](../../memo/141_background_eval_snapshot_ordering_10prd.md) | 残件有り | Background eval の network snapshot 順序保証 |
| [issue-20261103-07629c](../../management/issues/issue-20261103-07629c.md) | 未着手 | 1.2 | — | コード | PRD | 110_runner_app, 318_rl_core, 319_gui | UI | 不具合 | [PRD176](../../memo/176_qvaluepanel_manual_action_state_alignment_10prd.md) | 残件有り | QValuePanel の Q の表示と、手動で選んだ行動が当たる状態のずれ |
| [issue-20261103-8c3c6d](../../management/issues/issue-20261103-8c3c6d.md) | 未着手 | 1.2 | — | コード | PRD | 318_rl_core, 324_dqn, 321_agent_core, 322_env_individual | 学習基盤 | 機能追加 | [PRD171](../../memo/171_replay_eviction_sample_count_10prd.md) | 残件有り | replay の遷移が消えるまでの抽選回数と優先度を、PER の外の区分で測る |
| [issue-20261103-dbe666](../../management/issues/issue-20261103-dbe666.md) | 未着手 | 1.2 | — | コード | PRD | 322_env_individual, 318_rl_core, 311_dev_tools | Env | 調査 | [PRD163](../../memo/163_dropmerge_neet_diagnostics_10prd.md) | 残件有り | DropMerge の NEET 診断（trace で足りるか確かめてから指標化） |
| [issue-20261110-d21e5f](../../management/issues/issue-20261110-d21e5f.md) | 未着手 | 1.2 | — | コード | PRD | 318_rl_core, 325_muzero | 学習基盤 | 不具合 | — | 未確認 | MuZeroのReplayで、truncatedで終わるときに補う価値が、打ち切った次の状態ではなく最後のstep自身のroot_valueになっている |
| [issue-20261111-e90762](../../management/issues/issue-20261111-e90762.md) | 未着手 | 1.2 | — | コード | PRD | 318_rl_core | 学習基盤 | 不整合 | — | 未確認 | PERの抽選が埋まらないときに残りを一様の抽選で埋める代替が、明示した要求を黙って代替しない規約と食い違う |
| [issue-20261103-51220d](../../management/issues/issue-20261103-51220d.md) | 未着手 | 1.1 | — | コード | PRD | 318_rl_core, 321_agent_core, 322_env_individual, 317_metrics | 学習基盤 | 機能追加 | [PRD144](../../memo/144_episode_forensics_10prd.md) | 残件有り | Episode Forensics（特殊挙動エピソード自動捕捉・事後調査） |
| [issue-20261116-cc5942](../../management/issues/issue-20261116-cc5942.md) | 未着手 | 1.1 | — | コード | その他 | 318_rl_core, 321_agent_core, 314_tensor, 324_dqn | 学習基盤 | 性能改善 | — | 残件有り | 同じデータを何度もdevice間で転送している箇所を洗い出して減らす（Replayのobsとnext_obsの重なるフレーム、RNDの採点とActorが別々に送る観測など） |
| [issue-20261111-d1d12b](../../management/issues/issue-20261111-d1d12b.md) | 未着手 | 1.0 | — | コード | 監査 | 321_agent_core, 315_nn, 318_rl_core, 317_metrics, 316_config, 312_util, 313_profiling, 314_tensor, 322_env_individual, 110_runner_app | 学習基盤 | リファクタリング | — | 未確認 | 今は起こさないリファクタリング候補 16 群を暫定で 1 件にまとめた（手を付けるときに群ごとに分けて起票し直す） |
| [issue-20261103-cfe994](../../management/issues/issue-20261103-cfe994.md) | 未着手 | 0.9 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | イベント種別ごとにObserver interface・RunnerScopedラッパ・Notifierメソッド・MetricsLog派生を複製している |
| [issue-20261103-c5c791](../../management/issues/issue-20261103-c5c791.md) | 未着手 | 0.9 | — | コード | PRD | 322_env_individual, 318_rl_core | 学習基盤 | 調査 | [PRD179](../../memo/179_prd097_performance_remeasurement_10prd.md) | 残件有り | PRD097: 判定不能4構成の性能再測定とDropMerge serial追加費用の調査 |
| [issue-20261103-f40ae8](../../management/issues/issue-20261103-f40ae8.md) | 未着手 | 0.9 | — | コード | PRD | 317_metrics, 314_tensor, 318_rl_core | UI | 性能改善 | [PRD056](../../memo/056_timehistogram_fine_grid_10prd.md) | 残件有り | TimeHistogram fine-bin保持・描画時集約 |
| [issue-20261103-afe841](../../management/issues/issue-20261103-afe841.md) | 未着手 | 0.8 | — | コード | PRD | 321_agent_core, 324_dqn, 318_rl_core | 学習基盤 | 機能追加 | [PRD145](../../memo/145_staged_batch_replay_schedule_10prd.md) | 残件有り | DefaultDQN Staged Batch / Replay Schedule |
| [issue-20261103-969184](../../management/issues/issue-20261103-969184.md) | 未着手 | 0.6 | — | コード | PRD | 321_agent_core, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | [PRD174](../../memo/174_llm_actor_10prd.md) | 残件有り | LLM Actor（LLM の事前知識で行動する Actor。単独 Agent と DQN ハイブリッド） |
| [issue-20261119-497fce](../../management/issues/issue-20261119-497fce.md) | 未着手 | 0.6 | — | コード | その他 | 324_dqn, 318_rl_core | アルゴリズム | 機能追加 | — | 未確認 | 内的の価値を非エピソードにしても、n-stepの和がエピソードの終わりで切れ、Burdaらのように内的の収益がエピソードをまたいで続かない |
| [issue-20261103-145ee2](../../management/issues/issue-20261103-145ee2.md) | 未着手 | 0.5 | — | コード | PRD | 321_agent_core, 318_rl_core, 316_config | 学習基盤 | 機能追加 | [PRD152](../../memo/152_metric_targeted_hyperparam_control_10prd.md) | 残件有り | メトリクス目標値によるハイパラ自律調整 |
| [issue-20261103-9bd71e](../../management/issues/issue-20261103-9bd71e.md) | 未着手 | 2.0 | [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md) | コード | PRD | 321_agent_core, 318_rl_core, 324_dqn, 317_metrics | 学習基盤 | 機能追加 | [PRD142](../../memo/142_agent_prev_action_obs_10prd.md) | 残件有り | 直前行動観測の共通部品（`prev_action` フレーム段） |
| [issue-20261103-ca1a85](../../management/issues/issue-20261103-ca1a85.md) | 未着手 | 1.6 | [issue-20261103-1beaf9](../../management/issues/issue-20261103-1beaf9.md) | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | SerialとPipelineのDoStepが初期化・経験の組み立て・通知・カウンタ更新を別々に持ち、既に食い違っている |
| [issue-20261103-124da5](../../management/issues/issue-20261103-124da5.md) | 未着手 | 1.5 | [issue-20261103-3e1cf6](../../management/issues/issue-20261103-3e1cf6.md) | コード | PRD | 321_agent_core, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | [PRD167](../../memo/167_smdp_transition_duration_10prd.md) | 残件有り | SMDP 対応（遷移ごとの経過時間を割引に通す） |
| [issue-20261103-736e39](../../management/issues/issue-20261103-736e39.md) | 未着手 | 1.4 | [issue-20261103-b754ff](../../management/issues/issue-20261103-b754ff.md), [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md), [issue-20261103-9bd71e](../../management/issues/issue-20261103-9bd71e.md), [issue-20261103-75b8da](../../management/issues/issue-20261103-75b8da.md), [issue-20261107-949e8a](../../management/issues/issue-20261107-949e8a.md) | コード | PRD | 324_dqn, 321_agent_core, 318_rl_core, 322_env_individual, 110_runner_app | アルゴリズム | 機能追加 | [PRD173](../../memo/173_intrinsic_reward_novelty_10prd.md) | 残件有り | 内的報酬（NGU・E3B のエピソード内の新しさと RND）と方策の組 |
| [issue-20261103-6661d2](../../management/issues/issue-20261103-6661d2.md) | 未着手 | 1.3 | [issue-20261103-f39d10](../../management/issues/issue-20261103-f39d10.md), [issue-20261103-2d66d2](../../management/issues/issue-20261103-2d66d2.md) | コード | PRD | 318_rl_core, 324_dqn, 110_runner_app | 学習基盤 | 機能追加 | [PRD136](../../memo/136_best_checkpoint_10prd.md) | 残件有り | best checkpoint の自動保存（eval 自己ベスト更新時の重み保持） |
| [issue-20261103-b68e49](../../management/issues/issue-20261103-b68e49.md) | 未着手 | 1.3 | [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md) | コード | PRD | 110_runner_app, 318_rl_core, 322_env_individual | UI | 機能追加 | [PRD159](../../memo/159_input_mapping_joystick_10prd.md) | 残件有り | 操作入力の Joystick 対応と操作マッピングの設定化（EvalPanel） |
| [issue-20261103-578986](../../management/issues/issue-20261103-578986.md) | 未着手 | 0.6 | [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md), [issue-20261103-b68e49](../../management/issues/issue-20261103-b68e49.md) | コード | PRD | 318_rl_core, 319_gui, 110_runner_app | 学習基盤 | 機能追加 | [PRD160](../../memo/160_input_target_train_forced_action_10prd.md) | 残件有り | 操作入力の入力先切替と Train 側の強制行動（学習への反映） |

- 関連する課題: [issue-20261107-949e8a](../../management/issues/issue-20261107-949e8a.md) — AtariのRAMメトリクスに値の種類数とビットを数える畳み方が無く、Montezumaの訪れた部屋の数や持ち物を記録できない
着手できる 44 / 待ち 7 / 保留 0 / 次版 0 / 解決 0
- 関連する課題: [issue-20261103-b754ff](../../management/issues/issue-20261103-b754ff.md) — stucker.stack_keysに存在しないキー名を書いても検出されない
- 関連する課題: [issue-20261103-3e1cf6](../../management/issues/issue-20261103-3e1cf6.md) — DropMerge の評価スコア（本家互換スコアほか複数の軸）
- 関連する課題: [issue-20261103-2d66d2](../../management/issues/issue-20261103-2d66d2.md) — 999: train/eval mode 整理後の排他精査 PRD（草案）
<!-- issues:end -->

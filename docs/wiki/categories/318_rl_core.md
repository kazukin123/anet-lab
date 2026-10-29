# 318 強化学習 共通基盤

## 概要・責務

- ReplayBufferのExperience、N-step、PER、stack履歴、転送とprefetchの共通契約を引く。RunnerとObserverの実行・通知の所在もまとめる。（出典: [ReplayBuffer](../../design/150_replay_buffer.jp.md)、§1・§2）。
- TrainerとRunnerが学習・評価を実行し、Notifierがeventを配送する。値の収集と保存はObserverとMetricsLoggerへ分ける（出典: [可観測性](../../design/140_observability.jp.md)、§2）。
- Actor初期priorityのhint行は共通層では列の意味を解釈せずに運び、schema検証と推定は注入されたInitialPriorityEstimatorが担う。DQNの3列schemaの正本はDQN系Agent設計である（出典: [ReplayBuffer](../../design/150_replay_buffer.jp.md)、§2.4 PER）。

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
- 2026-09-20: ReplayBufferのframe履歴開始はPush時のepisode_startで確定し、n-step確定metadataから導出しない（出典: [ADR 0044](../../adr/0044-replay-frame-history-start-from-episode-start-at-push.md)、本文）。
- 2026-09-22: PRD 076: 終了時の評価セッション排水（出典: [PRD076](../../memo/done/076_eval_session_drain_on_exit_10prd.md)、題・目的）。
- 2026-09-22: 終了時の背景評価セッションは完走を既定とし、設定または上限時間による協調キャンセルで抜けられるようにして、部分集計は設けない（出典: [ADR 0045](../../adr/0045-eval-session-drain-on-exit-wait-or-discard.md)、本文）。
- 2026-09-24: 評価Runnerが新しいdevice指定を解決する接続へ移行した（出典: [PRD070](../../memo/070_device_spec_auto_10prd.md)、確定したD1・D9）。
- 2026-10-10: NN実行設定の移行に合わせObserverとTrainerの接続・回帰を確認した（出典: [PRD084](../../memo/084_noisynet_10prd.md)、§12 受け入れ条件・§13 実装順と複雑度監査）。
- 2026-10-19: Evaluation Trace の列はメトリクスの定義と分けて番号付きで宣言し、遷移の時点に合う target だけを読む（出典: [ADR 0057](../../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)、本文）。
- 2026-10-19: step trace の近似優先度は Agent が出す step 部品が計算し、Actor の設定キーで eval_panel にだけヒントを作らせることはしない（出典: [ADR 0058](../../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)、本文）。
- 2026-10-20: BatchEnv の Step は終端の lane をリセットせず、Runner が通知のあとに終わったエピソードをリセットする（後にADR0060で取消）（出典: [ADR 0059](../../adr/0059-batchenv-step-does-not-reset-runner-resets-ended-episodes-after-notification.md)、本文）。
- 2026-10-20: 遷移の結果は Step の結果で運び、速報値か確定値かは読むイベントで分ける（出典: [ADR 0060](../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)、本文）。
- 2026-10-20: 遷移結果をStepResultと通知eventで運び終端laneの値を正した（出典: [PRD097](../../memo/097_env_auto_reset_after_notification_10prd.md)、題・§4.1）。
- 2026-10-21: step traceの値取得・通知とAgent診断Providerの共通接続を加えた（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§4.2 D5〜D9）。
- 2026-10-24: 時間で変わる割引率は、schedule のときだけ抽選時に保存済みの step 報酬から収益を作り直す（出典: [ADR 0065](../../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)、本文）。
- 2026-10-24: ReplayBuffer は γ を自分で持たず、Push と Sample の入力として呼び出し側が渡す今の γ を使い、schedule の構成では抽選済みのstep報酬列と長さを保持し、prefetchを消費するSampleが返す時点のγで収益を作り直す。ring上書き後のstorageを読み直さない（出典: [PRD101](../../memo/101_exploration_schedule_profiled_value_10prd.md)、§4.4 D20）。
- 2026-10-29: EvalRunnerが最後のstepで行動を指定されたか（強制行動）を`$runner forced`として答える仕様を定めた（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、D3）。

## 選ばなかった形

- SampleIndicesをcallerに置いた内部plan/fetch分離 — 不採用: callerの待ちとReplayBuffer内部seamを増やす（出典: [PRD014](../../memo/done/014_async_loader_prefetch_10prd.md)、確定した設計判断）。
- Observerの周期判定をAgent内部のtriggerへ戻す形 — 不採用: roundより細かなevalは既存通知境界では実現できない（出典: [PRD053](../../memo/done/053_interval_gate_perf_ema_10prd.md)、5. スコープ外）。
- step traceをtraceチャネルへ記録する配線 — 範囲外: 鏡写しインスタンスを購読先にする配線と同じ関心である（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§6）。
- 学習側の厳密な優先度の列 — 不採用: EvalPanelには単一networkのforwardしかなくreplayも無いため学習側のonline/targetによる値は得られない（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§6）。
- Event/Observerの型の統一 — 範囲外: step traceとは別の関心。再開条件: 新しいEvent型を足す変更が出たとき（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§6）。
- episode開始通知とViewの開始表示 — 範囲外: 今回のゴールに効かず手動操作の時点ずれと同じ関心。再開条件: 999_qvaluepanel_manual_action_state_alignmentで手動の時点ずれを決めるとき（出典: [PRD097](../../memo/097_env_auto_reset_after_notification_10prd.md)、§5）。
- ResetEndedEpisodesの段・外から読む値の写し・予備ALE — 不採用: 段の案は非同期学習を5〜8%遅くし、写しは埋め合わせを残し、予備ALEはAtari固有になる（出典: [PRD097](../../memo/097_env_auto_reset_after_notification_10prd.md)、§5・§4.1 D1/D2）。
- Actor設定キーでeval_panelだけにhintを生成させる形 — 不採用: 設定が増え、autoの語が今の条件を表すものになり、n-step窓の所有がAgent外へ出る（出典: [ADR 0058](../../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)、本文）。
- 全targetを許し値の時点差は文書だけで説明する形 — 不採用: 終端行でリセット前後の値が混ざっても見た目で気付けない（出典: [ADR 0057](../../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)、Considered Options）。
- 速報値と確定値を別の鍵名に分ける形 — 不採用: Step結果が2組の値を持つ作りになり、現用の設定キー名も移行する必要がある（出典: [ADR 0060](../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)、Considered Options）。
- skip 用の replay を別に持つ（TempoRL 論文の形） — 不採用: PER・prefetch・ready・世代 key が 2 系統になり、obs の複製でメモリが 2 倍（出典: [PRD098 TempoRL](../../memo/098_temporl_skip_policy_10prd.md)、§4 D11、[ADR 0063](../../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)）。
- Push 時に同一行動の区間を確定して slot に足す形 — 不採用: 同じ起点から短い sub-skip を取れない（出典: [PRD098 TempoRL](../../memo/098_temporl_skip_policy_10prd.md)、§4 D11、[ADR 0063](../../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)）。
- 判断の境界を越えて同じ行動の連続を数える窓 — 不採用: 次の判断で同じ行動を選び直すかが次の状態に依存するため、確率的な env で長い窓に偏りが入る。窓は判断の旗で区切る（出典: [PRD098 TempoRL](../../memo/098_temporl_skip_policy_10prd.md)、§4 D13、[ADR 0063](../../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)）。→ PRD104 で、skip の出力を行動に使わない構成（補助学習のみモード。全 Actor がスキップ長を常に 1 にする）に限り採用。行動の変化で閉じる窓と、上限 J とエピソードの終わりだけで閉じる窓の 2 つ（出典: [PRD104](../../memo/104_temporl_skip_auxiliary_only_10prd.md)、§4 D3・D4）。
- k_max に比例する重みで論文の「接続ごとに等しい重み」に合わせる形 — 先送り: 一様な k で始める。再開条件: 短い k の偏りが skip の学習に効くと分かったら。後続の暫定 PRD では、全 k の一括学習（K）が全接続を等しく数える形でこの重みも扱う（出典: [PRD098 TempoRL](../../memo/098_temporl_skip_policy_10prd.md)、§8、[999 TempoRL の後続](../../memo/999_temporl_skip_followups_10prd.md)、§4 D7）。
- eviction 統計（last_evicted 系）も sampleable の範囲を基準にする形 — 不採用（却下）: 追い出される slot は必ず history margin を過ぎてから追い出されるので、wrap 後に統計が恒常的に 0 件になり、一度も引かれずに追い出された割合の指標が死ぬ（出典: [ADR 0024](../../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md)、Considered Options；[PRD050](../../memo/done/050_replay_ring_stack_margin_10prd.md)、0. 決定一覧 D7）。
- sampleable index を返す GetValidIndices1D を非 const にする形 — 不採用（採らない）: const な呼び出し元 4 箇所が、unique_ptr の operator-> が const を伝播しないことに依存して非 const を呼ぶ形になり、将来 propagate_const などで塞がれると通らなくなる（出典: [PRD077](../../memo/done/077_valid_index_scratch_reuse_10prd.md)、D4）。
- 初期優先度のヒントを TensorDict で任意の named tensor として運ぶ形 — 不採用（棄却）: 複数 value の CPU 化が複数の D2H になり、1 key へ pack すると単一 tensor の carrier と実質同じになる（出典: [ADR 0012](../../adr/0012-replay-initial-priority-hint-completion.md)、Considered Options）。
- 転送の in-flight 数を blocking cap で抑える形 — 不採用（入れない）: worker の Acquire と learner の future.get の間でデッドロックを生む。in-flight は呼び側の 1-deep prefetch で back-pressure され有界になる（出典: [PRD020](../../memo/done/020_device_transfer_common_part_10prd.md)、Solution（概要）設計判断 5）。
- 評価セッションで group ごとに固定の quota を割り当てる形 — 不採用（却下）: 不偏だが、episode 長が不均一なほど最も遅い group の quota 合計に律速される（出典: [PRD060](../../memo/done/060_eval_batch_episodes_10prd.md)、11. 却下した旧解決候補）。
- partial Reset・masked Step（採用権を持たない group を Step の対象から外す） — 先送り: P2 には導入しない。再開条件: P3 の実測で必要性が出たら、別の契約として設計する（出典: [PRD060](../../memo/done/060_eval_batch_episodes_10prd.md)、4.3 評価セッションと採用規則）。
- 保存済みの収益は Push 時の γ、bootstrap は学習時の γ の 2 時点を契約にする形 — 不採用: 変化中に学習の target の中へ既知のずれを作る。Replay が step 報酬を保存しているので作り直す方が安い（出典: [PRD101](../../memo/101_exploration_schedule_profiled_value_10prd.md)、§4.4 D20・§7、[ADR 0065](../../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)、Considered Options）。
- γ の schedule で常に抽選時に収益を作り直す形 — 不採用: 定数でも計算が増え、足す順を揃えないと既存 Run との checksum が変わる。定数は保存値の経路に留める（出典: [ADR 0065](../../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)、Considered Options）。
- γ を今回 constant だけに限る形 — 不採用: 器としては γ だけ空く。schedule のときだけ作り直す形で両立できた（出典: [PRD101](../../memo/101_exploration_schedule_profiled_value_10prd.md)、§7、[ADR 0065](../../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)、Considered Options）。
- 強制したかを行動の情報（`WithAction`の結果）で答える形 — 不採用: 基底・DQN系・MuZero系の行動の情報とテストの型を変え、「`WithAction`の結果＝強制」という意味を公開の型に付ける。強制するのはRunnerで、Runnerが答えれば1か所で済む（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、D3）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- PushとExtractのstorage shared/exclusive lockによる排他を維持する（出典: [PRD014](../../memo/done/014_async_loader_prefetch_10prd.md)、1.2 ReplayBuffer）。
- scratchの再利用でもindexの昇順を変えず、wrap・stack・n-step・PERの既存回帰を通す（出典: [PRD077](../../memo/done/077_valid_index_scratch_reuse_10prd.md)、D5: 昇順契約を維持する）。
- 公開ReplayBuffer API、replay item keyの符号化、sampling分布、PER completionとeviction統計のready基準を変更しない（出典: [PRD078](../../memo/done/078_replay_frame_history_integrity_10prd.md)、6.2 モジュールの責務）。
- 既存のHead/SN FP32保護と用途別AMP精度を維持する。Noisy無効の既存チェーンは決定論backendでmetrics checksumを比較し、sample時の乱数列の割当変更を無効側へ持ち込まない。比較の相手は実装前に保存した基準の実行体・依存DLL・出力とし、開始時の状態を再現できないHEADからの後追い比較は基準にしない（出典: [PRD084](../../memo/084_noisynet_10prd.md)、§8.2・§12・§13；[PRD060](../../memo/done/060_eval_batch_episodes_10prd.md)、6. 受入条件 8）。
- 読み口と配線の整理でmetrics出力・定義レコード・WARN・fail-fastの文言と時機を変えない。同seed・決定的モードの短Runでchecksum・defs・WARNを比較する（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§4.1 D3・§8 1）。
- ヒント計算の共有化前後でactor_approxの同seed・決定的Runの学習metrics checksumを一致させ、既定のEvalPanel networkとRunnerカウンタの通知前後の時機を変えない（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§4.2 D6/D7・§8 3）。
- Agentとreplayへ入る値、loss・Q、episode return・長さを保持する。Atari2構成と他Env・逐次/非同期の7構成で全点を比較し、実時間は交互3組、ばらつきが大きければ5組で性能低下を確認する（出典: [PRD097](../../memo/097_env_auto_reset_after_notification_10prd.md)、ゴールG3・§7 1/4）。
- slot に 1 step の記録を足しても、sampling 分布・item key・PER の completion と eviction 統計の ready 基準・Push と Extract の lock・index の昇順を変えない。スキップが無効なら ready の遅れは max(1, n_step) のまま（出典: [PRD098 TempoRL](../../memo/098_temporl_skip_policy_10prd.md)、§5.1・§6 の 11）。
- EvalRunnerに強制の値を足す前後で、PRD098の検証資材の段0の短縮条件のmetrics checksumを4環境とも一致させる（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、§7の受入6）。

## 関係

- 設計: [ReplayBuffer](../../design/150_replay_buffer.jp.md)、§1・§2。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [observers.hpp](../../../core/anet-core/include/anet/observers.hpp)
- [replay_buffer.hpp](../../../core/anet-core/include/anet/replay_buffer.hpp)
- [rl.hpp](../../../core/anet-core/include/anet/rl.hpp)
- [serialize.hpp](../../../core/anet-core/include/anet/serialize.hpp)
- [trainer.hpp](../../../core/anet-core/include/anet/trainer.hpp)
- [episode_end_test.cpp](../../../core/anet-core/src/episode_end_test.cpp)
- [muzero_rb.cpp](../../../core/anet-core/src/muzero_rb.cpp) / [muzero_rb.hpp](../../../core/anet-core/src/muzero_rb.hpp)
- [observers.cpp](../../../core/anet-core/src/observers.cpp) / [observers_test.cpp](../../../core/anet-core/src/observers_test.cpp)
- [replay_buffer_test.cpp](../../../core/anet-core/src/replay_buffer_test.cpp)
- [replay_buffer_impl.cpp](../../../core/anet-core/src/replay_buffer_impl.cpp) / [replay_buffer_impl.hpp](../../../core/anet-core/src/replay_buffer_impl.hpp)
- [rl.cpp](../../../core/anet-core/src/rl.cpp)
- [serialize.cpp](../../../core/anet-core/src/serialize.cpp)
- [trainer.cpp](../../../core/anet-core/src/trainer.cpp) / [trainer_test.cpp](../../../core/anet-core/src/trainer_test.cpp)

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
- [PRD070](../../memo/070_device_spec_auto_10prd.md) — PRD 070: device 指定の auto 対応とキー体系統一
- [PRD075](../../memo/done/075_episode_steps_and_eval_session_log_10prd.md) — PRD 075: エピソード長の汎用メトリクス化と評価セッションの実行ログ
- [PRD076](../../memo/done/076_eval_session_drain_on_exit_10prd.md) — PRD 076: 終了時の評価セッション排水
- [PRD077](../../memo/done/077_valid_index_scratch_reuse_10prd.md) — PRD 077: sampleable index 列の確保を持続バッファへ移す
- [PRD078](../../memo/done/078_replay_frame_history_integrity_10prd.md) — ReplayBuffer の観測履歴・bootstrap stack 整合性修復 PRD
- [PRD084](../../memo/084_noisynet_10prd.md) — NoisyNet (Noisy Networks for Exploration) PRD
- [PRD094](../../memo/094_evalpanel_step_history_10prd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD
- [PRD097](../../memo/097_env_auto_reset_after_notification_10prd.md) — 遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD

## 台帳の指摘

台帳のツールが作り直す。手で直さない。

<!-- registry:begin -->
- F-0006 P0 defect — Notifierが無ロックのvectorを、GUIスレッドのAttach/Detachと学習スレッドのNotifyで同時に触る
- F-0012 P1 refactor — イベント種別ごとにObserver interface・RunnerScopedラッパ・Notifierメソッド・MetricsLog派生を複製している
- F-0009 P2 defect — PipelineTrainRunnerのイベントで、経験と学習結果が1 stepずれた組になる
- F-0015 P2 refactor — SerialとPipelineのDoStepが初期化・経験の組み立て・通知・カウンタ更新を別々に持ち、既に食い違っている
- F-0017 P2 refactor — ObserverFactoryのコンストラクタが331行で、graphの節だけが共通の字句解析を使わずに独自に読む
- F-0025 P2 drift — train.runner_typeがpipeline以外の任意の文字列を黙ってserialとして扱う
- F-0010 P3 defect — ActionSpec::GetShapeが連続行動で次元数ぶんの1を並べた形を返す
- F-0027 P3 test-gap — MuZeroReplayBufferの価値ターゲットの打ち切りと、unroll窓のmaskにテストが無い
- F-0034 P3 refactor — InitialPriorityCompleterのロックの前提の@noteがGetStatsに触れていない
- F-0035 P3 test-gap — DefaultReplayBufferのコンストラクタの、priority storeとcompleterを対で持つ不変条件にテストが無い

[台帳の一覧](../../../reports/registry/findings.md)
<!-- registry:end -->

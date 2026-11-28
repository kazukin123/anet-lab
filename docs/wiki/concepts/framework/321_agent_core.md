---
title: 321 Agent 共通基盤
page_kind: フレームワーク
aliases:
  - 321_agent_core
description: Agent、Actor、Learnerの生成・行動・更新と、Agentが所有するResource、更新部品が所有するStateの境界を扱う。
updated: 2026-11-28
---
# 321 Agent 共通基盤

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [フレームワーク](../../index.md#フレームワーク)

## 概要・責務

- Agent、Actor、Learnerの生成・行動・更新と、Agentが所有するResource、更新部品が所有するStateの境界を扱う。（出典: [Agentと学習](../../../design/110_agents_and_learning.jp.md)、§2.1・§2.3・§3）。
- DefaultDQN の Actor は観測と CPU 上の開始の印を行動選択用の Forwarder に渡す。Learner は ReplayBuffer が返す spec で Sample 用を作る（出典: [Forwarder と Perception](../../../design/135_forwarder_and_perception.jp.md)、§3）。
- 強制行動は`MakeActionOptions::forced_actions`でlaneごとに渡す。Actorは方策の選択を終えてから強制を反映し、返した行動をRunnerが実行する。TempoRLでは強制したstepを長さ1の判断として記録し、そのlaneは次のstepで判断し直す（出典: [Agentと学習](../../../design/110_agents_and_learning.jp.md)、§2.2・§6.1）。
- Agentは基盤の外で書ける拡張点で、`anet/agent.hpp`・`anet/rl.hpp`で`AgentFactory`・`Agent`・`Actor`を実装し、`RegisterAgentFactory`で登録する（出典: [ANETフレームワーク全体概要](../../../design/010_framework_overview.jp.md)、§4.3）。
- `Actor::MakeAction`は呼び出しごとの選択を`MakeActionOptions`で受け（nn_traceの要求は既定で偽）、要求をActorやNotifierへ登録しない。Agentは、Runnerが系統の公開初期化関数を呼んで登録し、同じ関数を何度呼んでも登録は1回で、同じclass IDを直接登録すると例外になって先の登録が残る。DQNのprobe専用の乱数は、内側のLearnerがAgentの基のseedと用途ごとの名前から作って持つ（出典: [Agentと学習](../../../design/110_agents_and_learning.jp.md)、§2.1・§2.3・§2.4）。
- 登録済みAgentは`DefaultDQNAgent`・`RainbowAgent`（Runでは使わない）・`PPOAgent`・`MuZeroAgent`（試作）・`ImageClsAgent`である。Learnerは0件以上の更新結果を返し、1回の呼出しで何回更新するかとReplayBufferを使うかは具象実装が決め、更新が無ければ空の一覧を返す。`LearnEvent`を作るのは結果を受け取ったRunnerである（出典: [Agentと学習](../../../design/110_agents_and_learning.jp.md)、§2.1・§2.4・§6.2・§8）。
- DefaultDQN の TempoRL（`skip.enabled`）の Actor は、スキップを続けている lane も含めて primary を毎 step 全 lane で forward し、判断する lane だけが選んだ行動と長さ j を採り、残りの j−1 回は同じ行動を返す。skip の forward も続けている lane を含めて 1 回行い、`single` を含むどの `skip.max_length_mode` でも行う（出典: [DQN系Agent](../../../design/200_dqn_agents.jp.md)、§2.1.8・§6.5）。
- 方策ベース（方策を直接学ぶ）のAgentが使うcoreの共通部品（行動の分布`CategoricalDistribution`、advantageの推定`ComputeGae`、Headの合成`CompositeHeadFactory`、ためた経験の全体の読み出し`LaneTimeTensorStore::GetAll`）は、特定のAgentの名前を持たず、`anet-core-base`の公開ヘッダから使う。部品は計算だけを受け持ち、入力と同じdeviceで計算して値をCPUへ読み戻さず、設定キーを読まない。損失の式、rolloutをいつ更新に使うか、optimizer、設定の読み取りと値域の検証は使う側のAgentが持つ（出典: [方策ベースの学習の共通部品](../../../design/115_on_policy_learning.jp.md)、§1.1・§1.2）。
- `CategoricalDistribution`は、行（laneやミニバッチの1件）ごとのlogits`[行, 行動数]`から作る離散行動の分布で、抽選（logitsと同じdeviceの生成器を渡す）、確率が最大の行動、log確率、entropy、全行動の確率とlog確率を返す。logitsはfloat32にしてから`log_softmax`で正規化し、確率0の行動のentropyの項は0と数える。logitsが勾配を持てばlog確率とentropyも勾配を持ち、勾配を切るのは使う側である。logitsの未定義・型・rank・行動数2未満は構築で、行動の未定義・型・形は`ComputeLogProb`で止まる（出典: [方策ベースの学習の共通部品](../../../design/115_on_policy_learning.jp.md)、§2.1・§2.2）。

## 成立の経緯

- 2026-07-12: Actor初期優先度は平均Q空間の近似とし、Learner target-network評価/UQE targetの再現を見送る（出典: [ADR 0010](../../../adr/0010-actor-priority-mean-q-approx.md)、本文）。
- 2026-07-12: Replay優先度更新にはgeneration付きreplay item keyを使用する（出典: [ADR 0011](../../../adr/0011-generation-aware-replay-item-key.md)、本文）。
- 2026-07-13: Replay初期優先度ヒントを単一packed carrierとし、completionを専用coordinatorへ分離する（出典: [ADR 0012](../../../adr/0012-replay-initial-priority-hint-completion.md)、本文）。
- 2026-07-13: 追加 forward なしの近似 Actor PER 初期優先度 RFD（出典: [RFD0035](../../../memo/done/0035_approx_actor_priority_per_10rfd.md)、題・目的）。
- 2026-07-15: Actor network方針の既定値解決をAgentへ戻す（出典: [ADR 0013](../../../adr/0013-actor-network-resource-policy.md)、本文）。
- 2026-07-16: DefaultDQN Train Actor定期network snapshot RFD（出典: [RFD0036](../../../memo/done/0036_train_actor_periodic_snapshot_10rfd.md)、題・目的）。
- 2026-07-20: ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter（出典: [RFD0034](../../../memo/done/0034_imagecls_batch_input_10rfd.md)、題・目的）。
- 2026-08-09: IQN+UQE の Q 系値は同一 forward の risk-biased action score を使う（出典: [ADR 0019](../../../adr/0019-iqn-uqe-score-without-extra-forward.md)、本文）。
- 2026-08-10: 044: IQN tau sampling modes(stratified / systematic / antithetic)（出典: [RFD0044](../../../memo/done/0044_iqn_tau_stratified_sampling_10rfd.md)、題・目的）。
- 2026-08-11: RFD0047: IQN探索P0診断メトリクス（出典: [RFD0047](../../../memo/done/0047_iqn_search_p0_metrics_10rfd.md)、題・目的）。
- 2026-08-11: RFD0048: 分位tail探索診断メトリクス（出典: [RFD0048](../../../memo/done/0048_quantile_tail_metrics_10rfd.md)、題・目的）。
- 2026-08-21: Conv2d 可視化「実Forwardトレース化」実装指示書（出典: [RFD0006](../../../memo/done/0006_conv2d_visual_trace_10rfd.md)、題・目的）。
- 2026-08-21: DefaultDQNAgent Dueling: V/A ストリームを body DAG で分岐できるようにする（出典: [RFD0009](../../../memo/done/0009_dueling_dag_head_10rfd.md)、題・目的）。
- 2026-08-21: FusedAdamW（ATen `_fused_adamw_` 直接呼び出し）導入 仕様書（出典: [RFD0011](../../../memo/done/0011_fused_adamw_10rfd.md)、題・目的）。
- 2026-08-21: IQN 対応改修（quantile_mode 導入・bind `*` 演算子・IQNHead/IQNLearner）（出典: [RFD0001](../../../memo/done/0001_iqn_10rfd.md)、題・目的）。
- 2026-08-21: ImageCls BF16(AMP)対応 + 汎用 LinearHead による Head 化（出典: [RFD0033](../../../memo/done/0033_imagecls_bf16_head_10rfd.md)、題・目的）。
- 2026-08-21: PrefetchingReplayBuffer リファクタ（Learner 埋め込み prefetch を decorator へ分離 + 再現性修正）仕様書（出典: [RFD0014](../../../memo/done/0014_async_loader_prefetch_10rfd.md)、題・目的）。
- 2026-08-21: Sample+To プリフェッチ（ReplayBuffer サンプリング + H2D を GPU 学習と overlap）仕様書（出典: [RFD0013](../../../memo/done/0013_sample_prefetch_10rfd.md)、題・目的）。
- 2026-08-21: Transformed Bellman Operator (TBO) 導入 仕様書（出典: [RFD0008](<../../../memo/done/0008_Transformed Bellman Operator_10rfd.md>)、題・目的）。
- 2026-08-21: V2: 可視化層（metrics_image / HeatMap / 状態スイープ）の TensorDict 復旧 実装指示書（出典: [RFD0010](../../../memo/done/0010_image_metrics_v2_10rfd.md)、題・目的）。
- 2026-08-21: 【修正仕様】Ape-X型 空間探索（Vectorized Exploration）機能の追加（出典: [RFD0002](../../../memo/done/0002_vec_exp_10rfd.md)、題・目的）。
- 2026-08-22: Stack 次元のネイティブ対応(Reshape ハックを廃する根本対応)（出典: [RFD0057](../../../memo/done/0057_native_stack_dim_handling_10rfd.md)、題・目的）。
- 2026-08-27: 可塑性メトリクスは実 forward の branch capture と学習経路外の部分 forward の 2 チャネルで測る（出典: [ADR 0031](../../../adr/0031-plasticity-metrics-out-of-band-partial-forward.md)、本文）。
- 2026-08-29: Spectral Normalization は自前 power iteration + u/v named buffer で実装し、σ は毎 forward その場の W で再計算する（出典: [ADR 0032](../../../adr/0032-spectral-norm-self-impl-buffer-semantics.md)、本文）。
- 2026-08-29: 可塑性 weight norm（feature/readout 分割）RFD（出典: [RFD0063](../../../memo/done/0063_plasticity_weight_norm_10rfd.md)、題・目的）。
- 2026-08-29: 可塑性メトリクス（srank / dormant unit 率）RFD（出典: [RFD0062](../../../memo/done/0062_plasticity_metrics_10rfd.md)、題・目的）。
- 2026-08-30: NN Spectral Normalization（weight_norm.mode）RFD（出典: [RFD0065](../../../memo/done/0065_nn_spectral_norm_10rfd.md)、題・目的）。
- 2026-09-02: Policy churn は fixed probe で optimizer 前後を測り、target は sync 後の不一致を測る（出典: [ADR 0033](../../../adr/0033-policy-churn-fixed-probe-and-target-lag.md)、本文）。
- 2026-09-02: Policy churn メトリクス（35_agent_churn）RFD（出典: [RFD0066](../../../memo/done/0066_policy_churn_metrics_10rfd.md)、題・目的）。
- 2026-09-04: Actor Qヒントを3列へ拡張し、近似Actor初期優先度をMunchausenターゲットと同型にする（出典: [ADR 0036](../../../adr/0036-actor-q-hint-three-columns-munchausen.md)、本文）。
- 2026-09-04: MunchausenターゲットはDQN Learnerに局所化し、実空間で計算し、3つのlog-policy modeを持つ（出典: [ADR 0035](../../../adr/0035-munchausen-target-learner-local-real-space.md)、本文）。
- 2026-09-06: Munchausen RL（M-DQN / M-QR / M-IQN）RFD（出典: [RFD0067](../../../memo/done/0067_MunchausenRL_10rfd.md)、題・目的）。
- 2026-09-08: Actor 設定は名前付きカタログ参照とし、Agent インタフェースから RunMode を外す（出典: [ADR 0038](../../../adr/0038-actor-config-catalog-without-runmode.md)、本文）。
- 2026-09-10: replay 当てはまり診断は held-out 分割ではなく既存の抽選履歴フラグで群分けし、評価条件を学習条件から独立させる（出典: [ADR 0039](../../../adr/0039-replay-fit-sampling-history-groups-not-holdout.md)、本文）。
- 2026-09-11: RFD0073: replay 抽選履歴別の TD・損失メトリクス（出典: [RFD0073](../../../memo/done/0073_replay_fit_metrics_10rfd.md)、題・目的）。
- 2026-09-19: RFD0061: Actor 設定カタログと eval スロットごとの方策 / network 指定（出典: [RFD0061](../../../memo/done/0061_eval_slot_policy_override_10rfd.md)、題・目的）。
- 2026-09-20: ReplayBuffer の観測履歴・bootstrap stack 整合性修復 RFD（出典: [RFD0078](../../../memo/done/0078_replay_frame_history_integrity_10rfd.md)、題・目的）。
- 2026-09-24: Agentのdevice指定をauto対応の文字列へ統一した（出典: [RFD0070](../../../memo/done/0070_device_spec_auto_10rfd.md)、確定したD1・D3）。
- 2026-09-27: ActionContextを廃止しRNGと観測加工をActorへ直接移した（出典: [RFD0083](../../../memo/done/0083_action_context_retire_10rfd.md)、題・Context）。
- 2026-10-03: NoisyNet の ε は module に持たせず、呼び出し側所有の NN 実行状態と NN 実行設定キーで駆動する（出典: [ADR 0048](../../../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md)、本文）。
- 2026-10-10: ActorとLearnerの用途別forwardをNN実行設定の参照へ接続した（出典: [RFD0084](../../../memo/done/0084_noisynet_10rfd.md)、§4.2・§6・§7）。
- 2026-10-10: NN Forward の設定解釈を事前準備へ移し、用途別実行オブジェクトが State と Network の寿命を持つ（出典: [ADR 0051](../../../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md)、本文）。
- 2026-10-15: DefaultDQN の Learner が checkpoint を読み込む際に、optimizer の状態を引き継ぎ、設定値は現在の値を適用するようにした（出典: [RFD0091](../../../memo/done/0091_checkpoint_load_optimizer_options_10rfd.md)、§2・§4）。
- 2026-10-15: checkpoint の optimizer を復元した後に、現在の設定から作った option を param group ごとに適用する判断を記録した（出典: [ADR 0053](../../../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md)、決定）。
- 2026-10-16: Learnerの更新内で勾配ノイズを測り購読とEMAをAgent側へ接続した（出典: [RFD0092](../../../memo/done/0092_gradient_noise_scale_10rfd.md)、§4・§6・§9.1）。
- 2026-10-16: forward の持ち帰りは鍵要求型の capture 1 つに統一し、module の診断も同じ器で運ぶ（出典: [ADR 0055](../../../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md)、本文）。
- 2026-10-16: 勾配ノイズスケールは更新自身の batch を小バッチに分けて学習経路の中で測り、BF16 も対象にする（出典: [ADR 0054](../../../adr/0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16.md)、本文）。
- 2026-10-18: Learnerの購読時probeと更新結果へattention entropyを接続した（出典: [RFD0093](../../../memo/done/0093_attention_entropy_diagnostics_10rfd.md)、§4 決定した設計）。
- 2026-10-19: step trace の近似優先度は Agent が出す step 部品が計算し、Actor の設定キーで eval_panel にだけヒントを作らせることはしない（出典: [ADR 0058](../../../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)、本文）。
- 2026-10-21: Actorと共有するヒント計算を使うDQNのstep診断部品を加えた（出典: [RFD0094](../../../memo/done/0094_evalpanel_step_history_10rfd.md)、§4.2 D6）。
- 2026-10-24: 探索の ε・UQE の ε と τ・ラダーの両端を ProfiledValue で時間変化させ、時間の値とラダーの両端を別キーに分け、学習側の 7 値も同じ器にする（出典: [RFD0101](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md)、題・§1・§4）。
- 2026-10-24: 時間で変わる割引率は、schedule のときだけ抽選時に保存済みの step 報酬から収益を作り直す（出典: [ADR 0065](../../../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)、本文）。
- 2026-10-26: Actorのlane別継続状態と判断の旗を、共通の行動・学習設定へ接続した（出典: [RFD0098](../../../memo/0098_temporl_skip_policy_10rfd.md)、§5.5・§5.11）。
- 2026-10-28: skipの合算・遮断に加え、loss別のAdamで更新するsplitと評価セッションの集約値を接続した（出典: [RFD0103](../../../memo/done/0103_temporl_reference_alignment_10rfd.md)、§5.2・§5.4・§5.8）。
- 2026-10-29: Actorごとに毎step判断するsingleと上限を時間で変えるvariableを選べるようにした（出典: [RFD0104](../../../memo/done/0104_temporl_skip_auxiliary_only_10rfd.md)、§5.1・§5.3・§5.9）。
- 2026-10-30: Actor共通の設定へskip側のNN実行設定のモードとキーを足した（出典: [RFD0106](../../../memo/done/0106_temporl_skip_forward_noise_key_10rfd.md)、§5.1・§5.6）。
- 2026-11-11: DQN共用ヘッダに`#pragma once`を足し、`BatchUpdateResult::GetScalar`の本体と補助を.cppへ移して、名前を解析して受ける群の関数と、直接比べるkeyの表に分けた（出典: [RFD0122](../../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§4 D5〜D8）。
- 2026-11-12: 方式に依らない一生の新しさの口（`LifelongNoveltyEstimator`とそのFactory）と、RNDによる実装を足した（出典: [RFD0116](../../../memo/0116_rnd_lifelong_novelty_10rfd.md)、題・§1.2・§5.1）。
- 2026-11-12: RNDの推定器が、checkpointのoptimizerの設定を今の設定で置き換えた通知をINFOにした（出典: [RFD0117](../../../memo/0117_warn_policy_and_unread_config_10rfd.md)、§4 D10）。
- 2026-11-12: TD・QR・IQNの1回の更新の進行を基底の学習器の1か所へ寄せ、Munchausenの評価とtargetの式を共通の関数にした（出典: [RFD0125](../../../memo/0125_dqn_learner_update_skeleton_10rfd.md)、題・§1・§4 D1・D2・D10・D11）。
- 2026-11-13: Actor Qヒントを、価値の分解の有無に依らず常時5列（内的のHeadのQ_saとV_sの2列を足す）に広げると決めた。列の数をモードで変えず、使わない列を0にする判断は保つ（出典: [ADR 0075](../../../adr/0075-actor-q-hint-five-columns-carry-per-head-values.md)、本文、[RFD0121](../../../memo/0121_extrinsic_intrinsic_value_decomposition_10rfd.md)、§5 D12）。
- 2026-11-14: Learnerの診断指標の6つの群が購読の解析・間隔・発火・要求を共通の型で持ち、値の読み出しはpackの形の群も含めて共通にした。probe用の乱数5つは、使う側のLearnerが構築時に、Agentの基のseedから今と同じ名前で作って持つ（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、題・§5 D12・D13）。
- 2026-11-15: DQNの学習器が、外的のHeadと内的のHeadのそれぞれに目標値・損失を作り、優先度をHeadごとの誤差の重み付きの和にした。合わせたQはDQN側の1つの関数が作り、損失ごとに勾配のclipとAdamの状態を分けるモードを持つ。内的のHeadが行動の選択をどれだけ変えたかを見る診断（不一致、譲った外的の価値、外的の差、内的の差）をActorが行動を選ぶたびにlaneごとに作り、Evaluation Traceはlaneごとの値、学習のActorはlaneの平均、評価のActorはsessionの累積を、別の鍵で読む（出典: [RFD0121](../../../memo/0121_extrinsic_intrinsic_value_decomposition_10rfd.md)、§5 D4・D5・D12・D24〜D28・§6・§6.8・§8.4）。
- 2026-11-15: 一生の新しさの推定器に、重みも統計も変えずに採点する口（`EstimateWithoutUpdate`）、写しを作る口（`Clone`）、laneごとの診断値の鍵の一覧を足し、DQNのstep診断の部品が、列の鍵に内的報酬の鍵があるときだけ採点するようにした（出典: [RFD0127](../../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md)、§4 D1・D3・D4・§6.1）。
- 2026-11-16: RNDの採点の統計の区間を軽くするため、正規化と統計の集計を計算deviceで行うようにし、そのあとで2つの統計方式（whiten・variance_feedback）を型で分けた（出典: [RFD0130](../../../memo/0130_rnd_estimate_statistics_perf_10rfd.md)、題・§1.2・§5.1）。
- 2026-11-17: Actorがnn_traceを要求された呼び出しだけcaptureで取り出し、SelectAction系がcallbackの代わりにcaptureを受けるようにした（出典: [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md)、§4 D7・D9・D11・D12）。
- 2026-11-18: Agentは基盤の外の系統ごとのライブラリに置き、Headを書くための部品を基盤が公開する（出典: [ADR 0079](../../../adr/0079-agents-live-in-per-family-libraries-outside-core-and-head-authoring-parts-are-public.md)、本文）。
- 2026-11-19: RNDのwhitenで、warm-up中の誤差を誤差の統計から外す既定と、誤差の統計の取り方（累積・直近重み。既定は累積）、平均を引くかの選択を足した（出典: [RFD0123](../../../memo/0123_rnd_intrinsic_reward_scale_10rfd.md)、題・§1・§4 D2・D4・D7、冒頭の2026-11-19の改訂）。
- 2026-11-19: Agentを基盤の外の系統ごとのライブラリ（DQN・MuZero・ImageCls）へ出し、登録を各ライブラリの初期化関数に移した。基盤の外で実装する拡張点と公開ヘッダの規約（RFD0169）も統合して入れた（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§1・§4.1 D5・D7・D10、[RFD0169](../../../memo/0169_outside_core_extension_layout_10rfd.md)、冒頭の2026-11-18の改訂の行）。
- 2026-11-19: RNDのwhitenに、誤差を誤差の統計で割らない倍率の式を足し、Burdaらの内的報酬の形にできるようにする仕様を、暫定で起票した（出典: [RFD0191](../../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md)、冒頭の起票の行）。
- 2026-11-20: DQNのτの生成のエラー文・引数の検証関数・テストの名前を、実在しない部品名TauGeneratorから`GenerateTaus`にそろえた（出典: [RFD0190](../../../memo/0190_dqn_design_doc_alignment_10rfd.md)、§4 D19）。
- 2026-11-20: AgentRepositoryの登録が登録済みのclass IDを止め、初期化関数は何度呼んでも登録を1回だけ行うようにした（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5 D10・D11・§6.5）。
- 2026-11-21: Agentのfactoryの未知のclass IDのエラーと、未定義のActorのカタログ項目のエラーが、設定キーと項目の完全キーを示すようにした（出典: [RFD0194](../../../memo/0194_config_key_full_names_in_messages_10rfd.md)、§5 D6・§8.3）。
- 2026-11-21: RNDのHeadの初期化の設定も、NNの初期化の設定の1つの読み取りの口で読むようにした（出典: [RFD0193](../../../memo/0193_variance_scaling_weight_init_10rfd.md)、§5.2・§6.1 D5）。
- 2026-11-21: RNDの設定キーを、段階（統計・誤差・predictor）ごとのスコープとモード別の名前空間に組み替え、warm-upを段階ごとの待ちに分け、clipを誤差のモードごとの真偽のキーで持つ仕様を、グリルで決めた。誤差で割らない式（`rnd.error_norm.mode = none`）、一生の新しさの推定器の口の2つの操作（内的報酬を呼び出し側が0にするかを返す、Networkの構造図を返す）、lane×時間のtensorをためる汎用の置き場の部品も足す（出典: [RFD0191](../../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md)、§6.1 D2〜D5・D7〜D9・§8.4）。
- 2026-11-21: RNDの設定は、段階ごとのスコープとモード別の名前空間で持ち、被るキーは割り切る（出典: [ADR 0088](../../../adr/0088-rnd-config-is-scoped-by-stage-with-per-mode-namespaces.md)、本文）。
- 2026-11-22: 強制行動は`MakeAction`の入力とし、Actorの出力を実行された行動にする（出典: [ADR 0083](../../../adr/0083-forced-actions-are-make-action-input-and-actor-output-is-executed-action.md)、本文）。
- 2026-11-23: DQNのActorが、行動選択のPerceptionをForwarderの並びから作り、方策の選択・skipの適用のあとに強制行動を反映してからヒントを計算する形にした。TempoRLのskipが有効なActorは、強制したlaneの継続状態を捨て、そのstepを長さ1の判断として記録する。LearnerはSampleの観測（obs・next_obs・skipの窓の終点）のすべてに同じ並びのForwarderを当てる（出典: [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md)、§6.2 D11〜D13・§6.4 D31・§7.2）。
- 2026-11-23: テストのタグで使う文書の呼び名を RFD に改め、番号を 0 埋めの 4 桁にそろえた（出典: [RFD0195](../../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md)、§6 D12・§9.6）
- 2026-11-25: DQNのActorが、実行器の代わりに行動選択用のForwarderを持ち、episodeの開始の印を入力の鍵（`episode_start`）で渡すようにした。Learnerは、ReplayBufferを作ったあとに、ReplayBufferが答えた束のspecを渡してSample用のForwarderを作る（出典: [RFD0200](../../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.1 D1・§5.2 D14・§6.5）。
- 2026-11-25: 行ごとのlogitsから作る離散行動の分布`CategoricalDistribution`（抽選、確率が最大の行動、log確率、entropy、全行動の確率とlog確率。機能グループ`distribution`）を足した（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§6.1・§7.1）。
- 2026-11-28: DQN 系の共有部品で、Learner の設定検証と学習進行 State の責務をそろえ、未成立の UQE 診断を修正した（出典: [RFD0207](../../../memo/0207_dqn_small_fixes_and_refactors_10rfd.md)、§4・§5・§6）

## 選ばなかった形

- ImageClsのTrain Actorにも遅延snapshotを適用する形 — 不採用: episodeを持たず、shared networkの即時反映を維持する（出典: [RFD0036](../../../memo/done/0036_train_actor_periodic_snapshot_10rfd.md)、対象範囲）。
- ObservationPipelineクラスと専用ファイル・テストの追加 — 範囲外: 固定3段の観測加工はActor本体で満たせ、段の要件が揃う前のクラス化は作り直しになる。再開条件: 段の要件が揃う後続RFDで導入する。（出典: [RFD0083](../../../memo/done/0083_action_context_retire_10rfd.md)、複雑さ監査）。→ RFD0161 で、クラス1つではなく、Forwarderの並びとして採用（出典: [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md)、§6.4 D25・§6.6）。
- 独立した公開NoisyLinearブロック型 — 不採用: 既存LinearとHead最終射影へ対応する形を採る（出典: [RFD0084](../../../memo/done/0084_noisynet_10rfd.md)、§4.1）。
- AMPの実行設定への集約とノイズ込み重みのSN — 範囲外: 精度は用途ごとに異なり、ノイズ込みSNの推定・保持・比較契約も別の暫定RFDで裁定する（出典: [RFD0084](../../../memo/done/0084_noisynet_10rfd.md)、§8.1・§8.2・§14.2）。
- Noisyの有無フラグ・機能登録・実行条件の汎用比較と独立実行基盤 — 不採用: 固有の解釈と検証を具象の計算部品へ閉じる（出典: [RFD0084](../../../memo/done/0084_noisynet_10rfd.md)、§4.2・§14.2）。
- εを引き直す総ノイズ測定と曲率を含む臨界batch測定 — 先送り: 追加forward・専用乱数やヘッセ行列との積が要る。再開条件: 分岐の結果が条件付き値の読みと食い違いNoisy probeのQ差が大きいとき、または単純版とBの効き目が食い違ったとき（出典: [RFD0092](../../../memo/done/0092_gradient_noise_scale_10rfd.md)、§8）。
- PPO/SAC・複数損失・別optimizer向けの拡張口 — 先送り: 単一損失・AdamWの現用途で完結させる。再開条件: 対象Learner・観測したい損失とoptimizerが具体化したとき（出典: [RFD0092](../../../memo/done/0092_gradient_noise_scale_10rfd.md)、§8 5・§10.2）。
- 購読間隔ごとの別EMA — 不採用: 状態と系列が増える。独立した時間幅はrawの利用側での再平滑化に任せる（出典: [RFD0092](../../../memo/done/0092_gradient_noise_scale_10rfd.md)、§10.2）。
- head単位統計とattention entropy最小値 — 先送り: 層平均で足りる間は公開値と保持データを増やさない。再開条件: 層平均が下がったRunで内訳が必要になったとき。最小値は6層構成で読みにくくなったとき（出典: [RFD0093](../../../memo/done/0093_attention_entropy_diagnostics_10rfd.md)、§5・§7）。
- TQLのattention entropy正則化 — 先送り: anetでは本指標で集中の進行をまだ観測していない。再開条件: 層平均の集中と成績の頭打ち・下降が時期として重なったとき。因果は別途判断する（出典: [RFD0093](../../../memo/done/0093_attention_entropy_diagnostics_10rfd.md)、§7 1）。
- 学習側の厳密な優先度の列 — 不採用: EvalPanelには単一networkのforwardしかなくreplayも無いため学習側のonline/targetによる値は得られない（出典: [RFD0094](../../../memo/done/0094_evalpanel_step_history_10rfd.md)、§6）。
- moduleのbufferにεを置きreset_noiseで引き直す形 — 不採用: shared networkの同時forwardで競合し、snapshotとhard updateで別用途のεが混入する（出典: [ADR 0048](../../../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md)、Considered Options）。
- ReplayBufferから別に一様抽出する勾配ノイズprobe — 不採用: 実更新と抽選分布・IS重みが違い、forwardも増える（出典: [ADR 0054](../../../adr/0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16.md)、Considered Options）。
- BatchNormの専用検出・禁止 — 不採用: 独立サンプルとして解釈できる条件を文書化し、構成と解釈は利用側に任せる（出典: [RFD0092](../../../memo/done/0092_gradient_noise_scale_10rfd.md)、§10.2）。
- 未成立理由ごとの追加状態指標 — 不採用: 記録から理由を一意に復元できない場合も含め、追加公開契約を増やさない（出典: [RFD0092](../../../memo/done/0092_gradient_noise_scale_10rfd.md)、§10.2）。
- attention entropy専用のfeatureキー — 先送り: 今の構成ではTransformerがfeature閉包内にある。再開条件: Transformerがfeature閉包の外へ置かれたとき（出典: [RFD0093](../../../memo/done/0093_attention_entropy_diagnostics_10rfd.md)、§5）。
- 構築時dummy forwardで層数を数える追加API — 不採用: 黙ったNaNは最初の測定のfail-fastで防げ、APIを1つ増やす必要がない（出典: [RFD0093](../../../memo/done/0093_attention_entropy_diagnostics_10rfd.md)、§5）。
- Actor設定キーでeval_panelだけにhintを生成させる形 — 不採用: 設定が増え、autoの語が今の条件を表すものになり、n-step窓の所有がAgent外へ出る（出典: [ADR 0058](../../../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)、本文）。
- NetworkModel が特徴空間・行動価値・スキップ価値の 3 つの Network を持つ形 — 不採用: 切れ目は DAG の branch で表せ、checkpoint の形と Network 1 つ前提の経路へ波及する。補助 Network は primary の公開出力を入力に取り、その target は bootstrap に使わず target 側の Actor の行動にだけ使う（出典: [RFD0098 TempoRL](../../../memo/0098_temporl_skip_policy_10rfd.md)、§4 D3、[ADR 0062](../../../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md)）。
- 強制行動を Actor へ通知する事後 hook — 先送り: 決定点への入力（0161_observation_pipeline の単位 A）で合わせる。再開条件: 単位 A が入ったら、強制された lane のスキップ状態を捨てる（出典: [RFD0098 TempoRL](../../../memo/0098_temporl_skip_policy_10rfd.md)、§4 D10・§9）。→ RFD0161 で、決定点への入力の形を採用し、強制した lane のスキップの継続状態を捨てるようにした（出典: [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md)、§6.2 D7・D13）。
- Thompson sampling の τ の区間を `thompson.tau_low/high` で設定する形 — 不採用: 非 spatial の Thompson は τ を U[0,1]（IQN は全範囲）から引き、区間は存在しない。spatial は `spatial.uqe.tau.*` のラダーを使う（出典: [RFD0101](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md)、§2.1・§4.2 D9）。
- log ラダーの 0 の端を fail-fast にして 0 に届くラダーを新しい並べ方 `exponential` で書く形、下限をキー（`spatial.log_floor`）にする形、1e-4 への clamp を残す形 — 不採用: 下限 1e-4 で log に並べた列を書いた両端へ線形に伸縮する規則 1 つにし（両端が 1e-4 以上なら恒等）、モードやキーを増やさず、0 と書いて 1e-4 で動く形を残さず、端が 0 へ下がる途中も値を連続にする（出典: [RFD0101](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md)、§4.2 D8）。
- MuZero の温度 `temp_start/end/decay_steps` の ProfiledValue 化 — 範囲外: 試作 Agent。再開条件: 本線に戻すときに同じ器で足す（出典: [RFD0101](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md)、§3 非ゴール・§7）。
- NoisyNet の雑音の倍率を時間で変える — 先送り: σ は学習するパラメータで、外から掛ける倍率は forward の雑音の作り方に新しい入力が要る。再開条件: ε の形で差が説明できないとき（出典: [RFD0101](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md)、§7）。
- step の軸を learn_step やエピソード数にする — 先送り: 探索と β はすべて exp_step で動いている。再開条件: 別の軸が要る例が出たとき（出典: [RFD0101](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md)、§7）。
- `BatchUpdateResult::GetScalar`のkeyの全部を1枚の表にする形 — 不採用: 名前を解析して受ける群は名前から番号を読むので表に載らない（出典: [RFD0122](../../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§4 D6・§7）。
- `BatchUpdateResult::GetScalar`の直接比べるkeyも群ごとの関数に分け、中をifの連鎖のままにする形 — 不採用: keyの重複や届かない分岐が機械では見つからないまま残る（出典: [RFD0122](../../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§4 D6・§7）。
- RNDの誤差の統計を直近重み（バイアス補正EMA）を既定にする形 — 不採用: 時定数というenvごとの調整値と、覚えきったenvで微小な差を拾う帰結が既定に入る。KFMのRunで累積と直近重みを比べてから決め直すとしていたが、2026-11-19に、KFM 10Mの同じseedの2本どうしでε 0.01の評価が累積を下回ったのを見て、既定を累積のままにすると決めた（出典: [RFD0123](../../../memo/0123_rnd_intrinsic_reward_scale_10rfd.md)、冒頭の2026-11-19の改訂・§4 D2・§4.1）。
- RNDのwarm-upの中身を、predictorを学習させるかと誤差を統計に入れるかの独立したスイッチ2つで選ぶ形 — 不採用: 要求の無い組が2つ増え、うち1つは膨らみが再発する（出典: [RFD0123](../../../memo/0123_rnd_intrinsic_reward_scale_10rfd.md)、§4 D4・§4.1）。 → RFD0191 で採用（段階ごとの待ちの件数として分ける。出典: [RFD0191](../../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md)、§6.1 D8）
- RNDの前半は観測の統計だけ、後半はpredictorも学習させる2ステージのwarm-up — 先送り: 効果が未測定で設定キーが増える。再開条件: 新しい既定のRunの8M〜10Mの誤差の標準偏差が0.05を超えて残るか、2M〜3Mと9M〜10Mの比が1.5を超えたとき。KFMでは比が僅差で1.5を超えたが、warm-upを1,000,000件にしても評価が変わらず、2026-11-15に開き直さないと決めた（出典: [RFD0123](../../../memo/0123_rnd_intrinsic_reward_scale_10rfd.md)、冒頭の2026-11-15の改訂・§4.1）。 → RFD0191 で採用（待ちの件数の組として書ける形にする。効果は確かめない。出典: [RFD0191](../../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md)、§7.2）
- 平均を引かない形を `multiplier_mode` の3つ目の値にする形 — 不採用: clipの有無を表すキーに別の軸が混ざり、平均を引かずclipもしない形が書けない（出典: [RFD0123](../../../memo/0123_rnd_intrinsic_reward_scale_10rfd.md)、§4 D7・§4.1）。
- RNDのcheckpointの書式の識別子を一律に上げ、誤差の統計の取り方を中に書く形 — 不採用: 識別子は1つで済むが、今あるRNDのcheckpointが新しい実行体で読めなくなる。累積の書式と中身は変えず、直近重みだけ別の識別子にする（出典: [RFD0123](../../../memo/0123_rnd_intrinsic_reward_scale_10rfd.md)、§4 D11・§4.1）。
- 誤差を統計に入れ始める件数を整数1つ（0が今の中身）で持ち、warm-upの中身のモードを作らない形 — 不採用: 値に動作の切り替えを兼ねさせず、待たせるかどうかはモードのキー、件数は値のキーで表す（出典: [RFD0123](../../../memo/0123_rnd_intrinsic_reward_scale_10rfd.md)、§4 D4・§4.1）。
- 直近重みのモードを作らず累積だけにする形 — 不採用: 累積の書式を変えなければcheckpointの不利益が無く、直近重みを比べる手段を残せる（出典: [RFD0123](../../../memo/0123_rnd_intrinsic_reward_scale_10rfd.md)、§4 D2・D11・§4.1）。
- Burdaらの形（内的報酬の割引収益の標準偏差で割る） — 範囲外: laneごとの割引収益の保持と、外的・内的の価値を分ける話と組になり、RFD0121・0173に残す（出典: [RFD0123](../../../memo/0123_rnd_intrinsic_reward_scale_10rfd.md)、§1 非ゴール）。
- RNDの誤差で割らない式を、`statistics_mode` の3つ目の値で選ぶ形、真偽のキー（`error_norm.enabled`）で選ぶ形 — 不採用: 前者は観測の統計と誤差の式が1つのキーに混ざり、checkpointが値を持つので再開時に切り替えられない。後者は3つ目の式を足すときにキーを作り直すことになる（出典: [RFD0191](../../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md)、§6.1 D4・§6.4）。
- 誤差で割らない式のとき、使われない設定キーが既定と違えばWARN（またはINFO）を出す形 — 不採用: モードを変えたいだけなのに、関係ない設定を同時に直させることになる。読んで検証するだけにする（出典: [RFD0191](../../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md)、§6.2 D12・§6.4）。
- フレーム積層の履歴とReplayの観測の置き場を、lane×時間のtensorの置き場の部品へ載せ替える形 — 範囲外: RNDを使わないRunを含む全Runの経路に触る別の関心。再開条件: ためる仕組みの統合整理の課題に取り組むとき（出典: [RFD0191](../../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md)、§1.4・§6.1 D2・§8.5）。
- RNDのwarm-upを `warmup_mode` で束ねたままにする形、待ちの件数を `rnd` 直下に1つ持つ形 — 不採用: 段階をまたぐ束が残り、モードを切り替えたときに前のモード用に書いた件数が残る（出典: [RFD0191](../../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md)、§6.1 D8・§6.4）。
- 倍率のclipを、誤差の式に依らない独立の段階にする形 — 不採用: 上限が切る対象の単位が、標準化した値と生の誤差で変わる（出典: [RFD0191](../../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md)、§6.1 D9・§6.4）。
- RNDの新しく足すキーだけを段階ごとの形にし、今あるキーは今の名前のまま残す形 — 不採用: 2つの形が並び、設定とbatを2回直すことになる（出典: [RFD0191](../../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md)、§6.1 D7・§6.4）。
- 学習器の更新の末尾（PERの重み付けから結果の組み立てまで）だけを共通の関数にする形 — 不採用: 正規化から要素誤差までの順がTD・QR・IQNの3か所に残る（出典: [RFD0125](../../../memo/0125_dqn_learner_update_skeleton_10rfd.md)、§4 D2・§7）。
- 学習器の更新の進行を、関数を受け取る補助にして分布ごとの違いを渡す形 — 不採用: targetと要素誤差が仮想関数の上書きの型で、型が2つになる（出典: [RFD0125](../../../memo/0125_dqn_learner_update_skeleton_10rfd.md)、§4 D3・§7）。
- targetの式の関数がh空間の値も受け取り、旗で逆変換を切り替える形 — 不採用: 旗の渡し違いで逆変換が2回掛かる（出典: [RFD0125](../../../memo/0125_dqn_learner_update_skeleton_10rfd.md)、§4 D11・§7）。
- PERが無効のときの重みを、1を掛ける形にそろえる — 不採用: 値は同じで、Tensorの確保と掛け算が増える（出典: [RFD0125](../../../memo/0125_dqn_learner_update_skeleton_10rfd.md)、§4 D6・§7）。
- 学習器の更新のDEBUGログ（`element_loss=`・`weights=`）を消す形 — 不採用: 選んで残してあるログで、通常のRunに影響しない（出典: [RFD0125](../../../memo/0125_dqn_learner_update_skeleton_10rfd.md)、§4 D8・§7）。
- TD・QR・IQNの学習器のクラスを1つにまとめる形 — 範囲外: 写しを無くすのに要らず、構築の口と設定からクラスを選ぶ箇所が変わる（出典: [RFD0125](../../../memo/0125_dqn_learner_update_skeleton_10rfd.md)、§3.2・§7）。
- 価値の分解のためのHeadごとの目標の口を、学習器の更新を寄せるときに先に足す形 — 範囲外: RFD0121で作る。再開条件: RFD0121のグリルで口の形が決まったとき（出典: [RFD0125](../../../memo/0125_dqn_learner_update_skeleton_10rfd.md)、§3.4・§7）。
- replay fitの診断（`EvaluateReplayFit`）が別に書いている進行の前半を、学習器の更新の進行と1つにする形 — 範囲外: forwardの仕方・τの置き方・指標の有無が学習と違い、current側の計算の口をそのまま使えない。[課題4efa57](../../../management/issues/issue-20261112-4efa57.md)に記録した（出典: [RFD0125](../../../memo/0125_dqn_learner_update_skeleton_10rfd.md)、§2.4・§3.2・§7）。
- TD・QR・IQNの学習器に、基底の`UpdateFromSamples`を呼ぶだけの上書きを書く形 — 不採用: 仕事をしない層になる（出典: [RFD0125](../../../memo/0125_dqn_learner_update_skeleton_10rfd.md)、§4 D4・§7）。
- 入力の行数の検査を学習器の更新の進行に置かず、直前の`ValidateDeviceSamples`に任せる形 — 不採用: テストが`UpdateFromSamples`を直接呼ぶときに掛からなくなり、QRとIQNの今の動きが変わる（出典: [RFD0125](../../../memo/0125_dqn_learner_update_skeleton_10rfd.md)、§4 D7・§7）。
- Actorが要求の数を持ち、nn_traceを読む側がActorに登録する形 — 不採用: MakeActionの出力が引数に現れない値で決まり、Actorの基底に登録の口が増える（出典: [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md)、§4.1；[ADR 0072](../../../adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md)、Considered Options）。
- Actorがforwardを包み、SelectActionから引数を無くす形 — 不採用: 包むための新しい型が要る（出典: [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md)、§4.1）。
- nn_traceを読む側が、通知に載った観測からforwardをやり直して途中の出力を得る形 — 不採用: 行動選択に使ったforwardそのものの値を出す方針で、やり直すとNoisyNetの雑音やIQNのτの抽選で値が食い違う（出典: [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md)、§4.1；[RFD0006](../../../memo/done/0006_conv2d_visual_trace_10rfd.md)、§2）。
- nn_traceを作れないActorが要求を受けたときに、何も知らせずに黙って作らない形 — 不採用: Fail-Fast原則の表が「実装の欠けによる機能の一部の省略」をRunごとに1回のWARNと定めており、paneが空の理由が利用者に分からない（出典: [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md)、§4 D12）。
- step診断の内的報酬の鍵を、読まれたときに計算する形 — 不採用: 値を返すだけに見える読み出しがNetworkを通し、読む時機で計算の時点がずれる（出典: [RFD0127](../../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md)、§4 D3・§4.1）。
- RNDが有効なら、step診断が常に内的報酬を採点する形 — 不採用: 評価sessionの記録で`q_sa`だけを宣言したとき、全lane・全stepで使われない採点が走る（出典: [RFD0127](../../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md)、§4.1）。
- RNDのNetworkに、forwardで乱数を引く層（dropout・droppath）や内部の統計を書き換える層（BatchNorm）がある構成を検出してWARNする形 — 不採用: 学習の採点も同じforwardで今も検出しておらず、出荷設定には無い（出典: [RFD0127](../../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md)、§4 D11・§4.1）。
- 群の共通の型を作らず、診断の指標を名前で指す形にだけ直す形 — 不採用: 群を足すたびに12か所前後を直す形が残る。共通の型に載らない群だけはこの形にし、その群でも並びと参照が食い違えばビルドが止まる形にする（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、§5 D12・§5.1・§8 U1）。
- 診断の指標に名前を付けるだけで、指標の並びとの食い違いをビルドで止めない形 — 不採用: 並びを変えたときに、名前が黙って別の指標を指す形が残る（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、§5 D12・D19・§5.1）。
- 診断の共通の読み出しを、間隔のgateを持つ6群だけにする形 — 不採用: RFD0121が足す、毎更新に出すメトリクスが載らない（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、§5 D12・§5.1）。
- probe用の乱数の渡し口を、Learnerのコンストラクタと購読の設定の2つのまま残す形 — 不採用: 課題24d885の事象が残る（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、§5 D13・§5.1）。
- probe用の乱数を、購読されたときにAgentが作って渡す形 — 不採用: AgentとLearnerが同じ購読の判断を別々に書き、食い違うとRunの途中で止まるか黙って購読を捨てる（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、§2.5・§5 D13・§5.1）。
- probe用の乱数を、Agentが構築時に作ってLearnerへ渡す形 — 不採用: 使うのはLearnerだけで、使う側が作って持つほうが単純（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、§5 D13・§5.1）。
- probe用の乱数を、Learnerが購読されたときに群の型の中で作る形 — 不採用: 購読を設定し直すと乱数の列が頭に戻る置き場になる（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、§5 D13・§5.1）。
- probe用の乱数を、Learner自身のseedから作る形 — 不採用: 同じseedでもprobeの診断の値が変わり、診断を作り直す段の前後比較を一致で確かめられない。切り替えるなら別の段で、変わるのがprobeのtagだけだと確かめる（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、§5 D13・§5.1）。
- 診断の群の共通の型をAgentの共通基盤に置き、ImageClsAgentも載せる形 — 範囲外: リファクタリング5件の外で、前後比較の組がImageClsを通らない。再開条件: ImageClsに診断の群を足すとき（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、§1.3・§5 D22）。
- Learnerの購読を「開始前の宣言」と「呼び出しごとの要求」に分ける形 — 範囲外: 購読の口を変える別の関心。再開条件: 購読の口を変える要求が出たとき（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、§1.3・§5.1）。
- RNDの統計と正規化をCPUのまま、batchの集計の式だけを替える形 — 不採用: 見積もりは1 intakeあたり7〜8 msで、計算deviceへ寄せる試作の6.482 msより遅く、計算deviceへ送る量も今のままになる（出典: [RFD0130](../../../memo/0130_rnd_estimate_statistics_perf_10rfd.md)、§5.1の退けた案）。
- RNDの補助学習を別のスレッドから出す並列処理化 — 先送り: NGUの対応のあとに残らない見込みで、得が1 intakeの3〜4%にとどまる。再開条件: 性能の直しのあとも、RNDが1 intakeの1割以上を占めるとき（出典: [RFD0130](../../../memo/0130_rnd_estimate_statistics_perf_10rfd.md)、§5.1 D9）。
- RNDの正規化をNNの中の前処理にする形 — 不採用: 統計がNetworkの中身になり、checkpointの書式が変わる（出典: [RFD0130](../../../memo/0130_rnd_estimate_statistics_perf_10rfd.md)、§5.1 D3・退けた案）。
- RNDの統計方式の整理と性能の直しを同時に入れる形 — 不採用: 値を変えない直しを、同seedの結果の一致で確かめられなくなる（出典: [RFD0130](../../../memo/0130_rnd_estimate_statistics_perf_10rfd.md)、§5.1の退けた案）。
- RNDのtargetの出力の統計と診断値の読み戻しを1回にまとめる形、ボーナスの診断値の読み出しをまとめる形、CPUのスレッド数を増やす形 — 不採用: 置き場の変更だけでtargetの出力の統計が0.772 msから0.151 msになり、ボーナスの読み出しは0.033 ms、スレッド数は実時間が1.874倍と13.486倍に延びた（出典: [RFD0130](../../../memo/0130_rnd_estimate_statistics_perf_10rfd.md)、§5.1 D8）。
- Agentのライブラリを具象ごとの5本にする形 — 不採用: 変更の多いDQNの内部ヘッダが公開面になり、足される強制は今0行の依存に対するものである（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.3）。
- Agent全体を1本のライブラリにする形 — 不採用: MuZeroやImageClsがDQNの部品を読むことを、機械が止めない（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.3）。
- 系統のライブラリが基盤のsrcをincludeパスに足して内部ヘッダを読む形 — 不採用: Agentは基盤の公開ヘッダだけを読む、というゴールを満たさない（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.3）。
- テストを境界の対象にしない形、系統のテストが基盤のsrcを直接読む形 — 不採用: テストが基盤の公開面を素通りし、同じリポジトリにあることに頼る（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.3；[ADR 0080](../../../adr/0080-tests-read-own-library-internals-and-reach-other-libraries-through-public-headers-and-test-support.md)、Considered Options）。
- Envのテストと系統のテストが互いのライブラリを使ってよいとする形 — 不採用: 置き場が主題しだいで割れ、相手が壊れると使う側のテスト実行体が作れない（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.3）。
- 初期化関数は1回しか呼べない形にして、テストを直す形 — 不採用: `InitRL`が中で`InitNN`を呼ぶので`InitRL`を2回呼ぶだけで止まり、`InitNN`を公開する変更とテスト46か所の書き換えが要る（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5.1）。
- 同じ型のfactoryなら、重複登録を黙って通す形 — 不採用: 型の照会が要り、同じ型で中身の違う登録を見分けられない（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5.1）。
- 強制行動を反映したあとのヒントを、Actorの外で作り直す形 — 不採用（却下）: 行動から作る値が増えるたびに、差し替えの側も増える（出典: [ADR 0083](../../../adr/0083-forced-actions-are-make-action-input-and-actor-output-is-executed-action.md)、Considered Options）。
- RNDの観測の加工をForwarderの並びに載せる形、RNDの正規化の判定を値の尺度の属性に合わせる形 — 範囲外: RNDはbyteの画像をfloatに直して自前の統計で正規化していて、合わせるとAtariの動きが変わる。再開条件: RFD0191の実装が済み、RNDの入力にQの側の並びと同じ部品を当てる構成が要るようになったとき（出典: [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md)、§2 のNG3・§6.4 D33・§15）。
- 行動選択用のForwarderの出力のspecを、Actorを作るたびに比べる形 — 不採用: Runの途中で初めて止まりうる。Agentの構築のうちに比べる（出典: [RFD0200](../../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.6、§5.4 D21）。
- 部品が、使われる所ごとの関数と状態を自分で持つ形 — 不採用（却下）: Actorごとに部品を複製することになり、履歴の持ち主が定義と混ざる（出典: [ADR 0084](../../../adr/0084-forwarder-is-common-interface-and-chain-serves-requests-named-by-input-and-product.md)、Considered Options）。
- Actor のヒント計算に Learner の設定全体を渡す — 不採用: 行動選択に不要な学習方針まで Actor の契約に入り、設定の責務が広がる（出典: [ADR 0036](../../../adr/0036-actor-q-hint-three-columns-munchausen.md)、Considered Options）。
- 新規遷移の初期優先度だけを求めるためにネットワークを追加で forward する形 — 不採用。単一マシンでは Learner と同じ GPU 資源を消費し、既存の行動推論から Q 値を利用できる（出典: [RFD0035](../../../memo/done/0035_approx_actor_priority_per_10rfd.md)、問題）。
- DQN 設計文書の監査で、τ 生成を表す古い名前に合わせて設定やアルゴリズムまで変える案は範囲外とした。文書とコードに残った名前を実際の生成関数にそろえれば、指摘された食い違いは解消する（出典: [RFD0190](../../../memo/0190_dqn_design_doc_alignment_10rfd.md)、§2「ゴールと非ゴール」・§4 の D19）。
- テストのタグや検証資材だけ旧い呼び名を残す形 — 不採用: 現行のテストと参照手順に旧名が残り、同じ変更の参照が分かれる（出典: [RFD0195](../../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md)、§6 D9・D12）
- 行動の値を取り出すすべての経路に NaN の検査を広げる形 — 不採用: NaN を含みうる診断まで Debug 実行で停止する（出典: [RFD0207](../../../memo/0207_dqn_small_fixes_and_refactors_10rfd.md)、§4 の選んだ行動の値の取り出し）
- checkpoint 読み込み後に学習率など一部の値だけを個別に上書きする形 — 不採用: optimizer に設定を追加したとき、読み込み後に戻す値の一覧だけ追随が漏れる（出典: [RFD0091](../../../memo/done/0091_checkpoint_load_optimizer_options_10rfd.md)、§6）。
- 行動の分布を`rl`か`nn_head`の機能グループに置く形 — 不採用: `rl`は契約のヘッダで計算の部品を置かず、`nn_head`のHeadは分布の入力を出すだけである（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§7.8）。
- 行動の分布の共通の口（抽象の基底）を今回作る形 — 先送り: 実装が1つしか無い。再開条件: 連続行動を足すとき（entropyを持たない分布（tanhで縮めたGaussian）を含めて口を決める）（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§12 U14）。
- 連続行動の分布（対角のGaussian、tanhで縮めたGaussianとlog確率の補正）を同時に足す形 — 範囲外（非ゴール）: 行動の受け渡し（Env基底・経験・Viewer）の横断の改修で、別の関心である。再開条件: 連続行動のRFD（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§3.2 NG3・§7.8・§11）。
- DQN系のActorのNetworkの写しと同期（`CopySourceNetwork`・`UpdateSnapshot`・`Sync`。ImageClsにも同じ形）をcoreへ上げる形 — 先送り: 今回はDQN系を変えない（開発者の了承が無い）。再開条件: 写しを持つTrainのActorをPPO系で使うとき（rolloutごとの同期が要る）（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§10 C2・§3.2 NG1）。
- DQN系のoptimizerの組み立てと学習率の書き戻し（`BuildOptimizerOptions`・`SetupOptimizer`。ImageClsにも同じ形）をcoreへ上げる形 — 先送り: 今回はDQN系を変えない。移すとgroupの分け方（Noisyのσ、価値の分解）を呼ぶ側が渡す形になる。再開条件: parameterのgroupを分けた学習率をPPO系で使うとき（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§10 C3）。
- 購読ごとの診断の間引き（`DiagnosticDemandTable`ほか）をcoreへ上げる形 — 先送り: 今回はDQN系を変えない。再開条件: 鍵ごとの間隔で計算を間引く診断（可塑性など）をPPO系に足すとき（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§10 C4）。
- Perceptionの仕上げとspecの照合（`CompletePerception`・`RequirePerceptionSpecs`）をcoreへ上げる形 — 先送り: 今回はDQN系を変えず、PPO系はForwarderの並びを結線しない。再開条件: PPO系にForwarderの並びを結線するとき（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§10 C6・§12 U11）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../../reports/stats/trend.md)の「テスト充足」を見る。

- 明示的なActor::Syncは呼出時の強制同期として維持する。shared既定ではaction forward回数と保存形式を保つ（出典: [RFD0036](../../../memo/done/0036_train_actor_periodic_snapshot_10rfd.md)、目的）。
- Actorを追加しても既存Actorの行動列を変えず、同一checkpoint・seedの各スロットの結果を確認する（出典: [RFD0061](../../../memo/done/0061_eval_slot_policy_override_10rfd.md)、7.4・8）。
- 観測のstack→device転送→正規化の並び、乱数の派生・消費順、aux集合を変えない。同seed・config・exp予算のDefaultDQN/LunarLander/stack4でmetrics checksumを比べ、checkpointのraw checksumはゲートにしない（出典: [RFD0083](../../../memo/done/0083_action_context_retire_10rfd.md)、Context・確定した設計判断2〜4・11）。auxのうち、観測の2つ（重ねたあとと正規化のあと）と全行動のMunchausenの項はRFD0161で消え、観測の加工はForwarderの並びへ移った。今の確かめ方は、4環境・同seed・決定的なバックエンドのscalarとtraceの一致である（出典: [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md)、§6.2 D12・§6.4 D42・§9）。
- 既存のHead/SN FP32保護と用途別AMP精度を維持する。Noisy無効の既存チェーンは決定論backendでmetrics checksumを比較し、sample時の乱数列の割当変更を無効側へ持ち込まない。比較の相手は実装前に保存した基準の実行体・依存DLL・出力とする（RFD0060は基準を取り損ねた）（出典: [RFD0084](../../../memo/done/0084_noisynet_10rfd.md)、§8.2・§12・§13）。
- 診断ON/OFFで同seed・決定的モードの学習系列を比較し、学習の.grad・optimizer state・RNGを変えない。費用は反復A/Bで実測しprofiler合計でend-to-end増加率を代用しない（出典: [RFD0092](../../../memo/done/0092_gradient_noise_scale_10rfd.md)、§7.1・§7.2）。
- captureへ移す既存branch出力の値とdetachを保ち、captureなしforwardは不変とする。同seed・決定論設定の診断ON/OFFで学習由来metrics checksumとRNG・buffer等への非干渉を確認する（出典: [RFD0093](../../../memo/done/0093_attention_entropy_diagnostics_10rfd.md)、§6 1・5）。
- ヒント計算の共有化前後でactor_approxの同seed・決定的Runの学習metrics checksumを一致させ、既定のEvalPanel networkとRunnerカウンタの通知前後の時機を変えない（出典: [RFD0094](../../../memo/done/0094_evalpanel_step_history_10rfd.md)、§4.2 D6/D7・§8 3）。
- スキップ中も全 lane で forward し、行動だけを上書きする（全 lane の q_values を前提にする step 診断・Actor Q ヒント・spatial・Noisy・stacker・Push の契約を保つ）（出典: [RFD0098 TempoRL](../../../memo/0098_temporl_skip_policy_10rfd.md)、§4 D2・§5.5）。
- 探索のキーを ProfiledValue へ移すとき、既存の log ラダー（0.4〜0.01、0.05〜1e-4）と非 spatial の ε・τ の値を変えず、Atari・DropMerge・GridMaze・LunarLander（DropMergeとLunarLanderだけ最後のlaneを1e-4と書いた検査用設定） の短い決定的 Run の metrics checksum を実装前に保存した実行体と比べる。評価の時機は Actor が `UpdateSchedule` の `exp_step`、Learner が update の先頭の `exp_step`。PER β だけは今の時機（前回の更新の終わりに次回分を評価、初期値は step 0 の値）を保つ（出典: [RFD0101](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md)、§4.2 D7・D8・§4.4 D18・§6.1 1）。
- Munchausen の α・entropy_tau を時間で変えるときは、Actor の Q ヒント・Evaluation Trace の step 診断・Learner の target が同じ step で同じ値を使うことを確かめる（出典: [RFD0101](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md)、§4.4 D19・§6.2 5）。
- 新しい設定が既定のときは、着手前に保存した実行体・設定を使い、スキップ無効と既存の有効構成のmetrics checksumを一致させる。古い設定と新しい実行体を組み合わせる比較を保つ（出典: [RFD0103](../../../memo/done/0103_temporl_reference_alignment_10rfd.md)、§4 D14・§5.7・§7の受入1）。
- 既定のfull・decisionの組み合わせではActorの行動・乱数消費とReplayの抽選列を変えず、RFD0103の検証資材でmetrics checksumを一致させる（出典: [RFD0104](../../../memo/done/0104_temporl_skip_auxiliary_only_10rfd.md)、§5.7・§7の受入1）。
- 既定のsharedではskip側のseedの派生・時計・forwardの回数と順序を変えず、着手前の設定とのmetrics checksumを一致させる（出典: [RFD0106](../../../memo/done/0106_temporl_skip_forward_noise_key_10rfd.md)、§5.3・§7の受入1）。
- `BatchUpdateResult::GetScalar`を整理するときは、keyの名前・返す値・止まる条件と文言・群を引く順を変えない。認識するkeyの全部を値がある状態と無い状態で引く特性テストを先に足して前後で通し、RFD0098の検証資材の短いRun（4環境、seed 1、決定的モード）で時間と資源のtagを除く全scalarの点が一致することを確かめる（出典: [RFD0122](../../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§4 D6〜D8・§6 A5・A7）。
- RNDのwhitenの既定や統計の取り方を変えるときは、`warmup_mode = collect_errors` の組・`variance_feedback`・RND無効の3構成で、同seed・決定的モードの短いRunのmetrics checksumを、着手前に保存した実行体と一致させる（出典: [RFD0123](../../../memo/0123_rnd_intrinsic_reward_scale_10rfd.md)、§4 D5・§7.2 の 1）。
- `variance_feedback`（Taigaらの再現のモード）の中身とwarm-upは変えない（出典: [RFD0123](../../../memo/0123_rnd_intrinsic_reward_scale_10rfd.md)、§1 非ゴール・§4 D8）。
- RNDの累積と `variance_feedback` のcheckpointは、書式の識別子と中身を変えず、着手前に保存した実行体のcheckpointを同じ設定で読めることを確かめる（出典: [RFD0123](../../../memo/0123_rnd_intrinsic_reward_scale_10rfd.md)、§4 D11・§6.3・§7.2 の 6）。
- 学習器の更新の進行・Munchausenの評価・targetの式を変えるときは、乱数を引く位置と順、forwardの順、AMPの範囲の端、優先度の準備→最適化→優先度の反映の順を変えない。TD・QR・IQNのそれぞれにPER・skip・Munchausenの3方式・TBOの有無を組み合わせた31組の短いRun（seed 1、決定的モード、Debug。IQN・Munchausen・replay fitの診断の値も記録する）で、時間と資源のtagを除く全scalarの点を、着手前に保存した実行体と一致させる（出典: [RFD0125](../../../memo/0125_dqn_learner_update_skeleton_10rfd.md)、§4 D5・D13・D17・§5.4）。
- targetの式の共通の関数は実空間の値だけを受け取り、hは完成したtargetに1回だけ掛ける（出典: [RFD0125](../../../memo/0125_dqn_learner_update_skeleton_10rfd.md)、§4 D11・§5.3）。
- 学習器の更新の速さは、Releaseで着手前に保存した実行体と直した後の実行体を交互に3巡以上流し、更新1回あたりの実時間の中央値が直す前のRunの最大値を超えないことで確かめる（出典: [RFD0125](../../../memo/0125_dqn_learner_update_skeleton_10rfd.md)、§4 D16・§6 A9）。
- DefaultDQN系とImageClsのnn_traceの鍵と値を、captureへの統合の前後で変えない。TempoRLのskip側のforwardと、Learnerがtargetの方策を呼ぶSelectActionからは取り出さない（出典: [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md)、§3.1 G2・§5.2）。
- 推定器の読むだけの採点（`EstimateWithoutUpdate`）を挟んでも、`Estimate`・`Learn`の戻り値と`Diagnostics`の値が変わらないことを、whitenの累積・直近重みと`variance_feedback`の3構成で確かめる。どの方式でも、読むだけの採点の倍率が、直後にその観測1件を`Estimate`へ渡したときの倍率と一致する。写しを採点しても元の推定器は変わらない（出典: [RFD0127](../../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md)、§5.1・§7の1・2・4）。
- 診断の群を共通の型へ寄せるときは、メトリクスの名前とWARNの文言・回数を変えず、診断の6群をすべて購読する組を含む前後比較で全scalarの点を一致させ、更新の速さを交互3巡以上で確かめる（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、§5 D12・D16・D18、§6.4）。
- probe用の乱数の作り方を変えるときは、Agentの基のseedと乱数の名前を変えず、購読を設定し直しても作り直さない。名前付きのseedは作る順に依らず、probeは学習の乱数に触れないので、この2つを守れば診断の値も学習の値も変わらない（出典: [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md)、§2.5・§5 D13）。
- RNDの正規化と統計の置き場を変えるときは、RND無効の出荷の設定で同seed・決定的モードのmetrics checksumを着手前の実行体と一致させ、CUDAで計算するRND有効の構成で同seedのRunが2回一致することを確かめる。速さは、RNDなしとRNDありを同じ回・同じ実行体で3巡測り、中央値の比で見る（出典: [RFD0130](../../../memo/0130_rnd_estimate_statistics_perf_10rfd.md)、§5.1 D15・D16、§7）。
- RNDの設定キーを移すとき、式や補助学習の形を足すときは、RND無効・whitenの累積・whitenの直近重み・`variance_feedback`・`collect_errors` に当たる形の5構成で、同seed・決定的モードのmetricsの全scalar点を着手前の実行体と一致させる。誤差を標準化する式の待ちの間の動きと、`variance_feedback` の中身は変えない（出典: [RFD0191](../../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md)、§8.2・§9.2 の 1）。
- Agentのライブラリは基盤の公開ヘッダだけを読み、基盤は具象Agentを知らない。テストは上位や兄弟のライブラリに依存せず、ほかのライブラリは公開ヘッダとテスト用の口だけを通して使う。ターゲットの依存とincludeは、検索と道具の検査で確かめる（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§5.4・§6 のS3-1〜S3-5）。
- ソースを移す・分けるときは、着手前に保存した実行体と、RFD0128の前後比較の組（seed 1本、決定的なbackend、4 envを含む）で比べて、全scalar・設定の記録・WARNとERRORの行の一致を確かめ、旧パスの互換層（symlink・転送ヘッダ）を置かない（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.1 D22・D24・§6 のC3）。
- 強制行動を反映しても、方策の処理（forward、探索の乱数、診断の集計）は全laneで行い、強制していないlaneの結果と乱数を強制の有無で変えない。ヒントの期待値は、消したWithActionのテストの期待値を移して確かめる（出典: [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md)、§6.2 D11・§9 の受入3・4・§10）。
- 重ねる → 転送 → 正規化の並びと、乱数の消費順を変えない。4環境（LunarLander・GridMaze・DropMerge・Atari qbert）の同seed・決定的なバックエンドのRunで、scalarとtraceの全レコードの一致を確かめる（出典: [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md)、§6.5 D35・§9；[RFD0200](../../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.5 D25・§7）。
- episodeの開始の印は、CPUに置いた真偽の値のままForwarderへ渡し、deviceへ移さない（出典: [RFD0200](../../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.1 D1）。
- 同じ Actor の `MakeAction()` と `Sync()` は並行して呼ばず、Runner で呼び出しを直列化する（出典: [Agentと学習](../../../design/110_agents_and_learning.jp.md)、§2.1）。
- 空間探索で環境の並びを反転するときは、ε と τ を同じ向きに並べ、各 lane に割り当てるパラメータの組と分布を保つ（出典: [RFD0002](../../../memo/done/0002_vec_exp_10rfd.md)、§2.1）。
- activation の trace を要求しない target 方策と学習経路は従来の推論を保つ。Conv2d のエピソード間隔による録画の開始・終了も確認する（出典: [RFD0006](../../../memo/done/0006_conv2d_visual_trace_10rfd.md)、§8）。
- QR・IQN の上下の裾を診断しても、方策の行動、両方式の損失、raw priority、実際のリプレイ優先度と乱数列を変えない（出典: [RFD0048](../../../memo/done/0048_quantile_tail_metrics_10rfd.md)、Testing Decisions）。
- テストのタグを改名する場合は、タグを指定する手順も追随させ、既定のテストを途中の変更でも実行できることを確かめる（出典: [RFD0195](../../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md)、§6 D4・§9.6）
- 設定の読み取りと型を直接受ける Learner 構築の両方で不正値を拒否する。診断の変更では、行動・乱数・記録する既存キーを保つ（出典: [RFD0207](../../../memo/0207_dqn_small_fixes_and_refactors_10rfd.md)、§4・§7）
- optimizer の復元を変えるときは、保存前と同じモーメントと step を引き継ぎ、学習率などの値は現在の設定になることを、NoisyNet の有無を含む往復テストで確かめる（出典: [RFD0091](../../../memo/done/0091_checkpoint_load_optimizer_options_10rfd.md)、§7）。
- coreに方策ベースの共通部品を足すときも、DQN系（`src/algo/dqn`）のコードと動きは変えない。基準のコミットからの`src/algo/dqn`の差分が空で、`anet-algo-dqn-test.exe`が通ることで確かめる（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§3.2 NG1・§9 A14）。
- DQN系の部品（rolloutの並べ替え、Actorの写しと同期、optimizerの組み立て、診断の間引き、2つのHead、Perceptionの仕上げ、設定の検証の補助、RNDの段取り）をcoreへ移すときは、`anet-algo-dqn-test.exe`と`anet-core-base-test.exe`、設定の基準と`[shipped_config]`、同じseedのRunの全点の照合（移す前のReleaseのRunnerと、GridMaze・LunarLander・Q*bert・DropMergeで比べ、LunarLander・DropMerge・Atariは`backend.$=backend.@deterministic`。rolloutの並べ替えとRNDの段取りはRNDを有効にした構成を足す）、ライブラリの依存の検査で無回帰を確かめる。rolloutの並べ替えを移して乱数の消費の順が変わると、同じseedの結果が変わる（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§10 の無回帰の確かめ方・C1）。

## 外の事例

記述なし

## 関係

- 設計: [Agentと学習](../../../design/110_agents_and_learning.jp.md)、§2.1・§2.3・§2.4・§3・§6.2。強制行動は同§2.2・§6.1、観測の加工は[Forwarder と Perception](../../../design/135_forwarder_and_perception.jp.md)、§3。
- 一生の新しさの推定器（RND）: [RNDによる新しさの評価と内的報酬](../../../design/212_rnd_lifelong_novelty.jp.md)。
- 方策ベースの共通部品: [方策ベースの学習の共通部品](../../../design/115_on_policy_learning.jp.md)、§1・§2。advantageの推定は[318 強化学習 共通基盤](318_rl_core.md)、Headの合成は[315 ニューラルネットワーク](315_nn.md)、器の全体の読み出しは[314 テンソル処理](314_tensor.md)。
- 拡張点: [ANETフレームワーク全体概要](../../../design/010_framework_overview.jp.md)、§4.3。
- ユーザーガイド: [Run実行ガイド](../../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [dqn_test.cpp](../../../../src/algo/dqn/src/dqn_test.cpp)
- [dqn_actor.cpp](../../../../src/algo/dqn/src/dqn_actor.cpp) / [dqn_actor.hpp](../../../../src/algo/dqn/src/dqn_actor.hpp) / [dqn_actor_test.cpp](../../../../src/algo/dqn/src/dqn_actor_test.cpp)
- [dqn_assembler.cpp](../../../../src/algo/dqn/src/dqn_assembler.cpp) / [dqn_assembler.hpp](../../../../src/algo/dqn/src/dqn_assembler.hpp) / [dqn_assembler_test.cpp](../../../../src/algo/dqn/src/dqn_assembler_test.cpp)
- [dqn_config.cpp](../../../../src/algo/dqn/src/dqn_config.cpp) / [dqn_config.hpp](../../../../src/algo/dqn/src/dqn_config.hpp)
- [dqn_heads.cpp](../../../../src/algo/dqn/src/dqn_heads.cpp) / [dqn_heads.hpp](../../../../src/algo/dqn/src/dqn_heads.hpp)
- [dqn_learner.cpp](../../../../src/algo/dqn/src/dqn_learner.cpp) / [dqn_learner.hpp](../../../../src/algo/dqn/src/dqn_learner.hpp) / [dqn_learner_test.cpp](../../../../src/algo/dqn/src/dqn_learner_test.cpp)
- [dqn_math.cpp](../../../../src/algo/dqn/src/dqn_math.cpp) / [dqn_math.hpp](../../../../src/algo/dqn/src/dqn_math.hpp) / [dqn_math_test.cpp](../../../../src/algo/dqn/src/dqn_math_test.cpp)
- [dqn_munchausen_test.cpp](../../../../src/algo/dqn/src/dqn_munchausen_test.cpp)
- [dqn_network_model.cpp](../../../../src/algo/dqn/src/dqn_network_model.cpp) / [dqn_network_model.hpp](../../../../src/algo/dqn/src/dqn_network_model.hpp) / [dqn_network_model_test.cpp](../../../../src/algo/dqn/src/dqn_network_model_test.cpp)
- [agent.hpp](../../../../src/core/base/include/anet/agent.hpp)
- [distribution.hpp](../../../../src/core/base/include/anet/distribution.hpp)
- [intrinsic.hpp](../../../../src/core/base/include/anet/intrinsic.hpp)
- [agent.hpp](../../../../src/core/base/include/anet/test/agent.hpp)
- [intrinsic.hpp](../../../../src/core/base/include/anet/test/intrinsic.hpp)
- [agent.cpp](../../../../src/core/base/src/agent.cpp)
- [distribution.cpp](../../../../src/core/base/src/distribution.cpp) / [distribution_test.cpp](../../../../src/core/base/src/distribution_test.cpp)
- [intrinsic.cpp](../../../../src/core/base/src/intrinsic.cpp) / [intrinsic_test.cpp](../../../../src/core/base/src/intrinsic_test.cpp)
- [intrinsic_rnd.cpp](../../../../src/core/base/src/intrinsic_rnd.cpp) / [intrinsic_rnd.hpp](../../../../src/core/base/src/intrinsic_rnd.hpp)
- [agent.cpp](../../../../src/core/base/src_testing/agent.cpp)
- [intrinsic.cpp](../../../../src/core/base/src_testing/intrinsic.cpp)

## RFD の一覧

点検が作り直す。手で直さない。

- [RFD0034](../../../memo/done/0034_imagecls_batch_input_10rfd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [RFD0035](../../../memo/done/0035_approx_actor_priority_per_10rfd.md) — 追加 forward なしの近似 Actor PER 初期優先度 PRD
- [RFD0036](../../../memo/done/0036_train_actor_periodic_snapshot_10rfd.md) — DefaultDQN Train Actor定期network snapshot PRD
- [RFD0061](../../../memo/done/0061_eval_slot_policy_override_10rfd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [RFD0062](../../../memo/done/0062_plasticity_metrics_10rfd.md) — 可塑性メトリクス（srank / dormant unit 率）PRD
- [RFD0063](../../../memo/done/0063_plasticity_weight_norm_10rfd.md) — 可塑性 weight norm（feature/readout 分割）PRD
- [RFD0065](../../../memo/done/0065_nn_spectral_norm_10rfd.md) — NN Spectral Normalization（weight_norm.mode）PRD
- [RFD0066](../../../memo/done/0066_policy_churn_metrics_10rfd.md) — Policy churn メトリクス（35_agent_churn）PRD
- [RFD0067](../../../memo/done/0067_MunchausenRL_10rfd.md) — Munchausen RL（M-DQN / M-QR / M-IQN）PRD
- [RFD0070](../../../memo/done/0070_device_spec_auto_10rfd.md) — PRD 070: device 指定の auto 対応とキー体系統一
- [RFD0073](../../../memo/done/0073_replay_fit_metrics_10rfd.md) — PRD 073: replay 抽選履歴別の TD・損失メトリクス
- [RFD0078](../../../memo/done/0078_replay_frame_history_integrity_10rfd.md) — ReplayBuffer の観測履歴・bootstrap stack 整合性修復 PRD
- [RFD0083](../../../memo/done/0083_action_context_retire_10rfd.md) — ActionContext の廃止（Actor が RNG と frame stacker を直接持つ）
- [RFD0084](../../../memo/done/0084_noisynet_10rfd.md) — NoisyNet (Noisy Networks for Exploration) PRD
- [RFD0092](../../../memo/done/0092_gradient_noise_scale_10rfd.md) — 勾配ノイズの大きさ（gradient noise scale）の計測 PRD
- [RFD0093](../../../memo/done/0093_attention_entropy_diagnostics_10rfd.md) — TransformerEncoder の attention エントロピー診断 PRD
- [RFD0094](../../../memo/done/0094_evalpanel_step_history_10rfd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD
- [RFD0098](../../../memo/0098_temporl_skip_policy_10rfd.md) — TempoRL 対応（スキップ: 選んだ行動を続ける長さの学習）RFD
- [RFD0101](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md) — 探索と学習のパラメータを ProfiledValue で時間変化させる PRD
- [RFD0103](../../../memo/done/0103_temporl_reference_alignment_10rfd.md) — TempoRL 第二弾（著者の実装との差の洗い出しと是正）PRD
- [RFD0104](../../../memo/done/0104_temporl_skip_auxiliary_only_10rfd.md) — TempoRL の補助学習のみモード（毎 step 判断し、スキップの出力は補助タスクとしてだけ学ぶ）PRD
- [RFD0106](../../../memo/done/0106_temporl_skip_forward_noise_key_10rfd.md) — TempoRL の skip 側の forward のノイズを Q 側と分けて指定する（`full` でだけノイズが効く仕組みの切り分け）PRD
- [RFD0116](../../../memo/0116_rnd_lifelong_novelty_10rfd.md) — RND による一生の新しさだけの腕（内的報酬の第一の腕）暫定 RFD
- [RFD0117](../../../memo/0117_warn_policy_and_unread_config_10rfd.md) — RFD0117: WARN の意味の線引きと、読まれない設定の検出 RFD
- [RFD0121](../../../memo/0121_extrinsic_intrinsic_value_decomposition_10rfd.md) — 外的と内的の価値の分解（Q を外的報酬と内的報酬の 2 つの Head で学ぶ） RFD
- [RFD0122](../../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md) — RFD0122: 小さなリファクタリングとテストの空白をまとめて片づける
- [RFD0123](../../../memo/0123_rnd_intrinsic_reward_scale_10rfd.md) — RND の内的報酬の尺度（誤差の統計の取り方・warm-up・平均を引かない形・終端の mask） RFD
- [RFD0125](../../../memo/0125_dqn_learner_update_skeleton_10rfd.md) — RFD0125: DQN の学習器の 1 回の更新の進行と Munchausen の target を 1 か所に寄せる
- [RFD0127](../../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md) — Evaluation Trace で RND の内的報酬を見る（統計を変えない採点と step 診断の鍵）RFD
- [RFD0128](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md) — RFD0128: DQN 系 Agent まわりのリファクタリング 5 件をまとめて片づける（価値の分解の前に） RFD
- [RFD0130](../../../memo/0130_rnd_estimate_statistics_perf_10rfd.md) — RFD0130: RND の採点の統計の区間を軽くし、2 つの統計方式を型で分ける RFD
- [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md) — 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化 暫定 RFD
- [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md) — RFD0170: forward の名前付き出力を capture に一本化し、nn_trace を要求した step だけ作る RFD
- [RFD0182](../../../memo/0182_source_layout_10rfd.md) — ソース配置の見直し（置き場の規則、Agent の系統ごとのライブラリ、DQN のファイル分割、テストの参照範囲）RFD
- [RFD0190](../../../memo/0190_dqn_design_doc_alignment_10rfd.md) — 設計文書200（DQN系Agent）の記述をコード・設定・用語集・ADRへそろえる RFD
- [RFD0191](../../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md) — RFD0191: RND の設定を段階ごとに組み替え、誤差で割らない内的報酬の式と、ためてからエポックで学ぶ predictor の補助学習を足す（Burda らの形へ寄せる） RFD
- [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md) — RFD0192: 監査で見つかった不具合 5 件をまとめて直す（Notifier の同時実行、動画の画像の大きさ、強制行動の device、重複登録、QValuePanel の平滑化）
- [RFD0193](../../../memo/0193_variance_scaling_weight_init_10rfd.md) — RFD0193: 重みの初期化に、分散を倍率÷fan_in にする一様分布の方式を足し、初期化の設定の読み取りを 1 つの口にまとめる（Dopamine の IQN・Rainbow と同じ初期値） RFD
- [RFD0194](../../../memo/0194_config_key_full_names_in_messages_10rfd.md) — RFD0194: エラー・WARN・INFO が示す設定キーを完全キーにする（device の二重の fatal、指標名、用語の不整合 6 件を含む）
- [RFD0195](../../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md) — RFD0195: 文書の呼び名を RFD に改め、番号を 0 埋めの 4 桁にし、今使っている場所をすべて移す PRD
- [RFD0200](../../../memo/0200_forwarder_public_interface_rework_10rfd.md) — forwarder.hpp の公開の口の作り直し（作る側と作られるものを分け、spec を依頼ごとに渡す） RFD
- [RFD0207](../../../memo/0207_dqn_small_fixes_and_refactors_10rfd.md) — RFD0207: DQN 系の小さい直しとリファクタリングをまとめて行う（学習の設定の値域、診断値の NaN、fatal の重複、Learner の State と値の振り分け、写しの集約、特性テスト）
- [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md) — RFD0209: PPO 系（方策ベースの Agent の系統）と、方策ベースの学習に要る core の共通部品

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | RFD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261122-c1744a](../../../management/issues/issue-20261122-c1744a.md) | 対応中 | 2.6 | — | コード | RFD | 321_agent_core, 324_dqn | 学習基盤 | 機能改善 | [RFD0191](../../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md) | 残件有り | RND と補助学習の設定キーが段階とモードの関係を表しておらず、warm-up の 1 つの件数が 2 つの待ちを束ねていて、設計文書 212 から設定と動きを引きにくい |
| [issue-20261125-8eff2e](../../../management/issues/issue-20261125-8eff2e.md) | 未着手 | 5.0 | — | コード | 監査 | 321_agent_core | 学習基盤 | リファクタリング | — | 未確認 | DQN の Learner が、forward のあとに「価値の分解なら合わせた Q を作る」処理を 6 か所に条件つきで書き、同じ目的の DqnForwardExecutor と 2 つの道になっている |
| [issue-20261125-fa1be2](../../../management/issues/issue-20261125-fa1be2.md) | 未着手 | 5.0 | — | コード | 監査 | 321_agent_core | 学習基盤 | リファクタリング | — | 未確認 | ActorQHintConfig が、ヒントではなく行動の選択（価値の分解・β・TBO）と診断の割引率を決めていて、名前と役割が合わない |
| [issue-20261123-6b991d](../../../management/issues/issue-20261123-6b991d.md) | 未着手 | 4.0 | — | 文書 | 監査 | 311_dev_tools, 321_agent_core | 開発 | 調査 | — | 残件有り | 状態が中身と合わない課題が 3 件ある（対応中なのに実装計画も最近の作業も無い 23d185・52ffba、RFD が着手しないと書くのに未着手の 6f6c73） |
| [issue-20261125-fe014b](../../../management/issues/issue-20261125-fe014b.md) | 未着手 | 4.0 | — | コード | 監査 | 321_agent_core | 学習基盤 | 不具合 | — | 未確認 | 価値の分解の行動選択の診断（extrinsic_q_gap など）が、IQN の UQE では平均の Q ではなく risk の τ（全分布の問い合わせがあれば risk と全範囲を混ぜた τ）の平均で作られる |
| [issue-20261125-3cb74a](../../../management/issues/issue-20261125-3cb74a.md) | 未着手 | 3.5 | — | コード | 監査 | 321_agent_core | 学習基盤 | 不具合 | — | 未確認 | 価値の分解の split（feature_gradient=split）で勾配ノイズの指標を購読すると、内的の Head の state が外的の optimizer に無いため Adam 版が常に未成立になり、ERROR も WARN も無く tag が欠ける |
| [issue-20261115-e5a0eb](../../../management/issues/issue-20261115-e5a0eb.md) | 未着手 | 3.4 | — | コード | RFD | 315_nn, 321_agent_core, 317_metrics | 学習基盤 | 性能改善 | — | 残件有り | 可塑性メトリクスの計算が、500 learn step ごとにメインスレッドで約 2 秒走り、その間は学習が止まる |
| [issue-20261125-16374e](../../../management/issues/issue-20261125-16374e.md) | 未着手 | 3.3 | — | コード | 監査 | 321_agent_core, 318_rl_core | 学習基盤 | 不具合 | — | 未確認 | checkpoint の読み込みが parameter の device と形を今の Network と突き合わせず、CUDA で保存したものを agent.device=cpu で読むと Network が CUDA に載ったままになる（最初の forward で落ちる見込み） |
| [issue-20261111-585e60](../../../management/issues/issue-20261111-585e60.md) | 未着手 | 3.0 | — | コード | 監査 | 318_rl_core, 321_agent_core, 315_nn, 110_runner_app | 基盤 | リファクタリング | — | 残件有り | 名前やコメントが実際の動きと合わない関数・クラス・設定項目がある（BatchState の IsDone、DefaultAgentFactory、NetworkGraphVizConfig の 2 項目、Runner の 4 関数） |
| [issue-20261125-0e98cb](../../../management/issues/issue-20261125-0e98cb.md) | 未着手 | 3.0 | — | コード | 監査 | 321_agent_core | 学習基盤 | リファクタリング | — | 未確認 | UQE の τ の問い合わせと score の作り方が、行動選択の MakeUQEActionInfo と、skip・価値の分解のための関数に 2 回ずつ書かれ、探索の適用も 4 か所にある |
| [issue-20261125-7c9e47](../../../management/issues/issue-20261125-7c9e47.md) | 未着手 | 3.0 | — | コード | 監査 | 321_agent_core | 学習基盤 | リファクタリング | — | 未確認 | RndNoveltyEstimator の外側に統計と誤差の方式の分岐が 4 か所戻り、倍率を 1 にする条件が、更新する採点と読むだけの採点の 2 つに別々に書かれている |
| [issue-20261125-d1dfc6](../../../management/issues/issue-20261125-d1dfc6.md) | 未着手 | 2.8 | — | コード | 監査 | 321_agent_core, 324_dqn | 学習基盤 | リファクタリング | — | 未確認 | DQN の観測の加工（Perception）の組み立ての決まりが Agent・Learner・Actor の 3 ファイルに分かれ、依頼ごとに「作る → 補う → 照合する」を別々に書いている |
| [issue-20261111-17fea4](../../../management/issues/issue-20261111-17fea4.md) | 未着手 | 2.7 | — | コード | 監査 | 324_dqn, 321_agent_core | 学習基盤 | リファクタリング | — | 残件有り | 方策の種類（policy_type）と分布の表し方（quantile_mode）を文字列のまま持ち回り、数字の別名の受け付け方が関数ごとに違う |
| [issue-20261111-52ed74](../../../management/issues/issue-20261111-52ed74.md) | 未着手 | 2.7 | — | コード | 監査 | 315_nn, 321_agent_core | 学習基盤 | リファクタリング | — | 残件有り | 可塑性指標の一覧と重みノルムのキーの並びを、nn・DQN・ImageCls の 5〜6 か所で手で合わせている |
| [issue-20261111-e80c7b](../../../management/issues/issue-20261111-e80c7b.md) | 未着手 | 2.7 | — | コード | 監査 | 321_agent_core | 学習基盤 | リファクタリング | — | 残件有り | DQNActionInfo::GetScalar が 6 種の指標群を 1 つの関数（276 行）で処理し、「指定した行動を除いた最大値」の計算を 2 回写している |
| [issue-20261123-6bb71d](../../../management/issues/issue-20261123-6bb71d.md) | 未着手 | 2.5 | — | コード | 監査 | 315_nn, 321_agent_core, 317_metrics | 学習基盤 | 不整合 | — | 残件有り | コードの名前・エラー文・メトリクスの tag に CONTEXT.md の避ける語（feature key・mean Q・action churn）が残り、エラー文は実在しない設定キー名 feature_key を示す |
| [issue-20261103-2d66d2](../../../management/issues/issue-20261103-2d66d2.md) | 未着手 | 2.2 | — | コード | RFD | 324_dqn, 321_agent_core, 325_muzero, 315_nn | 学習基盤 | 調査 | [RFD0134](../../../memo/0134_network_lock_audit_10rfd.md) | 残件有り | 999: train/eval mode 整理後の排他精査 RFD（草案） |
| [issue-20261103-5addea](../../../management/issues/issue-20261103-5addea.md) | 未着手 | 2.2 | — | コード | RFD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [RFD0079](../../../memo/0079_cpp_internal_linkage_convention_10rfd.md) | 残件有り | RFD0079: .cpp の内部リンケージと namespace インデント規約 |
| [issue-20261103-cdac7c](../../../management/issues/issue-20261103-cdac7c.md) | 未着手 | 2.2 | — | コード | RFD | 321_agent_core | アルゴリズム | 機能改善 | [RFD0148](../../../memo/0148_target_update_step_axis_10rfd.md) | 残件有り | target network 更新の step 軸正規化（`soft_update_tau` / `hard_update_interval`） |
| [issue-20261111-4351bc](../../../management/issues/issue-20261111-4351bc.md) | 未着手 | 2.2 | — | コード | 監査 | 321_agent_core | 学習基盤 | リファクタリング | — | 残件有り | DQN の Actor が const の MakeAction の中で 11 個の mutable メンバを書き換えて状態を進め、一時的な出力を結果の public メンバで運んでいる |
| [issue-20261125-5cef87](../../../management/issues/issue-20261125-5cef87.md) | 未着手 | 2.2 | — | コード | 監査 | 321_agent_core | 学習基盤 | リファクタリング | — | 未確認 | 価値の分解の目標値（MakeDecomposedTargets）が、RFD0125 で 1 か所にした Munchausen の評価と hard target の行動選択を写し、分布ごとの違いを quantile_mode の分岐で書いている |
| [issue-20261103-72f807](../../../management/issues/issue-20261103-72f807.md) | 未着手 | 2.0 | — | コード | 監査 | 321_agent_core | アルゴリズム | 不具合 | — | 残件有り | use_td_clipが損失の前にTDをclampするため、clipを超えたサンプルの勾配が0になる |
| [issue-20261103-785e68](../../../management/issues/issue-20261103-785e68.md) | 未着手 | 2.0 | — | コード | RFD | 315_nn, 318_rl_core, 321_agent_core, 324_dqn, 319_gui, 320_env_core, 314_tensor, 312_util, 110_runner_app, 322_env_individual, 325_muzero | 開発 | リファクタリング | [RFD0169](../../../memo/0169_outside_core_extension_layout_10rfd.md) | 残件無し | anet-core の外で実装する拡張点とコード配置の規約 |
| [issue-20261103-9bd71e](../../../management/issues/issue-20261103-9bd71e.md) | 未着手 | 2.0 | — | コード | RFD | 321_agent_core, 318_rl_core, 324_dqn, 317_metrics | 学習基盤 | 機能追加 | [RFD0142](../../../memo/0142_agent_prev_action_obs_10rfd.md) | 残件有り | 直前行動観測の共通部品（`prev_action` フレーム段） |
| [issue-20261103-a39330](../../../management/issues/issue-20261103-a39330.md) | 未着手 | 2.0 | — | コード | RFD | 321_agent_core, 324_dqn | アルゴリズム | 機能追加 | [RFD0147](../../../memo/0147_iqn_first_priority_quantile_pair_10rfd.md) | 残件有り | RFD0147: IQN初回Learner priorityのquantile-pair方式 |
| [issue-20261111-54c7e1](../../../management/issues/issue-20261111-54c7e1.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core, 321_agent_core, 315_nn, 319_gui, 110_runner_app | 基盤 | リファクタリング | — | 残件有り | 呼ばれない関数・型・マクロと読まれないメンバ・届かない分岐が、core と Runner に合わせて 1,100 行以上残っている |
| [issue-20261119-9afcf4](../../../management/issues/issue-20261119-9afcf4.md) | 未着手 | 1.7 | — | コード | RFD | 311_dev_tools, 321_agent_core | 開発 | リファクタリング | — | 残件有り | テストごとに書かれている代役の Agent 8 個を、基盤のテスト用の口の代役へ寄せる |
| [issue-20261123-8cf911](../../../management/issues/issue-20261123-8cf911.md) | 未着手 | 1.7 | — | コード | 監査 | 318_rl_core, 321_agent_core | 学習基盤 | リファクタリング | — | 残件有り | Agent の拡張点 AgentFactory::CreateAgent が Notifier を受け取るが、本番の Agent 4 つはどれも使わず、使うのはテストの代役 1 つ（Observer の登録順を前にするため）だけ |
| [issue-20261103-0ef7b8](../../../management/issues/issue-20261103-0ef7b8.md) | 未着手 | 1.6 | — | コード | RFD | 317_metrics, 311_dev_tools, 316_config, 318_rl_core, 321_agent_core, 120_metrics_viewer, 110_runner_app | UI | 機能改善 | [RFD0108](../../../memo/0108_metrics_tag_numbering_10rfd.md) | 残件有り | メトリクスの tag の番号を全体で振り直す（群の番号と群の中の番号） |
| [issue-20261103-75b8da](../../../management/issues/issue-20261103-75b8da.md) | 未着手 | 1.5 | — | コード | RFD | 321_agent_core, 318_rl_core | アルゴリズム | 機能改善 | [RFD0168](../../../memo/0168_nstep_target_correction_10rfd.md) | 残件有り | n-step の目標値の補正（窓の途中の Munchausen の項・方策オフの補正） |
| [issue-20261103-e29c92](../../../management/issues/issue-20261103-e29c92.md) | 未着手 | 1.4 | — | コード | 監査 | 321_agent_core, 320_env_core, 319_gui, 315_nn | 基盤 | リファクタリング | — | 残件有り | 名前→factoryのRepositoryと自動登録マクロを4か所で別々に実装し、重複登録の扱いが食い違う |
| [issue-20261103-50644a](../../../management/issues/issue-20261103-50644a.md) | 未着手 | 1.4 | — | コード | RFD | 317_metrics, 320_env_core, 321_agent_core, 318_rl_core, 110_runner_app | UI | 機能追加 | [RFD0184](../../../memo/0184_eval_session_step_trace_10rfd.md) | 残件有り | 評価 session の step trace を Run の記録に残す |
| [issue-20261103-52bda0](../../../management/issues/issue-20261103-52bda0.md) | 未着手 | 1.4 | — | コード | RFD | 318_rl_core, 321_agent_core, 120_metrics_viewer, 311_dev_tools | UI | 機能改善 | [RFD0096](../../../memo/0096_metrics_step_axis_interval_10rfd.md) | 残件有り | メトリクスの横軸と間隔を数える単位を分ける |
| [issue-20261103-aa8701](../../../management/issues/issue-20261103-aa8701.md) | 未着手 | 1.4 | — | コード | RFD | 318_rl_core, 321_agent_core, 324_dqn, 320_env_core | 学習基盤 | 機能追加 | [RFD0143](../../../memo/0143_serialize_10rfd.md) | 残件有り | Run Serialization（Run 全体 Save / Load） |
| [issue-20261109-adc5c4](../../../management/issues/issue-20261109-adc5c4.md) | 未着手 | 1.4 | — | コード | 監査 | 321_agent_core, 320_env_core, 319_gui, 315_nn | 基盤 | リファクタリング | — | 残件有り | Agent・Env・View・NNモジュールの4つのRepositoryが同じ形を別々に実装し、登録と取り出しの関数名もそろっていない |
| [issue-20261112-4efa57](../../../management/issues/issue-20261112-4efa57.md) | 未着手 | 1.4 | — | コード | RFD | 321_agent_core | 学習基盤 | リファクタリング | — | 残件有り | replay fitの診断（EvaluateReplayFit）が学習の更新の前半を別に書き、分布の表し方での分岐を持っている |
| [issue-20261115-c89660](../../../management/issues/issue-20261115-c89660.md) | 未着手 | 1.4 | — | コード | その他 | 110_runner_app, 321_agent_core, 318_rl_core | UI | 機能追加 | — | 残件有り | Runner の Conv2dPanel と metrics.image の conv2d に、RND の target と predictor の畳み込みの出力が出ない |
| [issue-20261103-71c4f4](../../../management/issues/issue-20261103-71c4f4.md) | 未着手 | 1.3 | — | コード | RFD | 315_nn, 317_metrics, 321_agent_core | 学習基盤 | 機能追加 | [RFD0135](../../../memo/0135_nn_block_metrics_10rfd.md) | 残件有り | NN ブロック別 勾配メトリクス — 背景・目的・基本方針（暫定） |
| [issue-20261103-8c3c6d](../../../management/issues/issue-20261103-8c3c6d.md) | 未着手 | 1.2 | — | コード | RFD | 318_rl_core, 324_dqn, 321_agent_core, 322_env_individual | 学習基盤 | 機能追加 | [RFD0171](../../../memo/0171_replay_eviction_sample_count_10rfd.md) | 残件有り | replay の遷移が消えるまでの抽選回数と優先度を、PER の外の区分で測る |
| [issue-20261122-258ef8](../../../management/issues/issue-20261122-258ef8.md) | 未着手 | 1.2 | — | コード | RFD | 321_agent_core, 324_dqn, 314_tensor | 学習基盤 | 不具合 | — | 残件有り | 統計つきの観測の正規化で、Actor が Agent の排他の外で統計を読み、学習と並んで走る評価の Actor で読み書きが競合する |
| [issue-20261103-1edf84](../../../management/issues/issue-20261103-1edf84.md) | 未着手 | 1.1 | — | コード | RFD | 321_agent_core, 324_dqn, 315_nn | アルゴリズム | 機能追加 | [RFD0172](../../../memo/0172_bootstrapped_multihead_q_10rfd.md) | 残件有り | Bootstrapped DQN 型の複数の Head（1 本の Run の中に別々の価値関数を持つ） |
| [issue-20261103-51220d](../../../management/issues/issue-20261103-51220d.md) | 未着手 | 1.1 | — | コード | RFD | 318_rl_core, 321_agent_core, 322_env_individual, 317_metrics | 学習基盤 | 機能追加 | [RFD0144](../../../memo/0144_episode_forensics_10rfd.md) | 残件有り | Episode Forensics（特殊挙動エピソード自動捕捉・事後調査） |
| [issue-20261116-cc5942](../../../management/issues/issue-20261116-cc5942.md) | 未着手 | 1.1 | — | コード | その他 | 318_rl_core, 321_agent_core, 314_tensor, 324_dqn | 学習基盤 | 性能改善 | — | 残件有り | 同じデータを何度もdevice間で転送している箇所を洗い出して減らす（Replayのobsとnext_obsの重なるフレーム、RNDの採点とActorが別々に送る観測など） |
| [issue-20261103-6f6c73](../../../management/issues/issue-20261103-6f6c73.md) | 未着手 | 1.0 | — | コード | RFD | 321_agent_core | アルゴリズム | 機能追加 | [RFD0183](../../../memo/0183_temporl_skip_followups_10rfd.md) | 残件有り | TempoRL の後続（長い j の不確かさ・閉ループの割り込み・全 k の一括学習） |
| [issue-20261111-d1d12b](../../../management/issues/issue-20261111-d1d12b.md) | 未着手 | 1.0 | — | コード | 監査 | 321_agent_core, 315_nn, 318_rl_core, 317_metrics, 316_config, 312_util, 313_profiling, 314_tensor, 322_env_individual, 110_runner_app | 学習基盤 | リファクタリング | — | 残件有り | 今は起こさないリファクタリング候補 16 群を暫定で 1 件にまとめた（手を付けるときに群ごとに分けて起票し直す） |
| [issue-20261119-cb9a0f](../../../management/issues/issue-20261119-cb9a0f.md) | 未着手 | 1.0 | — | コード | その他 | 324_dqn, 321_agent_core | アルゴリズム | 機能追加 | — | 残件有り | RNDの観測の正規化に、学習の前に一様乱数の行動で統計を作る前段（Burdaら）が無い |
| [issue-20261103-f33167](../../../management/issues/issue-20261103-f33167.md) | 未着手 | 0.9 | — | コード | RFD | 324_dqn, 321_agent_core, 315_nn | アルゴリズム | 機能追加 | [RFD0107](../../../memo/0107_noisy_gain_lane_ladder_10rfd.md) | 残件有り | 学習の Actor の lane ごとに NoisyNet のノイズの強さを変える（ノイズのラダー） |
| [issue-20261103-afe841](../../../management/issues/issue-20261103-afe841.md) | 未着手 | 0.8 | — | コード | RFD | 321_agent_core, 324_dqn, 318_rl_core | 学習基盤 | 機能追加 | [RFD0145](../../../memo/0145_staged_batch_replay_schedule_10rfd.md) | 残件有り | DefaultDQN Staged Batch / Replay Schedule |
| [issue-20261122-4c0af5](../../../management/issues/issue-20261122-4c0af5.md) | 未着手 | 0.8 | — | コード | RFD | 318_rl_core, 314_tensor, 321_agent_core | 学習基盤 | リファクタリング | — | 残件有り | 観測をためる仕組み（フレーム積層の履歴、Replay の観測の置き場、補助学習の rollout の置き場）が別々に実装されていて、共通の置き場の部品へ載せ替えていない |
| [issue-20261103-969184](../../../management/issues/issue-20261103-969184.md) | 未着手 | 0.6 | — | コード | RFD | 321_agent_core, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | [RFD0174](../../../memo/0174_llm_actor_10rfd.md) | 残件有り | LLM Actor（LLM の事前知識で行動する Actor。単独 Agent と DQN ハイブリッド） |
| [issue-20261113-6f954f](../../../management/issues/issue-20261113-6f954f.md) | 未着手 | 0.6 | — | コード | RFD | 321_agent_core, 324_dqn | 学習基盤 | リファクタリング | — | 残件有り | Learner の診断の購読が、Run で使う診断の宣言と、呼び出しごとの要求を 1 つの口で兼ねている |
| [issue-20261122-ba42c9](../../../management/issues/issue-20261122-ba42c9.md) | 未着手 | 0.6 | — | コード | RFD | 321_agent_core, 314_tensor | 学習基盤 | 機能改善 | — | 残件有り | RND の観測の加工（byte を float に直す、自前の統計で正規化する）が、Q の側の Forwarder の並びと別に書かれている |
| [issue-20261103-145ee2](../../../management/issues/issue-20261103-145ee2.md) | 未着手 | 0.5 | — | コード | RFD | 321_agent_core, 318_rl_core, 316_config | 学習基盤 | 機能追加 | [RFD0152](../../../memo/0152_metric_targeted_hyperparam_control_10rfd.md) | 残件有り | メトリクス目標値によるハイパラ自律調整 |
| [issue-20261103-4a2b5e](../../../management/issues/issue-20261103-4a2b5e.md) | 未着手 | 0.5 | — | コード | RFD | 315_nn, 321_agent_core | 基盤 | リファクタリング | [RFD0165](../../../memo/0165_nn_head_projection_separation_10rfd.md) | 残件有り | NN Head の最終射影と出力変換の責務分離 |
| [issue-20261103-f993bc](../../../management/issues/issue-20261103-f993bc.md) | 未着手 | 0.5 | — | コード | RFD | 321_agent_core, 324_dqn, 325_muzero, 315_nn | 学習基盤 | リファクタリング | [RFD0164](../../../memo/0164_nn_forward_amp_consolidation_10rfd.md) | 残件有り | NN 実行設定への AMP の集約（全 Agent 対象） |
| [issue-20261115-d3e094](../../../management/issues/issue-20261115-d3e094.md) | 未着手 | 0.5 | — | コード | その他 | 324_dqn, 321_agent_core | アルゴリズム | 調査 | — | 残件有り | RNDの改良版のSND（targetも自己教師で学ぶ蒸留）を、RNDの評価が落ち着いたあとに試すかを検討する |
| [issue-20261103-124da5](../../../management/issues/issue-20261103-124da5.md) | 未着手 | 1.5 | [issue-20261103-3e1cf6](../../../management/issues/issue-20261103-3e1cf6.md) | コード | RFD | 321_agent_core, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | [RFD0167](../../../memo/0167_smdp_transition_duration_10rfd.md) | 残件有り | SMDP 対応（遷移ごとの経過時間を割引に通す） |
| [issue-20261103-736e39](../../../management/issues/issue-20261103-736e39.md) | 未着手 | 1.4 | [issue-20261103-9bd71e](../../../management/issues/issue-20261103-9bd71e.md), [issue-20261103-75b8da](../../../management/issues/issue-20261103-75b8da.md), [issue-20261107-949e8a](../../../management/issues/issue-20261107-949e8a.md) | コード | RFD | 324_dqn, 321_agent_core, 318_rl_core, 322_env_individual, 110_runner_app | アルゴリズム | 機能追加 | [RFD0173](../../../memo/0173_intrinsic_reward_novelty_10rfd.md) | 残件有り | 内的報酬（NGU・E3B のエピソード内の新しさと RND）と方策の組 |
| [issue-20261108-227b05](../../../management/issues/issue-20261108-227b05.md) | 未着手 | 0.8 | [issue-20261103-736e39](../../../management/issues/issue-20261103-736e39.md) | コード | RFD | 324_dqn, 321_agent_core | アルゴリズム | 調査 | — | 残件有り | RNDのpredictorを周期的に初期化してbonusを続かせる案は、GridMazeとLunarLanderの試行で利得が無く、実装を戻した |
| [issue-20261103-5a783c](../../../management/issues/issue-20261103-5a783c.md) | 未着手 | 0.6 | [issue-20261103-124da5](../../../management/issues/issue-20261103-124da5.md), [issue-20261103-75b8da](../../../management/issues/issue-20261103-75b8da.md) | コード | RFD | 321_agent_core, 322_env_individual | アルゴリズム | 機能追加 | [RFD0162](../../../memo/0162_decision_aligned_nstep_10rfd.md) | 残件有り | 次の判断まで遡る n-step（判断単位の可変長 n-step） |

着手できる 56 / 待ち 4 / 保留 0 / 次版 0 / 解決 0
- 関連する課題: [issue-20261107-949e8a](../../../management/issues/issue-20261107-949e8a.md) — AtariのRAMメトリクスに値の種類数とビットを数える畳み方が無く、Montezumaの訪れた部屋の数や持ち物を記録できない
- 関連する課題: [issue-20261103-3e1cf6](../../../management/issues/issue-20261103-3e1cf6.md) — DropMerge の評価スコア（本家互換スコアほか複数の軸）
<!-- issues:end -->

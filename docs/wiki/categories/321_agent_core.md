# 321 Agent 共通基盤

## 概要・責務

- Agent、Actor、Learnerの生成・行動・更新と、Agentが所有するResource、更新部品が所有するStateの境界を扱う。（出典: [Agentと学習](../../design/110_agents_and_learning.jp.md)、§2.1・§2.3・§3）。

## 成立の経緯

- 2026-07-12: Actor初期優先度は平均Q空間の近似とし、Learner target-network評価/UQE targetの再現を見送る（出典: [ADR 0010](../../adr/0010-actor-priority-mean-q-approx.md)、本文）。
- 2026-07-12: Replay優先度更新にはgeneration付きreplay item keyを使用する（出典: [ADR 0011](../../adr/0011-generation-aware-replay-item-key.md)、本文）。
- 2026-07-13: Replay初期優先度ヒントを単一packed carrierとし、completionを専用coordinatorへ分離する（出典: [ADR 0012](../../adr/0012-replay-initial-priority-hint-completion.md)、本文）。
- 2026-07-13: 追加 forward なしの近似 Actor PER 初期優先度 PRD（出典: [PRD035](../../memo/done/035_approx_actor_priority_per_10prd.md)、題・目的）。
- 2026-07-15: Actor network方針の既定値解決をAgentへ戻す（出典: [ADR 0013](../../adr/0013-actor-network-resource-policy.md)、本文）。
- 2026-07-16: DefaultDQN Train Actor定期network snapshot PRD（出典: [PRD036](../../memo/done/036_train_actor_periodic_snapshot_10prd.md)、題・目的）。
- 2026-07-20: ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、題・目的）。
- 2026-08-09: IQN+UQE の Q 系値は同一 forward の risk-biased action score を使う（出典: [ADR 0019](../../adr/0019-iqn-uqe-score-without-extra-forward.md)、本文）。
- 2026-08-10: 044: IQN tau sampling modes(stratified / systematic / antithetic)（出典: [PRD044](../../memo/done/044_iqn_tau_stratified_sampling_10prd.md)、題・目的）。
- 2026-08-11: PRD 047: IQN探索P0診断メトリクス（出典: [PRD047](../../memo/done/047_iqn_search_p0_metrics_10prd.md)、題・目的）。
- 2026-08-11: PRD 048: 分位tail探索診断メトリクス（出典: [PRD048](../../memo/done/048_quantile_tail_metrics_10prd.md)、題・目的）。
- 2026-08-21: Conv2d 可視化「実Forwardトレース化」実装指示書（出典: [PRD006](../../memo/done/006_conv2d_visual_trace_10prd.md)、題・目的）。
- 2026-08-21: DefaultDQNAgent Dueling: V/A ストリームを body DAG で分岐できるようにする（出典: [PRD009](../../memo/done/009_dueling_dag_head_10prd.md)、題・目的）。
- 2026-08-21: FusedAdamW（ATen `_fused_adamw_` 直接呼び出し）導入 仕様書（出典: [PRD011](../../memo/done/011_fused_adamw_10prd.md)、題・目的）。
- 2026-08-21: IQN 対応改修（quantile_mode 導入・bind `*` 演算子・IQNHead/IQNLearner）（出典: [PRD001](../../memo/done/001_iqn_10prd.md)、題・目的）。
- 2026-08-21: ImageCls BF16(AMP)対応 + 汎用 LinearHead による Head 化（出典: [PRD033](../../memo/done/033_imagecls_bf16_head_10prd.md)、題・目的）。
- 2026-08-21: PrefetchingReplayBuffer リファクタ（Learner 埋め込み prefetch を decorator へ分離 + 再現性修正）仕様書（出典: [PRD014](../../memo/done/014_async_loader_prefetch_10prd.md)、題・目的）。
- 2026-08-21: Sample+To プリフェッチ（ReplayBuffer サンプリング + H2D を GPU 学習と overlap）仕様書（出典: [PRD013](../../memo/done/013_sample_prefetch_10prd.md)、題・目的）。
- 2026-08-21: Transformed Bellman Operator (TBO) 導入 仕様書（出典: [PRD008](../../memo/done/008_Transformed%20Bellman%20Operator_10prd.md)、題・目的）。
- 2026-08-21: V2: 可視化層（metrics_image / HeatMap / 状態スイープ）の TensorDict 復旧 実装指示書（出典: [PRD010](../../memo/done/010_image_metrics_v2_10prd.md)、題・目的）。
- 2026-08-21: 【修正仕様】Ape-X型 空間探索（Vectorized Exploration）機能の追加（出典: [PRD002](../../memo/done/002_vec_exp_10prd.md)、題・目的）。
- 2026-08-22: Stack 次元のネイティブ対応(Reshape ハックを廃する根本対応)（出典: [PRD057](../../memo/done/057_native_stack_dim_handling_10prd.md)、題・目的）。
- 2026-08-27: 可塑性メトリクスは実 forward の branch capture と学習経路外の部分 forward の 2 チャネルで測る（出典: [ADR 0031](../../adr/0031-plasticity-metrics-out-of-band-partial-forward.md)、本文）。
- 2026-08-29: Spectral Normalization は自前 power iteration + u/v named buffer で実装し、σ は毎 forward その場の W で再計算する（出典: [ADR 0032](../../adr/0032-spectral-norm-self-impl-buffer-semantics.md)、本文）。
- 2026-08-29: 可塑性 weight norm（feature/readout 分割）PRD（出典: [PRD063](../../memo/done/063_plasticity_weight_norm_10prd.md)、題・目的）。
- 2026-08-29: 可塑性メトリクス（srank / dormant unit 率）PRD（出典: [PRD062](../../memo/done/062_plasticity_metrics_10prd.md)、題・目的）。
- 2026-08-30: NN Spectral Normalization（weight_norm.mode）PRD（出典: [PRD065](../../memo/done/065_nn_spectral_norm_10prd.md)、題・目的）。
- 2026-09-02: Policy churn は fixed probe で optimizer 前後を測り、target は sync 後の不一致を測る（出典: [ADR 0033](../../adr/0033-policy-churn-fixed-probe-and-target-lag.md)、本文）。
- 2026-09-02: Policy churn メトリクス（35_agent_churn）PRD（出典: [PRD066](../../memo/done/066_policy_churn_metrics_10prd.md)、題・目的）。
- 2026-09-04: Actor Qヒントを3列へ拡張し、近似Actor初期優先度をMunchausenターゲットと同型にする（出典: [ADR 0036](../../adr/0036-actor-q-hint-three-columns-munchausen.md)、本文）。
- 2026-09-04: MunchausenターゲットはDQN Learnerに局所化し、実空間で計算し、3つのlog-policy modeを持つ（出典: [ADR 0035](../../adr/0035-munchausen-target-learner-local-real-space.md)、本文）。
- 2026-09-06: Munchausen RL（M-DQN / M-QR / M-IQN）PRD（出典: [PRD067](../../memo/done/067_MunchausenRL_10prd.md)、題・目的）。
- 2026-09-08: Actor 設定は名前付きカタログ参照とし、Agent インタフェースから RunMode を外す（出典: [ADR 0038](../../adr/0038-actor-config-catalog-without-runmode.md)、本文）。
- 2026-09-10: replay 当てはまり診断は held-out 分割ではなく既存の抽選履歴フラグで群分けし、評価条件を学習条件から独立させる（出典: [ADR 0039](../../adr/0039-replay-fit-sampling-history-groups-not-holdout.md)、本文）。
- 2026-09-11: PRD 073: replay 抽選履歴別の TD・損失メトリクス（出典: [PRD073](../../memo/done/073_replay_fit_metrics_10prd.md)、題・目的）。
- 2026-09-19: PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、題・目的）。
- 2026-09-20: ReplayBuffer の観測履歴・bootstrap stack 整合性修復 PRD（出典: [PRD078](../../memo/done/078_replay_frame_history_integrity_10prd.md)、題・目的）。
- 2026-09-24: Agentのdevice指定をauto対応の文字列へ統一した（出典: [PRD070](../../memo/070_device_spec_auto_10prd.md)、確定したD1・D3）。
- 2026-09-27: ActionContextを廃止しRNGと観測加工をActorへ直接移した（出典: [PRD083](../../memo/083_action_context_retire_10prd.md)、題・Context）。
- 2026-10-03: NoisyNet の ε は module に持たせず、呼び出し側所有の NN 実行状態と NN 実行設定キーで駆動する（出典: [ADR 0048](../../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md)、本文）。
- 2026-10-10: ActorとLearnerの用途別forwardをNN実行設定の参照へ接続した（出典: [PRD084](../../memo/084_noisynet_10prd.md)、§4.2・§6・§7）。
- 2026-10-10: NN Forward の設定解釈を事前準備へ移し、用途別実行オブジェクトが State と Network の寿命を持つ（出典: [ADR 0051](../../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md)、本文）。
- 2026-10-16: Learnerの更新内で勾配ノイズを測り購読とEMAをAgent側へ接続した（出典: [PRD092](../../memo/092_gradient_noise_scale_10prd.md)、§4・§6・§9.1）。
- 2026-10-16: forward の持ち帰りは鍵要求型の capture 1 つに統一し、module の診断も同じ器で運ぶ（出典: [ADR 0055](../../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md)、本文）。
- 2026-10-16: 勾配ノイズスケールは更新自身の batch を小バッチに分けて学習経路の中で測り、BF16 も対象にする（出典: [ADR 0054](../../adr/0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16.md)、本文）。
- 2026-10-18: Learnerの購読時probeと更新結果へattention entropyを接続した（出典: [PRD093](../../memo/093_attention_entropy_diagnostics_10prd.md)、§4 決定した設計）。
- 2026-10-19: step trace の近似優先度は Agent が出す step 部品が計算し、Actor の設定キーで eval_panel にだけヒントを作らせることはしない（出典: [ADR 0058](../../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)、本文）。
- 2026-10-21: Actorと共有するヒント計算を使うDQNのstep診断部品を加えた（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§4.2 D6）。
- 2026-10-24: 探索の ε・UQE の ε と τ・ラダーの両端を ProfiledValue で時間変化させ、時間の値とラダーの両端を別キーに分け、学習側の 7 値も同じ器にする（出典: [PRD101](../../memo/101_exploration_schedule_profiled_value_10prd.md)、題・§1・§4）。
- 2026-10-24: 時間で変わる割引率は、schedule のときだけ抽選時に保存済みの step 報酬から収益を作り直す（出典: [ADR 0065](../../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)、本文）。

## 選ばなかった形

- ImageClsのTrain Actorにも遅延snapshotを適用する形 — 不採用: episodeを持たず、shared networkの即時反映を維持する（出典: [PRD036](../../memo/done/036_train_actor_periodic_snapshot_10prd.md)、対象範囲）。
- ObservationPipelineクラスと専用ファイル・テストの追加 — 範囲外: 固定3段の観測加工はActor本体で満たせ、段の要件が揃う前のクラス化は作り直しになる。再開条件: 段の要件が揃う後続PRDで導入する。（出典: [PRD083](../../memo/083_action_context_retire_10prd.md)、複雑さ監査）。
- 独立した公開NoisyLinearブロック型 — 不採用: 既存LinearとHead最終射影へ対応する形を採る（出典: [PRD084](../../memo/084_noisynet_10prd.md)、§4.1）。
- AMPの実行設定への集約とノイズ込み重みのSN — 範囲外: 精度は用途ごとに異なり、ノイズ込みSNの推定・保持・比較契約も別の暫定PRDで裁定する（出典: [PRD084](../../memo/084_noisynet_10prd.md)、§8.1・§8.2・§14.2）。
- Noisyの有無フラグ・機能登録・実行条件の汎用比較と独立実行基盤 — 不採用: 固有の解釈と検証を具象の計算部品へ閉じる（出典: [PRD084](../../memo/084_noisynet_10prd.md)、§4.2・§14.2）。
- εを引き直す総ノイズ測定と曲率を含む臨界batch測定 — 先送り: 追加forward・専用乱数やヘッセ行列との積が要る。再開条件: 分岐の結果が条件付き値の読みと食い違いNoisy probeのQ差が大きいとき、または単純版とBの効き目が食い違ったとき（出典: [PRD092](../../memo/092_gradient_noise_scale_10prd.md)、§8）。
- PPO/SAC・複数損失・別optimizer向けの拡張口 — 先送り: 単一損失・AdamWの現用途で完結させる。再開条件: 対象Learner・観測したい損失とoptimizerが具体化したとき（出典: [PRD092](../../memo/092_gradient_noise_scale_10prd.md)、§8 5・§10.2）。
- 購読間隔ごとの別EMA — 不採用: 状態と系列が増える。独立した時間幅はrawの利用側での再平滑化に任せる（出典: [PRD092](../../memo/092_gradient_noise_scale_10prd.md)、§10.2）。
- head単位統計とattention entropy最小値 — 先送り: 層平均で足りる間は公開値と保持データを増やさない。再開条件: 層平均が下がったRunで内訳が必要になったとき。最小値は6層構成で読みにくくなったとき（出典: [PRD093](../../memo/093_attention_entropy_diagnostics_10prd.md)、§5・§7）。
- TQLのattention entropy正則化 — 先送り: anetでは本指標で集中の進行をまだ観測していない。再開条件: 層平均の集中と成績の頭打ち・下降が時期として重なったとき。因果は別途判断する（出典: [PRD093](../../memo/093_attention_entropy_diagnostics_10prd.md)、§7 1）。
- 学習側の厳密な優先度の列 — 不採用: EvalPanelには単一networkのforwardしかなくreplayも無いため学習側のonline/targetによる値は得られない（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§6）。
- moduleのbufferにεを置きreset_noiseで引き直す形 — 不採用: shared networkの同時forwardで競合し、snapshotとhard updateで別用途のεが混入する（出典: [ADR 0048](../../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md)、Considered Options）。
- ReplayBufferから別に一様抽出する勾配ノイズprobe — 不採用: 実更新と抽選分布・IS重みが違い、forwardも増える（出典: [ADR 0054](../../adr/0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16.md)、Considered Options）。
- BatchNormの専用検出・禁止 — 不採用: 独立サンプルとして解釈できる条件を文書化し、構成と解釈は利用側に任せる（出典: [PRD092](../../memo/092_gradient_noise_scale_10prd.md)、§10.2）。
- 未成立理由ごとの追加状態指標 — 不採用: 記録から理由を一意に復元できない場合も含め、追加公開契約を増やさない（出典: [PRD092](../../memo/092_gradient_noise_scale_10prd.md)、§10.2）。
- attention entropy専用のfeatureキー — 先送り: 今の構成ではTransformerがfeature閉包内にある。再開条件: Transformerがfeature閉包の外へ置かれたとき（出典: [PRD093](../../memo/093_attention_entropy_diagnostics_10prd.md)、§5）。
- 構築時dummy forwardで層数を数える追加API — 不採用: 黙ったNaNは最初の測定のfail-fastで防げ、APIを1つ増やす必要がない（出典: [PRD093](../../memo/093_attention_entropy_diagnostics_10prd.md)、§5）。
- Actor設定キーでeval_panelだけにhintを生成させる形 — 不採用: 設定が増え、autoの語が今の条件を表すものになり、n-step窓の所有がAgent外へ出る（出典: [ADR 0058](../../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)、本文）。
- NetworkModel が特徴空間・行動価値・スキップ価値の 3 つの Network を持つ形 — 不採用: 切れ目は DAG の branch で表せ、checkpoint の形と Network 1 つ前提の経路へ波及する。補助 Network は primary の公開出力を入力に取り、その target は bootstrap に使わず target 側の Actor の行動にだけ使う（出典: [PRD098 TempoRL](../../memo/098_temporl_skip_policy_10prd.md)、§4 D3、[ADR 0062](../../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md)）。
- 強制行動を Actor へ通知する事後 hook — 先送り: 決定点への入力（999_observation_pipeline の単位 A）で合わせる。再開条件: 単位 A が入ったら、強制された lane のスキップ状態を捨てる（出典: [PRD098 TempoRL](../../memo/098_temporl_skip_policy_10prd.md)、§4 D10・§9）。
- 具象 Agent を anet-core の中のサブフォルダに分けるだけの形 — 不採用: 内部ヘッダ（Head 部品と Replay の生成）への依存が残り、責務の境界が CMake で強制されない。Env と同じ具象別の独立ライブラリ（`src/agents/<agent>/`）にし、`Init<Agent>()` を app が呼ぶ（出典: [PRD999 ソース配置](../../memo/999_source_layout_10prd.md)、§4・§5）。
- DQN 系を家族 1 ライブラリにまとめる形（DefaultDQN と Rainbow を同じライブラリに置く） — 先送り: 移動量は少ないが、Env と同じ「具象 1 つに 1 パス」から外れる。再開条件: グリルで具象別の境界の価値が移動量に見合わないと裁定したとき（出典: [PRD999 ソース配置](../../memo/999_source_layout_10prd.md)、§8 D3）。
- Thompson sampling の τ の区間を `thompson.tau_low/high` で設定する形 — 不採用: 非 spatial の Thompson は τ を U[0,1]（IQN は全範囲）から引き、区間は存在しない。spatial は `spatial.uqe.tau.*` のラダーを使う（出典: [PRD101](../../memo/101_exploration_schedule_profiled_value_10prd.md)、§2.1・§4.2 D9）。
- log ラダーの 0 の端を fail-fast にして 0 に届くラダーを新しい並べ方 `exponential` で書く形、下限をキー（`spatial.log_floor`）にする形、1e-4 への clamp を残す形 — 不採用: 下限 1e-4 で log に並べた列を書いた両端へ線形に伸縮する規則 1 つにし（両端が 1e-4 以上なら恒等）、モードやキーを増やさず、0 と書いて 1e-4 で動く形を残さず、端が 0 へ下がる途中も値を連続にする（出典: [PRD101](../../memo/101_exploration_schedule_profiled_value_10prd.md)、§4.2 D8）。
- MuZero の温度 `temp_start/end/decay_steps` の ProfiledValue 化 — 範囲外: 試作 Agent。再開条件: 本線に戻すときに同じ器で足す（出典: [PRD101](../../memo/101_exploration_schedule_profiled_value_10prd.md)、§3 非ゴール・§7）。
- NoisyNet の雑音の倍率を時間で変える — 先送り: σ は学習するパラメータで、外から掛ける倍率は forward の雑音の作り方に新しい入力が要る。再開条件: ε の形で差が説明できないとき（出典: [PRD101](../../memo/101_exploration_schedule_profiled_value_10prd.md)、§7）。
- step の軸を learn_step やエピソード数にする — 先送り: 探索と β はすべて exp_step で動いている。再開条件: 別の軸が要る例が出たとき（出典: [PRD101](../../memo/101_exploration_schedule_profiled_value_10prd.md)、§7）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- 明示的なActor::Syncは呼出時の強制同期として維持する。shared既定ではaction forward回数と保存形式を保つ（出典: [PRD036](../../memo/done/036_train_actor_periodic_snapshot_10prd.md)、目的）。
- Actorを追加しても既存Actorの行動列を変えず、同一checkpoint・seedの各スロットの結果を確認する（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、7.4・8）。
- 観測のstack→device転送→正規化の並び、乱数の派生・消費順、aux集合を変えない。同seed・config・exp予算のDefaultDQN/LunarLander/stack4でmetrics checksumを比べ、checkpointのraw checksumはゲートにしない（出典: [PRD083](../../memo/083_action_context_retire_10prd.md)、Context・確定した設計判断2〜4・11）。
- Agent のライブラリは anet-core の公開ヘッダと自分のライブラリだけに依存し、anet-core は Agent のライブラリに依存しない。ソースを移すときは 4 env の metrics checksum の一致と全テストで挙動不変を確かめ、旧パスの互換層（symlink・転送ヘッダ）を置かない（出典: [PRD999 ソース配置](../../memo/999_source_layout_10prd.md)、§4 8・§6）。
- 既存のHead/SN FP32保護と用途別AMP精度を維持する。Noisy無効の既存チェーンは決定論backendでmetrics checksumを比較し、sample時の乱数列の割当変更を無効側へ持ち込まない。比較の相手は実装前に保存した基準の実行体・依存DLL・出力とする（PRD060は基準を取り損ねた）（出典: [PRD084](../../memo/084_noisynet_10prd.md)、§8.2・§12・§13）。
- 診断ON/OFFで同seed・決定的モードの学習系列を比較し、学習の.grad・optimizer state・RNGを変えない。費用は反復A/Bで実測しprofiler合計でend-to-end増加率を代用しない（出典: [PRD092](../../memo/092_gradient_noise_scale_10prd.md)、§7.1・§7.2）。
- captureへ移す既存branch出力の値とdetachを保ち、captureなしforwardは不変とする。同seed・決定論設定の診断ON/OFFで学習由来metrics checksumとRNG・buffer等への非干渉を確認する（出典: [PRD093](../../memo/093_attention_entropy_diagnostics_10prd.md)、§6 1・5）。
- ヒント計算の共有化前後でactor_approxの同seed・決定的Runの学習metrics checksumを一致させ、既定のEvalPanel networkとRunnerカウンタの通知前後の時機を変えない（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§4.2 D6/D7・§8 3）。
- スキップ中も全 lane で forward し、行動だけを上書きする（全 lane の q_values を前提にする step 診断・Actor Q ヒント・spatial・Noisy・stacker・Push の契約を保つ）（出典: [PRD098 TempoRL](../../memo/098_temporl_skip_policy_10prd.md)、§4 D2・§5.5）。
- 探索のキーを ProfiledValue へ移すとき、既存の log ラダー（0.4〜0.01、0.05〜1e-4）と非 spatial の ε・τ の値を変えず、Atari・DropMerge（最後の lane を 1e-4 と書いた検査用設定）・GridMaze・LunarLander の短い決定的 Run の metrics checksum を実装前に保存した実行体と比べる。評価の時機は Actor が `UpdateSchedule` の `exp_step`、Learner が update の先頭の `exp_step`。PER β だけは今の時機（前回の更新の終わりに次回分を評価、初期値は step 0 の値）を保つ（出典: [PRD101](../../memo/101_exploration_schedule_profiled_value_10prd.md)、§4.2 D7・D8・§4.4 D18・§6.1 1）。
- Munchausen の α・entropy_tau を時間で変えるときは、Actor の Q ヒント・Evaluation Trace の step 診断・Learner の target が同じ step で同じ値を使うことを確かめる（出典: [PRD101](../../memo/101_exploration_schedule_profiled_value_10prd.md)、§4.4 D19・§6.2 5）。

## 関係

- 設計: [Agentと学習](../../design/110_agents_and_learning.jp.md)、§2.1・§2.3・§3。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [agent.hpp](../../../core/anet-core/include/anet/agent.hpp)
- [agent.cpp](../../../core/anet-core/src/agent.cpp)
- [dqn_based_test.cpp](../../../core/anet-core/src/dqn_based_test.cpp)
- [dqn_based_agent.cpp](../../../core/anet-core/src/dqn_based_agent.cpp) / [dqn_based_agent.hpp](../../../core/anet-core/src/dqn_based_agent.hpp) / [dqn_based_agent_test.cpp](../../../core/anet-core/src/dqn_based_agent_test.cpp)
- [dqn_based_heads.cpp](../../../core/anet-core/src/dqn_based_heads.cpp) / [dqn_based_heads.hpp](../../../core/anet-core/src/dqn_based_heads.hpp)
- [dqn_munchausen_test.cpp](../../../core/anet-core/src/dqn_munchausen_test.cpp)
- [muzero_based_agent.cpp](../../../core/anet-core/src/muzero_based_agent.cpp) / [muzero_based_agent.hpp](../../../core/anet-core/src/muzero_based_agent.hpp)

## PRD の一覧

点検が作り直す。手で直さない。

- [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [PRD035](../../memo/done/035_approx_actor_priority_per_10prd.md) — 追加 forward なしの近似 Actor PER 初期優先度 PRD
- [PRD036](../../memo/done/036_train_actor_periodic_snapshot_10prd.md) — DefaultDQN Train Actor定期network snapshot PRD
- [PRD057](../../memo/done/057_native_stack_dim_handling_10prd.md) — Stack 次元のネイティブ対応(Reshape ハックを廃する根本対応)
- [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [PRD062](../../memo/done/062_plasticity_metrics_10prd.md) — 可塑性メトリクス（srank / dormant unit 率）PRD
- [PRD063](../../memo/done/063_plasticity_weight_norm_10prd.md) — 可塑性 weight norm（feature/readout 分割）PRD
- [PRD065](../../memo/done/065_nn_spectral_norm_10prd.md) — NN Spectral Normalization（weight_norm.mode）PRD
- [PRD066](../../memo/done/066_policy_churn_metrics_10prd.md) — Policy churn メトリクス（35_agent_churn）PRD
- [PRD067](../../memo/done/067_MunchausenRL_10prd.md) — Munchausen RL（M-DQN / M-QR / M-IQN）PRD
- [PRD070](../../memo/070_device_spec_auto_10prd.md) — PRD 070: device 指定の auto 対応とキー体系統一
- [PRD073](../../memo/done/073_replay_fit_metrics_10prd.md) — PRD 073: replay 抽選履歴別の TD・損失メトリクス
- [PRD078](../../memo/done/078_replay_frame_history_integrity_10prd.md) — ReplayBuffer の観測履歴・bootstrap stack 整合性修復 PRD
- [PRD083](../../memo/083_action_context_retire_10prd.md) — ActionContext の廃止（Actor が RNG と frame stacker を直接持つ）
- [PRD084](../../memo/084_noisynet_10prd.md) — NoisyNet (Noisy Networks for Exploration) PRD
- [PRD092](../../memo/092_gradient_noise_scale_10prd.md) — 勾配ノイズの大きさ（gradient noise scale）の計測 PRD
- [PRD093](../../memo/093_attention_entropy_diagnostics_10prd.md) — TransformerEncoder の attention エントロピー診断 PRD
- [PRD094](../../memo/094_evalpanel_step_history_10prd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD

## 台帳の指摘

台帳のツールが作り直す。手で直さない。

<!-- registry:begin -->
- F-0013 P2 refactor — 名前→factoryのRepositoryと自動登録マクロを4か所で別々に実装し、重複登録の扱いが食い違う
- F-0014 P2 refactor — TD・QR・IQNのUpdateFromSamplesが同じ骨格（正規化→損失→PER重み→PER更新→最適化）を丸写ししている
- F-0019 P2 refactor — BatchUpdateResult::GetScalarがヘッダ内のinlineで234行、keyの比較43個のif連鎖
- F-0029 P2 defect — use_td_clipが損失の前にTDをclampするため、clipを超えたサンプルの勾配が0になる
- F-0021 P3 refactor — dqn_based_agent.hppにインクルードガードが無く、先頭のファイル名のコメントが古い

[台帳の一覧](../../../reports/registry/findings.md)
<!-- registry:end -->

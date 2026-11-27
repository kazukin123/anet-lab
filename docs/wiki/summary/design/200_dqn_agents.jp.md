---
title: DQN系Agent
page_kind: サマリ
material: "[[docs/design/200_dqn_agents.jp.md|200_dqn_agents.jp]]"
material_kind: 設計文書
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/methods/noisynet.md|NoisyNetの観測]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
updated: 2026-11-27
---
# DQN系Agent

## 元の材料

[DQN系Agent](../../../design/200_dqn_agents.jp.md) blob:154228c534d96b340bdf4f2cddd95f1ad8ca52d2

## 要点

- DefaultDQN と Rainbow は Agent・Actor・Learner の契約を共有し、学習用 Network と目標値用 Network を使う。
- 方策が Q 値や分位点から行動を選び、行動時の情報を経験と学習・診断へ渡す。
- ReplayBuffer から N-step の Sample を引き、設定した更新方法に応じた目標値と損失を計算する。
- DefaultDQN には Noisy、スキップ方策、Munchausen、外的・内的価値の分解などがあり、適用範囲と組み合わせが異なる。
- Actor が使用するモデルの同期、checkpoint に保存する状態、診断を計算するタイミングを区別する。

## 内容

### 2. DQN系の概要と基本概念

- 非 spatial の ThompsonSampling 方策で QR を使う場合は、呼出しごと・laneごとにτを`[0,1]`から一様に1つ引き、同じτを全行動へ使う。`uqe.use_tail_mean=false`ではその分位、`true`ではその分位から上側の平均で比較する。
- Double DQN、N-step、PER、勾配clip、AMPなどはLearner設定で組み合わせる。すべての組合せを「Rainbow」という名称だけから暗黙に仮定せず、Runの解決済み設定を確認する。
- DefaultDQNのActorは、観測をdeviceへ移し、CPU上の`episode_start`を加えて行動選択用Forwarderに渡す。得たPerceptionをPolicy呼出しと補助情報の付与に使う。RainbowのActorでは観測をdevice転送してからPolicyを呼ぶ。PolicyはLearnerへ依存せず、NetworkとRNGでActionを決定する。
- PERが有効で初期優先度の方式が`actor_approx`の場合、カタログ項目の名前や用途にかかわらず、Agentが作るすべてのActorは既存forwardからDQN固有の`float32[B,5]` hintを作る。列は選んだ行動の score、状態価値（通常は最大 score、Munchausen 時は soft value）、実空間の Munchausen bonus、内的 Q_sa、内的 V_s の順。combined では末尾2列が0で、分解時は先頭3列に外的 Head の値を使う。
- 内側の Learner は、成分ごとの割引率の設定が定数か schedule かで `needs_return_recompute` を決め、schedule の成分だけ抽選時に収益を作り直す。QR・IQN の PER に書き戻す優先度は、current と target の分位点の平均の差の絶対値に `per_eps` を足した値から作る（§2.5）。

### 3. コンポーネント定義

- `DefaultDQNAgent` — scaler、target用Policy、Train Actor snapshotなどを組み合わせる設定可能なDQN系Agent。
- `IntrinsicRewardCoordinator` — DefaultDQNのprivateな入れ子。内的報酬の採点、β、補助学習、診断、保存・読込、Evaluation Traceの採点と写しをまとめる。
- `RainbowAgent` — DQN 系の共通部品の責務分担を確かめる2つ目の Agent。Run では使わない。
- `DqnComponentAssembler` — 具象 Agent の設定型を受け取らず、Head・方策・Actor・Learner の生成と値の振り分けを共通化する。
- `RuntimeVars` — `learn_step`、PER betaなどDQN学習中に変化するState。
- `NetworkModel` — online/target Network、target更新、保存・読込をまとめるResource。
- `SpatialLadderSchedule` — 両端の時間の値と並べ方を所有し、laneのTensorと統計を保持する。
- `TDLearner` — 通常Qのとき選ばれる具象型。上書きを持たず、ReplayBufferとoptimizerを準備する。target・誤差・forward・TBOの変換は基底の`dqn::Learner`が担当する。
- `GenerateTaus` — IQNの5種のtau配置方式に従い、指定のdevice上にτを生成する自由関数。固定の範囲を受ける形と、laneごとの下限を受ける形がある。

### 5. 静的構造

- `LearnerInterface`は共通`anet::rl::Learner`、`DqnLearner`は内部`anet::rl::dqn::Learner`を表す。外側AgentはRunから見えるLearner facadeであり、共有mutexを取得してから内側Learnerを呼ぶ。

### 6. 主要フロー

- 共有Networkを使うActorはLearner更新との競合を避けるためshared lock内でPolicyを呼ぶ。clone Actorはprivate Networkを使い、forward中にsource Networkを参照しない。
- DefaultDQNは外側AgentでRewardをscaleし、ObservationNormalizerの統計を更新してから生Observationとscale済みRewardを内側へ渡す。Rainbowはこの前処理を持たず、同じinner Learner contractへ直接委譲する。
- Munchausen 有効時の Learner の target 計算では、hard action 選択を呼ばない。IQNはcurrent N本、target M本、必要ならfresh online N本の順にtauを生成する。target modeのbonusはM本、他modeはN本を使う。Munchausen 無効時の forward 順とRNG消費は維持する。target modeのplasticity captureはskip無効時`[2B,F]`、有効時`[3B,F]`を検証し、通常nextに対応するB〜2B行を返す。
- `Sync()`はstepを受け取らないため、強制同期後に最初に呼ばれたactionの`train_step`をage 0の基準にする。shared Actorと`sync_interval.*`未宣言のActorは定期snapshotを持たない。
- `learner.skip.sample_weight=uniform`が既定で、`kmax`では行ごとのskip lossへ`max_lengths / mean(max_lengths)`を掛ける。重みの和はBで、IS重みと積にする。TD値自体は変えない。PERは`|δQ| + learner.skip.priority_lambda * |δskip| + ε`で、係数の既定は0。同じIS重みを両lossへ掛ける。

### 7. DefaultDQNとRainbowの構成・設定

- Policy — `DefaultDQNAgent`はActorカタログとLearner targetを個別構成。epsilon-greedy、UQE、Thompson Sampling。`RainbowAgent`はActorごとのepsilon-greedyとLearner target用greedy。
- `intrinsic.enabled = false` では RND の Network・optimizer・統計を生成しない。分解と内的報酬スケーラの認識した設定値は無効時も型・列挙・値域を検証する。分解や内的スケーラの設定で、現行の診断対象に当たる未使用の明示的な非既定値は対象ごとに 1 回 WARN する。既知の診断キーは無効時も保持して NaN を返す。DQN は `intrinsic.hpp` の Factory を使う。
- IQN専用lossはcurrent側Nをsum、target側Mをmeanし、Huber項を`kappa`で除算する。`N = 1`では分散の不偏推定を使わず、`q_std`を明示的に0とする。分位数を持たない`quantile_mode=none`では`q_std`が成立しないため`NaN`を返し、0へ偽装しない。
- 通常/Fused AdamW は Noisy ON のとき σ 専用 group を使い、`learner.noisy_weight_decay=0` が既定。OFF は従来の単一 group と順序を維持する。μ-only の current は σ の gradient を作らず、σ を更新・decay しない。sample で成立した σ gradient は既存の全体 norm/clip に含まれる。checkpoint と snapshot の parameter に σ を含め、ε・時計・RNG は保存しない。checkpoint 引き継ぎ時は σ 専用 group にも現在の optimizer 設定を適用し、weight_decay は learner.noisy_weight_decay で決める。

### 8. lifetime・同期・保存読込

- 外側Agentが`RuntimeVars`、`NetworkModel`、Policy、inner Learner、scaler類、`IntrinsicRewardCoordinator`のlifetimeを所有する。 内側 Learner は、Agent の基の seed と用途ごとの名前から5種類の probe RNG を構築して所有し、購読の再設定でも保持する。
- outer Agentの共有mutexはLearner更新、shared Actor forward、clone/sync copyの境界を直列化する。
- `split`のoptimizerは通常の4役割に`primary_skip`を加え、追加optimizerのmoment・stepを保存する。4役割のcheckpointからsplitへは追加状態を初期化して一度WARNし、5役割からstop／mergeへの読込は拒否する。復元後は追加optimizerにも現在のlr・eps・betasを適用し、weight decayは0に保つ。stop／mergeの保存形式は変更しない。
- `RainbowAgent`は現行コードで独自の`Save()` / `Load()`をoverrideせず、基底Agentのno-opを使う。DefaultDQNと同じcheckpoint対応があると仮定しない。
- NetworkのSN u/vはnamed bufferなので、DefaultDQN checkpointのonline/target Networkとともに保存・復元される。`Clone()`は構築seedを再利用して同じ構造を再構築してから、parameterとbufferを完全copyする。

### 9. 可観測性・性能・エラー

- step診断の`intrinsic.error`・`intrinsic.normalized_error`・`intrinsic.bonus`は群56の集約値とは別に、評価の遷移ごとの値を保持する。
- DefaultDQNは`train_actor_snapshot_interval`と`train_actor_snapshot_age`をActionInfoから公開する。定期snapshotがないActorでは両値を`NaN`とし、同期したactionのageは0になる。
- shared Actorは追加Networkを持たないがLearnerのunique lockと競合する。clone Actorは追加memoryと同期copyを使う代わりにforwardを分離する。
- shared ActorのdeviceがAgent deviceと異なる場合は、cloneを有効にするか同じdeviceを指定する。
- DefaultDQN の方策変化を測る `35_agent_churn` の Q 由来指標が、その `learn_step` で記録対象になった場合だけ、caller-owned `policy_churn_probe` RNGで完全なprobe batchを1回取得し、既存Observation正規化を適用する。件数は`learner.policy_churn.probe.batch_size`で指定する（既定1024）。
- `metrics.scalar.@replay_fit`を明示選択すると、`46_agent_replay_fit`の13指標を既定503 learner updatesごとに観測する。既定のprofileには含めない。各行のIntervalGateで当該更新の要求を合成し、母数だけ、片群だけ、TDだけ、損失だけ、実PERだけの要求を分ける。PER無効時の実PER平均と選択比はNaNで、それらだけの要求では群も評価しない。

### 10. テストと拡張時の確認事項

- 価値の分解と報酬スケーラを変えるときは、RFD0121検証資材の数値検査、checkpoint判定表、既定5構成の前後比較とAtari性能測定を確認する。
- 構築・設定読み込み・内的報酬・診断の責務を変えるときは、RFD0128検証資材の34構成の前後比較、旧checkpointの継続、Releaseの速度、指標の並びの破壊検査を確認する。
- TD・QR・IQN の更新の進行、Munchausen の評価、Bellmanターゲットの組み立てを変えるときは、RFD0125 検証資材の基準保存、31 組と BF16 の前後比較、target の式の数値検査、Release の性能比較を確認する。
- skipの更新方式・重み・session指標・構造の群を変えるときはRFD0103検証資材の数値、checkpoint、設定・Run比較を確認する。
- TempoRL の window・Actor・learner・checkpoint・指標を変える場合は、RFD0098検証資材 の契約テスト、4環境の無効時等価性、smoke、Atari性能比較を実行する。長時間学習の合否は短い検証とは分けて記録する。
- 外側Agent、共通DQN component、ReplayBufferのどのlayerへ責務を置くかを明示する。
- outer Agentが`CreateLearner()`で自身を返す場合、mutex取得と前後処理をinner Learner呼出しの外側へ保つ。
- Actorカタログ項目ごとのPolicy、source Network、shared/clone、`Sync()`契約をtestする。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [NoisyNetの観測](../../concepts/methods/noisynet.md)
- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
- [行動の時間尺度](../../concepts/methods/temporal_abstraction.md)

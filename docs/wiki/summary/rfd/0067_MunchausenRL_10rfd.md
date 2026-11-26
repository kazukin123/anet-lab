# Munchausen RL（M-DQN / M-QR / M-IQN）PRD

## 元の材料

[Munchausen RL（M-DQN / M-QR / M-IQN）PRD](../../../memo/done/0067_MunchausenRL_10rfd.md) blob:8c84f6ca82c4793d24cdd3a54dfc90387e72176a

## 要点

- Munchausen RL は自身の方策の scaled log probability を報酬側へ加え、次状態で soft な価値を使う。
- TD、QR、IQN の3種の Learner に同じ有効化契約を用意し、既定は無効にする。
- 論文の1-step 式と既存の n-step ターゲットの違いを整理し、報酬と bootstrap の扱いを決める。
- 近似 Actor 初期優先度を保つため、行動選択時の Q ヒントも扱う。
- 楽観的ターゲットでのリスク評価を含め、無効条件の数値と既存の挙動が一致することを確認する。

## 内容

### 数理契約

`scaled_log_policy <= 0` である。`tau` はIQNのtaus、target network更新率、gradient clipなどのtauとは別概念である。
分位点ごとに別の方策を作らない。soft価値側のscaled log-policyはclipしない。
終端maskはbootstrapだけへ適用する。TBO有効時はscalar Qまたは各分位点へ個別に `h^-1` を適用してから平均・方策・bonus・soft価値を求め、target完成後にだけ `h` を適用する。非線形なので `h^-1(mean(Z))` で代用しない。
TD Learner（`quantile_mode=none`）では `target_policy=UQE` が既存検証でfail-fastするため、soft楽観ターゲットは分位点表現でだけ成立する。

### 実装契約

既存のcurrent online forwardとtarget forwardを従来の順で完了する。
Actorは行動ごとのMunchausen項を一時auxとして保持する。`DQNActionInfo::WithAction` は行動差し替え時に `q_sa` と `munchausen_term` を再gatherし、`state_value` を維持する。hintがあるのに必要なauxが欠けていれば契約違反とする。
`enabled=false` のLearnerはMunchausen分岐へ入らず、数値経路とRNG消費を実装前から完全に変えない。

### テスト契約

productionへtest-only APIを追加せず、既存のforward-count probe、network mode観測、TauEcho fixtureを利用する。
diagnostics readbackがPER ON/OFFの両方で正しいoffsetと5値を返すことを確認する。PER OFFかつMunchausen diagnosticsだけが定義された場合も早期returnせず、pendingが有効になることを確認する。
`DqnInitialPriorityEstimator` がMunchausen込みtargetをTBO ON/OFFで再現し、OFF時の初期優先度数値が従来と一致することを検証する。
`target_policy` がGreedy / EpsilonGreedyのとき、Munchausen ONのtargetが平均スコア経路と同値であることを確認する。

### 現行実装で確認済みの前提

`ForwardTarget` はtarget側branch captureを常に同じforwardへ渡す。`target` modeの2B forwardをそのまま使うとcaptureも2Bになるため、PRD 062の既存意味を保つには後半Bへのnarrowが必要である。
BTRの非IQN経路はtarget current、IQN経路はfresh online currentでbonusを計算し、集約済みN-step returnへbonusを一度加え `gamma^n` bootstrapを使う。BTRに `online_reuse` の前例があるとは主張しない。
`target_policy` の既定はGreedyで、Atariの現行構成はtrain/evalともEpsilonGreedy・`use_optimistic_target=false` である。soft楽観ターゲットが実際に効くのはtrain_policy=UQEのDropMerge / LunarLander / GridMaze系構成である。

### リスクと制約

`target` はtarget forwardを2Bへ増やし、`online` は追加online forwardとIQN tau RNGを増やす。Learner律速のAtariではwall-clockへ現れる可能性があるため、合否ではなくmode別に記録する。
soft経路のnextスコアはM本の経験分位で、Atariの `M=8` では粗い。`fixed` だけが乱数非依存であり、`random` / `stratified` / `systematic` / `antithetic` は乱数を消費する。
同じ `target_policy.uqe_use_tail_mean` がhard / soft両経路を支配する。hard経路のTBO都合で `false` にした設定はsoft経路でも1点UQEになる。
`use_optimistic_target=true` はfail-fastしないため、env側A層に残った `true` が黙ってsoft楽観ターゲットになる。config dumpと初期化ログのスコア源で確認する。

### Complexity Audit

`MakeRiskBiasedScore` の抽出。切るとQRのhard経路とsoft経路でスコア定義が2箇所に複製され乖離する。
Munchausen ON + Thompson targetの構成時fail-fast。平均へ落とすとhard（サンプルごと乱択tauのrisk選択）とsoftで `target_policy` の意味が割れるため、soft版が未定義であることを明示する。
OFF保証を、Learner数値経路・RNG、標準Atari、actor_approxの優先度数値同値へ限定する。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [報酬尺度と割引率](../../concepts/methods/reward_scale_discount.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)

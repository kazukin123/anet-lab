---
title: 追加 forward なしの近似 Actor PER 初期優先度 PRD
page_kind: サマリ
material: "[[docs/memo/done/0035_approx_actor_priority_per_10rfd.md|0035_approx_actor_priority_per_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
updated: 2026-11-26
---
# 追加 forward なしの近似 Actor PER 初期優先度 PRD

## 元の材料

[追加 forward なしの近似 Actor PER 初期優先度 PRD](../../../memo/done/0035_approx_actor_priority_per_10rfd.md) blob:c3089243802f69efb2e1e7afc4625de6925e5400

## 要点

- PER の固定初期優先度が高いと新規遷移が抽選を占め、低いと未抽選のまま追い出されやすかった。
- 最大優先度で初期化する方式でも、新規遷移への優先度の集中は解消しなかった。
- Actor が行動選択の forward から得た情報をヒントとして渡し、追加の forward を行わず近似初期優先度を求める。
- 報酬・n-step 収益と終端条件がそろう段階でヒントを用い、Learner の実測優先度で更新する。
- 既定は fixed とし、ヒント欠損・終端・後続の優先度更新をそれぞれ検証する。

## 内容

### 問題

DropMerge 系の実行では、PER の新規遷移に固定 `per_initial_priority` を与えている。固定値が高いと、Learner 未更新の新規遷移がリプレイ優先度の質量を支配する。
固定値が低いと、一度もサンプリングされないまま追い出されやすい。実測でも、`replay_ratio = 1.0` では `per_sample_initial_ratio` と `last_evicted_never_sampled_ratio` の間に強い収支がある。
一般的なPER実装の既定である最大優先度初期化（現行実装では負の`per_initial_priority`で選択できる）も試した。しかし新規遷移が常にその時点の最大優先度で入るため、初期優先度の質量支配は固定高値と同等かそれ以上に強く、実測でも問題は解消しなかった。
行動選択時点では、報酬、n-step収益、ブートストラップ状態が確定していない。そのため、Actorが最終優先度まで計算する責務を持つのは不自然である。

### 解決方針

Actorは最終優先度を計算しない。既存の行動推論から得られる最小限のアルゴリズム固有payloadを、`BatchActionInfo`へ追加するoptionalな`ReplayInitialPriorityHint`に載せる。
TrainerとLearnerは、このメタデータを通常の`BatchExperience`と同じ経路で`ReplayBuffer`へ運ぶ。アルゴリズム固有の統合処理は行わない。
DQN Agentが注入する推定器は、Learnerと同じ割引率、TBO変換、PER epsilonを使って近似TD誤差を計算する。報酬スケールはAgentがPush前に適用済みで、保存済みtarget returnへ反映されているため、推定器では扱わない。truncated遷移は開始hintを検証したうえで、終端観測由来のbootstrap hintが存在しないため最大優先度初期化へフォールバックする。契約違反はバグとしてエラーで停止する。
優先度専用forward、Actor用target network、前後stepを結合するTrainer固有処理は追加しない。

### ユーザーストーリー

強化学習の実験者として、近似Actor初期優先度のためにネットワークforwardを増やしたくない。GPU時間をLearner更新に使うためである。
強化学習の実験者として、固定・最大・近似Actorの初期化方式を明示的に切り替えたい。同条件の実行を比較するためである。
強化学習の実験者として、仕様上bootstrap hintを利用できないtruncated遷移には最大優先度初期化を使いたい。有効化によって遷移が暗黙にリプレイ対象外にならないためである。
強化学習の実験者として、サンプリング後はLearnerのTD誤差を最終権威にしたい。Actorの近似情報を初回サンプリング前だけに限定するためである。

### 実装上の決定

`ActionPolicyConfig`へ重複するユーザー設定を追加しない。Agent構築時にLearnerのモードからReplay初期優先度ヒント出力の要否を決め、学習用Actorへ明示的に配線する。
`max`と`actor_approx`はPERを前提とする。`use_per = false`との組み合わせは、両設定と修正方法を示してエラーにする。
`actor_approx`でActor推定値を利用できない遷移のフォールバック初期化は、`max`モードと同一の最大優先度初期化とする。
Replay初期優先度ヒントを生成するのは`actor_approx`が有効な学習Actorだけとする。評価ActorとLearner内部のtarget action選択、ならびに`fixed`/`max`モードではヒントを生成しない。

### 設定

追加・変更する設定項目を示す。キーはDQN系AgentのLearner設定配下（例: `DefaultDQNAgent.learner.per_initial_priority_mode`）である。
`use_per = false`かつ`per_initial_priority_mode`が`fixed`以外の組み合わせは、両設定と修正方法を示してエラーにする。
`per_initial_priority`の負値は全モードで、`per_initial_priority_mode = max`への移行方法を示してエラーにする。NaN/Infも全モードでエラーにする。`fixed`以外の非負finite指定だけを、A/B比較のためエラーやWARNなしで無視する。

### テスト方針

外部から観測できる動作と安定したインターフェースを中心にテストし、production APIへtest-only経路を追加しない。任意leaf設定が必要な低水準テストは内部の物理slot APIを直接テストし、外部`UpdatePriorities`へraw indexを再導入しない。
既存のReplayBuffer、DQN/QR Learner、TBO、転送、先読み、Trainerの回帰テストを実行する。
実験採用時は、基準モード（`fixed`または`max`）とActor近似モードを同じ`exp_step`区間で比較する。3%を超えるスループット低下、未サンプル追い出し悪化、弱い優先度相関、終盤報酬低下を別々の棄却信号とする。

### 対象外

Actor側target network、Learner側target-network評価の正確な再現、正確なUQE/楽観的target選択（判断の記録と再訪条件はADR 0010）。
Action Masking連動のstate value補正。`max_a Q_online`が無効行動のQを拾い得る問題は、Action Masking基盤の導入時に再訪する。

## 主要な概念

- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [DropMerge](../../concepts/subjects/dropmerge.md)

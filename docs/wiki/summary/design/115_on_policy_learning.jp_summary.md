---
title: 方策ベースの学習の共通部品
page_kind: サマリ
material: "[[docs/design/115_on_policy_learning.jp.md|115_on_policy_learning.jp]]"
material_kind: 設計文書
concepts:
  - "[[docs/wiki/concepts/framework/326_ppo.md|326 Agent個別実装 PPO系列]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
updated: 2026-11-28
---
# 方策ベースの学習の共通部品

## 元の材料

[方策ベースの学習の共通部品](../../../design/115_on_policy_learning.jp.md) blob:06589448c3cd8d992a1794454f373605a39d30a7

## 要点

- 方策を直接学ぶ Agent が使う core の共通部品として、離散行動の分布 `CategoricalDistribution`、advantage と value の目標を GAE で作る `ComputeGae`、複数の Head を 1 つの Network に並べる `CompositeHeadFactory`、ためた経験の全体を読む `LaneTimeTensorStore::GetAll` の 4 つを扱う。
- 部品は特定の Agent の名前を持たず、`anet-core-base` の公開ヘッダから使う。計算だけを受け持ち、損失の式、rollout をいつ更新に使うか、optimizer は使う側の Agent が持つ。
- 部品は入力と同じ device で計算して結果も同じ device に返し、値を CPU へ読み戻さない。設定キーも読まず、設定の読み取りと値域の検証は使う側の Config が行う。
- `ComputeGae` は終端（`terminated`）と episode の終わり（`episode_end`）の印を分けて受け取り、打ち切りで終わった episode でも次の観測の value で先を見積もれるようにする。次の観測の value をどこから取るかは使う側が決める。
- `CompositeHeadFactory` では、中の Head が受け取る Body の出力のキーをどの branch へ向けるかの設定だけで、Body を共有するか分けるかが決まり、コードは変わらない。
- 不正な形・型・値域の入力は、構築か計算の前で止まる。
- 確かめは `anet-core-base-test` の 4 つのタグで行い、device を受けるテストは CPU と CUDA の両方で走る。

## 内容

### 2. 行動の分布

- `CategoricalDistribution` は行（lane やミニバッチの 1 件）ごとの logits `[行, 行動数]` から作り、`Sample`（logits と同じ device の生成器を渡す）、`GetMode`、`ComputeLogProb`、`ComputeEntropy`、全行動の確率と log 確率を返す。
- logits は float32 にしてから `log_softmax` で正規化し、確率 0 の行動の entropy の項は 0 と数える。勾配が要らない呼び出しで勾配を切るのは使う側である。

### 3. advantage の推定

- 入力は `[時刻, lane]` で時刻 0 が最も古い。`delta[t] = rewards[t] + γ·next_values[t]·(1 − terminated[t]) − values[t]`、`advantage[t] = delta[t] + γ·λ·(1 − episode_end[t])·advantage[t+1]`、`value_target[t] = advantage[t] + values[t]`。結果は入力と同じ形で、勾配を作らない。
- 打ち切りで先を見積もるときは `terminated` を偽、`episode_end` を真にし、`next_values` に Reset の前の観測の value を入れる。終端と同じに扱うなら両方を真にする。

### 4. Head の合成

- 組（`CompositeHeadEntry`）は `name`（parameter は `head.<name>.…`）、`feature_key`（`net.body.output.[キー]`）、`factory` を持ち、作った Head は組ごとの出力を 1 つの `TensorDict` にまとめる。実行器・Noisy の parameter・図の情報は中の Head のものを順に集める。
- 組の不備と名前の重複は factory の構築で、`feature_key` が Body の出力に無いことと出力のキーの重なりは Network の組み立て（`CreateHead`）で止まる。

### 5. ためた経験の全体の読み出し

- `GetAll` はたまっている範囲の全体をキーごとに `[ためた時刻の数, lane 数, …]` で、複製を作らずに返し、空の器では止まる。rollout の学習は器が埋まったら `GetAll` で読み、更新のあとに `Clear` で空にする。

### 7. テスト

- タグ `[distribution]`・`[advantage]`・`[composite_head]`・`[tensor_store]` で、手計算との一致、終端と episode の終わり、λ = 0・1、不正な入力で止まることを確かめる。

## 主要な概念

- [326 Agent個別実装 PPO系列](../../concepts/framework/326_ppo.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)

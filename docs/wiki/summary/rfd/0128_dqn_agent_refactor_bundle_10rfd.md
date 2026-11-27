---
title: 価値分解に先立つ DQN 系 Agent の整理
page_kind: サマリ
material: "[[docs/memo/0128_dqn_agent_refactor_bundle_10rfd.md|0128_dqn_agent_refactor_bundle_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
updated: 2026-11-26
---
# 価値分解に先立つ DQN 系 Agent の整理

## 元の材料

[RFD0128　価値分解に先立つ DQN 系 Agent の整理](../../../memo/0128_dqn_agent_refactor_bundle_10rfd.md) blob:6b8f84ee136c35be22b7563f409a7662919364d4

## 要点

- 外的・内的価値の分解で触れる箇所に集中していた五つの課題を先に整理する。
- 内的報酬の状態と採点、保存・読み込みに散っていた扱いをまとめる。
- 標本フィールドの順序と、DefaultDQN と Rainbow の Head・Actor・Learner 構築の共通部分を整理する。
- 実装前の実行体と設定を保存し、各課題を反映した区切りで前後比較を行う。
- メトリクス、設定記録、checkpoint 形式、乱数消費順を含めて学習挙動の等価性を調べる。

## 内容

### 目的と問題

2026-11-10〜11 のリファクタリング候補の抽出で、DQN 系の Agent まわりに、重複と散らばりが 5 件見つかった。どれも、価値の分解が変更を足す場所にある。

### 起草時点の現状

lane の抽選の乱数だけ標準ライブラリの `std::mt19937_64` で、同じクラスのほかの乱数は `anet::RandomGenerator` である。
`replay_item_keys`・`per_priority_sources` を CPU に置く書き方が違う。
Rainbow の値の振り分けは、ReplayBuffer の接頭辞を `"replaybuffer."` と直書きし、DefaultDQN は `ReplayBuffer::kKeyPrefix` を使う。
RFD0182（ソースの配置）は暫定・未着手で、DQN 系の共通部品を `dqn_based` のライブラリに、DefaultDQN と Rainbow をそれぞれ別のライブラリに置く案を持つ。

### 規約と過去の判断

学習器や Agent を変えるときは、乱数の派生と消費の順を変えず、同 seed・決定的モードの短い Run のメトリクスを、実装の前に保存した実行体と比べる（324 DQN の「変えるときの条件」）。
短い Run の前後比較は、RFD0122 と RFD0125 が同じファイルで行った。変更前の実行体と設定を保存し、seed 1・決定的モードで流して、時間と資源の tag を除く全 scalar の点を比べる（RFD0125 の検証資材）。

### 確定事項

方策と τ の検証の置き場。 `ValidateActionPolicy`・`ValidateTauRule` を、検証する型と同じ DQN 系の共通のヘッダーへ移す。中身・エラー文・呼ぶ時機（各 Agent の設定の読み取りの中）は変えない。
`ExperienceSamples::To` の未定義の tensor。 CPU の経路も CUDA の経路も、未定義の tensor をそのまま返す。
`ExperienceSamples::ToString`。 転送と同じフィールドの並びから作り、今は出していない `window` の 6 つと `sampled_rewards`・`sampled_window_rewards` も出す。
lane の抽選の乱数の種類。 変えない。

### 実装契約

Agent の値の振り分けは、鍵を引く順を変えない（DefaultDQN は内的報酬、`target_policy.`、schedule の鍵、`per_beta`、probe と noisy と ReplayBuffer、報酬スケーラ、観測の正規化の順）。
Head の選択の 6 通り（分布の表し方 3 種 × Dueling の有無）と、Learner の 3 種の選択。選ばれた Head と Learner の種類と、ログの文言。
学習を止めた構成（`learner.enabled = false`）で、診断の 6 群と `intrinsic.` の鍵が、それぞれ NaN を返すか未知の鍵になるかと、WARN の回数。
条件は RFD0125 の検証資材 の短縮条件のまま（seed 1、8 lane、batch 16、warmup 256、replay ratio 1、8,704 experience、評価停止、決定的 backend、FP32。GridMaze・LunarLander は CPU、Atari qbert・DropMerge は CUDA）。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)

---
title: TempoRL の Q 側・skip 側のノイズと τ を分けた評価
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-10-29_temporl-skip-forward-stage0.md|default-dqn/atari/2026-10-29_temporl-skip-forward-stage0]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
  - "[[docs/wiki/concepts/methods/noisynet.md|NoisyNetの観測]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/subjects/qbert.md|Qbert]]"
updated: 2026-11-26
---
# TempoRL の Q 側・skip 側のノイズと τ を分けた評価

## 元の材料

[TempoRL の Q 側・skip 側のノイズと τ を分けた評価](../../../../../experiments/default-dqn/atari/2026-10-29_temporl-skip-forward-stage0.md) blob:75f8c3811f569edff43b56acfbb131072ccaf2a5

## 要点

- 分割の主因の読みは、ノイズが「決めきれない」、τが「決めきれない」。skip側だけを切った全体突破率の差は、ノイズが18.50 [11.50, 25.00]ポイント、τが7.50 [0.75, 14.50]ポイントだった。
- 合算の主因の読みは、ノイズが「決めきれない」、τが「決めきれない」。skip側だけを切った全体突破率の差は、ノイズが16.50 [10.00, 22.75]ポイント、τが10.25 [4.50, 16.25]ポイントだった。
- 分割は評価seed1/2の各200ゲームを合わせた400ゲーム。合算は200ゲームから始め、hまたはkの95%区間が0.5を含んだ読みだけseed2を追加した。
- 判断率・平均スキップ長はseed別のsession scalarを並べる。平均どうしを単純平均してまとめてはいない。実時間は性能の比較に使わない。
- 分割・合算とも、保存済み重みで skip 側の摂動を操作した評価である。学習中の遅れが改善するかは未検証。

## 内容

### 条件と数え方

すべて `skip.max_length_mode=full`、最大長J=10。評価slotは `[greedy]` だが、方策の型はEpsilonGreedyでε=0とした。16 lane、1 Run 200ゲーム、frame skip=4、sticky action=0.25、開始時の最大noop=30、上限108,000 framesを共通にした。

### 分割

集中面でゲームが終わった数: G: 面6で119本; NQ: 面6で47本; NS: 面6で32本; NB: 面6で28本。

### 合算

集中面でゲームが終わった数: G: 面5で48本; NQ: 面5で30本; NS: 面5で28本; NB: 面5で32本。

### G4 の判断に使う差

G4の学習はまだ選択・実行していない。候補は、合算の構成でTrain Actorのskip側だけmu-onlyにする群、skip専用の方策のτをfixedにする群、または構造の `skip.head.projection.noisy.enabled=false` でActorとLearner両方のskip Headのノイズを消す群。後者は新しく学習する群であり、既存checkpointの評価条件とは違う。

### single の値と固定の寄与の目安

既存のsingle評価を使う。ビルドと採取時点が異なるため、固定の寄与は点推定の目安である。single/fullを同じ実装後ビルドで取り直した比較ではない。既存値の出典は先行の評価記録の探索ブロック01・02とPRD106 §2.2。

### 採用した Run

各Runの53_evalg/episodeは200件。seed1/2を併記した条件は、両方を合わせて400件で数えた。

### 先行の200ゲームとの比較

Qbert の分割条件で、第8面の突破率は先行の200ゲームではノイズなし66.0%、Q側・skip側ともにノイズあり41.5%だった。実装後の各400ゲームでは62.50%と42.25%。ビルドと採取条件が異なるため、効果量の分母には実装後のノイズなしだけを使った。

## 主要な概念

- [行動の時間尺度](../../../../concepts/methods/temporal_abstraction.md)
- [NoisyNetの観測](../../../../concepts/methods/noisynet.md)
- [τ配置とIQN比較](../../../../concepts/methods/tau_sampling.md)
- [Qbert](../../../../concepts/subjects/qbert.md)

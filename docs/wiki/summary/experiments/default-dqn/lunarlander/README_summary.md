---
title: DefaultDQNAgent / LunarLander 探索記録
page_kind: サマリ
material: "[[docs/experiments/default-dqn/lunarlander/README.md|default-dqn/lunarlander/README]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/subjects/lunarlander.md|LunarLander]]"
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-28
---
# DefaultDQNAgent / LunarLander 探索記録

## 元の材料

[DefaultDQNAgent / LunarLander 探索記録](../../../../../experiments/default-dqn/lunarlander/README.md) blob:ef21fa26a6bc56b6630318f2c8f4d0a63df19367

## 要点

- LunarLander は DropMerge より一回の実験コストが低く、IQN・UQE・Replay・quantile 本数の動作確認と回帰確認に使う。
- 候補条件の絞り込みと仮説生成を LunarLander で行い、DropMerge への採否は DropMerge の Run で確かめる。
- IQN と QR、NoisyNet の比較は環境別の campaign に分かれている。
- TempoRL は学習中の繰り返し、評価時の full/single、勾配を遮断した対照を別々に比べる。
- RND の有無による影響と predictor の周期リセットの検証も、この索引から引ける。

## 主要な概念

- [LunarLander](../../../../concepts/subjects/lunarlander.md)
- [DropMerge](../../../../concepts/subjects/dropmerge.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)

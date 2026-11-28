---
title: DefaultDQNAgent ハイパーパラメータ探索
page_kind: サマリ
material: "[[docs/experiments/default-dqn/README.md|default-dqn/README]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
updated: 2026-11-28
---
# DefaultDQNAgent ハイパーパラメータ探索

## 元の材料

[DefaultDQNAgent ハイパーパラメータ探索](../../../../experiments/default-dqn/README.md) blob:230e0c04b8faa36ee0acdc6f82f8d7d1e0d6cfad

## 要点

- DefaultDQNAgent の共通設定と、Atari・DropMerge・LunarLander・GridMaze の実験を結ぶ索引である。
- 同じ seed でも QR と IQN は乱数消費や NN が異なり、学習軌跡は分岐する。
- 比較前に容量・warmup・batch・RR・学習率・PER・target 更新・評価方策・評価時点・予算を実効設定で揃える。
- 単一 seed の最良得点だけで判断せず、到達時間、終盤窓、失敗した Run 数、seed 間の範囲を記録する。
- 報酬密度と行動の意味が異なる環境では最適値も変わり得る。Munchausen の温度は各環境の clip 率と bonus から評価する。

## 主要な概念

- [評価と比較の条件](../../../concepts/measurement/evaluation.md)
- [報酬尺度と割引率](../../../concepts/methods/reward_scale_discount.md)
- [τ配置とIQN比較](../../../concepts/methods/tau_sampling.md)

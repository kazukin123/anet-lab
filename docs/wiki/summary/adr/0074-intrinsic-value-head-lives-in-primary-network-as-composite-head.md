# 価値の分解の内的の Head は、補助 Network ではなく、primary の Network の合成 Head に置く

## 元の材料

[ADR 0074](../../../adr/0074-intrinsic-value-head-lives-in-primary-network-as-composite-head.md) blob:f4330765070fc481d2ff06209dd3eb6dbeb244fd

## 要点

- 価値分解では外的と内的の値を primary Network の合成 Head から別キーで返す。
- 一度の forward で両方を得て、既存の target 同期・Actor 複製・checkpoint の経路を使う。
- どう重みを付けて Q を合成するかは DQN 側が決める。
- 既定では外的・内的の勾配を合わせて1回 clip する。
- 価値分解の有無で Network の形が変わるため、異なる構成の checkpoint は読み込めない。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)

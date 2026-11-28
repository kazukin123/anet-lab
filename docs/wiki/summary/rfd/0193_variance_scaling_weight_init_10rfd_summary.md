---
title: fan_in に基づく一様分布の重み初期化
page_kind: サマリ
material: "[[docs/memo/0193_variance_scaling_weight_init_10rfd.md|0193_variance_scaling_weight_init_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/subjects/breakout.md|Breakout]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-28
---
# fan_in に基づく一様分布の重み初期化

## 元の材料

[RFD0193　fan_in に基づく一様分布の重み初期化](../../../memo/0193_variance_scaling_weight_init_10rfd.md) blob:2771046412b10fd0808317d927c369a25d42c4e0

## 要点

- Dopamine の IQN・Rainbow と初期化条件をそろえるため、分散を指定した倍率と fan_in から決める一様分布を追加する。
- 参照する構成は重みの分散が 0.577/fan_in、bias が0である。
- Breakout の比較では he・xavier から torch 既定へ変えたときの立ち上がりに違いがあり、初期値の幅を切り分ける必要があった。
- 複数箇所に分かれていた初期化設定の読み取りをまとめ、層による解釈の食い違いを防ぐ。
- 初期化方式、分布の幅、バイアスと従来方式の再現を確かめてから学習結果を比較する。

## 内容

### 切り分けの目的

Dopamine の IQN・Rainbow では、入力数 fan_in に応じて一様分布の幅を決め、bias はゼロにする。torch の既定は分散が 1/(3×fan_in) で、bias にも乱数を使う。比較済みの he・xavier も分布・幅の基準が違い、幅だけをそろえる既存設定がなかった。

### 設定と分布

`variance_scaling` は `variance_scale / fan_in` を分散にする一様分布を使う。倍率は分散の倍率として受け取り、既存の標準偏差の倍率と区別する。Dopamine の値は約0.577、torch と同じ幅で bias だけをゼロにする比較には1/3を使う。乱数は既存の torch の生成器を使う。

### 初期化設定の読み取り

起草時には初期化方式や係数を14か所で個別に読んでいた。共通の読み取り処理を設ける。通常の block と DQN・Rainbow の Head は読み取り時に検証する。RND と内的価値の Head では従来の検証時期を保つ。

### Breakout の比較条件

入力配置や τ を混ぜる位置を変えた比較では、seed 差10.7を超える差は出なかった。一方、同じ seed 2 で he・xavier を全層 torch 既定に替えると、学習側の iteration 15〜19 の平均は59.6から89.6へ変わった。Dopamine 公開5本は同区間119.7〜177.3だが、公開値が学習側か評価側かは不明であり、比較時にこの条件差を残す。

### 実装の受入と実験の判断

分布の範囲、fan_in の扱い、bias がゼロか、既存方式を選んだ場合の再現を確かめる。学習で効果を測るときは、同じ幅で bias だけゼロにした条件と比べ、幅と bias の影響を分ける。学習結果の良し悪しは方式を実装できたかの判定と分ける。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
- [Breakout](../../concepts/subjects/breakout.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)

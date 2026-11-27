---
title: Agent は基盤の外の系統ごとのライブラリに置き、Head を書くための部品を基盤が公開する
page_kind: サマリ
material: "[[docs/adr/0079-agents-live-in-per-family-libraries-outside-core-and-head-authoring-parts-are-public.md|0079-agents-live-in-per-family-libraries-outside-core-and-head-authoring-parts-are-public]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/framework/325_muzero.md|325 Agent個別実装 MuZero系列]]"
updated: 2026-11-26
---
# Agent は基盤の外の系統ごとのライブラリに置き、Head を書くための部品を基盤が公開する

## 元の材料

[ADR 0079](../../../adr/0079-agents-live-in-per-family-libraries-outside-core-and-head-authoring-parts-are-public.md) blob:f7b656f32e19cb8ffc77008549f9f414844f6076

## 要点

- DQN・MuZero・ImageCls は基盤外の系統別ライブラリに置く。
- 各系統の初期化関数をアプリが呼び、基盤が具象 Agent を知らずに済む。
- Head の実装に必要な型と N-step 経験の組み立てを基盤から公開する。
- Q や分位の Head を汎用層へ残す案ではアルゴリズム固有の概念が漏れる。
- 具象ごとに分割すると DQN 内部の公開面が広がり、全 Agent を1本にすると系統間の独立を強制できない。

## 主要な概念

- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [325 Agent個別実装 MuZero系列](../../concepts/framework/325_muzero.md)

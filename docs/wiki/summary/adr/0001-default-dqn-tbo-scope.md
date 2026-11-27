---
title: DefaultDQN に TBO の公開範囲を限定する
page_kind: サマリ
material: "[[docs/adr/0001-default-dqn-tbo-scope.md|0001-default-dqn-tbo-scope]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/framework/325_muzero.md|325 Agent個別実装 MuZero系列]]"
updated: 2026-11-26
---
# DefaultDQN に TBO の公開範囲を限定する

## 元の材料

[ADR 0001](../../../adr/0001-default-dqn-tbo-scope.md) blob:bc3fa8491fd923227bc2258e6fe28896b41a34c8

## 要点

- TBO は Bellman ターゲットを変換する。
- 有効にすると Q 値メトリクスの解釈も変わる。
- DefaultDQN の実験で設定できるようにする。
- Rainbow は元のアルゴリズム構成を保つため常に無効にする。
- MuZero 独自の値変換・分布表現は別に扱う。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [325 Agent個別実装 MuZero系列](../../concepts/framework/325_muzero.md)

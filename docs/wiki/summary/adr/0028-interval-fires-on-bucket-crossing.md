---
title: 飛び飛びのカウンタに対する interval を bucket-crossing で定義する
page_kind: サマリ
material: "[[docs/adr/0028-interval-fires-on-bucket-crossing.md|0028-interval-fires-on-bucket-crossing]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
updated: 2026-11-26
---
# 飛び飛びのカウンタに対する interval を bucket-crossing で定義する

## 元の材料

[ADR 0028](../../../adr/0028-interval-fires-on-bucket-crossing.md) blob:a0c2549d53b483910674e1d9e4e31ecacea8f020

## 要点

- exp_step や learn_step は一度に複数進み、剰余がゼロのときだけ発火させると観測が抜ける。
- IntervalGate は前回と異なる区間に入ったときに発火する。
- 複数区間を飛び越えても、失われた途中状態の代わりに同じ値を連続記録しない。
- 指定周期がカウンターの刻みより短ければ毎イベント1回が上限になる。
- メトリクスは初回も記録する。後の評価スケジュールでは先頭を既定で抑え、offset で時点をずらせる。

## 内容

### 周期と軸

複数の経験を一度に数える exp_step では interval の倍数を飛び越えることがある。剰余による完全一致を要求すると、観測頻度が並列数など意図しない設定に左右される。

### 観測できる範囲

前回発火した区間は Observer またはタグごとに保持する。途中の状態を持たないので追い掛けて複数回発火しない。評価のみ先頭回の抑制を別途適用する。

## 主要な概念

- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)

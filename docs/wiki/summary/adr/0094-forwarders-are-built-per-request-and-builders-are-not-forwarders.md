---
title: Forwarder は依頼ごとに作り、作る側（並びと部品）は Forwarder を兼ねない
page_kind: サマリ
material: "[[docs/adr/0094-forwarders-are-built-per-request-and-builders-are-not-forwarders.md|0094-forwarders-are-built-per-request-and-builders-are-not-forwarders]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
updated: 2026-11-26
---
# Forwarder は依頼ごとに作り、作る側（並びと部品）は Forwarder を兼ねない

## 元の材料

[ADR 0094](../../../adr/0094-forwarders-are-built-per-request-and-builders-are-not-forwarders.md) blob:5afbb9ca977ae1a1d1a48a538a2e3225f2faa491

## 要点

- 実行の入口を Forwarder に一本化し、依頼ごとに生成する。
- 履歴を使う変換も Forwarder が受け、開始の印は入力のキーで渡す。
- 並びや部品は Builder として、依頼に合う Forwarder を作る。
- 依頼の種類名と追加情報は RL 側が定義する。
- 共通の層に種類を列挙せず、どの依頼に対する spec かを生成時に決める。

## 主要な概念

- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)

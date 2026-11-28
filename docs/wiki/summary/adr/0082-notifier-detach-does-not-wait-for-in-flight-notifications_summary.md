---
title: Notifier の Detach は進行中の通知を待たず、Observer はロックの外で呼ぶ
page_kind: サマリ
material: "[[docs/adr/0082-notifier-detach-does-not-wait-for-in-flight-notifications.md|0082-notifier-detach-does-not-wait-for-in-flight-notifications]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/319_gui.md|319 GUI 共通基盤]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
updated: 2026-11-28
---
# Notifier の Detach は進行中の通知を待たず、Observer はロックの外で呼ぶ

## 元の材料

[ADR 0082](../../../adr/0082-notifier-detach-does-not-wait-for-in-flight-notifications.md) blob:4c12cf3de750c3148e526498056200b286d064c4

## 要点

- Notifier の Observer 一覧を複数スレッドが変更・参照できるように保護する。
- Detach は進行中の通知を待たずに返る。
- すでに開始した通知は外した Observer にも届きうるため、その間の寿命を Notifier が保つ。
- Observer を実行する間は Notifier のロックを外す。
- Observer は Panel などの参照先の寿命を確かめ、どのスレッドで破棄されても安全にする。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)

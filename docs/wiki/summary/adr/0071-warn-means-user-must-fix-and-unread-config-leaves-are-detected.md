---
title: WARN は利用者が直すべき状態に限り、読まれない設定キーは NoCare にせず検出する
page_kind: サマリ
material: "[[docs/adr/0071-warn-means-user-must-fix-and-unread-config-leaves-are-detected.md|0071-warn-means-user-must-fix-and-unread-config-leaves-are-detected]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-26
---
# WARN は利用者が直すべき状態に限り、読まれない設定キーは NoCare にせず検出する

## 元の材料

[ADR 0071](../../../adr/0071-warn-means-user-must-fix-and-unread-config-leaves-are-detected.md) blob:c3513f3c9041010151f4c0e4af2cd3b2633ae337

## 要点

- WARN は設定・コード・実行環境を利用者が確認して直す必要がある状態に使う。
- 意図的に休止した評価や設定どおりの checkpoint 上書きは INFO にする。
- 構築したカタログ項目に未読キーがあれば構築時に止める。
- その他の未読キーは Run 構築末にまとめて警告する。
- 長い Run の途中で出力の一部が欠ける場合は、一律に停止せず原因を警告する。

## 主要な概念

- [316 設定管理](../../concepts/framework/316_config.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)

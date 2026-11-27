---
title: 作業の種類でだけ要る規約の細則は品質確認観点を正本にし、AGENTS.md からは「作業ごとに読むもの」の表で導く
page_kind: サマリ
material: "[[docs/adr/0061-detail-rules-live-in-quality-viewpoints-and-agents-md-leads-to-them-by-reading-table.md|0061-detail-rules-live-in-quality-viewpoints-and-agents-md-leads-to-them-by-reading-table]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-26
---
# 作業の種類でだけ要る規約の細則は品質確認観点を正本にし、AGENTS.md からは「作業ごとに読むもの」の表で導く

## 元の材料

[ADR 0061](../../../adr/0061-detail-rules-live-in-quality-viewpoints-and-agents-md-leads-to-them-by-reading-table.md) blob:88ec703ded616a5d4d5c4da2355af5ff6ca1a7f2

## 要点

- 毎回読む AGENTS.md と実装レビュー観点にコーディング規約が重複していた。
- 作業種類に応じて読む細則は、品質確認観点の該当節に正本を置く。
- AGENTS.md の表と入口 skill から編集前に必要な節へ案内する。
- 理由や例を含む規約と、編集後に点検する観点を同じ場所で扱う。
- 広く使う原則や共通の作業方法は AGENTS.md に残す。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)

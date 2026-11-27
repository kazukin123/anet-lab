---
title: 課題が prd 欄で PRD を指し（N:1）、PRD の実装済みは課題から導く
page_kind: サマリ
material: "[[docs/adr/0069-issues-point-to-prds-n-to-1-and-prd-completion-is-derived.md|0069-issues-point-to-prds-n-to-1-and-prd-completion-is-derived]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-26
---
# 課題が prd 欄で PRD を指し（N:1）、PRD の実装済みは課題から導く

## 元の材料

[ADR 0069](../../../adr/0069-issues-point-to-prds-n-to-1-and-prd-completion-is-derived.md) blob:22f594cdc1a7ef49defe3913e7ee36dd47cbd549

## 要点

- 設計文書ごとに専用の窓口課題を置く方式では、実作業と窓口が一覧で重複した。
- 各課題が自分を直す設計文書を1本指し、複数の課題から同じ文書を指せる。
- 文書の実装済み判定は参照元の課題の判定と状態から導く。
- 課題が複数の設計文書を指すと、関係のない実装の完了まで待つことになる。
- 作業状態の正本は課題に置き、設計文書に同じ状態を書き重ねない。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)

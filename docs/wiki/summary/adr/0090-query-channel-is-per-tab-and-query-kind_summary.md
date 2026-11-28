---
title: query channel の単位を、ブラウザタブと問い合わせの種類の組にする
page_kind: サマリ
material: "[[docs/adr/0090-query-channel-is-per-tab-and-query-kind.md|0090-query-channel-is-per-tab-and-query-kind]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
updated: 2026-11-28
---
# query channel の単位を、ブラウザタブと問い合わせの種類の組にする

## 元の材料

[ADR 0090](../../../adr/0090-query-channel-is-per-tab-and-query-kind.md) blob:f8ee4c10eab8f19a61ce7d3749054d5ad4c2522b

## 要点

- サーバーが古い問い合わせを取り消す単位を、タブと問い合わせ種類の組にする。
- グラフ描画と範囲統計には別の channel を使う。
- 範囲統計のパネルを開いても、同じタブのグラフ取得を中断しなくて済む。
- 同一 channel では新しい要求が古い要求を取り消す。
- 別 channel は同時実行枠だけを共有し、サーバー側に種類別の取り消しロジックを加えない。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)

---
title: タグセットは追跡する作業用ファイルと jar 同梱の初期値に分け、初期値は作業用が無いときだけ写す
page_kind: サマリ
material: "[[docs/adr/0067-tag-sets-split-tracked-working-file-and-bundled-initial-copied-only-when-absent.md|0067-tag-sets-split-tracked-working-file-and-bundled-initial-copied-only-when-absent]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
updated: 2026-11-28
---
# タグセットは追跡する作業用ファイルと jar 同梱の初期値に分け、初期値は作業用が無いときだけ写す

## 元の材料

[ADR 0067](../../../adr/0067-tag-sets-split-tracked-working-file-and-bundled-initial-copied-only-when-absent.md) blob:074d068d1fc1360a27ced67c92df2011b790c582

## 要点

- 普段のタグセットは workspace 群の親に置く作業用ファイルに保存する。
- リリース用の初期タグセットは jar のリソースとして独立に持つ。
- 起動時に作業用ファイルがなければ初期値を写す。
- 既存ファイルがある場合はその内容を使い、上書き展開でも編集を保つ。
- すでに作業用タグセットを持つ環境に、新版の初期値は自動では加わらない。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)

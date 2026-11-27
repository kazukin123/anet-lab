---
title: Auto Recolor は選択中だった Run の色を変えず、新しく選んだ Run へ毎回最良の色を配る
page_kind: サマリ
material: "[[docs/adr/0050-auto-recolor-colors-only-newly-selected-runs.md|0050-auto-recolor-colors-only-newly-selected-runs]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
updated: 2026-11-26
---
# Auto Recolor は選択中だった Run の色を変えず、新しく選んだ Run へ毎回最良の色を配る

## 元の材料

[ADR 0050](../../../adr/0050-auto-recolor-colors-only-newly-selected-runs.md) blob:dd73288b5d1b4490a72cdddfc79d56ed07039a12

## 要点

- 自動配色はすでに選択していた Run の色を保持する。
- 追加した Run には基本色を参照せず、残る選択色との分離が最良の色を与える。
- 以前の閾値判定では十分離れた候補があっても似た基本色が残った。
- 選択を外しても残った Run の色を入れ替えない。新規選択は選択順で配る。
- 明示した Recolor は選択集合全体を再配色し、1本だけでも palette の先頭を使う。

## 内容

### 旧方式の問題

色間距離の閾値を満たす基本色を残すと、もっと離れた色が空いていても近い組を使い続けた。閾値を上げるだけでは選択を解除した時の既存色の入れ替わりも解決しない。

### 配色の手順

最初に残っているすべての選択色を確保してから、新しく選択した Run に順番に割り当てる。既選択の後ろにある色も考慮できる。利用者が Recolor を明示したときだけ既存の割り当ても変える。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)

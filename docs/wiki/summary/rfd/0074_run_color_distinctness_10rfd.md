---
title: 選択中 Run の色分離（Metrics Viewer）
page_kind: サマリ
material: "[[docs/memo/done/0074_run_color_distinctness_10rfd.md|0074_run_color_distinctness_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
  - "[[docs/wiki/concepts/framework/319_gui.md|319 GUI 共通基盤]]"
updated: 2026-11-26
---
# 選択中 Run の色分離（Metrics Viewer）

## 元の材料

[選択中 Run の色分離（Metrics Viewer）](../../../memo/done/0074_run_color_distinctness_10rfd.md) blob:a785584d782d11ddde608ecbca7145a5fabcc68c

## 要点

- 固定20色を workspace 全体へ先着順に割り当てる方式では、比較中の Run 同士の色が近くなる。
- 選択した Run の区別を優先して配色し、Run 自体の同一性は runId が担う。
- 手動の最適配色は現在色に依存せず、Auto は既存色をできるだけ保つ。
- 選択順に処理し、Run を追加しても既存 Run の色が動きにくくする。
- 色は比較ごとの表示状態として扱い、Run の永続属性にはしない。

## 内容

### 背景

`selectedRuns` は選択順の配列で、後から選んだ Run が末尾に付く。一度外して選び直すと末尾へ移動する。同 `bindRunListEvents`（1182-1185 行）、`setSelectedRuns`（1601 行）。
要点は 色が「workspace 内での位置」で決まり、「今の比較の構成」では決まらないことである。
比較したい 2 本が palette 上で隣り合っていれば似た色になり、20 離れていれば完全に同じ色になる。

### 仕様

チェックボックスの状態は `localStorage` の `anet.metricsviewer.autoRecolorEnabled` へ保存する。既定は ON（保存値が無い場合も ON）。
`minDist(c, [])` は +∞ とし、argmax の同値は `RUN_COLORS` の並び順で先勝ちとする。

### 残る制約

分離できる上限は palette の 20 色である。21 本目以降は同じ色が再び現れる。上限を上げるには palette の拡張が要り、それは別 PRD とする（近接ペア ΔE 0.0385 の緑 2 色の見直しを含む）。
既定 ON と しきい値 0.16 の組み合わせにより、基本色は選択した瞬間に上書きされるのが常態になる。 基本色が残るのは未選択行だけである。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)

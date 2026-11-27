---
title: Metrics表示は共通Step区間、保存LODはfactor4とする
page_kind: サマリ
material: "[[docs/adr/0052-metrics-display-uses-common-step-bins-and-factor4-storage.md|0052-metrics-display-uses-common-step-bins-and-factor4-storage]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
updated: 2026-11-26
---
# Metrics表示は共通Step区間、保存LODはfactor4とする

## 元の材料

[ADR 0052](../../../adr/0052-metrics-display-uses-common-step-bins-and-factor4-storage.md) blob:afa61b0fcdebdeb935d4c4ab44c911a8bd10d93e

## 要点

- 異なる記録頻度の Run も同じ範囲で比較できるよう、Step=0 を起点に共通区間を切る。
- 区間内は完成した保存 bucket を再利用し、境界だけ下位へ降りて集約する。
- 保存 LOD の倍率は16から4に変更する。
- 合成 SQLite で倍率4は16に比べ容量が約1.7倍、境界での全点参照が約5分の1だった。
- キャッシュは schema=2 で再構築し、原本、序数の主キー、generation の契約を保つ。
- 表示側の追記では、近傍外にも軽い概観を保ち、最終的に共通の Plotly figure から SVG を作って表示のずれを解消した。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)

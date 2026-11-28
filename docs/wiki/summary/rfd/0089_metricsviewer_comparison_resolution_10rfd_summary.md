---
title: MetricsViewerの比較粒度を表示Step範囲でそろえる
page_kind: サマリ
material: "[[docs/memo/done/0089_metricsviewer_comparison_resolution_10rfd.md|0089_metricsviewer_comparison_resolution_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
updated: 2026-11-28
---
# MetricsViewerの比較粒度を表示Step範囲でそろえる

## 元の材料

[MetricsViewerの比較粒度を表示Step範囲でそろえる](../../../memo/done/0089_metricsviewer_comparison_resolution_10rfd.md) blob:c25080b4c2577cc767553202fc62f84e04e2908b

## 要点

- Run やタグを増やしただけでグラフが粗くなり、選択順によって比較対象の粒度も食い違っていた。
- 表示 step 範囲と画面幅が同じなら、少なくとも4 Run まで同じ step 区間で比較する。
- 保存する LOD は factor 4 の階層とし、表示は step 0 起点の半開区間で min・max・平均・末尾を集約する。
- 区間に収まる保存 bucket を再利用し、平均は元の点数で重み付けする。
- 概観を先に表示し、詳細は画面近くで描画する。点数制限時は黙って粒度を落とさず明示する。

## 内容

### 目標と解決策

同じグラフは全Runで同じStep区間を比較し、少なくとも4Runまで、同じ表示範囲と画面幅ならRun数・選択タグ数によって粒度を落とさない。ズーム・パン・Reset View・Reload・Run選択順序によらずこの契約を守る。

### 利用者の要求と受け入れ条件

記録頻度と長さが異なるRunでも、各Step区間の集約値がマスタの直接計算と一致する。

### 確定事項

ユーザーの選択: 保存factor=4。形式変更時のキャッシュ再構築を受け入れる。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)

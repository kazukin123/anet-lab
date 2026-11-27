---
title: Metrics Viewerの表示粒度
page_kind: 測り方
aliases:
  - MetricsViewer
  - LOD
  - common Step bins
  - maxPoints
  - overview
description: 調査時には Run ごとの表示粒度が混在した。現行の Viewer は同じグラフの Run を共通の Step 区間で集約し、記録が無い区間を補間しない。
updated: 2026-11-26
---
# Metrics Viewerの表示粒度

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [測り方](../../index.md#測り方)

## 今の結論

調査時には Run ごとの表示粒度が混在した。現行の Viewer は同じグラフの Run を共通の Step 区間で集約し、記録が無い区間を補間しない。

## 分かっていること

### anet の知見

- `run_20261119-202718_mz_tg_g99ne_rr4_s2` のキャッシュでは最大の tag が 234,373 点、中央値が 6,000 点だった。最大の tag の後半だけを集計した読み込み済みの計測が 14.7 ms、全点を初めて集計した計測が 409 ms。範囲とキャッシュの状態が異なる（出典: [RFD0196](../../../memo/0196_metrics_viewer_run_comparison_10rfd.md)、§4「前提」の P18）。

- 調査時は共有点予算・RunごとのLOD選択・古いwindow再利用により、選択数と操作順で粒度が混在した。引き金はmemory pressureの検出ではなかった。（出典: [metrics_viewer_resolution_investigation_2026-10-10.md](../../../../reports/metrics_viewer_resolution_investigation_2026-10-10.md)、結論・原因箇所）。
- 4 Run × 3タグで確認した約0.92 MB は API が転送した JSON サイズだった。DOM・Plotly は stub で、ブラウザーの heap や全216タグの常駐メモリはこの調査では測っていない（出典: [MetricsViewer 複数 Run の表示粒度: 再現調査と対応方針](../../../../reports/metrics_viewer_resolution_investigation_2026-10-10.md)、制約と次の判断）。
- 同じグラフの Run は共通の Step 区間で集約する。同じ表示範囲・画面幅なら少なくとも4 Run までは選択数で粒度を下げず、欠測区間は補間しない（出典: [Run分析ユーザーガイド](../../../design/030_user_guide_analysis.jp.md)、§3）。

### 外の事例

なし

## 横断的知見

- 対立点: 調査時には Run ごとの粒度と古い表示範囲の再利用が混在を招いた。現行の操作ガイドでは、同じグラフの系列を同じ幅の Step 区間で読む。条件の違い: 前者は変更前の操作順に依存した再現、後者は同じ表示範囲・画面幅にそろえた比較である（出典: [表示粒度の再現調査](../../../../reports/metrics_viewer_resolution_investigation_2026-10-10.md)、結論・原因箇所、[Run分析ユーザーガイド](../../../design/030_user_guide_analysis.jp.md)、§3）。

## 見立て

なし

## 分かっていないこと

なし。

## 撤回・更新

なし。

## 関連

- [Auto Recolor は選択中だった Run の色を変えず、新しく選んだ Run へ毎回最良の色を配る](../../../adr/0050-auto-recolor-colors-only-newly-selected-runs.md)。
- [metrics query の latest-wins をサーバ側で強制し、その単位を query channel（ブラウザタブ）とする](../../../adr/0023-server-enforced-query-supersede-per-channel.md)。channel の単位は ADR 0090 で改めた。
- [query channel の単位を、ブラウザタブと問い合わせの種類の組にする](../../../adr/0090-query-channel-is-per-tab-and-query-kind.md)。
- [RFD0090: MetricsViewerのキャッシュ再構築を高速化し全ワークスペースで事前実行する](../../../memo/done/0090_metricsviewer_cache_rebuild_10rfd.md)。
- [RFD0089: MetricsViewerの比較粒度を表示Step範囲でそろえる](../../../memo/done/0089_metricsviewer_comparison_resolution_10rfd.md)。
- [MetricsViewer SQLite キャッシュ化(Phase 1: OOM 解消・LOD・viewport API)](../../../memo/done/0041_metrics_sqlite_cache_10rfd.md)。
- [共通Step区間を採った理由](../../../adr/0052-metrics-display-uses-common-step-bins-and-factor4-storage.md)、本文。
- [現行の共通Step区間とcache契約](../../../design/210_metrics_viewer.jp.md)、§2.7・§6.2・§6.3。
- [atari](../subjects/atari.md)。
- [evaluation](evaluation.md)。
- [仕組みと契約](../../../design/210_metrics_viewer.jp.md)。

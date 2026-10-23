# Metrics Viewerの表示粒度

## 今の結論

調査時の粒度混在は共有点予算・RunごとのLOD選択・古いwindowの再利用に起因し、引き金はmemory pressureの検出ではなかった。

## 分かっていること

- 調査時は共有点予算・RunごとのLOD選択・古いwindow再利用により、選択数と操作順で粒度が混在した。引き金はmemory pressureの検出ではなかった。（出典: [metrics_viewer_resolution_investigation_2026-10-10.md](../../../reports/metrics_viewer_resolution_investigation_2026-10-10.md)、結論・原因箇所）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [Auto Recolor は選択中だった Run の色を変えず、新しく選んだ Run へ毎回最良の色を配る](../../adr/0050-auto-recolor-colors-only-newly-selected-runs.md)。
- [metrics query の latest-wins をサーバ側で強制し、その単位を query channel（ブラウザタブ）とする](../../adr/0023-server-enforced-query-supersede-per-channel.md)。
- [PRD 090: MetricsViewerのキャッシュ再構築を高速化し全ワークスペースで事前実行する](../../memo/090_metricsviewer_cache_rebuild_10prd.md)。
- [PRD 089: MetricsViewerの比較粒度を表示Step範囲でそろえる](../../memo/089_metricsviewer_comparison_resolution_10prd.md)。
- [MetricsViewer SQLite キャッシュ化(Phase 1: OOM 解消・LOD・viewport API)](../../memo/done/041_metrics_sqlite_cache_10prd.md)。
- [共通Step区間を採った理由](../../adr/0052-metrics-display-uses-common-step-bins-and-factor4-storage.md)、本文。
- [現行の共通Step区間とcache契約](../../design/210_metrics_viewer.jp.md)、§2.7・§6.2・§6.3。
- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [仕組みと契約](../../design/210_metrics_viewer.jp.md)。

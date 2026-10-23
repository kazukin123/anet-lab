# Metrics Viewer

## 概要・責務

- MetricsマスタをRunごとのSQLiteキャッシュへ取り込み、range queryとブラウザ描画を通して長いRunを比較する。（出典: [Metrics Viewer](../../design/210_metrics_viewer.jp.md)、§1.1・§2.1）。
- JSONL/gzipをstreaming parseし、L0・factor4のLOD・TagStatsを同一transactionで更新する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§2.2 Metrics Viewer）。

## 成立の経緯

- 2026-07-27: Metricsキャッシュは破棄可能な従属導出物とする(マスタ=JSONL、PK=序数)（出典: [ADR 0015](../../adr/0015-metrics-cache-disposable-derivative.md)、本文）。
- 2026-08-01: 042: MetricsViewer 構造リファクタリング(挙動不変) — 取り込み/照会経路の可読性改善（出典: [PRD042](../../memo/done/042_metricsviewer_refactor_10prd.md)、題・目的）。
- 2026-08-13: PRD 049: metrics query の supersede 機構（出典: [PRD049](../../memo/done/049_metrics_query_supersede_10prd.md)、題・目的）。
- 2026-08-21: MetricsViewer SQLite キャッシュ化(Phase 1: OOM 解消・LOD・viewport API)（出典: [PRD041](../../memo/done/041_metrics_sqlite_cache_10prd.md)、題・目的）。
- 2026-08-21: Metrics Viewerの配置をリリースのapps配置へ移した（出典: [PRD043](../../memo/done/043_release_packaging_10prd.md)、2. リポジトリ再配置）。
- 2026-08-29: Metrics Viewer 定常取り込み I/O 抑制 PRD（出典: [PRD064](../../memo/done/064_metricsviewer_idle_ingest_io_10prd.md)、題・目的）。
- 2026-09-05: メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す（出典: [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md)、題・目的）。
- 2026-09-12: PRD 074: 選択中 Run の色分離（Metrics Viewer）（出典: [PRD074](../../memo/done/074_run_color_distinctness_10prd.md)、題・目的）。
- 2026-09-12: Run 色は比較ごとの役割とし、分離は選択中 Run の集合に対してだけ保証する（出典: [ADR 0041](../../adr/0041-run-color-is-comparison-role-not-run-identity.md)、本文）。

## 選ばなかった形

- 未選択Runの背景取り込みを廃止し、選択時だけ取り込む形 — 不採用: 定常I/Oの削減とは分け、既存の背景取り込みを保つ（出典: [PRD064](../../memo/done/064_metricsviewer_idle_ingest_io_10prd.md)、0.3 Non-goals）。
- WatchServiceなどのfilesystem通知によるevent駆動化 — 不採用: poll経路の無駄な完全検査を抑える範囲に絞る（出典: [PRD064](../../memo/done/064_metricsviewer_idle_ingest_io_10prd.md)、0.3 Non-goals）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- 通常追記ではcache世代を維持し、全再構築だけで世代を変える。commit済みprefix、tag隔離、Run-level errorのtransaction境界を変えない（出典: [PRD064](../../memo/done/064_metricsviewer_idle_ingest_io_10prd.md)、1.1 正本とcache）。
- priority 3・background 1、単一writer、完成行のtransaction境界でのworkspace切替を維持する（出典: [PRD064](../../memo/done/064_metricsviewer_idle_ingest_io_10prd.md)、1.2 Scheduling）。
- 色分離の変更でserver、Metricsキャッシュ、HTTP APIを変更しない（出典: [PRD074](../../memo/done/074_run_color_distinctness_10prd.md)、対象）。

## 関係

- 設計: [Metrics Viewer](../../design/210_metrics_viewer.jp.md)、§1.1・§2.1。
- ユーザーガイド: [Run分析ガイド](../../design/030_user_guide_analysis.jp.md)。
- 含む機能カテゴリ: なし。

## ファイル

点検が作り直す。手で直さない。

- [.checkstyle](../../../apps/metrics-viewer/.checkstyle)
- [.gitignore](../../../apps/metrics-viewer/.gitignore)
- [maven-wrapper.properties](../../../apps/metrics-viewer/.mvn/wrapper/maven-wrapper.properties)
- [build.bat](../../../apps/metrics-viewer/build.bat)
- [checkstyle.xml](../../../apps/metrics-viewer/checkstyle.xml)
- [metrics-viewer-cs-cleanup.xml](../../../apps/metrics-viewer/metrics-viewer-cs-cleanup.xml)
- [metrics-viewer-cs-formatter.xml](../../../apps/metrics-viewer/metrics-viewer-cs-formatter.xml)
- [mvnw](../../../apps/metrics-viewer/mvnw) / [mvnw.cmd](../../../apps/metrics-viewer/mvnw.cmd)
- [pom.xml](../../../apps/metrics-viewer/pom.xml)
- [MetricsViewerApplication.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/MetricsViewerApplication.java)
- [MetricsViewerSettings.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/config/MetricsViewerSettings.java)
- [MetricsCacheDatabase.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/infra/MetricsCacheDatabase.java)
- [MetricsSource.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/infra/MetricsSource.java)
- [RunScanner.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/infra/RunScanner.java)
- [GzipInputSessions.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/GzipInputSessions.java)
- [GzipSessionReader.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/GzipSessionReader.java)
- [IngestScheduler.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/IngestScheduler.java)
- [LoadingThread.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LoadingThread.java)
- [LodBucket.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodBucket.java)
- [LodIngestWriter.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodIngestWriter.java)
- [LodPageCache.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodPageCache.java)
- [MetricsApiException.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsApiException.java)
- [MetricsIngestor.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsIngestor.java)
- [MetricsQueryCoordinator.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsQueryCoordinator.java)
- [MetricsQueryPlanner.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsQueryPlanner.java)
- [MetricsRangeProjector.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsRangeProjector.java)
- [MetricsRepository.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsRepository.java)
- [MetricsService.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsService.java)
- [QueryCancelledException.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/QueryCancelledException.java)
- [RawFileReader.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/RawFileReader.java)
- [RunWarningRegistry.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/RunWarningRegistry.java)
- [SeriesAvailability.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/SeriesAvailability.java)
- [SourceReader.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/SourceReader.java)
- [WorkspaceManager.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/WorkspaceManager.java)
- [MetricTraceEncoder.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/util/MetricTraceEncoder.java)
- [HttpAccessLogFilter.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/HttpAccessLogFilter.java)
- [MetricsViewerController.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/MetricsViewerController.java)
- [WorkspaceController.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/WorkspaceController.java)
- [ApiError.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/ApiError.java)
- [GetMetricsRequest.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/GetMetricsRequest.java)
- [GetMetricsResponse.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/GetMetricsResponse.java)
- [GetRunsResponse.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/GetRunsResponse.java)
- [GetWorkspacesResponse.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/GetWorkspacesResponse.java)
- [IngestInfo.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/IngestInfo.java)
- [Issue.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/Issue.java)
- [LodProjection.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/LodProjection.java)
- [LodSummary.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/LodSummary.java)
- [MetricsSeriesRequest.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/MetricsSeriesRequest.java)
- [MetricsSeriesResult.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/MetricsSeriesResult.java)
- [MinMaxProjection.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/MinMaxProjection.java)
- [PrioritizeRunsRequest.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/PrioritizeRunsRequest.java)
- [RawProjection.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/RawProjection.java)
- [RunInfo.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/RunInfo.java)
- [RunStats.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/RunStats.java)
- [StrictStringDeserializer.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/StrictStringDeserializer.java)
- [SwitchWorkspaceRequest.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/SwitchWorkspaceRequest.java)
- [TagInfo.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/TagInfo.java)
- [TagStats.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/TagStats.java)
- [additional-spring-configuration-metadata.json](../../../apps/metrics-viewer/src/main/resources/META-INF/additional-spring-configuration-metadata.json)
- [spring-devtools.properties](../../../apps/metrics-viewer/src/main/resources/META-INF/spring-devtools.properties)
- [application.properties](../../../apps/metrics-viewer/src/main/resources/application.properties)
- [favicon.png](../../../apps/metrics-viewer/src/main/resources/static/favicon.png)
- [index.html](../../../apps/metrics-viewer/src/main/resources/static/index.html)
- [metrics-viewer.css](../../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.css) / [metrics-viewer.js](../../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.js)
- [MetricsCacheDatabaseIntegrationTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/infra/MetricsCacheDatabaseIntegrationTest.java)
- [RunScannerTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/infra/RunScannerTest.java)
- [CacheMaintenanceTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/CacheMaintenanceTest.java)
- [IngestSchedulerTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/IngestSchedulerTest.java)
- [LoadingThreadTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/LoadingThreadTest.java)
- [LodPageCacheTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/LodPageCacheTest.java)
- [MetricsApiIntegrationTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsApiIntegrationTest.java)
- [MetricsCacheIntegrationTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsCacheIntegrationTest.java)
- [MetricsIngestorIntegrationTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsIngestorIntegrationTest.java)
- [MetricsLodIntegrationTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsLodIntegrationTest.java)
- [MetricsQueryConcurrencyTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsQueryConcurrencyTest.java)
- [MetricsQueryCoordinatorTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsQueryCoordinatorTest.java)
- [MetricsQueryPlannerTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsQueryPlannerTest.java)
- [MetricsRepositorySnapshotIntegrationTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsRepositorySnapshotIntegrationTest.java)
- [MetricsRepositoryStepRangeIntegrationTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsRepositoryStepRangeIntegrationTest.java)
- [MetricsViewerSettingsTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsViewerSettingsTest.java)
- [SeriesAvailabilityTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/SeriesAvailabilityTest.java)
- [WorkspaceApiIntegrationTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/WorkspaceApiIntegrationTest.java)
- [WorkspaceManagerTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/WorkspaceManagerTest.java)
- [WorkspaceSnapshotIntegrationTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/WorkspaceSnapshotIntegrationTest.java)
- [ComparisonResolutionPlaywrightTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/ComparisonResolutionPlaywrightTest.java)
- [GraphInteractionPlaywrightTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/GraphInteractionPlaywrightTest.java)
- [HttpAccessLogFilterTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/HttpAccessLogFilterTest.java)
- [MetricsPlotPlaywrightTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/MetricsPlotPlaywrightTest.java)
- [MetricsViewerPlaywrightTestData.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/MetricsViewerPlaywrightTestData.java)
- [MetricsViewerPlaywrightTestSupport.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/MetricsViewerPlaywrightTestSupport.java)
- [OutlierRangePlaywrightTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/OutlierRangePlaywrightTest.java)
- [ReloadPlaywrightTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/ReloadPlaywrightTest.java)
- [RunListPlaywrightTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/RunListPlaywrightTest.java)
- [ScreenshotModePlaywrightTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/ScreenshotModePlaywrightTest.java)
- [SignedLogPlaywrightTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/SignedLogPlaywrightTest.java)
- [TagListPlaywrightTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/TagListPlaywrightTest.java)
- [WorkspaceSelectorPlaywrightTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/WorkspaceSelectorPlaywrightTest.java)

## PRD の一覧

点検が作り直す。手で直さない。

- [PRD064](../../memo/done/064_metricsviewer_idle_ingest_io_10prd.md) — Metrics Viewer 定常取り込み I/O 抑制 PRD
- [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md) — メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す

## 台帳の指摘

台帳のツールが作り直す。手で直さない。

<!-- registry:begin -->
- F-0003 P1 defect — window内に点が無いtagはgraph blockごと消え、全tag一括のReset View以外では戻せない
- F-0002 P2 defect — 未完成のLOD pageをbucketごとに全行読み直すため、LOD射影で読む行数が表示bucket数の2乗になる
- F-0004 P2 defect — DataFetcherの単一のabort枠を独立した呼び出し元が奪い合い、abortが失敗Toastや初期Run選択の喪失になる
- F-0005 P2 defect — 初期化中にScreenshotへ切り替えるとinit完了時にmodeが上書きされ、init失敗時はScreenshotから抜けられない

[台帳の一覧](../../../reports/registry/findings.md)
<!-- registry:end -->

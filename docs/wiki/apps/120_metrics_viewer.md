# Metrics Viewer

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [アプリケーション](../index.md#アプリケーション)

## 概要・責務

- MetricsマスタをRunごとのSQLiteキャッシュへ取り込み、range queryとブラウザ描画を通して長いRunを比較する。（出典: [Metrics Viewer](../../design/210_metrics_viewer.jp.md)、§1.1・§2.1）。
- JSONL/gzipをstreaming parseし、L0・factor4のLOD・TagStatsを同一transactionで更新する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§2.2 Metrics Viewer）。
- タグセットはTagSetStoreが起動時の検証と保存を、TagSetControllerがAPIを、TagSetMenuが選択・編集の操作を受け持つ。Workspaceをまたぐ保存済みの集合と、ブラウザの未保存の選択を分けて持つ（出典: [Metrics Viewer](../../design/210_metrics_viewer.jp.md)、§3.1・§8・§9.6）。

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
- 2026-10-10: 比較するRunの表示粒度を共通Step区間へそろえ、保存LODをfactor4にした（出典: [PRD089](../../memo/done/089_metricsviewer_comparison_resolution_10prd.md)、題・目標と解決策）。
- 2026-10-12: キャッシュ再構築の入力とSQLをまとめ、全workspaceの検査・背景取り込みと表示workspaceの寿命を分けた（出典: [PRD090](../../memo/done/090_metricsviewer_cache_rebuild_10prd.md)、題・§2・§4.1〜§4.4）。
- 2026-11-02: 名前を付けて保存したtagの集合（タグセット）をサーバー側の作業用ファイルに保存してworkspaceとブラウザをまたいで共有し、Tagsの下の1行と開閉する一覧で呼び出し（選択全体の置き換え）・Save・Revert・Save As・名前変更・削除・Deselectを行い、選択中のタグセットをlocalStorageに保つ仕様と、jarに同梱した初期値を作業用ファイルが無いときだけ写すことを定めた。作業用ファイルの既定はworkspaces-dirの直下で、gitで追跡しリリースの梱包からは除く（出典: [PRD111](../../memo/111_metrics_viewer_tag_sets_10prd.md)、題・§2・D1・D2・D5・D13〜D16）。
- 2026-11-02: タグセットは追跡する作業用ファイルと jar 同梱の初期値に分け、初期値は作業用が無いときだけ写す（出典: [ADR 0067](../../adr/0067-tag-sets-split-tracked-working-file-and-bundled-initial-copied-only-when-absent.md)、本文）。

## 選ばなかった形

- 任意factor設定・旧DB変換・新しい永続ストア — 不採用: 比較粒度の障害解消に不要で、マスタから再構築する既存の境界で足りる（出典: [PRD089](../../memo/done/089_metricsviewer_comparison_resolution_10prd.md)、複雑性監査）。
- 全Runを常にrawで保持する形 — 不採用: マスタ全点を保持せずに正確な区間集約ができる（出典: [PRD089](../../memo/done/089_metricsviewer_comparison_resolution_10prd.md)、複雑性監査）。
- 汎用ジョブ基盤と高度な予測prefetch — 先送り: 近傍読み込みの実測を先に行う。再開条件: 近傍読み込みでも応答性不足が実測されたとき（出典: [PRD089](../../memo/done/089_metricsviewer_comparison_resolution_10prd.md)、複雑性監査）。
- 再構築キュー用の新しい永続ストア — 不採用: process内で未処理Runと理由を持ち、起動時にキャッシュから必要な作業を再発見できる（出典: [PRD090](../../memo/done/090_metricsviewer_cache_rebuild_10prd.md)、§2）。
- 未選択Runの背景取り込みを廃止し、選択時だけ取り込む形 — 不採用: 定常I/Oの削減とは分け、既存の背景取り込みを保つ（出典: [PRD064](../../memo/done/064_metricsviewer_idle_ingest_io_10prd.md)、0.3 Non-goals）。
- WatchServiceなどのfilesystem通知によるevent駆動化 — 不採用: poll経路の無駄な完全検査を抑える範囲に絞る（出典: [PRD064](../../memo/done/064_metricsviewer_idle_ingest_io_10prd.md)、0.3 Non-goals）。
- 既存の queryRevision を supersede の連番に流用する形 — 不採用（流用しない）: queryRevision はユーザー操作でしか増えず、Auto Reload の経路では同じ値の POST が同時に飛びうる（出典: [PRD049](../../memo/done/049_metrics_query_supersede_10prd.md)、1. 現状の事実・2.1 query channelと連番）。
- C++ 側からの gzip 出力 — 不採用: PRD041 の実測の表に不採用の確定として記されている（理由の記述は無い）（出典: [PRD041](../../memo/done/041_metrics_sqlite_cache_10prd.md)、Context の実測）。
- schema の不一致・破損・ソース同一性の喪失のときに cache を migration する形 — 不採用: cache はマスタから再構築できる破棄可能な従属導出物なので、警告なしの全破棄・再構築で解決する（出典: [ADR 0015](../../adr/0015-metrics-cache-disposable-derivative.md)、本文；[PRD041](../../memo/done/041_metrics_sqlite_cache_10prd.md)、1. 原則 2）。
- タグセットにRunの組・表示設定（Log、percentile、LOD、viewport）・グラフの並び順・パターン指定・workspaceごとの区別を持たせる形 — 不採用（非ゴール）: 範囲の選択（出典: [PRD111](../../memo/111_metrics_viewer_tag_sets_10prd.md)、§2.2）。
- タグセットの呼び出しで、見えているtagの部分だけ置き換える形／今の選択に足し込む形 — 不採用: 呼んだときの選択（workspace・選択中のRun）で結果が変わる／観点の切替にClear Allが要る（出典: [PRD111](../../memo/111_metrics_viewer_tag_sets_10prd.md)、D1・§7）。
- タグセットの保存で、見えているtag（選択中のRunのtag）の部分だけ書き、Tags一覧に出ていない部分を残す形 — 不採用: ユーザーが単純さを選んだ（出典: [PRD111](../../memo/111_metrics_viewer_tag_sets_10prd.md)、D2・§7）。
- タグセットの入口をセレクト＋動詞ボタン、またはチップの横並びにする形 — 不採用: 名前の欄が約93pxに細る／数が増えると隠れ、名前変更・削除が隠れ操作になる（出典: [PRD111](../../memo/111_metrics_viewer_tag_sets_10prd.md)、D5・§7）。
- タグセットのAPIで名前をパスに入れる形、全体をGET・PUTする形 — 不採用: `%2F`をTomcatが既定で拒む／古い一覧を持つ側の保存が他の変更を消す（出典: [PRD111](../../memo/111_metrics_viewer_tag_sets_10prd.md)、D9・§7）。
- 選択中のタグセットを外す操作を、一覧の先頭の行・行の専用ボタン・開閉ボタンの中の×に置く形 — 不採用: 呼び出しと見分けがつかない／めったに使わないのに幅を取る／§3.3に無い形でタッチで押しにくい（出典: [PRD111](../../memo/111_metrics_viewer_tag_sets_10prd.md)、D10・§7）。
- 稼働中に外で変わったタグセットのファイル（手で直したもの、gitの切り替えで変わったもの）を、変更を検知して読み直す形、書く直前に照合して409で拒む形 — 不採用: 読み込みの経路を起動時の1つにし、照合も足さない単純さをユーザーが選んだ（出典: [PRD111](../../memo/111_metrics_viewer_tag_sets_10prd.md)、D12・§7）。
- タグセットの作業用ファイルの既定を作業ディレクトリ基準の固定値にする形、workspaces-dir直下で親が無ければ起動を止める形 — 不採用: 既存テストが起動で落ち、別のworkspace群での起動で元の群に書くか止まる／存在しないフォルダで起動する既存テストに手当てが要り、workspaces-dirが無いだけで起動が止まる（テストの手当て自体は本当の制約ではなく、ユーザーが手間と厳しさを比べて選ばなかった）（出典: [PRD111](../../memo/111_metrics_viewer_tag_sets_10prd.md)、D16・§7）。
- リリース用のタグセットを書き換え不可の「組み込み」の区画として作業用と並べる形／梱包で作業用の場所に置くだけの形 — 不採用: UI・APIが一回り大きくなり同名の扱いが要る／上書き展開で利用者の作業用ファイルを消す（出典: [ADR 0067](../../adr/0067-tag-sets-split-tracked-working-file-and-bundled-initial-copied-only-when-absent.md)、Considered Options；[PRD111](../../memo/111_metrics_viewer_tag_sets_10prd.md)、D13・§7）。
- タグセットの作業用ファイルをViewerのフォルダに置く形、初期値を追跡ファイルと設定キーで指す形 — 不採用: ユーザーがworkspace群と一緒に置くことを選んだ／設定キーが増え、Runnerのconfigフォルダに Viewer のファイルが混ざる（出典: [PRD111](../../memo/111_metrics_viewer_tag_sets_10prd.md)、D14・D15・§7）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- 共通Step区間への変更でもマスタと過去Run artifact、MinMax/Mean/Band・極値hover・全点TagStats・凡例・Log・percentile・Scroll Lock・Reloadを維持する（出典: [PRD089](../../memo/done/089_metricsviewer_comparison_resolution_10prd.md)、利用者の要求と受け入れ条件5・6）。
- 再構築をまとめて処理しても同一Stepの複数点・tagの序数・極値の同値時の選択・点数加重平均・非scalar行の順序を変えず、L0・LOD・統計・source位置を同じcommitで確定する（出典: [PRD090](../../memo/done/090_metricsviewer_cache_rebuild_10prd.md)、§4.1・§5の2〜4）。
- MetricsQueryCoordinator の critical section の中では、Statement.cancel、旧 request の完了待ち、Workspace・DB の lock の取得、query 本体の実行のどれも行わない（出典: [PRD049](../../memo/done/049_metrics_query_supersede_10prd.md)、2.3 supersede規則）。
- 通常追記ではcache世代を維持し、全再構築だけで世代を変える。commit済みprefix、tag隔離、Run-level errorのtransaction境界を変えない（出典: [PRD064](../../memo/done/064_metricsviewer_idle_ingest_io_10prd.md)、1.1 正本とcache）。
- priority 3・background 1、単一writer、完成行のtransaction境界でのworkspace切替を維持する（出典: [PRD064](../../memo/done/064_metricsviewer_idle_ingest_io_10prd.md)、1.2 Scheduling）。
- 色分離の変更でserver、Metricsキャッシュ、HTTP APIを変更しない（出典: [PRD074](../../memo/done/074_run_color_distinctness_10prd.md)、対象）。
- タグセットを足しても、新しいtagの自動選択、Select All・Clear All・Selected Onlyの動きは変えない。タグセットはMetricsキャッシュにもRunフォルダにも置かない（出典: [PRD111](../../memo/111_metrics_viewer_tag_sets_10prd.md)、D3・§5のE4・E16）。
- タグセットの作業用ファイルはリリースの梱包に含めず、残っていれば梱包を失敗させる。不正JSONの400への変換はworkspace切替とタグセットのControllerに限り、metrics・prioritizeの失敗の形には影響させない。既存のテストは、サイドパネルの操作行の一覧の期待にTagsの新しい行を足すほかは変えずに通る（出典: [PRD111](../../memo/111_metrics_viewer_tag_sets_10prd.md)、§6.1・§6.2・§8・E17）。

## 関係

- 設計: [Metrics Viewer](../../design/210_metrics_viewer.jp.md)、§1.1・§2.1。
- タグセットと保存: [Metrics Viewer](../../design/210_metrics_viewer.jp.md)、§3.1・§8・§9.6。操作は[Run分析ガイド](../../design/030_user_guide_analysis.jp.md)、§3。
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
- [TagSetStore.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/TagSetStore.java)
- [WorkspaceManager.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/WorkspaceManager.java)
- [MetricTraceEncoder.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/util/MetricTraceEncoder.java)
- [HttpAccessLogFilter.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/HttpAccessLogFilter.java)
- [MetricsViewerController.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/MetricsViewerController.java)
- [TagSetController.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/TagSetController.java)
- [WorkspaceController.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/WorkspaceController.java)
- [ApiError.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/ApiError.java)
- [DeleteTagSetRequest.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/DeleteTagSetRequest.java)
- [GetMetricsRequest.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/GetMetricsRequest.java)
- [GetMetricsResponse.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/GetMetricsResponse.java)
- [GetRunsResponse.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/GetRunsResponse.java)
- [GetTagSetsResponse.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/GetTagSetsResponse.java)
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
- [RenameTagSetRequest.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/RenameTagSetRequest.java)
- [RunInfo.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/RunInfo.java)
- [RunStats.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/RunStats.java)
- [SaveTagSetRequest.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/SaveTagSetRequest.java)
- [StrictStringDeserializer.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/StrictStringDeserializer.java)
- [SwitchWorkspaceRequest.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/SwitchWorkspaceRequest.java)
- [TagInfo.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/TagInfo.java)
- [TagSetInfo.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/TagSetInfo.java)
- [TagStats.java](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/TagStats.java)
- [additional-spring-configuration-metadata.json](../../../apps/metrics-viewer/src/main/resources/META-INF/additional-spring-configuration-metadata.json)
- [spring-devtools.properties](../../../apps/metrics-viewer/src/main/resources/META-INF/spring-devtools.properties)
- [application.properties](../../../apps/metrics-viewer/src/main/resources/application.properties)
- [initial_tag_sets.json](../../../apps/metrics-viewer/src/main/resources/initial_tag_sets.json)
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
- [TagSetApiIntegrationTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/TagSetApiIntegrationTest.java)
- [TagSetStoreTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/TagSetStoreTest.java)
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
- [TagSetPlaywrightTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/TagSetPlaywrightTest.java)
- [WorkspaceSelectorPlaywrightTest.java](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/WorkspaceSelectorPlaywrightTest.java)

## PRD の一覧

点検が作り直す。手で直さない。

- [PRD064](../../memo/done/064_metricsviewer_idle_ingest_io_10prd.md) — Metrics Viewer 定常取り込み I/O 抑制 PRD
- [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md) — メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| # | ID | 状態 | WSJF | 優先度 | 待ち | 種別 | 工程 | 領域 | 性質 | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [issue-20261103-a3574b](../../management/issues/issue-20261103-a3574b.md) | 未着手 | 3.0 | 中 | — | 成果物 | 監査 | 120_metrics_viewer | 不具合 | 残件有り | DataFetcherの単一のabort枠を独立した呼び出し元が奪い合い、abortが失敗Toastや初期Run選択の喪失になる |
| 2 | [issue-20261103-39ddb2](../../management/issues/issue-20261103-39ddb2.md) | 未着手 | 2.0 | 中 | — | 成果物 | 監査 | 120_metrics_viewer | 不具合 | 残件有り | 初期化中にScreenshotへ切り替えるとinit完了時にmodeが上書きされ、init失敗時はScreenshotから抜けられない |
| 3 | [issue-20261103-73a449](../../management/issues/issue-20261103-73a449.md) | 未着手 | 2.0 | 中 | — | 成果物 | 監査 | 120_metrics_viewer | 不具合 | 判断できない | window内に点が無いtagはgraph blockごと消え、全tag一括のReset View以外では戻せない |
| 4 | [issue-20261103-0ef7b8](../../management/issues/issue-20261103-0ef7b8.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 317_metrics, 311_dev_tools, 316_config, 318_rl_core, 321_agent_core, 120_metrics_viewer, 110_runner_app | 索引 | 未確認 | メトリクスの tag の番号を全体で振り直す（群の番号と群の中の番号）PRD |
| 5 | [issue-20261103-52bda0](../../management/issues/issue-20261103-52bda0.md) | 未着手 | 1.4 | 中 | — | 成果物 | PRD | 318_rl_core, 321_agent_core, 120_metrics_viewer, 311_dev_tools | 索引 | 未確認 | メトリクスの横軸と間隔を数える単位を分ける PRD |
| 6 | [issue-20261103-955e50](../../management/issues/issue-20261103-955e50.md) | 未着手 | 1.2 | 中 | — | 成果物 | PRD | 318_rl_core, 317_metrics, 110_runner_app, 311_dev_tools, 120_metrics_viewer | 索引 | 未確認 | EvalPanel のゲームを trace へ出す — dormant タグの trace を鏡写しインスタンスへ結び付け、trace 行の時刻で振り返る PRD |
| 7 | [issue-20261103-c16aab](../../management/issues/issue-20261103-c16aab.md) | 未着手 | 1.0 | 中 | — | 成果物 | PRD | 320_env_core, 322_env_individual, 312_util, 120_metrics_viewer, 311_dev_tools | 索引 | 未確認 | スカラー閾値越え率の汎用化（Observation 層 / Viewer 層） 暫定 PRD |

着手できる 7 / 待ち 0 / 保留 0 / 次版 0 / 解決 0
<!-- issues:end -->

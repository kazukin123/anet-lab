# Metrics Viewer

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [アプリケーション](../../index.md#アプリケーション)

## 概要・責務

- MetricsマスタをRunごとのSQLiteキャッシュへ取り込み、range queryとブラウザ描画を通して長いRunを比較する。（出典: [Metrics Viewer](../../../design/210_metrics_viewer.jp.md)、§1.1・§2.1）。
- JSONL/gzipをstreaming parseし、L0・factor4のLOD・TagStatsを同一transactionで更新する（出典: [アプリケーションとツール](../../../design/160_applications_and_tools.jp.md)、§2.2 Metrics Viewer）。
- タグセットはTagSetStoreが起動時の検証と保存を、TagSetControllerがAPIを、TagSetMenuが選択・編集の操作を受け持つ。Workspaceをまたぐ保存済みの集合と、ブラウザの未保存の選択を分けて持ち、未保存の変更は選択中でないセットの分もセットごとにブラウザへ残す（出典: [Metrics Viewer](../../../design/210_metrics_viewer.jp.md)、§3.1・§3.2・§7.4・§8・§9.6）。
- Runを比べるために、serverではRunConfigServiceとConfigTextReaderがRunの設定TXTをフレームワークの行規則で読み、RangeStatsQueryが表示範囲の統計をL0から計算する。browserではMainTabBarがGraphsとConfigを切り替え、ConfigTableViewが選んだRunの設定の差分を、KeywordMatcherがRun名・tagキー・設定キーの文字の絞り込みを、TagInfoPanelがtagの全期間と表示範囲の統計を受け持つ（出典: [Metrics Viewer](../../../design/210_metrics_viewer.jp.md)、§3.1・§3.2・§9.7・§9.8）。
- PlotlyはjarのWebJarから配信し、ブラウザが外部へ接続できない環境でもグラフを描ける（出典: [Metrics Viewer](../../../design/210_metrics_viewer.jp.md)、§11.5；[アプリケーションとツール](../../../design/160_applications_and_tools.jp.md)、§6）。
- ソースとMavenのビルド入力は`src/apps/metrics-viewer/`に置き、jarは`apps/metrics-viewer/bin/`に出る（出典: [ソース配置とライブラリの境界](../../../design/011_source_layout.jp.md)、「ライブラリとアプリ」「出力の置き場」）。

## 成立の経緯

- 2026-07-27: Metricsキャッシュは破棄可能な従属導出物とする(マスタ=JSONL、PK=序数)（出典: [ADR 0015](../../../adr/0015-metrics-cache-disposable-derivative.md)、本文）。
- 2026-08-01: 042: MetricsViewer 構造リファクタリング(挙動不変) — 取り込み/照会経路の可読性改善（出典: [RFD0042](../../../memo/done/0042_metricsviewer_refactor_10rfd.md)、題・目的）。
- 2026-08-13: RFD0049: metrics query の supersede 機構（出典: [RFD0049](../../../memo/done/0049_metrics_query_supersede_10rfd.md)、題・目的）。
- 2026-08-21: MetricsViewer SQLite キャッシュ化(Phase 1: OOM 解消・LOD・viewport API)（出典: [RFD0041](../../../memo/done/0041_metrics_sqlite_cache_10rfd.md)、題・目的）。
- 2026-08-21: Metrics Viewerの配置をリリースのapps配置へ移した（出典: [RFD0043](../../../memo/done/0043_release_packaging_10rfd.md)、2. リポジトリ再配置）。
- 2026-08-29: Metrics Viewer 定常取り込み I/O 抑制 RFD（出典: [RFD0064](../../../memo/done/0064_metricsviewer_idle_ingest_io_10rfd.md)、題・目的）。
- 2026-09-05: メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す（出典: [RFD0069](../../../memo/done/0069_metrics_trace_channel_10rfd.md)、題・目的）。
- 2026-09-12: RFD0074: 選択中 Run の色分離（Metrics Viewer）（出典: [RFD0074](../../../memo/done/0074_run_color_distinctness_10rfd.md)、題・目的）。
- 2026-09-12: Run 色は比較ごとの役割とし、分離は選択中 Run の集合に対してだけ保証する（出典: [ADR 0041](../../../adr/0041-run-color-is-comparison-role-not-run-identity.md)、本文）。
- 2026-10-10: 比較するRunの表示粒度を共通Step区間へそろえ、保存LODをfactor4にした（出典: [RFD0089](../../../memo/done/0089_metricsviewer_comparison_resolution_10rfd.md)、題・目標と解決策）。
- 2026-10-12: キャッシュ再構築の入力とSQLをまとめ、全workspaceの検査・背景取り込みと表示workspaceの寿命を分けた（出典: [RFD0090](../../../memo/done/0090_metricsviewer_cache_rebuild_10rfd.md)、題・§2・§4.1〜§4.4）。
- 2026-11-02: タグセットは追跡する作業用ファイルと jar 同梱の初期値に分け、初期値は作業用が無いときだけ写す（出典: [ADR 0067](../../../adr/0067-tag-sets-split-tracked-working-file-and-bundled-initial-copied-only-when-absent.md)、本文）。
- 2026-11-08: 名前を付けて保存したtagの集合（タグセット）をサーバー側の作業用ファイルに保存してworkspaceとブラウザをまたいで共有し、Tagsの下の1行と開閉する一覧で呼び出し（選択全体の置き換え）・Save・Revert・Save As・名前変更・削除・Deselectを行い、選択中のタグセットをlocalStorageに保つ機能と、jarに同梱した初期値を作業用ファイルが無いときだけ写す仕組みを足した。作業用ファイルの既定はworkspaces-dirの直下で、gitで追跡しリリースの梱包からは除く（出典: [RFD0111](../../../memo/0111_metrics_viewer_tag_sets_10rfd.md)、題・§2・D1・D2・D5・D13〜D16）。
- 2026-11-19: Metrics Viewerのソース（Mavenのプロジェクト一式）をsrc/apps/metrics-viewerへ移し、jarの出力先をapps/metrics-viewer/bin/metrics-viewer.jar、Mavenの中間物をout/の下、ログの出力先をトップのdata/の下にした（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.1 D11・D17・D20・§5.2）。
- 2026-11-22: Runを比べる作業に要る機能として、メインエリアのタブ（Graphs・Config）、選んだRunの設定を並べて違う行を見分ける設定表、Run一覧・tag一覧・設定キーの文字の絞り込み、グラフのtag名から出すtagの情報パネル（全期間と表示中の範囲の統計表）を足し、PlotlyをWebJarで同梱した（出典: [RFD0196](../../../memo/0196_metrics_viewer_run_comparison_10rfd.md)、題・§2・D1〜D9）。
- 2026-11-22: 設定表はRunフォルダの`config/config_data.txt`を、UTF-8として読めないバイト列と`$include`を失敗とするほかはC++の`Properties::Load`と同じ行の規則で読む（出典: [ADR 0089](../../../adr/0089-run-config-table-reads-config-data-txt-by-framework-reader-rules.md)、本文）。
- 2026-11-22: query channelの単位を、ブラウザタブと問い合わせの種類（グラフの描画、範囲統計）の組にした（出典: [ADR 0090](../../../adr/0090-query-channel-is-per-tab-and-query-kind.md)、本文）。
- 2026-11-23: 文書の呼び名をRFDに改めるのに合わせ、テストが使う一時フォルダと検証の出力先の名前の旧い呼び名（`prdNNN`）をRFDの4桁の番号（`rfdNNNN`）へ改めた（出典: [RFD0195](../../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md)、§5.2・§6 D1・D9）。

## 選ばなかった形

- 任意factor設定・旧DB変換・新しい永続ストア — 不採用: 比較粒度の障害解消に不要で、マスタから再構築する既存の境界で足りる（出典: [RFD0089](../../../memo/done/0089_metricsviewer_comparison_resolution_10rfd.md)、複雑性監査）。
- 全Runを常にrawで保持する形 — 不採用: マスタ全点を保持せずに正確な区間集約ができる（出典: [RFD0089](../../../memo/done/0089_metricsviewer_comparison_resolution_10rfd.md)、複雑性監査）。
- 汎用ジョブ基盤と高度な予測prefetch — 先送り: 近傍読み込みの実測を先に行う。再開条件: 近傍読み込みでも応答性不足が実測されたとき（出典: [RFD0089](../../../memo/done/0089_metricsviewer_comparison_resolution_10rfd.md)、複雑性監査）。
- 再構築キュー用の新しい永続ストア — 不採用: process内で未処理Runと理由を持ち、起動時にキャッシュから必要な作業を再発見できる（出典: [RFD0090](../../../memo/done/0090_metricsviewer_cache_rebuild_10rfd.md)、§2）。
- 未選択Runの背景取り込みを廃止し、選択時だけ取り込む形 — 不採用: 定常I/Oの削減とは分け、既存の背景取り込みを保つ（出典: [RFD0064](../../../memo/done/0064_metricsviewer_idle_ingest_io_10rfd.md)、0.3 Non-goals）。
- WatchServiceなどのfilesystem通知によるevent駆動化 — 不採用: poll経路の無駄な完全検査を抑える範囲に絞る（出典: [RFD0064](../../../memo/done/0064_metricsviewer_idle_ingest_io_10rfd.md)、0.3 Non-goals）。
- 既存の queryRevision を supersede の連番に流用する形 — 不採用（流用しない）: queryRevision はユーザー操作でしか増えず、Auto Reload の経路では同じ値の POST が同時に飛びうる（出典: [RFD0049](../../../memo/done/0049_metrics_query_supersede_10rfd.md)、1. 現状の事実・2.1 query channelと連番）。
- C++ 側からの gzip 出力 — 不採用: RFD0041 の実測の表に不採用の確定として記されている（理由の記述は無い）（出典: [RFD0041](../../../memo/done/0041_metrics_sqlite_cache_10rfd.md)、Context の実測）。
- schema の不一致・破損・ソース同一性の喪失のときに cache を migration する形 — 不採用: cache はマスタから再構築できる破棄可能な従属導出物なので、警告なしの全破棄・再構築で解決する（出典: [ADR 0015](../../../adr/0015-metrics-cache-disposable-derivative.md)、本文；[RFD0041](../../../memo/done/0041_metrics_sqlite_cache_10rfd.md)、1. 原則 2）。
- タグセットにRunの組・表示設定（Log、percentile、LOD、viewport）・グラフの並び順・パターン指定・workspaceごとの区別を持たせる形 — 不採用（非ゴール）: 範囲の選択（出典: [RFD0111](../../../memo/0111_metrics_viewer_tag_sets_10rfd.md)、§2.2）。
- タグセットの呼び出しで、見えているtagの部分だけ置き換える形／今の選択に足し込む形 — 不採用: 呼んだときの選択（workspace・選択中のRun）で結果が変わる／観点の切替にClear Allが要る（出典: [RFD0111](../../../memo/0111_metrics_viewer_tag_sets_10rfd.md)、D1・§7）。
- タグセットの保存で、見えているtag（選択中のRunのtag）の部分だけ書き、Tags一覧に出ていない部分を残す形 — 不採用: ユーザーが単純さを選んだ（出典: [RFD0111](../../../memo/0111_metrics_viewer_tag_sets_10rfd.md)、D2・§7）。
- タグセットの入口をセレクト＋動詞ボタン、またはチップの横並びにする形 — 不採用: 名前の欄が約93pxに細る／数が増えると隠れ、名前変更・削除が隠れ操作になる（出典: [RFD0111](../../../memo/0111_metrics_viewer_tag_sets_10rfd.md)、D5・§7）。
- タグセットのAPIで名前をパスに入れる形、全体をGET・PUTする形 — 不採用: `%2F`をTomcatが既定で拒む／古い一覧を持つ側の保存が他の変更を消す（出典: [RFD0111](../../../memo/0111_metrics_viewer_tag_sets_10rfd.md)、D9・§7）。
- 選択中のタグセットを外す操作を、一覧の先頭の行・行の専用ボタン・開閉ボタンの中の×に置く形 — 不採用: 呼び出しと見分けがつかない／めったに使わないのに幅を取る／§3.3に無い形でタッチで押しにくい（出典: [RFD0111](../../../memo/0111_metrics_viewer_tag_sets_10rfd.md)、D10・§7）。
- 稼働中に外で変わったタグセットのファイル（手で直したもの、gitの切り替えで変わったもの）を、変更を検知して読み直す形、書く直前に照合して409で拒む形 — 不採用: 読み込みの経路を起動時の1つにし、照合も足さない単純さをユーザーが選んだ（出典: [RFD0111](../../../memo/0111_metrics_viewer_tag_sets_10rfd.md)、D12・§7）。
- タグセットの作業用ファイルの既定を作業ディレクトリ基準の固定値にする形、workspaces-dir直下で親が無ければ起動を止める形 — 不採用: 既存テストが起動で落ち、別のworkspace群での起動で元の群に書くか止まる／存在しないフォルダで起動する既存テストに手当てが要り、workspaces-dirが無いだけで起動が止まる（テストの手当て自体は本当の制約ではなく、ユーザーが手間と厳しさを比べて選ばなかった）（出典: [RFD0111](../../../memo/0111_metrics_viewer_tag_sets_10rfd.md)、D16・§7）。
- 設定表の値をRunフォルダの`json/config_data.json`、またはキャッシュの`json_lines`から読む形 — 不採用: JSONは中身が同じでもCLIと読む元が分かれ、フレームワークの本体の形式でない。キャッシュは取り込み待ちと作り直し中のRunを出せない（出典: [RFD0196](../../../memo/0196_metrics_viewer_run_comparison_10rfd.md)、D2・§5.1）。
- 設定のTXTを書き出し側（`ToPropertiesString`）の形や`java.util.Properties`で読む形 — 不採用: 設定ファイルの様式は読み込み側が正本で、`java.util.Properties`は`\`をエスケープと読みWindowsのパスを壊す（出典: [RFD0196](../../../memo/0196_metrics_viewer_run_comparison_10rfd.md)、D3・§5.1）。
- 設定表でモジュールが読んだキーに「実効」の印を付ける形 — 不採用: ユーザーが要らないと選んだ（出典: [RFD0196](../../../memo/0196_metrics_viewer_run_comparison_10rfd.md)、D1・§5.1）。
- グラフと設定の切り替えをトグルボタン・別ページ・重ねるパネルにする形 — 不採用: 今の表示が押下色だけになる／選んだRunの受け渡しが要り選び直しを追わない／既存のパネルとの重ね順の取り決めが要る（出典: [RFD0196](../../../memo/0196_metrics_viewer_run_comparison_10rfd.md)、D4・§5.1）。
- tagの情報パネルの統計を全期間だけ・表示中の範囲だけにする形、2つを行で分ける形、グラフ見出しの1行の統計を残す形 — 不採用: ズームした範囲で比べられない／全期間と並べて見られない／行が2倍になる／二重の表示と読み違えが残る（出典: [RFD0196](../../../memo/0196_metrics_viewer_run_comparison_10rfd.md)、D5・D6・§5.1）。
- 文字の絞り込みをglobか正規表現にする形、絞り込み中の一括操作で文字の絞り込みも解除する形 — 不採用: 部分一致にも`*`が要る／`[train]`に逃がしが要る／見つけた行を1つずつ選ぶ手間が残る（出典: [RFD0196](../../../memo/0196_metrics_viewer_run_comparison_10rfd.md)、D7・D8・§5.1）。
- Plotlyを`static/`に置く形、buildのときにCDNから取る形 — 不採用: repoが数MB増えライセンスを手で足す／buildがCDNにつながり取得用のプラグインと検査値が要る（出典: [RFD0196](../../../memo/0196_metrics_viewer_run_comparison_10rfd.md)、D9・§5.1）。
- 設定以外のjson・ログ・`dot/`の表示、seedをまとめた帯、あとから系列を足す仕組み、キャッシュの容量、横軸の種類の表示 — 範囲外: 別の関心として別の課題かRFDで扱う（出典: [RFD0196](../../../memo/0196_metrics_viewer_run_comparison_10rfd.md)、§2.2）。
- サーバに問い合わせの種類ごとの取り消しを足す形 — 不採用: channelを分ければ今の取り消しの仕組みのまま満たせる（出典: [ADR 0090](../../../adr/0090-query-channel-is-per-tab-and-query-kind.md)、本文）。
- リリース用のタグセットを書き換え不可の「組み込み」の区画として作業用と並べる形／梱包で作業用の場所に置くだけの形 — 不採用: UI・APIが一回り大きくなり同名の扱いが要る／上書き展開で利用者の作業用ファイルを消す（出典: [ADR 0067](../../../adr/0067-tag-sets-split-tracked-working-file-and-bundled-initial-copied-only-when-absent.md)、Considered Options；[RFD0111](../../../memo/0111_metrics_viewer_tag_sets_10rfd.md)、D13・§7）。
- タグセットの作業用ファイルをViewerのフォルダに置く形、初期値を追跡ファイルと設定キーで指す形 — 不採用: ユーザーがworkspace群と一緒に置くことを選んだ／設定キーが増え、Runnerのconfigフォルダに Viewer のファイルが混ざる（出典: [RFD0111](../../../memo/0111_metrics_viewer_tag_sets_10rfd.md)、D14・D15・§7）。
- アプリのソースを`apps/`に残す形 — 不採用: 置き場の規則が「ライブラリのソースは`src`、アプリはソースも実行時の配置も`apps`」の2つになる（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.1 D11・§4.3）。
- Run 色を workspace の永続属性として保存する — 不採用: 選択する組ごとの色の見分けやすさを解決できず、削除や移動をした Run の色管理も要る（出典: [ADR 0041](../../../adr/0041-run-color-is-comparison-role-not-run-identity.md)、Considered Options）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../../reports/stats/trend.md)の「テスト充足」を見る。

- 共通Step区間への変更でもマスタと過去Run artifact、MinMax/Mean/Band・極値hover・全点TagStats・凡例・Log・percentile・Scroll Lock・Reloadを維持する（出典: [RFD0089](../../../memo/done/0089_metricsviewer_comparison_resolution_10rfd.md)、利用者の要求と受け入れ条件5・6）。
- 再構築をまとめて処理しても同一Stepの複数点・tagの序数・極値の同値時の選択・点数加重平均・非scalar行の順序を変えず、L0・LOD・統計・source位置を同じcommitで確定する（出典: [RFD0090](../../../memo/done/0090_metricsviewer_cache_rebuild_10rfd.md)、§4.1・§5の2〜4）。
- MetricsQueryCoordinator の critical section の中では、Statement.cancel、旧 request の完了待ち、Workspace・DB の lock の取得、query 本体の実行のどれも行わない（出典: [RFD0049](../../../memo/done/0049_metrics_query_supersede_10rfd.md)、2.3 supersede規則）。
- 通常追記ではcache世代を維持し、全再構築だけで世代を変える。commit済みprefix、tag隔離、Run-level errorのtransaction境界を変えない（出典: [RFD0064](../../../memo/done/0064_metricsviewer_idle_ingest_io_10rfd.md)、1.1 正本とcache）。
- priority 3・background 1、単一writer、完成行のtransaction境界でのworkspace切替を維持する（出典: [RFD0064](../../../memo/done/0064_metricsviewer_idle_ingest_io_10rfd.md)、1.2 Scheduling）。
- 色分離の変更でserver、Metricsキャッシュ、HTTP APIを変更しない（出典: [RFD0074](../../../memo/done/0074_run_color_distinctness_10rfd.md)、対象）。
- タグセットを足しても、新しいtagの自動選択、Select All・Clear All・Selected Onlyの動きは変えない。タグセットはMetricsキャッシュにもRunフォルダにも置かない（出典: [RFD0111](../../../memo/0111_metrics_viewer_tag_sets_10rfd.md)、D3・§5のE4・E16）。
- タグセットの作業用ファイルはリリースの梱包に含めず、残っていれば梱包を失敗させる。不正JSONの400への変換はworkspace切替とタグセットのControllerに限り、metrics・prioritizeの失敗の形には影響させない。既存のテストは、サイドパネルの操作行の一覧の期待にTagsの新しい行を足すほかは変えずに通る（出典: [RFD0111](../../../memo/0111_metrics_viewer_tag_sets_10rfd.md)、§6.1・§6.2・§8・E17）。
- 設定表・絞り込み・tagの情報パネルを足しても、Metricsキャッシュの様式と取り込み、既存のAPI、TagStatsとLODの作り方、グラフの描画、文字の絞り込みが空のときの一括操作と`Selected Only`の動き、RunInfoPanelの中身と出し方、既存の`localStorage`の12件の意味を変えない（出典: [RFD0196](../../../memo/0196_metrics_viewer_run_comparison_10rfd.md)、§6.4）。
- 設定ファイルの行の規則を変えるときは、C++の`Properties::Load`と、Viewerが設定表のために持つJavaの読み込みを同じ変更で直す（出典: [ADR 0089](../../../adr/0089-run-config-table-reads-config-data-txt-by-framework-reader-rules.md)、本文；[RFD0196](../../../memo/0196_metrics_viewer_run_comparison_10rfd.md)、§6.3）。
- TagStatsはcommit済みのL0の全点から、範囲統計は指定したstep範囲のcommit済みのL0の全点から作り、LODから導かない（出典: [Metrics Viewer](../../../design/210_metrics_viewer.jp.md)、§12.2の5）。
- LOD を変えるときは、完成した bucket だけを保存する経路と、表示範囲の端の部分 bucket を下位レベルから再集約する経路の両方を検証する（出典: [Metrics Viewer](../../../design/210_metrics_viewer.jp.md)、§8.2・§12.2）。
- 範囲を再取得してもドラッグで選んだ Pan を保持し、signed-log、scroll-lock、reload の既存の挙動を保つ（出典: [RFD0041](../../../memo/done/0041_metrics_sqlite_cache_10rfd.md)、§7）。

## 外の事例

記述なし

## 関係

- 設計: [Metrics Viewer](../../../design/210_metrics_viewer.jp.md)、§1.1・§2.1。
- タグセットと保存: [Metrics Viewer](../../../design/210_metrics_viewer.jp.md)、§3.1・§8・§9.6。操作は[Run分析ガイド](../../../design/030_user_guide_analysis.jp.md)、§3。
- Runの比較（設定表・範囲統計・同梱のPlotly）: [Metrics Viewer](../../../design/210_metrics_viewer.jp.md)、§3.1・§3.2・§9.7・§9.8・§11.5。比べる手順は[Run分析ガイド](../../../design/030_user_guide_analysis.jp.md)、§4。
- ソースと出力の置き場: [ソース配置とライブラリの境界](../../../design/011_source_layout.jp.md)、「ライブラリとアプリ」「出力の置き場」。
- ユーザーガイド: [Run分析ガイド](../../../design/030_user_guide_analysis.jp.md)。
- 含む機能カテゴリ: なし。

## ファイル

点検が作り直す。手で直さない。

- [.checkstyle](../../../../src/apps/metrics-viewer/.checkstyle)
- [.gitignore](../../../../src/apps/metrics-viewer/.gitignore)
- [maven-wrapper.properties](../../../../src/apps/metrics-viewer/.mvn/wrapper/maven-wrapper.properties)
- [build.bat](../../../../src/apps/metrics-viewer/build.bat)
- [checkstyle.xml](../../../../src/apps/metrics-viewer/checkstyle.xml)
- [metrics-viewer-cs-cleanup.xml](../../../../src/apps/metrics-viewer/metrics-viewer-cs-cleanup.xml)
- [metrics-viewer-cs-formatter.xml](../../../../src/apps/metrics-viewer/metrics-viewer-cs-formatter.xml)
- [mvnw](../../../../src/apps/metrics-viewer/mvnw) / [mvnw.cmd](../../../../src/apps/metrics-viewer/mvnw.cmd)
- [pom.xml](../../../../src/apps/metrics-viewer/pom.xml)
- [MetricsViewerApplication.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/MetricsViewerApplication.java)
- [MetricsViewerSettings.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/config/MetricsViewerSettings.java)
- [MetricsCacheDatabase.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/infra/MetricsCacheDatabase.java)
- [MetricsSource.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/infra/MetricsSource.java)
- [RunScanner.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/infra/RunScanner.java)
- [ConfigTextReader.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/ConfigTextReader.java)
- [GzipInputSessions.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/GzipInputSessions.java)
- [GzipSessionReader.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/GzipSessionReader.java)
- [IngestScheduler.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/IngestScheduler.java)
- [LoadingThread.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LoadingThread.java)
- [LodBucket.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodBucket.java)
- [LodIngestWriter.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodIngestWriter.java)
- [LodPageCache.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodPageCache.java)
- [MetricsApiException.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsApiException.java)
- [MetricsIngestor.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsIngestor.java)
- [MetricsQueryCoordinator.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsQueryCoordinator.java)
- [MetricsQueryPlanner.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsQueryPlanner.java)
- [MetricsRangeProjector.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsRangeProjector.java)
- [MetricsRepository.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsRepository.java)
- [MetricsService.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsService.java)
- [QueryCancelledException.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/QueryCancelledException.java)
- [RangeStatsQuery.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/RangeStatsQuery.java)
- [RawFileReader.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/RawFileReader.java)
- [RunConfigService.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/RunConfigService.java)
- [RunWarningRegistry.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/RunWarningRegistry.java)
- [SeriesAvailability.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/SeriesAvailability.java)
- [SourceReader.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/SourceReader.java)
- [TagSetStore.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/TagSetStore.java)
- [WorkspaceManager.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/WorkspaceManager.java)
- [MetricTraceEncoder.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/util/MetricTraceEncoder.java)
- [HttpAccessLogFilter.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/HttpAccessLogFilter.java)
- [MetricsViewerController.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/MetricsViewerController.java)
- [TagSetController.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/TagSetController.java)
- [WorkspaceController.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/WorkspaceController.java)
- [ApiError.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/ApiError.java)
- [DeleteTagSetRequest.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/DeleteTagSetRequest.java)
- [GetMetricsRequest.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/GetMetricsRequest.java)
- [GetMetricsResponse.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/GetMetricsResponse.java)
- [GetRunsResponse.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/GetRunsResponse.java)
- [GetTagSetsResponse.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/GetTagSetsResponse.java)
- [GetWorkspacesResponse.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/GetWorkspacesResponse.java)
- [IngestInfo.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/IngestInfo.java)
- [Issue.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/Issue.java)
- [LodProjection.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/LodProjection.java)
- [LodSummary.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/LodSummary.java)
- [MetricsSeriesRequest.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/MetricsSeriesRequest.java)
- [MetricsSeriesResult.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/MetricsSeriesResult.java)
- [MinMaxProjection.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/MinMaxProjection.java)
- [PrioritizeRunsRequest.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/PrioritizeRunsRequest.java)
- [RangeStats.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/RangeStats.java)
- [RangeStatsRequest.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/RangeStatsRequest.java)
- [RawProjection.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/RawProjection.java)
- [RenameTagSetRequest.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/RenameTagSetRequest.java)
- [RunConfigResponse.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/RunConfigResponse.java)
- [RunInfo.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/RunInfo.java)
- [RunStats.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/RunStats.java)
- [SaveTagSetRequest.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/SaveTagSetRequest.java)
- [StrictStringDeserializer.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/StrictStringDeserializer.java)
- [SwitchWorkspaceRequest.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/SwitchWorkspaceRequest.java)
- [TagInfo.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/TagInfo.java)
- [TagSetInfo.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/TagSetInfo.java)
- [TagStats.java](../../../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/TagStats.java)
- [additional-spring-configuration-metadata.json](../../../../src/apps/metrics-viewer/src/main/resources/META-INF/additional-spring-configuration-metadata.json)
- [spring-devtools.properties](../../../../src/apps/metrics-viewer/src/main/resources/META-INF/spring-devtools.properties)
- [application.properties](../../../../src/apps/metrics-viewer/src/main/resources/application.properties)
- [initial_tag_sets.json](../../../../src/apps/metrics-viewer/src/main/resources/initial_tag_sets.json)
- [favicon.png](../../../../src/apps/metrics-viewer/src/main/resources/static/favicon.png)
- [index.html](../../../../src/apps/metrics-viewer/src/main/resources/static/index.html)
- [metrics-viewer.css](../../../../src/apps/metrics-viewer/src/main/resources/static/metrics-viewer.css) / [metrics-viewer.js](../../../../src/apps/metrics-viewer/src/main/resources/static/metrics-viewer.js)
- [MetricsCacheDatabaseIntegrationTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/infra/MetricsCacheDatabaseIntegrationTest.java)
- [RunScannerTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/infra/RunScannerTest.java)
- [CacheMaintenanceTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/CacheMaintenanceTest.java)
- [ConfigTextReaderTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/ConfigTextReaderTest.java)
- [IngestSchedulerTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/IngestSchedulerTest.java)
- [LoadingThreadTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/LoadingThreadTest.java)
- [LodPageCacheTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/LodPageCacheTest.java)
- [MetricsApiIntegrationTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsApiIntegrationTest.java)
- [MetricsCacheIntegrationTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsCacheIntegrationTest.java)
- [MetricsIngestorIntegrationTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsIngestorIntegrationTest.java)
- [MetricsLodIntegrationTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsLodIntegrationTest.java)
- [MetricsQueryConcurrencyTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsQueryConcurrencyTest.java)
- [MetricsQueryCoordinatorTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsQueryCoordinatorTest.java)
- [MetricsQueryPlannerTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsQueryPlannerTest.java)
- [MetricsRepositorySnapshotIntegrationTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsRepositorySnapshotIntegrationTest.java)
- [MetricsRepositoryStepRangeIntegrationTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsRepositoryStepRangeIntegrationTest.java)
- [MetricsViewerSettingsTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsViewerSettingsTest.java)
- [SeriesAvailabilityTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/SeriesAvailabilityTest.java)
- [TagSetApiIntegrationTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/TagSetApiIntegrationTest.java)
- [TagSetStoreTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/TagSetStoreTest.java)
- [WorkspaceApiIntegrationTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/WorkspaceApiIntegrationTest.java)
- [WorkspaceManagerTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/WorkspaceManagerTest.java)
- [WorkspaceSnapshotIntegrationTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/WorkspaceSnapshotIntegrationTest.java)
- [ComparisonResolutionPlaywrightTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/ComparisonResolutionPlaywrightTest.java)
- [GraphInteractionPlaywrightTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/GraphInteractionPlaywrightTest.java)
- [HttpAccessLogFilterTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/HttpAccessLogFilterTest.java)
- [MetricsPlotPlaywrightTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/MetricsPlotPlaywrightTest.java)
- [MetricsViewerPlaywrightTestData.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/MetricsViewerPlaywrightTestData.java)
- [MetricsViewerPlaywrightTestSupport.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/MetricsViewerPlaywrightTestSupport.java)
- [OutlierRangePlaywrightTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/OutlierRangePlaywrightTest.java)
- [ReloadPlaywrightTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/ReloadPlaywrightTest.java)
- [RunComparisonPlaywrightTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/RunComparisonPlaywrightTest.java)
- [RunListPlaywrightTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/RunListPlaywrightTest.java)
- [ScreenshotModePlaywrightTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/ScreenshotModePlaywrightTest.java)
- [SignedLogPlaywrightTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/SignedLogPlaywrightTest.java)
- [TagListPlaywrightTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/TagListPlaywrightTest.java)
- [TagSetPlaywrightTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/TagSetPlaywrightTest.java)
- [WorkspaceSelectorPlaywrightTest.java](../../../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/WorkspaceSelectorPlaywrightTest.java)

## RFD の一覧

点検が作り直す。手で直さない。

- [RFD0064](../../../memo/done/0064_metricsviewer_idle_ingest_io_10rfd.md) — Metrics Viewer 定常取り込み I/O 抑制 PRD
- [RFD0069](../../../memo/done/0069_metrics_trace_channel_10rfd.md) — メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す
- [RFD0182](../../../memo/0182_source_layout_10rfd.md) — ソース配置の見直し（置き場の規則、Agent の系統ごとのライブラリ、DQN のファイル分割、テストの参照範囲）RFD
- [RFD0195](../../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md) — RFD0195: 文書の呼び名を RFD に改め、番号を 0 埋めの 4 桁にし、今使っている場所をすべて移す PRD
- [RFD0196](../../../memo/0196_metrics_viewer_run_comparison_10rfd.md) — RFD0196: MetricsViewer で Run を比べる作業に要る機能を足す（設定表、Run・tag の絞り込み、tag の情報パネル、Plotly の同梱）

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | RFD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261103-955e50](../../../management/issues/issue-20261103-955e50.md) | 調査中 | 1.2 | — | コード | RFD | 318_rl_core, 317_metrics, 110_runner_app, 311_dev_tools, 120_metrics_viewer | UI | 機能追加 | [RFD0081](../../../memo/0081_evalpanel_episode_trace_10rfd.md) | 残件有り | EvalPanel のゲームを trace へ出す — dormant タグの trace を鏡写しインスタンスへ結び付け、trace 行の時刻で振り返る |
| [issue-20261123-f3cad3](../../../management/issues/issue-20261123-f3cad3.md) | 未着手 | 6.0 | — | 文書 | 監査 | 110_runner_app, 120_metrics_viewer | UI | 不整合 | — | 残件有り | 設計160の静的構造の図が Viewer の SQLite cache と Optuna の DB に同じ node 名 DB を使い 1 つの node にまとまる。コードマップに EvalTracePanel・HeatMapPanel・Conv2dPanel・ErrorDialog が無い |
| [issue-20261125-fd5a4a](../../../management/issues/issue-20261125-fd5a4a.md) | 未着手 | 3.7 | — | コード | 監査 | 120_metrics_viewer | UI | リファクタリング | — | 未確認 | Viewer のブラウザで、詳細キャッシュが 64 MiB を超えたときの解放と先読みの停止、Copy Name の代替のコピー、分割した数列の復号を、通すテストが無い |
| [issue-20261103-0ef7b8](../../../management/issues/issue-20261103-0ef7b8.md) | 未着手 | 1.6 | — | コード | RFD | 317_metrics, 311_dev_tools, 316_config, 318_rl_core, 321_agent_core, 120_metrics_viewer, 110_runner_app | UI | 機能改善 | [RFD0108](../../../memo/0108_metrics_tag_numbering_10rfd.md) | 残件有り | メトリクスの tag の番号を全体で振り直す（群の番号と群の中の番号） |
| [issue-20261103-52bda0](../../../management/issues/issue-20261103-52bda0.md) | 未着手 | 1.4 | — | コード | RFD | 318_rl_core, 321_agent_core, 120_metrics_viewer, 311_dev_tools | UI | 機能改善 | [RFD0096](../../../memo/0096_metrics_step_axis_interval_10rfd.md) | 残件有り | メトリクスの横軸と間隔を数える単位を分ける |
| [issue-20261103-c16aab](../../../management/issues/issue-20261103-c16aab.md) | 未着手 | 1.0 | — | コード | RFD | 320_env_core, 322_env_individual, 312_util, 120_metrics_viewer, 311_dev_tools | UI | 機能追加 | [RFD0154](../../../memo/0154_scalar_threshold_rate_10rfd.md) | 残件有り | スカラー閾値越え率の汎用化（Observation 層 / Viewer 層） |

着手できる 6 / 待ち 0 / 保留 0 / 次版 0 / 解決 0
<!-- issues:end -->

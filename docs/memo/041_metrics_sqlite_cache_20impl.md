# MetricsViewer SQLiteキャッシュ化 実装メモ

## 概要

本メモは
[`041_metrics_sqlite_cache_10prd.md`](041_metrics_sqlite_cache_10prd.md)
を実装の正本として参照する。全点Javaヒープ保持、Kryo snapshot、差分follow API、
追記専用browser bufferを廃止し、Runごとの破棄可能なSQLite Metricsキャッシュ、
factor 16の多段LOD、range-only viewport APIへ置換する。

実装上のschema versionは`1`とする。SQLite接続はJDBCを直接使い、writerは取り込み
blockごと、readerはHTTP request内のRunごとに短命接続とする。`TagStats`は
commit済み有効L0全点の正確な統計として維持する。

## 未決事項監査

- ユーザー判断が必要なブロッカー: 0。
- repo evidenceで解決済み:
  - `RunScanner`は現行`metrics.jsonl`だけをRunとして列挙しているため、
    `.jsonl.gz`を同じRun作業セットへ加える。
  - `LoadingThread`、`MetricsRepository`、`MetricsSnapshot`は全点heap保持と
    Kryo保存を一体で担うため、SQLite writer schedulerとread queryへ責務を置換する。
  - `MetricsViewerController`とbrowserは`runTagMap`差分契約で結合しているため、
    server/client/dummy fixture/Playwrightを同じrange-only契約へ同時に切り替える。
  - 通常用とOptuna用launcherは
    `apps/runner/22_metrics_viewer_java*.bat`の2本である。
  - Metrics Viewerの正準設計説明は
    `docs/design/030_user_guide_analysis.jp.md`、
    `040_development_environment.jp.md`、
    `160_applications_and_tools.jp.md`にある。
- 本メモへ固定する前提:
  - Run/tag error codeは英語のstable identifierとして
    `invalid_json`、`invalid_record`、`invalid_step`、
    `tag_step_regression`、`gzip_corrupt`、`source_read_error`を使う。
  - query slot timeoutは`query_busy`を使う。
  - source identity、進捗bytes、Welfordの`M2`、LOD ordinalは内部契約とし、
    PRDで指定された値だけをPublic APIへ出す。
  - unknown request fieldはshape違反として400にする。
  - 取り込みblockの作業状態はDBから復元したtag単位stateへ閉じ、
    commit成功後だけ次blockへ引き継ぐ。
- PRD範囲外:
  - C++ `SqliteBackend`、bridge、gzip生成、複数runs-dir、Plotly置換。
  - 7.31 GiB Runは自動テスト後の手動受け入れとして残す。

## 主な変更

### 1. 文書と設定

- `CONTEXT.md`へ`TagStats`と`Metricsキャッシュ世代`を追加する。
- ADR 0015の履歴本文は維持し、`TagStats`分離、generation、tag隔離を
  follow-upへ記録する。
- `pom.xml`で`sqlite-jdbc:3.53.1.0`へ固定し、Kryoを削除する。
- Viewer設定を次へ置換し、起動時にfail-fast検証する。
  - `target-points-per-series=4000`
  - `max-points-per-request=500000`
  - `cache-memory-mb=256`
  - `max-concurrent-queries=2`
- 通常用/Optuna用batchと直接起動手順へ`-Xmx1g`を追加する。

### 2. SQLite cacheとsource identity

- `MetricsCacheDatabase`をDB lifecycleの深い境界とし、次を隠蔽する。
  - `application_id`、`user_version=1`、WAL、`busy_timeout=5000`
  - schema作成/検証、`quick_check`
  - Run単位read/write lockと短命connection
  - `.db/-wal/-shm`の安全な全破棄
  - `source_meta`とgeneration
- `MetricsSource`で`.jsonl`優先選択、`.jsonl.gz`、head/tail SHA-256、
  size/mtime、source kind変更を扱う。
- 同一prefixの正常追記だけを継続し、切り詰め、同サイズ上書き、
  より大きい差し替え、旧schema、corrupt DBは全再構築する。
- 旧`metrics_cache.kryo`は削除し、Parquetには触れない。

### 3. streaming ingest、LOD、TagStats

- `MetricsIngestor`が最大1,000,000完成行を1 transactionで処理する。
  巨大なJava行リストは作らず、raw lineを1本ずつparseしてDBへ反映する。
- JSONLは完全改行だけcommitする。gzipは同一`GZIPInputStream`をblock間で保持し、
  DB connectionだけ開閉する。
- tag単位write stateにnext ordinal、previous step、Welford state、
  各level最大15子のLOD accumulatorを復元する。
- factor 16の完成bucketだけをimmutable INSERTし、min/max/lastの
  L0 ordinalと実stepを全levelへ伝播する。
- fatal recordはblock rollback後にRun errorを別transactionで記録する。
  invalid scalar valueは行skip、step逆行はtagだけ隔離する。
- gzip正常中断時は次回起動で全再構築し、corrupt/truncated時は
  commit済みprefixを残してfingerprint変更まで再試行しない。

### 4. schedulerとmetadata

- `LoadingThread`を単一writer schedulerとして維持し、priority集合と背景集合を
  3 block対1 blockでRun round-robin処理する。
- `POST /api/runs/prioritize`はpriority集合を全置換する。
- Run消失時はactive gzip stream、scheduler state、server LRUを破棄する。
- `GET /api/runs.json`はgeneration、Run共通percentage、
  `pending/converting/ready/error`、scalar tag、`TagStats`、issueを返す。

### 5. range queryと固定page LRU

- `MetricsRepository`はSQLite read snapshotからrange queryを組み立て、
  DB schema/ingest stateを変更しないread側境界とする。
- queryはseriesをRunでgroup化し、1 HTTP requestにつきRunごとに1接続を使う。
- fair semaphoreで最大2 queryを実行し、5秒timeout時は
  `503 + Retry-After: 2 + query_busy`を返す。
- quota最低値とwater-fillingを実装し、最低quota合計超過は422の3 fieldを返す。
- 1 seriesにつきL0または単一LOD levelを選び、左右端だけをexact rangeへ合成する。
- serverでMinMax candidateの重複排除/ordinal順、Mean/Band summaryを
  projectionへ閉じる。
- `LodPageCache`は完成済みLODだけを
  `generation/run/tag/level/pageIndex`で1,024 bucket固定pageへ保持する。
  容量はprimitive arrayの実byte数で数え、L0/合成bucketは保存しない。

### 6. Public API

- `POST /api/metrics.json`を`series[]`のinclusive range要求へ置換する。
- request shapeはcontroller境界で検証し、unsafe step、逆range、
  不正`maxPoints`、廃止field、unknown fieldを400にする。
- 有効batchは入力順/件数を維持し、availabilityとissuesを独立して返す。
- f64 stepとf32 valueをlittle-endian Base64 chunkへencodeする。
- 旧`TagTrace`、`MetricsSnapshot`、`Point`、旧diff request/responseを削除する。

### 7. Frontend

- `DataFetcher`はquery単位AbortController、`DataCache`はseriesごとの
  immutableな3画面range windowを所有する。
- selection/viewportの`queryRevision`と、LOD/Log表示の
  `renderRevision`を分離する。
- Run行を即時toggleへ変更し、同一行350ms以内の2回目だけsoloにする。
  初回だけLatestを自動選択し、以後は空選択を維持する。
- viewport変更は150ms debounceし、coverage/解像度不足seriesを1 batchで取得する。
- latest followはrange再要求として実装し、過去rangeではmetricsをpollしない。
- pending/converting Runがある間はruns metadataを2秒pollし、
  選択/表示/変換中seriesだけrangeを2秒更新する。
- `#floating-controls`のScroll Lock直前へLOD mode selectを追加し、
  MinMax/Mean/Bandを同じprojectionから再fetchなしで描画する。
- graph headerへChan合成した`Min / Max / Avg / Std`とRun/tag issueを表示する。

### 8. 設計文書

- `docs/design/160_applications_and_tools.jp.md`のMetrics Viewer構造/sequenceを
  SQLite ingest、range query、viewport windowへ更新する。
- `docs/design/030_user_guide_analysis.jp.md`のRun選択、進捗、LOD mode、
  TagStats表示、`.jsonl.gz`認識を更新する。
- `docs/design/040_development_environment.jp.md`の直接起動を`-Xmx1g`へ更新する。

## テスト

- Public interface / surface:
  - HTTP `GET /api/runs.json`
  - HTTP `POST /api/metrics.json`
  - HTTP `POST /api/runs/prioritize`
  - Viewer起動設定
  - browser DOM、HTTP request、Plotly trace
- 優先behavior:
  1. tracer bullet: 小型JSONLをSQLiteへ取り込み、runs APIで
     generation、ready、100%、正確な`TagStats`を観測する。
  2. range APIでinclusiveなL0 raw projectionを返す。
  3. 16点超の入力で完成LODとactual extrema stepのMinMax projectionを返す。
  4. append/rebuild/generation、transaction rollback、invalid value、
     tag隔離、unknown non-scalarを追加する。
  5. gzip同値性、中断、corrupt、共通percentageを追加する。
  6. availability/issues、quota、priority 3:1、semaphore timeout、
     固定page LRUを追加する。
  7. browserのRun選択、window置換、polling、stale拒否、LOD mode、
     TagStats/error表示を追加し、既存signed-log/scroll-lock契約を維持する。
- TDD順序:
  - 上記を1 behaviorずつ`RED -> 最小GREEN`で進める。
  - RED中にproduction refactorを行わず、各縦スライスがGREENになってから
    重複を深いmoduleへ移す。
  - private helperやDB内部形状だけをmock/testせず、HTTP、設定、
    file/DB observable result、browser DOMから検証する。

## 検証

```powershell
cd viewers\metrics-viewer
mvn -B -Dtest=MetricsCacheIntegrationTest test
mvn -B -Dtest=MetricsApiIntegrationTest test
mvn -B -Dtest=PalettePlaywrightTest test
mvn -B test
mvn -B package
git diff --check
```

手動受け入れでは`run_20260721-201834_cnx-vit128★`を`-Xmx1g`で開き、
PRD 8.2の容量、folder move、progress、extrema step、Mean/Bandを確認する。

### 実施結果（2026-07-27）

- `mvn -B clean test`: 60件成功（backend/API 37件、Playwright 23件）。
- `mvn -B -DskipTests package`: `target/metrics-viewer.jar`生成成功。
- `node --check src/main/resources/static/metrics-viewer.js`: 成功。
- dependency tree: `sqlite-jdbc:3.53.1.0`を確認し、Kryo依存なし。
- `git diff --check`: 成功。
- 7.31 GiB RunによるPRD 8.2の手動受け入れは、本メモで定めた別枠のまま未実施。

## 前提

- 単一Viewer process、単一writer threadとする。
- sourceの正は`metrics.jsonl(.gz)`で、SQLiteは破棄可能なcacheである。
- 通常追記ではgenerationを維持し、全再構築だけで変更する。
- `TagStats`はrange/LODから独立したraw L0全体統計である。
- 旧metrics APIとの後方互換は設けない。
- userのRunner関連未コミット変更と、PRD 041に無関係なuntracked fileは変更しない。

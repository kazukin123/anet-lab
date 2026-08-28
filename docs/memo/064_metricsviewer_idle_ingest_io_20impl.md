# Metrics Viewer 定常取り込み I/O 抑制 実装メモ

## 概要

PRD 064を正本として、属性が変化していないterminal Runの定常pollをsource/cacheの
file attribute確認だけで終了させる。`MetricsIngestor`にprocess-localな検証済み観測を持たせ、
`IngestScheduler`にcycle-local exhaustedを持たせる。`WorkspaceManager.runIngestCycle()`の戻り値は
「cycle中にworkがあったか」ではなく「即時に次cycleを実行すべきか」を表す内部契約へ切り替える。

HTTP、設定、SQLite schema、Metricsマスタ形式、cache generation、frontendの公開契約は変更しない。

## 未決事項監査

- ユーザー判断が必要なブロッカー: 0。
- repo evidenceで解決済み:
  - `MetricsIngestor`はsource検証、block transaction、state確定を所有している。
  - `IngestScheduler`は4-slot cycle、3:1配分、Run作業セットrefreshを所有している。
  - `WorkspaceManager.runIngestCycle()`のbooleanは`LoadingThread`のsleep判断だけに使われている。
  - `MetricsCacheDatabase`はtest subclassで`prepare` / `openWrite`の呼出回数を観測できる。
- 本メモへ固定する前提:
  - Run keyとsource pathはabsolute normalized pathで比較する。
  - cache観測はmain `metrics_cache.db`の存在、size、mtimeだけを使う。
  - attribute取得不能時は観測を使わず、既存の完全検証経路へ戻る。
  - sourceが選択できないRunは、そのRunの検証済み観測を破棄する。
  - 現行設計書のsleep/retry記述だけを新しい内部契約へ同期する。
- PRD範囲外:
  - backoff、WatchService、恒久I/O counter、診断API、size/mtime偽装差替えの稼働中検出。

## 主な変更

### Phase 1: terminal Run fast path

- `MetricsIngestor`へ、Run pathをkeyとするvalidated observationを追加する。
- 観測値はsourceのnormalized path/kind/size/mtime、main cache DBの存在/size/mtime、
  terminal `IngestState`とする。
- `ingestBlock`の先頭で属性一致を判定し、一致時は`SourceReader`生成、fingerprint、
  `MetricsCacheDatabase.prepare` / `openWrite`より前にno-op outcomeを返す。
- 完全検証と必要なcommit、resource closeの後にだけ観測をpublishする。
- block中にsource属性が変化した場合、cache属性を取得できない場合、error永続化に失敗した場合は
  観測をpublishしない。
- `MetricsIngestor.retainRuns`でRun消失をpruneし、source消失時は該当Runを明示的にforgetする。
- workspace snapshot破棄とprocess restartは`MetricsIngestor` instance破棄により自然に全観測を破棄する。

### Phase 2: retry semanticsとcycle exhaustion

- `MetricsIngestor.IngestOutcome`へ`immediateRetry`を追加する。
- `CONVERTING`だけを`immediateRetry=true`とし、`READY`、`ERROR`、fast-path no-opはfalseとする。
- `IngestScheduler`はcycle開始時にexhausted Run集合を初期化する。
- terminal/no-op/失敗Runはそのcycleの残slotから除外する。
- no-opはslotを消費せず同じclass、次に反対classの候補へ譲る。terminal commitはslotを消費する。
- `CONVERTING`はexhaustedにせず、既存のpriority 3 : background 1とblock上限で継続する。
- `WorkspaceManager.runIngestCycle()`は4 slotの`immediateRetry`をOR集約する。
- workspace epoch変更だけは従来どおりtrueを返し、旧snapshotの残slotを捨てる。
- `LoadingThread`は即時再試行が不要なcycleで10秒sleepし、runtime failure後も従来どおりsleepする。

### ドキュメント同期

- `docs/design/210_metrics_viewer.jp.md`と`docs/design/160_applications_and_tools.jp.md`のうち、
  旧didWork基準のsleep説明とscheduler説明だけを新契約へ同期する。
- `CONTEXT.md`、ADR、PRD 041、設定資料は変更しない。

## テスト

- Public interface / surface:
  - `MetricsIngestor.ingestBlock`の`IngestOutcome`。
  - `IngestScheduler.runNextBlock`が行うRun選択とingest呼出順。
  - `WorkspaceManager.runIngestCycle`から`LoadingThread`へ渡る即時再試行boolean。
- 優先 behavior:
  1. tracer bullet: 同一`MetricsIngestor`の不変READY 2回目が、reader生成、fingerprint、
     DB prepare/openWriteなしでno-opになる。
  2. ERROR、不変cache、process restart、Run消失・再登録の観測lifecycle。
  3. append、truncate、同サイズoverwrite、source kind変更、cache属性変更で完全検証へ戻る。
  4. block中のsource変更とerror永続化失敗では観測をpublishしない。
  5. terminal/no-op Runは4-slot cycleで最大1回だけ検査される。
  6. terminal/no-opが空けたslotを反対classの`CONVERTING`が使う。
  7. 両classの`CONVERTING`は3:1順序と既存block上限を維持する。
  8. READY commitはsleep、CONVERTINGとworkspace epoch変更は即時cycle、runtime failureはsleepする。
- TDD順序:
  - behaviorごとに1つの失敗テストを追加してREDを確認する。
  - そのbehaviorを通す最小実装だけを加えてGREENにする。
  - GREEN後にだけ重複を整理し、関連testを再実行してから次behaviorへ進む。
  - production診断counterやtest-only public APIは追加しない。必要な観測はpackage-private collaboratorと
    test subclass/mockを使う。

## 実装前後のI/O実測

- `target/prd064-io/`配下に50 MiB以上の停止済みraw Run fixtureを作る。
- 変更前のappを同fixtureで起動し、cache READY後15秒warm-upしてからWindows process counterを
  1秒間隔で60 sample取得する。
- 実装後も同じJDK、heap、port、fixture、測定時間で再測定する。
- 前後でDB size/mtime、read/write総量近似・平均・最大、1 MiB/s超sampleを保存し、
  read/write各90%以上削減、周期spike 0、DB属性不変を確認する。
- 測定用fixtureと結果は`target/`内だけに置き、tracked Run artifactを変更しない。

## 検証

```powershell
cd C:\dev\anet-lab\apps\metrics-viewer
mvn -B -Dtest=MetricsIngestorIntegrationTest test
mvn -B -Dtest=IngestSchedulerTest,LoadingThreadTest,WorkspaceManagerTest test
mvn -B test

cd C:\dev\anet-lab
git -c safe.directory=C:/dev/anet-lab diff --check -- `
  apps/metrics-viewer/src/main `
  apps/metrics-viewer/src/test `
  docs/design/160_applications_and_tools.jp.md `
  docs/design/210_metrics_viewer.jp.md `
  docs/memo/064_metricsviewer_idle_ingest_io_20impl.md
```

## 前提

- JSONL/gzipの正本契約、SQLiteの破棄可能cache契約、短命connectionを維持する。
- 未選択READY Runも10秒ごとにattributeだけ確認する。
- `MAX_BLOCK_LINES=1_000_000`とactionable block間の3:1配分を維持する。
- main DB以外の`-wal` / `-shm`属性はfast-path identityに含めない。
- 内容、size、mtimeがすべて同一の差替えはprocess restartまで検出しない。
- 無関係なdirty/untracked fileを変更しない。

## 実装結果

- targeted test:
  - `MetricsIngestorIntegrationTest`: 22件成功。
  - `IngestSchedulerTest`: 8件成功。
  - `WorkspaceManagerTest`、`WorkspaceSnapshotTest`、`LoadingThreadTest`: 合計18件成功。
- full test: `mvn -B test`で165件成功、failure/error/skipは0件。
- 同一fixture実測:
  - source: 56,477,780 bytesの停止済みraw Run 1件。
  - 条件: 同一JDK、port、workspace、cache、READY後15秒warm-up、1秒間隔60 sample。
  - 変更前HEAD: read 10,627,632 bytes、平均177,127.2 bytes/s、最大1,180,848 bytes/s、
    1 MiB/s超9 sample、write 0 bytes。
  - 実装後: read 0 bytes、write 0 bytes、1 MiB/s超0 sample。
  - read削減率は100%。writeは変更前から0 bytesのため削減率を算出できないが、実装後も0 bytesで
    増加していない。
  - cache DBは両測定ともsize 21,204,992 bytes、mtime
    `2026-08-27T17:37:54.2512834Z`で測定前後不変。
- 実測artifactは`apps/metrics-viewer/target/prd064-io/`だけに保存し、tracked Run artifactは
  変更していない。

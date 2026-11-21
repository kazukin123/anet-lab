# Metrics Viewer 定常取り込み I/O 抑制 PRD

> 番号: 064
> 状態: **implementation ready**
> 対象: `apps/metrics-viewer` backend
> 起点: 2026-08-28、Metrics Viewer の `javaw.exe` で累積約 5.94 TB read / 0.90 TB write を観測し、停止・未選択 Run への定期アクセス有無を診断した。
> 関連: [041_metrics_sqlite_cache_10prd.md](041_metrics_sqlite_cache_10prd.md)、[ADR 0015](../../adr/0015-metrics-cache-disposable-derivative.md)、`CONTEXT.md`「Run作業セット」「Metricsマスタ」「Metricsキャッシュ」「Metricsキャッシュ世代」

---

## 0. Goal anchor

### 0.1 問題

Metrics Viewer は browser 接続の有無にかかわらず `LoadingThread` を起動し、選択中
workspace の Run作業セット全体を定期走査する。現行実装では、Metricsキャッシュが
`ready` で Metricsマスタにも変化がない Run に対しても、1回の検査で次を行う。

1. Metricsキャッシュへ接続し、header、schema、`source_meta`を検査する。
2. Metricsマスタの先頭とcommit済み末尾を各最大64 KiB読み、fingerprintを検証する。
3. raw Metricsマスタを開いてcommit済みoffsetへ移動する。
4. 新規完成行が0件でもwrite connectionを開き、同じ`source_meta`をUPSERTしてcommitする。

さらに1回のworkspace ingest cycleは4 slotあり、全Runがterminal/no-opの場合も同じRun集合を
各slotで再検査する。このため、変化のないRunが1件だけでも約10秒ごとに最大4回の完全検査と
no-op writeが発生する。

診断時の実プロセスでは、現在workspaceにRunが1件だけの状態で、25秒間に約10秒間隔のI/O
spikeを観測した。Process counterの最大値はread約11.35 MiB/s、write約3.97 MiB/sだった。
画像の累積値とprocess起動時刻から求めた平均はread約10.84 MiB/s、write約1.64 MiB/sである。
これらは物理disk bytesではなく、OS cacheを含むprocessの論理I/Oである。

短時間測定時のRunはMetricsマスタへ追記中だったため、観測値には正当なcache取り込みも含む。
一方、terminal/no-op Runにも同じ完全検査・write経路が適用されることはコード経路から確定している。

### 0.2 Goal

Metricsマスタへのライブ追従、Metricsキャッシュの再構築判断、初回/backlog変換のthroughputを
維持しながら、属性が変化していない`ready` / `error` Runの定常処理をfile attribute確認だけにする。

具体的には次を達成する。

1. 変化のないterminal RunではMetricsマスタ本文、fingerprint、SQLite connectionへ到達しない。
2. terminal/no-op Runを同一4-slot cycle内で再試行しない。
3. 未処理backlogがある`converting` Runだけは、従来どおりsleepなしで処理を継続する。
4. 通常のappend、truncate、overwrite、source kind変更、cache変更を検出したら、既存の完全検証へ戻る。
5. Task Managerで確認された約10秒周期のread/write spikeが、停止済みRunで消えることを実測する。

### 0.3 Non-goals

- Metricsマスタ=`metrics.jsonl(.gz)`、Metricsキャッシュ=`metrics_cache.db`という正本/派生物契約の変更。
- Metricsキャッシュschema、`source_meta`、cache generation、HTTP JSON形式の変更。
- `MAX_BLOCK_LINES=1,000,000`の縮小による見かけ上のI/O削減。
- priority/backgroundの3:1配分、単一writer、短命DB connectionという基本構造の廃止。
- 未選択Runの背景取り込み廃止、選択時lazy ingestへの変更。
- 完了Runの指数backoff、Java `WatchService`、filesystem通知によるevent駆動化。
- browser tab単位のpriority、page close時のpriority解除、複数tab競合の解決。
- 恒久的なI/O counter、診断HTTP API、運用dashboardの追加。
- Plotly、Run/Tag一覧、Auto Reload、ingest progress表示などfrontend挙動の変更。
- 内容、size、mtimeをすべて同一に偽装したMetricsマスタ差替えのprocess稼働中即時検出。
- 物理disk I/OまたはSSD write量を測定する基盤の追加。

## 1. Existing contracts to preserve

### 1.1 正本とcache

- Metricsマスタだけを正とし、Metricsキャッシュはいつでも全破棄・再構築可能な従属導出物とする。
- source kind、正常追記、truncate、同サイズoverwrite、より大きい別sourceへの差替え、schema不一致、
  cache接続不能に対する既存の判定結果を変えない。
- 全再構築だけがMetricsキャッシュ世代を変更し、通常追記では同じ世代を維持する。
- commit済みprefix、LOD、`TagStats`、tag隔離、Run-level errorのtransaction境界を変えない。

### 1.2 Scheduling

- Run作業セットは現在workspaceの`runs/`直下にあり、Metricsマスタを持つRun全体とする。
- priority側3 block、background側1 blockを1 cycleとする。
- 片側にactionable Runがなければ、もう片側へslotを譲る。
- 各取り込みblockの上限は1,000,000完成行とする。
- workspace切替要求が来た場合は、完成行のtransaction境界でyieldし、旧workspaceの残slotを捨てて
  新workspaceのcycleへ直ちに進む。
- browserが開かれていなくてもbackend取り込みは動作する。

### 1.3 Poll cadence

- 即時処理可能なbacklogがない場合のsleepは10秒を維持する。
- 未選択`ready` Runも10秒ごとに確認するが、attribute不変なら本文やDBへ触れない。
- priorityは処理可能Runの配分だけに影響し、terminal/no-op fast pathを無効化しない。

## 2. Decisions

| ID | Decision |
|---|---|
| D1 | `MetricsIngestor`が完全検証済みRun観測をprocess memoryだけに保持する。 |
| D2 | terminal Runのsource/cache属性が観測と一致すれば、SourceReader、fingerprint、DB prepare/writeより前で返す。 |
| D3 | 観測は`ready`または`error`の確定状態だけに作り、`pending` / `converting`には作らない。 |
| D4 | source/cache属性変化、process restart、観測済みRunの消失、workspace snapshot破棄では完全検証へ戻る。 |
| D5 | ingest結果へ`immediateRetry`を追加し、未処理backlogを表す`converting`だけを即時再試行する。 |
| D6 | `ready` / `error` / no-op Runは現在cycleでexhaustedとし、残slotでは再検査しない。 |
| D7 | 3:1はactionable backlog間の配分とする。terminal/no-op確認は配分slotを継続消費しない。 |
| D8 | low-frequency fingerprint audit、backoff、WatchServiceは導入しない。 |
| D9 | 自動回帰と同一fixtureのbefore/after process I/O測定を完了条件にする。 |

## 3. Phase 1: terminal Run fast path

### 3.1 Ownership

完全検証済み観測は`MetricsIngestor`が所有する。

理由:

- `IngestScheduler`はRun配分の所有者であり、source fingerprintやDB lifecycleを知るべきではない。
- `MetricsCacheDatabase`はDB lifecycleの境界だが、Metricsマスタ読取結果とscheduler結果を所有しない。
- `MetricsIngestor`は現在も`source → validation → read → transaction → ingest state`を連続して統括する。

観測を永続化しない。Metricsキャッシュやsidecarへ新しいfield/fileを追加しない。

### 3.2 Validated Run observation

`MetricsIngestor`は正規化済みRun pathをkeyに、少なくとも次の値を保持する内部recordを持つ。

| 区分 | 値 |
|---|---|
| source | 選択sourceの正規化path、kind、size、mtime |
| cache | `metrics_cache.db`の存在、size、mtime |
| result | 完全検証後に確定した`IngestState` (`READY`または`ERROR`) |

このrecordはpublic API、設定、DB schemaではない。class名やmember名は実装上選べるが、所有者、
比較値、生成/破棄条件は本節の契約に従う。

`metrics_cache.db-wal` / `metrics_cache.db-shm`は短命connectionの内部artifactであり、fast pathの
identityにはしない。main DBの属性変化を完全検証への復帰条件とする。

### 3.3 Fast-path order

1 Runの定期確認は次の順序で行う。

1. 従来どおり`.jsonl`優先でMetricsマスタを選び、file attributesを1回取得する。
2. `metrics_cache.db`の存在とfile attributesを取得する。
3. Run pathに完全検証済み観測があり、source/cache属性がすべて一致するか判定する。
4. 一致する場合、保存済みstateを使って`didWork=false`、`immediateRetry=false`で返す。
5. 一致しない場合、観測を破棄して既存のSourceReader/DB prepare/fingerprint/ingest経路を実行する。

fast pathでは次を行ってはならない。

- `MetricsSource.headSha256()` / `sha256Before()`。
- Metricsマスタ本文のopen、skip、read。
- SQLite JDBC connectionのopen。
- header/schema/`source_meta` query。
- `source_meta` UPSERT/DELETE、transaction commit。
- cache generation、ingest state、DB mtimeの更新。

file existence、kind選択、size、mtime取得は許可する。これは「content read 0」のcontentに含めない。

### 3.4 Observation publication boundary

観測は次の全条件を満たした後だけ作る。

1. 既存の完全検証が成功している。
2. 必要なread/write transactionがcommit済みである。
3. write/read connectionとSourceReaderがclose済みである。
4. 最終stateが`ready`または`error`である。
5. connection close後にsource/cache属性を再取得できる。
6. source属性が、そのblock開始時に選択したsource snapshotと一致する。

block実行中またはclose直後にsource属性が変化していた場合は観測を作らず、次cycleで完全経路を
再実行する。未処理source変化が明白な場合はsleep前の即時cycle対象としてよい。

errorを永続化するtransaction自体が失敗した場合も観測を作らない。同一失敗をfast pathで隠さず、
従来どおり次回完全経路で再試行する。

### 3.5 Invalidation

次の場合は該当Runの観測を破棄する。

- source path/kind/size/mtimeのいずれかが変わった。
- `metrics_cache.db`が生成、消失、size変更、mtime変更した。
- `RunScanner`の走査で観測済みRunの消失を確認した。
- workspace snapshotがretireされ、対応する`MetricsIngestor`が破棄された。
- Metrics Viewer processを再起動した。

`IngestScheduler.refreshWorkSet()`は、現在存在する正規化Run path集合を`MetricsIngestor`へ渡し、
消失Runの観測をpruneする。Runが後で同じ名前で再登録されても、最初の検査は完全経路になる。

1回の10秒interval内でRunを削除し、同じpath・size・mtimeで再登録して走査から消失を隠す操作は、
size/mtime偽装差替えと同じdeferred caseとする。WatchServiceは追加しない。

### 3.6 Source changes

fast pathから完全経路へ戻った後はPRD 041の既存契約をそのまま適用する。

| 変化 | 挙動 |
|---|---|
| raw JSONL正常追記 | 同じgenerationへ差分をcommitする。 |
| source truncate | cacheを全再構築する。 |
| 同サイズoverwrite | mtime変化で完全検証し、fingerprint不一致なら全再構築する。 |
| より大きい別source | size/mtime変化で完全検証し、prefix不一致なら全再構築する。 |
| `.jsonl` / `.jsonl.gz`切替 | kind変化で全再構築する。 |
| gzip属性変化 | immutable source変更として全再構築する。 |
| cache DB属性変化 | header/schema/source metadataを再検証し、無効なら全再構築する。 |
| error source属性変化 | error fast pathを解除し、既存の回復/再構築判定を実行する。 |

内容、size、mtimeをすべて同一に保ったprocess稼働中の書換えは、次のprocess起動時の完全検証まで
検出しなくてよい。この形の実事故が発生した場合だけ、低頻度fingerprint auditを別PRDで検討する。

## 4. Phase 2: retry semantics and cycle exhaustion

### 4.1 Ingest outcome

internal Java contractのingest結果へ、既存の`didWork`と`state`に加えて`immediateRetry`を持たせる。

| 結果 | `didWork` | `immediateRetry` |
|---|---:|---:|
| 属性不変fast path | false | false |
| 新規行を読み、snapshot末尾まで到達して`ready` | true | false |
| block上限等で未処理backlogが残り`converting` | true | true |
| source不変の`error` | false | false |
| 新しいfatal errorを永続化 | true | false |

`pending`は即時再試行判断として使用しない。正常なsourceを処理したblockは、終了時に`ready`または
`converting`へ確定する。workspace切替による新snapshotへの移行はingest stateとは別に、従来どおり
即時cycleを要求する。

`didWork`と`immediateRetry`を同一概念として扱わない。小さなappendをcommitしたことはworkだが、
既にsnapshot末尾へ到達していれば、確認目的だけの即時cycleは不要である。

### 4.2 Per-cycle exhausted set

`IngestScheduler`は4-slot cycleの開始時に、cycle-localなexhausted Run集合を空にする。

- `ready`、`error`、またはfast-path no-opを返したRunを現在cycleでexhaustedにする。
- exhausted Runは残slotの候補列挙から除外する。
- `converting`を返したRunはexhaustedにせず、round-robinと3:1配分に従って再処理できる。
- no-op Runをskipしたslotは、同じclassの次Run、次に反対classのactionable Runへ譲る。
- 実際にdata/errorをcommitしたterminal Runはそのslotを使用してよいが、残slotでは再検査しない。
- cycle終了後にexhausted集合を破棄する。次の10秒pollでは全Runのattributeを再確認する。

`refreshWorkSet()`、priority snapshot、priority/background cursor、scan中に追加されたpriorityをpruneしない
既存の競合対策は維持する。

### 4.3 LoadingThread sleep condition

`WorkspaceManager.runIngestCycle()`は4 slotの`immediateRetry`をOR集約する。`LoadingThread`は
「cycle中に何かcommitしたか」ではなく、「直ちに処理すべきbacklogまたはworkspace切替が残るか」で
sleepを決める。

- `immediateRetry=true`: sleepせず次cycleへ進む。
- 全Runがterminal/no-opで`immediateRetry=false`: 10秒sleepする。
- workspace epoch変更: 旧snapshotの残slotを捨て、sleepせず新snapshotへ進む。
- cycle例外: 従来どおりerrorを記録し、10秒sleep後に再試行する。

これにより、小さなappendを`ready`まで取り込んだ直後の「4回no-op確認cycle」を削除する。

### 4.4 3:1 interpretation

3:1はactionable ingest blockの公平性契約とする。属性確認やterminal/no-op結果を3:1の成果として
数えない。priorityに処理可能backlogがなくbackgroundにだけある場合はbackgroundがslotを使用し、
逆も同様とする。

両classに継続的な`converting` Runがある場合、8 actionable blockの処理順は従来テストと同じ
`priority, priority, priority, background`の反復になること。

## 5. Interface and compatibility

### 5.1 Unchanged public surface

次は変更しない。

- `GET /api/runs.json`、`POST /api/metrics.json`、`POST /api/runs/prioritize`。
- HTTP request/response field、ingest state文字列、percentage、generation。
- `application.properties`とlauncher引数。
- MetricsマスタJSONL/gzip形式。
- SQLite application ID、user version、table/column、`source_meta` key。
- frontend DOM、localStorage key、poll interval、Auto Reload interval。

### 5.2 Internal surface

変更対象となるinternal seamは次に限定する。

- `MetricsIngestor`: validated observation、fast path、observation prune、結果の`immediateRetry`。
- `IngestScheduler`: cycle-local exhausted集合とactionable block集約。
- `WorkspaceManager` / `LoadingThread`: `immediateRetry`に基づくcycle継続判定。

test都合だけのpublic API、production診断counter、singleton/global cacheは追加しない。package-private constructor、
spy、mock、temporary fixtureを使って回帰を固定する。

## 6. Test plan

### 6.1 Phase 1 targeted tests

`MetricsIngestorIntegrationTest`を中心に、次を追加する。

1. raw Runを`ready`まで取り込んだ後、同じ`MetricsIngestor`で再実行すると`didWork=false`、
   `immediateRetry=false`になり、DB prepare/openWriteとfingerprint/content readが呼ばれない。
2. fast pathを複数回実行しても`metrics_cache.db`のsize/mtime、generation、committed offset、stateが変わらない。
3. `error`を確定した同一sourceの2回目以降も完全経路へ入らない。
4. source appendでfast pathが解除され、追記行を同じgenerationへcommitする。
5. truncate、mtimeが変わる同サイズoverwrite、より大きい別source、source kind変更で既存の再構築判定へ戻る。
6. cache DBのsize/mtime変更、削除、再生成で完全検証へ戻る。
7. 新しい`MetricsIngestor` instanceではmemory観測がなく、初回完全検証する。
8. Run消失を`retainRuns`相当のpruneで確認した後、同名Runを再登録すると初回完全検証する。
9. block処理中にsource属性が変化した場合はterminal観測をpublishしない。
10. error永続化に失敗した場合は観測をpublishしない。

fingerprint/content read 0は、production counterではなく既存classのspy/mockまたはpackage-private collaboratorで
呼出回数として検証する。file attribute readまで0とはしない。

### 6.2 Phase 2 scheduler tests

`IngestSchedulerTest`、`LoadingThreadTest`、必要に応じて`WorkspaceSnapshotIntegrationTest`へ次を追加する。

1. terminal/no-op Runが1件の場合、4 slot cycleでingest試行は1回だけ。
2. terminal/no-op Runが複数の場合、各Runの試行は1 cycle最大1回。
3. priority側がterminal、background側が`converting`の場合、空いたslotをbackgroundが使う。
4. background側がterminal、priority側が`converting`の場合、priorityが空いたslotを使う。
5. 両側が継続`converting`の場合、actionable blockは3:1順序を維持する。
6. 小さなappendをcommitして`ready`になったcycleは、work有りでも`immediateRetry=false`となりsleepする。
7. block上限で`converting`が残るcycleはsleepせず継続する。
8. workspace epoch変更はterminal/no-op状態に関係なくsleepせず新snapshotへ進む。
9. runtime failureは従来どおりlog後にsleepし、次cycleで回復を試みる。
10. scan開始後に追加されたpriorityを古いscan結果でpruneしない既存回帰を維持する。

### 6.3 Regression commands

実装完了時は少なくとも次を実行する。

```powershell
cd C:\dev\anet-lab\apps\metrics-viewer
mvn -B -Dtest=MetricsIngestorIntegrationTest,IngestSchedulerTest,LoadingThreadTest test
mvn -B test

cd C:\dev\anet-lab
git -c safe.directory=C:/dev/anet-lab diff --check -- `
  apps/metrics-viewer/src/main `
  apps/metrics-viewer/src/test
```

テスト選択構文が使用中のSurefire versionで解釈できない場合は、3 classを個別実行してからfull suiteを実行する。
frontend変更はないが、full Maven suite内のPlaywrightも回帰として成功させる。

## 7. Manual I/O acceptance

### 7.1 Fixture

- `target/`配下の一時workspaceを使用し、tracked Run artifactを変更しない。
- Runは50 MiB以上の停止済みraw `metrics.jsonl` 1件とする。
- Metricsキャッシュを`ready`まで構築し、source fileを測定中変更しない。
- browser tab、Auto Reload、外部HTTP clientから測定中にrequestを送らない。
- service起動後、`GET /api/runs.json`で`ready`を確認してから15秒warm-upする。

### 7.2 Measurement

実装前と実装後で同じfixture、JDK、heap、port、測定時間を使う。listen portから対象PIDを特定し、
Windows process counterの`IO Read Bytes/sec`、`IO Write Bytes/sec`を1秒間隔で60 sample取得する。

測定直前と直後に次を保存する。

- `metrics_cache.db`のsizeとmtime。
- 60 sampleのread/write総量近似、平均、最大。
- 1 MiB/sを超えたsampleのtimestampと件数。

これはlogical process I/Oの比較であり、物理disk bytesの主張には使わない。

### 7.3 Pass criteria

次をすべて満たしたときだけ受入成功とする。

1. 実装後60秒で`metrics_cache.db`のsize/mtimeが不変。
2. read総量が同一fixtureの実装前比で90%以上減少。
3. write総量が同一fixtureの実装前比で90%以上減少。
4. 実装後にreadまたはwriteが1 MiB/sを超える約10秒周期sampleが0件。
5. 測定中のRun state、generation、tag count、`TagStats`が測定前後で不変。

環境ノイズで2〜4だけが失敗した場合、同条件で最大3回まで再測定し中央値を採用してよい。
DB size/mtimeが変化した場合はノイズ扱いせず失敗とする。

## 8. Implementation order and stop points

### Phase 1

1. 実装前I/O baselineを同一fixtureで保存する。
2. `MetricsIngestor`へvalidated observationとfast pathを追加する。
3. Run消失/workspace lifecycleのpruneを接続する。
4. Phase 1 targeted testsとfull Maven suiteを通す。

Phase 1だけでもfingerprint、DB検査、no-op writeという主要I/Oを除去できる。Phase 2で問題が見つかった場合、
Phase 1をgreenな停止点として残せる。

### Phase 2

1. ingest/cycle outcomeへ`immediateRetry`を追加する。
2. cycle-local exhaustedとactionable 3:1配分を実装する。
3. `LoadingThread`のsleep判断を`immediateRetry`へ切り替える。
4. Phase 2 targeted testsとfull Maven suiteを通す。
5. 60秒manual I/O acceptanceを実行し、before/after結果を実装報告へ記録する。

Phase 2はmetadata確認の4重実行と確認cycleを削減する。Phase 1のfast pathやcache correctnessを変更しない。

## 9. Complexity audit

### 9.1 Aggregate excess

| Mechanism | Verdict | 切った場合に戻る実害 |
|---|---|---|
| `MetricsIngestor` validated observation | keep | 10秒ごとのfingerprint、schema検査、no-op DB write |
| state別`immediateRetry`とcycle exhausted | keep | 同一Run集合の4重検査と確認cycle |
| 自動回帰＋60秒実測 | keep | 元のTask Manager症状に対する効果未証明 |
| 完了Run backoff | cut | fast path後のattribute確認だけで実害が未証明 |
| WatchService | cut | lifecycle・overflow・platform分岐を増やすだけで必須でない |
| tab別priority lifecycle | cut | fast path後は今回のI/O原因ではない |
| 恒久診断API/counter | cut | 一時的な受入測定で目的を満たす |
| 新設定・HTTP・DB schema | cut | 固定契約で足り、利用者選択を増やす必要がない |

### 9.2 Requirement reality

- 定常I/O、4重検査、no-op writeは実プロセスとコードで確認済み: keep。
- ライブ追従、通常差替え検出、backlog throughputは既存運用契約: keep。
- size/mtime偽装差替えは未発生の想定: incident発生までdefer-behind-gate。

### 9.3 Decision residue

fast pathとcycle exhaustedを採用した結果、stale priorityは定常重I/Oの原因ではなくなった。
priority lifecycle修正を本PRDへ残さない。3:1はterminal確認ではなくactionable blockの配分として残す。

### 9.4 Minimal-solution diff

必要なproduction差分は次の5点だけとする。

1. `MetricsIngestor`のvalidated observation。
2. 属性一致時のDB/source-content前fast return。
3. 消失Run/workspace lifecycleでの観測破棄。
4. `immediateRetry`とcycle exhausted。
5. `LoadingThread`のsleep判定変更。

### 9.5 Phase independence

Phase 1は重いI/Oを単独で削減し、Phase 2は重複metadata確認を単独で追加削減する。
どちらの完了点でも既存cache dataやpublic behaviorを悪化させず停止できる。

### 9.6 Success measurability

完了は「コードを変更した」ことではなく、test call count、DB mtime不変、logical I/O 90%以上削減、
10秒周期spike 0件で判定する。

## 10. Documentation impact

- `CONTEXT.md`の既存語彙で本契約を表現でき、新しいdomain termはない。変更しない。
- ADR 0015の正本/派生cache、短命connection、source不一致時再構築という決定を変更しない。
- 完全検証の呼出頻度とscheduler retryは容易に戻せる内部最適化であり、新ADRの3条件を満たさない。
- PRD 041は当時の設計記録として変更しない。本PRDが定常poll時の追加契約を定義する。

## 11. Completion checklist

- [ ] Phase 1 targeted testsがgreen。
- [ ] Phase 1後のfull Maven suiteがgreen。
- [ ] Phase 2 targeted testsがgreen。
- [ ] Phase 2後のfull Maven suiteがgreen。
- [ ] `git diff --check`がgreen。
- [ ] HTTP、設定、DB schema、frontend、Metricsマスタ形式に差分がない。
- [ ] 無関係なdirty/untracked fileを変更していない。
- [ ] 60秒manual I/O acceptanceの全条件を満たした。
- [ ] 実装報告にbaseline/finalのread/write総量、最大sample、DB size/mtimeを記録した。

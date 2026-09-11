# PRD 049: metrics query の supersede 機構

- 起票日: 2026-08-13
- 状態: implementation ready
- 対象: Metrics Viewer（`apps/metrics-viewer`）のquery同時実行制御とfrontend
- Topic Issue: 不具合対応 `#24`、MetricsViewer `#5`
- 関連: ADR 0023（本PRDで新設）、ADR 0015（Metricsキャッシュ=破棄可能導出物）、ADR 0021（Run分類=workspaceフォルダ）、PRD 046（workspace機構）
- 設計文書: `docs/design/210_metrics_viewer.jp.md`

## Context（背景・目的）

Metrics ViewerでRunやtagを素早く切り替えると、`POST /api/metrics.json`が約5秒後に`503 query_busy`を返し、画面に`Update failed`が出る。

原因は「サーバがqueryの放棄を知らないこと」である。frontendは既にlatest-winsを実装しており、`DataFetcher.fetchMetrics()`は新しいrequestを出す前に`abortMetrics()`で直前のrequestをabortする。しかしHTTP切断はServletスレッドから観測できず、abortされたrequestはサーバ側で最後まで実行を続け、query同時実行数を制限するfair semaphoreのpermitを保持し続ける。

その結果、既定の2 permitが「もう誰も結果を待っていないquery」で占有され、最新のqueryだけが5秒待って503を受け取る。ログの`active=2`はこの状態を指す。

本PRDは、frontendが既に表明しているlatest-winsの意図をサーバ側で強制する。すなわち、同じ画面操作系列に属する古いqueryをサーバが能動的に取り消し、permitを最新のqueryへ明け渡す。HTTP切断の検出には依存しない。

## 0. 決定一覧（グリル確定値）

| ID | 決定 |
|---|---|
| D1 | 直す範囲は「古いqueryをサーバ側で止める」。加えてworkspace切替時とアプリ終了時にも止める |
| D2 | supersedeの単位は**query channel**（1つのブラウザタブ）。別channelは相互にcancelせず、プロセス全体の同時実行枠だけを共有する |
| D3 | channel識別子と連番はHTTPヘッダで送る。**必須**とし、欠落・不正形式は`400 invalid_request` |
| D4 | 実行中queryの停止はループ要所のcancel checkpointを主とし、`Statement.cancel`を併用する |
| D5 | supersedeされたqueryの応答は`409` + `{"code":"superseded"}` |
| D6 | workspace切替は`SWITCHED`のときだけ旧epochのqueryを止める。`NO_OP` / `UNKNOWN`では止めない。cancel要求を出すだけで終了を待たない |
| D7 | 終了時は`LoadingThread`停止より前に全queryをcancelする。追加の待ち合わせは入れない |
| D8 | `getMetrics()`の取得順序を「検証 → 枠 → Workspace lease」へ変更する |
| D9 | cancelは専用の非チェック例外`QueryCancelledException`で伝え、既存3箇所の`catch (Exception)`の先頭で再throwする |
| D10 | channelごとの最新連番は件数上限付きLRU（固定値64）で保持する。設定項目は増やさない |
| D11 | 検証は状態機械の単体テスト、サービス層の並行テスト、Playwright回帰の3層 |
| D12 | `CONTEXT.md`へ`query supersede`と`query channel`を追加する |
| D13 | 成果物はPRD 049、ADR 0023、設計文書更新、`CONTEXT.md`用語2件 |
| D14 | （レビュー由来）`cancelWorkspace`はcancel済みepochのwatermarkを記録し、ticketはepoch束縛直後に`epoch <= watermark`なら自己cancelする |
| D15 | （レビュー由来）`cancelAll()`はterminal。以後の新規`run()`は実行せず即`QueryCancelledException` |
| D16 | （レビュー由来）channel識別子は長さ1〜128文字。超過は`400 invalid_request` |

## 1. 現状の事実（コード確認済み）

実装判断の前提となる事実を、確認済みの根拠とともに固定する。

| 事実 | 根拠 |
|---|---|
| semaphoreを通るのは`/api/metrics.json`のみ。`runs.json`と`prioritize`は枠の外にいる | `MetricsService.java:62,82,107` |
| `getMetrics()`はWorkspace leaseを先に取り、その後semaphoreを最大5秒待つ。枠待ちの間も旧workspaceのgzip資源をpinし続ける | `MetricsService.java:102-116` |
| queryが使う`MetricsRepository` / `RunScanner` / `LodPageCache`に`close()`はない。Workspace leaseが実際に守る寿命は`GzipInputSessions`だけである | 該当3クラスと`WorkspaceManager.java:331` |
| cancelの漏れ口は3箇所の`catch (Exception)`。`openQueryContext`はnullへ、`readSeriesInputs`はerror入力へ、`buildResult`は`availability=PENDING`へ丸める | `MetricsRepository.java:154,169,244` |
| `buildResult`の丸め先はHTTPエラーではない。cancel例外をここで捕まえると`200 OK`で「pendingの系列」として返り、frontendが成功として扱う | `MetricsRepository.java:244-252` |
| 長時間ブロックする単一DB文はない。`raw()`の最大50万行はJava側の`while (result.next())`ループ、`lowerBound`は1行取得 × 約log2(N)回、LOD page読みは1024行上限 | `MetricsRangeProjector.java:68`、`MetricsRepository.java:278`、`LodPageCache.java:135` |
| sqlite-jdbcはxerial 3.53.1.0で、`Statement.cancel()`は`sqlite3_interrupt`に対応する | `apps/metrics-viewer/pom.xml:79` |
| queryは対象Runごとのlifecycle READ lockを全series完了まで保持し、`prepare()`のWRITEと競合する。長いqueryは`LoadingThread`を`ingestGate`保持のまま待たせる | `MetricsRepository.java:107-129`、`MetricsCacheDatabase.java:123,188` |
| 静的ファイルは同じjarから配信されるため、frontendとサーバのバージョンずれは起きない。互換分岐は不要である | `src/main/resources/static/` |
| `queryRevision`はユーザー操作でしか増えず、Auto Reload経路は増やさずに`requestVisibleData()`を呼ぶ。同一値のPOSTが同時に飛びうるため連番へ流用できない | `metrics-viewer.js:1590,1664` |
| shutdown中の`acquireLease()`は`IllegalStateException`を投げるが`getMetrics()`が捕まえないため500になる | `WorkspaceManager.java:100` |

## 2. 契約

### 2.1 query channelと連番

**query channel**は1つのブラウザタブに対応するqueryの発行系列である。frontendはページロード時に`crypto.randomUUID()`でchannel識別子を1つ生成し、そのページが生存する間は変更しない。

**連番**はchannelごとに単調増加する整数で、POSTを発行するたびに増やす。既存の`queryRevision`は流用しない（1.の表のとおり、同一値のPOSTが同時に飛びうるため）。

連番の意味は「このchannelの中での新しさ」だけであり、channel間で比較しない。

### 2.2 HTTP契約

`POST /api/metrics.json`に次の2ヘッダを追加する。いずれも必須とする。

| ヘッダ | 値 |
|---|---|
| `X-Query-Channel` | channel識別子。長さ1〜128文字の文字列 |
| `X-Query-Sequence` | 連番。`0`以上のJavaScript safe integer |

欠落、空、長さ超過、数値として解釈できない、範囲外のいずれも既存の検証と同じ`400` `{"code":"invalid_request"}`で返す。channel識別子に長さ上限を置くのは、最新連番のLRUが件数64で有界でも1エントリの文字列長が無制限になることを防ぐためである。

任意ヘッダにはしない。任意にすると「未指定＝supersede対象外」という第2の実行経路が恒久化し、AGENTS.mdの「旧契約の実装を残さない」方針に反するためである。

`409`を応答へ追加する。

| status | body | 契機 |
|---|---|---|
| 409 | `{"code":"superseded","message":...}` | 同じchannelのより新しいqueryに追い出された、workspace切替またはshutdownで取り消された、より新しい連番が既に処理された後に遅れて到着した |

`503 query_busy`の契約は維持する。本PRD適用後、503が出るのは真に別channelが同時実行枠を使い切っている場合だけになる。

### 2.3 supersede規則

新設する`MetricsQueryCoordinator`が同時実行枠とsupersedeを一体で所有する。既存のsemaphoreはこれに置換・内包し、別の制限を重ねない。

- channelごとに「最新連番」と「実行中ticket」を持つ。
- 到着した連番が最新連番より大きい場合、そのchannelの実行中ticketと枠待ちticketをcancelし、自分を最新とする。
- 到着した連番が最新連番以下の場合、実行せず即座に`superseded`とする。
- 実行中ticketの消去はidentity一致時だけ行う。遅れて終了した旧requestが新ticketを消してはならない。
- 異なるchannelは相互にcancelしない。競合するのは同時実行枠だけである。
- 最新連番はqueryの完了時に消さない。保持はchannel件数の上限付きLRU（固定値64）とする。上限を超えて落とされたchannelから遅着したrequestは新規channel扱いで実行されるが、被害は「もう誰も見ていないrequestが1回だけ走る」ことに限られる。
- ticketは自身が取得したworkspace epochを保持する。epochはWorkspace leaseを取得した直後に束縛する。coordinatorはcancel済みepochのwatermark（`cancelWorkspace`が受けたepochの最大値）を保持し、ticketは束縛直後に`epoch <= watermark`なら自己cancelする。lease取得とepoch束縛の間に`cancelWorkspace`の走査が通り抜けるregistration raceを、`Statement`登録の再確認と対称の機構で閉じるためである（D14）。
- 実行中のcancelも、遅着による即時棄却（実行せず`superseded`）も、どちらも`run()`から`QueryCancelledException`で表面化させる。呼び出し側の`MetricsService`は409 `superseded`への変換を1経路で済ませる。

`MetricsQueryCoordinator`のcritical section内では、`Statement.cancel`、旧requestの完了待ち、Workspace/DB lockの取得、query本体の実行のいずれも行わない。

外部インタフェースは次の3つに閉じる。`Semaphore`、lock、`Statement`はインタフェースへ露出させない。

```java
<T> T run(QueryChannel channel, long sequence, QueryWork<T> work);
void cancelWorkspace(long epoch);
void cancelAll();
```

### 2.4 cancelの伝播

cancelは専用の非チェック例外`QueryCancelledException extends RuntimeException`で伝える。

cancel checkpointを次の位置へ置く。

- `MetricsRepository.query()`のRunループ（Runごとのcontextを開く前）
- `readSeriesInputs()`のseriesループ
- `lowerBound()`の二分探索の各反復
- `buildResult()`の射影開始前
- `MetricsRangeProjector.raw()`の行取り出しループ
- `MetricsRangeProjector.loadBuckets()`のbucketループ

checkpointを主機構とする根拠は1.の表のとおりで、長時間ブロックする単一DB文が存在しないためである。`Statement.cancel`は想定外に長い文への保険として併用する。実行中の`Statement`はticketへ登録し、登録直後にcancel状態を再確認して登録raceを閉じる。

既存3箇所の`catch (Exception)`は、先頭で`QueryCancelledException`だけを再throwする。「1つのRunやseriesの失敗で応答全体を失わせない」という既存の意図は変更しない。捕捉する例外の型は狭めない。狭めると`Missing L0 ordinal`のような既存の丸め対象が漏れ始め、本PRDの目的外の挙動変化が出るためである。

解放は内側から順に、`Statement` / `ResultSet`、SQLite transaction、Run connectionとlifecycle READ lock、Workspace lease、同時実行枠の順で行う。

### 2.5 workspace切替

`WorkspaceManager.switchWorkspace()`が`SwitchResult.SWITCHED`を確定したときだけ`cancelWorkspace(previous.epoch())`を呼ぶ。

- `NO_OP`（同じworkspaceの再選択）と`UNKNOWN`（存在しない名前、404）では何もcancelしない。誤入力や再選択で表示中のグラフを壊さないためである。
- cancel要求を出すだけで、queryの終了を待たない。`switchWorkspace()`は`ingestGate`を保持したHTTPスレッドで動くため、ここで待つとingestとshutdownを巻き込んで止める。
- cancel対象はticketが保持するepochで判別する。交換直後に開始した新epochのqueryはepochが異なるため巻き込まない。逆に、旧epochのleaseを取得済みでepoch束縛前のticketは走査をすり抜けうるが、2.3のwatermark再確認（D14）が束縛直後に自己cancelさせるため取り逃さない。この2つを合わせて、cancel要求と`current`の交換の順序に依存しない。

### 2.6 終了

`MetricsService`の`@PreDestroy`の順序を次へ変更する。

1. `coordinator.cancelAll()`
2. `loadingThread.terminateAndWait(30_000)`
3. `workspaceManager.shutdown()`

`cancelAll()`はterminalとする。以後の新規`run()`は実行せず即`QueryCancelledException`とする（D15）。1と3の間に到着したrequestが新ticketを登録してlifecycle READ lockを取り直し、2のjoinを再びブロックする穴を閉じるためである。

cancelはcheckpointで速やかに効くため、lifecycle READ lockがすぐ解放され、`LoadingThread`が`prepare()`のWRITEを取れるようになる。結果として既存の30秒joinが本来の目的どおり働く。新しい待ち時間定数は導入しない。

あわせて、shutdown中の`acquireLease()`が投げる`IllegalStateException`を`getMetrics()`で捕まえ、500ではなく`503 query_busy`相当として返す。

## 3. 実装範囲

### 3.1 新規

- `service/MetricsQueryCoordinator.java` — 2.3の契約を実装する。
- `service/QueryCancelledException.java` — 2.4の例外。`MetricsQueryCoordinator`の入れ子型でもよい。

### 3.2 変更

| ファイル | 変更内容 |
|---|---|
| `service/MetricsService.java` | `Semaphore`を撤去し`MetricsQueryCoordinator`へ委譲。順序を「検証 → `coordinator.run()` → その内側でWorkspace lease」へ（D8）。`QueryCancelledException`を409 `superseded`へ変換。`@PreDestroy`を2.6の順序へ。`IllegalStateException`を503へ丸める |
| `service/MetricsRepository.java` | 2.4のcheckpoint追加、3箇所の`catch`で`QueryCancelledException`を再throw、実行中`Statement`の登録・解除 |
| `service/MetricsRangeProjector.java` | `raw()`の行ループと`loadBuckets()`のbucketループへcheckpoint追加 |
| `service/WorkspaceManager.java` | `SWITCHED`確定時に`cancelWorkspace(previous.epoch())`を呼ぶ（2.5） |
| `view/MetricsViewerController.java` | 2ヘッダを受理し、`MetricsService`へ渡す。409応答の返却 |
| `resources/static/metrics-viewer.js` | ページロード時にchannel識別子を1つ生成。POST発行ごとに増える専用カウンタを連番に使う。`fetchMetrics()`で2ヘッダを送る。409 `superseded`を`_setUpdateFailure`と`_handleQueryError`の両方で対象外にする（`AbortError`と同じ扱い。除外しないとAuto Reload経路でconsole.errorが出続ける）。現行`fetchMetrics()`はstatusを持たない汎用`Error`をthrowするため、409を判別できるstatus保持エラーへ変更する。frontend側の`abortMetrics()`は帯域節約のため維持する |

### 3.3 文書

| ファイル | 変更内容 |
|---|---|
| `docs/design/210_metrics_viewer.jp.md` | §9.3（`POST /api/metrics.json`）のrequest契約へヘッダ2件、同節のエラー表へ409 `superseded`を追加し、503行の契機文を「query semaphore」から`MetricsQueryCoordinator`の記述へ差し替え。§10.1 lifetimeへ終了順序、§10.2 並行制御表の「query同時実行数」行を`MetricsQueryCoordinator`の記述へ差し替え |
| `CONTEXT.md` | §Metrics基盤へ`query supersede`と`query channel`を追加（D12） |
| `docs/adr/0023-server-enforced-query-supersede-per-channel.md` | 新設 |

## 4. 受け入れ基準

### 4.1 状態機械（新規 `MetricsQueryCoordinatorTest`）

- 同一channelで連番1が実行中に2が到着すると、1がcancelされ2が実行される。
- 1が枠待ち中でも、2の到着で1が即座に終了する。
- 異なるchannelは同時実行上限まで並行し、相互にcancelしない。
- 遅着した小さい連番は実行されず`superseded`になる。
- 実行中ticketの消去がidentity一致時だけ起き、遅れて終了した旧requestが新ticketを消さない。
- LRU上限を超えたchannelが落ちても、生存channelの新旧判定が壊れない。
- lease取得とepoch束縛の間に`cancelWorkspace`が走っても、束縛直後のwatermark再確認で自己cancelされる（D14）。

### 4.2 サービス層（既存 `MetricsQueryConcurrencyTest` を拡張）

- 古いqueryが409 `superseded`を返し、枠が解放され、新しいqueryが完走する。
- 真に別channelが枠を使い切った場合だけ`503 query_busy`になる。
- cancel後にRun lifecycle lock、Workspace lease、同時実行枠のいずれもリークしない。
- `SWITCHED`で旧epochのqueryが止まり、`NO_OP`と`UNKNOWN`では止まらない。
- 進行中のqueryとingestがある状態で終了処理が期限内に完了する。
- `cancelAll()`後の新規queryは実行されず409 `superseded`になる（D15）。
- ヘッダ欠落・不正形式・channel識別子の長さ超過が400 `invalid_request`になる。

### 4.3 Playwright回帰

- Run、tag、workspaceを高速に切り替えても`Update failed`が表示されない。

### 4.4 実行

```bash
mvn -f apps/metrics-viewer/pom.xml test
```

## 5. スコープ外

次は本PRDで扱わない。いずれも今回の症状の原因ではなく、独立に判断できる。

- `runs.json`を同時実行枠へ編入すること、およびmetadata用の予約枠。`getRuns()`はRunごとにread connectionを開いて全tagを読む重い処理を無制限に走らせているが、これは枠の外にいるため今回の503には寄与していない。
- `RunScanner.listRunId()`のキャッシュ化などquery自体の高速化。permit保持時間を縮める効果はあるが、計測が先である。
- 排他構造の整理。`GzipInputSessions`の防御的同期の除去、query用read-only viewとingest leaseの分離、`prepare()`のREAD先行検査、Run単位のread transaction解放、parallel ingest。
- 対話操作と背景取得（Auto Reload）の優先度付け。両者は同じchannelでlatest-winsに畳まれる。これは現在のfrontend挙動と同じであり、悪化しない。
- `MetricsCacheDatabase.cacheFilesExist`の未使用コード削除。

## 6. Further Notes

- `query channel`という語は多義になりやすい。`CONTEXT.md`の定義文で「1つのブラウザタブに対応する」ことを明示し、`_Avoid_`に`HTTPセッション`と`接続`を置く。
- ヘッダを採用したのは、将来`runs.json`（GET）へsupersedeを広げるときに方式を変えずに済むためである。requestボディへのフィールド追加ではGETに使えない。
- frontend側の`abortMetrics()`は残す。サーバ側supersedeが入っても、応答本体の転送を止められる利点は残る。

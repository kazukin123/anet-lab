# 042: MetricsViewer 構造リファクタリング(挙動不変) — 取り込み/照会経路の可読性改善

- 発端: 2026-08-01 実施の MetricsViewer アーキテクチャレビュー(取り込み・照会/API・フロント+テストの3系統監査、代表指摘は裏取り済み)。
- スコープ: レビューで「強く推奨」となった Java 側構造リファクタ 6 件(T1〜T6)。
  - スコープ外: 死にコード一掃・PalettePlaywrightTest 解体(別指示)、バグ疑い 9 件(挙動が変わるため別管理)、付録 A の 3 件(指示があるまで着手しない)。
- 実装: Codex。本書は self-contained(レビュー本体を参照しなくても実装できる)。
- 行番号は 2026-08-01 時点の作業ツリー基準。実装時は内容で特定すること。

## 0. 共通原則(全タスク)

1. **挙動不変**。HTTP 応答の JSON 形式・DB スキーマと格納値・状態文字列・キャッシュ破棄判定の真偽・ログ出力条件は一切変えない。例外は T3 の診断 reason 文言と破損した raw source metadata の安全側再構築(タスク内に明記)、および T2 で外部改変・破損により未知の DB state が入ったキャッシュを再構築前に直接読んだ場合の安全側 fallback とする。
2. **ADR0015(docs/adr/0015)の確定事項に触れない**: SQLite 短命接続(読み=リクエスト毎/書き=ブロック毎)、migration を書かない(不整合=全破棄再構築)、L0 PK=(tag_id, ordinal)、アプリ層 LRU は確定バケット限定、TagStats は LOD 非依存、step 逆行は tag 単位隔離、**LOD factor=16 の値**。
3. **既知バグに触れても現行挙動を維持する**。レビューでバグ疑いとして別管理中(例: ingestBlock で SQLException が "source_read_error" に誤分類される、issuedWarnings がプルーンされない等)。リファクタ中に気づいても本書では直さない。`// TODO` コメントの付記は可。
4. **テスト**: 各タスク完了ごとに `viewers/metrics-viewer` で `mvn test` 全緑。既存テストの修正は「内部構造変更への追従」のみ可、アサートの弱体化・削除は禁止。PalettePlaywrightTest は Edge 前提(assumeTrue)のため環境によっては skip される — その場合は残り全緑+コンパイル成功で可。
5. **コミット**: 1 タスク=1 コミット以上(タスク内分割は可、タスクを跨ぐコミットは不可)。実施順は T1→T2→T3→T4→T5→T6(T5 は T2 の enum に依存。T6 が最大工数のため最後)。
6. **配置**: 新規トップレベルクラスは最小限にし、既存ファイルへの同居(nested class・同パッケージ)を優先する。
7. 対象パッケージ: `io.github.kazukin123.anetlab.metricsviewer`(以下、クラス名のみで記載)。

---

## T1. LOD 幾何定数の一元化(工数: 小)

### 現状
「LODバケット=序数区間幅 factor16 の冪」というドメイン定数が裸のリテラルで散在している。

- LodIngestWriter.java:23-24(`% 16L`, `/= 16L`)、:132(`pending.size() < 16`)、:143-144(`!= 16`、"LOD parent requires exactly 16 children")、:182-186(`widthForLevel` 実装 #1)
- LodPageCache.java:144-148(`widthForLevel` 実装 #2)
- MetricsRangeProjector.java:238-246(selectLevel 内に width×16 の同等ループ=実装 #3。さらに `bucketCount * 3L > pointBudget` の `3` は、同クラス約 200 行先の「LOD バケットから min/max/last の 3 候補を排出する」ロジックと暗黙同期)
- MetricsRepository.java:222(`Math.min(50, plan.cap)` の `50`=1 系列あたり最低保証点数、無名)
- LodBucket.java: 13 連続の位置引数コンストラクタ。LodIngestWriter.java:107-121 の `fromLodRow` が `result.getLong(3), getLong(4), …` を位置渡ししており、列とフィールドの対応ミスをコンパイラが検出できない

### やること
1. 名前付き定数を 1 箇所(LodBucket を推奨)に定義し、全参照を置換:
   - `LOD_FACTOR = 16`
   - `POINTS_PER_LOD_BUCKET = 3`(min/max/last の 3 点。selectLevel の `×3` と候補排出側の両方から参照し、暗黙同期を明示化)
   - `MIN_POINTS_PER_SERIES = 50`
2. `widthForLevel` を 1 実装に統合し、3 箇所から利用する。
3. `fromLodRow` の列→フィールド対応を 1 箇所に閉じる(LodBucket に名前付き static ファクトリを置く、または列名ベース取得にする。位置引数の連続をなくすことが目的)。

### やらないこと
- 16 の値変更(ADR0015 確定)。SQL・スキーマ・出力の変更。

### 完了条件
- `16`・`3`・`50` の裸リテラルと widthForLevel 重複が対象箇所から消えている。`mvn test` 緑。

---

## T2. IngestState / SeriesAvailability の enum 化(工数: 小〜中)

### 現状
`"pending" / "converting" / "ready" / "error"` の状態機械が stringly-typed で 3 クラスに分散:

- 語彙定義: MetricsCacheDatabase.java:34-35(VALID_STATES)
- 遷移: MetricsIngestor.java:95, 169-171, 195(例: `state = cursor.eof && (!gzip || trailingBytes == 0) ? "ready" : "converting"`)
- 解釈: MetricsRepository.java:339-341(`isConverting` は `"pending".equals(state) || "converting".equals(state)` — **"pending" でも true を返し名前が実挙動とズレている**)、:407-409(calculatePercentage)

さらに系列応答の availability(`"ok" / "pending" / "not_found" / "empty"`、MetricsRepository.java:183, 195, 204-206)の "pending" は ingest state の "pending" と**別意味**で、読み手が 2 つの語彙空間を暗算で分離させられている。

### やること
1. `MetricsCacheDatabase` 内へ public nested enum
   `IngestState { PENDING, CONVERTING, READY, ERROR }` を導入する。
   - 各値はDBとHTTP JSONで共通の安定した小文字表現を保持し、`externalName()`で返す。`name()`、`toString()`、呼び出し側の小文字変換へ永続・wire表現を依存させない。
   - `fromDb(String)` はnull・未知値を拒否する。DB妥当性検査では非例外の`isValidDbValue(String)`を使用し、不正値を`PENDING`へ丸めず従来どおり全破棄再構築する。
   - 正規の書き込み経路では生成されない未知stateを再構築前に直接読んだ場合、`runs.json`は`pending`・0%・空タグ・generationなしへfallbackしてwarnを記録し、`metrics.json`は`pending`・`run/query_error`・projectionなしへfallbackする。未知文字列のechoは互換対象外とし、次回取り込み時に`state_invalid`として再構築する。
   - 実際に必要な判定`isStillIngesting()`を持たせる(PENDINGまたはCONVERTING)。未使用の`isTerminal()`は追加しない。
2. Java内部interfaceのstate型を`String`から`IngestState`へ差し替える。
   - 対象は`CachePreparation.state`、`CacheMetadata.state`、`IngestOutcome.state`。
   - DBへの書込時とruns.jsonの`IngestInfo`構築時だけ`externalName()`へ変換する。`IngestInfo.state`は`String`のままとし、enumをJSONへ直接公開しない。
3. `MetricsRepository.java`内へpackage-private top-level enum
   `SeriesAvailability { OK, PENDING, NOT_FOUND, EMPTY }`を導入する。
   - `SeriesPlan.availability`をenumへ差し替え、全代入・比較とprojection失敗時fallbackをenumへ統一する。
   - `MetricsSeriesResult.availability`は`String`のままとし、結果構築時だけ`externalName()`へ変換する。
4. `isConverting`を廃止し、呼び出し側を`IngestState.isStillIngesting()`へ置き換える。enum化は語彙と判定の一元化に限定し、状態遷移条件自体は変更しない。

### やらないこと
- 状態の追加・削除・遷移条件の変更。JSON/DB 表現の変更。
- Tag statusの`"error"`、error code/message、SQL制約、外部契約を確認するテスト文字列のenum化。

### 完了条件
- productionコードのIngestState / SeriesAvailabilityの代入・比較から対応する小文字リテラルが消え、文字列はDB/JSON変換点に限定されている。
- Java内部型を直接扱うテストとmock生成はenumへ追従してよい。DB値とHTTP JSONを直接確認する既存の小文字文字列アサートは無修正で緑となり、外部表現不変を証明する。
- 不正なDB stateが従来どおり無効DB判定・generation変更を伴う再構築へ進む統合試験がある。`mvn test`緑。

---

## T3. ソース同一性検査の「双子メソッド」統合(工数: 小〜中)

### 現状
MetricsCacheDatabase.java に「boolean 判定」と「理由診断」が同じ検査列を**順序違いで**二重実装した組が 2 つある(計約 150 行):

- 組 1: `matchesSource`(:227-257)と `describeSourceMismatch`(:259-304)
  - 同じ検査(サイズ・mtime・source_kind・先頭 64KiB ハッシュ・切詰め判定)を別順序で実装。IOException の扱いも非対称(matches=throw / describe=`"source_fingerprint_diagnosis_failed"` に握る)。診断側で同じ 64KiB ハッシュを再計算する無駄もある。
  - 帰結: 検査を 1 つ足すと双方の同期修正が必要で、**診断された reason が実際に失敗した検査と食い違い得る**。
- 組 2: `isValidDatabase`(:406-430)と `diagnoseInvalidDatabase`(:432-461)— REQUIRED_TABLES/COLUMNS 走査のコピー。

### やること
1. 各組を「valid なら null、無効なら理由文字列を返す」**単一の検査列**に統合する(例: `String checkSourceMismatch(...)` / `String checkDatabaseInvalid(...)`)。
2. boolean 版が必要な呼び出し元there は `reason == null` の薄い wrapper で維持(公開シグネチャの温存は任意。呼び出し元を直接 null 判定に書き換えてもよい)。
3. IOException 方針は組ごとに現行の外部挙動(破棄再構築に至るか否か)を維持する形で 1 箇所に明記する。

### 挙動不変の例外
- **判定の真偽(=キャッシュを破棄再構築するか)は、次の破損 metadata を除き完全不変**。
- raw `jsonl`・非 `ERROR` state で `source_mtime` が欠落または非数値の場合、旧実装はキャッシュを維持したが、新実装は `source_metadata_invalid` として全破棄再構築する。正規の書き込み経路では `source_mtime` が常に保存されるため、外部改変・破損データだけを安全側へ倒す変更として許容する。
- 診断ログに出る reason 文字列は「実際に最初に失敗した検査」を指すようになり、現行と変わるケースがある(これは修正目的そのもの)。reason 文字列をアサートしている既存テストがあれば追従修正可。

### 完了条件
- 検査列が組ごとに 1 実装。`mvn test` 緑(reason 追従修正を除き無修正)。

---

## T4. source_meta 読み書きの一機構化(工数: 中)

### 現状
`source_meta` テーブル(k/v)のキー文字列・型変換・既定値の知識が 3 クラス×6 メソッドに分散:

- 書込: MetricsCacheDatabase.java:384-391(initialize → `writeMeta`。:539-544 は**手動エスケープ+文字列連結 SQL**)と MetricsIngestor.java:507-522(updateSourceMeta)+ :548-557(upsertMeta。こちらは PreparedStatement)— 同一テーブルへ 2 作法
- 読取: MetricsCacheDatabase.java:499-506(readSourceMeta)と MetricsRepository.java:397(readMeta。同型の複製)
- 型変換: `parseLong(String, long fallback)` が MetricsCacheDatabase.java:546-553 と MetricsRepository.java:414 に 2 枚。**fallback 値まで別**(Database 側 -1 / Repository 側 0。実装時に現行値を確認)
- 型付き表現: `CacheMetadata` record は Database 側で定義されるが、組み立ては Repository:63-70 のみ。一方 query 経路(Repository:140, 170-172, 332-337)は生 Map + `getOrDefault("state", "pending")` でキー文字列を再散布 — 同じメタデータの読解が「型付き」と「生 Map」の二重表現

### やること
1. source_meta の read / write / 型変換 / キー定数を 1 箇所に集約する(名称例: `SourceMeta`。配置は MetricsCacheDatabase の nested class または infra パッケージ内。新規ファイルより同居優先)。
   - write は PreparedStatement に統一(文字列連結 SQL を廃止)。
   - read は `Connection → CacheMetadata` を 1 実装に。parseLong はここへ 1 枚だけ。**呼び出し元ごとの現行 fallback 値は維持**(既定値を引数で受けるか、呼び出し元別アクセサで吸収)。
2. Repository の query 経路(RunQueryContext 含む)は生 Map でなく CacheMetadata を保持・参照する。
3. T2 完了後なので、state は CacheMetadata 上で IngestState として扱ってよい(DB 表現は不変)。

### やらないこと
- source_meta のキー名・値形式・スキーマの変更(ADR0015: Phase 2 で C++ SqliteBackend がマスタ地位を DB へ移す際の移行判定の枢軸が `user_version` と `source_meta.source_kind`。**本タスクはその知識を 1 module に局所化するのが目的**であり、表現を変えてはならない)。

### 完了条件
- source_meta のキー文字列リテラルが SourceMeta(集約先)以外に出現しない。parseLong・readMeta の複製が消えている。`mvn test` 緑。

---

## T5. MetricsQueryPlanner の抽出(工数: 中)

### 現状
MetricsRepository.java が名前(Repository=データアクセス)に反して query 計画を内包:

- :161-253 `inspectSeries` / `allocatePointBudgets`(private)— water-filling 式の点数予算公平配分。:222 `plan.pointBudget = Math.min(50, plan.cap)`、:231 で `MetricsApiException(HttpStatus.UNPROCESSABLE_ENTITY, …)` を throw(HTTP ステータス決定が Repository 内)
- :453-471 `SeriesPlan` — mutable フィールド 10 個。availability の値によって有効なフィールド集合が変わる暗黙プロトコル(tagId/ordinalFrom/ordinalTo/cap は "ok" のときだけ有効、context は null あり得る)がどこにも書かれていない
- 帰結: 予算配分の境界ケース検証が「fixture ingest+MockMvc の全スタック統合テスト」経由でしかできない

### やること
1. 計画部(inspectSeries / allocatePointBudgets とその下請け)を **package-private の `MetricsQueryPlanner`** として抽出する(同パッケージ内新規クラス可。純関数構成: 入力=系列要求+読み出し済み tag 情報+settings、出力=SeriesPlan 群。SQL・Connection を持たない)。
   - Repository には SQL 読み(readTags / readMeta / lowerBound 等)だけ残す。
   - 422 throw は planner 内のままでよい(呼び出しチェーン経由で HTTP 応答は現行と同一)。
2. SeriesPlan の availability を T2 の `SeriesAvailability` に差し替え、**「状態別に有効なフィールド」をクラスコメントで宣言**する(record 分割までは要求しない)。
3. **特性テストを先行または同時に追加**: `MetricsQueryPlannerTest`(単体・純関数)で water-filling の境界を固定 — 均等割の端数、cap 混在(小 cap 系列がある場合の再配分)、`MIN_POINTS_PER_SERIES` 未満の cap、全系列 not_found/empty、予算不足で 422 になる境界。既存統合テストの期待値と一致すること。

### やらないこと
- 配分アルゴリズム・点数・応答 JSON の変更。ADR0015 の短命接続・LRU 構成の変更。

### 完了条件
- MetricsRepository から予算配分・422 判定のロジックが消え、planner 単体テストが追加されて緑。既存統合テスト無修正で緑。

---

## T6. ingestBlock に SourceReader seam(工数: 中〜大、最後に実施)

### 現状
MetricsIngestor.java:84-227 の `ingestBlock`(143 行)に gzip/raw の分岐が 12 箇所(:86, 92, 95, 108, 142, 164, 169, 195, 202, 207, 211, 222 付近)あり、以下の全関心と交差:

- 入力選択(`readLimitOffset = gzip ? Long.MAX_VALUE : source.size()`)
- オフセット会計(raw は `cursor.bytesRead++`=非圧縮 byte、gzip は `cursor.bytesRead = consumedBytes.getAsLong()`=BufferedInputStream が 64KiB 先読みした**圧縮消費 byte**。単位が違うのにフィールドは 1 つ。PRD041 §3.5 が規定済みだがコードに単位注記なし)
- 論理 EOF 判定・セッション寿命(GzipInputSessions の hasSession→prepare→acquire 3 段プロトコルが呼び手へ露出)・エラー分類(catch 節にも `if (gzip)`)

周辺の同時対象:
- `writeScalar`(:346-355)が 9 引数リレー(connection / PreparedStatement×2 / lodInsert / tagStates / warnings×2 / cursor)
- `DatabaseWriteRuntimeException`(:437-438, 687-691)— `tagStates.computeIfAbsent(tag, t -> loadTagState(connection, t))` の lambda が SQLException を投げられないための例外トンネル
- `BlockCursor`(:584-595)— 上記の単位混乱の置き場

### やること
1. **特性テストの確認を先行**: MetricsIngestorIntegrationTest が gzip の途中再開(trailing bytes あり)・raw 追記再開・corrupt 行 quarantine を覆っているか確認し、不足があれば ingestBlock を触る前にテストを追加する。
2. `SourceReader` interface を導入(名前は例。配置は MetricsIngestor 同居 nested interface 可):
   - 契約: 「完全な 1 行を返す(未完行は返さない)」「消費ソースオフセットを報告する」「論理 EOF / corrupt を判定する」。
   - adapter 2 つ: `RawFileReader`(raw。オフセット=非圧縮 byte)と `GzipSessionReader`(gzip。GzipInputSessions の 3 段プロトコルと CountingInputStream を内包し、オフセット=圧縮消費 byte)。
   - オフセットの**単位の違いを名前で明示**する(例: `consumedSourceBytes()` の Javadoc に raw/gzip の意味を明記。BlockCursor 側もフィールド名か契約コメントで単位を明示)。
3. `ingestBlock` は transaction 境界・状態遷移("ready"/"converting" 判定)・エラー分類だけにする。エラー分類の**現行挙動は誤分類含め原則維持**(共通原則 3。SQLException→"source_read_error" 経路は既知バグ疑いのため現状維持+TODO 可)。ただし下記5の`loadTagState`例外契約は明示的な許容差分とする。
4. `BlockWriteSession`(名前は例)を導入し、1 ブロック分の書込文脈(statements / lodInsert / tagStates / warnings / cursor)を 1 オブジェクトに畳んで writeScalar の 9 引数リレーを解消する。
5. `DatabaseWriteRuntimeException` トンネルを撤去: `computeIfAbsent` をやめ `Map.get` + null 時ロード(checked SQLException をそのまま伝播)にする。
   - **例外契約の裁定**: `loadTagState`内の`SQLException`も同じblock内の他の`SQLException`と同様に外側の分類へ到達し、rawでは`source_read_error`を記録してRunを`error`にする。旧実装でunchecked wrapperが外側のcatchを偶然迂回し、`converting`のまま次cycleへ再試行していた挙動は互換対象外とする。
   - `SQLException`全般を`source_read_error`へ分類する問題自体は既知バグ#1のままとし、T6では`loadTagState`だけ旧挙動へ戻さない。将来の一括修正時に同じblock内の全`SQLException`をまとめて再裁定する。

### やらないこと
- 上記`loadTagState`の例外契約を除く、取り込み結果(DB 内容・state 遷移・quarantine 条件・warn 条件・committed_offset の値)の変更。単一 writer thread 前提の変更。

### 完了条件
- `if (gzip)` が ingestBlock 系から消え、モード差が 2 つの adapter に閉じている。writeScalar 系の引数リレーと例外トンネルが消えている。`mvn test` 緑(gzip/raw 両モードの統合テスト含む)。

---

## 付録 A: スコープ外(指示があるまで着手しない)

レビューで「検討価値あり」となった 3 件。本書では実施しない。

1. Projection の sealed interface 化 — `MetricsRangeProjector.Projection.body` と `MetricsSeriesResult.projection` の `Object` 型消去 2 箇所を `sealed interface Projection permits RawProjection, LodProjection` へ(JSON 不変)。
2. エラー応答の一機構化 — MetricsViewerController の同一 catch 節×2 を `@RestControllerAdvice` へ集約、body を既存 `ApiError` record 系へ一本化。
3. フロント描画状態の一方向化 — metrics-viewer.js の capturePlotState(Plotly DOM からの吸い上げ)廃止、renderBySelection(138 行)の 3 分割。

また、レビューのバグ疑い 9 件(SQLException 誤分類 / LoadingThread 無言死 / replacePriority 競合窓 / issuedWarnings 無プルーン / body.error オーバーレイの pointer-events / onresize 直代入 / ポーリング失敗の無通知 / LodPageCache の世代内 stale page / cache-memory-mb=0 の読み増幅)は挙動変更を伴うため、本書とは別に個別裁定する。

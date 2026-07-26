# MetricsViewer SQLite キャッシュ化(Phase 1: OOM 解消・LOD・viewport API)

> 設計分担: Claude=設計/PRD、実装=Codex、Run/commit=ユーザー。
> 本書は self-contained。実装時は行番号ではなく、近傍のシンボル名で再検索する。
> 用語(Run作業セット / Metricsマスタ / Metricsキャッシュ / 序数 / LODバケット)は
> リポジトリルート `CONTEXT.md` の「Metrics基盤」節が正本。
> 対応 ADR: [0015-metrics-cache-disposable-derivative.md](../adr/0015-metrics-cache-disposable-derivative.md)

## Context(背景・目的)

MetricsViewer(`viewers/metrics-viewer`、Spring Boot 3.5.7 / Java 17)には3課題がある。

1. **大規模Runでサーバー側OOM(最優先)** — 根因はストレージ形式ではなくViewerの保持構造。
   `MetricsSnapshot`が`Map<TagInfo, List<Point>>`で**全Run×全tag×全点をヒープ保持**し、
   `MetricsRepository.findTagTraceDiff`の間引き(Min-Max-Last)は**転送時のみ**。
   `metrics_cache.kryo`スナップショットも全点を直列化しており、起動時ロードで同じ山を登る。
2. **`metrics.jsonl`の容量** — 無圧縮テキストでscalar 1点は約94〜97 bytes。
3. **拡張性** — 事後集計・統計クエリの土台がJSONL走査しかない。

さらに現行実装には解像度の構造問題がある。初回転送は全履歴を30,000点へ間引くが、
以後のdeltaは新着だけを独立に間引いて**クライアントの追記専用バッファ**
(`numeric-buffers.js`の`Int32Buffer` / `Float32Buffer`)へ積む。このため、
系列の先頭は1/2700に潰れ末尾は原寸という不均一密度が蓄積し、
ズームしても**再取得経路がないので精度が戻らない**。

### 実測(2026-07、設計根拠)

| 項目 | 値 |
|---|---|
| 対象Run | `run_20260721-201834_cnx-vit128★`(7.31 GiB)ほか |
| 総行数 / scalar行 | 80,936,134 / 80,936,120(非scalarは**14行**のみ) |
| tag数 | 98(別Run実測114) |
| scalar 1行 | 平均93.8 bytes |
| stepの巻き戻り | **全tagでゼロ**(非減少) |
| 同一stepの重複 | **1 tagで240,109件**(`20_eps/11_train_reward_ema`。256並列envが同一`exp_step`で複数終端するため正当) |
| SQLite行サイズ実測 | L0 `(tag_id, ordinal, step, value)` WITHOUT ROWID **18.86 B/点** / rowid表+index 35.73 B/点 / extrema位置を持たない旧LOD lean **52.36 B/bucket** |
| LOD行数(factor 16) | L0比**6.67%**(L1=316k / L2=19.8k / L3=1.2k / L4=78 per 5.06M点tag) |
| 容量見込み | 旧LOD leanではL0+LODが約**1.69 GiB**(JSONL 7.31 GiBの23%)。実step metadata追加後は再計測し、受け入れ上限30%で判定 |
| gzip実測 | `-6`で9.10x / 169 MB/s(参考。C++側gzip出力は不採用が確定) |

### 解決の骨子

- 各Runフォルダ内に破棄可能な**SQLiteキャッシュ`metrics_cache.db`**を構築する。
  L0全点、factor 16の多段LOD、`TagStats`、非scalar原文、ソース状態を保持し、
  マスタ`metrics.jsonl`または不変な`metrics.jsonl.gz`から再生成可能にする。
- 全点のJavaヒープ保持とKryoを廃止し、読み出しを短命SQLite接続によるviewport単位queryへ置換する。
- `TagStats`は廃止しない。commit済みの有効なL0全点に対する、
  viewportやLODレベルに依存しない正確な統計として維持する。
- LODはvalueだけでなくmin / max / lastが実際に発生したL0のordinalとstepを保持する。
  既存Min-Max-Lastと同様、MinMax modeでも極値を実stepへ描画する。
- キャッシュ全再構築ごとに`generation`を変更し、通常追記では維持する。
  サーバーとクライアントはgenerationをstaleデータの拒否に使う。
- サーバー/クライアントのメモリをRunサイズ・Run数から切り離し、
  表示中の系列数と設定済み上限だけに比例させる。

## 1. 原則(アーキテクチャ不変条件)

1. **マスタは1種類**:
   Runフォルダ内の正は`metrics.jsonl`。
   手動gzip済みの`metrics.jsonl.gz`は同一マスタの不変なライフサイクル段階として透過的に読む。
   両方あれば`.jsonl`を優先する。
2. **キャッシュは破棄可能**:
   `metrics_cache.db`、`metrics_cache.db-wal`、`metrics_cache.db-shm`は
   マスタから再構築可能な従属導出物とする。schema不一致、破損、ソース同一性喪失ではmigrationせず全再構築する。
3. **commit済みprefixだけを公開**:
   L0、LOD、`TagStats`、tag状態、非scalar原文、ソースoffsetを同一transactionで更新する。
   APIはread transaction開始時点のcommit済みsnapshotだけを見る。
4. **短命DB接続**:
   読みは1 HTTP要求・1 Runごと、書きは1取り込みブロックごとに接続を開閉する。
   ただしgzip変換中は同一`GZIPInputStream`をブロック間で保持するため、
   activeなgzip変換中Runのフォルダ移動だけは非対応とする。
5. **全自動変換と明示的優先度**:
   Run作業セット内の全Runを順次変換する。APIで指定されたpriority集合と背景集合を
   3ブロック:1ブロックで処理し、変換途中でもcommit済み分を表示する。
6. **解像度はviewportの関数**:
   空きヒープ量に応じて保持点数を増やさない。描画点数予算内でLODを選び、
   ズームすればL0へ到達する。余剰メモリは完成済みLODのLRUにだけ使う。
7. **単一Viewerプロセス**:
   複数Viewerプロセスによる協調取り込みは対象外。偶発的な同時接続でも
   WALと`busy_timeout`によりDBを破損させない。

## 2. SQLiteキャッシュ

### 2.1 ヘッダと初期化

JDBCを直接使用し、SQLite接続に次を設定する。

```sql
PRAGMA application_id = 0x414E4554;  -- "ANET"
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;          -- 接続ごと
```

- `user_version`は1から始まる単調増加のcache schema versionとする。
- 新規DBではDDLと初期`source_meta`を完了するまで`user_version=0`のままにし、
  初期化がすべて成功したときだけ現行versionを設定する。
- `application_id`不一致、`user_version`不一致、必須table/column欠落、
  `PRAGMA quick_check`失敗、接続不能は無効DBとして扱う。
- 無効DB、旧schema、corrupt DBは接続を閉じてから
  `.db` / `-wal` / `-shm`を削除し、generationを新しくして先頭から再構築する。

### 2.2 schema

```sql
CREATE TABLE tags(
  id                  INTEGER PRIMARY KEY,
  key                 TEXT UNIQUE NOT NULL,
  type                TEXT NOT NULL CHECK(type = 'scalar'),
  status              TEXT NOT NULL CHECK(status IN ('ok', 'error')),
  error_code          TEXT,
  error_message       TEXT,
  error_source_offset INTEGER,
  error_previous_step INTEGER,
  error_step          INTEGER
);

-- L0。ordinalはtag内でcommitされた有効点の出現順(0, 1, 2, ...)。
-- stepは座標でありidentityではない。重複を許し、非減少でなければならない。
CREATE TABLE scalars(
  tag_id  INTEGER NOT NULL,
  ordinal INTEGER NOT NULL,
  step    INTEGER NOT NULL,
  value   REAL NOT NULL,
  PRIMARY KEY(tag_id, ordinal)
) WITHOUT ROWID;

-- 完成した16個の子だけから作るimmutableなLOD。level=1..N。
-- bucket = ordinal / 16^levelの整数除算。
CREATE TABLE scalars_lod(
  tag_id      INTEGER NOT NULL,
  level       INTEGER NOT NULL,
  bucket      INTEGER NOT NULL,
  cnt         INTEGER NOT NULL,
  step_first  INTEGER NOT NULL,
  step_last   INTEGER NOT NULL,
  min_ordinal INTEGER NOT NULL,
  min_step    INTEGER NOT NULL,
  vmin        REAL NOT NULL,
  max_ordinal INTEGER NOT NULL,
  max_step    INTEGER NOT NULL,
  vmax        REAL NOT NULL,
  vmean       REAL NOT NULL,
  vlast       REAL NOT NULL,
  PRIMARY KEY(tag_id, level, bucket)
) WITHOUT ROWID;

-- commit済みの有効なL0全点に対する範囲非依存の統計。
CREATE TABLE tag_stats(
  tag_id     INTEGER PRIMARY KEY,
  count      INTEGER NOT NULL,
  mean       REAL NOT NULL,
  m2         REAL NOT NULL,
  min_value  REAL NOT NULL,
  max_value  REAL NOT NULL,
  min_step   INTEGER NOT NULL,
  max_step   INTEGER NOT NULL,
  last_value REAL NOT NULL
) WITHOUT ROWID;

-- 有効な非scalar JSON objectの原文。
CREATE TABLE json_lines(
  ordinal   INTEGER PRIMARY KEY,
  type      TEXT NOT NULL,
  tag       TEXT,
  step      INTEGER,
  timestamp TEXT,
  json      TEXT NOT NULL
);

CREATE TABLE source_meta(
  k TEXT PRIMARY KEY,
  v TEXT NOT NULL
) WITHOUT ROWID;
```

`source_meta`には少なくとも次を保存する。

| key | 意味 |
|---|---|
| `generation` | DB全再構築ごとに採番するUUID |
| `source_kind` | `jsonl`または`jsonl.gz` |
| `source_size` / `source_mtime` | 指紋採取時のソース属性 |
| `source_head_sha256` | ソース先頭64 KiBのSHA-256 |
| `source_commit_tail_sha256` | commit offset直前64 KiBのSHA-256 |
| `committed_offset` | 最後にcommitしたsource位置。JSONLでは完全改行までのbyte offset、gzipでは圧縮stream消費bytes |
| `state` | `pending` / `converting` / `caught_up` / `complete` / `error` |
| `error_code` / `error_message` | Run-level error。正常時は削除する |

設計上の制約:

- `scalars`と`scalars_lod`は`WITHOUT ROWID`を必須とする。
- step補助indexは作らない。step範囲からordinal範囲への変換は
  PK point queryによる二分探索で行う。
- LOD factorは16固定。L1は16個のL0、L2以降は16個の直下level完成bucketから作る。
  次levelに完成bucketを1つも作れなくなるまで保存し、target点数から深さを決めない。
- `min_ordinal/min_step`と`max_ordinal/max_step`は、extremaを持つ元のL0点の
  absolute ordinalと実stepを全levelで保持する。`step_last/vlast`も元の最終L0点を保持する。
- 16子を合成するとき、min / maxを持つchildのvalueだけでなくL0 ordinalとstepも引き継ぐ。
  同値のminまたはmaxが複数ある場合は、既存Min-Max-Lastと同じく最小ordinalを採用する。
- 完成bucket内で`min_ordinal` / `max_ordinal`は当該bucketのordinal範囲内でなければならない。
  `step_first <= min_step,max_step,step_last`を満たすが、step重複を許す。
- 未完成LOD bucketは永続化しない。各level最大15子から復元できる。
- SQLiteの`REAL`はf64として計算・保存し、API転送時だけvalue列をf32にする。
- `tag_stats.m2`は内部集計用であり、公開`TagStats`には露出しない。

### 2.3 `TagStats`契約

Javaの`TagStats`型は残し、公開fieldを次に固定する。

```text
TagStats {
  minStep,
  maxStep,
  count,
  lastValue,
  minValue,
  maxValue,
  mean,
  variance,
  stdDev
}
```

- 対象はそのtagでcommit済みの有効なL0全点。viewport、query range、LOD levelに依存しない。
- Welfordで`count` / `mean` / `M2`を更新し、母分散`variance=M2/count`を採用する。
- 1点では`variance=0`、`stdDev=0`。
- `minStep` / `maxStep`は受理済みL0点のstep範囲、`lastValue`は最大ordinalのvalue。
- `updatedAt`と公開`M2`は持たない。
- 無効valueとしてskipした行と、tag隔離後にskipした行は含めない。

## 3. 取り込みパイプライン

取り込みは単一writer threadで行い、1ブロックの上限を1,000,000完成行とする。
巨大な`List<MetricsFileLine>`を作らず、1行ずつstreaming parseして同じtransactionへ反映する。

### 3.1 マスタ選択とソース同一性

1. `metrics.jsonl`があれば選ぶ。なければimmutableな`metrics.jsonl.gz`を選ぶ。
2. 両方ある場合は`.jsonl`を優先し、RunごとにViewerプロセス中1回だけWARNする。
3. source kindが変わった場合は全再構築する。gzip変換中にJSONLが出現した場合も同じ。
4. size / mtimeに加えて次のSHA-256を照合する。
   - 先頭64 KiB
   - 保存済みcommit offset直前64 KiB
5. 保存済みprefixのhashが変化した場合は、現在のsizeが以前より大きくても
   「追記」ではなく別ファイルへの差し替えとして全再構築する。
6. JSONLのsizeがcommit offset未満になった場合、同サイズ上書き、切り詰めも全再構築する。
7. JSONLの指紋が一致しsizeが増えた場合だけ差分取り込みする。
8. gzipは不変ファイルとして扱い、完了後のsize / mtime / hash変化は全再構築する。

JSONLは完全な改行までだけcommitする。末尾の未終端行は次回追記まで保留し、
`committed_offset`を進めない。

旧`metrics_cache.kryo`を発見した場合は削除する。
Python版ツールが所有する`metrics_cache.parquet`には触れない。

### 3.2 1ブロックのtransaction

1ブロックで次を同一transactionに含める。

1. 有効なscalar点を`scalars`へINSERTする。
2. 16子が揃ったLOD bucketを`scalars_lod`へINSERTする。
3. Welford更新した現在値を`tag_stats`へUPDATEする。
4. tag初出または隔離状態を`tags`へ反映する。
5. 有効な非scalar JSON objectの原文を`json_lines`へINSERTする。
6. `source_meta`のoffset、指紋、進捗、状態を更新する。

commitに成功したときだけ次block用のin-memory作業状態を採用する。
transactionをrollbackした場合は、そのblockで更新したordinal、Welford、
LOD accumulator、WARN候補をすべて破棄し、DBのcommit済み状態から復元する。

再開時は各tagのL0件数と`tag_stats`を読み、各levelについて末尾の最大15子だけを
DBから読んで未完成LOD accumulatorを復元する。完成LOD行はimmutableでありUPDATEしない。
LOD accumulatorは`cnt/step_first/step_last/min_ordinal/min_step/vmin/`
`max_ordinal/max_step/vmax/vmean/vlast`を一体として保持し、rollback時は位置情報も破棄する。

### 3.3 行の妥当性とエラー規則

完成済み行には次の規則を適用する。

#### Run-level fatal error

次はblock全体をrollbackし、別の短命transactionでRunを`error`にする。
同一ソース指紋の間は自動再試行しない。

- 不正JSON
- JSON objectではないtop-level値
- typeごとの必須field欠落
- 整数でない、不正な、またはJavaScript安全整数範囲
  `[-(2^53-1), 2^53-1]`外のstep

scalarは`type` / `tag` / `step` / `value`を必須とする。
非scalarは少なくとも文字列`type`を必須とし、未知のtypeでも有効なJSON objectなら原文保存する。

#### 行だけskipするscalar value

次のscalar valueはその行だけskipし、ソースoffsetは進める。
WARNは`(Run, tag, reason)`ごとにViewerプロセス中1回だけ出す。

- `null`
- JSON number以外
- 非有限値
- f32へ変換すると非有限になる値

skip行にはordinalを割り当てず、`TagStats`とstep順序の基準にも含めない。

#### tag隔離

受理済みの直前L0点に対して`step < previousStep`となった場合:

- Run全体を止めず、そのtagだけ`status=error`へ隔離する。
- `tags`へerror code、詳細、source offset、previous step、current stepを保存する。
- 同じblock内で隔離行より前に処理した正常点は他の変更とともにcommitする。
- 隔離を検出した行と、それ以後の同tag行はskipする。別tagとRun変換は継続する。
- 隔離前までのL0、LOD、`TagStats`は公開し続ける。
- 隔離解除はソース変更による全再構築時だけ行う。

### 3.4 Run単位のDB取り込み進捗

進捗はsource kind固有の機能ではなく、各Runについて
「選択したMetricsマスタをSQLiteへどこまでcommitしたか」を表す共通契約とする。

- `ingestedBytes`は最後に成功したDB transactionまでに消費したsource bytes。
  JSONLでは完全改行までのcommit offset、gzipでは圧縮streamの消費bytesを使う。
- `sourceBytes`は選択中のMetricsマスタの現在file size。
- 表示percentageはstateを優先して次のように決める。
  - `pending`: 0%
  - `converting`: `floor(ingestedBytes * 100 / sourceBytes)`を0〜99%へclamp
  - `caught_up` / `complete`: 100%
  - `error`: 最後にcommitできた`ingestedBytes / sourceBytes`を0〜99%へclampし、error表示を併記
- JSONLとgzipでAPI field、percentage計算、UI表現を分けない。
  精度区分を示すfieldや、gzipだけに`~`などの印を付ける仕様は設けない。
- live JSONLへの追記で`sourceBytes`が増えた場合、percentageが一時的に下がり、
  stateが`caught_up`から`converting`へ戻ることを許容する。
- `sourceBytes=0`で処理対象行がないRunは`caught_up`または`complete`として100%とする。

### 3.5 gzip

- block間で同一`GZIPInputStream`を保持し、各blockではDB接続だけを開閉する。
- gzip変換中に正常終了またはプロセス中断し、cache stateが`converting`のまま残った場合、
  次回起動時にそのcacheを破棄して先頭から再構築する。gzip途中位置からのresumeはしない。
- corrupt / truncated gzipは、それまでにcommitした部分dataを残してRunを`error`にする。
  source fingerprintが変わるまで再試行しない。
- 進捗は3.4のRun共通契約に従う。EOF確定前は`converting`のため最大99%、
  最終commit後に`complete`となって100%を表示する。
- activeなgzip変換中はstream handleを保持するため、そのRunフォルダの移動・削除は非対応。
  `caught_up` / `complete` / `error`後はhandleを閉じる。

### 3.6 state

| state | 意味 |
|---|---|
| `pending` | Runは存在するが、有効なcache generationがまだない |
| `converting` | 現在のマスタに未処理の完成行またはgzip streamがある |
| `caught_up` | JSONLの現在の完全改行までcommit済み。将来追記され得る |
| `complete` | immutable gzipのEOFまでcommit済み |
| `error` | Run-level fatal error。commit済み部分dataは保持する |

tag隔離だけではRunを`error`にしない。

### 3.7 schedulingとpriority

- priority集合は`POST /api/runs/prioritize`の要求で全置換する。
- 選択(priority)側3ブロック、背景側1ブロックを1 cycleとする。
- 各集合内はRun単位round-robinで、1 Runが1回に処理するのは最大1ブロック。
- 片側に処理可能Runがなければ、もう片側を連続処理してwriterを遊ばせない。
- 全Runが`caught_up` / `complete` / `error`で、処理可能なsource変化もない場合だけ10秒sleepする。

## 4. Public API

旧metrics APIとの後方互換は設けない。

### 4.1 `GET /api/runs.json`

```text
RunInfo {
  id,
  generation: UUID | null,
  stats: {
    maxStep: safe-integer | null
  },
  ingest: {
    state: pending | converting | caught_up | complete | error,
    ingestedBytes,
    sourceBytes,
    error?: {
      code,
      message
    }
  },
  tags: [{
    key,
    type: "scalar",
    status: ok | error,
    stats: TagStats,
    error?: {
      code,
      message
    }
  }]
}
```

- `generation`はDB全再構築ごとに変わり、同じDBへの通常追記では変わらない。
  `pending`で有効DBがない間は`null`。
- `stats.maxStep`はcommit済み`TagStats.maxStep`の最大。点がなければ`null`。
- 非scalarは`RunInfo.tags`へ含めない。
- tag隔離およびRun-level errorでも、commit済みのtagと統計は返す。
- `ingestedBytes` / `sourceBytes`とpercentageの意味はsource kindによらず3.4の共通契約に従う。
  source kindや進捗の精度区分は公開しない。

### 4.2 `POST /api/metrics.json`

#### request

1 batch内でrange modeとfollow modeをseries単位に混在できる。

```jsonc
{
  "series": [
    {
      "runId": "run-a",
      "tagKey": "10_train/loss",
      "fromStep": 1000,
      "toStep": 2000,
      "maxPoints": 4000
    },
    {
      "runId": "run-b",
      "tagKey": "10_train/loss",
      "fromOrdinal": 123456,
      "maxPoints": 4000
    }
  ]
}
```

- range modeはinclusiveな`fromStep`と`toStep`を両方指定する。
- follow modeは`fromOrdinal`を指定する。
- `maxPoints`省略時は`metricsviewer.target-points-per-series`を使う。
- 次はrequest shape違反としてHTTP 400:
  - range fieldとfollow fieldの混在
  - `fromStep` / `toStep`の片方だけ
  - どちらのmode fieldもない
  - unsafe step
  - 負の`fromOrdinal`
  - `fromStep > toStep`
  - `maxPoints < 3`またはglobal上限超過
  - 空または型不正の`runId` / `tagKey`

shape検証と最低quota実現性検証を通過したbatchは、データ状態にかかわらず
入力順・入力件数を保ったHTTP 200を返す。series単位のpending / not_found / errorを
HTTP errorへ昇格させない。request全体を拒否する例外は、
最低quotaを満たせないHTTP 422とquery slotを取得できないHTTP 503だけとする。

#### response

```text
{
  data: [{
    runId,
    tagKey,
    generation: UUID | null,
    status: ok | pending | empty | not_found | error,
    pointBudget,
    snapshotEndOrdinal,
    nextOrdinal,
    error?: {
      code,
      message
    },
    segments: [Segment, ...]
  }, ...]
}

Segment {
  level,
  bucketWidth,
  ordinalFrom,
  ordinalTo,
  steps?,        // base64 little-endian f64[]。L0の実step
  values?,       // base64 little-endian f32[]。L0だけ
  firstSteps?,   // base64 little-endian f64[]。LOD bucket先頭の実step
  lastSteps?,    // base64 little-endian f64[]。LOD bucket末尾の実step
  minSteps?,     // base64 little-endian f64[]。minが発生した実step
  maxSteps?,     // base64 little-endian f64[]。maxが発生した実step
  minOrdinals?,  // base64 little-endian f64[]。minが発生したabsolute ordinal
  maxOrdinals?,  // base64 little-endian f64[]。maxが発生したabsolute ordinal
  mins?,         // base64 little-endian f32[]。LODだけ
  maxs?,
  means?,
  lasts?,
  cnts?          // base64 little-endian f64[]。LODだけ
}
```

- `segments`は`ordinalFrom`昇順で、隙間・重複なく連続する。
  `ordinalTo`は排他的終端。
- L0では`level=0`、`bucketWidth=1`、1点につきstep/valueを1つ返す。
- LODでは同じ添字の`firstSteps/lastSteps/minSteps/maxSteps/minOrdinals/`
  `maxOrdinals/mins/maxs/means/lasts/cnts`が1 bucketを表す。
- `cnts`はbucketが覆うL0点数。bucket先頭ordinalは`ordinalFrom`から
  それ以前の`cnts`を加算して求め、bucket末尾ordinalは`先頭+cnt-1`で求める。
- stepとordinal位置はf64で転送し、JavaScript安全整数範囲内を正確に表す。
- MinMax用の1 bucketの候補点は次の3つ。
  - `(minOrdinal, minStep, min)`
  - `(maxOrdinal, maxStep, max)`
  - `(bucket末尾ordinal, lastStep, last)`
  同一ordinalを重複排除してordinal昇順に並べる。これにより1 bucketは1〜3 verticesとなり、
  stepが重複していても既存Min-Max-Lastと同じL0出現順を復元できる。
- `snapshotEndOrdinal`は、そのRunのread transaction開始時点にcommit済みだった
  対象tagの総L0点数。
- `nextOrdinal`は常に今回返したordinal区間の排他的終端。
  空のfollow結果では要求`fromOrdinal`、まだordinalを解決できないpending / not_foundでは0とする。
- `error`でも既知のcommit済み区間を`segments`へ添付できる。

statusの判定:

| status | 意味 |
|---|---|
| `ok` | 要求区間をcommit済みdataで返した |
| `pending` | Run/cache/tagが変換途中で、現時点では要求を解決できない |
| `empty` | Run/tagは存在するが、要求区間に点がない |
| `not_found` | Runが消失した、または現在のcommit済みマスタにtagが存在しない |
| `error` | Run-level errorまたはtag隔離。commit済み部分dataは返し得る |

### 4.3 range解決とLOD tiling

- range modeのinclusive step範囲は、非減少L0に対する二分探索で
  `[first(step >= fromStep), first(step > toStep))`のordinal範囲へ変換する。
  重複stepを境界上もすべて含める。
- follow modeは`[fromOrdinal, snapshotEndOrdinal)`を対象とする。
- queryはread transaction内で同じsnapshotを使い、途中commitを混ぜない。
- 完成済み永続LODを内部に使い、要求範囲の左右端だけは
  永続LOD/L0を合成した部分bucketをquery時に最大2個作る。
  範囲外ordinalを含めず、欠落も作らない。
- query時に合成する部分bucketも、value集約だけでなく
  `step_first/step_last/min_ordinal/min_step/max_ordinal/max_step`を保持する。
- 内部は粗いlevel、右端ほど細かいlevelの順にsegmentを並べる。
  vertex予算に余裕があれば最新側をL0まで細かくする。
- query合成bucketはDBへ書かず、LRUにも保存しない。
- 旧`level` / `tail` / `watermark`単一応答は廃止する。

### 4.4 点数予算

`maxPoints`はpayload上の点数ではなく、Plotlyの描画vertices数である。

- L0点は1 vertex。
- LOD bucketはMinMaxのmin / max / lastまたはBandのmin / max / meanを基準に
  最大3 verticesとして計算する。
- Mean表示を選んでいても予算を縮小せず、mode変更後も同じcacheを再利用する。
- seriesごとの最低quotaは
  `min(50, requestedMaxPoints, availableVertices)`。
- 最低quota合計が`metricsviewer.max-points-per-request`を超える場合はHTTP 422とし、
  次を返す。

```json
{
  "seriesCount": 12000,
  "requiredMinimumPoints": 600000,
  "maxPointsPerRequest": 500000
}
```

- 最低quotaを配った後の余りは、未充足seriesへwater-fillingで公平に再配分する。
- 各series結果の`pointBudget`に最終配分値を明示する。
- 有効batchを入力末尾から黙ってdropしたり、比例縮小したりしない。

### 4.5 query concurrencyとserver LRU

- metrics query全体をfair semaphoreで制限する。既定の同時実行数は2。
- slot超過時は最大5秒待つ。取得できなければHTTP 503、
  `Retry-After: 2`、error code `query_busy`を返す。
- 1 HTTP要求内ではseriesをRunごとにgroup化し、Runごとに短命read connectionを1つ開閉する。
- binary chunk LRUへ保存できるのは、完成済み永続LODだけ。
- LRU keyは`generation / run / tag / level / bucket range`。
- L0とquery合成bucketはcacheしない。
- Run消失検出時は、そのRunのLRU entryを即時削除する。

### 4.6 `POST /api/runs/prioritize`

```json
{"runIds":["run-a","run-b"]}
```

- 要求の`runIds`をpriority集合の全体として扱い、既存集合を全置換する。
- 存在しないRun id、配列以外、文字列以外はHTTP 400。
- 重複idは集合として1件に畳む。
- 空配列、既に同じ集合、変換済みRunだけの指定も成功しHTTP 204。

## 5. Frontend

### 5.1 Run選択

- Run一覧をcheckboxから即時toggle行へ変更する。1回目の操作で即座にON/OFFする。
- 同じ行への350ms以内の2回目操作はsoloとし、そのRunだけをONにする。
- 空選択を許可する。
- ページ初回ロードでRun一覧を初めて得たときだけLatestを自動選択する。
- 以後の手動空選択、選択Run消失による空選択、新Run追加後の空選択は維持する。
- `Select All` / `Latest Only`は維持する。

### 5.2 Run単位のDB取り込み進捗

- Run行へ`ingestedBytes / sourceBytes`から求めたpercentageを表示し、
  同じ割合まで背景を左から塗る。JSONLとgzipで表示を分けない。
- `pending` / `converting`では`0%`〜`99%`、`caught_up` / `complete`では`100%`を表示する。
- `error`では最後にcommitできたpercentageを残し、Run-level errorまたは隔離tag数の警告を併記する。
- `100%`表示は低彩度にして変換中Runを目立たせるが、値自体は省略しない。
- `title`へ`ingestedBytes / sourceBytes`の完全値とstateを載せる。
- gzip固有の接頭辞、記号、tooltip、色分けは設けない。

### 5.3 Run消失とstale応答

- 次の`runs.json`でRun消失を検出した時点で、Run一覧、server LRU、
  ingest scheduler、client viewport cacheから除去する。
- clientはselection、viewport、renderの各変更でrequest revisionを増やし、
  旧metrics通信を`AbortController`でabortする。
- response適用時にもmetadata revision、selection、Run generation、
  request revisionを再検証する。どれかが古ければ描画へ反映しない。

### 5.4 viewport cacheと取得

- seriesごとのclient cacheは現在viewportの左右1画面ずつを加えた計3画面分だけ保持する。
- `plotly_relayout`後150ms debounceし、手元の粗いcacheを即描画する。
- coverage不足または解像度不足なら、同一cycleで不足している可視seriesを
  1件の`metrics.json` batchへまとめる。
- serverが予算を保証するため、client側stride decimationは削除する。
- `MAX_POINTS=10000000`と追記専用の無制限numeric bufferは廃止する。

### 5.5 followとpolling

- Plotlyがautorange中、またはviewport右端と最新stepの差がviewport幅の5%以内ならfollow状態。
- 過去rangeを表示中は、Auto Reload時もmetrics queryを送らない。
- followは`fromOrdinal=nextOrdinal`で差分取得する。
- followで蓄積したBand換算verticesが実効`pointBudget × 3`を超える前に、
  現在viewportをrange modeで再取得してcacheを置換する。
- `pending`または`converting`のRunが1つでもある間は、
  Auto Reload設定に関係なく`runs.json`を2秒間隔でpollする。
- metricsを2秒更新するのは、
  選択中・表示中・変換中という3条件を満たすseriesだけ。
- 対象Runが`caught_up` / `complete` / `error`になったら強制2秒metrics更新を止める。

### 5.6 tag発見と空状態

- 新しく発見した可視tagは、現在のTag Filterに一致するかどうかにかかわらず自動ONにする。
- 一度認識したtagをユーザーがOFFにした状態は、以後のmetadata refreshでも維持する。
- pending初期変換中もRun一覧と操作を表示し、全画面blockingにしない。
- 選択なしは`No selection.`。
- Run選択済みだがcommit済みscalarがない場合は`No metrics data.`。

### 5.7 LOD表示mode

追加先はside panelの`.global-controls`ではなく、画面右上へfixed表示されている
`#floating-controls`とする。これは現行で`Scroll Lock: ON/OFF`と
screenshot切替`⬅`を横並びにしている枠である。

DOM上はScroll Lockボタンの直前へ、compactなlabelとselectを追加する。

```html
<div id="floating-controls">
  <span id="lod-display-mode-control">
    <label for="lod-display-mode">LOD:</label>
    <select id="lod-display-mode">
      <option>MinMax</option>
      <option>Mean</option>
      <option>Band</option>
    </select>
  </span>
  <button id="btn-graph-scroll-lock">Scroll Lock: OFF</button>
  <button id="btn-screenshot-toggle">⬅</button>
</div>
```

- selectは全graphへ作用するglobal設定であり、graphごとのheaderやside panelには複製しない。
- `#floating-controls`の既存flex配置、右上位置、Scroll Lock、screenshot切替の挙動を維持する。
- `#lod-display-mode-control`はinline-flexとし、label/selectの高さを既存Scroll Lockボタンと揃え、
  320px幅でも`#floating-controls`全体を1行に収める。
- screenshot modeではScroll Lockと同様に`#lod-display-mode-control`を非表示にする。

```text
MinMax / Mean / Band
```

- 既定は`MinMax`。
- 選択値を`localStorage`へ保存する。
- mode変更では再fetchせず、同じsegment列を再描画する。
- L0は全modeでraw折れ線。
- LODの描画:
  - **MinMax**: bucketごとにmin / max / lastの候補点を作り、同一ordinalを重複排除して
    ordinal昇順に並べる。各点のxには`minStep` / `maxStep` / `lastStep`という
    元L0の実stepを使い、全bucketを1本の折れ線として連結する。
    `step_first`へmin/maxを縦置きする描画は禁止する。
  - **Mean**: `firstSteps`上の`means`を結ぶ折れ線。
  - **Band**: 各bucketの`firstStep`を共通xとして、下端min、上端maxを別traceで結び、
    その間を塗ってmean線を重ねる。
    minとmaxを同じtrace内で直接結ばないため、MinMaxのような縦線は作らない。
    `minStep/maxStep`はhoverで実際のextrema位置を示すために使う。
- MeanとBandは初期実装へ残すが、実Run上の可読性を手動受け入れで評価する。
  継続、見え方の調整、将来の削除はその実画面を見て別途判断する。

### 5.8 TagStats表示

グラフheaderのLogボタン右へ、選択Run全体を合成した次を表示する。

```text
Min / Max / Avg / Std
```

- raw L0値基準の各Run `TagStats`を合成し、viewport rangeには依存させない。
- `min=min(min_i)`、`max=max(max_i)`。
- `M2_i=variance_i × count_i`として復元し、Chan合成を使う。

```text
delta = mean_b - mean_a
count = count_a + count_b
mean  = mean_a + delta * count_b / count
M2    = M2_a + M2_b + delta^2 * count_a * count_b / count
```

- 最終`Avg=mean`、`Std=sqrt(M2/count)`。
- 既存数値formatterを再利用し、title属性へ合計countと省略前の完全値を載せる。

### 5.9 error表示

- tag隔離はTag一覧とグラフheaderへ警告色、`⚠`、詳細tooltipを表示する。
- Run行には隔離tag数を集約表示する。
- Run-level errorでもcommit済み部分dataをグラフへ表示する。
- error文字列はserverの詳細を保持しつつ、通常レイアウトを押し広げない。

## 6. 設定・依存・起動

### 6.1 設定

`application.properties`と
`META-INF/additional-spring-configuration-metadata.json`を次へ置換する。

| key | 既定値 | 起動時validation |
|---|---:|---|
| `metricsviewer.target-points-per-series` | 4000 | 3以上かつglobal上限以下 |
| `metricsviewer.max-points-per-request` | 500000 | 3〜1,000,000 |
| `metricsviewer.cache-memory-mb` | 256 | 0以上、0は無効、runtime max heapの50%以下 |
| `metricsviewer.max-concurrent-queries` | 2 | 1〜4 |

不正値は起動時にfail-fastする。

次の旧設定はcode、properties、metadata JSONから削除する。

- `metricsviewer.max-transfer-points-initial`
- `metricsviewer.max-transfer-points-delta`
- `metricsviewer.decimation.enabled`

`metricsviewer.runs-dir`、1ブロック1,000,000行、全Run停止時の10秒sleepは維持する。

### 6.2 依存

- `org.xerial:sqlite-jdbc:3.53.1.0`へ固定する。
  [xerial公式リリース](https://github.com/xerial/sqlite-jdbc/releases)
- `com.esotericsoftware:kryo`依存を削除する。
- JDBCを直接使用し、ORMやconnection poolを追加しない。

### 6.3 起動heap

- 通常用batchとOptuna用batchの両方へ`-Xmx1g`を追加する。
- 設計文書にある直接起動手順も`java -Xmx1g -jar ...`へ更新する。
- 1 GiBは受け入れ条件であり、巨大Runに合わせてheapを増やして解決しない。

## 7. テスト

### 7.1 SQLite統合

- `application_id` / `user_version` / 必須schema
- 初期化完了時だけ`user_version`確定
- 無効DB、旧schema、corrupt DBの`.db/-wal/-shm`削除
- source kind変更
- JSONL切り詰め
- 同サイズ上書き
- より大きい別ファイルへの差し替え
- 正常追記
- 全再構築時だけgeneration変更、通常追記では不変
- 旧Kryo削除、Parquet非変更

### 7.2 LOD

- 全完成levelの`cnt/step_first/step_last/min_ordinal/min_step/vmin/`
  `max_ordinal/max_step/vmax/vmean/vlast`
- 上位level合成後もmin / max / lastが元L0の実stepとordinalを保持
- min / max同値時に最小ordinalを選ぶ決定性
- 16子未満を永続化しない
- 再起動時に各level最大15子から未完成状態を復元
- range左右端の部分bucket合成
- ordinalが隙間・重複なく連続する複数segment
- MinMaxの候補点を同一ordinalで重複排除し、ordinal順へ復元
- Bandのmin / max / mean 3 traceを含むvertex予算
- `maxPoints=3`の最小予算
- 重複stepをinclusive境界にすべて含める

### 7.3 `TagStats`

- Welfordによる逐次更新
- Chanによる複数Run合成
- 1点の`variance=0` / `stdDev=0`
- transaction rollbackで統計も戻る
- tag隔離前までの統計を保持
- invalid valueと隔離後の行を含めない

### 7.4 parser

- 未終端行をcommitしない
- 不正な完成行でblock全体rollback
- 必須field欠落
- 無効scalar valueを行だけskip
- unsafe stepでRun error
- step逆行でtagだけ隔離
- 未知の非scalar typeを原文保存
- WARN one-shot

### 7.5 gzip

- 同一内容のJSONLとgzipから同じL0 / LOD / `TagStats`
- 正常変換中断後はcacheを破棄して先頭から再構築
- corrupt / truncated gzipはcommit済み部分を残してerror
- 同一fingerprintで再試行しない

### 7.6 APIとscheduler

- batch入力順と同件数
- `ok` / `pending` / `empty` / `not_found` / `error`
- errorとpartial segmentsの併存
- JSONL / gzip共通の`ingestedBytes` / `sourceBytes`とstate別percentage
- range / follow混在
- LODのactual extrema step / ordinal列とL0 raw列
- snapshotとgeneration
- quota不足422の3 field
- priority集合の全置換
- priority / backgroundの3:1公平性と片側empty時の連続処理
- fair semaphore timeoutの503と`Retry-After`

### 7.7 Playwright

- Run toggle / 350ms solo / 空選択
- 初回だけLatest
- Run消失
- pending / converting中の2秒poll
- JSONL / gzipで同一のRun進捗percentage表示
- 新tag自動ONと既知OFF維持
- `#floating-controls`内でScroll Lock左に表示されるLOD mode select
- screenshot modeで`#lod-display-mode-control`を非表示
- LOD mode永続化とmode変更時no-fetch
- MinMaxがmin / max / lastを実stepかつordinal順に描画し、`step_first`縦線を作らない
- Bandが`firstSteps`を共通xとするmin/max帯とmean線を描画
- `TagStats`表示と複数Run合成
- tag / Run error警告UI
- revision / generationによるstale response破棄
- zoom時の精細化
- follow後の再圧縮
- 既存signed-log、scroll-lock、reload契約の維持

## 8. 受け入れ基準

### 8.1 自動ゲート

- 小型fixtureでJSONL / gzip / invalid / quarantine / append / rebuildの全契約を再現できる。
- 負荷生成testで、行数に比例する巨大Java object列を作らずstreaming ingestできる。
- Maven testとpackage、Playwrightが成功する。
- metrics queryのheap使用量がRun全点数ではなく、同時query数、series数、
  point budget、LRU上限で有界になる。
- 狭いviewportへzoomすると最終的にL0 rawへ到達する。
- LOD MinMaxのextrema位置が、対応するL0のstep / ordinalと一致する。
- cache削除後の再構築で同じcommit済み表示と`TagStats`を再現する。

### 8.2 7.31 GiB Runの手動受け入れ

`run_20260721-201834_cnx-vit128★`は自動testと分離した手動受け入れとする。

- 通常用・Optuna用とも`-Xmx1g`でOOMせず安定動作する。
- checkpoint後の`metrics_cache.db` / `-wal` / `-shm`合計が
  JSONL sizeの30%以下。
- short-lived DB connection終了後、Windows上でRunフォルダを移動できる。
- activeなgzip変換中Runだけはフォルダ移動試験対象外。
- 変換途中でもcommit済み部分を表示でき、進捗が99%を超えて完了扱いにならない。
- MinMaxのスパイクと谷がL0 spot checkと同じ実stepに描画され、人工的な`step_first`縦線がない。
- BandとMeanを実Runで切り替え、Bandの帯の判読性、Meanの有用性、多系列時の見やすさを記録する。
  この評価では初期実装からmodeを削除せず、継続判断はfollow-upとする。

## 9. 実装開始ゲートと文書同期

本PRDの実装へ着手する前に、別変更として次を完了する。

1. `CONTEXT.md`へ正準語`TagStats`とMetricsキャッシュ世代を追加する。
2. ADR 0015の履歴本文は書き換えず、follow-upとして次を記録する。
   - `TagStats`をLODから分離する理由
   - generationによるcache同一性
   - step逆行をRun全体ではなくtag隔離にする理由
3. 最初の実装メモ`docs/memo/041_metrics_sqlite_cache_20impl.md`を作り、
   本PRDを実装source of truthとして参照する。

## 10. スコープ外

- C++ `SqliteBackend`によるDB直接出力
- 自動gzip化、gzip化tool
- `tb_bridge.py` / `mlflow_bridge.py`などbridge類のSQLite対応
- 複数runs-dir、他drive直接参照
- Plotlyの置換
- 記録側intervalの変更
- activeなgzip変換中Runのフォルダ移動
- 複数Viewerプロセスの協調書き込み
- 旧metrics APIとの後方互換

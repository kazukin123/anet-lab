# MetricsViewer SQLite キャッシュ化(Phase 1: OOM 解消・LOD・ビューポート API)

> 設計分担: Claude=設計/PRD、実装=Codex、Run/commit=ユーザー。
> 本書は self-contained。実装時は行番号ではなく、近傍のシンボル名で再検索する。
> 用語(Run作業セット / Metricsマスタ / Metricsキャッシュ / 序数 / LODバケット)は
> リポジトリルート `CONTEXT.md` の「Metrics基盤」節が正本。
> 対応 ADR: [0015-metrics-cache-disposable-derivative.md](../adr/0015-metrics-cache-disposable-derivative.md)

## Context(背景・目的)

MetricsViewer(`viewers/metrics-viewer`、Spring Boot 3.5.7 / Java 17)には 3 課題がある。

1. **大規模 Run でサーバー側 OOM(最優先)** — 根因はストレージ形式ではなく Viewer の保持構造。
   `MetricsSnapshot` が `Map<TagInfo, List<Point>>` で**全 Run×全 tag×全点をヒープ保持**し、
   `MetricsRepository.findTagTraceDiff` の間引き(Min-Max-Last)は**転送時のみ**。
   `metrics_cache.kryo` スナップショットも全点を直列化しており、起動時ロードで同じ山を登る。
2. **`metrics.jsonl` の容量** — 無圧縮テキストで scalar 1 点 ≈ 94〜97 bytes。
3. **拡張性** — 事後集計・統計クエリの土台が JSONL 走査しかない。

さらに現行実装には解像度の構造問題がある: 初回転送は全履歴を 30,000 点へ間引くが、
以後の delta は新着だけを独立に間引いて**クライアントの追記専用バッファ**
(`numeric-buffers.js` の `Int32Buffer`/`Float32Buffer`)へ積むため、
系列の先頭は 1/2700 に潰れ末尾は原寸という不均一密度が蓄積し、
ズームしても**再取得経路が無いので精度が戻らない**。

### 実測(2026-07、設計根拠)

| 項目 | 値 |
|---|---|
| 対象 Run | `run_20260721-201834_cnx-vit128★`(7.31 GiB)ほか |
| 総行数 / scalar 行 | 80,936,134 / 80,936,120(非 scalar は **14 行**のみ) |
| tag 数 | 98(別 Run 実測 114) |
| scalar 1 行 | 平均 93.8 bytes |
| step の巻き戻り | **全 tag でゼロ**(非減少) |
| 同一 step の重複 | **1 tag で 240,109 件**(`20_eps/11_train_reward_ema`、episode 系。256 並列 env が同一 exp_step で複数終端するため正当) |
| SQLite 行サイズ実測 | L0 `(tag_id,step,value)` WITHOUT ROWID **18.86 B/点** / rowid 表+index 35.73 B/点 / LOD lean **52.36 B/bucket** |
| LOD 行数(factor16) | L0 比 **6.67%**(L1=316k/L2=19.8k/L3=1.2k/L4=78 per 5.06M 点 tag) |
| 容量見込み | L0+LOD ≈ **1.69 GiB**(JSONL 7.31 GiB の 23%) |
| gzip 実測 | -6 で 9.10x / 169 MB/s(参考。C++ 側 gz 出力は不採用が確定) |

### 解決の骨子

- 各 Run フォルダ内に **SQLite キャッシュ `metrics_cache.db`**(L0 全点+多重解像度 LOD)を
  **マスタ `metrics.jsonl` から従属構築**する。ヒープ保持は廃止し、読み出しは都度 SQL。
- API を**ビューポートクエリ**(step 範囲+点数予算 → 適切な LOD レベル)へ置換。
  解像度は表示範囲の関数になり、ズームすれば最終的に L0 生データへ到達する。
  サーバー/クライアントのメモリは **Run サイズ・Run 数に非依存**(表示系列数のみに比例)で有界化。
- Phase 2(スコープ外)で C++ 側に `SqliteBackend` を追加して DB を直接出力し、
  JSONL を任意ダンプへ降格する構想。本 PRD のスキーマはその移行を壊さない形にする。

## 1. 原則(アーキテクチャ不変条件)

1. **マスタ 1 種**: Run フォルダ内のメトリクス記録の正は `metrics.jsonl` のみ。
   手動 gzip 済みの `metrics.jsonl.gz` は**同一マスタのライフサイクル段階**(不変ファイル)として
   透過に読む。マスタを 2 種並立させない。C++ 側の gz 出力・自動 gz 化ツールはやらない(確定済み)。
2. **キャッシュは破棄可能**: `metrics_cache.db` はいつ削除してもマスタから同一内容を再構築できる。
   様式・整合性が疑わしければ**警告なしに全破棄・再構築**する(migration は書かない)。
3. **短命接続**: SQLite 接続は読み=リクエスト毎、書き=取り込みブロック毎に開閉する。
   Run フォルダの出し入れ(=Run作業セットへの登録/解除。Windows はオープン中ファイルを
   移動・削除できない)をロックで阻害しない。ロック窓は現行 JSONL ポーリング並みの瞬間に保つ。
4. **全自動変換**: `runs` ディレクトリ内にあること自体が「見る」意思表示。
   起動時から全 Run を順次変換し、UI で選択された Run は変換キューの先頭へ。
   変換途中でも取り込み済み分から部分表示する。
5. **解像度はビューポートの関数、メモリ量はその帰結**: 空きメモリに応じて保持を増やす方式は
   採らない。余剰メモリは精度ではなく先読み(レイテンシ)に使う。

## 2. スキーマ(DDL)

```sql
-- ファイルヘッダ(テーブルに触れる前に判定できる様式識別)
PRAGMA application_id = 0x414E4554;  -- "ANET"
PRAGMA user_version   = 1;           -- キャッシュ様式バージョン。1 始まり単調増加
PRAGMA journal_mode   = WAL;
PRAGMA synchronous    = NORMAL;
-- 接続毎: PRAGMA busy_timeout = 5000;

CREATE TABLE tags(
  id   INTEGER PRIMARY KEY,
  key  TEXT UNIQUE NOT NULL,
  type TEXT NOT NULL               -- 現状 'scalar' 固定(将来拡張用)
);

-- L0: 全点。PK は序数(tag 内出現順 0,1,2,...)。step は座標値であり identity ではない
CREATE TABLE scalars(
  tag_id  INTEGER NOT NULL,
  ordinal INTEGER NOT NULL,
  step    INTEGER NOT NULL,        -- int64。非減少・非一意(重複 24 万件/Run の実測あり)
  value   REAL    NOT NULL,
  PRIMARY KEY(tag_id, ordinal)
) WITHOUT ROWID;

-- LOD: 序数バケット。bucket = ordinal / 16^level(整数除算)。level=1..N
CREATE TABLE scalars_lod(
  tag_id     INTEGER NOT NULL,
  level      INTEGER NOT NULL,
  bucket     INTEGER NOT NULL,
  cnt        INTEGER NOT NULL,
  step_first INTEGER NOT NULL,     -- バケット先頭点の step(範囲→序数変換の二分探索キー)
  vmin       REAL    NOT NULL,
  vmax       REAL    NOT NULL,
  vlast      REAL    NOT NULL,
  vsum       REAL    NOT NULL,     -- mean = vsum/cnt。cnt/vsum は加算合成可能
  PRIMARY KEY(tag_id, level, bucket)
) WITHOUT ROWID;

-- 非 scalar 行の原文保存(meta / json / video / Grid / CNBlock / Conv2d 等。実測 14 行/Run)
CREATE TABLE json_lines(
  ordinal   INTEGER PRIMARY KEY,   -- 非 scalar 行の出現順
  type      TEXT NOT NULL,
  tag       TEXT,
  step      INTEGER,
  timestamp TEXT,
  json      TEXT NOT NULL          -- マスタ行の原文
);

-- ソース指紋。様式バージョンはここに置かない(user_version が正本)
CREATE TABLE source_meta(
  k TEXT PRIMARY KEY,
  v
);
-- k: source_kind ('jsonl' | 'jsonl.gz') / source_size / source_mtime / ingested_bytes
```

設計注記:

- **WITHOUT ROWID が必須**(rowid 表+index は実測 1.9 倍の容量)。
- **step への補助インデックスは張らない**。step 範囲→序数範囲は
  「L1 の `step_first` を二分探索 → 序数範囲で L0/LOD を PK レンジスキャン」で引く
  (step 非減少は実測で保証。L1 は 5M 点 tag で 316k 行、二分探索は数回の point query)。
- **LOD レベル数は動的**: 各 tag で「最粗レベルのバケット数 ≤ `target-points-per-series`」に
  なるまで積む(5M 点 → L4、将来 100M 点 tag なら自動で L5)。factor は 16 固定。
- `value`/LOD 各値は REAL(f64)。転送時に f32 へ落とす(現行と同じ精度契約)。

## 3. 取り込みパイプライン(LoadingThread の再設計)

現行 `LoadingThread`(10 秒ポーリング、`MAX_LINES=1,000,000` 行/ブロック)の骨格と
選択 Run 優先の `request` 機構は流用し、`MetricsRepository.mergeMetrics`+kryo 保存を
SQLite 取り込みへ置換する。取り込みスレッドは現行同様 1 本(=DB への単一 writer)。

### 3.1 Run ごとの判定フロー(ポーリング毎)

```
1. マスタ特定: metrics.jsonl があればそれ。無ければ metrics.jsonl.gz。
   両方あれば jsonl を優先し WARN(Run ごと 1 回)。
2. metrics_cache.db が存在すれば様式チェック(ヘッダのみ、テーブルに触れない):
   application_id != 0x414E4554 または user_version != 現行値 → DB ファイル削除して再構築へ
3. ソース指紋チェック(source_meta):
   - source_kind が変わった(jsonl ⇄ jsonl.gz)      → 全破棄・再構築
   - size <  ingested_bytes(切り詰め・差し替え)      → 全破棄・再構築
   - size == ingested_bytes かつ mtime 変化(上書き疑い)→ 全破棄・再構築
   - size >  ingested_bytes                            → 差分取り込み
   - size == ingested_bytes かつ mtime 不変            → 何もしない
   ※ .gz は不変ファイル前提: ingested 済みなら size 一致以外は全て再構築
4. 取り込み: ブロック単位で JSONL をパースし DB へ書く(3.2)
```

旧 `metrics_cache.kryo` を Run フォルダに発見したら削除する(本 Viewer 自身の旧形式)。
`metrics_cache.parquet` には**触れない**(Python 版ツールの所有物)。

### 3.2 ブロック取り込み(トランザクション整合)

- パーサは現行 `MetricsFileReader`(Jackson streaming、1 byte 読みのオフセット管理、
  改行未達の書きかけ行をコミットしない設計)を流用し、次を拡張する:
  - `step` を **long** で読む(現行 int)。`MetricsFileLine.step` も long 化。
  - 非 scalar 行は**行の原文文字列**を保持する(`json_lines` 挿入用。scalar 行では保持しない)。
  - `.gz` の場合は `GZIPInputStream` を挟む。オフセットは**圧縮前ストリーム位置ではなく
    「gz は一括取り込み・追尾なし」**とし、ingested_bytes には圧縮ファイルサイズを記録する。
- **1 ブロック=1 トランザクション**で以下を原子的に commit する:
  1. scalar 行 → `scalars` へ INSERT(tag 初出時は `tags` へ INSERT し id 採番)
  2. 確定した LOD バケット → `scalars_lod` へ INSERT
  3. 非 scalar 行 → `json_lines` へ INSERT
  4. `source_meta.ingested_bytes` を当該ブロック末尾オフセットへ更新
  クラッシュしても「commit 済みブロックまで取り込んだ」状態が保たれ、
  ingested_bytes と DB 内容が常に整合する。書き込みは全テーブル **INSERT-only**。
- **序数採番**: tag ごとに次 ordinal をメモリ保持。再開時は
  `SELECT MAX(ordinal) FROM scalars WHERE tag_id=?`(PK で O(logN))+1 から。
- **LOD 逐次構築**: (tag, level) ごとに開バケットのアキュムレータ
  `{bucket, cnt, step_first, vmin, vmax, vlast, vsum}` をメモリ保持し、
  点 1 つにつき全レベルを更新(O(レベル数)=O(5)、クエリゼロ)。
  バケット境界(`ordinal % 16^level == 0` に到達)で確定行を INSERT してリセット。
  - **部分バケットは DB に書かない**(確定のみ)。再開・再起動時は、各レベルの
    確定境界 `floor(point_count / 16^level) * 16^level` 以降の点を L0 から読み直して
    アキュムレータを復元する(最深レベルでも高々 16^N−1 点の PK レンジスキャン)。
- 書き込み接続はブロック処理の間だけ開き、commit 後にクローズする(短命接続)。

### 3.3 変換進捗の公開

`ingested_bytes / source_size` を Run 情報として API に載せる(4.1)。
100% は「マスタ末尾まで取り込み済み」を意味し、ライブ Run では追記で再び 100% 未満に戻り得る。

## 4. クエリ / API

読み接続はリクエスト毎に開閉。`PRAGMA busy_timeout=5000` で取り込み commit と共存する
(WAL のため読みは書きをブロックしない)。

### 4.1 `GET /api/runs.json`(拡張)

現行 `GetRunsResponse{runs: RunInfo[]}` の `RunInfo{id, stats, tags}` に変換状態を追加する:

```
RunInfo {
  id, tags: [{key, type}],
  stats: { maxStep },           // 全 tag の最終 step の最大(L0 の tag 別末尾から導出)
  ingest: { state: 'pending'|'converting'|'live'|'complete',
            ingestedBytes, sourceBytes }   // 進捗% = ingestedBytes/sourceBytes
}
```

未変換 Run(DB 無し)も一覧に出す(`state='pending'`、tags 空)。
`TagStats`(mean/variance 等)の系列全体統計は最粗 LOD の畳み込みで導出できるため、
現行 `TagStats` クラスの全点更新実装は廃止する。
なお現行 `TagStats.getMinStep()` の int/long 比較バグ(空時に 2147483647 を返す)は
実装ごと消滅する。

### 4.2 `POST /api/metrics.json`(置換)

現行の `{runTagMap}` リクエストと `TagTrace` レスポンスを置換する。

**リクエスト** — 2 モードを series 単位で混在可:

```jsonc
{
  "series": [
    // (a) 範囲クエリ: 表示用。step 範囲+点数予算
    { "runId": "...", "tagKey": "...",
      "fromStep": 0, "toStep": 5000000,   // 省略時は全範囲
      "maxPoints": 4000 },
    // (b) 追尾クエリ: ライブ追従。序数起点(前回レスポンスの nextOrdinal)
    { "runId": "...", "tagKey": "...",
      "fromOrdinal": 5062232, "maxPoints": 4000 }
  ]
}
```

step は非一意なので**差分追尾の起点は step ではなく序数**で指定する(重複・欠落を排除)。
サーバーは総量予算 `max-points-per-request` を系列間で公平配分する:
`quota_i = min(範囲内実点数_i, 公平配分)` → 余りを未充足系列へ再配分 → 下限 50 点を保証。
それでも超過する場合は maxPoints を比例縮小する(黙って切り捨てない)。

**レベル選択**: 範囲クエリでは L1 `step_first` の二分探索で `fromStep..toStep` を
序数範囲 `[o1, o2)` に変換し、`level = clamp(ceil(log16((o2-o1) / maxPoints)), 0, 最深)`。
level=0 は L0 生を返す。

**レスポンス** — 列分離。数値配列は現行 `MetricTraceEncoder` の
base64 little-endian チャンク方式を流用(f64 エンコーダを追加):

```jsonc
{
  "data": [{
    "runId": "...", "tagKey": "...",
    "level": 2, "bucketWidth": 256,
    "steps":  "base64 f64[]",            // level>0: step_first / level=0: 生 step
    "mins":   "base64 f32[]", "maxs": "...", "means": "...", "lasts": "...",
    "cnts":   "base64 f32[]",            // level>0 のみ。level=0 は values のみ
    "values": "base64 f32[]",            // level=0 のみ
    "tail":   { "steps": "base64 f64[]", "values": "base64 f32[]" },
    "watermark":   4980736,              // 確定 LOD 末尾の step(表示用)
    "nextOrdinal": 5062232               // 次回追尾クエリの起点(識別用)
  }]
}
```

- **steps 転送は f64**(JS の Number に直結、2^53 まで正確)。pause→resume 700M step
  運用が視野にあるため int32 は採らない。
- **tail** = 選択レベルの確定バケット境界以降の L0 生。予算の一部(目安 maxPoints/4)を
  上限とし、超過時は最新側を優先する(粗いズームアウト中の tail 細部は視認不能のため)。
- **サーバー側クエリ結果キャッシュ**: 確定バケット由来の応答のみ、
  キー `(runId, tagKey, level, bucket範囲)` で上限 `cache-memory-mb` の LRU に保持
  (実装は手製 LinkedHashMap で可。依存追加はしない)。tail と未確定領域は毎回読み直す。
  Run フォルダ消失を検出したら該当 Run のエントリを即破棄する。

## 5. フロントエンド(`static/metrics-viewer.js` ほか)

### 5.1 ビューポート再取得型への転換

- `DataCache` の**追記専用・無制限バッファを廃止**し、系列ごとに
  「表示範囲の 2〜4 倍(oversample)を覆う取得済み区間+そのデータ」だけを保持する。
  範囲外は破棄。クライアント保持は 1 系列 ≈ 数万点で頭打ちになる。
- `plotly_relayout`(範囲変更)→ **150ms デバウンス** → 手元データで即時描画
  (プログレッシブ精細化: 粗いデータでまず描き、レスポンス到着で差し替え)→
  キャッシュ被覆判定 → 不足なら範囲クエリを fetch。
- ライブ追従: Auto Reload(現行 30 秒)は `runs.json` を常に叩き、
  **右端が最新に近い(追従状態の)系列のみ**追尾クエリ(`fromOrdinal`)を送る。
  過去範囲を表示中の系列は確定データが不変なのでポーリング不要。
- `decimateTrace` は**範囲 slice(二分探索)だけ残し、ストライド間引きを削除**する
  (点数上限はサーバーが保証する。ストライド間引きは min/max を落とすため禁止)。
  `MAX_POINTS=10000000` 定数は削除。

### 5.2 描画(全部「線」)

- 既定: level>0 では各バケットの min/max を x=step_first 上に 2 点並べた**単一折れ線**
  (密部が自然に帯状に見える、オシロスコープ方式)。level=0 は通常の折れ線。
- トグル: mean 線(means=vsum/cnt)/ min-max 帯塗り+mean 線(系列数が少ない時用)。
- ツールチップに cnt / min / max / mean を出せる(列分離レスポンスの customdata)。

### 5.3 Run 一覧のトグルリスト化(チェックボックス廃止)

現行 `renderRunList`/`bindRunListEvents` の `label+checkbox+<br>` 構造を、
タグ一覧(`#tag-list li.active`)と同じトグルリストへ置換する:

- **タップ/クリック=トグル**(即時発火、遅延なし)。
- **素早い同一行 2 回目タップ=solo**(その Run だけ残して他を全部外す)。
  1 回目で既にトグル済みなので判定遅延ゼロでダブルタップ solo が成立する。
  PC のダブルクリックも同義。`touch-action: manipulation` でモバイルの
  ダブルタップズームを抑止する。
- 色チップ(`span.run-color`)は維持。**空選択を許容**する(既存の「No selection.」
  表示があり、`Latest Only` ボタンで即復帰できるため。現行の
  「最後の 1 件は外せない」制約と行クリック=ラジオ動作は廃止)。
- 現行 CSS の不可視ハイライト(`.run-row input:checked + span` が色チップの
  インライン背景に負ける/`.run-row.active` を付与する JS が無い)はこの置換で消滅する。
- `Select All` / `Latest Only` ボタンは現行機能を維持する。

### 5.4 変換進捗の表示(場所を取らない)

- Run 行の**背景を左から進捗%まで塗る**(`linear-gradient(to right, rgba(42,107,149,0.28) p%, transparent p%)` 相当)。
  追加 DOM なし。現行の固定ダーク配色(#181818 面、選択ハイライト系 #1e4d70)に整合する
  暗青・低彩度とし、選択状態の視認を妨げない濃度にする。
- `state='complete'` で消灯。title 属性(既存 stats ツールチップ)に進捗%を併記する。
- `state='pending'|'converting'` の Run はグラフ選択可(部分表示で伸びていく)。

## 6. 設定

`application.properties` + `META-INF/additional-spring-configuration-metadata.json` を更新する。

| キー | 型 | 既定値 | 意味 |
|---|---|---|---|
| `metricsviewer.target-points-per-series` | Integer | 4000 | 系列あたり目標点数(≈2 点/px) |
| `metricsviewer.max-points-per-request` | Integer | 500000 | 1 リクエストの総点数予算 |
| `metricsviewer.cache-memory-mb` | Integer | 256 | サーバー側クエリ結果 LRU の上限 |

削除(コード・properties・metadata JSON から): `metricsviewer.max-transfer-points-initial` /
`metricsviewer.max-transfer-points-delta` / `metricsviewer.decimation.enabled`。
`metricsviewer.runs-dir`、ポーリング 10 秒、`MAX_LINES` 100 万行/ブロックは現行維持。

## 7. 依存・削除対象

- pom.xml: `org.xerial:sqlite-jdbc` を追加(本 pom の個別バージョン明記方式に合わせる)。
  `com.esotericsoftware:kryo` は削除。
- 削除・置換されるサーバー実装: `MetricsSnapshot`(全点ヒープ保持)、
  `MetricsRepository` の kryo save/load と `decimatePoints`、`Point`、
  `TagStats` の全点逐次更新(LOD 畳み込みへ)。
  `RunScanner` / `MetricsFileReader`(拡張の上)/ `MetricTraceEncoder`(f64 追加)/
  `LoadingThread` の骨格と request 機構は流用。
- 多重起動は非サポートを明記(単一 Viewer プロセス前提。偶発的な二重起動でも
  WAL+busy_timeout により破損はしない)。

## 8. テスト

旧 `MetricsRepositoryTest` / `LoadingThreadTest` / `MetricsFileReaderTest` は置換する。

1. **LOD 構築の厳密検証**: 既知系列(例: 1..N の三角波+単発スパイク)を取り込み、
   全レベルの cnt/vmin/vmax/vlast/vsum/step_first を期待値と完全一致で照合。
2. **整合性判定**: 様式不一致(user_version/application_id)・source_kind 切替・
   切り詰め・上書き疑い → 全破棄再構築 / 追記 → 差分。5 分岐を網羅。
3. **序数**: 重複 step の全点保持、再開時の MAX(ordinal) 継続、
   部分バケットのアキュムレータ復元(再起動を模擬)。
4. **ブロックトランザクション整合**: ブロック途中の失敗を模擬し、
   ingested_bytes と DB 内容が commit 境界で一致することを確認。
5. **gz 透過**: 同一内容の `.jsonl` と `.jsonl.gz` から同一キャッシュが構築されること。
6. **レベル選択と予算配分**: 範囲・maxPoints からの level 決定、系列間公平配分、下限保証。
7. **Playwright**(`PalettePlaywrightTest` を改修): Run 一覧のトグル/solo/空選択、
   進捗表示、ズーム時の再取得と精細化、追尾クエリでのライブ追従。

## 9. 受け入れ基準

1. サーバーヒープが Run サイズ・Run 数に**非依存**(表示系列数のみに比例)。
   7 GiB 級 Run を複数同時表示+全 tag 選択でも既定ヒープで安定。
2. ズームで解像度が回復し、狭い範囲では L0 生データに到達する(現行は初回間引きで固定)。
3. Viewer 稼働中に Run フォルダの出し入れ・リネームが成功する(変換中の当該 Run を除く)。
4. 変換進捗が Run 一覧で視認でき、変換中 Run も部分表示できる。
5. キャッシュ削除 → 再起動でマスタから同一表示が再現する。
6. `metrics_cache.db` 容量が JSONL の 25% 前後(実測見込み 23%)。

## 10. スコープ外(明記)

- **Phase 2**: C++ `SqliteBackend` による DB 直接出力、JSONL の任意ダンプ化、
  `dump_jsonl.py`。本 PRD のスキーマ・user_version・source_kind がその互換の枢軸。
- `tb_bridge.py` / `mlflow_bridge.py` の読み替え(当面 JSONL を読み続ければ動く)。
- gz 化ツール・自動 gz 化(手動 gzip で足りる)。
- 複数 runs-dir・他ドライブ直接参照(見たい Run は runs へ入れる運用)。
- Plotly 代替ライブラリ(点数が有界化された後に別件で再評価)。
- 記録側の interval 変更(LOD が容量面で優越するため不要になった)。

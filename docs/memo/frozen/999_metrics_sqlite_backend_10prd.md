# C++ SqliteBackend による metrics DB 直接出力(Phase 2・暫定)

> 凍結中(再開条件: PRD 041 の実装と安定運用)

> 状態: **backlog**。方式確定と実装は Phase 1
> ([041_metrics_sqlite_cache_10prd.md](../done/041_metrics_sqlite_cache_10prd.md))の実装と安定運用を待って行う。
> 本書は確定済み前提と未決論点を区別して保存する備忘であり、このまま実装に着手しない。
> 関連: [041_metrics_sqlite_cache_10prd.md](../done/041_metrics_sqlite_cache_10prd.md)、
> [adr/0015-metrics-cache-disposable-derivative.md](../../adr/0015-metrics-cache-disposable-derivative.md)、
> `CONTEXT.md`「Metrics基盤」節(Run作業セット / Metricsマスタ / Metricsキャッシュ / 序数 / LODバケット)。
> 設計分担: Claude=設計/PRD、実装=Codex、Run/commit=ユーザー。

## Context(背景・位置づけ)

Phase 1 では Metricsマスタは `metrics.jsonl` のままとし、Viewer が Run 毎の破棄可能キャッシュ
`metrics_cache.db` を従属構築する。Phase 2 は**マスタ地位そのものを JSONL から SQLite DB へ移す**:
C++ 側(`MetricsLogger` の backend)が Run 中に直接 `metrics.db` を書き、JSONL は必要時に
DB からダンプする任意成果物へ降格する。

動機の現在価値(Phase 1 設計時からの再評価):

1. **キャッシュ変換工程の消滅(主動機)** — Phase 1 では 7 GiB 級 Run の初回変換に数分かかる。
   C++ が最初から DB を書けば、Viewer は変換なしで即読みでき、`ingested_bytes` 追尾・
   整合性再構築の機構も不要になる。
2. **マスタ自体の SQL 可読性** — AI エージェントや分析ツールが未変換 Run にも
   Python 標準 `sqlite3` で即クエリできる(Phase 1 ではキャッシュ構築済み Run に限られる)。
3. **容量(副次)** — JSONL 比で実測見込み 23%(L0+LOD)。ただし「過去 Run は別ドライブへ
   フォルダごと退避」という運用が確立したため、当初最優先だった容量動機は副次に降格した。
   アーカイブ圧縮は手動 gzip(Phase 1 の透過読み)で足りている。

## 確定済み前提(041 / ADR 0015 から継承。ここは再議論しない)

- **スキーマは Phase 1 と同一系**: `tags` / `scalars`(PK=`(tag_id, ordinal)`、step は座標値)/
  `scalars_lod`(factor16 序数バケット)/ `json_lines` / `source_meta`。
  `PRAGMA application_id=0x414E4554` + `user_version` が様式判定、
  `source_meta.source_kind` がマスタ種別判定の枢軸。
- **ファイル名で役割を区別**: C++ が書くマスタは **`metrics.db`**、Viewer の従属キャッシュは
  `metrics_cache.db`。マスタは破棄不可、キャッシュは破棄可能という契約が名前から判別できる。
- **WAL 並行読み**: Run 中(C++ が writer)でも Viewer は read-only 接続で追従できる。
  Run フォルダ運用(Run作業セットへの出し入れ)は維持。Viewer 側は短命接続を続ける。
- **C++ 側の gz 出力はやらない**(確定済み。JSONL 圧縮は Phase 1 の手動 gzip + 透過読みの領分)。
- 実測前提: 書き込みレートは数千 scalar 行/s 程度(80.9M 行 / 約20h Run)。
  SQLite のバッチ INSERT + 定期 commit は 100k 行/s 以上を余裕で処理できる。

## 暫定方針(方向は固めたが、grill 未実施)

- 既存の backend 抽象(`metrics_logger.hpp` の `IBackend::Open/WriteJsonl/Flush`、
  実装 `JsonlBackend`)に `SqliteBackend` を追加する。
  ただし現行 interface は「1 行の json オブジェクトを書く」形であり、scalar 高速路
  (tag_id 解決済み・パース不要の型付き書き込み)を持たない。interface 形状の再設計は未決論点へ。
- sqlite3 は **amalgamation(sqlite3.c/h)を `third_party/` へ**置く(public domain、
  DLL 不要、CMake ターゲット 1 つ)。vcpkg 依存は増やさない。
- 書き込みは**バッチ INSERT + 約 1 秒毎の commit**、`PRAGMA journal_mode=WAL` /
  `synchronous=NORMAL`。commit 間隔がライブ反映レイテンシの上限になる
  (現行 ofstream バッファ+Viewer 10 秒ポーリングと同等以下)。
- **移行期は dual-write 設定**(`metrics.db` + `metrics.jsonl` 併記)を用意し、
  安定後に JSONL を止める。最終形では `viewers/metrics-tools/dump_jsonl.py`(新規、DB→JSONL)で
  「JSONL ダンプも必要に応じて可能」の要件を満たす。
- Viewer は Run フォルダに `metrics.db` があればそれをマスタとして直接読み、
  JSONL 取り込み・キャッシュ構築をスキップする(`metrics.jsonl` しか無い旧 Run は
  Phase 1 経路のまま)。両方あるときの優先規則は未決論点へ。

## 未決論点(本書の本体。Phase 2 着手時に grill で確定させる)

1. **LOD を誰が書くか(最大の分岐)**
   - 案(a) C++ が L0+LOD 両方を書く: Viewer は完全読み専になり構成が最简。
     ただし LOD アキュムレータ(tag×level の開バケット、部分バケット非永続、
     resume 時の復元)を C++ に再実装する。
   - 案(b) C++ は L0(+`json_lines`)のみ書き、Viewer が LOD だけを従属構築して
     `metrics_cache.db` に置く: 「マスタに Viewer が書かない」原則(ADR 0015)を
     最も素直に守れるが、ファイル 2 本構成が恒久化し、Phase 1 の取り込み機構の一部が残る。
   - 判断材料: C++ 実装コスト、Viewer 側機構の残存量、Run 中の LOD 追従レイテンシ。
     トレードオフ未評価。
2. **`IBackend` interface の形状**: 現行 `WriteJsonl(const json&)` は SqliteBackend では
   「json を組んで即分解」の無駄が出る。`LogScalar(tag, step, value)` 直通の型付き経路
   (例: `IBackend::WriteScalar(tag_id or string, step, value)`)を足すか、
   backend 内で json パースを許容するか。`MetricsLogger::LogScalar` は
   Observer callback から Train/Learn critical path 上で呼ばれる点に注意
   (性能測定・ProfileRange ルール適用対象)。
3. **書き込みスレッド設計**: commit バッチングの所在。
   - 呼び出しスレッド内バッファ+時間契機 flush(現行 JsonlBackend の mutex 構成に近い)
   - 専用 writer スレッド+lock-free/mutex キュー(critical path から I/O を完全排除)
   - いずれでも `Flush()` 境界(`FlushRunOutputs`: pause/save/shutdown)で確定 commit する契約は維持。
4. **クラッシュ時の未 commit ロス**: 約 1 秒分の損失は現行 ofstream バッファロスと同等か広いか。
   強制 kill が日常の運用で許容幅を明文化する(WAL により commit 済みは保証される)。
5. **`json_lines` 相当の書き込み経路**: `LogJsonInternal`(config dump)、meta(start)、
   video 行、`json/` `config/` ディレクトリへのバラ書きとの整合。Run 再開(resume)時の
   meta 行の扱いも含む。
6. **resume(同一 Run dir への追記)時の継続**: ordinal 採番と LOD アキュムレータの復元は
   Phase 1 の再開手順(MAX(ordinal) + L0 読み直し)をそのまま流用できる見込み。要確認のみ。
7. **旧 JSONL Run の一括変換ツールの要否**: Run作業セット運用では「見たい Run だけ
   Phase 1 キャッシュが吸収」で足りる可能性が高い。別ドライブの大量アーカイブを
   一括変換する動機が実際に生じるかを見てから決める。
8. **`metrics.db` と `metrics.jsonl` が両方ある Run の優先規則**: dual-write 期の正常形と、
   片方だけ手で消された異常形の区別。source_kind と突き合わせた fail-fast / WARN 設計。
9. **tb_bridge.py / mlflow_bridge.py の読み替え時期**: 当面 JSONL(dual-write / ダンプ)を
   読み続ければ動く。DB 直読への移行は優先度低のまま独立に判断。
10. **DB ファイル運用**: WAL checkpoint の契機、Run close 時の checkpoint(TRUNCATE)、
    長期 Run での `-wal` サイズ、VACUUM の要否(INSERT-only なので断片化は小さい見込み)。

## スコープ外

- Phase 1 の設計・実装内容の再説明(041 を正とする)。
- TensorBoard / MLFlow ブリッジの機能拡張。
- 記録側 interval・メトリクス定義の変更。

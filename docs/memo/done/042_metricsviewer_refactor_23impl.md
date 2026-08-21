# PRD 042 T6 SourceReader seam 実装計画

## 概要

本書をT6の実装正本とする。raw/gzipの読み取り差を`SourceReader`へ閉じ込め、
`MetricsIngestor.ingestBlock`をtransaction境界・状態遷移・エラー分類へ限定する。
HTTP API、SQLite schema、取り込み結果は変更しない。

## 実装内容

- package-private `SourceReader`と`RawFileReader` / `GzipSessionReader`を追加する。
  - `prepare()`で`hasSession -> database.prepare -> acquire/open`を完結させる。
  - 完全行と行末時点のsource offsetをcallbackへ渡す。
  - 読了結果はlogical EOF、最終消費offset、commit可能offset、未終端bytesを保持する。
  - raw offsetは非圧縮byte、gzip offsetは圧縮stream消費byteであることを明記する。
  - rawは取得時点のsource sizeを論理末尾とし、未終端行を保留したまま`ready`にできる。
  - gzipは同一sessionの展開済みdataを次blockへ引き継ぎ、`converting`ではsessionを保持し、
    `ready` / `error`では閉じる。
- `ingestBlock`はmode非依存の読了結果から`ready` / `converting`を決定する。
  rawの`IOException` / `SQLException`を`source_read_error`にする経路と、
  gzipの`SQLException`を同errorへ丸めない既存経路を維持する。
  `loadTagState`内の`SQLException`もchecked例外としてrawの同じ分類へ到達し、
  Runを`error`にすることを明示的な許容差分とする。
- private `BlockWriteSession`へConnection、PreparedStatement、LOD insert session、
  tag状態、警告集合、runIdを集約し、`writeScalar`の9引数リレーを解消する。
- tag状態は`Map.get`後、未登録時だけ`loadTagState()`して`put`する。
  `loadTagState()`はchecked `SQLException`を直接伝播し、
  `DatabaseWriteRuntimeException`を削除する。
- 新規コメントは日本語とし、offset単位、gzip session寿命、transaction境界、
  既知のSQLException分類を維持する理由を記録する。

## 互換性

- public constructor、`ingestBlock`、`IngestOutcome`、HTTP JSON、SQLite schema・metadata値は変更しない。
- `loadTagState`の`SQLException`がraw Runを`source_read_error`へ遷移させる裁定を除き、
  DB内容、generation、state遷移、`didWork`、quarantine・warn条件、
  `committed_offset`、block上限、LOD・TagStatsを維持する。
- `GzipInputSessions`のbean所有とschedulerの`retainRuns` / `closeAll`経路を維持する。
- `10prd`、`20impl`〜`22impl`、ADR0015、`CONTEXT.md`、T1〜T5、付録A、既知バグは変更しない。

## TDDと検証

- 実装前baseline: `mvn -B test`で82件成功。
- `MetricsIngestorIntegrationTest`へ、64 KiB未満のgzipを2行/blockで3回処理する
  characterizationを追加する。
  - 初回は2点をcommitして`converting`。
  - 圧縮offsetがファイル末尾でも同一sessionの展開済み行から次blockで4点まで進む。
  - generation、offset、各blockの`didWork`を現行値で固定する。
  - 最終blockで全5点、`ready`となりsessionが終了する。
- raw追記再開、未終端行、step逆行tag隔離、gzip再起動・破損、LOD・TagStatsは
  既存テストをcharacterizationとして維持する。

```powershell
mvn -B "-Dtest=MetricsIngestorIntegrationTest" test
mvn -B "-Dtest=MetricsIngestorIntegrationTest,MetricsCacheDatabaseIntegrationTest,IngestSchedulerTest,MetricsLodIntegrationTest" test
mvn -B test
git diff --check
```

- 新規テスト追加後は全83件成功を期待する。
- `rg`で`ingestBlock`系の`if (gzip)`、9引数`writeScalar`、
  `DatabaseWriteRuntimeException`、tag状態取得の`computeIfAbsent`が消え、
  raw/gzip差がreader adapter内だけにあることを確認する。

## 実装結果

- gzip同一session継続のcharacterizationは既存実装でGREENとなり、人工的なREDは作らなかった。
  初回で圧縮offsetがsource sizeへ到達した後も、同じgeneration・offsetのまま
  展開済み行を2回目のblockへ引き継ぐ現行挙動を固定した。
- `SourceReader`、`RawFileReader`、`GzipSessionReader`を追加し、
  `ingestBlock`からraw/gzipの直接分岐と混在offset cursorを削除した。
- `BlockWriteSession`へ書込文脈を集約し、`writeScalar`を3引数へ縮小した。
  tag状態は`Map.get`と明示的なload/putへ変更し、checked `SQLException`を直接伝播する。
- `loadTagState`の`SQLException`は他のblock内SQLExceptionと同じ外側catchへ到達し、
  raw Runを`source_read_error`へ遷移させる。旧unchecked wrapperによる偶発的な再試行は
  互換対象外とし、SQLException誤分類の一括修正は既知バグ#1へ残した。
- 検証結果:
  - clean build後の`MetricsIngestorIntegrationTest`: 14件成功。
  - 対象4クラス: 30件成功。
  - `mvn -B test`: 83件成功、失敗・error・skipなし。
  - `git diff --check`: 成功。
  - 静的検査: `ingestBlock`系の`if (gzip)`、`DatabaseWriteRuntimeException`、
    tag状態取得の`computeIfAbsent`、9引数`writeScalar`は0件。

## 前提

- 無関係なdirty worktreeを保持する。
- commit・pushは行わない。

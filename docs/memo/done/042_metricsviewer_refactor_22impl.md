# 042: MetricsViewer 構造リファクタリング T3〜T5 実装メモ

- 対象 PRD: `docs/memo/042_metricsviewer_refactor_10prd.md`
- 対象タスク: T3〜T5
- 状態: 実装・検証完了（2026-08-01）

## 概要

キャッシュ判定、`source_meta`、query planning を T3→T4→T5 の順に一元化し、既存の HTTP JSON、DB schema、通常時挙動を維持する。本書を T3〜T5 の実装正本とする。

## 主な変更

- T3 では `matchesSource` / `describeSourceMismatch` を単一の `checkSourceMismatch` へ、`isValidDatabase` / `diagnoseInvalidDatabase` を単一の `checkDatabaseInvalid` へ統合する。source 検査順は kind、generation、state、数値 metadata、切詰め、gzip session、fingerprint、state 固有の size / mtime とする。fingerprint の `IOException` は従来どおり呼び出し元へ伝播する。例外として、raw `jsonl`・非 `ERROR` state の `source_mtime` 欠落または非数値は、外部改変・破損を安全側へ倒して `source_metadata_invalid` で再構築する。
- T4 では `MetricsCacheDatabase` 内の public nested `SourceMeta` へ、キー定数、単一 SELECT、数値変換、PreparedStatement による初期化、進捗更新、error 更新を集約する。Database 側の不正数値 fallback `-1` と Repository 側の表示用 fallback `0` を維持する。
- `CacheMetadata` は `IngestState` と state 解析エラーを明示的に保持し、未知 state 時も runs 側の warn 付き fallback と metrics 側の既存 run issue + `query_error` を維持する。`RunQueryContext` は生 Map でなく `CacheMetadata` を保持する。
- T5 では package-private の `MetricsQueryPlanner` を新設する。Repository は request 内で Run ごとに 1 本の Connection を保持し、metadata、tag、count、ordinal 境界を Connection を含まない入力へ変換する。Planner は availability、issue、cap、water-filling、422 判定と系列最低点数`MIN_POINTS_PER_SERIES = 50`を所有する。
- `SeriesPlan` には generation、tagId、ordinal 範囲、cap / pointBudget が有効になる条件をコメントで明記する。計画と Projection は Repository が保持する同一 snapshot を参照する。

## Interface と互換性

- HTTP API、JSON 形状、SQLite schema・キー・値、接続寿命、LOD / LRU、配分結果を変更しない。
- T3 はキャッシュ破棄判定を維持し、reason だけ単一検査列で最初に検出した具体的理由へ統一する。
- `SourceMeta` と `MetricsQueryPlanner` は Java 内部境界とし、外部 API や設定は追加しない。
- T6、付録 A、既知バグ 9 件は対象外とする。

## テスト

- Public interface / surface: `MetricsCacheDatabase.prepare`、取り込み経路、`/api/runs.json`、`/api/metrics.json`、package-private module `MetricsQueryPlanner`。
- T3 / T4 は純粋な構造変更のため人工的な RED は作らず、characterization を補強して GREEN を維持する。
- T3 は deep validator failure、source identity、invalid schema / state / generation、再構築 generation を確認する。
- T4 は不正 `committed_offset` の `source_metadata_invalid`、不正 `source_size` の表示用 fallback、未知 state、取り込み state、error metadata を確認する。
- T5 は次を 1 ケースずつ RED→GREEN する。
  1. 2 系列・総予算 101 の均等割端数 `51 / 50`。
  2. cap `55 / 100`・総予算 140 の再配分 `55 / 85`。
  3. cap `20 / 80`・総予算 100 の配分 `20 / 80`。
  4. 全系列 `not_found / empty` の予算 0。
  5. 最低必要量 150・上限 149 の 422 と既存 3 フィールド。

## 検証

```powershell
mvn -B "-Dtest=MetricsCacheDatabaseIntegrationTest" test
mvn -B "-Dtest=MetricsCacheDatabaseIntegrationTest,MetricsIngestorIntegrationTest,MetricsApiIntegrationTest" test
mvn -B "-Dtest=MetricsQueryPlannerTest,MetricsApiIntegrationTest" test
mvn -B test
git diff --check
```

`rg` で双子メソッド、`source_meta` キーリテラル、`readMeta` / `parseLong` 重複、Repository 内の予算配分・422 判定、Planner 内の SQL 依存が消えたことを確認する。

## 実装結果

- T3: source 判定を `checkSourceMismatch`、database 判定を `checkDatabaseInvalid` に統合した。deep validation failure は `deep_validation_failed`、不正な数値 metadata は `source_metadata_invalid` として再構築し、fingerprint の `IOException` は判定結果へ変換せず伝播する。
- T4: public nested `SourceMeta` に全キー、単一 SELECT、数値 fallback、PreparedStatement による初期化・進捗更新・error 更新を集約した。`CacheMetadata` は typed state と解析エラーを保持し、未知 state の runs / metrics fallback を維持した。
- T5: SQL 非依存の package-private `MetricsQueryPlanner` を追加した。Repository は request 内で Run ごとの同一 snapshot を計画から Projection 完了まで保持し、その Connection を含まない `SeriesInput` を Planner へ渡す。Planner は availability、issue、cap、water-filling、422、系列最低50点を決定する。
- `MetricsQueryPlannerTest` は指定 5 ケースを 1 件ずつ RED→GREEN し、snapshot 競合 regression と characterization を合わせて対象 36 テストが成功した。
- レビュー指摘対応: 計画用 Connection を閉じて Projection 用に再 open する二段化を撤回した。計画と Projection の間へ全再構築が割り込めないこと、および応答 generation が計画時 snapshot と一致することを `MetricsRepositorySnapshotIntegrationTest` で固定した。
- レビュー指摘対応: raw `jsonl`・非 `ERROR` state の不正な `source_mtime` を再構築する真偽変更を許容例外として PRD に明記し、欠落・非数値の両方を `MetricsCacheDatabaseIntegrationTest` で固定した。

## 検証結果

- `mvn -B "-Dtest=MetricsCacheDatabaseIntegrationTest,MetricsRepositorySnapshotIntegrationTest,MetricsQueryPlannerTest,MetricsApiIntegrationTest" test`: 25 tests、failure / error / skipped はすべて 0。
- `mvn -B test`: 82 tests、failure / error / skipped はすべて 0。
- R1〜R6レビュー修正の対象テスト: 30 tests、failure / error / skipped はすべて0。
- R1〜R6レビュー修正後の`mvn -B test`: 82 tests、failure / error / skipped はすべて0。
- `git diff --check`: 問題なし。
- 静的検査: T3 の双子メソッド、service 側の `source_meta` キーリテラル、`readMeta` / `parseLong` / `upsertMeta` 重複、Repository 内の配分・422 判定、Planner 内の SQL 依存はいずれも 0 件。`SourceMeta` の metadata SELECT は 1 個。

## 前提

- `10prd`はレビューで許容したT3の例外だけを追記する。後続レビューの最終実装境界を`20impl`、`21impl`、本書へ反映し、`CONTEXT.md`とADR0015は変更しない。
- 無関係な dirty worktree を保持する。
- commit / push は行わない。

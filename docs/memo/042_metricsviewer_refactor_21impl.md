# 042: MetricsViewer 構造リファクタリング T1 実装メモ

- 対象 PRD: `docs/memo/042_metricsviewer_refactor_10prd.md`
- 対象タスク: T1 `LOD 幾何定数の一元化`
- 状態: 実装・検証済み（2026-08-01）

## 概要

PRD 042 の T1「LOD 幾何定数の一元化」だけを実装する。LOD factor、LOD バケットから描画する候補点数、系列ごとの最低保証点数、LOD 幅計算、DB 行から `LodBucket` への変換を一元化し、外部挙動は変更しない。

## 主な変更

- `LodBucket` に package-private の `LOD_FACTOR = 16`、`POINTS_PER_LOD_BUCKET = 3`、`MIN_POINTS_PER_SERIES = 50` と、唯一の `widthForLevel(int)` を配置する。新しい入力検証は追加せず、既存の計算と overflow 挙動を維持する。
- `LodBucket.fromLodRow(ResultSet, int)` を追加し、`bucket`、`cnt`、`step_first` などを列名で取得する。`LodIngestWriter` の連続した列番号と位置引数による変換をこの factory へ置き換える。
- `LodIngestWriter` と `LodPageCache` の重複した幅計算を削除し、復元、16 子集約、ページ検索を共通定義へ接続する。例外文言は `LOD parent requires exactly 16 children` のまま維持する。
- `MetricsRangeProjector` の level 選択と min/max/last 候補生成を共通定義へ接続する。3 候補は一括生成し、生成件数が `POINTS_PER_LOD_BUCKET` と一致する不変条件を置く。`MetricsRepository` の最低 50 点配分は `MIN_POINTS_PER_SERIES` へ置き換える。
- SQL、SQLite schema、DB 値、HTTP JSON、LOD level、bucket 幅、点数配分、ログ条件は変更しない。

## Interface と互換性

- public API の追加・変更は行わない。追加する定数、factory、幅計算は service package 内だけで使用する。
- LOD factor は ADR0015 どおり 16 固定とし、level 0 / 1 / 2 の幅 1 / 16 / 256 を維持する。
- T2 の enum 実装には触れず、T3〜T6、付録 A、既知バグは対象外とする。

## テスト

- Public interface / surface: `/api/metrics.json` と既存の Metrics キャッシュ利用経路。
- 優先 behavior: factor 16 の level 選択、境界 bucket、min/max/last、上位 level 合成、系列最低 50 点配分を既存統合テストで確認する。
- TDD 順序: 挙動追加ではない純粋なリファクタリングのため人工的な RED は作らない。既存テストを characterization baseline とし、幅計算、DB 行 factory、候補数と最低点数の各変更単位で GREEN を維持する。内部 helper を直接固定する新規単体テストは追加しない。

## 検証

```powershell
mvn -B "-Dtest=MetricsLodIntegrationTest,LodPageCacheTest,MetricsIngestorIntegrationTest,MetricsApiIntegrationTest" test
mvn -B test
git diff --check
```

対象 production コードを `rg` で検査し、`widthForLevel` の実装が 1 個だけであることと、指定箇所の裸の `16`、`3`、`50` が消えていることを確認する。

実施結果:

- 初回T1対象テスト: 25 tests、Failures 0、Errors 0、Skipped 0
- レビュー指摘対応の対象テスト: 33 tests、Failures 0、Errors 0、Skipped 0
- レビュー指摘対応後の全テスト: 71 tests、Failures 0、Errors 0、Skipped 0
- production 37 files、test 19 filesの全再コンパイル成功
- `widthForLevel` の実装は `LodBucket` の 1 個だけ
- 指定箇所の裸の `16`、`3`、`50` は検出なし
- `git diff --check`: エラーなし

## 前提

- 無関係な dirty worktree を保持する。
- `CONTEXT.md`とADR0015は変更しない。`042_metricsviewer_refactor_20impl.md`は、レビューで裁定した未知DB stateの契約明文化に限って更新する。
- コミット、push は明示依頼がない限り行わない。

# Metrics ViewerのLOD集約数の記述整合 暫定PRD

> 2026-11-03 番号を 999 から 934 へ振り直した（PRD114）。

> 2026-11-01: [PRD110 pin対応の実装記録](done/110_runner_main_pane_dynamic_centre_22impl.md)の独立レビューで見つかった範囲外の指摘を、ユーザーの「課題として記録する」の回答により別件として記録した。今回は記録のみで、文書の修正には着手していない。

## 問題

[設計160の日本語版](../design/160_applications_and_tools.jp.md)で、Metrics ViewerのLOD集約数の説明が一致していない。

- §2.1の`MetricsIngestor` / `LodIngestWriter`の行は「factor 4 LOD」と記載している（記録時の47行）。
- §7.1のMetrics Viewerの段落は「LODは完成した16子だけから作り」と記載している（記録時の389行）。

2026-11-01に読み取りで確認した現行実装は4子を集約する。

- [LodBucket.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodBucket.java)の`LOD_FACTOR`は`4`。
- [LodIngestWriter.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodIngestWriter.java)の`appendChild`は子が`LOD_FACTOR`件そろうと親を生成し、`combine`も子の数が同じ値であることを検証している。

この食い違いはpin対応の変更前から存在する。PRD110のpane操作とは切り離して扱う。

## ゴールと範囲

- LODの集約数の説明を現行実装と整合させ、同じ設計文書内で数値が食い違わないようにする。
- 対応時には設計160の日英と関連する説明を確認し、同じ旧記述が残る箇所を特定する。
- 本件は説明の整合を対象とし、LODの集約アルゴリズムや保存形式の変更は要求しない。

## 対応時の確認と受入基準

1. 対応時点の`LOD_FACTOR`と親を生成する条件を読み直し、文書に書く値の根拠を示す。
2. 設計160の関連箇所と日英の説明が、確認した集約数・生成条件に一致する。
3. 旧記述の検索結果と修正範囲を記録する。PRD110のpin実装には変更を加えない。

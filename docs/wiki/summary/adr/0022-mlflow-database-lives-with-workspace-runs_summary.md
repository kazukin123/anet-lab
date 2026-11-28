---
title: MLflow DB は選択 workspace の runs 配下に置く
page_kind: サマリ
material: "[[docs/adr/0022-mlflow-database-lives-with-workspace-runs.md|0022-mlflow-database-lives-with-workspace-runs]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
updated: 2026-11-28
---
# MLflow DB は選択 workspace の runs 配下に置く

## 元の材料

[ADR 0022](../../../adr/0022-mlflow-database-lives-with-workspace-runs.md) blob:956c1ce5f654e613ff5ed0331baa7768c1bd595b

## 要点

- MLflow bridge と server は同じ SQLite DB を読み書きする。
- 選択した workspace の runs/mlflow.db に DB を置く。
- 監視対象 Run と DB は workspace 単位で一緒に移せる。
- Run の走査は run_* 内の metrics.jsonl に限り、この DB を Run と数えない。
- runs/ 直下に補助 DB を置く例外は MLflow に限る。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)

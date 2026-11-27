---
title: Run 分類は物理 workspace フォルダで行い、workspace を自己完結の箱とする
page_kind: サマリ
material: "[[docs/adr/0021-run-classification-by-workspace-folder.md|0021-run-classification-by-workspace-folder]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-26
---
# Run 分類は物理 workspace フォルダで行い、workspace を自己完結の箱とする

## 元の材料

[ADR 0021](../../../adr/0021-run-classification-by-workspace-folder.md) blob:82b725fd4f1029d318710743d8e4745ce60044f9

## 要点

- 実験系列は物理 workspace で分類し、設定、Run、Optuna の履歴と成果物を一緒に持つ。
- フォルダの退避・復元だけで入力と結果の対応を維持できる。
- 分類タグの外部 DB では、移動や削除と DB の同期が別途必要になる。
- 明示した設定での Runner 起動は自己完結の扱いとし、自動 workspace 解決を介さない。
- Viewer は1プロセスで1 workspace を開く。別 workspace の同時閲覧には別ポートを使う。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)

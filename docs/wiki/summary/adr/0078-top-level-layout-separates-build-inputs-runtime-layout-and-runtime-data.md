---
title: トップの階層は、ビルドの入力（src）・実行時の配置（apps）・実行時にできるもの（data）で分け、ビルドは apps へ直接出す
page_kind: サマリ
material: "[[docs/adr/0078-top-level-layout-separates-build-inputs-runtime-layout-and-runtime-data.md|0078-top-level-layout-separates-build-inputs-runtime-layout-and-runtime-data]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
updated: 2026-11-26
---
# トップの階層は、ビルドの入力（src）・実行時の配置（apps）・実行時にできるもの（data）で分け、ビルドは apps へ直接出す

## 元の材料

[ADR 0078](../../../adr/0078-top-level-layout-separates-build-inputs-runtime-layout-and-runtime-data.md) blob:6ccd4a6e3871be156e5c6eb38f75876690489017

## 要点

- ビルド入力を src、実行時の配置を apps、Run や利用者の状態を data に分ける。
- 配布しない中間生成物などは out に置く。
- 日常ビルドでは実行物を apps へ直接出してすぐ起動できるようにする。
- 設定は実行時の場所で編集し、ビルドによる再コピーを必要としない。
- 開発と配布の配置と起動 bat を共通にし、梱包時に同梱するものを選ぶ。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)

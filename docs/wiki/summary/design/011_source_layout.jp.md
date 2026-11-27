---
title: ソース配置とライブラリの境界
page_kind: サマリ
material: "[[docs/design/011_source_layout.jp.md|011_source_layout.jp]]"
material_kind: 設計文書
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
updated: 2026-11-26
---
# ソース配置とライブラリの境界

## 元の材料

[ソース配置とライブラリの境界](../../../design/011_source_layout.jp.md) blob:1407913cee3d27e3e307a2627ea53495dd1d7701

## 要点

- `src/core/base/` が Agent・Env・NN・ReplayBuffer・Observer の共通契約と実装を持つ。
- Agent の各系統は `src/algo/`、環境は `src/env/` に置き、基盤の公開ヘッダを使う。Runner が各ライブラリの公開初期化関数を呼ぶ。
- 単体テストは対象ライブラリに置く。他ライブラリへは公開ヘッダとテスト用の口からアクセスする。
- 組み合わせ自体を確かめるテストは `src/apps/` に専用アプリを置く。
- 中間ファイルは `out/build/`、単体テストと共通 DLL は `out/test/`、Runner と Viewer の配布物はそれぞれ `apps/` 配下に出す。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤](../../concepts/framework/311_dev_tools.md)
- [強化学習の共通基盤](../../concepts/framework/318_rl_core.md)
- [Runner](../../concepts/apps/110_runner_app.md)
- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)

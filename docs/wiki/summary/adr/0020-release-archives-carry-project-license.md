---
title: リリース zip 直下にプロジェクトのライセンス文書を置く
page_kind: サマリ
material: "[[docs/adr/0020-release-archives-carry-project-license.md|0020-release-archives-carry-project-license]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
updated: 2026-11-26
---
# リリース zip 直下にプロジェクトのライセンス文書を置く

## 元の材料

[ADR 0020](../../../adr/0020-release-archives-carry-project-license.md) blob:b3bc54104b0bf31fce44e95e046b4171a91168f5

## 要点

- 実行用 zip と Doxygen zip を個別に配布できるようにする。
- プロジェクトの LICENSE と NOTICE を各 zip の直下に置く。
- 実行物はリポジトリ配置の写しとし、licenses/ は第三者のライセンスに使う。
- Viewer の jar にも META-INF 内のライセンス、NOTICE、第三者情報を同梱する。
- リリースの生成後に必須の収録ファイルを検査する。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)

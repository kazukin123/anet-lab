---
title: 解析用メタデータは Runner が出力し、解析側は設定から再導出しない
page_kind: サマリ
material: "[[docs/adr/0029-analysis-metadata-emitted-by-runner.md|0029-analysis-metadata-emitted-by-runner]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
updated: 2026-11-28
---
# 解析用メタデータは Runner が出力し、解析側は設定から再導出しない

## 元の材料

[ADR 0029](../../../adr/0029-analysis-metadata-emitted-by-runner.md) blob:2260e330814c288c2ff0ad0a63090ef53bb6b049

## 要点

- 評価の step 値は同じ設定名でも、どの Runner のカウンターを使ったかで座標系が異なる。
- 一つの定数を掛けても異なる評価系列を正しい横軸へ復元できない。
- Runner が解決したメトリクス定義を metrics.defs としてメトリクス原本に出す。
- 解析ツールは設定解決の C++ 規則を再実装せず、この定義を読む。
- 独立の定義ファイルを必要とせず、metrics の記録だけを受け取った場合も解析できる。

## 内容

### 同名の軸が違う場合

評価の通常イベントとエピソード末イベントが別のカウンターを運ぶと、両方が exp_step と書かれていても比較可能とは限らない。設定ファイルには実行時の対応を確定する情報が足りなかった。

### 解析側への伝達

定義を専用ファイルに分けず、メトリクス原本とともに読み出せるようにする。古い Run は定義を持たないので、新しい記録と同じ精度で座標系を知れるとは限らない。

## 主要な概念

- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)

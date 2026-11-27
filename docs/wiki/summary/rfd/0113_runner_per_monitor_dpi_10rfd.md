---
title: Runner のモニター別 DPI とログの字下げ
page_kind: サマリ
material: "[[docs/memo/0113_runner_per_monitor_dpi_10rfd.md|0113_runner_per_monitor_dpi_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/319_gui.md|319 GUI 共通基盤]]"
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-26
---
# Runner のモニター別 DPI とログの字下げ

## 元の材料

[RFD0113　Runner のモニター別 DPI とログの字下げ](../../../memo/0113_runner_per_monitor_dpi_10rfd.md) blob:9841782d1da93c4025d2781aea2f59f5aa843019

## 要点

- GDI スケーリングが Per-Monitor V2 の指定を上書きし、拡大環境での表示と字下げの試験に影響していた。
- 原本の暫定案は Runner と GUI テストの GDI スケーリング宣言を外し、窓を置いたモニターの DPI に描画を合わせる。
- 寸法を DIP で記すこと、利用時点で窓の実画素へ換算することは、この案を採る場合の実装方針である。
- ログ本文の字下げは表示された文字位置から求め、測れない間の扱いも決める。
- 150% と175% のモニター間で窓を移動し、見た目の寸法と文字位置が追従するか確認する。

## 内容

### 調査時の問題

GDI スケーリングの宣言が Per-Monitor V2 の指定より優先され、窓が受け取る DPI と実際の拡大率が合わなかった。描画のぼやけと Logs の字下げを調べる契機になった。

### 未合意の描画方法

原本は以下を再グリルで決める暫定の既定とした。GDI の宣言を外し、固定寸法は100%時の大きさで記して使用時に FromDIP で換算する。窓やフォントは wx の DPI 変更に任せ、アプリが画素で覚えた値だけを持ち主が計算し直す。

### 字下げと受入の案

ログ本文の開始位置を RichEdit 自身の表示位置から測る。まだ一行もないときなどは GDI の値を仮に使う。案を採る場合は既定モニターの GUI テストと、100%・150%・175%の環境やモニター間移動の目視を組み合わせ、学習速度への影響を調べる。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)

---
title: リリース ZIP を CMake で組み立てる
page_kind: サマリ
material: "[[docs/memo/0185_release_packaging_cmake_10rfd.md|0185_release_packaging_cmake_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
updated: 2026-11-28
---
# リリース ZIP を CMake で組み立てる

## 元の材料

[RFD0185　リリース ZIP を CMake で組み立てる](../../../memo/0185_release_packaging_cmake_10rfd.md) blob:b8c555ce9cc5a85b925a4a8e05e2bd471b4c2337

## 要点

- 実行用 ZIP と Doxygen ZIP の内容・検査・圧縮が GitHub Actions 内の PowerShell にあり、手元では同じ手順を使えなかった。
- CMake の install と CPack に配布内容を記述し、ワークフロー preset でまとめる案を置いた。再グリル前の既定である。
- 開発機と CI の両方で同じ組み立て経路を使えるようにする。
- 実行体、設定、依存 DLL や文書の配置を既存の配布契約と照合する。
- できた ZIP の内容と展開後の利用経路を検査し、CI の配布操作と組み立てそのものを分ける。

## 内容

### 起草のきっかけ

配布物の配置・検査・圧縮を GitHub Actions 内の処理で行っていたため、同じ ZIP を手元だけでは作りにくかった。開発用の出力を汚さず、開発機と CI で同じ中身を検証することを目指した。

### 再グリル前の暫定案

案ではリリース用スイッチを有効にしたときだけ版番号、JDK、Doxygen 等を要求する。ビルド木に生成した実行物と文書を CMake の install ルールで配置し、CPack で実行用、Doxygen、デバッグシンボルの ZIP に分ける。CI での配信とパッケージの組み立ては分ける。

### 配布内容の検査案

実行体の隣に置く DLL を開発時のコピーと配布時の配置で共通にし、小さな ZIP だけでも付属物や不足を検査できるようにする。三つの ZIP、チェックサム、preset やディレクトリ構成は原本の暫定案で、個別方式は決定前である。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)

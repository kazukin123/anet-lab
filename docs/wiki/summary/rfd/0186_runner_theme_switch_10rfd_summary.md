---
title: 起動中の Runner でライト・ダークテーマを切り替える
page_kind: サマリ
material: "[[docs/memo/0186_runner_theme_switch_10rfd.md|0186_runner_theme_switch_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/319_gui.md|319 GUI 共通基盤]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
updated: 2026-11-28
---
# 起動中の Runner でライト・ダークテーマを切り替える

## 元の材料

[RFD0186　起動中の Runner でライト・ダークテーマを切り替える](../../../memo/0186_runner_theme_switch_10rfd.md) blob:86c9fc179d26a0ebc5b378f5eb7aedff6fb65fd0

## 要点

- Windows のアプリモードを切り替えると、一部だけが新テーマへ移り、古い色と混在した。
- 再起動を避けたい長い Run を想定し、配色を一か所で更新する暫定案を置いた。実装方法は合意前である。
- Trace やログを含む表示領域ごとに、再描画すべき色と部品の状態を整理する。
- 検証では同じテーマで最初から起動した画面と比べ、テーマ変更後の色がそろうことを確認する。
- 表示の変更中も Run と閲覧対象を保つため、再起動に依存しない更新方法を扱う。

## 内容

### 観測した症状

起動後に Windows のライト／ダークのアプリモードを切り替えると、Runner の一部は新しい外観になる一方、アプリが保持していた色が残る領域があった。参照する画面は同じテーマを選んで最初から起動した Runner である。

### 再グリル前の再描画案

切り替えの通知中は色を確定せず、複数の通知が落ち着いてから、その時点のシステム色で一度だけ塗り直す案を置いた。QValue の表や行のない Evaluation Trace の領域はアプリの対象に、標準コントロールは wx 側の処理に任せる案である。

### 採用を決める前の確かめ方

通知と再描画の時点に読み取れる色を記録し、遅延の仕方が有効かを確かめる。スクロール位置、列幅、表示対象などを保てるかを、両方向のテーマ切り替えで比較する。通知の時期や wx に残る不具合への対処は検証結果を待つ。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)

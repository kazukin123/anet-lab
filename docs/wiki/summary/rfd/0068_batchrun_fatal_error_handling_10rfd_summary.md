---
title: batchrun の致命エラー処理（ダイアログ抑止・終了コード・bat 検出）
page_kind: サマリ
material: "[[docs/memo/done/0068_batchrun_fatal_error_handling_10rfd.md|0068_batchrun_fatal_error_handling_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
updated: 2026-11-28
---
# batchrun の致命エラー処理（ダイアログ抑止・終了コード・bat 検出）

## 元の材料

[batchrun の致命エラー処理（ダイアログ抑止・終了コード・bat 検出）](../../../memo/done/0068_batchrun_fatal_error_handling_10rfd.md) blob:000a8e1127a715d4c87122dc94d32a93c7c3174f

## 要点

- batchrun の致命エラーがダイアログ待ちになり、後続の実験が進まなくなる事例を扱う。
- app.show_error_dialog の既定は online で有効、batchrun で無効にする。
- 致命エラーは記録と flush を行い、ダイアログの表示有無によらず非ゼロ終了で呼び出し元から検出できる。
- チェックポイント保存など回復可能な UI 操作の失敗は Run を継続し、終了コードに含めない。
- bat から失敗を識別できることと、通常実行時の報告経路を確認する。

## 内容

### バッチがダイアログの前で止まった事例

Atari の連続実行で Phoenix がエラーになってから Qbert の開始まで13分48秒かかった。通常の切り替えは3〜5秒で、残りはエラーのダイアログを人が閉じるまでの待ち時間だった。

### モードごとの既定とエラー通知

`app.online.show_error_dialog=true`、`app.batchrun.show_error_dialog=false` を設定の選択チェーンに置き、アプリは解決後の単一キーだけを読む。致命エラーではログを書き出して非ゼロで終了し、バッチは終了コードで失敗を識別する。worker での例外も main thread の同じ通知経路へ渡す。About やファイル選択、workspace 選択などの操作用ダイアログはこのキーの対象外。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)

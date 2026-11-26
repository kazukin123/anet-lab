# RFD0071: 捕まらない fatal の可視化（worker catch・set_terminate・エラーログ write-through）

## 元の材料

[RFD0071: 捕まらない fatal の可視化（worker catch・set_terminate・エラーログ write-through）](../../../memo/0071_uncaught_fatal_visibility_10rfd.md) blob:b7a168cdccda9ff599778c63aca9bdb06ff3157c

## 要点

- CUDA 非搭載機で Runner が無言終了し、通常のダイアログにも Run ログにも原因が残らなかった事例を扱う。
- worker が実行するタスクを例外境界で囲み、失敗時も待ちタスク数を更新する。
- worker の例外は main thread の既存の fatal 報告に渡す。
- 通常の例外経路に乗らない終了にも terminate handler とエラーログの即時書き込みを用意する。
- 観測手段の整備を目的とし、非搭載機固有の根本原因をこの段階で断定しない。

## 内容

### どの失敗経路を記録するか

RunnerThread の失敗には致命エラーの通知があったが、PinnedThreadPool でタスクが例外を投げたときはログや終了コードが残らなかった。非搭載機に依存する事例で原因が一つに絞れていないため、通常の worker 例外と `std::terminate` を通る異常終了の両方で痕跡を残す。

### タスク境界・終了処理・ログ

タスクの呼び出しを try/catch で囲み、例外時も未処理件数を減らし、main thread の既存の fatal 経路に渡す。worker だけを直すと noexcept 内の例外を見落とし、terminate handler だけでは worker の失敗を通常の fatal 表示や終了コードへ渡せない。エラーログは書き込み時点でフラッシュし、終了直前のメッセージまでファイルに残す。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [312 ユーティリティ](../../concepts/framework/312_util.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)

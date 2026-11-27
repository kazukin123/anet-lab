---
title: AttachScoped が返す Observer の解除ハンドル
page_kind: サマリ
material: "[[docs/memo/0146_notifier_scoped_detach_10rfd.md|0146_notifier_scoped_detach_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/319_gui.md|319 GUI 共通基盤]]"
updated: 2026-11-26
---
# AttachScoped が返す Observer の解除ハンドル

## 元の材料

[RFD0146　AttachScoped が返す Observer の解除ハンドル](../../../memo/0146_notifier_scoped_detach_10rfd.md) blob:1b99a4ea1451d942d924ee6d27c46f7fc9493813

## 要点

- スコープ付き登録では内部に wrapper が登録され、呼び出し元の Observer を Detach へ渡しても同じ対象として見つからなかった。
- 登録時に返したハンドルをそのまま Detach に使えるよう、実際に登録した wrapper を返す。
- 通常の Attach と同じ使い方にそろえ、呼び出し側が登録方法によって解除を変える必要をなくす。
- Runner の停止後に解除するという寿命の前提を明示し、現在の呼び出し経路を一緒に修正する。
- 終了時にしか解除しないため表面化しにくかったことと、API の約束自体の不一致を分けて扱う。

## 内容

### Context / Problem Statement

`Notifier::Attach()` および `Attach<T>()` は「戻り値＝Detach へ渡すもの」という不変条件を満たしている。`AttachScoped` だけがこれを破っており、本件はその不変条件の回復である。
現在は一致するものが無いため `erase(end(), end())` となり、要素の移動も破棄も起きない＝実質読み取りだけ。一方 `Notifier` は mutex を持たず、`Notify` は `for (auto obs : train_observers_)` でメンバ vector を直接走査する。

### 要件

`Notify` は Trainer スレッドから observer 列を走査する。`Notifier` は同期機構を持たない。
本修正以前は `Detach` が実質 no-op だったため違反しても顕在化しなかったが、修正後は実際に列を書き換える。
新タグ `[notifier][scoped]`。`TestRunner` を 2 体用意し、`FunctionTrainObserver` の呼び出し回数で検証する。

### 設計判断(why。self-contained のため明記)

多重継承 `T` は `static_assert` で弾く。wrapper を 1 つ返す契約と、複数 wrapper を attach する現行実装は両立しない。現行 0 件なので実害なくコンパイル時に固定できる。「起こり得ない経路のエラー処理を足さない」と fail-fast を両立させる最小手段。
RunnerFrame の回避実装は同一変更内で撤去する。回避が不要になったのは本修正の結果であり、残すと解消済みの問題を指すコメントが残留する(クリーンブレーク方針「移行漏れとして扱う」)。

### 影響・移行

挙動変化: `Detach` が実際に observer 列を書き換えるようになる。現行の全 detach 経路は Runner 停止後のため安全だが、前提が暗黙から明示に変わる(R3)。
template 版の戻り型が `shared_ptr<T>` から wrapper 型に変わるため、inner の concrete 型を必要とする新規利用は `Attach<T>` を使うことになる。現行に該当利用はない。

### Testing Decisions

回帰: `anet-core-test` 全体。既知の失敗以外が増えないこと。

### 受け入れ基準

AC1: `AttachScoped()` の戻り値を `Detach()` へ渡すと、以降その observer へ通知が届かない(Train 系・Learn 系の両方でテスト済み)。
AC2: 4 パネルと trainer.cpp に差分が生じない(呼び出し側の宣言・Detach 行が無変更のままビルド・動作する)。
AC4: `RunnerFrame::AttachTrainStatusObserver` が `AttachScoped` 経由になり、本件を指す回避コメントが残っていない。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)

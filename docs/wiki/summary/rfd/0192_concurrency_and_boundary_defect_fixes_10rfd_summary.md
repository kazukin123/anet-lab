---
title: 通知・画像・強制行動・登録・Q 値表示の境界修正
page_kind: サマリ
material: "[[docs/memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md|0192_concurrency_and_boundary_defect_fixes_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/319_gui.md|319 GUI 共通基盤]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
updated: 2026-11-28
---
# 通知・画像・強制行動・登録・Q 値表示の境界修正

## 元の材料

[RFD0192　通知・画像・強制行動・登録・Q 値表示の境界修正](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md) blob:11d04b6d9e37a7a830a03927fc7f9ce70c1469c4

## 要点

- 並行して行われる Observer の登録・解除と通知で、登録列の読み書きが競合しないようにする。
- 動画に渡す画像の大きさを検証し、不整合な入力で範囲外を読まないようにする。
- 強制行動に使う Tensor の device と、行動を処理する側の device の整合を取る。
- 重複登録を黙って上書きする経路を点検し、異なる部品の登録が隠れないようにする。
- QValuePanel の平滑化が境界条件で誤って止まる問題も、他の四件と独立に検証する。

## 内容

### 問題

直近の Atari の Run（`run_20261119-202718_mz_tg_g99ne_rr4_s2`）では、Observer は TRAIN 19・LEARN 134・EPISODE_END 27・SESSION_END 54 個付いていた（ログの `Notifier:` の行を数えた）。
`Notify` は、行動の 1 step ごとと、学習のあった step ごとに呼ばれる。1 step は NN の forward と Env の step を含むので、足す量は 1 step の 0.01% 程度の見込みである。

### ゴールと非ゴール

`Notifier` の `Attach`・`Detach`・`Notify` を別々のスレッドから同時に呼んでも、Observer の列が壊れない。呼んでよいスレッドと時点が、`Notifier` と関係するクラスに書かれている。
`actor_approx` かつ GPU 推論の評価 Actor に強制行動を渡しても止まらず、ヒントは渡した行動の値になる。
4 つの Repository を 1 つの形にまとめ、登録の関数名をそろえる案 — 範囲外。課題 issue-20261109-adc5c4 が扱う。制約ではなく別の関心で、重複登録の扱いは 1 つずつ直せる。
`ComputeQValueHeatMap` を分割する案（課題 issue-20261111-90fa5c）と、HeatMap の描画の重複を直す案（課題 issue-20261103-e6dbb4） — 範囲外。制約ではなく別の関心である。

### 確定事項

`Notifier` の `Attach`・`Detach`・`Notify`・`Clear`・`Shutdown`・`WillBlockOnShutdown`・`LogObservers` は、どのスレッドから同時に呼んでも、Observer の列が壊れない。
`Detach` と `Clear` は、進行中の通知を待たずに戻る。戻った時点で始まっていた通知は、外した Observer を呼びうる。その通知が終わるまで、Observer は `Notifier` が生かしておく。`Detach` が戻ったあとに始まった通知は、外した Observer を呼ばない。
`Notifier` は、Observer を呼ぶあいだ、自分のロックを持たない。Observer の中から `Attach`・`Detach`・`Notify` を呼んでよい。Observer の中で呼んだ `Attach`・`Detach` の効き目は、次に始まる通知からである。`Shutdown`・`WillBlockOnShutdown`・`LogObservers` も、Observer を呼ぶあいだロックを持たない。
Observer を呼ぶ順は登録順のまま、Observer が動くスレッドは `Notify` を呼んだスレッドのまま変えない。

### 実装契約

例外の扱いは変えない。Observer が投げた例外は、今と同じく `Notify` の呼び出し元へ伝わり、その回の残りの Observer は呼ばれない。退けた案は、例外を捕まえて残りの Observer を呼び続ける形である。制約ではないが、失敗を握り潰すことになり、題材に要らない。
WARN は、既存の「Run ごとに対象につき 1 回」の口で出し、対象は tag にする。Run を作り直すと、また 1 回出る。
`InitRL` は 1 回だけにしない。backend の設定を呼ぶたびに当て直す関数で、中で呼ぶ `InitNN` が 1 回だけになれば足りる。退けた案は、`InitRL` も 1 回だけにする形である。制約ではないが、設定を変えて呼び直すテストが、2 回目から設定を当てられなくなる。
RFD0056: `Notifier` と `VideoLogger` は説明に出るだけである。順序の決まりは無い。

### 受入

テストの実行体は、`anet-core-base-test`（基盤）、`anet-algo-dqn-test`・`anet-algo-muzero-test`・`anet-algo-imagecls-test`（Agent の系統）、`anet-env-atari-test`・`anet-env-dropmerge-test`・`anet-env-gridmaze-test`・`anet-env-imagecls-test`・`anet-env-lunarlander-test`、`AnetRLRunner-gui-test`である。`AnetRLRunner-gui-test` は、ビルドの既定の対象に入らず、`ctest` にも登録が無いので、手でビルドして起動する。

### 成功の測り方

題材の 5 件の判定が「残件無し」になる。
実装のあと、学習中の Panel の開閉と、終了時に、`Notifier` の Observer の列を走査する箇所で落ちた dump が出ない。出たら、その経路を題材 80c355 の対応内容に足して、判定を「残件有り」に戻す。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)

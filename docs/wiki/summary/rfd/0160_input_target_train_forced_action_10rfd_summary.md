---
title: 操作入力を Eval と Train の間で切り替える案
page_kind: サマリ
material: "[[docs/memo/0160_input_target_train_forced_action_10rfd.md|0160_input_target_train_forced_action_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/319_gui.md|319 GUI 共通基盤]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
updated: 2026-11-28
---
# 操作入力を Eval と Train の間で切り替える案

## 元の材料

[RFD0160　操作入力を Eval と Train の間で切り替える案](../../../memo/0160_input_target_train_forced_action_10rfd.md) blob:e5bf23386d8309a0625c21432bbb30093ef7dc6e

## 要点

- Eval と Train の入力先を切り替える暫定案で、入力先や強制行動の経路を再グリルで裁定する。
- 既存の手動操作は評価用の個体に届き、学習データを集める lane の行動を人が指定できなかった。
- 実行中に入力先を Eval と Train の間で切り替えられるようにする。
- Train を選んだ場合は操作対象の lane で実際に指定行動を実行し、その結果の遷移を通常どおり Replay に入れる。
- Actor の提案と実行した行動を混同せず、Q のヒントなど行動依存の情報の整合を保つ。
- 人のプレイを学習へ入れる最初の範囲を扱い、専用のデモ学習アルゴリズムとは分けて検討する。

## 内容

### ゴール / 非ゴール

設定で固定する方式ではない。Eval 側の挙動は対 RFD のまま変えない。

### 提案

現在の入力先はステータスバーに常時表示する（どちらに向いているか分からない事故を避ける）。切替はログ 1 行（`Input target: train lane=0` の形）。
`eval` のときは対 RFD の経路そのまま。`train` のときは Eval 側へは何も流れない（Eval は Actor で回り続ける）。
Train は 128 lane を同時に進めるので、人が操作している 1 lane 以外の 127 lane も同じ速さに落ちる。人間操作中の学習は「遅い」を受け入れる。

### 候補を採る場合の受入条件

その step の経験が ReplayBuffer に入り、`UpdateFromBatch` に含まれる（強制 lane の遷移が sample されうる。Learner 側に特別扱いが無いこと）。
強制 lane の Actor Q ヒント・Munchausen 項・（142 後は）prev_action が実行行動と一致する。`WithAction` 相当の再 gather コードが Train 側に存在しない。
`train_min_step_interval_ms` を設定すると、入力先が `train` の間だけ step 間隔が下限以上になり、`eval` へ戻すと制限が消える。

### 関連

098 TempoRL / 観測パイプライン単位A: skip有効時は強制したlaneの継続状態を捨て、強制したstep自体を `decision=true` で記録する。Replay窓は判断間の行動一致を前提とし、不一致はSample時にfail-fastするため、trainへの入力注入はこの2条件を満たすActorを前提にする。
035（Actor Q ヒント）、067（log π 項）— 強制行動が Actor 出力になることで整合が自動になる対象。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)

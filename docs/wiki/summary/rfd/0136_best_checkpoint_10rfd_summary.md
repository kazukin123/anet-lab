---
title: 評価の自己ベストを記録する checkpoint 案
page_kind: サマリ
material: "[[docs/memo/0136_best_checkpoint_10rfd.md|0136_best_checkpoint_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/framework/313_profiling.md|313 プロファイリング]]"
updated: 2026-11-28
---
# 評価の自己ベストを記録する checkpoint 案

## 元の材料

[RFD0136　評価の自己ベストを記録する checkpoint 案](../../../memo/0136_best_checkpoint_10rfd.md) blob:7a8a8cc8c819006e41dd5fc33f7e3a3c43522675

## 要点

- 保存する重みと評価結果の対応を保てるかを検討する草案であり、保存方法・負荷・バックグラウンド評価との同期は未決である。
- 評価成績が最高だった時点の重みを保持したいが、単一エピソードの評価値には大きなばらつきがある。
- 複数回の評価値を比べて最大値を選ぶと、観測ノイズによる過大評価を伴う。
- 判定に使う評価の集計法や改善幅と、重みを書き出す頻度を分けて検討する。
- 候補には更新時にメモリへ複製して後で書く方式、すぐに保存する方式、定期保存との併用がある。
- バックグラウンド評価と Learner の同時更新に対する整合、モデル複製のメモリ量と I/O 負荷が未決である。

## 内容

### Context / Problem Statement

長期 Run では ピークを打ったあと 10〜20% 劣化することが 3 例で確認されている。
現状は残らない。 重みが書かれるのは Run 終了時（GUI の close ハンドラ）の 1 回だけで、定期 checkpoint の機構が存在しない。上記 3 Run のピーク時点の重みは既に失われている。

### 問題

IMPALA で 1 回 50.1MB。序盤は best が頻繁に更新されるため、無条件にディスクへ書くと I/O が支配的になりうる。加えて `WriteTorchObject` が `ostringstream` を経由するため、保存のたびに同サイズのホストメモリを確保する。

### 候補を採る場合の目標契約

best が一度も更新されない場合でも、現行の `agent_close.anet` の挙動は変わらない（非退行）

### 解決候補

現行と同じ 3 点セット（学習再開も視野に入れる場合）

### 未調査事項

原典 DQN の `NeuralQLearner.lua` における `best_network` の初期化位置（更新が `train_agent.lua` 側であることは確認済み）

## 主要な概念

- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [313 プロファイリング](../../concepts/framework/313_profiling.md)

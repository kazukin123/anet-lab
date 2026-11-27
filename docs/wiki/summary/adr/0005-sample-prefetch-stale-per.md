---
title: ReplayBuffer サンプリング + H2D を 1ステップ stale で先読みし GPU 学習と overlap する（AsyncLoader 方式）
page_kind: サマリ
material: "[[docs/adr/0005-sample-prefetch-stale-per.md|0005-sample-prefetch-stale-per]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
updated: 2026-11-26
---
# ReplayBuffer サンプリング + H2D を 1ステップ stale で先読みし GPU 学習と overlap する（AsyncLoader 方式）

## 元の材料

[ADR 0005](../../../adr/0005-sample-prefetch-stale-per.md) blob:332ed2b763b971f3893b37f153d9aeee2282b7a4

## 要点

- DQN の学習開始時に ReplayBuffer の抽選・収集と CPU から GPU への転送で約13 ms の空きが観測された。
- 専用スレッドが次の1バッチだけ先読みし、現在のバッチの GPU 更新と重ねる。
- PrefetchingReplayBuffer に処理順を集め、保管中データの整合性は ReplayBuffer のロックで保つ。
- 先読み開始後の Push も同じ FIFO に遅延投入する。
- 優先度は1ステップ前の値で抽選される。直列経路との厳密比較では先読みを無効にする。

## 内容

### 先読みの順序

抽選と経験の収集に約6.5 ms、GPU への転送に約6.5 ms かかっていた。専用の copy stream と event で転送完了を伝え、学習用 stream は使う直前に待つ。

### 比較と適用範囲

抽選だけを呼び出し側に残す案では抽選の費用を隠せない。先読み有効どうしは同じ seed で抽選列を再現できるが、直列との一致は求めない。複数バッチの先読みと MuZero は対象外とした。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)

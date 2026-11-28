---
title: Replay初期優先度ヒントを単一packed carrierとし、completionを専用coordinatorへ分離する
page_kind: サマリ
material: "[[docs/adr/0012-replay-initial-priority-hint-completion.md|0012-replay-initial-priority-hint-completion]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
updated: 2026-11-28
---
# Replay初期優先度ヒントを単一packed carrierとし、completionを専用coordinatorへ分離する

## 元の材料

[ADR 0012](../../../adr/0012-replay-initial-priority-hint-completion.md) blob:4b7ae73dd1f978542fc1cecbbbdbd3536b508983

## 要点

- replay 初期優先度のヒントは連続 float32 の単一バッチテンソルで運ぶ。
- 共通層は形と転送を扱い、各列の意味や有効性はアルゴリズムの推定器に任せる。
- InitialPriorityCompleter が完成待ち、フォールバック、初期優先度の種類を一元管理する。
- ReplayBuffer が保管資源を持ち、sampleable 範囲は ValidIndexManager が判定する。
- DQN の当初2列は Munchausen で3列、価値分解で5列に変わった。強制行動時も実行行動を決めてから同じ推論結果でヒントを作る。

## 内容

### 共通層と DQN の境界

名前付きテンソルを複数運ぶと、複数回の CPU 転送が必要になる。1本のテンソルなら共通層の検査を dtype・rank・batch に限り、列の検証を推定器に任せられる。

### 確定処理

completer が lane 別の待ち列と初期優先度の計算方法を管理する。列数が正しくない場合と非有限な値は区別し、確定済みの経験を抽選に出せる条件は共通の範囲判定に従う。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)

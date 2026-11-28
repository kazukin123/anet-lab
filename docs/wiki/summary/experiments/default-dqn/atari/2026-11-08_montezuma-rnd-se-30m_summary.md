---
title: Montezuma's Revenge の RND（サンプル効率優先・30M）
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-11-08_montezuma-rnd-se-30m.md|default-dqn/atari/2026-11-08_montezuma-rnd-se-30m]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/subjects/montezuma_revenge.md|Montezuma's Revenge]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
updated: 2026-11-28
---
# Montezuma's Revenge の RND（サンプル効率優先・30M）

## 元の材料

[Montezuma's Revenge の RND（サンプル効率優先・30M）](../../../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-se-30m.md) blob:1d2fb74b557a33f902afb56c6393d6d6dad5c5b4

## 要点

- 学習 Run の Greedy（ε=0、τ 固定）は 20.4M まで全ゲーム 0 点で、25.45M に 243.8（400 点は 9/16）だった。
- ε=0.01・τ 乱数の評価は 14.21M から 22.21M まで 5 回続けて 400（10/10）だった。
- 学習側で 500 点のゲームが増えたのは 24M 以降だった。
- RND の bonus の平均は 24〜25M に 0.0144 で、1M 前（0.0062）と 1M 後（0.0033）より高く、26〜27M に 0.0013 だった。
- 30M の重みを 4 条件・各 32 ゲームで測ると、ε=0・τ 固定が 259.4（400 点以上 19/32）、ε=0・τ 乱数が 362.5（29/32）、ε=0.01・τ 固定が 403.1（32/32）、ε=0.01・τ 乱数が 412.5（32/32）だった。
- 20.07M の重みは、4 条件とも全ゲーム 400 点だった。

## 内容

### 目的と比較条件

Montezuma で、サンプル効率優先プロファイルに RND を載せた構成の 30M までの到達を測る。あわせて、学習 Run の中で Greedy が 400 点に届くのが ε=0.01 の評価より遅かったことについて、評価時の ε と τ を入れ替えて測る。

Train Actor の探索は、lane ごとの ε ラダー（0.4〜0.01、log）と UQE。τ は 32 本・乱数。

### 探索ブロック01: 学習 Run の評価

0.2M のゲームは長さの中央値 127 step で終わった。5.25M 以降は、10.3M の `53_evalg` の 1 本を除いて全て打ち切りだった。400 点のゲームの `room_max` は 2、それ以外は 1 だった。

### 探索ブロック02: 学習側

ログの Game 行を exp step の 2M の窓で数えた。部屋 7 の列は `room_max` が 7 のゲームの数。

### 探索ブロック03: RND の bonus と Q・loss

`metrics_cache.db` の値を exp step の窓で平均した。

## 主要な概念

- [内的報酬による探索の比較](../../../../concepts/methods/intrinsic_novelty.md)
- [Montezuma's Revenge](../../../../concepts/subjects/montezuma_revenge.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
- [τ配置とIQN比較](../../../../concepts/methods/tau_sampling.md)

---
title: "Atari Breakout: UQE 探索方策の掃引 — 2026-09-07"
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-09-07_uqe-exploration.md|default-dqn/atari/2026-09-07_uqe-exploration]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/subjects/breakout.md|Breakout]]"
updated: 2026-11-28
---
# Atari Breakout: UQE 探索方策の掃引 — 2026-09-07

## 元の材料

[Atari Breakout: UQE 探索方策の掃引 — 2026-09-07](../../../../../experiments/default-dqn/atari/2026-09-07_uqe-exploration.md) blob:264dc1f2e7ac2ee99949356d364e33140985738e

## 要点

- train 側の壁突破率は、この軸の判定に使えない。
- eval では UQE は中立だった。
- 確度が高いのは「上端 0.7 は悪い」だけ。
- 学習は破綻しない。 4 点とも 50M を完走し、壁突破率は全点で基準を上回る水準（12.55〜13.62%、train 軸）。UQE を入れても壊れないことは確定した（探索ブロック 01）。
- Breakout は楽観・悲観の軸を測る env として不適である。

## 内容

### 探索ブロック 01: UQE ラダー両端の十字掃引

p75 / p90 はほぼ同一で、p10 / p25 だけが持ち上がっている。 ε ラダーは上端 0.4 の env が意図的に下手に打つため train エピソードの 3 分の 1 が 100 点未満になる。 UQE は `uqe_eps = 0.05` でランダム行動がほぼ無く、全 env が楽観 greedy で打つ。 train スコアの差は挙動方策の差であり、学習した方策の質ではない。

`≥600` が 434 → 197〜234 と半減しているのも同じ理由で、ε ラダーの低 ε env が稼いでいた高得点エピソードが、 UQE では「全 env がそこそこ」に均されたためと読める。

### 判断

Breakout はブロックを消すだけのゲームで、行動のリスク構造がほとんど無い。楽観的に見積もるか悲観的に見積もるかで 方策が変わる余地が小さく、UQE の効果を測る env として不適である。実測でも eval が基準と同水準に留まった。

## 主要な概念

- [τ配置とIQN比較](../../../../concepts/methods/tau_sampling.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
- [Breakout](../../../../concepts/subjects/breakout.md)

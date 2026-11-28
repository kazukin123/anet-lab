---
title: Policy churn メトリクス（35_agent_churn）PRD
page_kind: サマリ
material: "[[docs/memo/done/0066_policy_churn_metrics_10rfd.md|0066_policy_churn_metrics_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/phenomena/plasticity.md|可塑性指標と成績]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/subjects/breakout.md|Breakout]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-28
---
# Policy churn メトリクス（35_agent_churn）PRD

## 元の材料

[Policy churn メトリクス（35_agent_churn）PRD](../../../memo/done/0066_policy_churn_metrics_10rfd.md) blob:834d18c5d10c7407fa573188defd47d05cdc40a7

## 要点

- Breakout では表現指標の改善が評価スコアや壁の突破率の改善と対応しない実験があった。
- ReplayBuffer から一様に取った同じ状態で、更新前後の online network を比べる。
- greedy 行動が変わる割合と expected Q の変化を1回の Learner 更新に対応づける。
- 同じ測定時点に online と target が選ぶ greedy 行動の違いも取る。
- 診断用の測定が学習の入力や方策を変えないことを条件にする。

## 内容

### 用語と数理契約

churn の action は ActionPolicy の探索、UQE、Thompson Sampling から独立した、network の expected Q に対する `argmax` とする。

### メトリクス契約

EMA は Q 由来 6 種すべてに漏れなく用意する。利用予定の有無で取捨せず、後から番号を振り直さずに済ませるためである。非有限値は EMA 更新の前段で除外されるため（`observers.cpp` の `!std::isfinite` ガード）、NaN が EMA 状態を汚染しない。alpha は疎な系列の慣行に合わせて 0.1 とする。
LearnEvent が存在しない warmup 中は、メトリクス点自体を出さない。

### Probe 契約

churn は ReplayBuffer から一様・非復元で 1024 件を 1 回抽出する。plasticity と共有するのは sampler の仕組みだけであり、plasticity の実バッチや既定 4096 件は共有・流用しない。

### 将来実装の受入条件

同一ビルド内の非干渉: RNG 移行後の同じビルドで churn ON / OFF を比較し、学習 parameter、通常 Replay sample 系列、plasticity probe 系列、既存 metrics checksum が一致する。RNG クリーンブレーク前後の `4x_probe_*` checksum は比較対象外とする。

## 主要な概念

- [可塑性指標と成績](../../concepts/phenomena/plasticity.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [Breakout](../../concepts/subjects/breakout.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)

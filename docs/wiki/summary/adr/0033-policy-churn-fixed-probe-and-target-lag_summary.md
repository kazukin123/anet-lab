---
title: Policy churn は fixed probe で optimizer 前後を測り、target は sync 後の不一致を測る
page_kind: サマリ
material: "[[docs/adr/0033-policy-churn-fixed-probe-and-target-lag.md|0033-policy-churn-fixed-probe-and-target-lag]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
updated: 2026-11-28
---
# Policy churn は fixed probe で optimizer 前後を測り、target は sync 後の不一致を測る

## 元の材料

[ADR 0033](../../../adr/0033-policy-churn-fixed-probe-and-target-lag.md) blob:f837fea09355dae8de9a0783b832ce9d58335268

## 要点

- 方策の変化は一様に非復元抽選した同じ probe 上で greedy action を比較する。
- online 方策を optimizer の更新前後に測る。
- target の遅れは同期後の online と target の不一致として測る。
- 固定した IQN 分位点と FP32 の推論を使い、丸め・再抽選を測定差から除く。
- 測定の乱数器を独立にし、学習系列と他の診断周期に干渉しないようにする。
- 最初の公開範囲は DefaultDQN。hard update のどの時点で測ったかは target_sync_age で別に記録する。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)

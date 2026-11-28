---
title: Agentと学習
page_kind: サマリ
material: "[[docs/design/110_agents_and_learning.jp.md|110_agents_and_learning.jp]]"
material_kind: 設計文書
concepts:
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/framework/325_muzero.md|325 Agent個別実装 MuZero系列]]"
updated: 2026-11-28
---
# Agentと学習

## 元の材料

[Agentと学習](../../../design/110_agents_and_learning.jp.md) blob:61b9e537fa06a7f22e2798308425c2a8f0123466

## 要点

- Agent は Actor と Learner が使う状態と資源を管理し、具象 Agent が生成・保存の契約を実装する。
- Actor の方策や参照する Network は ActorRequest の actor_key でカタログから解決する。
- 同じ Actor に MakeAction と Sync を同時に呼ばないよう、Runner 側が直列化する。
- Learner は Experience を受け取り、更新が成立したかを実行側へ返す。
- モデルの所有と複製、同期、checkpoint の保存・読込は、学習中と評価中でどの状態を見るかを決める。

## 内容

### 2. 基本概念と外部contract

- `Agent`はRunから利用する入口であり、ActorとLearnerを生成する。device取得、保存・読込、可視化用functionの公開境界でもある。
- Runner が系統の公開初期化関数を呼んで登録し、`AgentRepository`がclass IDから`AgentFactory`を解決する。

### 3. コンポーネント定義

- `AgentFactory` — EnvSpec、BatchEnvSpec、device、ConfigData、seedから具象Agentを構築するinterface。
- `Actor` — BatchStateからBatchActionInfoを生成し、必要に応じて推論Resourceを同期するinterface。

### 6. 処理フロー

- `LearnEvent`を生成するのはLearnerではなく、更新結果を受け取ったRunnerである。LearnerがReplayBufferを使うか、1回の呼出しで何回更新するかは具象実装が決める。

### 7. 設定・lifetime・エラー・性能特性

- EnvSpec、BatchEnvSpec、device、seed、ConfigDataはfactoryから具象Agentへ渡す。
- Train Actor、configured Eval Actor、EvalPanel Actorは用途ごとに生成される。共有ResourceとActor-private Resourceを区別する。

### 8. テストと拡張時の確認事項

- 系統の公開初期化関数から `RegisterAgentFactory` を呼び、一意な class ID で factory を登録する。
- metric、可視化、error、性能計測を共通interfaceへ無理に混在させず、既存のProvider・Observer・profile境界へ接続する。

## 主要な概念

- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [325 Agent個別実装 MuZero系列](../../concepts/framework/325_muzero.md)

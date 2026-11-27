---
title: 共有 Network と Actor 専用コピーの排他監査
page_kind: サマリ
material: "[[docs/memo/0134_network_lock_audit_10rfd.md|0134_network_lock_audit_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/framework/325_muzero.md|325 Agent個別実装 MuZero系列]]"
updated: 2026-11-26
---
# 共有 Network と Actor 専用コピーの排他監査

## 元の材料

[RFD0134　共有 Network と Actor 専用コピーの排他監査](../../../memo/0134_network_lock_audit_10rfd.md) blob:9a1b2d974922ea270ae13b12902ff063212c6ffa

## 要点

- これは排他の監査計画である。Actor と Learner が競合するか、修正を要するかは原本時点の調査対象である。
- 学習と評価モードを整理した後の、共有 Network と Actor 専用コピーの保護範囲を調べる草案である。
- 推論、重み更新、複製、可視化用の読み出しがどのロックを必要とするかを照合する。
- 構築後に Actor を追加できるか、実行中に重みを読み込めるかなど、寿命に依存する経路を分ける。
- Actor の観測正規化と Learner の統計更新の並行性も調査対象に含める。
- 安全な経路、前提付きで安全な経路、実際に競合しうる経路を分類してから修正の要否を判断する。

## 内容

### Context

Dropout 導入前の前提整理として、NN module の `train()` / `eval()` mode 契約を見直した。
`TrainingModeGuard` の呼び出しが常に `unique_lock` 内にあることを体系的に確認できていない。

### Problem

現状のコードで安全な経路、ライフサイクル仮定に依存している経路、実際に race し得る経路を分類し、最小修正案を決められる状態にする。

### Goals

DQN / Rainbow / ImageCls / MuZero について、Actor / Learner / Agent / NetworkModel の lock 契約を棚卸しする。

### Non-Goals

lock-free 化、fine-grained lock 化、性能最適化を主目的とした設計変更。

### Current Contract

`ForwardOnlineWithTrain()` は mode を変更するため、呼び出し側が `unique_lock` を保持している前提とする。

### Audit Targets

`Actor::Sync()` と同一 Actor の `MakeAction()` が並行し得るか。並行し得る場合、actor-private `network_` の `CopyTo()` と forward が race しないか。

### Required Output

現状の lock と lifecycle で安全と判断できる経路。

### Acceptance Criteria

shared network を変更する train/eval / optimizer step / load / target update 経路について、`unique_lock` の有無が説明されている。

### Draft Recommendation

`TrainingModeGuard` 本体に lock を持たせるのではなく、呼び出し側の Agent/Learner lock 契約を明示する方針を維持する。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [325 Agent個別実装 MuZero系列](../../concepts/framework/325_muzero.md)

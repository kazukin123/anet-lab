---
title: ActionContext の廃止（Actor が RNG と frame stacker を直接持つ）
page_kind: サマリ
material: "[[docs/memo/done/0083_action_context_retire_10rfd.md|0083_action_context_retire_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-28
---
# ActionContext の廃止（Actor が RNG と frame stacker を直接持つ）

## 元の材料

[ActionContext の廃止（Actor が RNG と frame stacker を直接持つ）](../../../memo/done/0083_action_context_retire_10rfd.md) blob:db3345bf9bae73ac9e74c7ceac2aac999ae58773

## 要点

- ActionContext が RNG とフレームスタックという別々の関心を束ね、観測の正規化は Actor に分かれていた。
- Actor が必要な乱数器と frame stacker を直接所有する形に整理する。
- 行動選択前の観測加工とその依存を、具体的な Actor から確認できるようにする。
- 生成に必要な情報は ActorRequest を通じて Agent に渡す。
- 行動の数値や乱数消費を保ちながら、既存の Actor の生成・評価経路を移す。

## 内容

### Actor の依存を整理する理由

DQN の行動前処理は、フレームの stack と device への転送を ActionContext が担当し、正規化は Actor 自身が担当していた。乱数も context 経由で取得しており、単に観測を転送する構成でも context を必要としていた。

### 行動選択に至る順序

Actor が専用乱数器、任意の frame stacker、device を直接所有する。stacker があるときはその device 上で stack と転送を行い、ないときは Actor が従来と同じ同期転送を1回だけ行う。その後に既存の `obs_norm_` で正規化する。乱数は `ActorRequest.seed` から作り、Actor だけが消費する。

### 同じ行動系列を確認する条件

ReplayBuffer の学習用 stack 再構成・Learner の正規化と、可視化用の経路は従来どおりにする。変更前に固定した seed・設定・終了ステップで短い Run を取得し、変更後も同じ条件で各指標の全点を比較する。Rainbow の Run 比較は今回の検証範囲から外した。

## 主要な概念

- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)

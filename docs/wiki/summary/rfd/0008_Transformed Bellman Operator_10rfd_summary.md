---
title: Transformed Bellman Operator (TBO) 導入 仕様書
page_kind: サマリ
material: "[[docs/memo/done/0008_Transformed Bellman Operator_10rfd.md|0008_Transformed Bellman Operator_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
updated: 2026-11-28
---
# Transformed Bellman Operator (TBO) 導入 仕様書

## 元の材料

[Transformed Bellman Operator (TBO) 導入 仕様書](<../../../memo/done/0008_Transformed Bellman Operator_10rfd.md>) blob:657910e3edeef89b85c2ac0999ec9f2c86650d2f

## 要点

- DQN の大きな Q 値を可逆な非線形関数で圧縮して学習する。
- Bellman 更新では bootstrap 値を元の尺度へ戻してから報酬と割引を適用し、再び変換する。
- 変換と逆変換は Tensor の各要素に適用する。
- 線形項 εx により、大きな値でも逆変換できる尺度を保つ。
- 将来の Munchausen 報酬を見据えるが、この仕様の変更対象は Transformed Bellman Operator に限る。

## 内容

### Context（背景・目的）

DQN 系エージェントで Q 値が発散・暴走しやすい問題への対策として、Transformed Bellman Operator（Pohlen et al. 2018 "Observe and Look Further", R2D2/Ape-X 系）を導入する。Bellman ターゲットを可逆な圧縮関数 `h` で写像することで、報酬スケールに依存せず大きなターゲット値を抑え、学習を安定化させる。

### Q 値メトリクス（学習 scalar の h 空間生値 ＋ h⁻¹ 逆変換値）

h 空間の生値は従来 tag を維持する（学習が実際に見ている値）。tag 命名は既存規約（TensorBoard 由来のメトリクス識別子）に準拠し、`name`（Runner 等のインスタンス識別子）とは混同しない。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [報酬尺度と割引率](../../concepts/methods/reward_scale_discount.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)

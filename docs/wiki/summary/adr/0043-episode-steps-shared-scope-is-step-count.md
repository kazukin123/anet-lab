---
title: エピソード長は SHARED でも Step() 回数とし、episode return の総和とは非対称にする
page_kind: サマリ
material: "[[docs/adr/0043-episode-steps-shared-scope-is-step-count.md|0043-episode-steps-shared-scope-is-step-count]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/320_env_core.md|320 Env 共通基盤]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/subjects/imagecls.md|ImageCls]]"
updated: 2026-11-26
---
# エピソード長は SHARED でも Step() 回数とし、episode return の総和とは非対称にする

## 元の材料

[ADR 0043](../../../adr/0043-episode-steps-shared-scope-is-step-count.md) blob:c8465333c6a17cf6033c9cd36963dde9cf3cf4bd

## 要点

- PER_LANE のエピソード長はその lane で開始から完了までに呼んだ Step の回数とする。
- SHARED でも batch 全体で呼んだ Step の回数を使う。
- 共有報酬の return は全 lane・全 step を合計するため、長さと集約方法が異なる。
- 長さに lane 数を掛けると環境の長さと並列度が混ざる。
- ImageCls では1 epoch に必要な minibatch 数として解釈できる。
- Atari が残機を1エピソードと数える場合、実ゲーム全体の長さと framework の episode_steps は境界が異なる。

## 内容

### 値が表すもの

エピソードの長さを lane 数倍すると、実行並列度を増やしただけで長く見えてしまう。SHARED の場合も1グループが何回の Step を要したかで比較する。

### return との違い

報酬は1エピソードに寄与した全ての値を足すのに対し、長さは共同で進めた1回を1と数える。ImageCls では epoch の実行に要した minibatch 数がそのまま対応する。

## 主要な概念

- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [ImageCls](../../concepts/subjects/imagecls.md)

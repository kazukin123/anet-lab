---
title: replay の slot に 1 step の reward と終わり方を持ち、サンプル時に差し替え可能な窓の規則で可変長の区間を組む
page_kind: サマリ
material: "[[docs/adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md|0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
updated: 2026-11-26
---
# replay の slot に 1 step の reward と終わり方を持ち、サンプル時に差し替え可能な窓の規則で可変長の区間を組む

## 元の材料

[ADR 0063](../../../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md) blob:f5a6b6bc169a50afd29f9dc6f36cb61a333acae3

## 要点

- Replay は各 slot に1 step の報酬と終了条件を記録する。
- 抽選時に窓の規則を適用し、可変長の区間とその収益を作る。
- 別バッファとの二重管理や、Push 時に特定の窓だけを先に計算する必要がなくなる。
- 窓の上限を抽選可能になるまでの遅れと容量の下限に含める。
- 補助学習だけでスキップ長を1に固定する構成では、行動の変化または終了と上限で窓を閉じる規則も使える。

## 内容

### 窓の作成

Push の時点で固定した n-step の結果だけを残すと、行動が続いた区間などを後から別の規則で集計できない。step ごとの記録を保持し、抽選した開始点から設定された上限までを見る。

### 既存経路との関係

窓を使う条件でも抽選分布と item key は維持し、優先度確定と追い出し統計の ready 判定は通常の n-step を参照する。後から追加した区切り規則は、実際の行動をスキップしない補助学習専用の条件に限る。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [行動の時間尺度](../../concepts/methods/temporal_abstraction.md)

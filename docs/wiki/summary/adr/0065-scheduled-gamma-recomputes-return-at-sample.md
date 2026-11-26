# 時間で変わる割引率は、schedule のときだけ抽選時に保存済みの step 報酬から収益を作り直す

## 元の材料

[ADR 0065](../../../adr/0065-scheduled-gamma-recomputes-return-at-sample.md) blob:0c7a8161047982e4ccc8586fa51aa312b46a1a3a

## 要点

- 割引率が時間で変わると、Push 時の n-step 収益と学習時の bootstrap の割引率がずれる。
- schedule を使うときは抽選時に保存済み step 報酬と現在の割引率から収益を再計算する。
- 割引率が定数なら Push 時の収益をそのまま使う。
- 定数条件にも再計算を課すと計算量が増え、加算順によって既存の値が変わりうる。
- 後の価値分解ではこの規則を報酬成分ごとに適用する。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [報酬尺度と割引率](../../concepts/methods/reward_scale_discount.md)

---
title: MunchausenターゲットはDQN Learnerに局所化し、実空間で計算し、3つのlog-policy modeを持つ
page_kind: サマリ
material: "[[docs/adr/0035-munchausen-target-learner-local-real-space.md|0035-munchausen-target-learner-local-real-space]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
updated: 2026-11-26
---
# MunchausenターゲットはDQN Learnerに局所化し、実空間で計算し、3つのlog-policy modeを持つ

## 元の材料

[ADR 0035](../../../adr/0035-munchausen-target-learner-local-real-space.md) blob:c38cd7ee19fc5f5e5badeabb3769b30d49832dc6

## 要点

- Munchausen target は実行行動の scaled log-policy を報酬側に加え、次状態では soft 価値を使う。
- ターゲットは実空間で組み立て、完成後にだけ値変換をかける。
- log-policy の取得には target、online の新規推論、既存 online 出力の再利用を用意する。
- 再利用と新規推論では費用や train/eval 条件が異なるため別々に比較できる。
- 後の追記で通常方式との式を MakeBellmanTarget にまとめ、実空間へ戻した bootstrap を受け取るようにした。

## 内容

### ターゲットの計算

先頭遷移の実行行動について bonus を作り、N-step 収益と soft bootstrap を組み合わせる。TBO を使う場合も、分位点ごとに実空間へ戻した後で方策や平均を求める。変換済みの値を直接平均して代用しない。

### 推論条件と費用

target の追加 forward、online の独立した eval 推論、train 側の既存出力の再利用を区別する。再利用は追加の乱数を消費しないが、前の2方式とは推論モードや鮮度が異なる。

### 比較時の制約

楽観的な UQE スコアを用いると bonus 差もそのスコアに依存し、論文の action gap や churn に関する性質をそのまま期待できない。違法行動のマスクはこの決定では扱わない。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [報酬尺度と割引率](../../concepts/methods/reward_scale_discount.md)
- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)

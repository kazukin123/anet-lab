---
title: Atari-5 外ゲームでの実時間優先プロファイル 100M 探索履歴
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-09-23_realtime-profile-100m.md|default-dqn/atari/2026-09-23_realtime-profile-100m]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/subjects/kung_fu_master.md|kung_fu_master（KFM）]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
updated: 2026-11-28
---
# Atari-5 外ゲームでの実時間優先プロファイル 100M 探索履歴

## 元の材料

[Atari-5 外ゲームでの実時間優先プロファイル 100M 探索履歴](../../../../../experiments/default-dqn/atari/2026-09-23_realtime-profile-100m.md) blob:b96fb113df4e3ca69f0b7c652e76afef4dfc81f7

## 要点

- 同予算の BTR とはほぼ互角。 50.7M（202.8M frames）までの ε=0 断面の最良は seed 1 が 68,404（HNS 303.2%）、 seed 2 が 44,417（196.5%）。BTR の 53,302（236.0%）に対して +28.3% / −16.7%、 2 seed 平均は 56,411（249.8%）で +5.8%。1 seed で出した「+28.3%」は、良い seed を引いた値だった（探索ブロック 01）。
- 学習途中は seed 間のブレが大きい。
- 50M の頭打ちは seed 1 だけに出た。
- `q_gap` の縮小と `action_churn` の半減は、成績が落ちなかった seed 2 にも同じ大きさで出た。
- 最終スコアの断絶（74,900 → 84,200）は 75,000 点の 1UP による。
- 点が入る step 1 回あたりの点数は seed で違う。

## 内容

### 探索ブロック 01: kung_fu_master 100M と BTR 比較

seed 1 だけが 50M で山をつけ、終盤は 2 本が並ぶ。 ε=0 断面（100 エピソード）。

### 探索ブロック 02: 最終スコアの断絶と 75,000 点の 1UP

ライフを失うと、その面の最初へ戻る。 同じ方法で確かめた。歩いている間だけ変わる RAM 4 バイト （0xC6 / 0xC7 / 0xD3 / 0xD4。攻撃しているフレームでは 1 回も変わらない）が、ゲーム開始時の値へ戻る。 画面上段のタイマーは 2000 に戻り、スコアはそのまま残る。確認したのは 1 面だけ。

### 探索ブロック 03: 報酬のクリップと 1 step あたりの点数

seed 1 の頭打ちはクリップ後 return にも同じ割合で出ている（50-60M の 320.9 → 90-100M の 265.5、−17.3%）ので、 エージェントの目的（点が入った step の数）とスコアのずれでは説明できない。

### pending

面の進行。 ボス撃破や面の突破は測っていない（面の番号が RAM のどこにあるかは未特定）。

## 主要な概念

- [kung_fu_master（KFM）](../../../../concepts/subjects/kung_fu_master.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
- [Atari横断](../../../../concepts/subjects/atari.md)

---
title: DropMerge の評価スコア（本家互換スコアほか複数の軸）暫定 RFD
page_kind: サマリ
material: "[[docs/memo/0085_dropmerge_eval_score_axes_10rfd.md|0085_dropmerge_eval_score_axes_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
updated: 2026-11-28
---
# DropMerge の評価スコア（本家互換スコアほか複数の軸）暫定 RFD

## 元の材料

[DropMerge の評価スコア（本家互換スコアほか複数の軸）暫定 RFD](../../../memo/0085_dropmerge_eval_score_axes_10rfd.md) blob:8241bc1f23a97692b0d5fe9c71c0ce91e4eb98d8

## 要点

- DropMerge のゲーム成績を、学習用の報酬とは別の尺度で表示するための暫定 RFD である。
- episode_return には合体の報酬に加え、SpawnBlocked や Overflow、NoDropTimeout の罰も含まれる。
- 本家スイカゲームに対応する得点を設けると、人が見た合体成績と報酬の差を追える。
- 評価で必要になる場合に備え、複数のスコアの軸を並べる構想を示す。
- 報酬設計の変更と評価の尺度選びを別々に判断できるようにする。

## 内容

### 目的と問題

評価の物差しを、学習の報酬から切り離す。人間が画面で見る「ゲームの点数」を、本家スイカゲームと同じ得点体系で出せるようにし、評価に支障がある場合に備えて複数の軸を持てるようにする。

### 提案の骨子

DropMerge スコア: 今の `fruit_scores` の合計（現行の `episode_score_` と同じ値）。過去の Run との連続性のため。

### User Stories

研究者として、最良の Run の点数を、本家スイカゲームの人間の記録と同じ物差しで比べたい。最終目標（人間のチャンピオン超え）の進み具合を測るため。

## 主要な概念

- [DropMerge](../../concepts/subjects/dropmerge.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)

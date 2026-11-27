---
title: 自己模倣による価値学習
page_kind: 手法
aliases:
  - SIL
  - SAIL
  - self-imitation learning
  - 自己模倣
description: 過去に得た高い収益を使う方法には、良い軌跡を再利用できる利点と、偶然の高収益を繰り返し学ぶ偏りがある。
updated: 2026-11-27
---
# 自己模倣による価値学習

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [手法](../../index.md#手法)

## 今の結論

過去に得た高い収益を使う方法には、良い軌跡を再利用できる利点と、偶然の高収益を繰り返し学ぶ偏りがある。

## 分かっていること

### anet の知見

なし

### 外の事例

- Montezuma の比較では、自己模倣に探索報酬と複数ステップの更新を組み合わせた群が最高点だった。 → 詳しくは [Montezuma’s Revenge](../subjects/montezuma_revenge.md)（出典: [自己模倣の調査](../../../../reports/sil_value_based_lower_bound_survey_2026-11-24.md)、§3）。
- 決定的な Atari を扱った MFEC の予備実験では、記憶を最大収益で更新する方法をやめ、観測収益で上書きすると性能が落ちた。確率的報酬下の最大値の偏りを、その実験で直接検証したものではない（出典: [Survey: 自己模倣（SIL）を価値ベースへ広げた手法と、確率的な環境での偏り](../../../../reports/sil_value_based_lower_bound_survey_2026-11-24.md)、7. 偏りへの対策と効果）。
- 文献の比較条件: SAIL の論文の Montezuma の値は最終の値で、期間の平均を別に書く（RND-IQN は最終 161、期間の平均 45）。指標は 1M step ごとの 200 回の値で、総学習量は書いていない。Montezuma の値の seed の数は書いておらず、seed は部分集合の比較が 6、全ゲームが 3 とする（出典: [montezuma_published_scores_survey_2026-11-27.md](../../../../reports/montezuma_published_scores_survey_2026-11-27.md)、§3.2）。

## 横断的知見

なし

## 見立て

なし

## 分かっていないこと

なし

## 撤回・更新

なし

## 関連

なし

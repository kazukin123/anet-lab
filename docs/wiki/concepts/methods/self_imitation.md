---
title: 自己模倣による価値学習
page_kind: 手法
aliases:
  - SIL
  - SAIL
  - self-imitation learning
  - 自己模倣
description: 過去に得た高い収益を使う方法には、良い軌跡を再利用できる利点と、偶然の高収益を繰り返し学ぶ偏りがある。
updated: 2026-11-28
---
# 自己模倣による価値学習

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [手法](../../index.md#手法)

## 今の結論

過去に得た高い収益を使う方法には、良い軌跡を再利用できる利点と、偶然の高収益を繰り返し学ぶ偏りがある。

## 分かっていること

### anet の知見

なし

### 外の事例

- Montezuma の比較では、自己模倣に探索報酬と複数ステップの更新を組み合わせた群が最高点だった。 → 詳しくは [Montezuma’s Revenge](../subjects/montezuma_revenge.md)（出典: [自己模倣の調査](../../../../reports/survey/sil_value_based_lower_bound_survey_2026-11-24.md)、§3）。
- 決定的な Atari を扱った MFEC の予備実験では、記憶を最大収益で更新する方法をやめ、観測収益で上書きすると性能が落ちた。確率的報酬下の最大値の偏りを、その実験で直接検証したものではない（出典: [Survey: 自己模倣（SIL）を価値ベースへ広げた手法と、確率的な環境での偏り](../../../../reports/survey/sil_value_based_lower_bound_survey_2026-11-24.md)、7. 偏りへの対策と効果）。
- 方策ベースの自己模倣には、収益の上位の軌跡だけを持つ記憶を使う手法がある。GASIL は割引収益で上位 K 本（大きさは step 数で 1,000・10,000）、SIDP は割り引かない和の優先度つきキューで 10 本を持つ。どちらも Atari の実験は無い（出典: [作業ノート: 自己模倣の記憶の持ち方の事例](../../../../reports/notes/self_imitation_memory.md)、項目 1・2）。
- 価値ベースへ広げた手法（Optimality Tightening、EMDQN、Tang の下限 Q 学習、SAIL、GEM、DB-ADP）には、上位の何本だけを残す方式が見つからなかった（出典: [Survey: 自己模倣（SIL）を価値ベースへ広げた手法と、確率的な環境での偏り](../../../../reports/survey/sil_value_based_lower_bound_survey_2026-11-24.md)、9. 総合評価）。
- 上位の軌跡を真似る自己模倣は、確率的な環境では過去の上位の軌跡を真似るだけでは最適方策に戻りにくい、と SIDP は 2 本腕のバンディットの例で述べる。GASIL も、決定的な環境を要する点を今後の課題とする（出典: [作業ノート: 自己模倣の記憶の持ち方の事例](../../../../reports/notes/self_imitation_memory.md)、項目 1・2）。
- 原典の SIL を PPO に足した版は、MuJoCo の 6 課題・10 seed のうち Swimmer・Walker2d・Ant で PPO より良く、報酬を 20 step ごとにまとめて渡す版では PPO との差が大きくなった。PPO の 1 回の更新ごとに SIL の更新を 10 回足し、SIL の損失には新旧の方策の比を使わない（出典: [作業ノート: 自己模倣の記憶の持ち方の事例](../../../../reports/notes/self_imitation_memory.md)、項目 4）。
- 2022 年以後の SIL の後続には、手続き生成のまれな報酬の環境（MiniGrid・ProcGen）で内的動機づけや再生の多様化と組む手法（Andres ら 2022・2023）と、PPO の上で成功した軌跡を一様に再生する on-policy の手法（Chaudhary ら 2026）がある。どれも要旨だけを読んだ（出典: [作業ノート: 自己模倣の記憶の持ち方の事例](../../../../reports/notes/self_imitation_memory.md)、項目 5）。
- 文献の比較条件: SAIL の論文の Montezuma の値は最終の値で、期間の平均を別に書く（RND-IQN は最終 161、期間の平均 45）。指標は 1M step ごとの 200 回の値で、総学習量は書いていない。Montezuma の値の seed の数は書いておらず、seed は部分集合の比較が 6、全ゲームが 3 とする（出典: [montezuma_published_scores_survey_2026-11-27.md](../../../../reports/survey/montezuma_published_scores_survey_2026-11-27.md)、§3.2）。

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

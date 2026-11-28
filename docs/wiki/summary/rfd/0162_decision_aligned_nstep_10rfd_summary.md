---
title: 次の判断時点で打ち切る可変長 n-step 案
page_kind: サマリ
material: "[[docs/memo/0162_decision_aligned_nstep_10rfd.md|0162_decision_aligned_nstep_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-28
---
# 次の判断時点で打ち切る可変長 n-step 案

## 元の材料

[RFD0162　次の判断時点で打ち切る可変長 n-step 案](../../../memo/0162_decision_aligned_nstep_10rfd.md) blob:852c63cc295d4fd159b56f3a7357ff3db9968594

## 要点

- 次の判断までを一つの窓として学習する構成は暫定案で、効果の確認と受入条件の選択を残している。
- DropMerge の NOOP が連続すると、n-step が1の学習では待った結果が一度の更新で1 step しか遡らない。
- 待機中の窓を延ばし、次に行動を判断する時点の状態を待機開始時の価値に反映する案である。
- 決定時点にそろえる窓と固定長の窓で、遡る距離と bootstrap の時点を比べる。
- 終了や上限が途中に来る場合の境界、Replay が返す収益と割引を明示する必要がある。
- 待機の費用や割引率そのものの変更とは区別し、可変長窓の学習効果を確かめる。

## 内容

### 目的と問題

待っている間の遷移では n-step の窓を延ばし、次に判断した時点で区切る。待ち始めた step の価値を、次に DROP を決める step の価値で直接答え合わせできるようにする。

### 起草時点の現状

QR51 + Munchausen τ0.03 100M（`run_20261003-081944_dmfb_qr_munch`）でも、10〜22M に長考型の NEET が出た。14M 付近の録画（EvalPanel、target ネット + Greedy）では、NoDropTimeout にならない範囲で NOOP を続けてから DROP していた。評価のエピソード長は 12〜14M に 1,458 step まで伸びたが、NoDropTimeout で終わった回は 10〜15M に約 1 割だけで、大半は締切の手前で DROP していた（2026-10-03 集計）。締切の直前で落とすので、NoDropTimeout の罰を受ける回は少ない。

### 提案の骨子

判断かどうかは、0167_smdp_transition_duration の経過時間から決める案を既定にする（DROP 単位で数えるとき、τ = 0 の遷移は待ち）。別のフラグを持たないので、時間の数え方と区切りが食い違わない。

### User Stories

研究者として、NEET の長考型（締切の直前だけ落とす）が、少ない更新で崩れるようにしたい。
研究者として、時間の数え方（0167_smdp_transition_duration）を変えたときに、n-step の区切りも自動でそれに合ってほしい。

### 決めること

窓の上限の長さ（ショットクロックの 100 step に合わせるか）。
step 単位で数えるとき（τ = 1 のまま）にも使えるよう、Env が判断かどうかのフラグを返す案を残すか。

## 主要な概念

- [行動の時間尺度](../../concepts/methods/temporal_abstraction.md)
- [報酬尺度と割引率](../../concepts/methods/reward_scale_discount.md)
- [DropMerge](../../concepts/subjects/dropmerge.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)

---
title: 訪問回数と擬似カウントによる探索
page_kind: 手法
aliases:
  - CTS
  - PixelCNN
  - pseudo-count
  - SimHash
  - φ-EB
description: 密度が観測の前後でどう変わるか、またはどの特徴を同じ状態と数えるかで、未訪問状態へのボーナスが変わる。Montezuma では価値更新の方法にも結果が左右された。
updated: 2026-11-27
---
# 訪問回数と擬似カウントによる探索

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [手法](../../index.md#手法)

## 今の結論

密度が観測の前後でどう変わるか、またはどの特徴を同じ状態と数えるかで、未訪問状態へのボーナスが変わる。Montezuma では価値更新の方法にも結果が左右された。

## 分かっていること

### anet の知見

なし

### 外の事例

- CTS は、同じ状態の観測前後の確率から「一度観測すると 1 増える」回数 N̂ を求める。学習で確率が下がらないとき非負で、経験分布を使うと実カウントに一致する。DQN のボーナスは β / √(N̂ + 0.01)、β=0.05（出典: [内的報酬の手法の全体像](../../../../reports/survey/intrinsic_reward_landscape_survey_2026-11-18.md)、§3.1）。
- PixelCNN は負の予測利得を 0 にし、学習時刻に応じた縮尺を用いた。#Exploration は SimHash・手設計の BASS 等で観測を離散化し、φ-EB は価値近似の特徴に密度を置く（出典: [内的報酬の手法の全体像](../../../../reports/survey/intrinsic_reward_landscape_survey_2026-11-18.md)、§3.1・3.3）。
- CTS 付き DQN では Montezuma の 50M フレームで 15 部屋、ボーナスなしで 2 部屋。MMC を外すとボーナスがあっても成績が落ちた。150M フレーム中の最高点の 3 seed 平均は DQN-CTS 3,705.5、DQN-PixelCNN 2,514.3。一方全 57 本では PixelCNN が 52 ゲームで CTS を上回った（出典: [内的報酬の手法の全体像](../../../../reports/survey/intrinsic_reward_landscape_survey_2026-11-18.md)、§3.2）。
- Montezuma の RAM から手設計したハッシュに敵の位置を含めると、遠くの敵を眺め続け 1,672 点、除くと 5,661 点。Freeway の φ-EB でも β が大きいと交通を眺めてボーナスが続いた。MiniHack では観測中の時刻で同じ状態が繰り返さず、元の NovelD の初訪問条件が効かなかった（出典: [内的報酬の手法の全体像](../../../../reports/survey/intrinsic_reward_landscape_survey_2026-11-18.md)、§3.4）。
- NGU はエピソード内 k 近傍カーネルで回数を近似し、E3B の楕円ボーナスは one-hot のとき訪問回数の逆数に戻る。原典が着想として挙げるのはそれぞれ状態行動の回数と PC-PG であり、CTS からの直接の継承は書いていない（出典: [内的報酬の手法の全体像](../../../../reports/survey/intrinsic_reward_landscape_survey_2026-11-18.md)、§3.3）。
- PixelCNN の論文は、Monte Carlo 収益を混ぜた更新を多段の方法と位置づけたが、ゲームによって成績を上げも下げもした → 詳しくは [n-step の収益](nstep_return.md)（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/survey/nstep_action_gap_survey_2026-11-27.md)、§3.4）。
- PixelCNN の論文の著者は、Retrace(λ) を使う Reactor では軌跡を切る性質がボーナスの効きを抑えたと推測した → 詳しくは [n-step の収益](nstep_return.md)（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/survey/nstep_action_gap_survey_2026-11-27.md)、§3.4）。
- Taïga らの Rainbow の上の CTS・PixelCNN の Montezuma の値は、学習の時点ごとの表がある → 詳しくは [内的報酬による探索の比較](intrinsic_novelty.md)（出典: [montezuma_published_scores_survey_2026-11-27.md](../../../../reports/survey/montezuma_published_scores_survey_2026-11-27.md)、§2.2）。
- β-DQN（Zhang ら、AAMAS 2025）は、Replay から状態ごとの行動の頻度（β）を学び、まれな行動を選ぶ被覆の方策、過大評価を確かめる方策、純粋な活用の方策を組にしてエピソードの中で切り替え、どれを使うかをスライディング窓のバンディットで決める。MiniGrid の成功率の平均は DQN 0.49 に対して 0.83、MinAtar の収益の平均は 36.44 に対して 52.44（各 10 seed、5M step）で、状態の被覆は CliffWalk の訪問の分布図で定性的に示しただけである（論文の HTML の前半だけを読んだ）（出典: [novelty_bonus_long_horizon_reach_survey_2026-11-27.md](../../../../reports/survey/novelty_bonus_long_horizon_reach_survey_2026-11-27.md)、§5.5・§9）。
- 回数・擬似回数のボーナスは非定常で最適化しにくいとして、十分統計量を状態に足して定常にする手法がある（要旨だけ読んだ）。 → 詳しくは [内的報酬による探索の比較](intrinsic_novelty.md)（出典: [rnd_dense_reward_reset_collapse_survey_2026-11-27.md](../../../../reports/survey/rnd_dense_reward_reset_collapse_survey_2026-11-27.md)、§4.5）。

## 横断的知見

なし

## 見立て

なし

## 分かっていないこと

なし

## 撤回・更新

なし

## 関連

- [内的報酬の手法比較](intrinsic_novelty.md)。

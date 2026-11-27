---
title: 優先度付き経験再生（PER）
page_kind: 手法
aliases:
  - prioritized experience replay
  - PER
  - TD 誤差
  - 初期優先度
description: TD 誤差で遷移を選ぶと、予測できない誤差にも抽出が偏る。優先度の初期値・更新方法・重要度重みをそろえる必要がある。
updated: 2026-11-27
---
# 優先度付き経験再生（PER）

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [手法](../../index.md#手法)

## 今の結論

TD 誤差で遷移を選ぶと、予測できない誤差にも抽出が偏る。優先度の初期値・更新方法・重要度重みをそろえる必要がある。

## 分かっていること

### anet の知見

- DropMerge の replay ratio 1・固定初期優先度1.0の定常時の速報では、Learner がまだ更新していない遷移が抽選の約0.73、優先度総量の約0.74を占めた。追い出された遷移のうち未抽選だった割合は約0.26だった。初期優先度を下げると未抽選の追い出しが増え、最大優先度で初期化しても未更新の遷移への集中は解消しなかった。（出典: [RFD0132 の旧草案](../../../memo/dropped/0132_actor_priority_per_10rfd.md)、Problem Statement・Further Notes、[正式採番後の RFD0035](../../../memo/done/0035_approx_actor_priority_per_10rfd.md)、問題）。
- anet の `per_initial_priority_mode = max` は、Learner が更新した優先度の最大値で、新しい遷移を初期化する。最大値を押し上げるのは、実際に適用された有限な Learner の優先度だけである。DropMerge の `dmfb_47b_scr_ctl`（2026-11-01。DropMerge の TempoRL（scratch 第 1 弾）と、対照の新旧の差の探索履歴）の実効設定では `fixed`、Atari の Montezuma の `mz_n10_lr3e4_s1`（2026-11-22。Montezuma's Revenge の RND（Taiga ら寄せの構成、β と Munchausen））と LunarLander の `ll_rnd_s3`（2026-11-08。RNDの害の確認とpredictorの周期リセット）の実効設定では `max` だった（出典: [RFD0035](../../../memo/done/0035_approx_actor_priority_per_10rfd.md)、実装上の決定 1・6；[2026-11-01_temporl-and-control-gap.md](../../../experiments/default-dqn/dropmerge/2026-11-01_temporl-and-control-gap.md)、参照 Run；[run_20261101-081421_dmfb_47b_scr_ctl.txt](../../../experiments/default-dqn/dropmerge/config/run_20261101-081421_dmfb_47b_scr_ctl.txt)、`DefaultDQNAgent.learner.per_initial_priority_mode = fixed`；[2026-11-08_montezuma-rnd-taiga-aligned.md](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、目的と比較条件；[run_20261122-174627_mz_n10_lr3e4_s1.txt](../../../experiments/default-dqn/atari/config/run_20261122-174627_mz_n10_lr3e4_s1.txt)、`DefaultDQNAgent.learner.per_initial_priority_mode = max`；[2026-11-08_rnd-harm-check-and-predictor-reset.md](../../../experiments/default-dqn/lunarlander/2026-11-08_rnd-harm-check-and-predictor-reset.md)、Runと実効config；[run_20261108-070626_ll_rnd_s3.txt](../../../experiments/default-dqn/lunarlander/config/run_20261108-070626_ll_rnd_s3.txt)、`DefaultDQNAgent.learner.per_initial_priority_mode = max`）。

### 外の事例

- 確認した BTR の `max_prio` は一度観測した最大値を維持する。大きな TD 誤差を持つ遷移がバッファから消えた後も、新規遷移がその優先度を継承する（出典: [Code Reading: BTR 公開実装](../../../../reports/btr_code_reading_2026-08-31.md)、3. 新規に判明した点「3.1 PER 初期優先度は「単調非減少の過去最高水位」」）。
- 公開実装のコードコメントでは重要度重みの指数に β の代わりに α を使うことが意図しない実装だと記されていた。論文本文の説明だけで再実装するとこの点が一致しない（出典: [Survey: BTR (Beyond The Rainbow)](../../../../reports/btr_hyperparams_survey_2026-08-26.md)、11. 批判・懸念）。
- ReLo の7×7格子では一様乱数の報酬を置いた状態への遷移に PER の抽出が集中し、序盤に本来のゴールを学べなかった。予測できない TD 誤差の大きさと追加学習の価値が一致しない例である（出典: [Survey: PER の優先度の弱点と代替](../../../../reports/per_priority_weaknesses_survey_2026-11-18.md)、2. TD 誤差の優先度の弱点「2.1 確率的な報酬と張り付き」）。
- 優先抽出と重要度重みの組み合わせは目的損失の形にも影響する。LAP は特定の損失の下で、優先抽出した期待勾配が別の損失による一様抽出と一致する条件を示した（出典: [Survey: PER の優先度の弱点と代替](../../../../reports/per_priority_weaknesses_survey_2026-11-18.md)、2. TD 誤差の優先度の弱点「2.3 損失の形との等価性」）。
- 割引率の違う複数の Q を補助として学ぶ多地平の補助タスク（Fedus ら）を PER と組むと、10 ゲーム中 4 ゲーム（Pong、Venture、Gravitar、Zaxxon）で成績が下がった。著者は全 Head の TD 誤差を平均して優先度にしたためと説明し、最大の γ で優先度を付ける案を予備の結果として付録 E に置く（出典: [skip_auxiliary_task_noise_survey_2026-11-27.md](../../../../reports/skip_auxiliary_task_noise_survey_2026-11-27.md)、§1.2）。
- de Bruin ら（JMLR 2018、DDPG）は、TD 誤差で引くと状態空間の全体で近似の精度がそろうので、学習と同じ開始の分布で試すなら望ましくないが、広い状態へ汎化させたいなら重要度の補正を掛けないほうがよいかもしれないと書く（出典: [novelty_bonus_long_horizon_reach_survey_2026-11-27.md](../../../../reports/novelty_bonus_long_horizon_reach_survey_2026-11-27.md)、§5.2）。
- Fu ら（2019、arXiv v1）の Q 学習の単体試験（格子の世界などを含む表形式の 8 領域、2 層の ReLU の NN、正解の Q と比べ、標本の不足は切り離す）では、状態・行動の一様な重み、Replay の重み（学習中のすべての方策の平均）、TD 誤差の重みが、どの NN の大きさでも最も高い収益になり、方策自身の分布（on-policy）は常によいわけではなかった。重みの分布のエントロピーが高いほど収益が高い傾向で、原典は状態の被覆を最大にする探索の動機になると書く（出典: [novelty_bonus_long_horizon_reach_survey_2026-11-27.md](../../../../reports/novelty_bonus_long_horizon_reach_survey_2026-11-27.md)、§5.3）。
- PER の原典は、優先抽出は経験の分布を制御なしに変え、方策と状態の分布が固定でも推定が収束する解を変えるとし、重要度の重みは β = 1 で非一様な抽出を打ち消すと書く。偏りのない更新が要るのは学習の終わりの収束の近くとして、β を学習の終わりに 1 へ届くよう線形に上げる（出典: [rnd_predictor_data_munchausen_survey_2026-11-27.md](../../../../reports/rnd_predictor_data_munchausen_survey_2026-11-27.md)、§4.4）。

## 横断的知見

- 補完: BTR の調査報告は重要度重みの指数に α が使われる点を指摘し、後続のコード読解は β が受け取られても実行中に0へ戻る箇所を特定した。論文上の β を変えるだけで実装の抽出・補正が追従するとはいえない。条件の違い: 前者は論文記載と公開コードの差、後者は固定した版の PER 更新処理を直接確認した（出典: [btr_hyperparams_survey](../../../../reports/btr_hyperparams_survey_2026-08-26.md)、11. 批判・懸念。[btr_code_reading](../../../../reports/btr_code_reading_2026-08-31.md)、3. 新規に判明した点「3.2 `beta` は死にコード」）。

## 見立て

なし

## 分かっていないこと

- KFM の実時間優先の構成では、1 遷移が寿命中に引かれる回数の期待値は約 1.0、4 件に 1 件は一度も引かれずに消え、`per_alpha` 0.2 では優先度が 20 倍でも引かれやすさは 1.8 倍だった。α を上げた試しは、ボス用セット（PER α 0.6 を sticky 0・低い ε・クリップ外しと同時に入れ、30M までボスに 1 度も当てなかった）だけで、DropMerge で `per_alpha` を 0.2 から 0.1 へ弱めた分岐は明確な改善が無かった。優先度を強めると、まれな場面（ボス戦など）の遷移を引く回数が増え、成績に効くか（出典: [0173_intrinsic_reward_novelty_10rfd.md](../../../memo/0173_intrinsic_reward_novelty_10rfd.md)、§2.3；[2026-10-15_kfm-boss-route.md](../../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md)、現時点のまとめ 4；[2026-07-27_longrun-batch-replay.md](../../../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md)、探索ブロック 02。開発者の発言（2026-10-21））。

## 撤回・更新

なし

## 関連

なし

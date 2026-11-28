---
title: Model-based Atariの評価
page_kind: 手法
aliases:
  - SimPLe
  - MuZero
  - EfficientZero
  - Atari100k
  - world model
description: SimPLeの低データ性能の優位は、後に調整されたmodel-freeの結果で相殺された。world model内の学習コストと評価手順への批判が残っている。
updated: 2026-11-27
---
# Model-based Atariの評価

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [手法](../../index.md#手法)

## 今の結論

SimPLeの低データ性能の優位は、後に調整されたmodel-freeの結果で相殺された。world model内の学習コストと評価手順への批判が残っている。

## 分かっていること

### anet の知見

なし

### 外の事例

- SimPLeの低データ性能は後に調整したmodel-freeの結果で相殺され、world model内の学習コストと評価手順の批判が残った。調査は取得できなかった一次資料も明記している。（出典: [simple_model_based_atari_survey_2026-09-24.md](../../../../reports/survey/simple_model_based_atari_survey_2026-09-24.md)、13. 総合評価・14. 調査の限界）。
- SimPLe の著者は実環境の相互作用を減らせる一方、動画予測モデルと方策を学ぶ計算を含めると標準的なモデルフリー学習より実時間が長いことを認めた。実環境のサンプル数と計算費は異なる比較軸である（出典: [Survey: SimPLe(Model-Based Reinforcement Learning for Atari)の受容・批判・後続研究](../../../../reports/survey/simple_model_based_atari_survey_2026-09-24.md)、4. ICLR 2020 の査読「4.3 著者の回答と一般コメント」）。
- 文献の比較条件: Qiao ら（ECAI 2024）は、モデルベースの RL で agent を周期的にリセットすると MBPO の Hopper で成績が下がったとし、代わりに world model をリセットした（MBPO は各 ensemble の最後の 2 つの隠れ層を 2×10^4 step ごと、DreamerV2 は RSSM の遷移の predictor を 2×10^5 step ごと）。DreamerV2 の DeepMind Control 20 本の IQM は 762 から 783 になった。探索のボーナスは扱っていない（出典: [rnd_dense_reward_reset_collapse_survey_2026-11-27.md](../../../../reports/survey/rnd_dense_reward_reset_collapse_survey_2026-11-27.md)、§3.4）。

## 横断的知見

- 補完: SimPLe の100Kは実環境との相互作用回数で、予測モデルを使った学習には別に計算時間がかかる。Atari の速度を比較する際には、FPS が生フレーム数か agent step 数か、環境の実行だけか方策の学習も含むかを確かめる。経験を節約する結果から実時間の短縮は導けない。条件の違い: SimPLe は実環境と予測モデル内の計算、実時間の調査は FPS の単位と学習時間の扱いを区別する（出典: [simple_model_based_atari_survey](../../../../reports/survey/simple_model_based_atari_survey_2026-09-24.md)、4. ICLR 2020 の査読「4.3 著者の回答と一般コメント」。[atari_walltime_frontier_survey](../../../../reports/survey/atari_walltime_frontier_survey_2026-09-09.md)、§1.1・§1.2（frame・agent step と FPS の単位）、§5.1・§5.2（EnvPool と Sample Factory の計測条件））。

## 見立て

なし

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [Agentと学習](../../../design/110_agents_and_learning.jp.md)。
- [atari](../subjects/atari.md)。
- [evaluation](../measurement/evaluation.md)。
- [仕組みと契約](../../../design/200_dqn_agents.jp.md)。

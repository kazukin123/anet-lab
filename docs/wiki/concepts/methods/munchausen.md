---
title: Munchausen RL
page_kind: 手法
aliases:
  - Munchausen
  - M-DQN
  - M-IQN
  - munch
  - IQN+M
  - QR+M
  - log π
  - ma0.7
description: Munchausen は RND を使わない Breakout と DropMerge では系列で最大の単独の上乗せだったが、RND を足した Montezuma では、有効にした Run が Greedy の 2500 点にほとんど届かなかった。Montezuma の比較は組ごとに 1 本で、無効の群は Double DQN も有効にした 2 変数の比較である。
updated: 2026-11-27
---
# Munchausen RL

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [手法](../../index.md#手法)

## 今の結論

Munchausen は RND を使わない Breakout と DropMerge では系列で最大の単独の上乗せだったが、RND を足した Montezuma では、有効にした Run が Greedy の 2500 点にほとんど届かなかった。Montezuma の比較は組ごとに 1 本で、無効の群は Double DQN も有効にした 2 変数の比較である。
DropMerge では、Munchausen の bonus と γ × soft gap を足した 1 step あたりの正味の項は全窓で負で、毎 step の加点が NEET を生むという仮説は否定された。
Munchausen（M-DQN・M-IQN）と RND などの内的報酬を組んだ論文・公開の結果・実装は、2026-11-27 の調査では見つからなかった。
温度 τ は Q 値差を基準に効く。報酬尺度が違う環境への適用では、clip 率と bonus の平均を確かめ、その環境の Run で採否を決める。

## 分かっていること

### anet の知見

- Munchausen RLについて: hard125 基準からの 1 変数（＋強制される Double OFF）で、25-50M 壁突破率が **11.35 / 11.52%** 対 基準 6.00 ± 0.30%（N=4）。**本キャンペーン最大の単一効果**（差 5.435pt / Welch t = 31.2）。`use_double_dqn=false` 単独の対照は無効果なので**寄与はすべて Munchausen**。機序は相対 action gap の 3.6 倍化。**2026-09-06 に 100M へ延長し、95-100M で 18.68% / `≥600` 8.66% と最終窓が最高値**（50M の 1.62 倍）。**旧 net では RR4 と併用できなかった**（8.85%）が、**BTR NN構成の下では RR4 + Munchausen が最良構成になっている**（25-48M 15.77%）ので、この制約は NN構成に条件付きだった（[2026-09-01](../../../experiments/default-dqn/atari/2026-09-01_btr-feedback-arms.md) 探索ブロック 28・29・31、[2026-09-12](../../../experiments/default-dqn/atari/2026-09-12_btr-network-port.md) 探索ブロック 01）（出典: [2026-09-01_btr-feedback-arms.md](../../../experiments/default-dqn/atari/2026-09-01_btr-feedback-arms.md) / [2026-09-12_btr-network-port.md](../../../experiments/default-dqn/atari/2026-09-12_btr-network-port.md) / [2026-09-08_replay-ratio-mechanism.md](../../../experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md)、現時点のまとめ・対応する探索ブロック）。
- Munchausen（τ0.03、α0.9、clip −1、Double DQN OFF）について: IQNの100M checkpointの評価専用Runで、Eval1 1,524→1,796（+18%）、Greedy 1,572→1,831（+16%）。2 seedの差は1〜2%。QR51ではGreedyのNEETは消えたが、UQEでの利得はseedで+4% / −7%に割れた（[Atari からのフィードバック](../../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md) 探索ブロック01・04）（出典: [2026-10-01_atari-feedback.md](../../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ・対応する探索ブロック）。
- Munchausen α 0.7（B1024/RR1の230Mから）について: 次状態のエントロピーが0.835→0.935に上がり、Qの1位と2位の差は−13%、NoisyNetで行動が変わる割合は8.6%→11.0%、Qの水準は+4.6%になったが、60Mでは評価に差が出なかった（評価専用Runで−0.6%、約1.4σ）（出典: [2026-10-15_gradient-noise-batch.md](../../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、探索ブロック15 探索結果・考察。Run: run_20261022-064721_dmfb_45e_42j_ma0.7）。
- Munchausen の bonus と γ × soft gap を足した 1 step あたりの正味の項は、soft gap を記録した 3 本で 90–100M に −0.037〜−0.045 で、全窓で負だった。「毎 step の加点で生き延びるほど得になり NEET を生む」という仮説は否定された。当時の分析の −0.125〜−0.004 は学習ステップの窓で切った値で、上は経験数の窓で引き直した値である（出典: [2026-10-01_atari-feedback.md](../../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、探索ブロック04 探索結果・考察・判断更新。Run は探索ブロック04の4本: run_20261003-000810_dmfb_munch / run_20261003-235455_dmfb_munch_s2 / run_20261003-081944_dmfb_qr_munch / run_20261003-160717_dmfb_qr_munch_s2）。
- Munchausen の温度 τ は Q 値差を基準に効く。報酬尺度が違う環境への適用では 36_agent_munchausen の 03_clip_ratio と 04_bonus_mean を確認し、その環境の Run で採否を決める（出典: [DefaultDQNAgent ハイパーパラメータ探索](../../../experiments/default-dqn/README.md)、Munchausenの温度と別Envへの適用）。

### 外の事例

- Munchausen の論文は、極限で action gap が (1+α)/(1−α) 倍になることを示し、温度 τ → 0 で Advantage Learning に帰着するとした。M-DQN は n-step を使わず、M-IQN は 3-step の収益の IQN と組み合わせ、1-step の M-IQN も僅差で Rainbow を上回り、3-step の M-IQN は 1-step をはっきり上回った。n ごとの action gap は報告していない（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/nstep_action_gap_survey_2026-11-27.md)、§4.2）。
- Munchausen の論文が Asterix で測った action gap は、M-DQN と Advantage Learning がどちらも Adam の DQN より大きかった（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/nstep_action_gap_survey_2026-11-27.md)、§4.2）。
- Munchausen の論文の脚注 5 は 1-step の収益を使うと書き、M-IQN を 3-step の収益の IQN と組み合わせたという本文の記述と食い違う（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/nstep_action_gap_survey_2026-11-27.md)、§4.2・§5）。
- Munchausen の原典は、softmax の方策による探索は Q の尺度に依存するとして、主な結果では ε-greedy で行動した。Enduro では Q が 0 に近い序盤に τ 0.03 の softmax がほぼ一様になり、報酬をほとんど得られずに学べなかった。既定値は α 0.9・τ 0.03・l0 −1 で、少数の Atari のゲームで調整した（出典: [rnd_predictor_data_munchausen_survey_2026-11-27.md](../../../../reports/rnd_predictor_data_munchausen_survey_2026-11-27.md)、§5.2）。
- Munchausen の原典の表 3（学習中の最大点、100 エピソード・3 seed の平均）では、Montezuma's Revenge は M-DQN 0・M-IQN 0 だった。主な実験は 200M フレーム・3 seed・sticky actions で、原典は内的報酬や難探索のゲームを論じない（出典: [rnd_predictor_data_munchausen_survey_2026-11-27.md](../../../../reports/rnd_predictor_data_munchausen_survey_2026-11-27.md)、§5.2）。
- BTR の Phoenix の表（200M フレーム、3 seeds）の action gap は、BTR 0.282、Munchausen なし 0.055 で、n-step を変えた列は無い（出典: [btr_hyperparams_survey_2026-08-26.md](../../../../reports/btr_hyperparams_survey_2026-08-26.md)、9.2 Table 2 (Phoenix)；[nstep_action_gap_survey_2026-11-27.md](../../../../reports/nstep_action_gap_survey_2026-11-27.md)、§4.2）。
- 調査の範囲: Munchausen（M-DQN・M-IQN）と RND などの内的報酬を組んだ論文・公開の結果・実装と、組んで悪くなる・良くなる理由を論じた文献は、2026-11-27 の調査では見つからなかった。離散行動のエントロピー正則化 Q 学習と内的報酬を Atari の難探索で組んだ報告も無く、Agent57 と MEME の読めた範囲にも Munchausen の記述は無い（探した範囲: Munchausen RL の本文と付録、Agent57・MEME、Munchausen と RND・内的報酬・難探索、エントロピー正則化 Q 学習と RND の web 検索）（出典: [rnd_predictor_data_munchausen_survey_2026-11-27.md](../../../../reports/rnd_predictor_data_munchausen_survey_2026-11-27.md)、§2・§5.1）。

## 横断的知見

- 対立点: Munchausen は RND を使わない Breakout と DropMerge では系列で最大の単独の上乗せだったが、RND を足した Montezuma では、有効にした Run が Greedy の 2500 点にほとんど届かなかった。Breakout（RR1・hard C=125、50M）の 25-50M の壁突破率は 11.35% / 11.52% で、基準は 6.000 ± 0.304（N=4）、Double DQN だけを切った対照は 6.33% だった。DropMerge は IQN の 100M checkpoint の評価専用 Run で Greedy が 1,572 → 1,831 だった。Montezuma（Taiga ら寄せ、β 0.01、seed 1）は 20.2〜29.2M の Greedy の 2500 点が無効 99/160、有効 5/160 で、無効の構成から作った n-step 10 の seed 2 は 24.2M 以降の Greedy が 5,662.5〜6,600.0 だった。BTR の原典は Impala の効果が最大（+142% IQM）で、Munchausen と IQN は Phoenix のような精密な制御に強く効くと書く（Phoenix で外すと ε=0 の点が 330k から 184k）。条件の違い: Montezuma の無効の群は Double DQN も有効にした 2 変数の比較で、組ごとに 1 本である。Breakout と DropMerge に RND は無く、同じ題材で RND の有無と Munchausen の有無を組んで比べた Run は無い（出典: [Atari Breakout: BTR ソース差分の群別スクリーニング — 2026-09-01](../../../experiments/default-dqn/atari/2026-09-01_btr-feedback-arms.md)、現時点のまとめ 39・40・探索ブロック 28；[DropMerge への Atari 知見のフィードバック実験（FB シリーズ）の探索履歴](../../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ 1；[Montezuma's Revenge の RND（Taiga ら寄せの構成、β と Munchausen）](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 1・38；[Survey: BTR (Beyond The Rainbow) — ハイパーパラメータと Atari 評価プロトコル](../../../../reports/btr_hyperparams_survey_2026-08-26.md)、§9.1・§9.2）。

## 見立て

- Montezuma の RND で Munchausen を有効にすると Greedy の 2500 点にほとんど届かなかったのは、外的報酬も内的報酬もほぼ 0 の間、Munchausen の soft な bootstrap（方策で重み付けした Q − τ log π）が game over で切られるまで価値を足し続け、死なずに長く続けることを学ぶ向きに働いたためではないか。その効きは内的報酬より桁で大きく、RND の新しさの向きを覆ったのではないか。 状態: 未検証
  - 支える事実: (1) Taiga らの式の RND で Munchausen 有効の Run は、Q の最大の平均が 0.11（0〜0.5M）から 0.80（3.5M〜）へ上がり、3M 台に終わった 49 本はすべて打ち切りだった。同じ式で Munchausen 無効・Double DQN 有効の Run の Q の最大の平均は、0.5M 窓で 0.005〜0.012 だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せの構成、β と Munchausen）](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 5・探索ブロック 08）。 (2) β 0.01・Munchausen 有効の Run は、16〜28M に学習側の終わったゲームの 74〜89% が打ち切りで、β 0.01・Munchausen 無効の seed 1 は 16〜30M に 4〜11%、seed 2 は 24〜30M に 2〜4% だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せの構成、β と Munchausen）](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 8）。 (3) β 0.01・Munchausen 有効の Run の 12〜16M の 4M 窓の平均は、Munchausen の項（`36_agent_munchausen/04_bonus_mean`）が −0.130、内的報酬が 1.6e-03 だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せの構成、β と Munchausen）](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、探索ブロック 09）。 (4) Munchausen の bootstrap は、行動の最大値だけでなく方策で重み付けした `Q − τ log π` を使い、報酬へ足す `α τ log π(a|s)` は一般に非正である（出典: [DQN系Agent](../../../design/200_dqn_agents.jp.md)、§2.1.7）。
  - 合わない事実: (1) β 0.1 では Munchausen 無効・Double DQN 有効の Run も、Greedy の 2500 点が 22.2M まで 0 本だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せの構成、β と Munchausen）](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 1）。 (2) Munchausen 無効で warm-up 200,000 の既定の Run でも、10〜12M の学習側の打ち切りは seed 1 が 49 / 123、seed 2 が 62 / 102 だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せの構成、β と Munchausen）](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、探索ブロック 09）。 (3) Munchausen 無効・β 0.01 の seed 2 の Greedy は、10.2〜29.2M の 320 ゲームのうち打ち切りでないのが 4 本だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せの構成、β と Munchausen）](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 10）。 (4) Taiga らの式で Munchausen 無効の Run は、learn step 6 万前後で probe の dead 率 0.94・`04_srank` 53 に崩れていた（出典: [Montezuma's Revenge の RND（Taiga ら寄せの構成、β と Munchausen）](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 5）。
  - 条件の違い: Munchausen の有効・無効を比べた組は、どれも seed 1 の 1 本どうしで（無効の側には seed 2 の b001_s2 もある）、Munchausen 無効の群は Double DQN を有効にした 2 変数の比較である。Taiga らの式の 2 本は 3.87M と 2.53M で止めた。記録にある Munchausen の値は報酬へ足す項で、bootstrap の中の τ log π の分の大きさは記録に無い。RND を切って Munchausen の有無だけを比べた Run は無い。
  - 確かめ方: 今ある Run で、β 0.1 と β 0.01 の Munchausen 有効・無効の 4 本の学習側のゲームの長さ、失った命の数、`37_agent_qtd/11_q_max_mean` を同じ 2M 窓で並べる（今ある Run の集計）。分けるには、RND を切って Munchausen の有無だけを変えた Montezuma の Run が要る（新しい Run が要る）。
  - 出どころ: AI（開発者の質問を起点、2026-11-10）

## 分かっていないこと

- ε ラダーのランダムな DROP は、Munchausen の bonus で大きく負の値を受けて DROP 側の Q を押し下げ、NEET の入口になっているか（出典: [2026-10-01_atari-feedback.md](../../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、探索ブロック08 考察。候補の機序として挙げられ、検証されていない。Munchausen 無しの QR51 でも NEET が出たので唯一の機序ではない）？

## 撤回・更新

なし

## 関連

- [Breakout](../subjects/breakout.md)。
- [DropMerge](../subjects/dropmerge.md)。
- [Montezuma's Revenge](../subjects/montezuma_revenge.md)。
- [Phoenix](../subjects/phoenix.md)。
- [内的報酬による探索の比較](intrinsic_novelty.md)。
- [報酬尺度と割引率](reward_scale_discount.md)。
- [NoisyNetの観測](noisynet.md)。
- [行動の時間尺度](temporal_abstraction.md)（action gap を広げる作用素と、TempoRL の有無による log π の clip 率）。
- [n-step の収益](nstep_return.md)。
- 仕組み: [DQN系Agent](../../../design/200_dqn_agents.jp.md) §2.1.7 Munchausen RL。

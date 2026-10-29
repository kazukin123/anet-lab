# NoisyNetの観測

## 今の結論

LunarLanderの3 seedではNoisyNetの成績差はseed幅内で、行動変化率は終盤に2〜4%へ縮んだ。DropMergeではεラダーへの上乗せではNEETが残り、NoisyNetへの置換で抑えられた。BTRの公開実装はNoisyNetを切ってもBreakoutで論文と互角だった。Atariの実時間優先50MでεラダーをNoisyNetに置き換えると7ゲームとも基準を下回り、BreakoutではNoisyNet自体の差は±0でラダーを外したぶんだった。qbertの50M終了checkpoint（seed1）を評価時だけノイズありにすると、スキップなしでは8面突破率が上がり、TempoRL分割のfullでは下がった。

## 分かっていること

- LunarLanderでは初期の行動変化率が大きく、終盤は2〜4%へ縮んだ。成績差は3-seedの幅内で、有効・無効を確定していない。（出典: [2026-10-06_noisynet.md](../../experiments/default-dqn/lunarlander/2026-10-06_noisynet.md)、現時点のまとめ）。
- DropMergeではεラダーへの上乗せと置換の結果が異なり、置換した構成でNEETが抑えられた。（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ6。Run: run_20261008-015339_dmfb_21_nzrep）。
- batch増加腕は同じσでもQの変化と行動の食い違いが小さく、学習側の上乗せにはノイズ感度の違いが混ざった。（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ5）。
- DropMerge の停滞区間で B を上げずに学習率を 7.1e-5 に下げた腕は、NoisyNet で行動が変わる割合がほとんど下がらず（10.0%、対照 10.5%）、学習側の上乗せ（+1.4〜1.7%）が学習中の評価の上乗せ（+1.2〜1.8%）とほぼ同じだった。B を上げた腕のように学習側が評価より大きく上がる形ではなかった（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、探索ブロック14 探索結果・考察。Run: run_20261019-184443_dmfb_42b_34a_lr7.1e-5）。
- DropMerge の B1024/RR1 で Munchausen α を 0.9 から 0.7 に下げると、同じ σ のまま NoisyNet で行動が変わる割合が 8.6% から 11.0% に上がり、次状態の方策のエントロピーも 0.835 から 0.935 に上がったが、60M では評価に差が出なかった（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、探索ブロック15 探索結果・考察。Run: run_20261022-064721_dmfb_45e_42j_ma0.7）。
- LunarLander の学習開始時（learn step 0）の probe では、ノイズだけを変えると greedy 行動の 71〜74% が μ-only と食い違い、行動のエントロピーは 1.33〜1.37 で 4 行動の上限 ln 4 ≈ 1.386 に近かった。この大きな効きは、Head の V/A 射影を orthogonal・gain 0.01 で初期化するので射影の ‖σ‖/‖μ‖ が σ0/gain = 50 になることと整合し、readout 群全体の初期値 0.635〜0.640 は計算値 0.637 と一致した（出典: [2026-10-06_noisynet.md](../../experiments/default-dqn/lunarlander/2026-10-06_noisynet.md)、現時点のまとめ6・7）。
- BTR の train の ε は、時定数 2M 遷移の指数減衰で 1.0 から 0.01 へ下がる全 env 共通の 1 つの値で、25M 遷移で切られて NoisyNet だけになる。評価の ε は 31.5M ごろまで 0.01、以降 0 である（出典: [2026-09-27_btr-gap-qbert-bz.md](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、探索ブロック01 探索観点；[2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、探索ブロック03「BTR の公開 CSV に対して」）。
- BTR は ε を module の buffer に持ち、choose_action ごとに online net を、learn_call ごとに target net を引き直す。online net は学習時に引き直さない。hard update は ε の buffer も複製するので、置換 step では target の ε が online の ε になる（出典: [084_noisynet_10prd.md](../../memo/084_noisynet_10prd.md)、§2）。
- BTR の公開実装を NoisyNet を切って（--noisy 0）Breakout で走らせると、41–91M frames の評価の平均は 504.5 で、論文の NoisyNet あり（490.3）と互角だった（出典: [2026-09-08_replay-ratio-mechanism.md](../../experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md)、探索ブロック10「NoisyNet を切っても論文と互角」。Run: run_20260911-095808_btr_noisyoff_wsl（BTR の出力で anet-lab の Run ではない））。
- Atari の実時間優先 50M で train の ε ラダーを NoisyNet に置き換える（ε 0）と、eval2 40-50M は 7 ゲームとも基準のどの seed も下回った（Breakout 470.1 対 546.0、battle_zone 53,335 対 107,497 / 135,890、double_dunk −23.9 対 17.1 / 20.5 など）（出典: [2026-10-11_noisynet-realtime.md](../../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md)、現時点のまとめ1・探索ブロック 01。Run: `run_20261011-182123_breakout_rr1_50m_nz_eps0` ほか同記録の 7 本）。
- Breakout の 2 × 2 では、ε ラダーを外すと NoisyNet の有無によらず eval2 25-50M が約 70〜85 点下がり、NoisyNet の有無の差はラダーありで +15.3、ラダーなしで +0.8 だった（出典: [2026-10-11_noisynet-realtime.md](../../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md)、現時点のまとめ2・探索ブロック 02。Run: `run_20261014-121541_breakout_rr1_50m_nz_ladder`, `run_20261014-194942_breakout_rr1_50m_eps0`, `run_20261011-182123_breakout_rr1_50m_nz_eps0`, `run_20260919-235619_rr1_va_hard125_cap2m_100m`）。
- NoisyNet を ε ラダーに足すと、battle_zone と double_dunk では成績が戻ったが、立ち上がりが基準より約 10M・約 20M 遅れた（battle_zone の eval2 40-50M 100,955、double_dunk は eval2 が正になるのが 40-45M）（出典: [2026-10-11_noisynet-realtime.md](../../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md)、現時点のまとめ3・探索ブロック 02。Run: `run_20261015-030627_a5_battle_zone_rr1_50m_nz_ladder`, `run_20261015-110003_a5_double_dunk_rr1_50m_nz_ladder`）。
- Atari の probe では、ノイズで greedy 行動が μ-only と食い違う割合は学習開始時 27〜71% で、40-50M の平均は 0.0〜4.9%（ε 0 の 7 本）だった。readout weight の ‖σ‖/‖μ‖ は 0.869 から 40-50M の平均で 0.48〜0.81 になった。double_dunk の ε 0 は 5M 以降ノイズが行動を 1 度も変えず、相転移も起きなかった（出典: [2026-10-11_noisynet-realtime.md](../../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md)、現時点のまとめ4・5、探索ブロック 03。Run: `run_20261013-012055_a5_double_dunk_rr1_50m_nz_eps0` ほか同記録の Run）。
- TempoRLを有効にしたDefaultDQNでは、skipのHeadの射影がQ側の`head.projection`のNoisyNetの設定を引き継ぎ、Actorのskip側（スキップ価値）のforwardはQ側（行動価値関数）と同じNN実行設定で動く（乱数のseedは別）。qbertの分割（50M、seed1）の45〜50MでLearnerのprobeがノイズでQ側の選ぶ行動を変える割合は0.038で、スキップなし（0.019）の2倍だった。skip側の割合は測っていない（出典: [106_temporl_skip_forward_noise_key_10prd.md](../../memo/106_temporl_skip_forward_noise_key_10prd.md)、§2.1・§2.2の3。Run: `run_20261028-215839_a5_qb_aln_t103spl_s1` / `run_20261025-231207_a5_qb_aln_btreps_s1`）。
- qbertのサンプル効率優先の50M終了checkpoint（seed1）を、評価時だけノイズあり（`nn_forward_key=act_sample`、τは固定）にして各200ゲーム評価すると、8面突破率はスキップなしでGreedyの50.5%から61.5%に上がり、TempoRL分割のfullでは66.0%から41.5%に下がった。分割のsingleは68.5%と66.5%で同じ水準だった（出典: [qbertの行動固定を評価時だけ外す比較](../../experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md)、探索ブロック02。Run: `run_20261029-112021_evnt_off_a` / `run_20261029-113339_evnt_off_n` / `run_20261029-111656_evnt_spl_a_c` / `run_20261029-112623_evnt_spl_n_c` / `run_20261029-120441_evnt_spl_a_s` / `run_20261029-115448_evnt_spl_n_s`）。

## 分かっていないこと

- 学習のActorのlaneの一部をNoisyNetのノイズの弱い（無い）動き方にすると、Greedyの評価で同じ得点で終わる決まった失敗の道が減るか（Greedyの道の状態がreplayに少ないために直らない、という見立て）（出典: [PRD107](../../memo/107_noisy_gain_lane_ladder_10prd.md)、§1・§3）？
- battle_zone と double_dunk で、NoisyNet を ε ラダーに足すと立ち上がりが基準より遅れるのはなぜか（出典: [2026-10-11_noisynet-realtime.md](../../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md)、pending）？
- double_dunk の ε 0 で、ノイズが行動を変えなくなったのに ‖σ‖/‖μ‖ が 0.81 に留まるのはなぜか（出典: [2026-10-11_noisynet-realtime.md](../../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md)、pending）？

## 撤回・更新

なし。

## 関連

- [ニューラルネットワーク](../../design/130_neural_networks.jp.md)。
- [NN Forward の設定解釈を事前準備へ移し、用途別実行オブジェクトが State と Network の寿命を持つ](../../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md)。
- [NoisyNet の ε は module に持たせず、呼び出し側所有の NN 実行状態と NN 実行設定キーで駆動する](../../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md)。
- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。
- [学習の Actor の lane ごとに NoisyNet のノイズの強さを変える暫定 PRD](../../memo/107_noisy_gain_lane_ladder_10prd.md)。
- [TempoRL の skip 側の forward のノイズを Q 側と分けて指定する PRD](../../memo/106_temporl_skip_forward_noise_key_10prd.md)。

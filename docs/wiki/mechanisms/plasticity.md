# 可塑性指標と成績

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機構](../index.md#機構)

## 今の結論

同じ勾配step付近の摩耗はreplay ratio間で似ていたが、付随する新規データ量と成績の推移は異なった。崩壊時に膨らんだのはfeature側の重みで、readout側ではなかった。SiLUやLN512で機序指標が改善しても、RR8短期screeningの成績は改善しなかった。MontezumaのRNDで価値を外的と内的の2つのHeadに分けたRunは、分けない2本よりprobeのdormant率とdead率が高かった。そこへ内的の割引率の分離と損失ごとの更新を足したRunは、dormant率とdead率がさらに高いまま、Greedyで2500点以上にそろった。同じ割引率の分離で損失を合わせて1つのoptimizerで更新した形は、別々に更新した形より、2 seedともprobeのdormant率とdead率が低かった。

## 分かっていること

- 同勾配step付近の損傷はRR間で似たが、付随する新規データ量は異なった。dead ratioは設定内の転換点と、設定間の水準比較で意味が違う。（出典: [2026-08-28_plasticity.md](../../experiments/default-dqn/atari/2026-08-28_plasticity.md)、現時点のまとめ。Run: run_20260828-062103_plasticity_rr8_breakout）。
- 設定修正後も同勾配座標の摩耗と、RRごとの崩壊・天井・上昇の違いが見られた。（出典: [2026-08-29_bundle-fix-rebaseline.md](../../experiments/default-dqn/atari/2026-08-29_bundle-fix-rebaseline.md)、現時点のまとめ。Run: run_20260829-190614_a5_breakout_rr4）。
- SiLUやLN512で機序指標を改善できても、RR8短期screeningでscore改善は得られなかった。q_gapの定常値が動かないことから因果を導く主張は撤回済みである。（出典: [2026-08-30_protection-screening.md](../../experiments/default-dqn/atari/2026-08-30_protection-screening.md)、現時点のまとめ1〜5）。
- 全層spectral_capは重みの実効ノルムを抑えて発散しなかったが、RR4の成績は動かなかった。（出典: [2026-09-08_replay-ratio-mechanism.md](../../experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md)、現時点のまとめ6・7。Run: run_20260908-073326_rr4_capall_munch）。
- Breakout の RR8・5M では、崩壊（eval ピーク後の低下、q_gap −57%）と同期する srank の低下や dormant の増加は無かった。重みを feature 側と readout 側に分けて測ると、膨らんだのは feature 側（RR8・5M で 1.47 倍）で readout 側は 1.10 倍、成長比は全腕で 4.2〜5.6 倍だった（出典: [0063_plasticity_weight_norm_10rfd.md](../../memo/done/0063_plasticity_weight_norm_10rfd.md)、Context（背景・目的）、初回検証 Run run_20260828-000954_plasticity_rr8_breakout；[2026-08-28_plasticity.md](../../experiments/default-dqn/atari/2026-08-28_plasticity.md)、現時点のまとめ3・探索ブロック01）。
- サンプル効率優先の Atari-5 では、readout の重みノルムが 5 本すべてで増え続けた（5-10M → 45-50M で battle_zone 47 → 80、qbert 70 → 140 など）。実時間優先は 34〜41 でほぼ横ばいだった（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、探索ブロック02「内部指標」）。
- MontezumaのTaigaら寄せの構成＋RND（seed 1、30M、learn step約94万）で、probeのdormant率はβ 0.01・Munchausen無効のRunでlearn step 15万〜30万の0.32から90万〜94万の0.54へ上がり続け、β 0.1・Munchausen有効のRunでは同じ範囲で0.19〜0.34だった。前者はGreedyで2500点に届き、後者は届かなかった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ7・探索ブロック07。Run: run_20261110-003002_mz_tg_nmdd_b001_s1、run_20261108-232006_mz_tg_rnd_s1）。
- 同じ構成でRNDをTaigaらの式（β 5e-05）にし、Munchausen無効・Double DQN有効にしたRunは、learn step 2万ごとの窓でprobeのdormant率が0.73から0.96、dead率が0.61から0.94、`04_srank`が211から53、`03_feature_norm`が0.35から0.06になり、Qの最大の平均は0.005〜0.012で平らだった（learn step約8万、2.53Mで止めた）（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、探索ブロック08。Run: run_20261109-221347_mz_tg_taiga_nmdd_s1）。
- 同じ構成のRFD0123の既定（β 0.01・Munchausen無効）で、価値を外的と内的の2つのHeadに分けたRun（seed 1）は、4M窓のprobeのdormant率が16M以降0.46〜0.50、dead率が0.24〜0.25だった。分けない2本（seed 1・2）は同じ窓でdormant率0.30〜0.35、dead率0.11〜0.15。分けたRunはGreedyで30Mまで500点を超えず、分けない2本は26.2M・29.2Mに2500点以上に届いた（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ12・15・23・探索ブロック18。Run: run_20261115-055326_mz_tg_r121_vd_s1、run_20261112-043034_mz_tg_r123_b001_s1、run_20261112-222903_mz_tg_r123_b001_s2）。
- 同じ分解のRunの内的の割引率を0.99に分けてgame overで切らず、外的と内的の損失を別々のAdamで更新したRun（seed 1、probeは5,000 learn stepごと。分解だけのRunは500）は、4M窓のprobeのdormant率が0.59〜0.70、dead率が0.38〜0.52、srankが730〜982だった。分解だけのRunは同じ窓で0.30〜0.50・0.12〜0.25・990〜1,405、分解なしのseed 1はdormant率0.18〜0.35・dead率0.06〜0.15。このRunはGreedyで15.2Mに2500点に届き、20.2〜29.2Mの`53_evalg`で146/160が2500点以上だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ25・28・探索ブロック18・20。Run: run_20261116-005336_mz_tg_g99ne_split_s1）。
- 同じ割引率の分離のまま外的と内的の損失を合わせて1つのoptimizerで更新したRunと、2つの形をseed 2で流したRunを足すと、20〜26Mと26〜30Mの平均で、probeのdormant率は別々の形が0.64・0.62（seed 1）と0.55・0.56（seed 2）、合わせた形が0.48・0.50と0.44・0.45、dead率は別々の形が0.40・0.38と0.33・0.34、合わせた形が0.27・0.27と0.21・0.23だった。20〜30MのGreedy（320ゲーム）の平均は、合わせた形が3,160.3・2,790.9、別々の形が2,394.4・2,265.3だった。seed 2の2本は実行体のbuildがseed 1の2本と違う。合わせた形のreplayの容量を2倍にしたRunは、dormant率0.41・0.48、dead率0.19・0.22で、Greedyの平均は1,919.4だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ29・33・探索ブロック20・21。Run: run_20261116-173557_mz_tg_r121_g99ne_s1、run_20261117-142732_mz_tg_r121_g99ne_s2、run_20261116-005336_mz_tg_g99ne_split_s1、run_20261118-000153_mz_tg_g99ne_split_s2、run_20261118-111145_mz_tg_g99ne_cap4m_s1）。

## 分かっていないこと

なし

## 撤回・更新

- 2026-09-08: 全層でノルムを抑えると発散するため両立点が無い → spectral_capの当該構成は抑制と非発散を両立したがscoreは改善しなかった（出典: [RR機序](../../experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md)、現時点のまとめ6・7）。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。
- [可塑性メトリクスは実 forward の branch capture と学習経路外の部分 forward の 2 チャネルで測る](../../adr/0031-plasticity-metrics-out-of-band-partial-forward.md)。
- [Montezuma's Revenge の RND（Taiga ら寄せの構成、β と Munchausen）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)。

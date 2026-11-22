# Replay ratioとlane窓

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機構](../index.md#機構)

## 今の結論

Breakoutの該当構成ではlane窓16,384の2点がε=0で685.12・685.30、窓8,192が615.73だった。効果の機序は未測定である。MontezumaのTaigaら寄せ＋RNDでlane窓を16,384から32,768へ広げたRun（1本、実行体のbuildも違う）は、Greedyの20〜30Mの平均が3,160から1,919へ下がり、終盤にεの小さいlaneのゲームが長く続いた。同じMontezumaの構成でreplay ratioを8から4にした2本（seed 1・2）は、30Mまで2500点の道に留まり、Greedyで剣で敵を倒すゲームが半数になった最初の回が26.2M・18.2M（replay ratio 8の2本は18.2M・18.2M）で、replay ratio 4のままn-stepを10にした2本は12.2M・14.2Mだった。

## 分かっていること

- Breakoutの該当構成ではlane窓16,384の2点はε=0で685.12と685.30、8,192は615.73だった。更新数だけの説明は3点を追わず、機序は未測である。（出典: [2026-09-15_replay-lane-window.md](../../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md)、現時点のまとめ1〜6。Run: run_20260914-231706_rr4_btrfull_envs128_50m / run_20260915-203706_rr4_btrfull_envs128_cap2m_50m）。
- B512で更新頻度を揃えたRR4はRR4/B256と同水準で、RR2はそれらより良かった。この条件では勾配頻度より経験再利用回数に成績が追従した。（出典: [2026-09-08_replay-ratio-mechanism.md](../../experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md)、現時点のまとめ1〜3。Run: run_20260909-142738_rr4_b512_munch / run_20260908-201708_rr2_munch）。
- PQLの連続制御での容量比例不要という結果と、Atariのlane窓の観測を同一条件の反証として扱わない。（出典: [replay_lane_window_survey_2026-09-17.md](../../../reports/replay_lane_window_survey_2026-09-17.md)、2. 分散アクター構成における容量設計）。
- Breakout の lane 窓 12,288 の Run は exp_step 27,092,992 で bad_alloc により止まった。確保地点は、当時 Sample() のたびに capacity サイズで行われていた ValidIndexManager::GetValidIndices1D の vector の reserve で、要求サイズ 12,582,951 byte は replay_capacity 1,572,864 × 8 + 39 と一致した。全 workspace の bad_alloc 停止 9 本のうち、サイズとスタックが残っているのはこの 1 本だけだった（出典: [2026-09-15_replay-lane-window.md](../../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md)、付随: bad_alloc の確保地点。Run: run_20260919-190647_rr4_btrsn12_envs128_cap1536k_50m）。
- MontezumaのTaigaら寄せ＋RND（価値の分解、内的の割引率の分離、損失を合わせて更新、128 env、seed 1、30M）で、replayの容量を2,097,152から4,194,304（lane窓16,384から32,768）にしたRunは、20〜30MのGreedy（320ゲーム）の平均が1,919.4（窓16,384のRunは3,160.3）、ε=0.01の評価は5回で2,356.0（同3,574.0）だった。Greedyで2500点以上が初めて出たのは21.2M（同15.2M）。26.5〜30Mにはε 0.01〜0.025とε 0.025〜0.16のlaneの1ゲームの長さの平均が15,218・10,458 step（同2,080・1,553）で、28M以後に1本もゲームを終えなかったlaneが128本中54本あった。設定のdumpの差は容量だけだが実行体のbuildも違い、どちらも1本である（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ33・34・探索ブロック20・21。Run: run_20261118-111145_mz_tg_g99ne_cap4m_s1、run_20261116-173557_mz_tg_r121_g99ne_s1）。
- 同じMontezumaの構成（損失を合わせて更新、lane窓16,384）でreplay ratioを8から4にし、targetの更新と評価・probeの間隔をexp stepで揃えた2本（seed 1・2、2026-11-19のbuild）は、Greedy（`53_evalg`）で剣で敵を倒すゲームが16本中8本以上になった最初の回が26.2M・18.2M（replay ratio 8の2本は同じ数え方で18.2M・18.2M）で、30Mまで学習側も評価も2500点の道に留まった（replay ratio 8の2本は学習側で3500点の道へ移った）。replay ratio 4のままn-stepを3から10にした2本（2026-11-21のbuild）は12.2M・14.2Mだった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ29・36・37・探索ブロック23。Run: run_20261119-144336_mz_tg_g99ne_rr4_s1、run_20261119-202718_mz_tg_g99ne_rr4_s2、run_20261121-211321_mz_g99ne_n10_rr4_s1、run_20261122-030405_mz_g99ne_n10_rr4_s2）。

## 分かっていないこと

- Breakoutでlane窓を伸ばして成績が変わった機序は何か（出典: [2026-09-15のcampaign](../../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md)、現時点のまとめ2・探索ブロック02「何が未測か」）？
- lane 窓の下側の崖（Breakout では 4,096 で質的に破綻し、12,288 は破綻しない）は、他のゲームでも同じか（出典: [2026-09-21_realtime-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、pending「lane 窓の下側」。他ゲームでは 1 本も測っていない）？
- Montezumaでlane窓を32,768へ広げたRunの遅れと終盤の長いゲームは、窓の効果か、Runごとのぶれか、実行体のbuildの差か。1本だけで、分けていない（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ33）？

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [Montezuma's Revenge](../subjects/montezuma_revenge.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。

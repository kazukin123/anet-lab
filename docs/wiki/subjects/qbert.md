# Qbert

## 今の結論

低εラダーの64 env・容量1Mでは、45–50Mの第8面突破率（29,000点以上かつround_clear≥8）が4 seedで45.3・32.5・0.7・0.1%だった。容量だけ・env数だけを変えた腕は1.0%・0.1%に留まり、初突破が早い腕でも突破が定着しない例があった。seed 4 は得点ループに入り、得点だけで数えた突破率は第8面の突破を表さなかった。

TempoRLの50M終了checkpoint（合算・遮断・重み0.3、各seed1）は、μ-only評価で行動の固定を外すとすべて平均点が上がった。gの点推定は3.82・0.77・0.96で、PRD104の規則では評価時の固定が主因と読める。学習中の遅れの原因や別seedへの一般化は未確認。分割のcheckpointは、評価時にNoisyNetのノイズやτのランダムさを入れるとfullでだけ突破率が下がった。スキップなしは2 seedとも、Greedyの突破率が学習中と同じ選び方より低かった。

## 分かっていること

- サンプル効率優先の50Mの合算のcheckpoint（seed1）を、`[greedy]`の評価sessionと同じ条件のEvaluation Traceで見ると（3ゲーム）、最後でない残機を失った前後で`V_s`はほとんど変わらず（4回で−0.4〜+0.0）、最後の1機のときだけ0へ崩れた。最上段から外へ出る飛び降りが、価値の上で最善の行動として選ばれた例があった。同じcheckpointのGreedyの評価200ゲームでは、45本が5面で終わり、そのうち43本が14,550点・ゲームの長さ1,063〜1,066 stepにそろっていた。（出典: [999_eval_session_step_trace_10prd.md](../../memo/999_eval_session_step_trace_10prd.md)、§1）。
- 50M終了checkpoint（seed1、各200ゲーム）を評価時の行動の選び方を変えて評価すると、8面突破率はGreedy（μ-only・τ固定）／ノイズあり／τランダム／両方の順に、スキップなしで50.5／61.5／66.0／60.0%、TempoRL分割のfullで66.0／41.5／50.5／37.5%、分割のsingleで68.5／66.5／70.0／66.5%だった。両方はε=0の学習中のTrain Actorと同じ選び方である。（出典: [qbertの行動固定を評価時だけ外す比較](../../experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md)、現時点のまとめ6・探索ブロック02。Run: `run_20261029-112021_evnt_off_a` ほか同ブロックの12本）。
- スキップなしのseed2（`run_20261028-132638_a5_qb_aln_btreps_s2`）の50M終了checkpointは、学習中と同じ選び方で44.5%・29,279.6点、Greedyで25.5%・26,705.8点だった（各200ゲーム）。学習の最後の400ゲームは49.8%、最後の1,000ゲームは67.5%で、seed1（86.5%・81.4%）より低い。（出典: [qbertの行動固定を評価時だけ外す比較](../../experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md)、現時点のまとめ7・探索ブロック03。Run: `run_20261029-142927_evnt_off2_nt` / `run_20261029-143538_evnt_off2_a`）。

- PRD103のsplitの50M終了checkpoint（seed1）を追加評価すると、full/single各200ゲームの平均点は26,913.6/31,358.9、突破率は63.5%/64.0%だった。評価時の固定を外すと平均点は上がったが、突破率の差は0.5ポイントだった。元3本とは学習時のビルド・構成が違う参考行である。（出典: [qbertの行動固定を評価時だけ外す比較](../../experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md)、現時点のまとめ5・探索ブロック01・追加splitと別枠の評価。Run: `run_20261029-131020_t104_stage0_split`）。

- TempoRLの50M終了checkpoint3本（合算・遮断・重み0.3、各seed1）をfull/single各200ゲームで比べると、singleの平均点は46,495.0・33,000.3・33,859.8となり、fullの29,756.6・29,257.6・26,389.1からすべて改善した。singleの突破率は97%・96%・100%。無効は34,142.4点・突破率50.5%で、平均点と突破率の序列は一致しない。評価時に固定を外した結果であり、学習中のデータの偏りや補助学習だけの効果は未確認。（出典: [qbertの行動固定を評価時だけ外す比較](../../experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md)、探索ブロック01・未確定と今回の区切り。Run: `run_20261029-105620_t104_stage0_merge` / `run_20261029-120945_t104_stage0_stop` / `run_20261029-121756_t104_stage0_lw03` / `run_20261029-105005_t104_stage0_off`）。

- RR4でも第8面の壁が残り、低εのlaneほど深い面へ届いた。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ・探索ブロック02。Run: run_20260926-022413_a5_qbert_btr50m）。
- BTR型のεスケジュールだけでは差が埋まらず、初突破の時期はseedで大きくぶれ、早く出ても定着しない例があった。（出典: [2026-09-27_btr-gap-qbert-bz.md](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ。Run: run_20260927-173004_a5_qbert_btrsched_s1 / run_20260927-035604_a5_qbert_btr50m_eps005_end0）。

- εラダー1e-4〜0.05、64 env・replay容量1M（lane窓16,384）の素の3 seedでは、45–50Mの第8面突破率が45.6 / 32.6 / 0.7%だった。seed1の45.7M・ε=0評価は平均32,948で102本すべて突破し、BTR公開CSVの45M・33,105と同水準だった。（出典: [2026-09-27_btr-gap-qbert-bz.md](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ4・5、探索ブロック03。Run: `run_20260928-065041_a5_qbert_btr50m_eps005_env64`, `run_20260929-025249_a5_qbert_btr50m_eps005_env64_seed2`, `run_20261001-015829_a5_qbert_btr50m_eps005_env64_seed3`）。
- 容量だけを下げた128 env・1Mは45–50Mの突破率1.0%、env数だけを下げた64 env・2Mは0.1%だった。初突破は全腕で18.5〜45.9Mに散らばり、早い初突破だけでは終盤の突破定着に結び付かなかった。（出典: [2026-09-27_btr-gap-qbert-bz.md](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ6・7、探索ブロック03。Run: `run_20260928-202307_a5_qbert_btr50m_eps005_cap1m`, `run_20260929-162737_a5_qbert_btr50m_eps005_env64_cap2m`, `run_20260928-134117_a5_qbert_btr50m_eps005_sync25`）。
- 64 env・1MにUQEまたはViTを加えた45–50Mの突破率は45.8% / 27.4%で、素のenv64のseed幅内だった。ViTの実時間は7.56h、素のenv64は6.25〜6.84hだった。（出典: [2026-09-27_btr-gap-qbert-bz.md](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ10、探索ブロック04。Run: `run_20260930-044852_a5_qbert_btr50m_env64_uqe`, `run_20260930-110739_a5_qbert_btr50m_eps005_env64_vit`, `run_20260928-065041_a5_qbert_btr50m_eps005_env64`, `run_20260929-025249_a5_qbert_btr50m_eps005_env64_seed2`, `run_20261001-015829_a5_qbert_btr50m_eps005_env64_seed3`）。
- RAMのround_clear=8には第8面へ入ってすぐ終わるゲームも含まれ、当該比較の突破判定はround_clear=8かつ29,000点以上だった。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ13、探索ブロック03。Run: `run_20260926-022413_a5_qbert_btr50m`, `run_20260927-035604_a5_qbert_btr50m_eps005_end0`）。
- 実時間優先 100M（seed 2）でも第 8 面で止まった。得点は 3,625 点刻みの段（1 段 = 1 面）に分かれ、91.1M の ε=0 断面では第 7 面に着いた 97 本のうち 58 本が抜けたのに、第 8 面に着いた 58 本は 1 本も抜けなかった。93,010 ゲーム中の第 8 面突破は train の 2 本だけだった。第 8 面に着いた時点の残機は記録していないので、面の難しさとそれまでの消耗は分けられていない（出典: [2026-09-21_realtime-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、現時点のまとめ13・探索ブロック04。Run: `run_20260923-163113_a5_qbert_rr1_100m_seed2`）。
- エミュレータ調査では、RAM 0x82（今の面の立方体の側面色。ROM の 20 面ぶんの表から面の開始時に写す）は 8 面だけ 0x00（黒）だった（出典: [082_atari_ram_progress_metrics_10prd.md](../../memo/082_atari_ram_progress_metrics_10prd.md)、Context / Problem Statement「他のゲームの RAM（2026-09-24）」）。
- 実時間優先 seed 2 の 100M の最終 checkpoint を学習なしで評価すると、ε=0 の 100 本中 93 本が第 8 面に着き、抜けたのは 0 本だった。EvalPanel（sticky 0・Greedy）の画面では第 8 面の立方体の側面が描かれず上面だけになり、側面の色が ROM の表で黒なので、観測の前処理が作った見え方ではない（出典: [2026-09-21_realtime-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、現時点のまとめ18・探索ブロック 05。Run: `run_20260923-222008_atari_qbert`、読み込んだ checkpoint は `run_20260923-163113_a5_qbert_rr1_100m_seed2`）。
- 64 env・1M の seed 4 は、25M 以降は第 8 面に着いたまま 29,000 点を超え（25-34M の 29,000 点以上の 964 本中 955 本が round_clear = 7）、34M 以降は面が進まないまま点だけが増えるゲーム（100,000 点超が 646 本、最高 996,450）に入った。45-50M の train は 29,000 点以上が 48.6% なのに round_clear ≥ 8 は 0.2%、両方を満たすのは 0.1% だった。それまでの 6 本は得点だけと両方で数えた突破率の差が 0.6pt 以内だった（出典: [2026-09-27_btr-gap-qbert-bz.md](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ13・探索ブロック 06。Run: `run_20261009-145225_a5_qbert_btr50m_eps005_env64_seed4`）。
- ε ラダーを 1e-4〜0.05 へ下げて効いたのは低 ε の lane の割合で、同じ ε での腕前ではなかった。同じ ε 帯の第 8 面到達率は旧ラダーとほぼ同じ（0.01〜0.016 で 38.8% 対 37.6%、0.025〜0.04 で 18.1% 対 16.0%）で、代わりに序盤が遅くなった（ε=0 断面 5.3M で 3,848 対 6,436）（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ12・探索ブロック03。Run: `run_20260927-035604_a5_qbert_btr50m_eps005_end0`, `run_20260926-022413_a5_qbert_btr50m`）。
- 64 env・1M の seed 1 と 128 env・2M の ε 腕では、25M 以降の lane の ε 帯ごとの第 8 面到達率はほぼ同じで（ε 0.001 未満で 68.4% 対 66.5%）、変わったのは着いた後に抜けるか（突破率 26.1% 対 1.3%）だった。q_gap・行動の入れ替わり・online と target の不一致は、抜けた Run と抜けなかった Run を分けなかった。上位 4 本が 64 env・1M 系 5 本に偶然集まる確率は約 1%、素の 3 seed 対 3 seed だけなら 20% だった（出典: [2026-09-27_btr-gap-qbert-bz.md](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ5・9・探索ブロック03。Run: `run_20260928-065041_a5_qbert_btr50m_eps005_env64`, `run_20260927-035604_a5_qbert_btr50m_eps005_end0`）。
- BTR の公開 CSV と比べると、20M で 24,264 対 ε 腕 19,921（ε=0.01 評価の 20-25M）と既に差があった。BTR の平均は 30M で 29,000 点を超えるのに対し、ε 腕の突破の開始（41.5M）はそれより 10M 以上遅い（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、探索ブロック03「BTR の公開 CSV に対して」。Run: `run_20260927-035604_a5_qbert_btr50m_eps005_end0`）。
- grad clip（grad_clip_tau 10）の発動率は 10-20M で 45.7% と高いが、削った勾配の量は最大 12.4% で、クリップされた step の倍率は中央値 0.84〜0.90 だった（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ14・探索ブロック02「内部指標」。Run: `run_20260926-022413_a5_qbert_btr50m`）。
- 実時間優先 50M で ε ラダーを NoisyNet に置き換える（ε 0）と、eval2 40-50M は 12,144（基準 seed 1 / 2 は 20,650 / 20,103）で、学習中のゲームも ε=0 断面も round_clear は最大 4（第 5 面より先へ進まない）だった（出典: [2026-10-11_noisynet-realtime.md](../../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md)、現時点のまとめ1・探索ブロック 01。Run: `run_20261012-015046_a5_qbert_rr1_50m_nz_eps0`）。

## 分かっていないこと

- 最後でない残機を失っても`V_s`がほとんど下がらないことは、評価sessionの多くのゲームで成り立つか。14,550点で終わるゲームは、残機を失う面・stepがそろった同じ道筋をたどっているか（出典: [999_eval_session_step_trace_10prd.md](../../memo/999_eval_session_step_trace_10prd.md)、§1・§3のG3）？
- μ-onlyの評価で固定を外した改善が、合算・遮断・重み0.3のcheckpointでもNoisyNetの摂動つきの行動で得られるか。別seedでも得られるか。学習中のデータの偏りと補助タスクの効果はどれだけ残るか（出典: [qbertの行動固定を評価時だけ外す比較](../../experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md)、未確定と今回の区切り・探索ブロック02）？

- BTRとの観測の縮小方式の差を減らすと、第8面以降への進行が改善するか（出典: [PRD099](../../memo/099_atari_resize_mode_10prd.md)、§3「学習効果の確かめ方」）？
- 64 env・容量1Mで突破が増えた効き方の仕組みは何か（出典: [2026-09-27のcampaign](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ6・探索ブロック03「効き方の仕組みは測れていない」）？
- 64 env・1M の seed 4 だけが、第 8 面の中で点を重ね、そのあと面が進まないまま点だけが増えるゲームへ入ったのはなぜか（出典: [2026-09-27のcampaign](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、pending）？

## 撤回・更新

- 2026-10-29: 「μ-onlyの評価で固定を外した改善が、NoisyNetの摂動つきの行動でも得られるかは未確認」→「分割のcheckpointでは、ノイズありでもsingleが66.5%でfullの41.5%を上回った。合算・遮断・重み0.3では未確認」（出典: [qbertの行動固定を評価時だけ外す比較](../../experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md)、探索ブロック02）。
- 2026-10-23: 第 8 面の突破を 29,000 点以上のゲームで数える（64 env・1M の 3 seed で 45.6・32.6・0.7%） → 得点ループの Run があるので 29,000 点以上かつ round_clear ≥ 8 で数える（4 seed で 45.3・32.5・0.7・0.1%）。seed 1〜3 の判定は変わらない（出典: [2026-09-27_btr-gap-qbert-bz.md](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、メタデータ「判定指標」・現時点のまとめ13・探索ブロック 06。Run: `run_20261009-145225_a5_qbert_btr50m_eps005_env64_seed4`）。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [PRD099](../../memo/099_atari_resize_mode_10prd.md)（縮小方式だけを変える比較）。

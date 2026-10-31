# DropMerge

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [題材](../index.md#題材)

## 今の結論

IQNとQR51の順位はUQE評価とGreedy評価で逆転した。NoisyNetへの探索置換はNEETを抑えたが、batch増加の学習側の上乗せの大部分はノイズ感度の差だった。NEET期にはNOOPと最善DROPの差が0付近まで潰れ、NoDropTimeoutの終局は投了ではなくNEETだった。

## 分かっていること

- 成熟checkpointでB512/RR2が有利という初期判断は、後の長期評価で弱められた。1窓の比較を長期の普遍的優位とは扱わない。（出典: [2026-07-27_longrun-batch-replay.md](../../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md)、現時点のまとめ11・探索ブロック07）。
- random tau IQNとQR51の初期100M比較ではQRが上だったが、各1 Runであり最終汎化の優劣は確定していない。層化とantitheticは後の複製でrandomより良い傾向を示した。（出典: [2026-08-09_iqn.md](../../experiments/default-dqn/dropmerge/2026-08-09_iqn.md)、現時点のまとめ。Run: run_20260811-113930_dm_iqn32-b256-rr100 / run_20260812-003400_dm_qr51-b256-rr100）。
- IQNがQR51に負けるという判断はUQE評価との相互作用で、Greedyでは順位が変わった。Munchausenはこの系列で改善を示したが、AtariのReLUなどはそのまま持ち込めなかった。（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ）。
- εラダーを残してNoisyNetを足してもNEETを防げず、εをNoisyNetで置換した基準ではNEETが抑えられた。RR4の同stepの優位とRR1の実時間の優位は両立する。（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ6〜9。Run: run_20261008-015339_dmfb_21_nzrep / run_20261006-135602_dmfb_20_rr4sn）。
- batch増加による学習側の上乗せの大部分はNoisyNetへの感度差で説明され、targetネットの評価専用Runの最終差は小さかった。（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ3〜6。Run: run_20261017-185400_dmfb_42a_34a_ctl）。

- batch_size=512, replay_ratio=2.0について: 170→230Mの分岐（42D）で評価専用Runの平均点は対照と±0.5%。学習側の+0.5〜1.2%はNoisyNetの揺らぎへの感度の差だった。7月のcy07/cy08も学習中の評価は300Mの間ほぼ横ばいで、B256の枝との差は1〜3%（[勾配ノイズの機序](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md) 探索ブロック05・06・09）。scratch開始の80M RunではQバブルと条件付きNEETが発生した（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ・対応する探索ブロック）。

- batch_size=128, replay_ratio=1.25について: cy01の立ち上がりとQ / lossは安定。IQN対応後binaryのQR対照も35.59Mまで旧cy01と同水準で正常closeした（出典: [2026-07-27_longrun-batch-replay.md](../../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md)、現時点のまとめ・対応する探索ブロック）。

- batch_size=256, replay_ratio=1.0について: scratch QR51が100Mで過去B128/RR1.25 QRと同等の終盤rewardを維持し、同条件IQNより約9%高速だった。RR1によるReplay coverage低下は残る（出典: [2026-08-09_iqn.md](../../experiments/default-dqn/dropmerge/2026-08-09_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- batch_size=512, replay_ratio=1.0について: 50→150Mの分岐（41b）で分岐直後に学習側+2.4%上がり、30〜40Mで消えた。学習中の評価は対照と同点（[勾配ノイズの機序](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md) 探索ブロック02）。100→170Mの分岐（31d・34b）の学習側+0.6〜2.0%は評価で確かめていない（[Atari からのフィードバック](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md) 探索ブロック12・14）（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ・対応する探索ブロック）。

- batch_size=1024, replay_ratio=2.0について: 170→230Mの分岐（42H）で学習中の評価は分岐後20Mまでに対照より約3%高いが、40–60Mで対照が追いつき、最終の評価専用Runも+0.3%。実時間は対照の2.2倍（同 探索ブロック04・12）（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ・対応する探索ブロック）。

- batch_size=1024, replay_ratio=1.0について: 計算量・実時間は対照とほぼ同じ。170→230Mの分岐（42J）で、学習中の評価は対照が40–60Mで着く水準に0–20Mで着き、最終の評価専用Run（Eval1・Eval3）も+0.9%（約2.4σ）。分岐後20〜60Mの学習中の評価は横ばいで、勾配ノイズの比も対照並みに戻った。230Mから続けると、評価専用Runは2,051→2,099（290M）→2,148（350M）と60Mごとに+2.3%伸び、230–290Mの学習中の評価はB256/RR1の系列より約1.6%高いまま同程度の傾きで伸びた。B2048はGPUメモリが溢れて1/4の速度、B4096は最初の更新で止まる（同 探索ブロック03・08・12・13・15・16）（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ14・対応する探索ブロック。Run: run_20261018-225933_dmfb_42j_34a_b1024rr1 / run_20261021-001853_dmfb_45a_42j_ctl / run_20261022-132905_dmfb_46a_45a_ctl）。

- batch_size=256, replay_ratio=2.0について: 取りこぼしは29%→5.6%に減ったが、ダブルスイカ率は増えず、評価専用Runは−0.8%。重みの増え方は対照の約3倍（同 探索ブロック10・13）（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ・対応する探索ブロック）。

- 学習率 7.1e-5（B256/RR1、checkpoint継続）について: 50→150Mの分岐（41c）で分岐直後に学習側+2.3%上がり、30〜40Mで消えた。学習中の評価は対照と同点（同 探索ブロック02）（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ・対応する探索ブロック）。停滞区間（170→230M）の同じ腕（42B）では、学習中の評価の上乗せが60M続き（+1.2〜1.8%）、最終の評価専用Runも+1.0%（約2.5σ）でB1024/RR1と同じだった。230Mから続けた45dも評価専用Runで+1.5%伸びた。230MからのB1024/RR1にさらに学習率7.1e-5を足した腕（45b）は、online と target の食い違いが最も小さく（4.6%）なったが、評価の上積みは無かった（−0.4%）（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ13・14、探索ブロック14・15。Run: run_20261019-184443_dmfb_42b_34a_lr7.1e-5 / run_20261022-011640_dmfb_45d_42b_lr71e-6 / run_20261021-071918_dmfb_45b_42j_lr71e-6）。

- Munchausen（τ0.03、α0.9、clip −1、Double DQN OFF）について: IQNの100M checkpointの評価専用Runで、Eval1 1,524→1,796（+18%）、Greedy 1,572→1,831（+16%）。2 seedの差は1〜2%。QR51ではGreedyのNEETは消えたが、UQEでの利得はseedで+4% / −7%に割れた（[Atari からのフィードバック](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md) 探索ブロック01・04）（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ・対応する探索ブロック）。

- Munchausen α 0.7（B1024/RR1の230Mから）について: 次状態のエントロピーが0.835→0.935に上がり、Qの1位と2位の差は−13%、NoisyNetで行動が変わる割合は8.6%→11.0%、Qの水準は+4.6%になったが、60Mでは評価に差が出なかった（評価専用Runで−0.6%、約1.4σ）（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、探索ブロック15 探索結果・考察。Run: run_20261022-064721_dmfb_45e_42j_ma0.7）。

- 学習中のActorの全laneに一定のε 0.02（B1024/RR1の230Mから）について: replayの行動のMunchausenのτ log πが約1.9倍負になり、次状態のエントロピーは0.835→0.745に下がり、Qの1位と2位の差は+12%と、方策はかえって尖った。評価専用Runは−1.3%（約3σ）。最後の2Mで終局がNoDropTimeoutに寄り、最終checkpointの評価専用Runでは86%がNoDropTimeoutで終わった。学習中の評価の60Mの平均は対照と同点圏（−0.4%）（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、探索ブロック15 探索結果・考察。Run: run_20261021-172358_dmfb_45c_42j_eps0.02）。

- IQN と QR51について: 「IQNはQR51に負ける」は評価方策UQEとの相互作用で、Greedyで評価するとIQN 1,572対QR51 1,552。Munchausen入りの2 seed平均はIQN 1,784対QR51 1,671（同 探索ブロック02・04）（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ・対応する探索ブロック）。

- ReLU（Aストリーム、τ埋め込み、main_feature末尾）について: Aストリームは50Mで差なし、全部ReLUはmain_feature末尾の90%がdormant、τ ReLUはQバブルで−30%（50M、Munchausen前の基準。同 探索ブロック03）（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ・対応する探索ブロック）。

- target の hard 更新（C=200）について: IQN+Mでsoftの2 seedより3方策とも低く、QR+MではNoDropTimeoutが39〜58%出た（同 探索ブロック05）（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ・対応する探索ブロック）。

- NoDropTimeout 罰 −20について: 評価専用Runの時間切れの終局が21〜62%から0〜5%へ。待ちは締切直前に落とす長考型に変わっただけで、成績は−10と同水準（同 探索ブロック06）（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ・対応する探索ブロック）。

- ε ラダーを NoisyNet に置き換える（uqe_eps_start=0、τ ラダーは残す）について: NEETの入口はε ラダーの上端のlane。εを0にするだけだと評価専用Runで−7〜−11%、NoisyNetで置き換えると同水準に戻りNEETは0。spatial explorationを切ると−5〜−10%（同 探索ブロック08）。学習側の平均点にはNoisyNetの揺らぎへの感度が混ざる（[勾配ノイズの機序](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md) 探索ブロック06）（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ・対応する探索ブロック）。

- n-step 3について: ε ラダー基準の上で長考型NEETに入り、評価専用Runで−10〜−12%（[Atari からのフィードバック](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md) 探索ブロック07）（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ・対応する探索ブロック）。

- replay ratio 4について: NoisyNet基準で100Mの学習中の評価が+4.1%（1 Run、Runごとのぶれの内側）。同じ実時間ではRR1の300M継続が約10%上。ε ラダーの上のRR4+SNは−11〜−16%（同 探索ブロック07・09・11）（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ・対応する探索ブロック）。

- 共有トランク Linear_2048について: RR1で512と4,096はどちらも−8%、列の位置を残す縮約（map16）は−21%、Transformer 8層は−9%（同 探索ブロック10）（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ・対応する探索ブロック）。

- γ0.999について: 100Mでnzrepと同点（−0.6%）（同 探索ブロック13）（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ・対応する探索ブロック）。

- PER 初期優先度（per_initial_priority_mode=max、per_initial_priority=0.03）について: 一度もsampleされずに捨てられる割合は設計どおり23% / 35%に動いたが、100→120Mの学習側の平均点は±0.8%以内（同 探索ブロック12）（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ・対応する探索ブロック）。

- 評価方策のUQE τについて: 0.5と0.85の差は4 checkpoint中1つで約+1%（約2σ）、他は差なし。1組目の+1.8%は別の局の組で再現しなかった（同 探索ブロック07）（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ・対応する探索ブロック）。

- 腕の判定方法について: NoisyNetを使う構成では、学習側の平均点に揺らぎへの感度の差が混ざる。online ネットのEval2は同じ設定の複製でも2.2%ずれた（target の2本は0.0%）。最終checkpointだけでは到達の速さの差を見落とす。学習中の評価は1セッション1局なら平均点の2%以上、Eval1・Eval3を8局にすると20M窓で1%程度の差が読める（実時間+9%）（同 探索ブロック05〜07・11・12）（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ・対応する探索ブロック）。学習中の評価の値は終局の罰（−10、NoDropTimeoutは−20）を含む報酬の合計で、評価専用Runの平均点（ゲームの点）とは種類が違う。罰をそろえると、×8の最後の10Mと最終の評価専用Runの差は8本で平均−0.1%（−1.5〜+0.8%）で、系統的には低く出ない（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ15・探索ブロック11 判断更新）。

- alpha=5e-5 / 7.5e-5について: checkpoint load により AdamW の param group options が復元され、設定ファイル上の alpha が実効学習率になっていない可能性が高い（出典: [2026-07-27_longrun-batch-replay.md](../../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md)、現時点のまとめ・対応する探索ブロック）。

- per_alpha=0.1について: 単一分岐では明確な改善がなく、NEET 増加の懸念もあった。確度は低い（出典: [2026-07-27_longrun-batch-replay.md](../../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md)、現時点のまとめ・対応する探索ブロック）。

- IQN 32/32 randomについて: 同一B256/RR1のQR51に対し、90–100MのEval target rewardが約8%、Policy rewardが約13%低い。Q/TD/lossは安定し正常close（出典: [2026-08-09_iqn.md](../../experiments/default-dqn/dropmerge/2026-08-09_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- IQN tau samplingについて: 両方式とも2-seedでrandom改善を再現し、seed 2の50Mでも同率。全stratum被覆と2-seed Targetからstratifiedを標準化。systematic / fixedは棄却（出典: [2026-08-09_iqn.md](../../experiments/default-dqn/dropmerge/2026-08-09_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- IQN tau本数について: M探索では変更利得なし。K8は2-seedで20M首位を再現し50Mまで安定したが、K32比の速度利得がなくPolicy marginも低い。K8/K32長期matched A/Bは保留（出典: [2026-08-09_iqn.md](../../experiments/default-dqn/dropmerge/2026-08-09_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- batch sizeの段階的引き上げについて: 停滞区間（170→230M）でBを上げた3腕は、評価で同じ水準に早く着いたが、60Mでは対照が追いつき、最終の評価専用Runの差は+0.2〜0.9%だった。学習側の水準の段差の大部分はNoisyNetの揺らぎへの感度の差（[勾配ノイズの機序](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)）（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ・対応する探索ブロック）。

- scratch B512/RR2の観察では、ResetのNOOP marginが負でも、数回DROPした後に待機を選ぶ状態依存性が見られた。Reset直後の診断だけでは途中の待機を否定できない（出典: [長期Run](../../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md)、探索ブロック10「NOOP、終端、映像所見」・考察。single-seedとbinary差があるため原因は未確定）。

- NoDropTimeout と SpawnBlocked の終局の罰は同じ −10 で、100 step 先の −10 は γ0.997 で −7.4 に割り引かれる。負けが確定した局面では待つ方が 2.6 点得になる（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、探索ブロック02 考察）。

- Munchausen の bonus と γ × soft gap を足した 1 step あたりの正味の項は、soft gap を記録した 3 本で 90–100M に −0.037〜−0.045 で、全窓で負だった。「毎 step の加点で生き延びるほど得になり NEET を生む」という仮説は否定された。当時の分析の −0.125〜−0.004 は学習ステップの窓で切った値で、上は経験数の窓で引き直した値である（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、探索ブロック04 探索結果・考察・判断更新。Run は探索ブロック04の4本: run_20261003-000810_dmfb_munch / run_20261003-235455_dmfb_munch_s2 / run_20261003-081944_dmfb_qr_munch / run_20261003-160717_dmfb_qr_munch_s2）。

- QR51 素の 100M の NEET 期（8〜30M）には、評価の NOOP と最善 DROP の差（UQE スコア）が −0.020〜−0.026 から −0.001〜−0.002 に潰れ、NOOP の選択率は 74〜84% だった。基準 IQN は −0.021〜−0.033 のままで、選択率は 0〜19% だった。QR51+Munchausen τ0.03 は健全な時期に差が −0.10〜−0.16 に開いたが、NEET 期（10〜22M）には 12〜14M に +0.024 まで振れ、選択率は 86% に達した（出典: [999_dropmerge_neet_diagnostics_10prd.md](../../memo/999_dropmerge_neet_diagnostics_10prd.md)、§2。Run: run_20261002-031314_dmfb_qr / run_20261001-191232_dmfb_base / run_20261003-081944_dmfb_qr_munch）。

- 評価で時間切れの瞬間に置ける列が無かった割合は、どの Run でもほぼ 0 だった。NoDropTimeout の終局は投了ではなく NEET である（出典: [999_dropmerge_neet_diagnostics_10prd.md](../../memo/999_dropmerge_neet_diagnostics_10prd.md)、§2）。

- 割引は Agent の step 単位で掛かる（γ0.997、n-step 1）。use_instant_drop = true では DROP の 1 step は落とした果物が着地するまで物理を進め、NOOP の 1 step は 1 フレームしか進めないが、割引はどちらも 1 回である（出典: [999_smdp_transition_duration_10prd.md](../../memo/999_smdp_transition_duration_10prd.md)、§1）。

- IQN+M・n-step 1・RR1・soft 更新の基準から hard 更新・n-step 3・RR4+SN のどれかに変えた 3 腕は、どれも Q の差（h 空間）が基準の約 2〜3 倍に開き、NOOP の差が 0 付近から上へ寄って長考型の NEET が深まった（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、探索ブロック07 考察。Run: run_20261006-033243_dmfb_19_nstep3 / run_20261006-135602_dmfb_20_rr4sn）。

## 分かっていないこと

- ε ラダーのランダムな DROP は、Munchausen の bonus で大きく負の値を受けて DROP 側の Q を押し下げ、NEET の入口になっているか（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、探索ブロック08 考察。候補の機序として挙げられ、検証されていない。Munchausen 無しの QR51 でも NEET が出たので唯一の機序ではない）？

## 撤回・更新

- 2026-10-01: IQNがQR51より劣るという一般化 → UQEとGreedyで比較結果が異なり、評価方策との相互作用があった（出典: [Atariフィードバック](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ2）。
- 2026-10-19: 成熟checkpointのB512/RR2の大幅優位 → 後続の長期比較で初期判断を弱めた（出典: [長期batch/replay](../../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md)、現時点のまとめ11）。
- 2026-10-18: 停滞区間（170→230M）では B を上げた順に学習側の平均点の水準が上がり、B が停滞に効く → 評価専用 Run の平均点は 3 本（2,032 / 2,027 / 2,038）で差がなく、学習側の差は NoisyNet の揺らぎへの感度の差だった。10/19 に、学習中の評価でも B を上げると同じ水準に早く着き、60M で対照が追いつくと読み直された（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、探索ブロック04 判断更新・探索ブロック05・06・12。Run: run_20261017-185400_dmfb_42a_34a_ctl / run_20261017-235549_dmfb_42d_34a_b512rr2 / run_20261018-074714_dmfb_42h_34a_b1024rr2）。
- 2026-10-23: B1024/RR1（42J）の学習中の評価は分岐後 20M までに上がり、その後は横ばい → 横ばいは分岐後 20〜60M の 40M のことで、230M から続けると評価専用 Run は 60M ごとに +2.3%（2,051 → 2,099 → 2,148）伸び、350M でも止まっていない（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ14・探索ブロック15 考察・探索ブロック16。Run: run_20261021-001853_dmfb_45a_42j_ctl / run_20261022-132905_dmfb_46a_45a_ctl）。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [campaignにない独自の分析](../../experiments/default-dqn/dropmerge/README.md#独自の分析)。

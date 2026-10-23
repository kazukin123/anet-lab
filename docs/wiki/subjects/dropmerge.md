# DropMerge

## 今の結論

IQNとQR51の順位はUQE評価とGreedy評価で逆転した。NoisyNetへの探索置換はNEETを抑えたが、batch増加の学習側の上乗せの大部分はノイズ感度の差だった。

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

- batch_size=1024, replay_ratio=1.0について: 計算量・実時間は対照とほぼ同じ。170→230Mの分岐（42J）で、学習中の評価は対照が40–60Mで着く水準に0–20Mで着き、最終の評価専用Run（Eval1・Eval3）も+0.9%（約2.4σ）。その後は横ばいで、勾配ノイズの比も対照並みに戻った。B2048はGPUメモリが溢れて1/4の速度、B4096は最初の更新で止まる（同 探索ブロック03・08・12・13）（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ・対応する探索ブロック）。

- batch_size=256, replay_ratio=2.0について: 取りこぼしは29%→5.6%に減ったが、ダブルスイカ率は増えず、評価専用Runは−0.8%。重みの増え方は対照の約3倍（同 探索ブロック10・13）（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ・対応する探索ブロック）。

- 学習率 7.1e-5（B256/RR1、checkpoint継続）について: 50→150Mの分岐（41c）で分岐直後に学習側+2.3%上がり、30〜40Mで消えた。学習中の評価は対照と同点（同 探索ブロック02）（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ・対応する探索ブロック）。

- Munchausen（τ0.03、α0.9、clip −1、Double DQN OFF）について: IQNの100M checkpointの評価専用Runで、Eval1 1,524→1,796（+18%）、Greedy 1,572→1,831（+16%）。2 seedの差は1〜2%。QR51ではGreedyのNEETは消えたが、UQEでの利得はseedで+4% / −7%に割れた（[Atari からのフィードバック](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md) 探索ブロック01・04）（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ・対応する探索ブロック）。

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

- 腕の判定方法について: NoisyNetを使う構成では、学習側の平均点に揺らぎへの感度の差が混ざる。online ネットのEval2は同じ設定の複製でも2.2%ずれた（target の2本は0.0%）。最終checkpointだけでは到達の速さの差を見落とす。学習中の評価は1セッション1局なら平均点の2%以上、Eval1・Eval3を8局にすると20M窓で1%程度の差が読める（実時間+9%）（同 探索ブロック05〜07・11・12）（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ・対応する探索ブロック）。

- alpha=5e-5 / 7.5e-5について: checkpoint load により AdamW の param group options が復元され、設定ファイル上の alpha が実効学習率になっていない可能性が高い（出典: [2026-07-27_longrun-batch-replay.md](../../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md)、現時点のまとめ・対応する探索ブロック）。

- per_alpha=0.1について: 単一分岐では明確な改善がなく、NEET 増加の懸念もあった。確度は低い（出典: [2026-07-27_longrun-batch-replay.md](../../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md)、現時点のまとめ・対応する探索ブロック）。

- IQN 32/32 randomについて: 同一B256/RR1のQR51に対し、90–100MのEval target rewardが約8%、Policy rewardが約13%低い。Q/TD/lossは安定し正常close（出典: [2026-08-09_iqn.md](../../experiments/default-dqn/dropmerge/2026-08-09_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- IQN tau samplingについて: 両方式とも2-seedでrandom改善を再現し、seed 2の50Mでも同率。全stratum被覆と2-seed Targetからstratifiedを標準化。systematic / fixedは棄却（出典: [2026-08-09_iqn.md](../../experiments/default-dqn/dropmerge/2026-08-09_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- IQN tau本数について: M探索では変更利得なし。K8は2-seedで20M首位を再現し50Mまで安定したが、K32比の速度利得がなくPolicy marginも低い。K8/K32長期matched A/Bは保留（出典: [2026-08-09_iqn.md](../../experiments/default-dqn/dropmerge/2026-08-09_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- batch sizeの段階的引き上げについて: 停滞区間（170→230M）でBを上げた3腕は、評価で同じ水準に早く着いたが、60Mでは対照が追いつき、最終の評価専用Runの差は+0.2〜0.9%だった。学習側の水準の段差の大部分はNoisyNetの揺らぎへの感度の差（[勾配ノイズの機序](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)）（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ・対応する探索ブロック）。

- scratch B512/RR2の観察では、ResetのNOOP marginが負でも、数回DROPした後に待機を選ぶ状態依存性が見られた。Reset直後の診断だけでは途中の待機を否定できない（出典: [長期Run](../../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md)、探索ブロック10「NOOP、終端、映像所見」・考察。single-seedとbinary差があるため原因は未確定）。

## 分かっていないこと

なし

## 撤回・更新

- 2026-10-01: IQNがQR51より劣るという一般化 → UQEとGreedyで比較結果が異なり、評価方策との相互作用があった（出典: [Atariフィードバック](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ2）。
- 2026-10-19: 成熟checkpointのB512/RR2の大幅優位 → 後続の長期比較で初期判断を弱めた（出典: [長期batch/replay](../../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md)、現時点のまとめ11）。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [campaignにない独自の分析](../../experiments/default-dqn/dropmerge/README.md#独自の分析)。

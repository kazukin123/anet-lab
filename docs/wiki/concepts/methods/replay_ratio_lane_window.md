# Replay ratioとlane窓

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [手法](../../index.md#手法)

## 今の結論

Breakout の比較では、lane 窓 16,384 の 2 条件は ε=0 で 685.12 と 685.30、窓 8,192 は 615.73 だった。効いた仕組みは未測定である。
容量と env 数を読むときは、各 lane が保持する経験の数と、同じデータから何回更新するかを分ける。

## 分かっていること

### anet の知見

- Breakoutの該当構成ではlane窓16,384の2点はε=0で685.12と685.30、8,192は615.73だった。更新数だけの説明は3点を追わず、機序は未測である。（出典: [2026-09-15_replay-lane-window.md](../../../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md)、現時点のまとめ1〜6。Run: run_20260914-231706_rr4_btrfull_envs128_50m / run_20260915-203706_rr4_btrfull_envs128_cap2m_50m）。
- B512で更新頻度を揃えたRR4はRR4/B256と同水準で、RR2はそれらより良かった。この条件では勾配頻度より経験再利用回数に成績が追従した。（出典: [2026-09-08_replay-ratio-mechanism.md](../../../experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md)、現時点のまとめ1〜3。Run: run_20260909-142738_rr4_b512_munch / run_20260908-201708_rr2_munch）。
- PQLの連続制御での容量比例不要という結果と、Atariのlane窓の観測を同一条件の反証として扱わない。（出典: [replay_lane_window_survey_2026-09-17.md](../../../../reports/replay_lane_window_survey_2026-09-17.md)、2. 分散アクター構成における容量設計）。
- Breakout の lane 窓 12,288 の Run は exp_step 27,092,992 で bad_alloc により止まった。確保地点は、当時 Sample() のたびに capacity サイズで行われていた ValidIndexManager::GetValidIndices1D の vector の reserve で、要求サイズ 12,582,951 byte は replay_capacity 1,572,864 × 8 + 39 と一致した。全 workspace の bad_alloc 停止 9 本のうち、サイズとスタックが残っているのはこの 1 本だけだった（出典: [2026-09-15_replay-lane-window.md](../../../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md)、付随: bad_alloc の確保地点。Run: run_20260919-190647_rr4_btrsn12_envs128_cap1536k_50m）。
- MontezumaのTaigaら寄せ＋RND（価値の分解、内的の割引率の分離、損失を合わせて更新、128 env、seed 1、30M）で、replayの容量を2,097,152から4,194,304（lane窓16,384から32,768）にしたRunは、20〜30MのGreedy（320ゲーム）の平均が1,919.4（窓16,384のRunは3,160.3）、ε=0.01の評価は5回で2,356.0（同3,574.0）だった。Greedyで2500点以上が初めて出たのは21.2M（同15.2M）。26.5〜30Mにはε 0.01〜0.025とε 0.025〜0.16のlaneの1ゲームの長さの平均が15,218・10,458 step（同2,080・1,553）で、28M以後に1本もゲームを終えなかったlaneが128本中54本あった。設定のdumpの差は容量だけだが実行体のbuildも違い、どちらも1本である（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ33・34・探索ブロック20・21。Run: run_20261118-111145_mz_tg_g99ne_cap4m_s1、run_20261116-173557_mz_tg_r121_g99ne_s1）。
- 同じMontezumaの構成（損失を合わせて更新、lane窓16,384）でreplay ratioを8から4にし、targetの更新と評価・probeの間隔をexp stepで揃えた2本（seed 1・2、2026-11-19のbuild）は、Greedy（`53_evalg`）で剣で敵を倒すゲームが16本中8本以上になった最初の回が26.2M・18.2M（replay ratio 8の2本は同じ数え方で18.2M・18.2M）で、30Mまで学習側も評価も2500点の道に留まった（replay ratio 8の2本は学習側で3500点の道へ移った）。replay ratio 4のままn-stepを3から10にした2本（2026-11-21のbuild）は12.2M・14.2Mだった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ29・36・37・探索ブロック23。Run: run_20261119-144336_mz_tg_g99ne_rr4_s1、run_20261119-202718_mz_tg_g99ne_rr4_s2、run_20261121-211321_mz_g99ne_n10_rr4_s1、run_20261122-030405_mz_g99ne_n10_rr4_s2）。
- BTR NN 構成の RR2 は未測定である。実時間優先・サンプル効率優先のどちらの選択も変わらないため、追加探索から外した（出典: [Atari 探索索引](../../../experiments/default-dqn/atari/README.md)、追加探索と複製の運用方針）。
- name_this_game の RR1・旧 NN は容量 2M→4M で age・count が 98〜102% 増えた。未抽選率は 56.0%→55.5%、共有する Greedy 8 点は平均 −2.2%、ε=0.01 の終盤は −2.5%。各 1 seed で、容量増による悪化は確定できない（出典: [実時間優先の Atari-5 記録](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック 02）。
- Breakout の PER で一度も抽出されずに消えた割合は、RR1 の0.249から RR2 の0.0265、RR4 の0.00012へ低下した。抽出回数の増加と、後半成績の改善は一致しなかった（出典: [Atari Breakout: 高 replay_ratio で成績が落ちる機序](../../../experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md)、現時点のまとめ。Run: `run_20260907-235121_rr4_hard125_munch`・`run_20260908-201708_rr2_munch`・`run_20260909-142738_rr4_b512_munch`）。
- Breakout・128環境の容量比較では、lane 窓32,768の ε=0 評価は16,384の2 seed 平均を、対応する9断面で平均42.4点（±10.8）上回った。終了時の評価は719.85対688.86だった。16,384で効果が頭打ちになるとはいえない（出典: [Atari Breakout: replay の lane 窓と eval コスト](../../../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md)、現時点のまとめ。Run: `run_20260919-093925_rr4_btrsn12_envs128_cap4m_50m`・`run_20260919-014649_rr4_btrsn12_envs128_cap2m_50m_s2`）。
- DropMerge の成熟 checkpoint からの分岐では、B256/RR1 と B512/RR2 の長期評価の差は1〜3%だった。初回 cy08 の単一窓から読んだ大きな差は、同じ窓の再実行に当てはまらなかった（出典: [長期 Run における batch / replay 探索履歴](../../../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md)、現時点のまとめ・2026-10-19 判断更新。Run: `run_20260805-115101_apx-ll_cy07`・`run_20260807-162046_apx-ll_cy08_b256`・`run_20260808-023912_apx-ll_cy08`）。
- 観測履歴の不整合の露出が最大だった lane 窓4,096で修正前後を比較した。ε=0 の7断面で差は +1.5 ± 16.8 点で、窓を広げた効果をこの不整合では説明できなかった。（出典: [Atari Breakout: replay の lane 窓と eval コスト](../../../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md)、現時点のまとめ。Run: `run_20260920-053733_rr4_btrsn12_envs128_cap512k_50m`・`run_20260920-153956_rr4_btrsn12_envs128_cap512k_50m_fix`）。

### 外の事例

- 文献では、Replayの容量を1Mから10Mへ増やした利得にPERの有無はほとんど影響せず、PERを外したRainbowでも+17.3%の利得が残った（Fedusら。著者は直感に反する結果とする）。DQNの基準設定（Replay 10^6、4遷移ごとに32件）では各遷移が平均8回引かれ、BTRのreplay ratioは環境stepあたり1/64（batch 256）、Replayは2^20である。1遷移あたりの抽選回数を変えてPERの効き方を直接測った研究は、調査では見つからなかった（出典: [PER の優先度の弱点と代替のサーベイ](../../../../reports/per_priority_weaknesses_survey_2026-11-18.md)、§6）。
- Pardo らはタイムアウト後の bootstrap を継続すると、大きいバッファを不利にした現象がほぼ消えることを示した。エピソード長200に対して短いバッファも含む比較で、容量と終端処理を合わせて読む必要がある（出典: [Survey: replay バッファの「1 アクターあたりの履歴窓」とエピソード長の比](../../../../reports/replay_lane_window_survey_2026-09-17.md)、3. エピソード単位 replay とエピソード長「エピソード境界の扱いがバッファサイズ感度を左右する」）。
- Replay を介してエピソード内の新しさを学ぶ方式では、保存した内的報酬を読む方法と履歴から再計算する方法で、サンプリング時に必要な文脈が異なる。 → 詳しくは[内的報酬による探索の比較](intrinsic_novelty.md)。（出典: [Survey: エピソード内の新しさのボーナスを価値ベース（DQN・Replay）に載せるときの論点](../../../../reports/episodic_bonus_value_based_survey_2026-11-18.md)、3. NGU 以外のエピソード内ボーナスと off-policy・3.3 Replay からの計算し直し）。
- 同じ遷移が何度抽出されても TD 誤差が消えない場合があり、優先抽出の頻度と学習に役立つ量は一致しない。 → 詳しくは[優先度付き経験再生（PER）](prioritized_replay.md)。（出典: [Survey: PER の優先度の弱点と代替](../../../../reports/per_priority_weaknesses_survey_2026-11-18.md)、2. TD 誤差の優先度の弱点・2.1 確率的な報酬と張り付き）。
- Ape-X で遷移を8回複製して置換速度をそろえても性能は戻らなかったが、複製は実効容量も小さくする。鮮度だけを変えた比較ではない。（出典: [Survey: replay バッファの「1 アクターあたりの履歴窓」とエピソード長の比](../../../../reports/replay_lane_window_survey_2026-09-17.md)、5. 批判と留保）。
- 逐次サンプルの理論と固定サイズの replay buffer を使う実験では、時間刻みを変えたときに保持する実時間の範囲が異なる。 → 詳しくは[行動の時間尺度](temporal_abstraction.md)。（出典: [Making Deep Q-learning Methods Robust to Time Discretization](../../../../reports/tallec2019_time_discretization_paper_2026-10-03.md)、論文が述べている限界）。

## 横断的知見

- 対立点: lane 窓を 16,384 から 32,768 に増やしたとき、Breakout では Greedy の 9 点ペア平均が +42.4 点、name_this_game では共有 8 点の平均が −2.2% だった。条件の違い: ゲームに加え、前者は RR4・BTR 系 NN、後者は RR1・旧 NN。どの違いが効果を変えたかは分離できない（出典: [Breakout](../../../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md)、容量を倍にした比較、[name_this_game](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック 02）。
- 対立点: 旧ネットワークの Breakout では、バッチを大きくして更新頻度をそろえても RR4 の後半突破率は RR2 を下回った。BTR 型ネットワークに替えると旧型に対する利得は RR4 に現れ、RR1・RR2 では旧型を下回った。再利用回数の比較結果はネットワーク構成に依存した。条件の違い: 旧ネットの比較はミニバッチと RR の操作、移植試験はネットワークと RR の組み合わせである（出典: [2026-09-12_btr-network-port](../../../experiments/default-dqn/atari/2026-09-12_btr-network-port.md)、現時点のまとめ、`run_20260912-074000_rr4_btrfull_munch`・`run_20260914-012640_rr1_btrfull_50m`・`run_20260914-091740_rr2_btrfull_50m`。[2026-09-08_replay-ratio-mechanism](../../../experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md)、現時点のまとめ、`run_20260908-201708_rr2_munch`・`run_20260909-142738_rr4_b512_munch`）。
- 補完: Breakout の測定では、lane あたりの履歴窓と貪欲エピソード長を比べると評価差を整理できた。文献調査で容量をエピソード長で明示的に割った実装例は HER で、目的は目標の再抽出である。共通の尺度を使っても Breakout の閾値が文献で再現されたわけではない。条件の違い: Breakout は貪欲評価と lane の履歴窓を比較し、HER はエピソード内の目標を引き直す（出典: [replay_lane_window_survey](../../../../reports/replay_lane_window_survey_2026-09-17.md)、3. エピソード単位 replay とエピソード長「エピソード長でバッファ容量を正規化した唯一の実例」。[2026-09-15_replay-lane-window](../../../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md)、現時点のまとめ）。

## 見立て

なし

## 分かっていないこと

- Breakoutでlane窓を伸ばして成績が変わった機序は何か（出典: [2026-09-15のcampaign](../../../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md)、現時点のまとめ2・探索ブロック02「何が未測か」）？
- lane 窓の下側の崖（Breakout では 4,096 で質的に破綻し、12,288 は破綻しない）は、他のゲームでも同じか（出典: [2026-09-21_realtime-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、pending「lane 窓の下側」。他ゲームでは 1 本も測っていない）？
- Montezumaでlane窓を32,768へ広げたRunの遅れと終盤の長いゲームは、窓の効果か、Runごとのぶれか、実行体のbuildの差か。1本だけで、分けていない（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ33）？

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [Montezuma's Revenge](../subjects/montezuma_revenge.md)。
- [evaluation](../measurement/evaluation.md)。
- [仕組みと契約](../../../design/200_dqn_agents.jp.md)。

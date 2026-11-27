# 計測の負荷と比較条件

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [測り方](../../index.md#測り方)

## 今の結論

TimeHistogram の調査では 22.37 秒中 16.26 秒を全生値の再構築が占め、出力間隔を広げても取得と再構築の量は減らなかった。
Breakout の可塑性 probe は interval 100 で throughput が約 2.1% 下がり、500 ではほぼ差がなかった。

## 分かっていること

### anet の知見

- `learner.use_rb_prefetch=true` での Nsight 計測では、`Sample()` 末尾で `PrefetchedBatch` を毎バッチ破棄し、`cudaEventDestroy` に代表値で +7.459 ms かかった。H2D と learner stream を同期する event の record/wait は必要で、再利用の対象は event ハンドルだった（出典: [RFD0020](../../../memo/done/0020_device_transfer_common_part_10rfd.md)「Context / Problem Statement」）。

- 2026-07のTimeHistogram調査では、画像の出力intervalを下げても全valid値の取得と全生値の再構築は減らず、TrainThreadを同期的に止めていた。Run `run_20260720-141314` の22.37秒中16.26秒が再構築だった。この記録は当時の経路の実測であり、現行の全診断へ同じ比率を適用しない（出典: [RFD0056](../../../memo/0056_timehistogram_fine_grid_10rfd.md)、問題・前提）。
- Breakoutの可塑性probeは、interval 100では約2.1%のthroughput低下、500ではほぼ差なしだった。probe batchを変えるとdead ratioの見え方も変わるため、測定条件を腕の差から分ける（出典: [可塑性較正](../../../experiments/default-dqn/atari/2026-08-28_plasticity.md)、現時点のまとめ7・8。Run: `run_20260829-094914_ab_s2_probe_b4096_5m` ほか）。
- eval schedule は既定で発火列の先頭 1 回を抑止し、初期の重みを測る評価を走らせない。`warmup_count=0` で初回も評価でき、`offset` は発火時点をずらす。開始時点を分けても、セッションが長ければ実行時間は重なる（出典: [Observability](../../../design/140_observability.jp.md)、eval schedule と背景評価の待機ログの説明；[RFD0158](../../../memo/0158_eval_schedule_first_fire_10rfd.md)、§1.3・§4 D2・D8）。
- 背景評価では学習は止まらず、次の発火時に前回のセッションが終わっていない場合だけ学習が待たされる。Atari-5 の learn_step 0 の greedy_dist セッションでは、未学習の貪欲方策が全エピソードで 27,000 step の上限に張り付いた name_this_game と double_dunk で、次の発火時に学習が 692.6s と 885.5s 止まった（合計約 26 分。5 ゲーム × 50M のスイープに対して 0.9%）（出典: [0158_eval_schedule_first_fire_10rfd.md](../../../memo/0158_eval_schedule_first_fire_10rfd.md)、§1.2「実測（2026-09-21）」）。
- eval1 と eval2 の同位相の発火による Train の定期停止は、既に診断された別件として扱われている。背景評価が interval をまたぐと次の gate で学習が待ち、DropMerge では eval1・eval2 が同位相で発火する（出典: [RFD0052](../../../memo/done/0052_eval_schedule_separation_10rfd.md)、6. スコープ外；[RFD0060](../../../memo/done/0060_eval_batch_episodes_10rfd.md)、9. コスト見積もり）。
- この実験機は 1 時間で最大 8% ドリフトするため、性能 A/B の腕をブロックに並べると符号を誤る。RR8・1M・eval 無効の 7 腕 × 3 巡をラウンドロビンで回して 3 巡の中央値で読むと、3 本がバックグラウンド負荷で低い外れ値になっており、1 巡だけでは actual が probe より高いという逆の結論が出た（出典: [可塑性較正](../../../experiments/default-dqn/atari/2026-08-28_plasticity.md)、探索ブロック02。Run: run_20260828-104700 〜 141420）。
- 剰余判定の発火のころ、Pong の num_envs 512 が 64 より 2.65 倍速く見えたのは、learn_step が round ごとに飛び飛びに進んで eval の回数が 4 倍違ったためで、eval 頻度を揃えると 1,264 対 1,198 steps/s で差は消えた（出典: [RFD0053](../../../memo/done/0053_interval_gate_perf_ema_10rfd.md)、Context 不具合 A。Run: run_20260815-031815_atari_pong_n512 / run_20260815-035220_atari_pong_n64 / run_20260815-052657_atari_pong）。
- 当時の perf メトリクスは 200ms 以上たまった窓の最新値をサンプル単位で拾っていたため、eval で止まった長い窓の時間の重みが失われ、真値 478 steps/s の Run を 1,830 steps/s と表示していた。真の throughput は 90_perf/90_elapse_hour の差分からしか得られなかった（出典: [RFD0053](../../../memo/done/0053_interval_gate_perf_ema_10rfd.md)、Context 不具合 B）。
- 同 seed で train の loss・weight だけが Run ごとに割れ、eval は再現した原因は、SDPA が CUDA で mem-efficient backend に落ち、その backward が勾配を atomic 加算することだった。forward は決定的なので eval は再現した（出典: [RFD0015](../../../memo/done/0015_deterministic_algorithms_10rfd.md)、Context（背景・目的））。
- KFM の実時間優先＋RND（128 lane、CUDA、Release）で、EvalPanel を再生したまま Evaluation Trace の pane を閉じた状態と開いた状態を交互に 3 回ずつ測ると、学習の exp step/s の中央値は通常の RND で 3,633 と 3,164、Taiga らの式で 249 と 233 だった。推定器と β の写しは定期の Actor 同期（既定で 10 秒ごと）のたびに 1 回入って 1 回 152〜189 ms で、RFD0127 が見積もった数ミリ秒の約 50 倍だった。pane を閉じた状態では採点も写しも行われなかった（出典: [RND の解説](../../../design/212_rnd_lifelong_novelty.jp.md)、§9.8 Evaluation Trace で読む値）。
- deterministic_algorithms = true は false に比べて約 11〜13% 遅い。SDPA の backward に限らず scatter・index_add・reduction などの決定版を全 op に強制する広域のスイッチだからで、「attention は実時間のボトルネックでないので無視できる」という当初の見積もりは誤りだった（出典: [ADR 0006](../../../adr/0006-deterministic-algorithms.md)、Consequences）。
- 実時間優先の Atari-5 では 50M の 5 本で評価が 208.8M 環境 step、学習が 250M。評価による学習停止は合計 4.75 時間だった。Greedy は 10 回で 155.0M、ε=0.01 は 199 回で 53.8M を使った（出典: [実時間優先の Atari-5 記録](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック 01）。
- 固定した name_this_game 方策を 100 ゲームずつ 5 回評価すると、評価 lane を 16 から 128 に増やしても実時間は 14 分 45 秒から 11 分 36 秒への短縮にとどまった。ゲーム数と総環境 step は同じで、8 コアの実験機で測った（出典: [実時間優先の Atari-5 記録](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック 01）。
- 100M の 5 本では評価負荷が学習の 6〜30% に下がり、4.1〜5.4 時間で完走した。double_dunk の初期 2 回は全エピソードが 27,000 step 上限に達し、Greedy 評価の 82% の環境 step を占めた（出典: [実時間優先の Atari-5 記録](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック 04）。
- Atari の RND 性能測定では、3巡目は3群とも速くなった。各群6本の中央値同士の比（既定/なし1.661、Taiga/なし14.754）は、巡ごとの比を平均した値とは異なる（出典: [AtariのRND性能測定](../../../experiments/default-dqn/atari/2026-11-12_rnd_performance.md)、現時点のまとめ。Run: `run_20261112-205944_prd116_ap_r01_off`・`run_20261112-210039_prd116_ap_r02_on`・`run_20261112-210153_prd116_ap_r03_taiga`）。
- GridMaze の CPU・256 lane で、warm-up 後の165,632経験にかかった実時間は、RND なしが中央値23.261秒、既定の RND が31.148秒だった。なし・あり・あり・なしの順に3巡し、各群6本で計算した。試走2本と Tracy 用の1本は除いた（出典: [GridMaze の RND 性能測定](../../../experiments/default-dqn/gridmaze/2026-11-07_rnd_performance.md)、条件と比較方法・通常比較の結果、`run_20261107-230211_prd116_gp_r01_off`・`run_20261107-230238_prd116_gp_r02_on` ほか）。
- 評価終了ログの elapsed は Actor の Sync 前から SessionEnd の通知後までの経過秒数である。背景評価によって学習が待った時間の計測には使えない（出典: [可観測性](../../../design/140_observability.jp.md)、§7.2）。
- LunarLander 級の小型 MLP の DQN 更新では、融合前の AdamW がパラメータ約10個を個別に処理し、1 step に100回超の CUDA カーネルが起動した。Tracy と同期実行の測定で optimizer の起動負荷を確認した（出典: [RFD0011](../../../memo/done/0011_fused_adamw_10rfd.md)、Context（背景・目的））。
- DropMerge の TransformerEncoder を調べた3本のトレースでは、非同期実行の CPU 区間に GPU の待ち時間が載った。CUDA 起動を同期した計測では自己注意が1.571msで、層全体2.428msの65%だった（出典: [RFD0012](../../../memo/done/0012_sdpa_attention_10rfd.md)、Context（背景・目的））。
- ReplayBuffer の先読み導入前のプロファイルでは、PER の抽選と経験収集、同期 H2D 転送がそれぞれ約6.5ms/step を占め、その間 GPU は計算を待っていた（出典: [RFD0013](../../../memo/done/0013_sample_prefetch_10rfd.md)、Context（背景・目的））。
- RFD0097 の再実装で実行体 v2 を比べたとき、DropMerge serial は5反復の中央値が2,889.554から2,689.727 exp/sへ下がった。変更前の最大値と最小値の比から1を引いた値が19.2386%で、判定基準2%を超えたため、6.9155%低下を確定した性能劣化とは扱えなかった。後続の native 値所有への変更後はこの速度を測っていない（出典: [RFD0179](../../../memo/0179_rfd0097_performance_remeasurement_10rfd.md)、「背景と問題」「22implでの関連変更」）。
- Montezuma・64 lane・CUDA の Tracy の内訳では、Taiga らの設定の RND は 1 intake あたり 170.18ms で、RND の時間の 93.20% が Learn（optimizer step 35.90%・backward 28.73%）だった。既定の RND は 1 intake あたり 13.097ms で、Estimate が 82.68%、そのうち統計の区間が RND 全体の 62.66% だった。CPU 側の区間の時間である（出典: [2026-11-12_rnd_performance.md](../../../experiments/default-dqn/atari/2026-11-12_rnd_performance.md)、Tracy による内訳）。

### 外の事例

- BYOL-Explore の Atari は学習 TPUv2 4 個、actor CPU 400 個、推論 TPUv2 4 個。E3B は逆行列の rank-1 更新で特徴次元の二乗時間となり、比較で 1.5〜2 倍遅かった。AMA は RTX 2080Ti の Space Invaders で約 12 時間、RND・ICM の追加実時間はこの調査では未確認（出典: [内的報酬の手法の全体像](../../../../reports/intrinsic_reward_landscape_survey_2026-11-18.md)、§2.1・4.3・7）。
- BTR のハードウェア表で H100 が RTX 4090 より遅かった比較には環境の供給側の律速が関わる。別のベクトル環境実装で learner が律速なら同じ順位を想定できない。（出典: [Survey: Atari の実時間効率フロンティア](../../../../reports/atari_walltime_frontier_survey_2026-09-09.md)、11. 批判・懸念・11.3 ハード世代差が補正されていない）。

## 横断的知見

- 共通点: RND の既定設定は、同じ環境内で経験数をそろえた比較で、GridMaze と Atari の両方で実時間を延ばした。GridMaze は CPU で165,632経験、Atari は Montezuma・CUDA・64 lane で89,984経験であり、両環境の絶対秒数は直接比べられない。条件の違い: GridMaze は CPU、Montezuma は CUDA・64 lane で、測定区間の経験数も異なる（出典: [2026-11-07_rnd_performance](../../../experiments/default-dqn/gridmaze/2026-11-07_rnd_performance.md)、条件と比較方法・通常比較の結果、`run_20261107-230211_prd116_gp_r01_off`・`run_20261107-230238_prd116_gp_r02_on`。[2026-11-12_rnd_performance](../../../experiments/default-dqn/atari/2026-11-12_rnd_performance.md)、条件と比較方法・通常比較の結果、`run_20261112-205944_prd116_ap_r01_off`・`run_20261112-210039_prd116_ap_r02_on`）。

## 見立て

- Montezuma の Taiga ら寄せの Run では、Greedy の評価（16 ゲーム）がほぼ全部 27,000 step の打ち切りまで続くので、Atari-5 の初期の Greedy と同じく、背景の評価が次の発火までに終わらず、学習を待たせているのではないか。 状態: 未検証
  - 支える事実: (1) seed 2 の Run の 10.2〜29.2M の `53_evalg` 320 ゲームのうち、打ち切りでないのは 4 本だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 10）。 (2) 割引率の分離と損失ごとの更新を足した Run では、15.2〜29.2M の `53_evalg` の 15 回のうち 11 回は、16 ゲームとも打ち切りだった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、探索ブロック 20）。 (3) 3500 点以上の道へ移った Run を含む 5 本の、20.2〜29.2M の `53_evalg` の打ち切りは、160 ゲーム中 125〜159 本だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、探索ブロック 21）。 (4) 未学習の貪欲方策が全エピソードで 27,000 step の上限に張り付いた name_this_game と double_dunk では、次の発火時に学習が 692.6s・885.5s 止まった（出典: [RFD0158](../../../memo/0158_eval_schedule_first_fire_10rfd.md)、§1.2）。 (5) Atari-5 の 100M では、上限まで終わらないセッションが evalg の予算を押し上げ、seed 2 の double_dunk は 0.2M と 10.3M の 2 回が上限に張り付いて evalg の 82% を占めた（出典: [2026-09-21_realtime-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック 04）。
  - 合わない事実: (1) 記録は、Run の速さの違い（`90_perf/12_exp_step_per_sec` が 867 と 1,012 など）を build とメイン機の再起動で説明していて、評価の負荷には触れていない（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、探索ブロック 21）。
  - 条件の違い: Atari-5 の事例は learn step 0 の ε=0 の評価で、Montezuma は学習の途中の 16 ゲームの Greedy である。Montezuma は replay ratio 8 で、RND を使う。
  - 確かめ方: Montezuma の Taiga ら寄せの Run のログで、`[greedy]` の発火の `waited for previous session` の行を数え、待ちの回数と秒数を出す（今ある Run の集計で済む。行の読み方は [140_observability.jp.md](../../../design/140_observability.jp.md) の 7.2）。
  - 出どころ: AI

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [可観測性](../../../design/140_observability.jp.md)、§2・§7。
- [eval schedule は既定で学習直後に評価せず、発火の時点を offset でずらす](../../../adr/0070-eval-schedule-skips-warmup-fires-by-default-and-shifts-with-offset.md)。
- [RND の解説](../../../design/212_rnd_lifelong_novelty.jp.md)、§9.5 計算する場所と費用・§9.8。
- [RFD0092の計測契約](../../../memo/done/0092_gradient_noise_scale_10rfd.md)。
- [学習経路内のGNS計測の判断](../../../adr/0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16.md)。
- [勾配ノイズの知見](../phenomena/gradient_noise_batch.md)。

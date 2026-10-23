# 計測の負荷と比較条件

## 今の結論

当時のTimeHistogramでは22.37秒中16.26秒を全生値の再構築が占め、出力intervalを下げても取得・再構築の量は減らなかった。Breakoutの可塑性probeはinterval100で約2.1%のthroughput低下、500ではほぼ差がなかった。性能A/Bは1時間で最大8%のドリフトがありラウンドロビンの3巡の中央値で読んだ。剰余判定のeval頻度差と200ms窓のperf値は実時間の比較を誤らせ、決定論モードは約11〜13%遅い。

## 分かっていること

- 2026-07のTimeHistogram調査では、画像の出力intervalを下げても全valid値の取得と全生値の再構築は減らず、TrainThreadを同期的に止めていた。Run `run_20260720-141314` の22.37秒中16.26秒が再構築だった。この記録は当時の経路の実測であり、現行の全診断へ同じ比率を適用しない（出典: [PRD056](../../memo/056_timehistogram_fine_grid_10prd.md)、問題・前提）。
- Breakoutの可塑性probeは、interval 100では約2.1%のthroughput低下、500ではほぼ差なしだった。probe batchを変えるとdead ratioの見え方も変わるため、測定条件を腕の差から分ける（出典: [可塑性較正](../../experiments/default-dqn/atari/2026-08-28_plasticity.md)、現時点のまとめ7・8。Run: `run_20260829-094914_ab_s2_probe_b4096_5m` ほか）。
- 背景評価では学習は止まらず、次の発火時に前回のセッションが終わっていない場合だけ学習が待たされる。Atari-5 の learn_step 0 の greedy_dist セッションでは、未学習の貪欲方策が全エピソードで 27,000 step の上限に張り付いた name_this_game と double_dunk で、次の発火時に学習が 692.6s と 885.5s 止まった（合計約 26 分。5 ゲーム × 50M のスイープに対して 0.9%）（出典: [999_eval_schedule_first_fire_10prd.md](../../memo/999_eval_schedule_first_fire_10prd.md)、Context「実測」）。
- eval1 と eval2 の同位相の発火による Train の定期停止は、既に診断された別件として扱われている。背景評価が interval をまたぐと次の gate で学習が待ち、DropMerge では eval1・eval2 が同位相で発火する（出典: [PRD052](../../memo/done/052_eval_schedule_separation_10prd.md)、6. スコープ外；[PRD060](../../memo/done/060_eval_batch_episodes_10prd.md)、9. コスト見積もり）。
- この実験機は 1 時間で最大 8% ドリフトするため、性能 A/B の腕をブロックに並べると符号を誤る。RR8・1M・eval 無効の 7 腕 × 3 巡をラウンドロビンで回して 3 巡の中央値で読むと、3 本がバックグラウンド負荷で低い外れ値になっており、1 巡だけでは actual が probe より高いという逆の結論が出た（出典: [可塑性較正](../../experiments/default-dqn/atari/2026-08-28_plasticity.md)、探索ブロック02。Run: run_20260828-104700 〜 141420）。
- 剰余判定の発火のころ、Pong の num_envs 512 が 64 より 2.65 倍速く見えたのは、learn_step が round ごとに飛び飛びに進んで eval の回数が 4 倍違ったためで、eval 頻度を揃えると 1,264 対 1,198 steps/s で差は消えた（出典: [PRD053](../../memo/done/053_interval_gate_perf_ema_10prd.md)、Context 不具合 A。Run: run_20260815-031815_atari_pong_n512 / run_20260815-035220_atari_pong_n64 / run_20260815-052657_atari_pong）。
- 当時の perf メトリクスは 200ms 以上たまった窓の最新値をサンプル単位で拾っていたため、eval で止まった長い窓の時間の重みが失われ、真値 478 steps/s の Run を 1,830 steps/s と表示していた。真の throughput は 90_perf/90_elapse_hour の差分からしか得られなかった（出典: [PRD053](../../memo/done/053_interval_gate_perf_ema_10prd.md)、Context 不具合 B）。
- 同 seed で train の loss・weight だけが Run ごとに割れ、eval は再現した原因は、SDPA が CUDA で mem-efficient backend に落ち、その backward が勾配を atomic 加算することだった。forward は決定的なので eval は再現した（出典: [PRD015](../../memo/done/015_deterministic_algorithms_10prd.md)、Context（背景・目的））。
- deterministic_algorithms = true は false に比べて約 11〜13% 遅い。SDPA の backward に限らず scatter・index_add・reduction などの決定版を全 op に強制する広域のスイッチだからで、「attention は実時間のボトルネックでないので無視できる」という当初の見積もりは誤りだった（出典: [ADR 0006](../../adr/0006-deterministic-algorithms.md)、Consequences）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [可観測性](../../design/140_observability.jp.md)、§2・§7。
- [PRD092の計測契約](../../memo/092_gradient_noise_scale_10prd.md)。
- [学習経路内のGNS計測の判断](../../adr/0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16.md)。
- [勾配ノイズの知見](gradient_noise_batch.md)。

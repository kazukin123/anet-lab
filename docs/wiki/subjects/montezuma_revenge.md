# Montezuma's Revenge

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [題材](../index.md#題材)

## 今の結論

サンプル効率優先に RND（β 0.1）を載せた 30M（1 本、RND なしの対照なし）では、ε=0.01 の評価が 14.21M から 400 点（鍵と扉 1 枚）を続けた一方、Greedy は 20.4M まで 0 点で、25.45M に 9/16 が 400 点だった。30M の重みの Greedy は 400 点以上が 19/32 で、届かないゲームは部屋 1 のまま打ち切りまで続いた。評価時に ε 0.01 か τ の乱数を足すと 29〜32/32 になった。20.07M の重みは、評価の 4 条件とも全ゲーム 400 点だった。Taiga ら寄せの構成で β・Munchausen・Double DQN を変えた RND の Run では、β 0.01・Munchausen 無効の 2 本（seed 1・2）の Greedy が 20.2〜29.2M に 99/160・81/160 のゲームで 2500 点に届いて回ごとに揺れた。同じ β 0.01 で Munchausen を有効にした Run（seed 1）は 5/160 で、多くが 500 点（剣を取って敵を倒さない）で止まった。β 0.1 の 2 本は Munchausen の有無に依らず Greedy で 2500 点が 0 本だった（有効は 29.2M まで、無効は 22.2M まで）。seed 2 の Run と Munchausen 有効の Run の Greedy は、ほぼすべてのゲームが 27,000 step の打ち切りで、命を残して終わった。Taiga らの式の RND は、Munchausen の有無に依らず鍵より先へ進まなかった。

## 分かっていること

- RAM の番号付き定義は `1:room_max`（0x83 の最大）と `4:x_min`（0xAA の最小）。面の区切りが無いので、1 には到達した最大の部屋番号を使い、開始の部屋は 1、開始の左の部屋は 0 である（出典: [220_atari_env.jp.md](../../design/220_atari_env.jp.md)、§4.9 RAM メトリクス）。
- サンプル効率優先＋RND の 30M（seed 1）で、Greedy（ε=0・τ 固定、16 ゲーム、約 5.05M ごと）は 0.2〜20.4M が全ゲーム 0 点、25.45M が 243.8（400 点 9、100 点 3、0 点 4）だった。同じ重みの別 session は 20.4M に 25.0（400 点 1/16）、25.45M に 231.2（9/16）。5.25M 以降の Greedy のゲームは、1 本を除いて 108,000 フレームの打ち切りで終わった（出典: [2026-11-08_montezuma-rnd-se-30m.md](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-se-30m.md)、現時点のまとめ1・探索ブロック01。Run: run_20261108-003056_mz_se_rnd_k32_s1）。
- 同じ Run の ε=0.01・τ 乱数の評価（10 ゲーム、約 2.0M ごと）は、14.21〜22.21M の 5 回が 400 点で揃い、24.21M は 360（400 点のゲームは部屋 7 に入って game over）、26.21M は 120、28.21M は 400 だった（出典: [2026-11-08_montezuma-rnd-se-30m.md](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-se-30m.md)、現時点のまとめ2・探索ブロック01。Run: run_20261108-003056_mz_se_rnd_k32_s1）。
- 同じ Run の学習側では、2M の窓で 500 点のゲームが 22〜24M の 0 本から 24〜26M の 87 本、26〜28M の 106 本へ増え、部屋 7 に入ったゲームは 24〜26M に 206 本だった（22〜24M は 0 本）（出典: [2026-11-08_montezuma-rnd-se-30m.md](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-se-30m.md)、現時点のまとめ3・探索ブロック02。Run: run_20261108-003056_mz_se_rnd_k32_s1）。
- 30M の重みを評価だけで 4 条件・各 32 ゲーム測ると、ε=0・τ 固定 259.4（400 点以上 19/32）、ε=0・τ 乱数 362.5（29/32）、ε=0.01・τ 固定 403.1（32/32）、ε=0.01・τ 乱数 412.5（32/32）だった。500 点（`room_max` 6）は ε=0.01 の 2 条件だけに出た。ε=0・τ 固定で 400 に届かなかった 13 本は部屋 1 のまま打ち切りまで続いた（出典: [2026-11-08_montezuma-rnd-se-30m.md](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-se-30m.md)、現時点のまとめ5・探索ブロック04。Run: run_20261108-110233_mz_se_ev30m_2x2）。
- exp step 20,066,688 で保存した重みは、4 条件とも全ゲーム 400 点だった。学習 Run の中で 0/16 だった Greedy の session（exp step 20,400,384）は、この重みより約 5,200 learn step あとである（出典: [2026-11-08_montezuma-rnd-se-30m.md](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-se-30m.md)、現時点のまとめ6・探索ブロック04。Run: run_20261108-111508_mz_se_ev20m_2x2）。

- 同じ構成（Run 名に `mz_tg_` を持つ群）から β・Munchausen・Double DQN だけを変えた RND の Run（seed 1、割引率 0.997、n-step 3）。β 0.1・Munchausen 有効・Double DQN 無効の Run は、Greedy（evalg・evalgs、各 16 ゲーム、約 1M ごと）で 29.2M まで 2500 点が 0 本、1 回の平均の最高は 462.5（27.2M）、ε 0.01 の評価は 10.2〜28.21M に 400〜490 で、学習側の最高は 2500 点（12・13・17・28M 台）だった。β 0.01・Munchausen 無効・Double DQN 有効の Run は、14.2M に Greedy で初めて 2500 点が出て、14.2〜29.2M の 16 回で 2500 点は evalg 170/256・evalgs 172/256。1 回あたり 0/16〜16/16 で揺れ、落ちた回（20.2M・27.2M・28.2M）は 500 点（剣を取って敵を倒さない）が主だった。ε 0.01 の評価は 14.21〜28.21M に 74/80 が 2500 点、学習側は 16〜30M の 2M 窓で終わったゲームの 81〜93% が 2500 点だった（出典: [外的と内的の価値の分解の暫定 PRD](../../memo/121_extrinsic_intrinsic_value_decomposition_10prd.md)、§2.1、[Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ1〜3・探索ブロック02〜04。Run: run_20261108-232006_mz_tg_rnd_s1、run_20261110-003002_mz_tg_nmdd_b001_s1）。
- 同じ構成で β 0.1・Munchausen 無効・Double DQN 有効の Run は、22.2M まで Greedy（evalg・evalgs）で 2500 点が 0 本で、16.2〜20.2M はおおむね 16 ゲームとも 400 点だった。ε=0.01 の評価は 12.21〜22.21M に 400〜450。学習側の 2500 点は 12〜14M に 7 本で、そのあと 22.63M まで 0 本だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ1・3・探索ブロック02〜04。Run: run_20261110-120903_mz_tg_nmdd_b01_s1、22.63M で止めた Run）。
- 同じ β 0.1・Munchausen 無効の Run は、2〜5M に左の扉（部屋 0）を開け、6M 台に右の扉へ移った。400 点以上のゲームのうち左の扉は 3M 台 171/171、5M 台 487/517、6M 台 3/74 で、左の扉の間は 400 点より上のゲームが無かった。β 0.01 の Run は 4M 台から右の扉が主だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ6・探索ブロック05。Run: run_20261110-120903_mz_tg_nmdd_b01_s1、run_20261110-003002_mz_tg_nmdd_b001_s1）。
- 同じ構成で学習のゲームを 40,000 フレームで切った Run（β 0.1・Munchausen 有効、評価の打ち切りは 108,000 のまま）は、学習側で 16〜17.52M に 2500 点が 69 本出たが、Greedy（`53_evalg`）の最高は 13.2M の 412.5（500 点 10/16）で、2500 点は 0 本だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、探索ブロック01・02・04。Run: run_20261109-105448_mz_tg_rnd_tr40k_s1）。
- 同じ構成で β 0.01・Munchausen 有効・Double DQN 無効の Run（seed 1）は、Greedy で 17.2M に初めて 2500 点が出たが、20.2〜29.2M の evalg 160 ゲームで 2500 点は 5 本、500 点が 106 本だった（β 0.01・Munchausen 無効の Run は同じ回で 99 本）。ε 0.01 の評価は 20.21〜28.21M に 7/50 が 2500 点で、16.21〜20.21M は 30 本とも 500 点。学習側の 2500 点は 20〜22M の 26%（53 本中 14 本）がいちばん多く、28〜30M は 3% だった。16〜28M の 2M 窓では終わったゲームの 74〜89% が打ち切りで（β 0.01・Munchausen 無効の Run は 16〜30M に 4〜11%）、Greedy は 1.2〜29.2M の 29 回すべてで 16 ゲームとも打ち切りで終わった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ1・3・8・探索ブロック02〜04・09。Run: run_20261110-233727_mz_tg_rnd_b001_s1）。
- β 0.01・Munchausen 無効・Double DQN 有効を seed 2 で走らせた Run は、Greedy で 18.2M に初めて 2500 点が出て、20.2〜29.2M の evalg で 81/160（21.2M は 0/16、28.2M は 15/16）、ε 0.01 の評価は 20.21〜28.21M に 32/50 が 2500 点（seed 1 は 45/50）だった。学習側の 2500 点は、2M 窓で終わったゲームの 16〜24M が 28〜45%、24〜30M が 72〜87%。seed 1 の Run と同じ実行体（PRD123 より前の build）で、実効の違いは seed だけだが、誤差の標準偏差は seed 1 の Run の 0.55〜0.56 倍、内的報酬の平均は 8M 以降の 4M 窓で 1.6〜3.2 倍だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ1〜3・9・探索ブロック02〜04・10。Run: run_20261111-123702_mz_tg_nmdd_b001_s2）。
- seed 2 の Run の Greedy は、10.2〜29.2M の evalg 320 ゲームのうち 316 本が打ち切りで終わり、20.2〜29.2M の終わったときの lives の平均は 3.50〜4.88 だった（seed 1 の Munchausen 無効は 2.00〜4.00、有効は 2.13〜3.44）。EvalPanel（Greedy、sticky 0）で 29.94M の重みを見ると、鍵を取った直後（ep_step 92、Q は 18 行動とも 2.692〜2.698）に UPFIRE を押して ep_step 95 に命を失い、剣を取ったあとは梯子の下で DOWN を押し続けた（Q は 1.654〜1.663、DOWN 1.663・UP 1.659）。手で梯子の上へ動かすと敵を倒して 2500 点になり、4 step あとに命を失った。そのあと部屋 1 で DOWN を押し続け（lives 2、Q は 0.105〜0.107）、そのままゲームが終わった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ10・探索ブロック09・10。Run: run_20261111-123702_mz_tg_nmdd_b001_s2）。
- 2500 点の内訳は、鍵 100、扉 300、部屋 6 の剣 100、剣を持って部屋 2 へ戻り敵を倒した 2,000。部屋 1 の鍵は使ったあと、ゲームの時計で約 16,000 フレームごとに戻り、その時刻に部屋 1 の外にいるときだけ戻る（ALE、sticky 0、frame_skip 1 の確認で、Run ではない）（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、Run 以外の確認）。

## 分かっていないこと

なし

## 撤回・更新

- 2026-11-12: 「seed 2 の Run は PRD123 の実装後の実行体で `intrinsic.rnd.warmup_mode = learn_predictor` が効いた（seed 1 との差は seed と内的報酬の大きさのどちらによるか）」→「seed 1 と同じ PRD123 より前の実行体で、新しいキーは読まれておらず、実効の違いは seed だけ」。同じ日の取り込みの誤りを直し、問いを外した（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ9・探索ブロック10。Run: run_20261111-123702_mz_tg_nmdd_b001_s2）。
- 2026-11-12: 「Greedy の 2500 点の有無を分けたのは β と Munchausen のどちらか（β 0.01・Munchausen 有効の組は測っていない）」→ β 0.01・Munchausen 有効の Run は 20.2〜29.2M に 5/160 で、同じ回の β 0.01・Munchausen 無効の Run（99/160）を大きく下回った。seed 1 では、β 0.01 と Munchausen 無効がそろった Run だけが届き続けた（各組 1 本）（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ1・探索ブロック02。Run: run_20261110-233727_mz_tg_rnd_b001_s1）。
- 2026-11-12: β 0.1・Munchausen 無効の Run を「22.47M まで集計した走行中の Run」→「22.63M で止めた Run」。学習側の 2500 点は 22.63M まで 0 本のまま（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、目的と比較条件・探索ブロック04。Run: run_20261110-120903_mz_tg_nmdd_b01_s1）。
- 2026-11-10: β 0.01 の Run の学習側で「15M 以降、終わったゲームの 83〜91% が 2500 点」→「16〜30M の 2M 窓で 81〜93%」。窓の取り方を記録に合わせて集計し直した（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、探索ブロック04）。

## 関連

- [atari](atari.md)。
- [220 §4.9 RAM メトリクス](../../design/220_atari_env.jp.md)。
- [内的報酬による探索の比較](../mechanisms/intrinsic_novelty.md)。
- [評価と比較の条件](../mechanisms/evaluation.md)。
- [τ配置とIQN比較](../mechanisms/tau_sampling.md)。
- [RND の腕の暫定 PRD](../../memo/116_rnd_lifelong_novelty_10prd.md)。
- [外的と内的の価値の分解の暫定 PRD](../../memo/121_extrinsic_intrinsic_value_decomposition_10prd.md)。
- [RND の内的報酬の尺度の PRD](../../memo/123_rnd_intrinsic_reward_scale_10prd.md)。
- [Montezuma's Revenge の RND（Taiga ら寄せの構成、β と Munchausen）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)。
- [Montezuma's Revenge の RND（サンプル効率優先・30M）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-se-30m.md)。

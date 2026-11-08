# Montezuma's Revenge

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [題材](../index.md#題材)

## 今の結論

サンプル効率優先に RND（β 0.1）を載せた 30M（1 本、RND なしの対照なし）では、ε=0.01 の評価が 14.21M から 400 点（鍵と扉 1 枚）を続けた一方、Greedy は 20.4M まで 0 点で、25.45M に 9/16 が 400 点だった。30M の重みの Greedy は 400 点以上が 19/32 で、届かないゲームは部屋 1 のまま打ち切りまで続いた。評価時に ε 0.01 か τ の乱数を足すと 29〜32/32 になった。20.07M の重みは、評価の 4 条件とも全ゲーム 400 点だった。

## 分かっていること

- RAM の番号付き定義は `1:room_max`（0x83 の最大）と `4:x_min`（0xAA の最小）。面の区切りが無いので、1 には到達した最大の部屋番号を使い、開始の部屋は 1、開始の左の部屋は 0 である（出典: [220_atari_env.jp.md](../../design/220_atari_env.jp.md)、§4.9 RAM メトリクス）。
- サンプル効率優先＋RND の 30M（seed 1）で、Greedy（ε=0・τ 固定、16 ゲーム、約 5.05M ごと）は 0.2〜20.4M が全ゲーム 0 点、25.45M が 243.8（400 点 9、100 点 3、0 点 4）だった。同じ重みの別 session は 20.4M に 25.0（400 点 1/16）、25.45M に 231.2（9/16）。5.25M 以降の Greedy のゲームは、1 本を除いて 108,000 フレームの打ち切りで終わった（出典: [2026-11-08_montezuma-rnd-se-30m.md](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-se-30m.md)、現時点のまとめ1・探索ブロック01。Run: run_20261108-003056_mz_se_rnd_k32_s1）。
- 同じ Run の ε=0.01・τ 乱数の評価（10 ゲーム、約 2.0M ごと）は、14.21〜22.21M の 5 回が 400 点で揃い、24.21M は 360（400 点のゲームは部屋 7 に入って game over）、26.21M は 120、28.21M は 400 だった（出典: [2026-11-08_montezuma-rnd-se-30m.md](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-se-30m.md)、現時点のまとめ2・探索ブロック01。Run: run_20261108-003056_mz_se_rnd_k32_s1）。
- 同じ Run の学習側では、2M の窓で 500 点のゲームが 22〜24M の 0 本から 24〜26M の 87 本、26〜28M の 106 本へ増え、部屋 7 に入ったゲームは 24〜26M に 206 本だった（22〜24M は 0 本）（出典: [2026-11-08_montezuma-rnd-se-30m.md](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-se-30m.md)、現時点のまとめ3・探索ブロック02。Run: run_20261108-003056_mz_se_rnd_k32_s1）。
- 30M の重みを評価だけで 4 条件・各 32 ゲーム測ると、ε=0・τ 固定 259.4（400 点以上 19/32）、ε=0・τ 乱数 362.5（29/32）、ε=0.01・τ 固定 403.1（32/32）、ε=0.01・τ 乱数 412.5（32/32）だった。500 点（`room_max` 6）は ε=0.01 の 2 条件だけに出た。ε=0・τ 固定で 400 に届かなかった 13 本は部屋 1 のまま打ち切りまで続いた（出典: [2026-11-08_montezuma-rnd-se-30m.md](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-se-30m.md)、現時点のまとめ5・探索ブロック04。Run: run_20261108-110233_mz_se_ev30m_2x2）。
- exp step 20,066,688 で保存した重みは、4 条件とも全ゲーム 400 点だった。学習 Run の中で 0/16 だった Greedy の session（exp step 20,400,384）は、この重みより約 5,200 learn step あとである（出典: [2026-11-08_montezuma-rnd-se-30m.md](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-se-30m.md)、現時点のまとめ6・探索ブロック04。Run: run_20261108-111508_mz_se_ev20m_2x2）。

## 分かっていないこと

なし。

## 撤回・更新

なし。

## 関連

- [atari](atari.md)。
- [220 §4.9 RAM メトリクス](../../design/220_atari_env.jp.md)。
- [内的報酬による探索の比較](../mechanisms/intrinsic_novelty.md)。
- [評価と比較の条件](../mechanisms/evaluation.md)。
- [τ配置とIQN比較](../mechanisms/tau_sampling.md)。
- [RND の腕の暫定 PRD](../../memo/116_rnd_lifelong_novelty_10prd.md)。
- [Montezuma's Revenge の RND（サンプル効率優先・30M）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-se-30m.md)。

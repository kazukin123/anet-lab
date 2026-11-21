# Double Dunk

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [題材](../index.md#題材)

## 今の結論

サンプル効率優先のε=0評価は20.4Mで22.1、45.7Mで23.2（BTR比101%）だった。実時間優先seed2は50.7Mで21.5、91.1Mで23.0に達した。

## 分かっていること

- 実時間優先では25M前後の相転移で負のscore帯を抜け、初回20Mの不振はその手前で切った観測だった。（出典: [2026-09-21_realtime-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、現時点のまとめ2。Run: run_20260922-020046_a5_double_dunk_rr1_50m）。
- サンプル効率優先は20M付近で高い水準に達し、特徴の痩せがあっても成績の天井へ届いた。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ6・9。Run: run_20260927-101048_a5_double_dunk_btr50m）。

- サンプル効率優先のε=0評価は20.4Mで22.1、45.7Mで23.2（BTR比101%）だった。実時間優先seed2は50.7Mで21.5、91.1Mで23.0だった。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ6、探索ブロック02。Run: `run_20260927-101048_a5_double_dunk_btr50m` / [実時間優先](../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、現時点のまとめ13・探索ブロック04。Run: `run_20260924-144556_a5_double_dunk_rr1_100m_seed2`）。
- 実時間優先 50M で ε ラダーを NoisyNet に置き換える（ε 0）と、50M まで相転移が起きず、eval2 は −23.9 前後、ε=0 断面は −0.7〜−1.0 のままで、ノイズで行動が変わる割合は 5M 以降 0.0% だった。NoisyNet をラダーに足すと相転移は戻ったが、eval2 が正になるのは 40-45M（基準は 20-25M）で、40-50M は 11.5、45.7M の ε=0 断面は 18.5 だった（出典: [2026-10-11_noisynet-realtime.md](../../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md)、現時点のまとめ3・4、探索ブロック 01〜03。Run: `run_20261013-012055_a5_double_dunk_rr1_50m_nz_eps0`, `run_20261015-110003_a5_double_dunk_rr1_50m_nz_ladder`）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。

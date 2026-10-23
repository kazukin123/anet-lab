# Double Dunk

## 今の結論

サンプル効率優先のε=0評価は20.4Mで22.1、45.7Mで23.2（BTR比101%）だった。実時間優先seed2は50.7Mで21.5、91.1Mで23.0に達した。

## 分かっていること

- 実時間優先では25M前後の相転移で負のscore帯を抜け、初回20Mの不振はその手前で切った観測だった。（出典: [2026-09-21_realtime-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、現時点のまとめ2。Run: run_20260922-020046_a5_double_dunk_rr1_50m）。
- サンプル効率優先は20M付近で高い水準に達し、特徴の痩せがあっても成績の天井へ届いた。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ6・9。Run: run_20260927-101048_a5_double_dunk_btr50m）。

- サンプル効率優先のε=0評価は20.4Mで22.1、45.7Mで23.2（BTR比101%）だった。実時間優先seed2は50.7Mで21.5、91.1Mで23.0だった。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ6、探索ブロック02。Run: `run_20260927-101048_a5_double_dunk_btr50m` / [実時間優先](../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、現時点のまとめ13・探索ブロック04。Run: `run_20260924-144556_a5_double_dunk_rr1_100m_seed2`）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。

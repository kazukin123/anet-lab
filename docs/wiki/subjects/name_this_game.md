# Name This Game

## 今の結論

サンプル効率優先50Mは実時間優先50M比+24%、BTR比102%だった。45.7M・ε=0の最良28,465は、実時間優先seed2・100Mの最良24,851を上回った。

## 分かっていること

- RR1と旧netではcapacity 2M→4Mでageやcountが増えても成績差はブレ幅内だった。Breakoutの結果とはゲーム・RR・netが同時に違い、原因は分離されていない。（出典: [2026-09-21_realtime-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、現時点のまとめ5・6。Run: run_20260921-213901_a5_name_this_game_rr1_50m）。
- サンプル効率優先は同50Mの実時間優先より良かったが、45.5M以降のmetrics欠落分はRunログから評価値を復元した。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ5。Run: run_20260926-151320_a5_name_this_game_btr50m）。

- サンプル効率優先50Mでは実時間優先50M比+24%、BTR比102%だった。ε=0の最良は45.7Mの28,465で、実時間優先seed2・100Mの最良24,851を上回った。45.5M以降のmetricsは容量切れで欠け、評価値はRunログから復元された。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ5、探索ブロック02。Run: `run_20260926-151320_a5_name_this_game_btr50m`）。
- 実時間をそろえると、実時間優先 seed 2 の 100M（4.76h）と同じ時間でサンプル効率優先は約 31M にいて、ε=0 断面は 30.5M の 24,941 と、実時間優先 seed 2・100M の最良 24,851 に並んだ。1 ゲームが約 1 万 step と長く、評価セッションが走る時間が全体の 64.7% を占め、評価で失った時間は約 1.25h（全体 7.76h の 16%）だった（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、探索ブロック02「name_this_game」。Run: `run_20260926-151320_a5_name_this_game_btr50m`, `run_20260924-095955_a5_name_this_game_rr1_100m_seed2`）。
- replay_capacity 2M → 4M（lane 窓 16,384 → 32,768、RR1・旧 net）の 1 seed の比較の −2.2〜−2.5% からは「改善しなかった」とは言えるが「悪化した」とは言えない。ただし Breakout 並みの効果があれば 2.9〜3.7σ の差として検出できたはずの水準だった（出典: [2026-09-21_realtime-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック02「留保」。Run: `run_20260921-213901_a5_name_this_game_rr1_50m`, `run_20260922-142005_a5_name_this_game_rr1_cap4m_50m`）。
- 実時間優先 50M で ε ラダーを NoisyNet に置き換える（ε 0）と、序盤が遅く（eval2 の 0-25M の 5M 窓は 1,913〜8,034、基準 seed 1 は 7,082〜20,331）、40-50M は 20,540（基準 seed 1 / 2 は 22,903 / 21,544）だった（出典: [2026-10-11_noisynet-realtime.md](../../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md)、現時点のまとめ1・探索ブロック 01。Run: `run_20261013-102213_a5_name_this_game_rr1_50m_nz_eps0`）。

## 分かっていないこと

- lane窓の効果がBreakoutと異なった原因は、ゲーム・RR・NN構成のどれか（出典: [2026-09-21のcampaign](../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、現時点のまとめ6・探索ブロック02。3軸を同時に変えたため原因は未分離）？

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。

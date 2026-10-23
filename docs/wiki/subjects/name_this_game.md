# Name This Game

## 今の結論

サンプル効率優先50Mは実時間優先50M比+24%、BTR比102%だった。45.7M・ε=0の最良28,465は、実時間優先seed2・100Mの最良24,851を上回った。

## 分かっていること

- RR1と旧netではcapacity 2M→4Mでageやcountが増えても成績差はブレ幅内だった。Breakoutの結果とはゲーム・RR・netが同時に違い、原因は分離されていない。（出典: [2026-09-21_realtime-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、現時点のまとめ5・6。Run: run_20260921-213901_a5_name_this_game_rr1_50m）。
- サンプル効率優先は同50Mの実時間優先より良かったが、45.5M以降のmetrics欠落分はRunログから評価値を復元した。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ5。Run: run_20260926-151320_a5_name_this_game_btr50m）。

- サンプル効率優先50Mでは実時間優先50M比+24%、BTR比102%だった。ε=0の最良は45.7Mの28,465で、実時間優先seed2・100Mの最良24,851を上回った。45.5M以降のmetricsは容量切れで欠け、評価値はRunログから復元された。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ5、探索ブロック02。Run: `run_20260926-151320_a5_name_this_game_btr50m`）。

## 分かっていないこと

- lane窓の効果がBreakoutと異なった原因は、ゲーム・RR・NN構成のどれか（出典: [2026-09-21のcampaign](../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、現時点のまとめ6・探索ブロック02。3軸を同時に変えたため原因は未分離）？

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。

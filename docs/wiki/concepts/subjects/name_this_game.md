# Name This Game

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [題材](../../index.md#題材)

## 今の結論

サンプル効率優先50Mは実時間優先50M比+24%、BTR比102%だった。45.7M・ε=0の最良28,465は、実時間優先seed2・100Mの最良24,851を上回った。

## 分かっていること

### anet の知見

- 容量を倍にすると経験の古さは変わったが、得点の改善は確認できなかった → 詳しくは [Replay ratio と lane 窓](../methods/replay_ratio_lane_window.md)（出典: [実時間優先の Atari-5 記録](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック 02）。
- サンプル効率優先は同50Mの実時間優先より良かったが、45.5M以降のmetrics欠落分はRunログから評価値を復元した。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ5。Run: run_20260926-151320_a5_name_this_game_btr50m）。
- サンプル効率優先50Mでは実時間優先50M比+24%、BTR比102%だった。ε=0の最良は45.7Mの28,465で、実時間優先seed2・100Mの最良24,851を上回った。45.5M以降のmetricsは容量切れで欠け、評価値はRunログから復元された。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ5、探索ブロック02。Run: `run_20260926-151320_a5_name_this_game_btr50m`）。
- 実時間をそろえると、実時間優先 seed 2 の 100M（4.76h）と同じ時間でサンプル効率優先は約 31M にいて、ε=0 断面は 30.5M の 24,941 と、実時間優先 seed 2・100M の最良 24,851 に並んだ。1 ゲームが約 1 万 step と長く、評価セッションが走る時間が全体の 64.7% を占め、評価で失った時間は約 1.25h（全体 7.76h の 16%）だった（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、探索ブロック02「name_this_game」。Run: `run_20260926-151320_a5_name_this_game_btr50m`, `run_20260924-095955_a5_name_this_game_rr1_100m_seed2`）。
- 容量増による悪化を判定できるだけの複製はない → 詳しくは [Replay ratio と lane 窓](../methods/replay_ratio_lane_window.md)（出典: [実時間優先の Atari-5 記録](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック 02）。
- 実時間優先 50M で ε ラダーを NoisyNet に置き換える（ε 0）と、序盤が遅く（eval2 の 0-25M の 5M 窓は 1,913〜8,034、基準 seed 1 は 7,082〜20,331）、40-50M は 20,540（基準 seed 1 / 2 は 22,903 / 21,544）だった（出典: [2026-10-11_noisynet-realtime.md](../../../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md)、現時点のまとめ1・探索ブロック 01。Run: `run_20261013-102213_a5_name_this_game_rr1_50m_nz_eps0`）。
- 初回20Mで12,316点に達し、人間基準8,049点を超えた。15–20Mでも上昇が続いた（出典: [Atari-5 横断スイープ 探索履歴](../../../experiments/default-dqn/atari/2026-08-27_atari5.md)、現時点のまとめ・参照 Run。Run: `run_20260827-152639_a5_name_this_game_apex`）。

### 外の事例

なし

## 横断的知見

なし

## 見立て

なし

## 分かっていないこと

- lane窓の効果がBreakoutと異なった原因は、ゲーム・RR・NN構成のどれか（出典: [2026-09-21のcampaign](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、現時点のまとめ6・探索ブロック02。3軸を同時に変えたため原因は未分離）？

## 撤回・更新

なし。

## 関連

- [atari](atari.md)。
- [evaluation](../measurement/evaluation.md)。
- [Breakout と結果が異なった Replay 容量比較](../methods/replay_ratio_lane_window.md#横断的知見)。

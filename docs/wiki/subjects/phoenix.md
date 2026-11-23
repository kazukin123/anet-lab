# Phoenix

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [題材](../index.md#題材)

## 今の結論

サンプル効率優先45.7M・ε=0の最良399,895はBTR Table A2比94%だった。同断面の100本中44本が27,000 step上限で打ち切られた。

## 分かっていること

- 初回20Mでは長時間生存とscore低下が同時に起き、途中でbad allocationにより終了した。（出典: [2026-08-27_atari5.md](../../experiments/default-dqn/atari/2026-08-27_atari5.md)、現時点のまとめ6。Run: run_20260827-162911_a5_phoenix_apex）。
- サンプル効率優先の45.7M断面では100本中44本が27,000 stepの上限に届き、上限が成績を抑え始めた。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ4。Run: run_20260925-174945_a5_phoenix_btr50m）。

- サンプル効率優先45.7M・ε=0の最良は399,895で、実時間優先の2.2〜2.6倍、BTR Table A2の427,481の94%だった。同断面の100本中44本は27,000 step上限で打ち切られ、打ち切り組の平均は454,277だった。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ4、探索ブロック02。Run: `run_20260925-174945_a5_phoenix_btr50m`）。
- 45.7M の ε=0 断面では 1 ゲームあたり 162.4 ウェーブを抜けており、1 ウェーブあたり約 2,460 点は 40.6M（約 2,430 点）とほぼ同じで、得点の伸びは抜けたウェーブ数の伸びだった。5.3M の断面では 100 本中 59 本が 27,000 step の上限で打ち切られ、得点は約 27,000 点で、稼がずに生き残る状態だった（10.3M では 0 本）（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、探索ブロック02「phoenix」。Run: `run_20260925-174945_a5_phoenix_btr50m`）。
- 実時間優先 50M で ε ラダーを NoisyNet に置き換える（ε 0）と、eval2 40-50M は 62,951（基準 seed 1 / 2 は 114,159 / 86,981）だった（出典: [2026-10-11_noisynet-realtime.md](../../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md)、現時点のまとめ1・探索ブロック 01。Run: `run_20261012-090712_a5_phoenix_rr1_50m_nz_eps0`）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。

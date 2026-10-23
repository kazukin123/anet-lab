# Phoenix

## 今の結論

サンプル効率優先45.7M・ε=0の最良399,895はBTR Table A2比94%だった。同断面の100本中44本が27,000 step上限で打ち切られた。

## 分かっていること

- 初回20Mでは長時間生存とscore低下が同時に起き、途中でbad allocationにより終了した。（出典: [2026-08-27_atari5.md](../../experiments/default-dqn/atari/2026-08-27_atari5.md)、現時点のまとめ6。Run: run_20260827-162911_a5_phoenix_apex）。
- サンプル効率優先の45.7M断面では100本中44本が27,000 stepの上限に届き、上限が成績を抑え始めた。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ4。Run: run_20260925-174945_a5_phoenix_btr50m）。

- サンプル効率優先45.7M・ε=0の最良は399,895で、実時間優先の2.2〜2.6倍、BTR Table A2の427,481の94%だった。同断面の100本中44本は27,000 step上限で打ち切られ、打ち切り組の平均は454,277だった。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ4、探索ブロック02。Run: `run_20260925-174945_a5_phoenix_btr50m`）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。

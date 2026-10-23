# Replay ratioとlane窓

## 今の結論

Breakoutの該当構成ではlane窓16,384の2点がε=0で685.12・685.30、窓8,192が615.73だった。効果の機序は未測定である。

## 分かっていること

- Breakoutの該当構成ではlane窓16,384の2点はε=0で685.12と685.30、8,192は615.73だった。更新数だけの説明は3点を追わず、機序は未測である。（出典: [2026-09-15_replay-lane-window.md](../../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md)、現時点のまとめ1〜6。Run: run_20260914-231706_rr4_btrfull_envs128_50m / run_20260915-203706_rr4_btrfull_envs128_cap2m_50m）。
- B512で更新頻度を揃えたRR4はRR4/B256と同水準で、RR2はそれらより良かった。この条件では勾配頻度より経験再利用回数に成績が追従した。（出典: [2026-09-08_replay-ratio-mechanism.md](../../experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md)、現時点のまとめ1〜3。Run: run_20260909-142738_rr4_b512_munch / run_20260908-201708_rr2_munch）。
- PQLの連続制御での容量比例不要という結果と、Atariのlane窓の観測を同一条件の反証として扱わない。（出典: [replay_lane_window_survey_2026-09-17.md](../../../reports/replay_lane_window_survey_2026-09-17.md)、2. 分散アクター構成における容量設計）。

## 分かっていないこと

- Breakoutでlane窓を伸ばして成績が変わった機序は何か（出典: [2026-09-15のcampaign](../../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md)、現時点のまとめ2・探索ブロック02「何が未測か」）？

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。

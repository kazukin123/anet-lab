# Atari横断

## 今の結論

学習中最良のAtari-5回帰で、実時間優先のBTR比は50M・seed1で66.8%、100M・seed2で83.8%だった。サンプル効率優先50Mは458.3、BTR比82.5%に達した。

## 分かっていること

- 2026-08の調査では、ALEはC++ APIで状態・RAM・flavorを扱える環境として整理された。前処理とROM・ライセンスの条件は別に確認する。（出典: [atari_env_survey_2026-08-13.md](../../../reports/atari_env_survey_2026-08-13.md)、8. 総合評価）。
- 初回20Mスイープの55.4% HNS推定は、複数ゲームがまだ伸びていた時点の途中経過である。（出典: [2026-08-27_atari5.md](../../experiments/default-dqn/atari/2026-08-27_atari5.md)、現時点のまとめ。Run: run_20260827-133241_a5_battle_zone_apex / run_20260827-172101_a5_qbert_apex）。
- 実時間優先50Mでは5ゲーム中4つが終盤も上昇中だった。ゲーム間throughput差の多くはeval負荷に由来した。（出典: [2026-09-21_realtime-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、現時点のまとめ。Run: run_20260921-155634_a5_qbert_rr1_50m / run_20260922-063422_a5_phoenix_rr1_50m）。
- サンプル効率優先は同じ50Mで改善したが、同程度のAtari-5水準へ至る実時間は実時間優先より長かった。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ。Run: run_20260925-174945_a5_phoenix_btr50m / run_20260926-022413_a5_qbert_btr50m）。
- PQNとBTRなどの時間比較ではsticky actions、episodic life、frame予算とhardwareを揃える必要がある。環境だけのFPSを学習込みthroughputと同一視しない。（出典: [atari_walltime_frontier_survey_2026-09-09.md](../../../reports/atari_walltime_frontier_survey_2026-09-09.md)、12. 総合評価・13. 調査限界）。

- 実時間優先50M・seed1のeval2・40–50M窓ではAtari-5 medianが358.0% HNS、回帰が344.8%だった。学習中最良を使うBTRの集計では回帰370.7対555.2、BTR比66.8%だった。（出典: [2026-09-21_realtime-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、現時点のまとめ1・3、探索ブロック01。Run: `run_20260921-155634_a5_qbert_rr1_50m`, `run_20260921-183247_a5_battle_zone_rr1_50m`, `run_20260921-213901_a5_name_this_game_rr1_50m`, `run_20260922-020046_a5_double_dunk_rr1_50m`, `run_20260922-063422_a5_phoenix_rr1_50m`）。
- 実時間優先100M・seed2では90–100M窓の回帰444.9、median463.3で、5ゲームとも同Run内の最高窓だった。学習中最良の回帰465.5はBTR555.2の83.8%だった。（出典: [2026-09-21_realtime-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、現時点のまとめ12・14、探索ブロック04。Run: `run_20260923-163113_a5_qbert_rr1_100m_seed2`, `run_20260923-203734_a5_battle_zone_rr1_100m_seed2`, `run_20260924-095955_a5_name_this_game_rr1_100m_seed2`, `run_20260924-144556_a5_double_dunk_rr1_100m_seed2`, `run_20260925-005343_a5_phoenix_rr1_100m_seed2`）。
- サンプル効率優先50Mの学習中最良の回帰は458.3、BTR555.2の82.5%。同50Mの実時間優先の2 seedを8.9〜23.6%上回った。ゲーム別BTR比はphoenix94%、name_this_game102%、double_dunk101%、battle_zone68%、qbert60%だった。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ1・3、探索ブロック01。Run: `run_20260925-174945_a5_phoenix_btr50m`, `run_20260926-022413_a5_qbert_btr50m`, `run_20260926-082748_a5_battle_zone_btr50m`, `run_20260926-151320_a5_name_this_game_btr50m`, `run_20260927-101048_a5_double_dunk_btr50m`）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [replay_ratio_lane_window](../mechanisms/replay_ratio_lane_window.md)。
- [campaignにない独自の分析](../../experiments/default-dqn/atari/README.md#独自の分析)。

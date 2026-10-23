# 行動の時間尺度

## 今の結論

TempoRLは行動へのコミット長を学ぶ設定で素のQ学習より速い学習を、FiGARはA3C・TRPO・DDPGへの行動反復追加で改善を報告した。小さい時間刻みでQ学習が失敗する条件も報告されている。

## 分かっていること

- TempoRLは行動へのコミット長を学ぶ設定で素のQ学習より速い学習を報告するが、anetでの実測ではない。（出典: [biedenkapp2021_temporl_paper_2026-10-03.md](../../../reports/biedenkapp2021_temporl_paper_2026-10-03.md)、アブストラクト）。
- FiGARはA3C・TRPO・DDPGへ行動反復を組み合わせた性能改善を報告している。DQNやDropMergeへそのまま移植した結果ではない。（出典: [sharma2017_figar_paper_2026-10-03.md](../../../reports/sharma2017_figar_paper_2026-10-03.md)、要旨）。
- 小さな時間刻みでQ学習が失敗する条件と、時間刻み変更に頑健な方法の実験が報告されている。単なるframe skip変更の同値性を保証する結果ではない。（出典: [tallec2019_time_discretization_paper_2026-10-03.md](../../../reports/tallec2019_time_discretization_paper_2026-10-03.md)、7. 結論）。
- action gapを広げる作用素について、最適性の十分条件と自転車・Atari実験が報告された。現行のMunchausenの効果量と同じ証拠にはしない。（出典: [bellemare2016_increasing_action_gap_paper_2026-10-03.md](../../../reports/bellemare2016_increasing_action_gap_paper_2026-10-03.md)、概要）。
- DropMerge では 1 step の NOOP は盤面をほとんど変えないので、その価値は次の step で最善手を打つ価値を γ で 1 回割り引いたものとほぼ同じになり、最善の DROP との本来の差は (1−γ)·V ≈ 0.3% しかない。推定の揺れで順位が入れ替わる（出典: [999_smdp_transition_duration_10prd.md](../../memo/999_smdp_transition_duration_10prd.md)、§1）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。

# Target更新の比較

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機構](../index.md#機構)

## 今の結論

n=2でhard C=125だけが低分散という主張は、n=4で撤回された。DropMergeのIQN＋Munchausenでは同時定数のhard C=200がsoftの2 seedを下回り、NOOPも増えた。

## 分かっていること

- target更新はenv step・gradient step・soft係数の単位が混在し、target networkを使うだけで収束を保証しない反例も報告されている。（出典: [target_network_update_rate_survey_2026-09-03.md](../../../reports/target_network_update_rate_survey_2026-09-03.md)、1. 単位の不一致という前提・7. 批判と留保）。
- n=2でhard C=125だけが低分散という主張はn=4で撤回された。env時間換算が一致する設定事実と、成績順位の主張を分ける。（出典: [2026-09-01_btr-feedback-arms.md](../../experiments/default-dqn/atari/2026-09-01_btr-feedback-arms.md)、探索ブロック27・28）。
- DropMergeのIQN+Munchausenでは同時定数のhard C=200がsoftの2 seedを下回り、NOOPも増えた。（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ5）。

## 分かっていないこと

なし

## 撤回・更新

- 2026-09-06: hard C=125だけがPoissonの範囲内というn=2の判断 → n=4ではその主張を撤回（出典: [BTR腕別screening](../../experiments/default-dqn/atari/2026-09-01_btr-feedback-arms.md)、探索ブロック28）。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。

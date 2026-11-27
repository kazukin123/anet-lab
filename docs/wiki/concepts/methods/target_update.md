# Target更新の比較

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [手法](../../index.md#手法)

## 今の結論

n=2でhard C=125だけが低分散という主張は、n=4で撤回された。DropMergeのIQN＋Munchausenでは同時定数のhard C=200がsoftの2 seedを下回り、NOOPも増えた。

## 分かっていること

### anet の知見

- n=2でhard C=125だけが低分散という主張はn=4で撤回された。env時間換算が一致する設定事実と、成績順位の主張を分ける。（出典: [2026-09-01_btr-feedback-arms.md](../../../experiments/default-dqn/atari/2026-09-01_btr-feedback-arms.md)、探索ブロック27・28）。
- DropMergeのIQN+Munchausenでは同時定数のhard C=200がsoftの2 seedを下回り、NOOPも増えた。（出典: [2026-10-01_atari-feedback.md](../../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ5）。
- Nature DQN の付録にある target の10,000 parameter updates と、公開 Lua 実装の10,000 agent steps は異なる。後者は update_freq=4 で2,500 parameter updates に相当する（調査での換算）（出典: [Survey: target network 更新速度（hard C / soft τ）の設定根拠と調整方法](../../../../reports/target_network_update_rate_survey_2026-09-03.md)、1. 単位の不一致という前提）。

### 外の事例

- target更新はenv step・gradient step・soft係数の単位が混在し、target networkを使うだけで収束を保証しない反例も報告されている。（出典: [target_network_update_rate_survey_2026-09-03.md](../../../../reports/target_network_update_rate_survey_2026-09-03.md)、1. 単位の不一致という前提・7. 批判と留保）。
- 調査した収束解析では hard 更新と soft 更新にそれぞれ別の十分条件を与えている。単一の換算で両者の性能まで等価にする定理は見つからなかった（出典: [Survey: target network 更新速度（hard C / soft τ）の設定根拠と調整方法](../../../../reports/target_network_update_rate_survey_2026-09-03.md)、3. 更新速度と収束・バイアスの理論）。
- DDPG は soft target を使う理由を説明したが、調査の範囲では τ=0.001 に決めた比較実験や選定経緯を確認できなかった。（出典: [Survey: target network 更新速度（hard C / soft τ）の設定根拠と調整方法](../../../../reports/target_network_update_rate_survey_2026-09-03.md)、2. 主要アルゴリズムの設定値と選定根拠）。
- target の更新速度の感度について、Fu らは、τ を 9 点掃引して 1 桁変えても、データ効率の劣化は 19%（batch size は 52%）に留まるとした（出典: [target_network_update_rate_survey_2026-09-03.md](../../../../reports/target_network_update_rate_survey_2026-09-03.md)、7. 批判と留保）。

## 横断的知見

なし

## 見立て

なし

## 分かっていないこと

なし

## 撤回・更新

- 2026-09-06: hard C=125だけがPoissonの範囲内というn=2の判断 → n=4ではその主張を撤回（出典: [BTR腕別screening](../../../experiments/default-dqn/atari/2026-09-01_btr-feedback-arms.md)、探索ブロック28）。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../measurement/evaluation.md)。
- [仕組みと契約](../../../design/200_dqn_agents.jp.md)。

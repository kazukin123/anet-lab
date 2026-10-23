# 勾配ノイズとbatch

## 今の結論

GNSと有効batchの対応はoptimizer・曲率・抽出方法・予算に依存する。DropMergeでは臨界batchが学習とともに増えたが、batch増加の最終target評価の差は学習側の差より小さかった。

## 分かっていること

- GNSと有効batchの対応はoptimizer・曲率・抽出・予算に依存するため、大きなGNSだけでB増加を決めない。（出典: [gradient_noise_observation_and_control_survey_2026-10-15.md](../../../reports/gradient_noise_observation_and_control_survey_2026-10-15.md)、1. 調査の結論・8. 総合評価と調査の限界）。
- DropMergeで臨界batchは学習とともに増えたが、B増加の最終target評価の差は学習側の差より小さかった。（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ1・3〜5）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。

# 報酬尺度と割引率

## 今の結論

調査ではDQN系・約200M framesで報酬clipだけを外した整ったA/Bは確認できなかった。DropMergeのγ0.999は100Mで基準とほぼ同点だったが、序盤と中盤の推移は異なった。

## 分かっていること

- 調査ではDQN系・約200M framesのままclipだけを外す整ったA/Bを確認できず、別予算・別手法の結果をその代替にはしていない。（出典: [reward_scale_discount_survey_2026-09-23.md](../../../reports/reward_scale_discount_survey_2026-09-23.md)、8. 総合評価・9. 調査の限界）。
- DropMergeのγ0.999は100Mで基準とほぼ同点だったが、序盤と中盤で推移が異なった。（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ10）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [checkpoint 引き継ぎでは optimizer の options を現在の設定で置き換える](../../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md)。
- [checkpoint 引き継ぎで optimizer の設定（学習率ほか）を効かせる PRD](../../memo/091_checkpoint_load_optimizer_options_10prd.md)。
- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。

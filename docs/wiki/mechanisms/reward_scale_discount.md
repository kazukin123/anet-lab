# 報酬尺度と割引率

## 今の結論

調査ではDQN系・約200M framesで報酬clipだけを外した整ったA/Bは確認できなかった。DropMergeのγ0.999は100Mで基準とほぼ同点だったが、序盤と中盤の推移は異なった。

## 分かっていること

- 調査ではDQN系・約200M framesのままclipだけを外す整ったA/Bを確認できず、別予算・別手法の結果をその代替にはしていない。（出典: [reward_scale_discount_survey_2026-09-23.md](../../../reports/reward_scale_discount_survey_2026-09-23.md)、8. 総合評価・9. 調査の限界）。
- DropMergeのγ0.999は100Mで基準とほぼ同点だったが、序盤と中盤で推移が異なった。（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ10）。
- 報酬クリップの代わりに Pop-Art や h 変換などを使うと、集計ではクリップとほぼ同等かやや劣った。A2C ではクリップが Pop-Art に対して有意な優位を保ち、R2D2 系の単一ヘッドでは集計で最良だったのはクリップだった（出典: [reward_scale_discount_survey_2026-09-23.md](../../../reports/reward_scale_discount_survey_2026-09-23.md)、3.3）。
- h だけで 0.997 を超える γ を安定させたと報告した原典は見つからなかった。γ = 0.999 の Ape-X DQfD では TC loss が必要で、γ = 0.9999 の固定は族とメタコントローラで回避されていた（出典: [reward_scale_discount_survey_2026-09-23.md](../../../reports/reward_scale_discount_survey_2026-09-23.md)、4.3）。
- BBF の公式スコア CSV から調査が計算すると、γ と n のアニールの効果の大部分は n のスケジュールによるもので、γ のスケジュールの効果は小さかった。BTR は γ を 0.97 から 0.997 へアニールしても有意差がなかったと報告している（BattleZone の 1 ゲーム）（出典: [reward_scale_discount_survey_2026-09-23.md](../../../reports/reward_scale_discount_survey_2026-09-23.md)、5.2）。
- h の ε の既定値は、rlax、Acme の R2D2 設定、SEED RL の R2D2 のいずれも 1e-3 だった（出典: [reward_scale_discount_survey_2026-09-23.md](../../../reports/reward_scale_discount_survey_2026-09-23.md)、6.2）。

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

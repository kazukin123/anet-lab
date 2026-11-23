# 報酬尺度と割引率

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機構](../index.md#機構)

## 今の結論

調査ではDQN系・約200M framesで報酬clipだけを外した整ったA/Bは確認できなかった。DropMergeのγ0.999は100Mで基準とほぼ同点だったが、序盤と中盤の推移は異なった。遷移ごとの割引でエピソード型と継続型を統一する定式化の保証は固定方策の評価と線形近似までで、DQNとReplayの分布は範囲外である。

## 分かっていること

- 調査ではDQN系・約200M framesのままclipだけを外す整ったA/Bを確認できず、別予算・別手法の結果をその代替にはしていない。（出典: [reward_scale_discount_survey_2026-09-23.md](../../../reports/reward_scale_discount_survey_2026-09-23.md)、8. 総合評価・9. 調査の限界）。
- DropMergeのγ0.999は100Mで基準とほぼ同点だったが、序盤と中盤で推移が異なった。（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ10）。
- 報酬クリップの代わりに Pop-Art や h 変換などを使うと、集計ではクリップとほぼ同等かやや劣った。A2C ではクリップが Pop-Art に対して有意な優位を保ち、R2D2 系の単一ヘッドでは集計で最良だったのはクリップだった（出典: [reward_scale_discount_survey_2026-09-23.md](../../../reports/reward_scale_discount_survey_2026-09-23.md)、3.3）。
- h だけで 0.997 を超える γ を安定させたと報告した原典は見つからなかった。γ = 0.999 の Ape-X DQfD では TC loss が必要で、γ = 0.9999 の固定は族とメタコントローラで回避されていた（出典: [reward_scale_discount_survey_2026-09-23.md](../../../reports/reward_scale_discount_survey_2026-09-23.md)、4.3）。
- BBF の公式スコア CSV から調査が計算すると、γ と n のアニールの効果の大部分は n のスケジュールによるもので、γ のスケジュールの効果は小さかった。BTR は γ を 0.97 から 0.997 へアニールしても有意差がなかったと報告している（BattleZone の 1 ゲーム）（出典: [reward_scale_discount_survey_2026-09-23.md](../../../reports/reward_scale_discount_survey_2026-09-23.md)、5.2）。
- 割引を MDP でなくタスクの一部として遷移ごとに γ(s,a,s′) と書くと、エピソードの終わりを「開始状態へ普通に遷移しつつ γ=0 を返す遷移」として書け、エピソード型と継続型を吸収状態なしで統一できる。状態ごとの割引では、仮想の状態を足さないと終わる遷移と終わらない遷移を区別できない場合がある。遷移によって割引が違うと最適方策の選択そのものが変わり、定数割引と平均報酬の対応は成り立たない。保証は固定した方策の評価と線形の関数近似までで、max を含む制御の作用素、DQN、Replay の分布は範囲外である（出典: [white2017_unifying_task_specification_paper_2026-11-04.md](../../../reports/white2017_unifying_task_specification_paper_2026-11-04.md)、2.1・C. 割引と平均報酬・キューの論点への答え 1・2）。
- h の ε の既定値は、rlax、Acme の R2D2 設定、SEED RL の R2D2 のいずれも 1e-3 だった（出典: [reward_scale_discount_survey_2026-09-23.md](../../../reports/reward_scale_discount_survey_2026-09-23.md)、6.2）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [checkpoint 引き継ぎでは optimizer の options を現在の設定で置き換える](../../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md)。
- [checkpoint 引き継ぎで optimizer の設定（学習率ほか）を効かせる RFD](../../memo/done/0091_checkpoint_load_optimizer_options_10rfd.md)。
- [atari](../subjects/atari.md)。
- [行動の時間尺度](temporal_abstraction.md)（SMDP の経過時間と NOOP の価値）。
- [evaluation](../mechanisms/evaluation.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。

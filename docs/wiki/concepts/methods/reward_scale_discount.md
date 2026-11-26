# 報酬尺度と割引率

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [手法](../../index.md#手法)

## 今の結論

調査ではDQN系・約200M framesで報酬clipだけを外した整ったA/Bは確認できなかった。DropMergeのγ0.999は100Mで基準とほぼ同点だったが、序盤と中盤の推移は異なった。遷移ごとの割引でエピソード型と継続型を統一する定式化の保証は固定方策の評価と線形近似までで、DQNとReplayの分布は範囲外である。

## 分かっていること

### anet の知見

- DropMergeのγ0.999は100Mで基準とほぼ同点だったが、序盤と中盤で推移が異なった。（出典: [2026-10-01_atari-feedback.md](../../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ10）。
- BBF の公式スコア CSV から調査が計算すると、γ と n のアニールの効果の大部分は n のスケジュールによるもので、γ のスケジュールの効果は小さかった。BTR は γ を 0.97 から 0.997 へアニールしても有意差がなかったと報告している（BattleZone の 1 ゲーム）（出典: [reward_scale_discount_survey_2026-09-23.md](../../../../reports/reward_scale_discount_survey_2026-09-23.md)、5.2）。
- Munchausen の温度 τ は Q 値差を基準に効く。報酬尺度が違う環境への適用では 36_agent_munchausen の 03_clip_ratio と 04_bonus_mean を確認し、その環境の Run で採否を決める（出典: [DefaultDQNAgent ハイパーパラメータ探索](../../../experiments/default-dqn/README.md)、Munchausenの温度と別Envへの適用）。
- Kung-Fu Master では報酬クリップを外して1/200に縮小するとボスに挑む回数が増えたが、勝ち方は定着しなかった。（出典: [kung_fu_master のボス行きの定着 探索履歴](../../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md)、現時点のまとめ・報酬クリップの比較、`run_20261015-210918_kfm_rr1_50m_noclip200_seed1`・`run_20261016-042902_kfm_rr1_50m_noclip200_seed2`）。
- 有限の軌跡でポテンシャル差を足すと終端の項が残る。調査の検算では、この項が方策に依存する場合の順位保存は保証できず、環境の終了と timeout の境界条件を分けた。（出典: [LLMの採点を報酬へ加えるときの保証と目標切り替え](../../../../reports/llm_reward_shaping_guarantees_2026-11-07.md)、4. 終端と学習中の変化）。

### 外の事例

- 調査ではDQN系・約200M framesのままclipだけを外す整ったA/Bを確認できず、別予算・別手法の結果をその代替にはしていない。（出典: [reward_scale_discount_survey_2026-09-23.md](../../../../reports/reward_scale_discount_survey_2026-09-23.md)、8. 総合評価・9. 調査の限界）。
- 報酬クリップの代わりに Pop-Art や h 変換などを使うと、集計ではクリップとほぼ同等かやや劣った。A2C ではクリップが Pop-Art に対して有意な優位を保ち、R2D2 系の単一ヘッドでは集計で最良だったのはクリップだった（出典: [reward_scale_discount_survey_2026-09-23.md](../../../../reports/reward_scale_discount_survey_2026-09-23.md)、3.3）。
- h だけで 0.997 を超える γ を安定させたと報告した原典は見つからなかった。γ = 0.999 の Ape-X DQfD では TC loss が必要で、γ = 0.9999 の固定は族とメタコントローラで回避されていた（出典: [reward_scale_discount_survey_2026-09-23.md](../../../../reports/reward_scale_discount_survey_2026-09-23.md)、4.3）。
- 割引を MDP でなくタスクの一部として遷移ごとに γ(s,a,s′) と書くと、エピソードの終わりを「開始状態へ普通に遷移しつつ γ=0 を返す遷移」として書け、エピソード型と継続型を吸収状態なしで統一できる。状態ごとの割引では、仮想の状態を足さないと終わる遷移と終わらない遷移を区別できない場合がある。遷移によって割引が違うと最適方策の選択そのものが変わり、定数割引と平均報酬の対応は成り立たない。保証は固定した方策の評価と線形の関数近似までで、max を含む制御の作用素、DQN、Replay の分布は範囲外である（出典: [white2017_unifying_task_specification_paper_2026-11-04.md](../../../../reports/white2017_unifying_task_specification_paper_2026-11-04.md)、2.1・C. 割引と平均報酬・キューの論点への答え 1・2）。
- h の ε の既定値は、rlax、Acme の R2D2 設定、SEED RL の R2D2 のいずれも 1e-3 だった（出典: [reward_scale_discount_survey_2026-09-23.md](../../../../reports/reward_scale_discount_survey_2026-09-23.md)、6.2）。
- Agent57 は内外の報酬の尺度と分散に合わせて価値 Network を分け、actor ごとに割引なしの外的収益で方策を選ぶ。MEME は胴体共有・head 分離で中央のバンディットに変更し、head 一つでも同程度だった。両者の設定と学習予算は異なる（出典: [内的報酬の手法の全体像](../../../../reports/intrinsic_reward_landscape_survey_2026-11-18.md)、§4.2・6.3）。
- 有界で固定したポテンシャル、無限期間、割引率が1未満の条件ではポテンシャル差の割引和が開始状態だけで決まり、方策順位を保存する。有限の打ち切りでは最終項の境界条件も確認する必要がある（出典: [LLMの採点を報酬へ加えるときの保証と目標切り替え](../../../../reports/llm_reward_shaping_guarantees_2026-11-07.md)、1. 何を保証する定理か）。
- 固定 γ の IMPALA 比較では0.995、0.998、0.999の順に211.9%、208.5%、114.9%だった。試した条件では割引率を大きくするほど良いという関係はなかった（出典: [Survey: 報酬のスケール処理と割引率(報酬クリップの代替、h(x)・Pop-Art、γ の設定とスケジュール)](../../../../reports/reward_scale_discount_survey_2026-09-23.md)、5. 学習中に γ と n-step を変えるスケジュール「5.3 メタ勾配による γ の適応」）。
- White の例では同じ状態へ入る2つの遷移のうち片方だけが終了する。次状態だけで決める割引では元の状態集合のまま表せず、遷移依存割引なら区別できる。仮想状態を追加する表現も可能である（出典: [Unifying Task Specification in Reinforcement Learning](../../../../reports/white2017_unifying_task_specification_paper_2026-11-04.md)、2. 問題の一般的な定式化（Generalized problem formulation）「2.1 エピソード型と継続型の統一（Unifying episodic and continuing specification）」）。
- 既定方策へ制御を渡せる MDP では、費用の範囲によって常に任せる・常に制御する両極端を避けられる。下限の推定には通常得られない最適価値などが関係する。（出典: [Lazy-MDPs: Towards Interpretable Reinforcement Learning by Learning When to Act](../../../../reports/jacq2022_lazy_mdps_learning_when_to_act_paper_2026-11-04.md)、6. いつどう行動するかの学習（Learning When and How to Act））。
- R2D2 の5ゲーム・3 seed の Clipped 条件は、h 変換を外すのと報酬をクリップするのを同時に変える。両者の単独寄与はこの比較から決まらない。（出典: [Survey: 報酬のスケール処理と割引率(報酬クリップの代替、h(x)・Pop-Art、γ の設定とスケジュール)](../../../../reports/reward_scale_discount_survey_2026-09-23.md)、3. 報酬クリップの有無を統制した比較・3.3 代替手段を使うと集計ではクリップとほぼ同等かやや劣る）。
- 報酬と割引を条件付きで同時に標本化しても、次状態を固定した条件付き期待を用いれば期待更新式で遷移ごとの割引として扱える。（出典: [Unifying Task Specification in Reinforcement Learning](../../../../reports/white2017_unifying_task_specification_paper_2026-11-04.md)、A. 確率的な割引を含む定式化（More general formulation with probabilistic discounts））。

## 横断的知見

なし

## 見立て

なし

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [checkpoint 引き継ぎでは optimizer の options を現在の設定で置き換える](../../../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md)。
- [checkpoint 引き継ぎで optimizer の設定（学習率ほか）を効かせる RFD](../../../memo/done/0091_checkpoint_load_optimizer_options_10rfd.md)。
- [atari](../subjects/atari.md)。
- [行動の時間尺度](temporal_abstraction.md)（SMDP の経過時間と NOOP の価値）。
- [evaluation](../measurement/evaluation.md)。
- [仕組みと契約](../../../design/200_dqn_agents.jp.md)。

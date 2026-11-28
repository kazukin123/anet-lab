---
title: 報酬尺度と割引率
page_kind: 手法
aliases:
  - reward clip
  - TBO
  - h(x)
  - Pop-Art
  - gamma
  - discount
  - 遷移ごとの割引
  - transition-based discounting
description: 調査ではDQN系・約200M framesで報酬clipだけを外した整ったA/Bは確認できなかった。DropMergeのγ0.999は100Mで基準とほぼ同点だったが、序盤と中盤の推移は異なった。遷移ごとの割引でエピソード型と継続型を統一する定式化の保証は固定方策の評価と線形近似までだった。
updated: 2026-11-27
---
# 報酬尺度と割引率

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [手法](../../index.md#手法)

## 今の結論

調査ではDQN系・約200M framesで報酬clipだけを外した整ったA/Bは確認できなかった。DropMergeのγ0.999は100Mで基準とほぼ同点だったが、序盤と中盤の推移は異なった。遷移ごとの割引でエピソード型と継続型を統一する定式化の保証は固定方策の評価と線形近似までで、DQNとReplayの分布は範囲外である。

## 分かっていること

### anet の知見

- DropMergeのγ0.999は100Mで基準とほぼ同点だったが、序盤と中盤で推移が異なった。（出典: [2026-10-01_atari-feedback.md](../../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ10）。
- BBF の公式スコア CSV から調査が計算すると、γ と n のアニールの効果の大部分は n のスケジュールによるもので、γ のスケジュールの効果は小さかった。BTR は γ を 0.97 から 0.997 へアニールしても有意差がなかったと報告している（BattleZone の 1 ゲーム）（出典: [reward_scale_discount_survey_2026-09-23.md](../../../../reports/survey/reward_scale_discount_survey_2026-09-23.md)、5.2）。
- Munchausen の温度は Q 値差を基準に効くので、報酬尺度が違う環境ではその環境の Run で採否を決める → 詳しくは [Munchausen RL](munchausen.md)（出典: [DefaultDQNAgent ハイパーパラメータ探索](../../../experiments/default-dqn/README.md)、Munchausenの温度と別Envへの適用）。
- Kung-Fu Master では報酬クリップを外して1/200に縮小するとボスに挑む回数が増えたが、勝ち方は定着しなかった。（出典: [kung_fu_master のボス行きの定着 探索履歴](../../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md)、現時点のまとめ・報酬クリップの比較、`run_20261015-210918_kfm_rr1_50m_noclip200_seed1`・`run_20261016-042902_kfm_rr1_50m_noclip200_seed2`）。
- 有限の軌跡でポテンシャル差を足すと終端の項が残る。調査の検算では、この項が方策に依存する場合の順位保存は保証できず、環境の終了と timeout の境界条件を分けた。（出典: [LLMの採点を報酬へ加えるときの保証と目標切り替え](../../../../reports/survey/llm_reward_shaping_guarantees_2026-11-07.md)、4. 終端と学習中の変化）。
- Atari の Montezuma の `mz_n10_lr3e4_s1`（2026-11-22。Montezuma's Revenge の RND（Taiga ら寄せの構成、β と Munchausen））の実効設定では、報酬クリップあり（`AtariEnv.reward_clip = true`）だった（出典: [2026-11-08_montezuma-rnd-taiga-aligned.md](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、目的と比較条件；[run_20261122-174627_mz_n10_lr3e4_s1.txt](../../../experiments/default-dqn/atari/config/run_20261122-174627_mz_n10_lr3e4_s1.txt)、`AtariEnv.reward_clip = true`）。

### 外の事例

- 調査ではDQN系・約200M framesのままclipだけを外す整ったA/Bを確認できず、別予算・別手法の結果をその代替にはしていない。（出典: [reward_scale_discount_survey_2026-09-23.md](../../../../reports/survey/reward_scale_discount_survey_2026-09-23.md)、8. 総合評価・9. 調査の限界）。
- 報酬クリップの代わりに Pop-Art や h 変換などを使うと、集計ではクリップとほぼ同等かやや劣った。A2C ではクリップが Pop-Art に対して有意な優位を保ち、R2D2 系の単一ヘッドでは集計で最良だったのはクリップだった（出典: [reward_scale_discount_survey_2026-09-23.md](../../../../reports/survey/reward_scale_discount_survey_2026-09-23.md)、3.3）。
- h だけで 0.997 を超える γ を安定させたと報告した原典は見つからなかった。γ = 0.999 の Ape-X DQfD では TC loss が必要で、γ = 0.9999 の固定は族とメタコントローラで回避されていた（出典: [reward_scale_discount_survey_2026-09-23.md](../../../../reports/survey/reward_scale_discount_survey_2026-09-23.md)、4.3）。
- 割引を MDP でなくタスクの一部として遷移ごとに γ(s,a,s′) と書くと、エピソードの終わりを「開始状態へ普通に遷移しつつ γ=0 を返す遷移」として書け、エピソード型と継続型を吸収状態なしで統一できる。状態ごとの割引では、仮想の状態を足さないと終わる遷移と終わらない遷移を区別できない場合がある。遷移によって割引が違うと最適方策の選択そのものが変わり、定数割引と平均報酬の対応は成り立たない。保証は固定した方策の評価と線形の関数近似までで、max を含む制御の作用素、DQN、Replay の分布は範囲外である（出典: [white2017_unifying_task_specification_paper_2026-11-04.md](../../../../reports/paper/white2017_unifying_task_specification_paper_2026-11-04.md)、2.1・C. 割引と平均報酬・キューの論点への答え 1・2）。
- h の ε の既定値は、rlax、Acme の R2D2 設定、SEED RL の R2D2 のいずれも 1e-3 だった（出典: [reward_scale_discount_survey_2026-09-23.md](../../../../reports/survey/reward_scale_discount_survey_2026-09-23.md)、6.2）。
- Agent57 は内外の報酬の尺度と分散に合わせて価値 Network を分け、actor ごとに割引なしの外的収益で方策を選ぶ。MEME は胴体共有・head 分離で中央のバンディットに変更し、head 一つでも同程度だった。両者の設定と学習予算は異なる（出典: [内的報酬の手法の全体像](../../../../reports/survey/intrinsic_reward_landscape_survey_2026-11-18.md)、§4.2・6.3；[value_decomposition_head_nstep_survey_2026-11-27.md](../../../../reports/survey/value_decomposition_head_nstep_survey_2026-11-27.md)、§2.1・§2.2）。
- 有界で固定したポテンシャル、無限期間、割引率が1未満の条件ではポテンシャル差の割引和が開始状態だけで決まり、方策順位を保存する。有限の打ち切りでは最終項の境界条件も確認する必要がある（出典: [LLMの採点を報酬へ加えるときの保証と目標切り替え](../../../../reports/survey/llm_reward_shaping_guarantees_2026-11-07.md)、1. 何を保証する定理か）。
- 固定 γ の IMPALA 比較では0.995、0.998、0.999の順に211.9%、208.5%、114.9%だった。試した条件では割引率を大きくするほど良いという関係はなかった（出典: [Survey: 報酬のスケール処理と割引率(報酬クリップの代替、h(x)・Pop-Art、γ の設定とスケジュール)](../../../../reports/survey/reward_scale_discount_survey_2026-09-23.md)、5. 学習中に γ と n-step を変えるスケジュール「5.3 メタ勾配による γ の適応」）。
- White の例では同じ状態へ入る2つの遷移のうち片方だけが終了する。次状態だけで決める割引では元の状態集合のまま表せず、遷移依存割引なら区別できる。仮想状態を追加する表現も可能である（出典: [Unifying Task Specification in Reinforcement Learning](../../../../reports/paper/white2017_unifying_task_specification_paper_2026-11-04.md)、2. 問題の一般的な定式化（Generalized problem formulation）「2.1 エピソード型と継続型の統一（Unifying episodic and continuing specification）」）。
- 既定方策へ制御を渡せる MDP では、費用の範囲によって常に任せる・常に制御する両極端を避けられる。下限の推定には通常得られない最適価値などが関係する。（出典: [Lazy-MDPs: Towards Interpretable Reinforcement Learning by Learning When to Act](../../../../reports/paper/jacq2022_lazy_mdps_learning_when_to_act_paper_2026-11-04.md)、6. いつどう行動するかの学習（Learning When and How to Act））。
- R2D2 の5ゲーム・3 seed の Clipped 条件は、h 変換を外すのと報酬をクリップするのを同時に変える。両者の単独寄与はこの比較から決まらない。（出典: [Survey: 報酬のスケール処理と割引率(報酬クリップの代替、h(x)・Pop-Art、γ の設定とスケジュール)](../../../../reports/survey/reward_scale_discount_survey_2026-09-23.md)、3. 報酬クリップの有無を統制した比較・3.3 代替手段を使うと集計ではクリップとほぼ同等かやや劣る）。
- 報酬と割引を条件付きで同時に標本化しても、次状態を固定した条件付き期待を用いれば期待更新式で遷移ごとの割引として扱える。（出典: [Unifying Task Specification in Reinforcement Learning](../../../../reports/paper/white2017_unifying_task_specification_paper_2026-11-04.md)、A. 確率的な割引を含む定式化（More general formulation with probabilistic discounts））。
- Pop-Art の論文は、クリップを外すとゲームごとの勝ち負けが大きく入れ替わるとし、Ms. Pac-Man と Time Pilot で方策が変わる例を挙げる（出典: [reward_scale_discount_survey_2026-09-23.md](../../../../reports/survey/reward_scale_discount_survey_2026-09-23.md)、3.4）。
- n ごとに action gap を測った実験や、n-step と action gap を結ぶ理論は、調査では見つからなかった → 詳しくは [n-step の収益](nstep_return.md)（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/survey/nstep_action_gap_survey_2026-11-27.md)、対象の問いと結論・§3.5・§7）。
- 推定した Q から作る Greedy の方策の損失は、action gap の規則性 ζ ≥ 0 によって O(‖Q̂ − Q*‖^(1+ζ)) で抑えられ、ζ > 0 なら Q の誤差の大きさから予想するより損失が小さい（Farahmand、NeurIPS 2011。抄録で確かめた）（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/survey/nstep_action_gap_survey_2026-11-27.md)、§1.2）。
- DQN 系の Atari の Agent で、1 回の更新ごとに Greedy の行動が変わる状態の割合（policy churn）は、1-step の Double DQN で約 9%、5-step の R2D2 で約 6% だった（割引率ほかの設定も違い、n を変えた比較ではない）。著者は、action gap が小さいと小さな近似誤差で次善の行動が最善を追い越すという仮説を立て、Advantage Learning で churn が大きく減り action gap が大きくなることを示した。n を変えて churn や action gap を測った実験は論文に無い（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/survey/nstep_action_gap_survey_2026-11-27.md)、§1.3・§3.5）。
- policy churn の論文では、Advantage Learning は churn が減っても、ε = 0 の評価で成績が高かった（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/survey/nstep_action_gap_survey_2026-11-27.md)、§1.3）。
- 補正なしの n-step は、分散が小さい代わりに固定点が偏る → 詳しくは [n-step の収益](nstep_return.md)（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/survey/nstep_action_gap_survey_2026-11-27.md)、§2.1）。
- policy churn の論文は、補正なしの多段の更新がよく効くことを churn で説明した → 詳しくは [n-step の収益](nstep_return.md)（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/survey/nstep_action_gap_survey_2026-11-27.md)、§2.2）。
- 軌跡を切らない Peng's Q(λ) は、行動の方策が Greedy の方策をゆっくり追うなら最適な方策に収束する → 詳しくは [n-step の収益](nstep_return.md)（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/survey/nstep_action_gap_survey_2026-11-27.md)、§2.2）。
- n-step の収益の分散は n とともに増え、λ 収益などの複合の収益は n-step より分散が大きくならない → 詳しくは [n-step の収益](nstep_return.md)（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/survey/nstep_action_gap_survey_2026-11-27.md)、§2.3）。
- Rainbow は多段の目標の利点を報酬を速く伝えることとし、外すと序盤も最終の成績も下がった → 詳しくは [n-step の収益](nstep_return.md)（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/survey/nstep_action_gap_survey_2026-11-27.md)、§3.1）。
- Mountain Car の DQN では、大きい n が序盤と全体の成績で最もよかった → 詳しくは [n-step の収益](nstep_return.md)（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/survey/nstep_action_gap_survey_2026-11-27.md)、§3.2・§5）。
- 疎な報酬の迷路では、n-step の DQN のほうが単段の DQN より経路が短かった → 詳しくは [n-step の収益](nstep_return.md)（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/survey/nstep_action_gap_survey_2026-11-27.md)、§3.3）。
- Episodic Backward Update の論文は、一様に引く単段の更新は疎で遅れた報酬では非効率だとし、その Atari の比較は n-step を含まない → 詳しくは [n-step の収益](nstep_return.md)（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/survey/nstep_action_gap_survey_2026-11-27.md)、§3.3）。
- Munchausen の論文は、極限で action gap が広がることを示し、M-IQN は多段の収益の IQN と組み合わせた → 詳しくは [Munchausen RL](munchausen.md)（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/survey/nstep_action_gap_survey_2026-11-27.md)、§4.2）。
- Munchausen の論文が測った action gap は、M-DQN と Advantage Learning が Adam の DQN より大きかった → 詳しくは [Munchausen RL](munchausen.md)（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/survey/nstep_action_gap_survey_2026-11-27.md)、§4.2）。
- Munchausen の論文は、使った収益の段数について脚注と本文が食い違う → 詳しくは [Munchausen RL](munchausen.md)（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/survey/nstep_action_gap_survey_2026-11-27.md)、§4.2・§5）。
- 割引率を入力に取って任意の時間尺度の価値を 1 つの網で予測する Γ-nets（Sherstan ら、AAAI 2020）は、補助タスクとして役立つかを調べることを次の課題に挙げ、補助タスクの実験は無い（出典: [skip_auxiliary_task_noise_survey_2026-11-27.md](../../../../reports/survey/skip_auxiliary_task_noise_survey_2026-11-27.md)、§1.6）。
- Munchausen の原典は、softmax の方策による探索は Q の尺度に依存するとして、主な結果では ε-greedy で行動した → 詳しくは [Munchausen RL](munchausen.md)（出典: [rnd_predictor_data_munchausen_survey_2026-11-27.md](../../../../reports/survey/rnd_predictor_data_munchausen_survey_2026-11-27.md)、§5.2）。
- 文献の比較条件: Yu らは、SAC のエントロピーの報酬は終端でない step にだけ足され、終端のあとの Q は TD の backup で 0 になるので、終端の前の報酬を期待値で膨らませ、エントロピーが正なら生き続ける向きに働くと書き（負なら逆）、これを reward inflation と呼ぶ。5 ノードの SimpleChain（終端でない 0〜3 番の報酬 −0.05）で α 0.2 を固定すると、1 step のエントロピーの報酬は約 0.14、差し引き約 0.09 で agent は終端を避け、SAC はほとんど進まなかった。BipedalWalker では前へ進まずに 1600 step の上限まで続く失敗が約 1.2M step まで続いた（出典: [rnd_predictor_data_munchausen_survey_2026-11-27.md](../../../../reports/survey/rnd_predictor_data_munchausen_survey_2026-11-27.md)、§5.3）。
- 文献の比較条件: Yu らの対処は、方策の評価からエントロピーの報酬を外す SACLite と、移動平均を引いて平均 0 にする SACZero で、大規模な比較では SACLite が SAC と同等か上回ることが多かった。扱うのは連続制御の SAC で、Munchausen と内的報酬の組には触れず、arXiv の版に会議の記載は無い（出典: [rnd_predictor_data_munchausen_survey_2026-11-27.md](../../../../reports/survey/rnd_predictor_data_munchausen_survey_2026-11-27.md)、§5.3）。

## 横断的知見

なし

## 見立て

- DropMerge の NEET（NOOP と最善の DROP の差が 0 付近に潰れる）と、Montezuma の Greedy が梯子の下や鍵の下で止まること（選んだ行動と進む行動の差が 0.002〜0.003）は、どちらも「今進む」と「1 step 遅れて進む」の価値の差が (1−γ)·V しかなく、推定の揺れで順位が入れ替わる、同じ形ではないか。 状態: 未検証
  - 支える事実: (1) DropMerge の NEET 期は、差が −0.020〜−0.026 から −0.001〜−0.002 に潰れ、NOOP の選択率は 74〜84% だった（出典: [RFD0163](../../../memo/0163_dropmerge_neet_diagnostics_10rfd.md)、§2）。 (2) DropMerge では、1 step の NOOP の価値と最善の DROP との本来の差は (1−γ)·V ≈ 0.3% しかなく、推定の揺れで順位が入れ替わる（出典: [RFD0167](../../../memo/0167_smdp_transition_duration_10rfd.md)、§1）。 (3) Montezuma の seed 2 の Run は、梯子の下で DOWN を押し続け、Q は 18 行動とも 1.654〜1.663（DOWN 1.663、UP 1.659）で、1 step 遅れたときの差は約 0.005 だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、探索ブロック 10）。 (4) Montezuma の部屋 2 の梯子の下は、選ばれた行動と進むのに要る行動の差が 0.003 だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 14・探索ブロック 11）。 (5) Montezuma の延長の 50M の重みは、部屋 6 で選ばれた行動と右へ進む行動の差が 0.002、部屋 14 で 18 行動の幅が 0.003 だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 20・探索ブロック 16）。
  - 合わない事実: (1) Montezuma の部屋 14 の鍵の真下は、鍵を取る UP が選ばれた行動より 0.027 低く、1 step 遅れたときの差（約 0.005）より大きい（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 14）。 (2) DropMerge で ε を 0 にすると、NoisyNet の有無によらず NEET は出なかった（NOOP の差 −0.18）。差が割引の分しかない構造があっても、ε ラダーが無ければ NEET に入らなかった（出典: [2026-10-01_atari-feedback.md](../../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、探索ブロック 08）。
  - 条件の違い: DropMerge の NOOP は物理を 1 frame 進める待ちで、Montezuma の止まりは進まない行動の繰り返しである。DropMerge の値は評価の UQE のスコアの差、Montezuma は EvalPanel（Greedy、sticky 0）の Q である。γ はどちらも 0.997 である。
  - 確かめ方: Montezuma の Greedy の止まった局面を EvalPanel か step trace で集めて、選んだ行動と進む行動の差を (1−γ)·V と並べる。外的の γ を下げた Run で、Greedy の打ち切りの割合を比べる（新しい Run が要る）。
  - 出どころ: AI
- Montezuma で命が残っている間に危ない動きで命を失い続けるのは、最後でない命を失う損が、ゲームオーバーか打ち切りの先にしか出ず、価値にほとんど現れないためではないか。 状態: 未検証
  - 支える事実: (1) Montezuma の r123 の exp step 29,892,992 の重みを EvalPanel（Greedy、sticky 0）で見ると、部屋 1 で鍵を取って右の扉から出る間に落下で命を 1 つ失い、部屋 2 で敵を避けて動き回る間に命を 2 つ失った（出典: [2026-11-08_montezuma-rnd-taiga-aligned.md](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、探索ブロック11）。 (2) Montezuma の g99ne_split の学習側のゲームは、どの 2M 窓でも平均で 4.94〜5.00 個の命を失って終わった（出典: [2026-11-08_montezuma-rnd-taiga-aligned.md](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 27）。
  - 合わない事実: (1) Montezuma で Munchausen を有効にした rnd_b001 の学習側のゲームは、16〜28M に 2M 窓ごとに終わったゲームの 74〜89% が打ち切りで終わった（出典: [2026-11-08_montezuma-rnd-taiga-aligned.md](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 8）。 (2) Montezuma の b001_s2 の Greedy は、10.2〜29.2M の 320 ゲームのうち打ち切りでないのが 4 本で、終わったときの lives の平均は 20.2〜29.2M に 3.50〜4.88 だった（出典: [2026-11-08_montezuma-rnd-taiga-aligned.md](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 10）。
  - 条件の違い: 支える事実の (1) は EvalPanel の 1 ゲームの観察で、(2) は学習側のゲームの窓の平均である。支える事実の Run（r123・g99ne_split）と合わない事実の Run（rnd_b001・b001_s2）は、Munchausen と RND の設定が違う。
  - 確かめ方: Montezuma の Greedy の評価に step trace（`lives` の列）を足して、最後でない命を失った step の前後の V の差を数える（評価だけの新しい Run が要る）。学習で命の喪失を終端にした Run で、命を失う回数と先への進みが変わるかを比べる（新しい Run が要る）。
  - 出どころ: 開発者の発言（2026-11-12）

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

# reports 索引(サーベイ)

`anet-survey-queue` が読む索引。済みは実施したサーベイ、キューは次に回す候補。キューへの追加と `ready` 化は人が行う。状態が `候補` の行は実行されない。

## 済み

| 日付 | ファイル | テーマ | 関連 |
|---|---|---|---|
| 2026-08-13 | [action_query_papers_2026-08-13.md](action_query_papers_2026-08-13.md) | Action Query 関連論文 | |
| 2026-08-13 | [aqt_performance_and_visibility_2026-08-13.md](aqt_performance_and_visibility_2026-08-13.md) | AQT の性能と知名度の監査 | |
| 2026-08-13 | [atari_env_survey_2026-08-13.md](atari_env_survey_2026-08-13.md) | Atari RL 環境(ALE、ゲーム、ラッパー) | PRD 051 |
| 2026-08-14 | [real_robot_offline_rl_candidates_2026-08-14.md](real_robot_offline_rl_candidates_2026-08-14.md) | 実ロボットのサンプル効率・オフライン学習候補 | docs/StampFly.md |
| 2026-08-26 | [btr_hyperparams_survey_2026-08-26.md](btr_hyperparams_survey_2026-08-26.md) | BTR のハイパーパラメータと Atari 評価プロトコル | Atari campaign |
| 2026-08-31 | [btr_code_reading_2026-08-31.md](btr_code_reading_2026-08-31.md) | BTR 公開実装のコードリーディング | Atari campaign |
| 2026-09-03 | [target_network_update_rate_survey_2026-09-03.md](target_network_update_rate_survey_2026-09-03.md) | target network 更新速度(hard C / soft τ) | Atari campaign |
| 2026-09-09 | [atari_walltime_frontier_survey_2026-09-09.md](atari_walltime_frontier_survey_2026-09-09.md) | Atari の実時間効率フロンティア | Atari campaign |
| 2026-09-23 | [reward_scale_discount_survey_2026-09-23.md](reward_scale_discount_survey_2026-09-23.md) | 報酬のスケール処理と割引率(クリップの代替となる価値の再スケーリング h(x)・Pop-Art、γ の設定と annealing) | Atari campaign(2026-09-23 kung_fu_master) |
| 2026-09-24 | [simple_model_based_atari_survey_2026-09-24.md](simple_model_based_atari_survey_2026-09-24.md) | SimPLe(Model-Based RL for Atari、ICLR 2020)の受容・批判・後続研究(Atari 100k の起点、world model 系の比較表での扱い、企業の world model での参照) | |
| 2026-10-03 | [biedenkapp2021_temporl_paper_2026-10-03.md](biedenkapp2021_temporl_paper_2026-10-03.md) | TempoRL: 行動と続ける長さを別々の Q で学ぶ(ICML 2021) | docs/memo/086_dropmerge_macro_actions_10prd.md、docs/memo/098_temporl_skip_policy_10prd.md |
| 2026-10-03 | [sharma2017_figar_paper_2026-10-03.md](sharma2017_figar_paper_2026-10-03.md) | FiGAR: 行動と反復回数を別々の方策で選ぶ細粒度の行動反復(ICLR 2017) | docs/memo/086_dropmerge_macro_actions_10prd.md |
| 2026-10-05 | [dong2026_tql_paper_2026-10-05.md](dong2026_tql_paper_2026-10-05.md) | TQL: Transformer 価値関数の attention 崩壊と、層ごとのエントロピー誘導(ICML 2026) | reports/action_query_papers_2026-08-13.md、apps/runner/config/DropMerge.txt(net.branch.@SuikaImpala2_ViT128) |
| 2026-10-18 | [llm_prior_knowledge_rl_survey_2026-10-18.md](llm_prior_knowledge_rl_survey_2026-10-18.md) | LLM の事前知識で RL の探索と報酬を補助する手法(説明書やゲーム知識からの補助報酬、LLM の好みを使った内的報酬、LLM による目標の提案) | Atari campaign(2026-09-23 kung_fu_master のボス、2026-09-24 qbert の 8 面) |
| 2026-10-23 | [tallec2019_time_discretization_paper_2026-10-03.md](tallec2019_time_discretization_paper_2026-10-03.md) | Tallec et al. 2019: 時間離散化への頑健性と Deep Advantage Updating（arXiv v2 の本文・付録を原典照合） | docs/memo/999_smdp_transition_duration_10prd.md、docs/memo/086_dropmerge_macro_actions_10prd.md |
| 2026-11-04 | [few_seed_comparison_reporting_survey_2026-11-04.md](few_seed_comparison_reporting_survey_2026-11-04.md) | 少数 seed での比較と報告(seed ブレ、max の偏り、公表値との比べ方、IQM・信頼区間) | Atari campaign、PRD 103 |
| 2026-11-04 | [bellemare2016_increasing_action_gap_paper_2026-10-03.md](bellemare2016_increasing_action_gap_paper_2026-10-03.md) | Bellemare et al. 2016: 一貫ベルマン作用素と advantage learning | PRD 956 |
| 2026-11-04 | [white2017_unifying_task_specification_paper_2026-11-04.md](white2017_unifying_task_specification_paper_2026-11-04.md) | White 2017: 遷移ごとの割引で episodic と continuing を統一 | PRD 956 |
| 2026-11-04 | [springenberg2024_offline_actor_critic_large_models_paper_2026-11-04.md](springenberg2024_offline_actor_critic_large_models_paper_2026-11-04.md) | Springenberg et al. 2024: Perceiver-Actor-Critic | action_query_papers_2026-08-13.md、DropMerge の net.branch.@SuikaImpala2_ViT128 |
| 2026-11-04 | [lee2024_uncertainty_aware_temporally_extended_actions_paper_2026-11-04.md](lee2024_uncertainty_aware_temporally_extended_actions_paper_2026-11-04.md) | Lee et al. 2024(UTE): skip 価値のアンサンブルと不確かさでの j 選択 | PRD 958 |
| 2026-11-04 | [jacq2022_lazy_mdps_learning_when_to_act_paper_2026-11-04.md](jacq2022_lazy_mdps_learning_when_to_act_paper_2026-11-04.md) | Jacq et al. 2022(Lazy-MDP): 既定の方策に任せる行動と罰 η | PRD 958 |
| 不明 | [tau_sampling_modes_survey.md](tau_sampling_modes_survey.md) | 分布強化学習の τ サンプリング/配置(統合版) | PRD 044 |
| 不明 | [_survey_tau_core_papers.md](_survey_tau_core_papers.md) | τ の選び方・配置方式(主要論文) | PRD 044 |
| 不明 | [_survey_tau_implementations.md](_survey_tau_implementations.md) | IQN / FQF / QR-DQN の τ 生成実装 | PRD 044 |
| 不明 | [_survey_tau_mc_theory.md](_survey_tau_mc_theory.md) | モンテカルロ積分の分散低減と τ | PRD 044 |
| 不明 | [_survey_tau_risk_adaptive.md](_survey_tau_risk_adaptive.md) | τ の非一様分布・distortion・risk | PRD 044 |

## キュー

| 優先 | 状態 | 種別 | テーマ | 関連 | 論点メモ |
|---|---|---|---|---|---|
| 7 | 候補 | survey | NoisyNet の現代的な扱い(Rainbow 以降で残っているか、ε-greedy との使い分け、実装の落とし穴) | docs/memo/done/084_noisynet_10prd.md | 採用判断に必要な根拠 |
| 8 | 候補 | survey | Q 値統計によるハイパーパラメータ誤設定の検出(q_max の伸び率、過大評価指標) | docs/memo/999_q_statistics_config_screening_10prd.md | スクリーニング指標の先行例 |
| 9 | 候補 | survey | 可塑性喪失の指標と対策の最新動向(dormant units、weight norm、reset) | docs/memo/done/062, 063 | 062 の指標群に欠けがないか |
| 13 | 候補 | survey | PER の優先度の弱点と代替(TD 誤差が経験の貴重さを表さない場面、確率的な報酬での優先度の張り付き、優先度の古さ、優先度を手前の遷移へ流す方式、損失の形との等価性、収益や gain×need による優先) | docs/memo/999_replay_eviction_sample_count_10prd.md、Atari campaign(2026-10 kung_fu_master のボス撃破が定着しない件) | Schaul et al. 2016「Prioritized Experience Replay」、Brittain et al. 2019「Prioritized Sequence Experience Replay」、Fujimoto et al. 2020「An Equivalence between Loss Functions and Non-uniform Sampling in Experience Replay」(LAP)、Oh et al. 2018「Self-Imitation Learning」、Mattar & Daw 2018「Prioritized memory access explains planning and hippocampal replay」、Lahire et al. 2022「Large Batch Experience Replay」を起点に。分布型(IQN・QR)での優先度の定義の違い(分位点平均の差、分位点 Huber 損失、KL)と、τ のサンプリングの揺らぎが優先度に乗る問題。珍しい大報酬の遷移の優先度が学習中にどう推移するかを測った例。α・β の選び方(BTR 0.2、Rainbow 0.5、原論文 0.6)と、1 遷移が寿命中に約 1 回しか引かれない低い replay ratio での PER の効き。遡り(PSER、EBU)・収益ベース(SIL)・gain×need の Atari での効果量。区分ごとの優先度や抽選回数を計器として出している例 |

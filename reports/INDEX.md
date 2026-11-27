# reports 索引(サーベイ)

`anet-survey-queue` が読む索引。済みは実施したサーベイ、キューは次に回すサーベイ。キューの行はすべて、優先の数が小さいものから回す。キューへの追加は人が行う。

## 済み

| 日付 | ファイル | テーマ | 関連 |
|---|---|---|---|
| 2026-11-27 | [rnd_dense_reward_reset_collapse_survey_2026-11-27.md](rnd_dense_reward_reset_collapse_survey_2026-11-27.md) | RNDの効き方と崩れ。外的報酬が密なゲームでのゲーム別の結果、predictorを初期化し直す事例、内的報酬の崩れと可塑性を結ぶ研究 | RFD0202（過去の質問の外の調べもの） |
| 2026-11-27 | [rnd_predictor_data_munchausen_survey_2026-11-27.md](rnd_predictor_data_munchausen_survey_2026-11-27.md) | RNDの学び方と組み合わせ。NGU系の倍率の統計の取り方、predictorが学ぶデータとPERの補正、Munchausenと内的報酬の組 | RFD0202（過去の質問の外の調べもの） |
| 2026-11-27 | [value_decomposition_head_nstep_survey_2026-11-27.md](value_decomposition_head_nstep_survey_2026-11-27.md) | 外的と内的の価値の分け方。Headだけ分ける形とNetworkごと分ける形、内的のn-stepの和とエピソードの境の扱い | RFD0202（過去の質問の外の調べもの） |
| 2026-11-27 | [montezuma_published_scores_survey_2026-11-27.md](montezuma_published_scores_survey_2026-11-27.md) | Montezuma's Revengeの公開の成績と測り方。TaïgaらとRND原典の評価の条件、内的報酬を足さない学習器の点 | RFD0202（過去の質問の外の調べもの） |
| 2026-11-27 | [qbert_published_scores_levels_survey_2026-11-27.md](qbert_published_scores_levels_survey_2026-11-27.md) | Q*bertの公表の点は何面まで進んだ点か。得点の仕組み、公表値の統計量、同じ面で点を重ねる振る舞いと不具合の報告 | RFD0202（過去の質問の外の調べもの） |
| 2026-11-27 | [weight_init_normalization_adam_survey_2026-11-27.md](weight_init_normalization_adam_survey_2026-11-27.md) | 正規化付きの層とAdamでの重みの初期値の大きさの効き方。実効学習率、weight decay、公開実装の初期値 | RFD0202（過去の質問の外の調べもの） |
| 2026-11-27 | [nstep_action_gap_survey_2026-11-27.md](nstep_action_gap_survey_2026-11-27.md) | n-stepとGreedyの行動間の差（action gap）。報酬の伝わる速さ、n-stepの偏りと分散、policy churn | RFD0202（過去の質問の外の調べもの） |
| 2026-11-27 | [skip_auxiliary_task_noise_survey_2026-11-27.md](skip_auxiliary_task_noise_survey_2026-11-27.md) | 行動を続ける長さの学習を補助タスクやノイズと組む事例。多地平の補助タスク、εz-greedy、SNAP、TAAC | RFD0202（過去の質問の外の調べもの） |
| 2026-11-27 | [novelty_bonus_long_horizon_reach_survey_2026-11-27.md](novelty_bonus_long_horizon_reach_survey_2026-11-27.md) | 新しさのボーナスで、長い移動の先やふだん通らない状態へ届くか。好奇心だけの到達、密な報酬での局所解、tandem、human starts | RFD0202（過去の質問の外の調べもの） |
| 2026-11-24 | [sil_value_based_lower_bound_survey_2026-11-24.md](sil_value_based_lower_bound_survey_2026-11-24.md) | 自己模倣（SIL）を価値ベースへ広げた手法（Optimality Tightening・EMDQN・下限Q学習・SAIL・GEM・DB-ADP）の損失・記憶の持ち方・難探索の数値・分布型での扱いと、確率的な環境（sticky actions）での偏りと対策 | RFD0201 |
| 2026-11-20 | [torchrl_survey_2026-11-20.md](torchrl_survey_2026-11-20.md) | TorchRLの設計と現状。TransformsとPRD161の観測パイプラインの比較（CatFramesのReplay側の重ね直し、spec変換、正規化の凍結、LastAction）、DQN系・RNDの対応と公開結果（比較元にはならず、Dopamine・CleanRL・RLeXploreの条件を併記）、実時間の性能、C++から使えるか、採用状況と批判 | docs/memo/161_observation_pipeline_10prd.md、docs/design/150_replay_buffer.jp.md、docs/design/212_rnd_lifelong_novelty.jp.md |
| 2026-11-18 | [per_priority_weaknesses_survey_2026-11-18.md](per_priority_weaknesses_survey_2026-11-18.md) | PERの優先度の弱点と代替。確率的な報酬での張り付き、優先度の古さ、損失の形との等価性（LAP）、α・β、分布型での優先度、遡る系・収益ベース・gain×need の効果量 | docs/memo/171_replay_eviction_sample_count_10prd.md、docs/management/issues/issue-20261103-8c3c6d.md |
| 2026-11-18 | [episodic_bonus_value_based_survey_2026-11-18.md](episodic_bonus_value_based_survey_2026-11-18.md) | エピソード内の新しさのボーナスを価値ベース（DQN・Replay）に載せるときの論点。NGU系の実装、off-policyでの扱い、Replayに入れた内的報酬の古さと外的・内的の分け方 | docs/memo/173_intrinsic_reward_novelty_10prd.md、docs/management/issues/issue-20261103-736e39.md |
| 2026-11-18 | [intrinsic_reward_landscape_survey_2026-11-18.md](intrinsic_reward_landscape_survey_2026-11-18.md) | 内的報酬の手法の全体像。予測誤差系・回数と密度の系・エピソード内の新しさと方策の組・戻る系、RNDとICMの使い分け、detachment・derailmentへの対処、ベンチマークごとの基準 | docs/memo/173_intrinsic_reward_novelty_10prd.md、docs/wiki/concepts/methods/intrinsic_novelty.md、docs/management/issues/issue-20261115-d3e094.md |
| 2026-11-07 | [llm_rl_reading_guide_2026-11-07.md](llm_rl_reading_guide_2026-11-07.md) | LLM/RL再サーベイ47論文の索引、関心別の読み順、研究上の問いと費用の比較方法 | 11冊の案内 |
| 2026-11-07 | [llm_rl_distillation_and_planning_2026-11-07.md](llm_rl_distillation_and_planning_2026-11-07.md) | LVLM2PとPriorZero。小さな方策への蒸留、公開PPOの分布、世界モデルとの相互学習と費用 | LLM/RL再サーベイ |
| 2026-11-07 | [llm_rl_action_advice_2026-11-07.md](llm_rl_action_advice_2026-11-07.md) | 行動事前分布・観測ヒント・APMの3論文。候補数、cache、KL付き目的を区別 | LLM/RL再サーベイ |
| 2026-11-07 | [llm_rl_goals_and_reflection_2026-11-07.md](llm_rl_goals_and_reflection_2026-11-07.md) | ELLM・SGRL・LLM-TALE・LAGEA。目標生成、行動の制約、失敗の振り返りと単一GPUの条件 | LLM/RL再サーベイ |
| 2026-11-07 | [llm_rl_language_grounding_2026-11-07.md](llm_rl_language_grounding_2026-11-07.md) | ONI・RLang翻訳・DECKARDの3論文。非同期注釈、語彙とラベル解析、経験による依存関係の修正 | LLM/RL再サーベイ |
| 2026-11-07 | [llm_rl_algorithm_discovery_2026-11-07.md](llm_rl_algorithm_discovery_2026-11-07.md) | LLMによるRL更新則発見の3論文と固定実装。独立探索・環境seed・評価先調整・候補学習費用を区別 | LLM/RL再サーベイ |
| 2026-08-13 | [action_query_papers_2026-08-13.md](action_query_papers_2026-08-13.md) | Action Query 関連論文 | |
| 2026-08-13 | [aqt_performance_and_visibility_2026-08-13.md](aqt_performance_and_visibility_2026-08-13.md) | AQT の性能と知名度の監査 | |
| 2026-08-13 | [atari_env_survey_2026-08-13.md](atari_env_survey_2026-08-13.md) | Atari RL 環境(ALE、ゲーム、ラッパー) | PRD 051 |
| 2026-08-14 | [real_robot_offline_rl_candidates_2026-08-14.md](real_robot_offline_rl_candidates_2026-08-14.md) | 実ロボットのサンプル効率・オフライン学習候補 | docs/StampFly.md |
| 2026-08-26 | [btr_hyperparams_survey_2026-08-26.md](btr_hyperparams_survey_2026-08-26.md) | BTR のハイパーパラメータと Atari 評価プロトコル | Atari campaign |
| 2026-08-31 | [btr_code_reading_2026-08-31.md](btr_code_reading_2026-08-31.md) | BTR 公開実装のコードリーディング | Atari campaign |
| 2026-09-03 | [target_network_update_rate_survey_2026-09-03.md](target_network_update_rate_survey_2026-09-03.md) | target network 更新速度(hard C / soft τ) | Atari campaign |
| 2026-09-09 | [atari_walltime_frontier_survey_2026-09-09.md](atari_walltime_frontier_survey_2026-09-09.md) | Atari の実時間効率フロンティア | Atari campaign |
| 2026-09-17 | [replay_lane_window_survey_2026-09-17.md](replay_lane_window_survey_2026-09-17.md) | replay の 1 アクターあたりの履歴窓(容量 ÷ env 数)とエピソード長の比。容量と最古方策年齢の分解、n-step・replay ratio との相互作用、分散アクター構成の容量設計、エピソード単位 replay、評価 ε で順位が変わる現象 | Atari campaign(2026-09-15 replay-lane-window)、docs/wiki/concepts/methods/replay_ratio_lane_window.md |
| 2026-09-23 | [reward_scale_discount_survey_2026-09-23.md](reward_scale_discount_survey_2026-09-23.md) | 報酬のスケール処理と割引率(クリップの代替となる価値の再スケーリング h(x)・Pop-Art、γ の設定と annealing) | Atari campaign(2026-09-23 kung_fu_master) |
| 2026-09-24 | [simple_model_based_atari_survey_2026-09-24.md](simple_model_based_atari_survey_2026-09-24.md) | SimPLe(Model-Based RL for Atari、ICLR 2020)の受容・批判・後続研究(Atari 100k の起点、world model 系の比較表での扱い、企業の world model での参照) | |
| 2026-10-03 | [biedenkapp2021_temporl_paper_2026-10-03.md](biedenkapp2021_temporl_paper_2026-10-03.md) | TempoRL: 行動と続ける長さを別々の Q で学ぶ(ICML 2021) | docs/memo/086_dropmerge_macro_actions_10prd.md、docs/memo/098_temporl_skip_policy_10prd.md |
| 2026-10-03 | [sharma2017_figar_paper_2026-10-03.md](sharma2017_figar_paper_2026-10-03.md) | FiGAR: 行動と反復回数を別々の方策で選ぶ細粒度の行動反復(ICLR 2017) | docs/memo/086_dropmerge_macro_actions_10prd.md |
| 2026-10-05 | [dong2026_tql_paper_2026-10-05.md](dong2026_tql_paper_2026-10-05.md) | TQL: Transformer 価値関数の attention 崩壊と、層ごとのエントロピー誘導(ICML 2026) | reports/action_query_papers_2026-08-13.md、apps/runner/config/DropMerge.txt(net.branch.@SuikaImpala2_ViT128) |
| 2026-10-15 | [gradient_noise_observation_and_control_survey_2026-10-15.md](gradient_noise_observation_and_control_survey_2026-10-15.md) | 勾配ノイズの観測と対処(GNS などの観測方法、batch size と学習率を変える事例と反証、momentum・Adam・importance sampling・SVRG・clipping の限界、RFD0092 への考察) | RFD0092、docs/wiki/concepts/phenomena/gradient_noise_batch.md |
| 2026-10-18 | [llm_prior_knowledge_rl_survey_2026-10-18.md](llm_prior_knowledge_rl_survey_2026-10-18.md) | LLM の事前知識で RL の探索と報酬を補助する手法(説明書やゲーム知識からの補助報酬、LLM の好みを使った内的報酬、LLM による目標の提案) | Atari campaign(2026-09-23 kung_fu_master のボス、2026-09-24 qbert の 8 面) |
| 2026-10-23 | [tallec2019_time_discretization_paper_2026-10-03.md](tallec2019_time_discretization_paper_2026-10-03.md) | Tallec et al. 2019: 時間離散化への頑健性と Deep Advantage Updating（arXiv v2 の本文・付録を原典照合） | docs/memo/167_smdp_transition_duration_10prd.md、docs/memo/086_dropmerge_macro_actions_10prd.md |
| 2026-11-04 | [few_seed_comparison_reporting_survey_2026-11-04.md](few_seed_comparison_reporting_survey_2026-11-04.md) | 少数 seed での比較と報告(seed ブレ、max の偏り、公表値との比べ方、IQM・信頼区間) | Atari campaign、PRD 103 |
| 2026-11-04 | [bellemare2016_increasing_action_gap_paper_2026-10-03.md](bellemare2016_increasing_action_gap_paper_2026-10-03.md) | Bellemare et al. 2016: 一貫ベルマン作用素と advantage learning | PRD 167 |
| 2026-11-04 | [white2017_unifying_task_specification_paper_2026-11-04.md](white2017_unifying_task_specification_paper_2026-11-04.md) | White 2017: 遷移ごとの割引で episodic と continuing を統一 | PRD 167 |
| 2026-11-04 | [springenberg2024_offline_actor_critic_large_models_paper_2026-11-04.md](springenberg2024_offline_actor_critic_large_models_paper_2026-11-04.md) | Springenberg et al. 2024: Perceiver-Actor-Critic | action_query_papers_2026-08-13.md、DropMerge の net.branch.@SuikaImpala2_ViT128 |
| 2026-11-04 | [lee2024_uncertainty_aware_temporally_extended_actions_paper_2026-11-04.md](lee2024_uncertainty_aware_temporally_extended_actions_paper_2026-11-04.md) | Lee et al. 2024(UTE): skip 価値のアンサンブルと不確かさでの j 選択 | PRD 183 |
| 2026-11-04 | [jacq2022_lazy_mdps_learning_when_to_act_paper_2026-11-04.md](jacq2022_lazy_mdps_learning_when_to_act_paper_2026-11-04.md) | Jacq et al. 2022(Lazy-MDP): 既定の方策に任せる行動と罰 η | PRD 183 |
| 2026-11-07 | [llm_assisted_rl_resurvey_2026-11-07.md](llm_assisted_rl_resurvey_2026-11-07.md) | LLM・VLMで外部RLを補助する21論文。報酬・観測・探索・課題・更新則、比較条件と失敗例 | 2026-10-18のLLM事前知識サーベイ |
| 2026-11-07 | [llm_assisted_rl_reproducibility_2026-11-07.md](llm_assisted_rl_reproducibility_2026-11-07.md) | LLM-Explorer・LIMEN・BIRD・Robo-Dopamineの固定commit調査とRoboRewardモデルカード | LLMとRLの再サーベイ |
| 2026-11-07 | [llm_reward_shaping_guarantees_2026-11-07.md](llm_reward_shaping_guarantees_2026-11-07.md) | LLM報酬整形の保証と目標切り替え。2026年preprintの式に対する反例と有理数での検算 | LLMとRLの再サーベイ |
| 2026-11-07 | [llm_generated_control_and_environments_2026-11-07.md](llm_generated_control_and_environments_2026-11-07.md) | 追加5論文。制御コード・世界モデル・探索prompt・学習環境、生成runと環境seedの違い | LLMとRLの再サーベイ |
| 2026-11-07 | [llm_rl_small_models_and_costs_2026-11-07.md](llm_rl_small_models_and_costs_2026-11-07.md) | 追加4論文。小型VLM・軽い探索方策・経験なし報酬指定・0.5B提案器、総計算費用の境界 | LLMとRLの再サーベイ |
| 2026-11-07 | [llm_rl_exploration_information_2026-11-07.md](llm_rl_exploration_information_2026-11-07.md) | 追加2論文。iLLMの4-bit・バッチ実行とAtari比較、IGEの復帰付き探索、前回記述の訂正 | LLMとRLの再サーベイ |
| 不明 | [tau_sampling_modes_survey.md](tau_sampling_modes_survey.md) | 分布強化学習の τ サンプリング/配置(統合版) | PRD 044 |
| 不明 | [_survey_tau_core_papers.md](_survey_tau_core_papers.md) | τ の選び方・配置方式(主要論文) | PRD 044 |
| 不明 | [_survey_tau_implementations.md](_survey_tau_implementations.md) | IQN / FQF / QR-DQN の τ 生成実装 | PRD 044 |
| 不明 | [_survey_tau_mc_theory.md](_survey_tau_mc_theory.md) | モンテカルロ積分の分散低減と τ | PRD 044 |
| 不明 | [_survey_tau_risk_adaptive.md](_survey_tau_risk_adaptive.md) | τ の非一様分布・distortion・risk | PRD 044 |

## キュー

| 優先 | 種別 | テーマ | 関連 | 論点メモ |
|---|---|---|---|---|

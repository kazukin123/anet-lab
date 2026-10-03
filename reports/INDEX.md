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
| 不明 | [tau_sampling_modes_survey.md](tau_sampling_modes_survey.md) | 分布強化学習の τ サンプリング/配置(統合版) | PRD 044 |
| 不明 | [_survey_tau_core_papers.md](_survey_tau_core_papers.md) | τ の選び方・配置方式(主要論文) | PRD 044 |
| 不明 | [_survey_tau_implementations.md](_survey_tau_implementations.md) | IQN / FQF / QR-DQN の τ 生成実装 | PRD 044 |
| 不明 | [_survey_tau_mc_theory.md](_survey_tau_mc_theory.md) | モンテカルロ積分の分散低減と τ | PRD 044 |
| 不明 | [_survey_tau_risk_adaptive.md](_survey_tau_risk_adaptive.md) | τ の非一様分布・distortion・risk | PRD 044 |

## キュー

| 優先 | 状態 | 種別 | テーマ | 関連 | 論点メモ |
|---|---|---|---|---|---|
| 2 | ready | survey | 少数 seed での比較と報告(ゲーム単位の seed ブレ、必要な seed 数、IQM・信頼区間) | Atari campaign(2026-09-21 Atari-5、2026-09-23 kung_fu_master) | Agarwal et al. 2021「Deep RL at the Edge of the Statistical Precipice」、Henderson et al. 2018「Deep RL that Matters」、Patterson et al. 2024「Empirical Design in RL」を軸に。1 ゲームあたりの seed ブレはどの程度と報告されているか。学習中の最良値(max)を報告する方式の偏りはどれくらいか。seed 数の違う公表値と 2 seed の結果をどう比べるのが妥当か |
| 3 | 候補 | survey | NoisyNet の現代的な扱い(Rainbow 以降で残っているか、ε-greedy との使い分け、実装の落とし穴) | docs/memo/999_noisynet_10prd.md | 採用判断に必要な根拠 |
| 4 | 候補 | survey | Q 値統計によるハイパーパラメータ誤設定の検出(q_max の伸び率、過大評価指標) | docs/memo/999_q_statistics_config_screening_10prd.md | スクリーニング指標の先行例 |
| 5 | 候補 | survey | 可塑性喪失の指標と対策の最新動向(dormant units、weight norm、reset) | docs/memo/done/062, 063 | 062 の指標群に欠けがないか |
| 6 | ready | survey | LLM の事前知識で RL の探索と報酬を補助する手法(説明書やゲーム知識からの補助報酬、LLM の好みを使った内的報酬、LLM による目標の提案) | Atari campaign(2026-09-23 kung_fu_master のボス、2026-09-24 qbert の 8 面) | Wu et al. 2023「Read and Reap the Rewards」、Klissarov et al. 2024「Motif」、Du et al. 2023「ELLM」を起点に。ALE の Atari で試した例と効果の大きさ。LLM を呼ぶ頻度とコスト(step ごとには呼ばない設計、答えのキャッシュ)。出来事の検出(RAM・物体検出)をどう作っているか。外部知識を入れた結果をベンチマークとどう分けて報告しているか。VLM を報酬に使う系(CLIP など)との違い |

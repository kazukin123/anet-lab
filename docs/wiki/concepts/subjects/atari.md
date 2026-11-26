# Atari横断

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [題材](../../index.md#題材)

## 今の結論

学習中最良のAtari-5回帰で、実時間優先のBTR比は50M・seed1で66.8%、100M・seed2で83.8%だった。サンプル効率優先50Mは458.3、BTR比82.5%に達した。ただしBTR比は分母と分子で評価回数とseed数がそろわず、BTRの公開CSVを評価10回のmaxで取ると92%（90〜96%）になる。

## 分かっていること

### anet の知見

- Atari観測の縮小は、anetのadaptive average poolingとBTRのOpenCV INTER_AREAで、非整数比の画素境界の重み付けが異なる。areaという名前だけでは数値一致を意味せず、OpenCVの丸めも経路で異なる。Qbertの成績への影響は未検証である（出典: [RFD0099](../../../memo/done/0099_atari_resize_mode_10rfd.md)、§2.1・§3）。
- Pong は v5 プリセット・num_envs 64 で 1M step から動き出し、2.5M で eval1 +7.10 だった（出典: [2026-08-17_baseline.md](../../../experiments/default-dqn/atari/2026-08-17_baseline.md)、現時点のまとめ7）。
- Atari-5 回帰の係数は battle_zone 0.3820・name_this_game 0.3108・phoenix 0.1241・qbert 0.0805・double_dunk 0.0679（切片 0）で、battle_zone と name_this_game の重みが大きい（出典: [2026-09-21_realtime-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、メタデータ「集約」）。
- 初回20Mスイープの55.4% HNS推定は、複数ゲームがまだ伸びていた時点の途中経過である。（出典: [2026-08-27_atari5.md](../../../experiments/default-dqn/atari/2026-08-27_atari5.md)、現時点のまとめ。Run: run_20260827-133241_a5_battle_zone_apex / run_20260827-172101_a5_qbert_apex）。
- 実時間優先50Mでは5ゲーム中4つが終盤も上昇中だった。ゲーム間throughput差の多くはeval負荷に由来した。（出典: [2026-09-21_realtime-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、現時点のまとめ。Run: run_20260921-155634_a5_qbert_rr1_50m / run_20260922-063422_a5_phoenix_rr1_50m）。
- サンプル効率優先は同じ50Mで改善したが、同程度のAtari-5水準へ至る実時間は実時間優先より長かった。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ。Run: run_20260925-174945_a5_phoenix_btr50m / run_20260926-022413_a5_qbert_btr50m）。
- 実時間優先50M・seed1のeval2・40–50M窓ではAtari-5 medianが358.0% HNS、回帰が344.8%だった。学習中最良を使うBTRの集計では回帰370.7対555.2、BTR比66.8%だった。（出典: [2026-09-21_realtime-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、現時点のまとめ1・3、探索ブロック01。Run: `run_20260921-155634_a5_qbert_rr1_50m`, `run_20260921-183247_a5_battle_zone_rr1_50m`, `run_20260921-213901_a5_name_this_game_rr1_50m`, `run_20260922-020046_a5_double_dunk_rr1_50m`, `run_20260922-063422_a5_phoenix_rr1_50m`）。
- 実時間優先100M・seed2では90–100M窓の回帰444.9、median463.3で、5ゲームとも同Run内の最高窓だった。学習中最良の回帰465.5はBTR555.2の83.8%だった。（出典: [2026-09-21_realtime-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、現時点のまとめ12・14、探索ブロック04。Run: `run_20260923-163113_a5_qbert_rr1_100m_seed2`, `run_20260923-203734_a5_battle_zone_rr1_100m_seed2`, `run_20260924-095955_a5_name_this_game_rr1_100m_seed2`, `run_20260924-144556_a5_double_dunk_rr1_100m_seed2`, `run_20260925-005343_a5_phoenix_rr1_100m_seed2`）。
- サンプル効率優先50Mの学習中最良の回帰は458.3、BTR555.2の82.5%。同50Mの実時間優先の2 seedを8.9〜23.6%上回った。ゲーム別BTR比はphoenix94%、name_this_game102%、double_dunk101%、battle_zone68%、qbert60%だった。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ1・3、探索ブロック01。Run: `run_20260925-174945_a5_phoenix_btr50m`, `run_20260926-022413_a5_qbert_btr50m`, `run_20260926-082748_a5_battle_zone_btr50m`, `run_20260926-151320_a5_name_this_game_btr50m`, `run_20260927-101048_a5_double_dunk_btr50m`）。
- BTR 比の分母と分子は統計量がそろっていない。手元のサンプル効率優先 50M は 1 seed・評価約 10 回の学習中最良で、BTR の Table A2 から計算した 555.2 は 3 seed・評価 200 回の max である。BTR の公開 CSV から調査が計算した Atari-5 回帰は max（K=200）526.7、max（K=10）497.8（取り始めをずらすと 476.8〜509.9）、最後の 10 回の平均 479.8 で、K=10 で取ると 458.3 は 92%（90〜96%）になる。これも 1 seed の max と 3 seed 平均と推定した曲線の max の比で、評価の ε も違う。調査は BTR 比を 10% 以上の幅のある帯として読むことを案に挙げる（出典: [少数 seed での比較と報告のサーベイ](../../../../reports/few_seed_comparison_reporting_survey_2026-11-04.md)、4.3・8.2 の 7）。
- 実時間優先 50M で train の ε ラダーを NoisyNet に置き換える（ε 0）と、Breakout・Atari-5・kung_fu_master の 7 ゲームとも eval2 40-50M が基準のどの seed も下回った。Breakout の 2 × 2 では下がったのはラダーを外したぶんで、NoisyNet をラダーに足した battle_zone と double_dunk は成績が戻ったが立ち上がりが約 10M・約 20M 遅れた（出典: [2026-10-11_noisynet-realtime.md](../../../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md)、現時点のまとめ1〜3。Run: `run_20261011-182123_breakout_rr1_50m_nz_eps0` ほか同記録の Run）。
- ViT ハイブリッドは CNN 以外の基本構成として常設し、Breakout 単独の成績で採否を決めない。この方針は Atari 以外にも適用する（出典: [Atari 探索索引](../../../experiments/default-dqn/atari/README.md)、campaign に無い比較の方針）。
- 小差の順位付けには複製が必要で、追加測定は判断に使う比較に絞る → 詳しくは [評価と比較の条件](../measurement/evaluation.md)（出典: [探索索引](../../../experiments/default-dqn/atari/README.md)、追加探索と複製の運用方針）。
- 実時間優先 50M の 2 seed の HNS 比はゲーム別に 0.76〜1.27 だったが、40〜50M の回帰値は 344.8 と 358.9。battle_zone と phoenix の差が逆向きで、集約値に現れにくかった（出典: [実時間優先の Atari-5 記録](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック 04）。
- Atari-5 の評価負荷は発火回数よりもゲーム数と長いエピソードの影響が大きかった → 詳しくは [計測の負荷](../measurement/diagnostic_cost.md)（出典: [実時間優先の Atari-5 記録](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック 01・04）。
- 初回スイープの平均には、人間基準を超えた Name This Game とランダム基準を下回った Double Dunk の両方が含まれる。（出典: [Atari-5 横断スイープ 探索履歴](../../../experiments/default-dqn/atari/2026-08-27_atari5.md)、現時点のまとめ。Run: `run_20260827-152639_a5_name_this_game_apex`・`run_20260827-142947_a5_double_dunk_apex`）。

### 外の事例

- 2026-08の調査では、ALEはC++ APIで状態・RAM・flavorを扱える環境として整理された。前処理とROM・ライセンスの条件は別に確認する。（出典: [atari_env_survey_2026-08-13.md](../../../../reports/atari_env_survey_2026-08-13.md)、8. 総合評価）。
- 同じ調査の時点では、新規に始めるなら v5 相当（sticky actions 0.25・frameskip 4、Machado et al. 2018 の勧告）が標準で、Atari 100k 系の事例と比べるときは sticky actions なしが慣行、評価の集計は median ではなく IQM と信頼区間が現行標準とされた（出典: [atari_env_survey_2026-08-13.md](../../../../reports/atari_env_survey_2026-08-13.md)、5.2・5.3・5.5・8. 総合評価）。
- PQNとBTRなどの時間比較ではsticky actions、episodic life、frame予算とhardwareを揃える必要がある。環境だけのFPSを学習込みthroughputと同一視しない。（出典: [atari_walltime_frontier_survey_2026-09-09.md](../../../../reports/atari_walltime_frontier_survey_2026-09-09.md)、12. 総合評価・13. 調査限界）。
- 過去の収益や行動の系列を使う手法は、sticky actions の有無で値が変わる。The Brute は sticky actions 0.25 で試した 5 ゲームとも大きく下がり（Asterix 6,909 → 308、Seaquest 621 → 81 など）、同じ条件の DQN はほぼ変わらなかった。SAIL（DQN）の AL-DQN に対する相対改善の平均は、sticky actions なしの +231.6% に対し、ありは +101.4% だった。SIL 本体・Optimality Tightening・EMDQN の値は sticky actions なしの評価である（出典: [自己模倣（SIL）を価値ベースへ広げた手法のサーベイ](../../../../reports/sil_value_based_lower_bound_survey_2026-11-24.md)、§6.2・§6.3）。
- A.L. と P.A.L. の比較は Atari の確率的設定を含む。過大評価を減らしたことが成績差の原因かは未確定である。 → 詳しくは[行動の時間尺度](../methods/temporal_abstraction.md)。（出典: [Increasing the Action Gap: New Operators for Reinforcement Learning](../../../../reports/bellemare2016_increasing_action_gap_paper_2026-10-03.md)、論文が述べている限界）。
- 同じゲームの報告でも Atari 100k と v5 の57ゲーム設定で sticky actions の取り扱いが違う。 → 詳しくは[評価と比較の条件](../measurement/evaluation.md)。（出典: [Survey: Atari RL 環境](../../../../reports/atari_env_survey_2026-08-13.md)、7. 批判・懸念・7.2 プロトコル分裂の注意点）。
- 公開された Breakout の評価系列に多数の評価回が含まれていても、seed 間のばらつきとは区別が必要である。 → 詳しくは[評価と比較の条件](../measurement/evaluation.md)。（出典: [Code Reading: BTR 公開実装](../../../../reports/btr_code_reading_2026-08-31.md)、7. 調査の限界）。
- 確認した BTR の Atari 実行条件は sticky actions 0.25 と最大30回の初期 no-op を併用していた。論文本文だけから読み取れない条件を実装側で補った。（出典: [Survey: BTR (Beyond The Rainbow)](../../../../reports/btr_hyperparams_survey_2026-08-26.md)、5. Atari 環境設定と前処理・5.2 noop_max — 論文に記載なし、コードで 30）。
- TextAtari の表題にある予算と、調査した版の評価手順で実際に使った相互作用数は異なる。 → 詳しくは[評価と比較の条件](../measurement/evaluation.md)。（出典: [LLMを活用したRLの再サーベイ](../../../../reports/llm_assisted_rl_resurvey_2026-11-07.md)、5. 探索と行動を補助する・5.3 TextAtari）。
- Read and Reward の Atari での実証と、Motif の NetHack、ELLM の Crafter の実証は環境も使える意味情報も異なる。 → 詳しくは[LLMの外部知識と探索](../methods/llm_prior_knowledge.md)。（出典: [Survey: LLM の事前知識で RL の探索と報酬を補助する手法](../../../../reports/llm_prior_knowledge_rl_survey_2026-10-18.md)、1. 結論と手法の見取り図）。

## 横断的知見

なし

## 見立て

なし

## 分かっていないこと

- BTR の Table A2 の値は seed ごとの max の平均か。Table A2 から計算した 555.2 は公開 CSV の max（526.7）より 5.4% 高く、BattleZone・Phoenix・qbert はこの向きだが、NameThisGame と DoubleDunk は逆向きである（出典: [少数 seed での比較と報告のサーベイ](../../../../reports/few_seed_comparison_reporting_survey_2026-11-04.md)、4.3）？

## 撤回・更新

なし。

## 関連

- [replay_ratio_lane_window](../methods/replay_ratio_lane_window.md)。
- [評価と比較の条件](../measurement/evaluation.md)。
- [campaignにない独自の分析](../../../experiments/default-dqn/atari/README.md#独自の分析)。
- [複製がない比較と少数 seed で言える範囲](../measurement/evaluation.md#横断的知見)。
- [Phoenix の二つのプロファイルでの得点とウェーブ数](phoenix.md#横断的知見)。

# wiki の目録

[規約](README.md)・[作業の記録](log.md)。出典は各ページから正本をたどる。

## 題材

- [実ロボット課題の選定](subjects/robot_tasks.md) — 実機候補の調査では、自動resetと成功検知が可能なアーム・治具・制御リグが挙がった。価格が安くても長時間の自律収集が成立するとは確定していない。 （別名: robot, Offline RL, reset, 実ロボット）
- [Name This Game](subjects/name_this_game.md) — サンプル効率優先50Mは実時間優先50M比+24%、BTR比102%だった。45.7M・ε=0の最良28,465は、実時間優先seed2・100Mの最良24,851を上回った。 （別名: NameThisGame, name_this_game, ntg）
- [Double Dunk](subjects/double_dunk.md) — サンプル効率優先のε=0評価は20.4Mで22.1、45.7Mで23.2（BTR比101%）だった。実時間優先seed2は50.7Mで21.5、91.1Mで23.0に達した。 （別名: DoubleDunk, double_dunk）
- [Phoenix](subjects/phoenix.md) — サンプル効率優先45.7M・ε=0の最良399,895はBTR Table A2比94%だった。同断面の100本中44本が27,000 step上限で打ち切られた。 （別名: Phoenix, phoenix）
- [Battle Zone](subjects/battle_zone.md) — 低εラダーのε=0最良は118,388、旧ラダーは115,533だった。64 env・容量1Mの2 seedは40–50M評価が94,621・86,616で、128 env・2Mの106,389を下回った。 （別名: BattleZone, battle_zone, bz）
- [Qbert](subjects/qbert.md) — 低εラダーの64 env・容量1Mでは45–50Mの第8面突破率が3 seedで45.6・32.6・0.7%だった。容量だけ・env数だけを変えた腕は1.0%・0.1%に留まり、初突破が早い腕でも突破が定着しない例があった。 （別名: Qbert, qbert, Q*bert）
- [LunarLander](subjects/lunarlander.md) — matched 3-seedのQR/IQN比較ではホバリング離脱時期に明確な差がなかった。持続的T180到達はQR1/3対IQN3/3、終盤reward平均は144.3対186.0だった。 （別名: LunarLander, ll, ホバリング）
- [DropMerge](subjects/dropmerge.md) — IQNとQR51の順位はUQE評価とGreedy評価で逆転した。NoisyNetへの探索置換はNEETを抑えたが、batch増加の学習側の上乗せの大部分はノイズ感度の差だった。 （別名: DropMerge, dm, Suika, NEET）
- [Atari横断](subjects/atari.md) — 学習中最良のAtari-5回帰で、実時間優先のBTR比は50M・seed1で66.8%、100M・seed2で83.8%だった。サンプル効率優先50Mは458.3、BTR比82.5%に達した。 （別名: Atari, ALE, Atari-5, Atari5）
- [Breakout](subjects/breakout.md) — LN512はRR4の天井を押し上げた一方、RR1では対照のブレ幅内だった。BTR型NNとSNの効果はreplay ratioと学習予算に依存し、SN無しの早期優位が後期に反転した。 （別名: Breakout, breakout）
- [kung_fu_master](subjects/kung_fu_master.md) — 実時間優先3 seedのうちseed3はボス行きを学び、15M以降の評価の面クリアはほぼ100%だったが、試した設定変更では定着しなかった。最終スコア75,000点の断絶は1UPへの到達に対応し、ボスや面の進行は示さなかった。 （別名: KFM, kfm, Kung-Fu Master, KungFuMaster）

## 機構

- [計測の負荷と比較条件](mechanisms/diagnostic_cost.md) — 当時のTimeHistogramでは22.37秒中16.26秒を全生値の再構築が占め、出力intervalを下げても取得・再構築の量は減らなかった。Breakoutの可塑性probeはinterval100で約2.1%のthroughput低下、500ではほぼ差がなかった。 （別名: 計測負荷, overhead, TimeHistogram）
- [Metrics Viewerの表示粒度](mechanisms/metrics_resolution.md) — 調査時の粒度混在は共有点予算・RunごとのLOD選択・古いwindowの再利用に起因し、引き金はmemory pressureの検出ではなかった。 （別名: MetricsViewer, LOD, common Step bins, maxPoints, overview）
- [LLMの外部知識と探索](mechanisms/llm_prior_knowledge.md) — 調査ではAtariの条件付き改善が見つかった一方、KungFuMasterのボス攻略やqbert第8面を直接改善した実証は見つからなかった。 （別名: LLM, VLM, Motif, ELLM, Read and Reward, LLM-Explorer）
- [Model-based Atariの評価](mechanisms/model_based_evidence.md) — SimPLeの低データ性能の優位は、後に調整されたmodel-freeの結果で相殺された。world model内の学習コストと評価手順への批判が残っている。 （別名: SimPLe, MuZero, EfficientZero, Atari100k, world model）
- [行動の時間尺度](mechanisms/temporal_abstraction.md) — TempoRLは行動へのコミット長を学ぶ設定で素のQ学習より速い学習を、FiGARはA3C・TRPO・DDPGへの行動反復追加で改善を報告した。小さい時間刻みでQ学習が失敗する条件も報告されている。 （別名: TempoRL, FiGAR, action repeat, SMDP, action gap）
- [Action queryとattentionの証拠](mechanisms/action_query_attention.md) — Action Q-Transformerの性能上の優位は一部ゲームに限られた。TQLはattention entropyを制御した条件で大きな価値Transformerの改善を報告したが、anetで同じ改善は未確認である。 （別名: AQT, ActionQuery, Transformer, TQL, attention）
- [報酬尺度と割引率](mechanisms/reward_scale_discount.md) — 調査ではDQN系・約200M framesで報酬clipだけを外した整ったA/Bは確認できなかった。DropMergeのγ0.999は100Mで基準とほぼ同点だったが、序盤と中盤の推移は異なった。 （別名: reward clip, TBO, h(x), Pop-Art, gamma, discount）
- [Target更新の比較](mechanisms/target_update.md) — n=2でhard C=125だけが低分散という主張は、n=4で撤回された。DropMergeのIQN＋Munchausenでは同時定数のhard C=200がsoftの2 seedを下回り、NOOPも増えた。 （別名: target network, hard update, soft update, τ, C）
- [勾配ノイズとbatch](mechanisms/gradient_noise_batch.md) — GNSと有効batchの対応はoptimizer・曲率・抽出方法・予算に依存する。DropMergeでは臨界batchが学習とともに増えたが、batch増加の最終target評価の差は学習側の差より小さかった。 （別名: GNS, gradient noise scale, Bnoise, batch size）
- [NoisyNetの観測](mechanisms/noisynet.md) — LunarLanderの3 seedではNoisyNetの成績差はseed幅内で、行動変化率は終盤に2〜4%へ縮んだ。DropMergeではεラダーへの上乗せではNEETが残り、NoisyNetへの置換で抑えられた。 （別名: NoisyNet, noisy, nzrep, μ-only）
- [評価と比較の条件](mechanisms/evaluation.md) — Breakoutではε=0.01で見えない腕の差がε=0に現れ、SNの効果の符号も20Mと50Mで逆転した。DropMergeでは同設定複製のtarget評価が一致した一方、online評価にはずれが残った。 （別名: eval, Greedy, ε, target, seed, 評価）
- [τ配置とIQN比較](mechanisms/tau_sampling.md) — 直接比較を確認できなかった初期調査に対し、DropMergeの2 seedではstratifiedとantitheticが有力だった。M削減の速度利得とM64による十分なlate改善は確認されなかった。 （別名: IQN, QR, tau, stratified, antithetic, quantile）
- [可塑性指標と成績](mechanisms/plasticity.md) — 同じ勾配step付近の摩耗はreplay ratio間で似ていたが、付随する新規データ量と成績の推移は異なった。SiLUやLN512で機序指標が改善しても、RR8短期screeningの成績は改善しなかった。 （別名: plasticity, srank, dormant, dead ratio, SN）
- [Replay ratioとlane窓](mechanisms/replay_ratio_lane_window.md) — Breakoutの該当構成ではlane窓16,384の2点がε=0で685.12・685.30、窓8,192が615.73だった。効果の機序は未測定である。 （別名: RR, replay ratio, replay_ratio, lane window, capacity）

## アプリケーション

- [Metrics Viewer](apps/120_metrics_viewer.md) — MetricsマスタをRunごとのSQLiteキャッシュへ取り込み、range queryとブラウザ描画を通して長いRunを比較する。（別名: 120_metrics_viewer）。
- [RunnerApp](apps/110_runner_app.md) — RunnerFrame・TrainPanel・EvalPanel・QValuePanelを通してRunの起動、学習とGUI、専用評価の接続をまとめる（別名: AnetRLRunner, AnetRLRunner.exe）。

## 機能カテゴリ

- [325 Agent個別実装 MuZero系列](categories/325_muzero.md) — MuZeroAgentは登録済みAgentの試作実装である。（別名: 325_muzero）。
- [324 Agent個別実装 DQN系列](categories/324_dqn.md) — DefaultDQNAgentとRainbowAgentが共通DQN componentを合成する具象実装を扱う。分類上同じ範囲にあるImageClsAgentも一覧に載せる。（別名: 324_dqn）。
- [322 Env個別実装](categories/322_env_individual.md) — 共通Env contract上のCartPole、GridMaze、LunarLander、DropMerge、ImageCls、Atariなどの具象環境の所在をまとめる。（別名: 322_env_individual）。
- [321 Agent 共通基盤](categories/321_agent_core.md) — Agent、Actor、Learnerの生成・行動・更新と、Agentが所有するResource、更新部品が所有するStateの境界を扱う。（別名: 321_agent_core）。
- [320 Env 共通基盤](categories/320_env_core.md) — SingleDiscreteEnvとBatchEnv、spec・state・Reset・Step、factoryとBuilderの共通契約を扱う。（別名: 320_env_core）。
- [319 GUI 共通基盤](categories/319_gui.md) — Env固有ViewをRunner GUIへ接続し、AuiLayoutFrameがdockサイズ往復・遷移時同期・paneとメニューの連動を担う。（別名: 319_gui）。
- [318 強化学習 共通基盤](categories/318_rl_core.md) — ReplayBufferのExperience、N-step、PER、stack履歴、転送とprefetchの共通契約を引く。RunnerとObserverの実行・通知の所在もまとめる。（別名: 318_rl_core）。
- [317 可視化・メトリクス](categories/317_metrics.md) — eventからscalar、trace、画像、動画、GraphVizとartifactへ変換し、値の取得と保存の境界をまとめる。（別名: 317_metrics）。
- [316 設定管理](categories/316_config.md) — ConfigDataに設定を集約し、ConfigResolverがinclude・selection・Run・CLI・継承とprofileを解決する。（別名: 316_config）。
- [315 ニューラルネットワーク](categories/315_nn.md) — TensorDictから設定でNetworkを構築し、Bodyのbranch DAGとHead、forward、clone/sync、可視化の境界をまとめる。（別名: 315_nn）。
- [314 テンソル処理](categories/314_tensor.md) — device転送とframe stackを担い、sampleのTensorとCPUに残すmetadata、非同期転送の寿命を扱う。（別名: 314_tensor）。
- [313 プロファイリング](categories/313_profiling.md) — ProfileRangeとmacroで関数・処理phaseをCPU/GPU profilerへ安定名で記録する。（別名: 313_profiling）。
- [312 ユーティリティ](categories/312_util.md) — 設定からRunを組み立てる実行基盤のうち、初期化、seed、schedule、log、application共通処理の所在を引く。（別名: 312_util）。
- [311 開発ツール](categories/311_dev_tools.md) — 開発・検証、分析ツール、探索harnessとCIの関係を引く（別名: 311_dev_tools, 開発プロセス, テスト基盤）。

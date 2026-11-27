# wiki の目録

[anet-lab](../../README.md) › wiki の目録

[規約](README.md)・[作業の記録](log.md)。出典は各ページから正本をたどる。

## 題材

- [GridMaze](concepts/subjects/gridmaze.md) — 6×10 の迷路ではスキップ長を選ぶ方策によって判断頻度は変わったが、目標に届く速さと終盤の評価は同程度だった。（別名: GridMaze, 6×10, 迷路）

- [実ロボット課題の選定](concepts/subjects/robot_tasks.md) — 実機候補の調査では、自動resetと成功検知が可能なアーム・治具・制御リグが挙がった。価格が安くても長時間の自律収集が成立するとは確定していない。 （別名: robot, Offline RL, reset, 実ロボット）
- [Name This Game](concepts/subjects/name_this_game.md) — サンプル効率優先50Mは実時間優先50M比+24%、BTR比102%だった。45.7M・ε=0の最良28,465は、実時間優先seed2・100Mの最良24,851を上回った。 （別名: NameThisGame, name_this_game, ntg）
- [Double Dunk](concepts/subjects/double_dunk.md) — サンプル効率優先のε=0評価は20.4Mで22.1、45.7Mで23.2（BTR比101%）だった。実時間優先seed2は50.7Mで21.5、91.1Mで23.0に達した。 （別名: DoubleDunk, double_dunk）
- [Phoenix](concepts/subjects/phoenix.md) — サンプル効率優先45.7M・ε=0の最良399,895はBTR Table A2比94%だった。同断面の100本中44本が27,000 step上限で打ち切られた。 （別名: Phoenix, phoenix）
- [Battle Zone](concepts/subjects/battle_zone.md) — 低εラダーのε=0最良は118,388、旧ラダーは115,533だった。64 env・容量1Mの2 seedは40–50M評価が94,621・86,616で、128 env・2Mの106,389を下回った。 （別名: BattleZone, battle_zone, bz）
- [Qbert](concepts/subjects/qbert.md) — サンプル効率優先の1 seedでは初期値をPyTorch既定へ戻すと序盤の評価がBTRの帯に入り、50Mで第8面の突破が増えた。実時間優先では改善しなかった。第9面で長く点を重ねるゲームがあり、高得点を面の進行と分けて読む。TempoRLの評価時の固定・ノイズの差は学習への効果と区別する。（別名: Qbert, qbert, Q*bert）
- [LunarLander](concepts/subjects/lunarlander.md) — 同条件の 3 seed では、ホバリングを抜ける時期に QR と IQN で明確な差はなかった。持続的に T180 に達したのは QR が 1 本、IQN が 3 本だった。 （別名: LunarLander, ll, ホバリング）
- [DropMerge](concepts/subjects/dropmerge.md) — IQNとQR51の順位はUQE評価とGreedy評価で逆転した。NoisyNetへの探索置換はNEETを抑えたが、batch増加の学習側の上乗せの大部分はノイズ感度の差だった。TempoRLのscratch（分離×合算、1本）はGreedyで無効の対照の下で、新ビルドの対照は旧ビルドより30M以降に低いが、決定的モードでは新旧のexeの学習系列が30Mまで一致した。 （別名: DropMerge, dm, Suika, NEET）
- [Atari横断](concepts/subjects/atari.md) — 学習中最良のAtari-5回帰で、実時間優先のBTR比は50M・seed1で66.8%、100M・seed2で83.8%だった。サンプル効率優先50Mは458.3、BTR比82.5%に達した。BTR比は評価回数とseed数がそろわず、公開CSVを評価10回のmaxで取ると92%（90〜96%）になる。観測縮小のarea表記は数値一致を意味しない。 （別名: Atari, ALE, Atari-5, Atari5, BTR比）
- [Breakout](concepts/subjects/breakout.md) — LN512はRR4の天井を押し上げた一方、RR1では対照のブレ幅内だった。BTR型NNとSNの効果はreplay ratioと学習予算に依存し、SN無しの早期優位が後期に反転した。 （別名: Breakout, breakout）
- [BeamRider](concepts/subjects/beam_rider.md) — サンプル効率優先の 50M では、スキップなしの 2 seed で Greedy の成績と到達セクターが大きく分かれた。 （別名: BeamRider, beam_rider, Beamrider, br）
- [Freeway](concepts/subjects/freeway.md) — RAM の 0xEA は効果音のタイマーで、91 に入った回数が車に当たった回数と一致した。面とボスが無く、RAM の番号は 4 だけを置いた。 サンプル効率優先10M（1 seed）ではスキップなしとTempoRLの合算が上限付近で並んだ。 （別名: Freeway, freeway, fw）
- [ImageCls](concepts/subjects/imagecls.md) — Food-101のスクラッチ学習の到達点はResNet18ish_hr＋cosine＋Mixup/CutMixの86.0〜86.4%（seed 1の3本）で、同じ学習条件のConvNeXt Atto・Femto（stem stride 2）は83.1・83.7%だった。 （別名: ImageCls, imagecls, imgcls, Food-101, food101, 画像分類）
- [Montezuma's Revenge](concepts/subjects/montezuma_revenge.md) — サンプル効率優先＋RND（β 0.1）の 30M、1 seed では、ε=0.01 の評価は 14.21M から鍵と扉で 400 点を繰り返した。Greedy の主評価は 20.4M まで全ゲーム 0 点。同じ重みの別評価では 20.4M に 16 ゲーム中 1 ゲームが 400 点だった。この Run には RND なしの対照がない。 （別名: Montezuma, montezuma_revenge, MontezumaRevenge, mz, mz_tg, Taiga, b001, b01, rnd_b001, b001_s2, r123, r123_s2, c50m, b0003, vd, r121, g99ne_split, g99ne, g99ne_s2, g99ne_split_s2, g99ne_cap4m, cap4m, rbw_tg, rbw_tg_s2, mz_rbw_rndtg, merge, split, 価値の分解, reward_clip, g99ne_rr4, g99ne_rr4_s2, n10, n10_s2, RR4, n-step, reuse4, lr3e4, RFD0191）
- [kung_fu_master](concepts/subjects/kung_fu_master.md) — 実時間優先の 3 seed のうち seed 3 だけがボス行きを学び、15M 以降は評価でほぼ毎回面をクリアした。既定の設定を 10M で 16 seed 試すと 7 本がボスに挑んだが、撃破÷命中は上がらなかった。 （別名: KFM, kfm, Kung-Fu Master, KungFuMaster, r123b003, r123ce, b003w1m）

## 手法

- [自己模倣による価値学習](concepts/methods/self_imitation.md) — 過去に得た高い収益を使う方法には、良い軌跡を再利用できる利点と、偶然の高収益を繰り返し学ぶ偏りがある。（別名: SIL, SAIL, self-imitation learning, 自己模倣）

- [優先度付き経験再生（PER）](concepts/methods/prioritized_replay.md) — TD 誤差で遷移を選ぶと、予測できない誤差にも抽出が偏る。優先度の初期値・更新方法・重要度重みをそろえる必要がある。（別名: prioritized experience replay, PER, TD 誤差, 初期優先度）

- [訪問回数と擬似カウントによる探索](concepts/methods/pseudo_counts.md) — 密度が観測の前後でどう変わるか、またはどの特徴を同じ状態と数えるかで、未訪問状態へのボーナスが変わる。Montezuma では価値更新の方法にも結果が左右された。 （別名: CTS, PixelCNN, pseudo-count, SimHash, φ-EB）
- [訪問済み状態へ戻る探索](concepts/methods/return_and_explore.md) — 有望な状態を記憶し、そこへ戻ってから探索する。シミュレータの復元、行動列の再生、目標条件付き方策では、戻れる環境と必要な環境操作が異なる。 （別名: Go-Explore, detachment, derailment, archive, DTSIL, MEGA, IGE）
- [LLMの外部知識と探索](concepts/methods/llm_prior_knowledge.md) — Atari・ロボットで効いた条件と失敗。小型モデル、報酬整形の保証、更新則発見、言葉と環境の接続。LAGEAの単一GPU条件を確認。KungFuMasterのボス・Qbert第8面への直接の実証は未確認。 （別名: LLM, VLM, Motif, ELLM, Read and Reward, LLM-Explorer, Robometer, CriticHack, LIMEN, iLLM, Intelligent Go-Explore, DiscoGen, LPO, ONI, RLang, DECKARD, SGRL, LLM-TALE, LAGEA）
- [Model-based Atariの評価](concepts/methods/model_based_evidence.md) — SimPLeの低データ性能の優位は、後に調整されたmodel-freeの結果で相殺された。world model内の学習コストと評価手順への批判が残っている。 （別名: SimPLe, MuZero, EfficientZero, Atari100k, world model）
- [行動の時間尺度](concepts/methods/temporal_abstraction.md) — TempoRL は行動とその繰り返し長を学ぶ。LunarLander、qbert、DropMerge には、学習中の繰り返しと評価時のスキップを分けた比較がある。 （別名: TempoRL, FiGAR, action repeat, SMDP, action gap, temporl, UTE, Lazy-MDP, lazy-MDP, 遷移ごとの割引）
- [Action queryとattentionの証拠](concepts/methods/action_query_attention.md) — Action Q-Transformerの性能上の優位は一部ゲームに限られた。TQLはattention entropyを制御した条件で大きな価値Transformerの改善を報告したが、anetで同じ改善は未確認である。PACの出力クエリの利点は連続制御の1比較だけで、DropMergeの今のHeadより良い根拠にならない。 （別名: AQT, ActionQuery, Transformer, TQL, attention, PAC, Perceiver, 行動クエリ）
- [報酬尺度と割引率](concepts/methods/reward_scale_discount.md) — 調査ではDQN系・約200M framesで報酬clipだけを外した整ったA/Bは確認できなかった。DropMergeのγ0.999は100Mで基準とほぼ同点だったが、序盤と中盤の推移は異なった。遷移ごとの割引でエピソード型と継続型を統一する定式化の保証は固定方策の評価と線形近似までだった。 （別名: reward clip, TBO, h(x), Pop-Art, gamma, discount, 遷移ごとの割引, transition-based discounting）
- [Target更新の比較](concepts/methods/target_update.md) — n=2でhard C=125だけが低分散という主張は、n=4で撤回された。DropMergeのIQN＋Munchausenでは同時定数のhard C=200がsoftの2 seedを下回り、NOOPも増えた。 （別名: target network, hard update, soft update, τ, C）
- [NoisyNetの観測](concepts/methods/noisynet.md) — LunarLander の 3 seed では NoisyNet の成績差は seed のぶれの範囲に収まり、終盤の行動変化率は 2〜4% だった。DropMerge では ε ラダーへの追加で NEET が残り、置き換えると抑えられた。 （別名: NoisyNet, noisy, nzrep, nz_eps0, nz_ladder, μ-only）
- [τ配置とIQN比較](concepts/methods/tau_sampling.md) — DropMerge の 2 seed では stratified と antithetic が有力だった。M を減らした速度上の利得と、M64 による終盤の十分な改善は確認できなかった。 （別名: IQN, QR, tau, stratified, antithetic, quantile, PER）
- [Replay ratioとlane窓](concepts/methods/replay_ratio_lane_window.md) — Breakout の比較では、lane 窓 16,384 の 2 条件は ε=0 で 685.12 と 685.30、窓 8,192 は 615.73 だった。効いた仕組みは未測定である。 （別名: RR, replay ratio, PER, replay_ratio, lane window, capacity, cap4m, RR4, g99ne_rr4, n10）
- [内的報酬による探索の比較](concepts/methods/intrinsic_novelty.md) — NGU 型ボーナスと E3B の比較は、環境・学習器・ほかのボーナスとの組み合わせに依存する。MiniHack の結果から Atari での順位は決められない。 （別名: MEME, DEEP, RLeXplore, Taïga, エピソード内の新しさ,  NGU, E3B, RND, whiten, 誤差の統計, warm-up, 直近重み, 終端の mask, intrinsic reward, 内的報酬, 新しさ, 一生の新しさ, Taiga, 周期リセット, predictor reset, 価値の分解, Agent57, 外的の Head, 内的の Head）
- [重みの初期値](concepts/methods/weight_init.md) — 重みの初期値を torch の既定に戻した効きは構成で分かれた。qbert のサンプル効率優先（SN 付きの残差 conv、1 seed）と Breakout の Nature 型（seed 2 の 1 対）では上がり、qbert の実時間優先（残差に SN が無く、胴体の後ろに LayerNorm、1 seed）では従来の初期値の 2 本を下回った。（別名: 初期値, 初期化, init, he, xavier, orthogonal, 直交, PyTorch の既定, torch の既定, variance scaling, Kaiming, aln_init）
- [Munchausen RL](concepts/methods/munchausen.md) — Munchausen は RND を使わない Breakout と DropMerge では系列で最大の単独の上乗せだったが、RND を足した Montezuma では、有効にした Run が Greedy の 2500 点にほとんど届かなかった。Montezuma の比較は組ごとに 1 本で、無効の群は Double DQN も有効にした 2 変数の比較である。（別名: Munchausen, M-DQN, M-IQN, munch, IQN+M, QR+M, log π, ma0.7）
- [n-step の収益](concepts/methods/nstep_return.md) — Montezuma で n-step を 3 から 10 にした 2 本は、Greedy で剣で敵を倒すゲームが 16 本中 8 本以上になった最初の回が 12.2M・14.2M（n-step 3 の 2 本は 26.2M・18.2M）で、外的の Q の 1 位と 2 位の差は 4M 窓 8 つのうち 5 つで大きかった。2 組は実行体の build も違う。（別名: n-step, n_step, nstep, nstep3, n10, 多段の目標, 多段の更新, λ 収益, Retrace, Peng's Q(λ), MMC）

## 測り方

- [計測の負荷と比較条件](concepts/measurement/diagnostic_cost.md) — TimeHistogram の調査では 22.37 秒中 16.26 秒を全生値の再構築が占め、出力間隔を広げても取得と再構築の量は減らなかった。 （別名: 計測負荷, overhead, TimeHistogram, Evaluation Trace）
- [Metrics Viewerの表示粒度](concepts/measurement/metrics_resolution.md) — 調査時には Run ごとの表示粒度が混在した。現行の Viewer は同じグラフの Run を共通の Step 区間で集約し、記録が無い区間を補間しない。（別名: MetricsViewer, LOD, common Step bins, maxPoints, overview）
- [評価と比較の条件](concepts/measurement/evaluation.md) — 1 ゲーム 2 seed で確認できるのは観測した方策どうしの差とその幅までである。BTR の公開 CSV では学習中の最良値が最後の 10 回の平均をゲームの中央値で 1.15 倍上回った。 （別名: eval, Greedy, ε, target, seed, 評価, lane, seed数, max報告, IQM, 信頼区間）

## 現象

- [勾配ノイズとbatch](concepts/phenomena/gradient_noise_batch.md) — GNSと有効batchの対応はoptimizer・曲率・抽出方法・予算に依存する。DropMergeでは臨界batchが学習とともに増えたが、batch増加の最終target評価の差は学習側の差より小さかった。 （別名: GNS, gradient noise scale, Bnoise, batch size）
- [可塑性指標と成績](concepts/phenomena/plasticity.md) — 同じ勾配 step 付近での摩耗は replay ratio が違っても似ていたが、その間に入る新規データ量と成績の推移は異なった。崩壊時に重みが膨らんだのは feature 側だった。 （別名: plasticity, srank, dormant, dead ratio, SN）

## アプリケーション

- [Metrics Viewer](concepts/apps/120_metrics_viewer.md) — MetricsマスタをRunごとのSQLiteキャッシュへ取り込み、range queryとブラウザ描画を通して長いRunを比較する。（別名: 120_metrics_viewer, 設定表, 範囲統計, Run比較）。
- [RunnerApp](concepts/apps/110_runner_app.md) — RunnerFrame・TrainPanel・EvalPanel・QValuePanelを通してRunの起動、学習とGUI、専用評価の接続をまとめる（別名: AnetRLRunner, AnetRLRunner.exe, テーマ, ダークモード, dark mode, DPI, 高DPI, gdiScaling, PerMonitorV2）。

## フレームワーク

- [325 Agent個別実装 MuZero系列](concepts/framework/325_muzero.md) — MuZeroAgentは登録済みAgentの試作実装である。（別名: 325_muzero）。
- [324 Agent個別実装 DQN系列](concepts/framework/324_dqn.md) — DefaultDQNAgentとRainbowAgentが共通DQN componentを合成する具象実装を扱う。分類上同じ範囲にあるImageClsAgentも一覧に載せる。（別名: 324_dqn）。
- [322 Env個別実装](concepts/framework/322_env_individual.md) — 共通Env contract上のCartPole、GridMaze、LunarLander、DropMerge、ImageCls、Atariなどの具象環境の所在をまとめる。（別名: 322_env_individual, GridMaze, seed_mode, DropMerge, NOOP, noop_penalty, RAMメトリクス, distinct_seen）。
- [321 Agent 共通基盤](concepts/framework/321_agent_core.md) — Agent、Actor、Learnerの生成・行動・更新と、Agentが所有するResource、更新部品が所有するStateの境界を扱う。（別名: 321_agent_core）。
- [320 Env 共通基盤](concepts/framework/320_env_core.md) — SingleDiscreteEnvとBatchEnv、spec・state・Reset・Step、factoryとBuilderの共通契約を扱う。（別名: 320_env_core）。
- [319 GUI 共通基盤](concepts/framework/319_gui.md) — Env固有ViewをRunner GUIへ接続し、AuiLayoutFrameがdockサイズ往復・遷移時同期・paneとメニューの連動を担う。（別名: 319_gui）。
- [318 強化学習 共通基盤](concepts/framework/318_rl_core.md) — ReplayBufferのExperience、N-step、PER、stack履歴、転送とprefetchの共通契約を引く。RunnerとObserverの実行・通知の所在もまとめる。（別名: 318_rl_core, Notifier, eval_schedule, 強制行動, ForcedActions, MakeActionOptions）。
- [317 可視化・メトリクス](concepts/framework/317_metrics.md) — eventからscalar、trace、画像、動画、GraphVizとartifactへ変換し、値の取得と保存の境界をまとめる。（別名: 317_metrics, HeatMap, TimeHistogram, Histgram, SweepedHeatMap, wxImage）。
- [316 設定管理](concepts/framework/316_config.md) — ConfigDataに設定を集約し、ConfigResolverがinclude・selection・Run・CLI・継承とprofileを解決する。（別名: 316_config, 読まれない設定キー, 在庫, 完全キー）。
- [315 ニューラルネットワーク](concepts/framework/315_nn.md) — TensorDictから設定でNetworkを構築し、Bodyのbranch DAGとHead、forward、clone/sync、可視化の境界をまとめる。（別名: 315_nn, ForwardCapture, stack軸）。
- [314 テンソル処理](concepts/framework/314_tensor.md) — device転送とframe stack、観測の加工（Forwarderの並び）を担い、sampleのTensorとCPUに残すmetadata、非同期転送の寿命を扱う。（別名: 314_tensor, LaneTimeTensorStore, tensor_store, Forwarder, ForwarderBuilder, Perception, FrameStack, 観測の加工）。
- [313 プロファイリング](concepts/framework/313_profiling.md) — ProfileRangeとmacroで関数・処理phaseをCPU/GPU profilerへ安定名で記録する。（別名: 313_profiling）。
- [312 ユーティリティ](concepts/framework/312_util.md) — 設定からRunを組み立てる実行基盤のうち、初期化、seed、schedule、log、application共通処理の所在を引く。（別名: 312_util, warn_once）。
- [311 開発ツール](concepts/framework/311_dev_tools.md) — 開発・検証、分析ツール、探索harnessとCIの関係を引く（別名: 311_dev_tools, 開発プロセス, テスト基盤, MKLDNN, oneDNN, リリース, release, CPack, 台帳, issue, 課題記録, 索引のissue, RFDの課題, rfd欄, RFD, 文書の呼び名, 4桁の番号, Obsidian, まとめページ, summary page, anet-summary-page, 要約, compaction, hook, compact_cue）。

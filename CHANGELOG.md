# anet-lab Change Log

注記: v0.1.0 以降の変更を記載する。それ以前(V1 系の開発履歴)は git log を参照。
1 行 = RFD・機能・不具合事象の 1 単位。末尾の (RFDnnnn) は `docs/memo/` の RFD 番号、
(ADRnnnn) は `docs/adr/` の設計判断。ハイパラ調整・実験記録・Run 設定だけの更新は載せない
(`docs/experiments/` 側で追う)。

更新のタイミング: リリース前にまとめて更新する。拾う先は前回のタグから main までの `git log` と、
`docs/memo/done/` へ移した RFD。


## 0.4.0: (2026-11-23 リリース)

GitHub Release: https://github.com/kazukin123/anet-lab/releases/tag/v0.4.0
(設定体系の全面移行、workspace、Atari、内的報酬(RND)と価値の分解、TempoRL、NoisyNet、Evaluation Trace、ソース配置の再編)。

後方互換は既定の要件としない(AGENTS.md「クリーンブレーク方針」)。v0.3.0 の設定・成果物をこの版で使うときに
読み替えが要る変更を、次の 3 つの一覧にまとめる。

非互換(設定を書き換えないと起動できない):

- 設定体系をプロファイル(`@name`)と選択チェーン(`X.$ = A > B`)へ全面移行。旧 `*.baseline.*` の直書きは通らない(RFD0059, ADR0030)。
- Run 設定の root を `train.*` から `run.*` へ改名(`run.seed`、`run.train.num_envs`、`run.eval.[tag]`)(RFD0061)。
- eval のスケジュールを `run.eval_schedule.[tag].interval` へ分離。interval は必須で、0 が OFF(RFD0052, ADR0027)。
- 評価スロットの方策は Actor カタログ `<Agent>.actor.[key]` へ移動。`run.eval.[tag].actor` で名前参照する(RFD0061, ADR0038)。
- device 指定を `agent.device` / `env.device` / `run.eval_device` に統一。旧 `*_device_type` / `*_device_index` は無効(RFD0070)。
- QR-DQN の `use_qr` / `num_quantiles` は `quantile_mode = none|qr|iqn` / `qr.num_quantiles` へ(RFD0001, ADR0018)。
- 探索・学習の時間の値を ProfiledValue へ統一。`eps_start`・`learner.alpha`・`per_beta_start` などの旧キーと新キーの対応は、節末の「RFD0101 ProfiledValue のキー対応」(RFD0101, ADR0064)。
- Head の設定キーを `head.*` の下へ移動。`head_init.mode` / `head_init.manual_gain` は `head.init.mode` / `head.init.manual_gain`、DefaultDQN の `use_dueling_net` は `head.use_dueling`(RainbowAgent は `use_dueling_net` のまま)(RFD0098)。
- 共用の特徴境界を DQN の `learner.final_feature_branch_key`、ImageCls の `final_feature_branch_key` へ改名。過去 dump の再生時は旧 `learner.plasticity.feature_key` / `plasticity.feature_key` を読み替える(RFD0093)。
- GridMaze の `randomize_seed` を廃止し、`seed_mode`(`process` / `episode`、既定 `process`)へ移行。`seed` を書けばその値で固定する(RFD0102)。
- Runner の LogPanel の表示 level を `app.log_level` から `app.log_panel.level` へ改名。
- Conv2d 可視化の `conv2d.mergin_x` / `conv2d.mergin_y` を `conv2d.margin_x` / `conv2d.margin_y` へ改名(RFD0182)。
- 読まれない設定キーを検出する。実体化したカタログ項目の中のものは構築時に止め、ほかは Run の構築の終わりに WARN の一覧へ出す。旧キーが残った設定は直す(RFD0117, ADR0071)。
- Run は `data/workspaces/<ws>/runs` へ、Env 選択は workspace の `config/_main.txt` へ。`app.runs_dir` は指定禁止(RFD0046, RFD0182, ADR0021, ADR0078)。
- ソースを `src/` の下へ再編。Runner の appdata は `data/appdata/`、実行時の道具は `apps/tools/`、Metrics Viewer の jar は `apps/metrics-viewer/bin/`、Python の仮想環境はリポジトリ直下の `.venv` に置く(RFD0182, ADR0078)。
- Runner は引数省略時に workspace 選択ダイアログを出す。bat / CLI からは `run.$=run.@name` で Run プロファイルを選ぶ(RFD0046, RFD0072)。
- Metrics Viewer は workspace 単位で Runs を読む。Optuna 用の別起動 bat `22_metrics_viewer_java_optuna.bat` は廃止(RFD0046)。
- frame stack 用の Reshape ブロック(`ReS4F8` 等)は不要になり、NN 配線に残っていると shape が合わない(RFD0057)。
- C++ の拡張点: Actor の `ActionContext` を廃止し、Actor が RNG と frame stacker を直接持つ(RFD0083)。NN の forward は用途ごとに準備した実行へ移し(RFD0084, ADR0051)、名前付き出力は capture に一本化した(RFD0170, ADR0072)。Agent は系統ごとのライブラリに分かれ、基盤の公開ヘッダ `include/anet/` だけを読む(RFD0182, ADR0079)。
- ビルドに wxWidgets 3.3.3 が必要。DLL を差し替えると 3.3.1 でビルドした旧 exe は起動できない。

非互換(黙って挙動・値が変わる):

- 既定キー `?=` を導入。`=` は個別指定としてチェーンの供給値に勝つため、共通ファイルに残した `=` は再確認する(RFD0072, ADR0040, ADR0042)。
- eval は `eval_batch_size`(lane 数)×`eval_episodes`(本数)の N 本平均を 1 点として記録する。1 エピソード 1 点ではない(RFD0060, ADR0034)。
- eval schedule は先頭の発火を既定で 1 回抑止する(`warmup_count=1`)。学習直後の評価が要るタグは 0 を指定し、`offset`(既定 0)で learn step の発火列をずらす(RFD0158, ADR0070)。
- interval 系の発火が bucket-crossing 化され、perf メトリクスは時間重み EMA になった(RFD0053, ADR0028)。
- EMA メトリクスに初期値のバイアス補正が入り、序盤の値が変わる(RFD0045)。
- Env の遷移の結果は Step の結果で運ぶ。エピソード完了時の値は target `$env_step` で読み、`$env` は今の状態だけを返す。`$env ... @train` で完了時の値を読んでいたメトリクス定義は `$env_step ... @episode_end` へ移す(RFD0097, ADR0060)。
- checkpoint から分岐した Run では、設定ファイルの optimizer 設定(学習率・weight_decay・eps)が実効値になる。moment と step は checkpoint から引き継ぐ(RFD0091, ADR0053)。
- ReplayBuffer は ring 折り返しで stack 履歴を失った transition を sample 候補から外す(RFD0050, ADR0024)。
- 観測履歴は Push 時にエピソード開始から始める。旧実装で混入していた前エピソードのフレームは出ない(RFD0078, ADR0044)。
- GridMaze で解けない seed の迷路は、派生 seed で決定的に引き直す。解ける seed の迷路は変わらない(RFD0100)。
- LunarLander の既定の Run で TempoRL を有効にした(`run.$ = run.@munch > run.@skip`)(RFD0098)。
- Atari のメトリクスを `game_score` / `game_len` / `game_frames` / `hns57` / `hns49` に改名。
- 解析用メタデータ `metrics.scalar.defs` / `metrics.trace.defs` を Runner が出力。解析側は設定から再導出しない(RFD0054, RFD0069, ADR0029)。
- Run 終了時に background 評価セッションの完走を `app.drain_timeout_sec`(既定 3600 秒)まで待つ(RFD0076, ADR0045)。
- nn_trace は要求した step だけ生成する。Conv2d pane を閉じた間の Trace 行には中間出力を保持しない。Actor と Runner の step は `MakeActionOptions` を受け、NN の中間出力は capture に統合した(RFD0170)。

v0.3.0 のあとに入った機能の設定キーの移し替え(開発中の Run の設定を読むとき):

- RND の設定を段階・モード別に移行(RFD0191, ADR0088)。`intrinsic.rnd` の `statistics_mode` は `statistics.mode`、`obs_norm.*` は `statistics.whiten.obs_norm.*`、feedback の観測 clip は `statistics.variance_feedback.obs_norm.clip_range`。`error_norm.use_centering` / `average_mode` / `ema_steps` は `error_norm.standardize.*`。`multiplier_mode` / `multiplier_max` は選ぶ方式の `use_clipping` / `multiplier_max` へ移す(`error_norm.standardize`、`error_norm.none`、`statistics.variance_feedback`)。`warmup_mode` / `warmup_steps` は、`learn_predictor` なら `error_norm.standardize.warmup_steps`、`collect_errors` なら標準化の待ちを 0 にして `predictor.warmup_steps`、feedback なら `predictor.warmup_steps` に移す。`intrinsic.learn_ratio` / `intrinsic.learn_batch_size` は `intrinsic.learn.intake.ratio` / `intrinsic.learn.intake.batch_size`。feedback の倍率 clip の既定は偽。
- RND の出力次元を `intrinsic.rnd.output_dim` から `intrinsic.rnd.head.output_dim` へ移動。
- TempoRL の GridMaze の skip 方策を epsilon-greedy へ、Atari の評価 session 名を `greedy` へ、長さ別 tag の添字を 2 桁へ移行。split の checkpoint は追加の optimizer 状態を持ち、stop / merge への読み込みは拒否する(RFD0103)。

全般:

- 設定体系をプロファイル / 選択チェーンモデルへ全面移行し、ConfigResolver による解決基盤を導入(RFD0059, ADR0030)。
- 設定継承の既定キー `?=` と個別指定 `=` を区別し、選択の最終値参照と Run プロファイルの優先段を導入(RFD0072, ADR0040, ADR0042)。
- device 指定を 3 キーへ統一し既定 `auto` を導入。CUDA 非搭載機では CPU で起動する(RFD0070)。
- Run 設定の root を `train.*` から `run.*` へ改名(RFD0061)。
- 探索・学習の時間の値(ε、UQE の ε と τ、学習率、PER β、同期間隔など)を ProfiledValue に統一し、時間変化を同じ書き方で指定できるようにした(RFD0101, ADR0064, ADR0065)。
- 読まれない設定キーの検出を追加。WARN は利用者が直すべき事象だけに絞り、正常な Run で出続けていた WARN を解消(RFD0117, ADR0071)。
- エラー・WARN・INFO が示す設定キーを、設定ファイルにそのまま書ける完全キーにした(RFD0194, ADR0087)。
- Env ログの prefix を設定記法 `env.[<Env>].[<lane>]:` に統一。
- Notifier の同時呼び出しで Observer の列が壊れる問題、VideoLogger と QValuePanel の範囲外読み、GPU 推論の評価 Actor に強制行動を渡すと止まる問題、同名の重複登録が黙って上書きされる問題を修正(RFD0192, ADR0082)。
- 設定の bool 値が txt 保存時に true/false ではなく 1/0 になる問題を修正。
- worker スレッドの停止通知の取りこぼしでハングする場合がある問題を修正。
- メモリ確保失敗時に要求サイズを含む診断ログを出すよう強化。
- Tracy 起動時のシンボル一括ロードを止め、PDB ファイルのロックを解放。
- Tracy 使用時に、終了したスレッドのヒープを回収してメモリ枯渇を防ぐよう修正。

Agent・学習・評価:

- IQN(Implicit Quantile Networks)に対応。`quantile_mode`、bind `*` 演算子、IQNHead / IQNLearner(RFD0001, ADR0018, ADR0019)。
- IQN の tau 配置方式を 5 方式(stratified / systematic / antithetic を含む)へ拡張(RFD0044)。
- Munchausen RL(M-DQN / M-QR / M-IQN)に対応(RFD0067, ADR0035, ADR0036)。
- NoisyNet(Noisy Networks for Exploration)に対応。Body の Linear と Head の射影に学習可能なノイズを入れ、Actor・Learner ごとにノイズの保持と共有の仕方を選べる。ノイズの効きを見る診断を追加(RFD0084, ADR0048)。
- TempoRL に対応。選んだ行動を何 step 続けるかを skip Network で学ぶ(`skip.enabled`)(RFD0098, ADR0062, ADR0063)。
- TempoRL に loss 別の Adam 更新 `split`、skip loss の重み `kmax`、greedy 評価 session の判断率を追加(RFD0103)。
- TempoRL に、毎 step 判断して skip の出力を補助タスクとしてだけ学ぶモードと、Actor ごとのスキップ長の幅の指定を追加(RFD0104)。
- TempoRL の skip 側の forward の NN 実行設定を、Actor ごとに Q 側と分けて指定できるようにした(RFD0106)。
- RND(Random Network Distillation)による一生の新しさの内的報酬を追加(`intrinsic.enabled`)。target と predictor の Network を個別に指定できる(RFD0116)。
- RND に warm-up と内的報酬の倍率の選択肢を追加。序盤の大きな誤差で膨らんだ誤差統計に内的報酬が抑えられる問題を修正(RFD0123)。
- RND の採点を高速化し、2 つの統計方式を型で分けた(RFD0130)。
- RND に生の予測誤差による内的報酬と、rollout 単位の predictor の補助学習を追加。設定を段階・モード別に組み替えた(RFD0191, ADR0088)。
- Q を外的報酬と内的報酬の 2 つの Head で学ぶ価値の分解を追加(`intrinsic.value_mode = decomposed`)。内的の割引率と終端の扱い、行動を選ぶときの β を選べる(RFD0121, ADR0073, ADR0074, ADR0075)。
- 評価セッション契約を導入。L 並列 / N エピソード集計で 1 点を記録する(RFD0060, ADR0034)。
- eval 定義と定期スケジュールを `run.eval.[tag]` / `run.eval_schedule.[tag]` に分離(RFD0052, ADR0027)。
- eval schedule に `offset` と `warmup_count` を追加。既定では学習直後に評価しない。評価専用 Run は `warmup_count=0` を明示する(RFD0158, ADR0070)。
- Actor 設定カタログを導入し、評価スロットごとに方策・network・clone を名前で選べるようにした(RFD0061, ADR0038)。
- 学習を完全停止した評価専用 Run 向けに `learner.enabled` を追加。
- Run 終了時に background 評価セッションの終了を待つ(または破棄する)排水を追加(RFD0076, ADR0045)。
- DefaultDQN の stack 次元をネイティブ軸として扱い、Reshape ハックを廃止(RFD0057)。
- checkpoint から分岐した Run で、設定ファイルの optimizer 設定が効かない問題を修正(RFD0091, ADR0053)。
- ウォームアップの開始・終了と、評価セッションの進み(30 秒ごと)をログへ出すよう追加。
- 学習器の 1 回の更新の進行と Munchausen の target を 1 か所に寄せ、DQN 系 Agent の責務分担を整理(挙動不変)(RFD0122, RFD0125, RFD0128)。
- IQN 診断の同期を外し、Windows では Env の worker の優先度を下げて学習スレッドへコアを譲るよう修正(性能)。

ReplayBuffer:

- ring 折り返しで stack 履歴を失った transition を sample 候補から除外。ready / sampleable を分離(RFD0050, ADR0024)。
- 観測履歴と bootstrap stack の整合性が壊れる問題を修正。履歴は Push 時にエピソード開始から始める(RFD0078, ADR0044)。
- sampleable index 列の確保を持続バッファ化(RFD0077)。
- RB 履歴整合性テストを追加。`episode_start` 境界の既知失敗は shouldfail 化し、契約裁定を起票。
- 1 step ごとの報酬と終端を保持し、長さの変わる目標値の窓で標本を作れるようにした(RFD0098, ADR0063)。
- 報酬の成分(外的・内的)を別々に保持し、成分ごとの割引率で収益を作る(RFD0121, ADR0073)。
- 割引率を時間で変えるときは、標本の時点で収益を作り直す(RFD0101, ADR0065)。

NN:

- Spectral Normalization(`weight_norm.mode`)に対応。自前実装で buffer 意味論を定義(RFD0065, ADR0032)。
- forward を用途ごとに準備した実行へ移し、公開 API と既定経路を整理(RFD0084, ADR0051)。
- forward の名前付き出力を capture に一本化し、nn_trace は要求した step だけ作る(RFD0170, ADR0072)。
- 重みの初期化に、分散を倍率 ÷ fan_in にする一様分布の方式 `variance_scaling` を追加(Dopamine の IQN・Rainbow と同じ初期値)。初期化の設定の読み取りを 1 つの口にまとめた(RFD0193)。

Env:

- Env の遷移の結果を Step の結果で運び、終端の lane でも通知の中でリセット前の値を読めるようにした。最後の学習の失敗を終了コードへ反映し、BatchEnv の worker の例外を呼び出し元へ返す(RFD0097, ADR0060)。
- AtariEnv を追加。ALE を C++ で直結し、前処理は自前、オプショナルビルド(RFD0051, ADR0025, ADR0026)。
- Atari: reset / episodic_life を標準実装に準拠させ、`game_*` メトリクスへ改名し HNS(hns57 / hns49)を追加。
- Atari: スコア相転移計測用メトリクス(`game_score.ge.[N]`)とゲーム完了ログを追加。
- Atari: ゲーム固有の進行(面クリア・ボス撃破・ボス命中)を RAM から番号付き汎用キーで取るメトリクスを追加(RFD0082, ADR0046)。
- Atari: BeamRider・Freeway・Montezuma の RAM メトリクスと、kung_fu_master の `player_hp_dec` を追加。
- Atari: RAM メトリクスの畳み方に、値の種類数 `distinct_seen` とビットの変化 `bit_rise_count` / `bit_fall_count` を追加(RFD0126)。
- Atari: 観測の縮小方式を `AtariEnv.resize_mode` で選べるようにした。既定は今の方式(RFD0099)。
- Atari: AtariView は既定で元画像だけを出し、中クリックで表示を切り替える。
- Atari: Atari-5 / Atari-100k のバッチ定義を追加。
- Atari: sound 有効時に SDL_AUDIODRIVER を自動設定し、WASAPI 無音を回避。
- Atari: AtariView でテキスト情報が見切れる問題を修正。
- Atari: 実効値が意図と違っていた `noop_max` / `adam_eps` を修正。
- GridMaze: 迷路を必ず解ける形にした。解けない seed は派生 seed で決定的に引き直す(RFD0100)。
- GridMaze: 迷路の seed を、指定値で固定するか、起動単位 / エピソード単位で選べるようにした(`seed_mode`)(RFD0102)。
- LunarLander: LunarLanderView に風と運動の矢印を追加し、中クリックで表示を切り替える。
- LunarLander: 既定の Run で TempoRL を有効にした(RFD0098)。
- DropMerge: 行動モードごとの NOOP の意味と、NOOP への罰の判定が合っていない問題を修正(RFD0115)。

メトリクス・可観測性:

- メトリクス trace チャネル(1 件 = 1 レコード)と SessionEnd イベントを追加(RFD0069, ADR0037)。
- メトリクス定義に購読先・eval 条件・clip 指定を追加(RFD0069)。
- Runner が解析用メタデータ `metrics.defs` を出力(RFD0054, ADR0029)。
- EMA の初期値バイアス補正を導入(RFD0045)。
- interval 発火を bucket-crossing 化し、perf メトリクスを時間重み EMA 化(RFD0053, ADR0028)。
- 可塑性メトリクス(srank / dormant unit 率)と feature / readout 分割の weight norm を追加(RFD0062, RFD0063, ADR0031)。
- policy churn メトリクスを追加(RFD0066, ADR0033)。
- replay 抽選履歴別の TD・損失メトリクス(replay_fit)を追加(RFD0073, ADR0039)。
- IQN 探索 P0 診断メトリクスと分位 tail 診断メトリクスを追加(RFD0047, RFD0048)。
- 共通のエピソード長指標(episode_steps)と評価セッション・待機ログを追加(RFD0075, ADR0043)。
- 実際の更新から勾配ノイズスケールを測る 15 指標を追加。`metrics.scalar.@gradnoise` の購読で有効化し、BF16 と feature / readout に対応(RFD0092, ADR0054)。
- TransformerEncoder の層平均 attention エントロピーを追加。`run.@attention` / `metrics.scalar.@attention` の購読時だけ更新前の probe で測り、`48_agent_attention` へ出す(RFD0093, ADR0055)。
- メトリクスの target `$env_step` を追加し、エピソード完了時の値を `@episode_end` で読めるようにした(RFD0097)。
- HeatMap 系の画像で、メモリの確保と解放の方法が合っていない問題を修正(RFD0115)。
- VideoLogger が ffmpeg の process の終了通知のあとに解放済みのオブジェクトを触る問題を修正。

Runner:

- workspace 機構を導入。Run を workspace フォルダで分類し、Runner / Viewer / Optuna を統合(RFD0046, ADR0021, ADR0022)。
- Evaluation Trace パネルを追加。EvalPanel の遷移ごとの step trace を表で見て、行を選ぶか再生すると EvalPanel・QValuePanel・Conv2d がその時点を出す(RFD0094, ADR0057, ADR0058)。
- Evaluation Trace: 列見出しのクリックでの並べ替えと、行のダブルクリックでのその step の通知のログ出力を追加。Trace モード中は矢印キーで強制行動が起きず、強制した step を見分けられる(RFD0105)。
- Evaluation Trace: 列のドラッグでの入れ替えと、横スクロール中も step 列を出す表示を追加。RND の内的報酬の列と、価値の分解の Head ごとの値の列を追加(RFD0121, RFD0127, RFD0129)。
- 起動時に RunManager を別スレッドで構築し、起動直後の画面の乱れと「応答なし」を解消(RFD0109)。
- Train View を隠したり float したりできるようにした。Train と Eval のどちらかの主 pane が Centre を占め、起動時の主 pane は `app.main_view`(`train` / `eval` / `both`、既定 `train`)で選ぶ(RFD0110, ADR0066)。
- LogPanel に操作バー(Auto-scroll・折り返し・字下げ・検索・Copy all・Clear log)と、未確認の WARN / ERROR の件数表示を追加。
- ステータスバーに起動中・起動完了と Eval の速度切替を表示。Eval の step_per_frame をメニューから変えられるようにした。
- ツールバー 4 本(対象軸)、View FPS メニュー、checkpoint 保存を追加(RFD0055)。
- AlwaysOnTop に対応。FPS メニューに 15、0.5、0.1 を追加。
- batchrun 向けにエラーダイアログ抑止・終了コード・bat 側の致命エラー検出を追加(RFD0068)。
- Eval pane 表示切替時の window 自動リサイズを廃止。右クリックの Eval resume 時も EvalPanel を表示。
- pane 見出しと workspace 履歴の視認性を改善。
- TrainPanel の View data を表示状態や FPS によらず毎 step 作っていた問題を修正(性能)。
- 初期ウィンドウの大きさを画面の作業領域に収めるよう修正。
- Q-Values pane の高さ・幅と、補助列を開いたときのツールバーの並びを修正。最大化中の pane の開閉や Reset Layout で最大化が解けない問題を修正。
- Close 後の pane の並べ直しと、終了時の背景評価の排水の前に train observer を外す処理で落ちる問題を修正。
- EvalPanel の FPS を 0 にするとエラーになる問題を修正。
- wxWidgets 3.3.3 移行の影響で TrainPanel が真っ黒になる問題を修正。

Metrics Viewer:

- workspace 切替に対応(RFD0046)。
- 複数 Run の比較粒度を表示 Step 範囲でそろえ、先読み表示を追加(RFD0089, ADR0052)。
- キャッシュ再構築を高速化し、全 workspace で事前に行うようにした(RFD0090)。
- タグセット(名前付きのタグ選択の保存と呼び出し)を追加。切り替えても未保存の変更をセットごとに残す(RFD0111, ADR0067)。
- Run 設定を比べる設定表、Run・tag の絞り込み、tag の情報パネル(統計)を追加。Plotly を jar に同梱し、CDN から読まない(RFD0196, ADR0089, ADR0090)。
- Run 情報パネルとグラフ高さスライダーを追加。Reload の失敗を Toast で知らせる。
- Reload を高速化。スクリーンショットモードで Reload 中のスピナーを表示。
- percentile 範囲を下限・上限の巡回ボタンに分けた。
- 放棄された query が permit を握り続け、素早い操作で `Update failed` になる問題を修正(サーバ側 supersede)(RFD0049, ADR0023)。
- 不変 Run の定常取り込み I/O を抑制(RFD0064)。
- 選択中 Run の色を分離。Recolor ボタンと AutoRecolor を追加。AutoRecolor は新しく選んだ Run だけを色付けする(RFD0074, ADR0041, ADR0050)。
- グラフのホバー表示を自前描画に置き換え。終了済み系列に別 step の値が混ざる問題を修正。
- タグ別 percentile Y 軸範囲と表示設定の永続化を追加。
- Runs に SelectedOnly トグルを追加。
- スクリーンショットモードでスワイプによる縦スクロールに対応。スクロール位置が飛ぶ問題を修正。
- スマホでグラフをタップしてもポイント情報が出ない問題を修正。
- 統計表示に隠れて Plotly modebar が操作できない問題を修正。
- グラフが消える問題、サンプルが 1 件だけのグラフが描かれない問題、Tags のホバーでスクロールが効かない問題、Scroll Lock 中のスワイプで慣性スクロールが効かない問題を修正。
- JDK だけでビルドできるようにした。Eclipse のインポートエラーと、起動時のデータの参照先を修正。

ツール:

- AI 向け Run 解析 CLI `inspect_run.py` を追加。subcommand 分割と trace-csv(RFD0054)。`env_step` に対応(RFD0097)。
- Optuna ハーネスを workspace へ統合(RFD0046)。
- workspace metrics の一括 gzip 圧縮(`70_compress_workspace_metrics.bat`)を追加。
- Atari-5 の 5 本の Run を 1 つのベンチマークとして集計するスクリプトを追加。
- Atari-5 専用 bat を `11_batch_run.bat` / `12_batch_run.bat` へ統合。
- BTR の変換器に RAM カウンタの取り込みとスコア閾値の指定を追加。
- MLflow bridge が config のコメント行で落ちる問題を修正。
- optuna ハーネスの実設定との不整合を修正。

ビルド・CI・リリース:

- ソースを `src/` の下へ再編し、Agent を系統ごとのライブラリに分けた。テストは自分のライブラリの内部と、ほかのライブラリの公開ヘッダ・テスト用の口だけを読む(RFD0182, ADR0078, ADR0079, ADR0080)。
- wxWidgets を 3.3.3 へ更新。find_package を Config モードへ一本化し、DLL コピー処理を追加。
- runtime DLL コピーのパスを CMAKE_SOURCE_DIR 基準に修正。
- Release ビルドで最適化を保ったまま PDB を作るよう修正。
- Release / RelWithDebInfo の preset で CMake Tools が build type を Debug と誤認する問題を修正。VSCode などからのビルドに対応。
- ENV 別 workspace 枠(`<Env>-01/config/_main.txt`)をリリース zip へ同梱。新しい workspace の初回起動の既定を CPU 版の LunarLander にした。
- 作業用のタグセットのファイルをリリース zip から外し、jar に初期値を入れる(RFD0111)。
- Debug 版のテストを単独やタグ指定で回すと oneDNN の assert で止まる問題を修正(RFD0124)。
- 出荷設定を読むテストを `[shipped_config]` タグにまとめた。
- docs/ を GitHub Pages へ配信するワークフローを追加。

ドキュメント・開発プロセス:

- 文書の呼び名を PRD から RFD(Request for Discussion)に改め、番号を 0 埋めの 4 桁にした(RFD0195, ADR0093)。旧い名前と新しい名前の対応: skill `anet-create-prd`・`anet-impl-prd` → `anet-create-rfd`・`anet-impl-rfd`、ファイル名 `NNN_<slug>_10prd.md` → `NNNN_<slug>_10rfd.md`、課題の欄 `prd` → `rfd` と `phase: PRD` → `phase: RFD`、コミット件名の記号 `PRDnnn` → `RFDnnnn`、テストのタグ `[prdNNN…]` → `[rfdNNNN…]`、検証資材 `testdata/prdNNN/` → `testdata/rfdNNNN/`。
- 9xx の番号の RFD 57 本を、起票日の順に 900 未満の連番へ振り直した(RFD0131, ADR0077)。
- 設計文書の英語版(`*.en.md`)を追加。
- アーキテクチャ Atlas(5 図と索引)と生成 skill `anet-archify-atlas` を追加。
- AI ハーネス動線マップと生成 skill `anet-harness-map` を追加(RFD0080)。
- レビューエージェントと品質確認観点(`docs/quality/`)を追加し、AGENTS.md の細則を観点へ移して「作業ごとに読むもの」の表から読む形にした。レビューを頼む skill `anet-review` を追加(RFD0087, ADR0049, ADR0061)。
- 知見と実装のトピック別まとめを置く LLM-Wiki(`docs/wiki/`)と、取り込み・点検の skill `anet-ingest-wiki` / `anet-lint-wiki` を追加(RFD0095, ADR0056)。
- 課題の記録を 1 課題 1 ファイル(`docs/management/issues/`)へ移し、課題の作成・一覧・更新の skill を追加。課題の種類を仕事の種類にし、RFD とは課題の `rfd` 欄で結ぶ(RFD0114, RFD0120, ADR0068, ADR0069, ADR0081)。
- 開発支援 skill 群(anet-stats / audit / housekeeping / survey-queue / translate-docs、prepare-commit)を追加。
- skill を改名(`grill-with-docs` → `anet-create-rfd`、`implement-prd-with-docs` → `anet-impl-rfd`、`prepare-commit` → `anet-prepare-commit`)。会話の手戻りからハーネスの改善策を選ぶ `anet-improve` と、会話のまとめページを作る `anet-summary-page` を追加(RFD0197)。
- grill skill に goal anchor と簡素化パスを追加(RFD0058)。グリルの前に、判断に要る前提を開発者と共有する手順を追加(RFD0189)。
- 新しいコミットを AI へ知らせる hook と、ディスク全体の再帰検索を止める hook を追加。
- RFD を状態別フォルダ(done / frozen / dropped)へ振り分け、バックログを Phase 別に採番。
- 設計文書 200(DQN 系 Agent)をコード・設定・用語集・ADR にそろえた。文書・コメント・設定の小さな食い違いと、Metrics Viewer の LOD の集約数の記述を直した(RFD0190, RFD0119, RFD0188)。
- AGENTS.md にクリーンブレーク方針、責任境界、Fail-Fast 原則、GetScalar 規約、実験記録と実効 config の保存ルール、Run 命名規約を追記。
- AGENTS.md に、直訳を避ける語の表、コミットメッセージの文言の規則、RFD の外の変更で ADR を起こす条件、クラス宣言の書き方、モードごとの名前空間で設定キーを持つ規約を追記(ADR0076, ADR0091, ADR0092)。
- Atari / DropMerge / LunarLander の実験記録と、サーベイ(SimPLe、報酬スケールと割引率、target 更新速度、BTR)を追加。
- TempoRL・NoisyNet・RND と価値の分解(Montezuma)・qbert・kung_fu_master の実験記録と、サーベイ(NGU・E3B、LLM と RL、勾配ノイズ、内的報酬と PER、TorchRL、FiGAR・TQL・TempoRL の論文解説)を追加。

### RFD0101 ProfiledValue のキー対応

探索・学習の時間の値をProfiledValueへ統一した。定数はbare、段の定数だけは `phase.[name].value` を使う。非constantの下へ継承したbareは保持するが評価・dumpしない。DropMergeとLunarLanderのlogラダー末端0を維持し、中間laneも旧系列から変わる。

| 旧キー | 新キー |
|---|---|
| `eps_start` | `epsilon.start` |
| `eps_end` | `epsilon.end` |
| `eps_decay_steps` | `epsilon.steps` |
| `uqe_eps_start` | `uqe.epsilon.start` |
| `uqe_eps_end` | `uqe.epsilon.end` |
| `uqe_eps_decay_steps` | `uqe.epsilon.steps` |
| `uqe_tau_start` | `uqe.tau.start` |
| `uqe_tau_end` | `uqe.tau.end` |
| `uqe_tau_decay_steps` | `uqe.tau.steps` |
| `use_spatial_exploration` | `spatial.enabled` |
| `spatial_scale_type` | `spatial.scale` |
| `uqe_use_tail_mean` | `uqe.use_tail_mean` |
| `sync_interval.value` | `sync_interval` |
| `learning_rate.value` | `learning_rate` |
| `learner.alpha` | `learner.learning_rate` |
| `per_beta_start` | `per_beta.start` |
| `per_beta_end` | `per_beta.end` |
| `per_beta_step` | `per_beta.steps` |

時間変化は `type = linear` を明示し、旧decay長が0ならconstantとbare開始値へ移す。spatialの旧start/endは `spatial.epsilon.first/last`・`spatial.uqe.epsilon.first/last`・`spatial.uqe.tau.first/last` へ別に写す。共有baseのspatial epsilonは0.4/0.01。既存のbare `weight_decay`・`gamma`・Munchausen値・soft update τの行はそのまま定数になる。PER βは更新末尾に次の抽選要求用の値を評価し、適用時機を維持する。（RFD0101）


## 0.3.0: (2026-08-09 リリース)

GitHub Release: https://github.com/kazukin123/anet-lab/releases/tag/v0.3.0
(実行用 zip と Doxygen zip を添付した初回のパッケージ配布)。

非互換:

- 重み初期化モード `WeightInitConfig.mode` を int から string へ変更(PRD027, ADR0008)。
- Env の RunMode 指定を Reset / Step 引数からコンストラクタへ移動。DefaultBatchEnvFactory を BatchEnvBuilder へ改名(PRD034)。
- Run root への `config.txt` 出力を廃止。実効設定は `config/config_data.txt` に一本化。
- Metrics Viewer を `apps/metrics-viewer` へ移動し、起動 bat を `apps/` 直下へ集約(PRD043, ADR0017)。
- `backend.cudnn_benchmark` の既定を off へ変更(true で同 seed 再現性が失われるため)。
- BatchNorm / LayerNorm は既定で FP32 強制。`app.metrics_logger.video_codec` の既定は `auto`。
- libtorch を 2.12.0(cu130)へ更新。

全般:

- ConfigReader と、アニーリング用の共通部品 ProfiledValue を追加(PRD031)。
- Config の書式エラーを例外で明示的に検出するよう修正。
- 設定グループ名と同じ接頭辞のキーが誤ってマージされる問題を修正。
- ログファイルが定期処理時と終了時に Flush されない問題を修正。
- 空き容量不足などでメトリクス書き込みに失敗した場合はエラーログだけ出し、システムエラーにしないよう修正。
- プレフィクス付きログを共通部品化(PRD038)。

Agent・学習・評価:

- Ape-X 相当の Actor 由来 PER 初期優先度(追加 forward なしの近似)に対応(PRD035, ADR0010〜0012)。
- Train Actor の定期 network snapshot(モデル同期)を追加(PRD036)。
- EvalEpisode の `clone_model` 設定を追加。既定値の解決は Agent が持つ(ADR0013)。
- Agent が Actor 生成時に EnvSpec の入出力互換性を検証するよう追加(PRD034)。
- Spatial Exploration の ENV lane 割り当てを反転し、TrainView に現在の方策に近い内容が映るよう修正。
- DQN の CPU モードで AMP 設定が不整合になる問題を修正。
- warmup 完了後は CanUpdate() で ReplayBuffer::Size() の呼び出しを省くよう修正(性能)。
- EpisodeEvalObserver で例外が起きてもエラー終了しない問題を修正。

ReplayBuffer:

- n-step の done 近傍でフレームスタックが崩れる問題を修正。Extractor の scalar read を高速化。
- PER 優先度を float から double へ変更。`per_initial_mass_ratio` 等が正しく測定されない問題を修正。
- generation 付き priority 更新の検証と atomicity を補強。重複更新の last-wins 契約を保証(ADR0011)。
- `per_clipped_count` を実際に clip された件数へ修正。PER の設定警告を Learner 単位で発行。
- completion 責務と sampleable 境界を分離(ADR0012)。

NN:

- ConvNeXt v1 の CNBlock を導入し、関連定義を `nn_cnx.txt` へ分離(PRD028)。
- `net.config_profile` によるブロック群パラメータ補間を追加。NN ブランチ毎のカスタマイズに対応(PRD029)。
- 汎用 LinearHead を追加し、LinearHeadFactory を標準化(PRD033)。
- IMPALA 風 Stem を追加(DropMerge baseline に採用)。
- Autocast(BF16)が正しく動作しない問題を修正。CPU モードでの AMP 不整合を修正。

Env:

- ImageCls: native batch 実装へ移行し、BatchEnvFactory の repository 登録に対応(PRD034, ADR0009)。
- ImageCls: BF16(AMP)学習に対応(PRD033)。
- ImageCls: Mixup / CutMix を導入(PRD030)。
- ImageCls: ProfiledValue による learning_rate の cosine アニーリング(PRD031)。
- ImageCls: FusedAdamW 対応、暫定の永続化(保存・復元)、メトリクスと PROFILE 範囲を拡充。
- ImageCls: Decode と Augmentation を同じスレッドプールで実行し性能向上。
- DropMerge: 直前行動観測(prev-action trio と DROP 列マーカー)を追加(PRD040)。
- DropMerge: NoLegal 裁定を blocked-persistence 化する `use_no_legal_adjudication` を追加(PRD039, ADR0014)。
- DropMerge: View に前回エピソードの終了理由を表示。NEET 問題診断用メトリクスを追加。
- LunarLander: 直前 action を観測に含める `obs_include_action` を追加(PRD032)。
- Env にインスタンス名を導入し、Env ログへ name prefix を自動付与(PRD037, PRD038)。

メトリクス・可観測性:

- VideoLogger にコーデック `auto` 選択と ffmpeg 異常の fail-fast 検出を追加(PRD025)。

Runner:

- パネルレイアウトを最適化し、AuiLayoutFrame を分離(ADR0016)。

Metrics Viewer:

- キャッシュを Kryo から SQLite へ移行。OOM 解消、LOD、viewport API(PRD041, ADR0015)。
- 取り込み / 照会経路の構造リファクタリング(挙動不変)(PRD042)。
- スクロールロックモードと Reset View ボタンを追加。
- 起動時のキャッシュチェックを高速化。LOD ページキャッシュの読み増幅と追記追従を改善。
- 起動時に Runs ディレクトリのパスをログ出力。
- 後から追加された tag が描画に反映されない問題を修正。
- Pan 後に Zoom モードへ強制的に戻る問題を修正。
- 最上部のグラフ領域が Scroll Lock ボタンと重なる問題を修正。
- 初期ロードエラーから Reload で復帰できない問題を修正。
- 取り込みスレッドが予期しない例外で無言停止する問題を修正。
- DB 書き込み失敗をソース読み取りエラーと区別し、更新障害情報を要求種別ごとに常設表示。
- 警告抑止と取り込み優先度の寿命規約を修正。イベントハンドラの直代入を addEventListener へ変更。

ツール:

- MLflow bridge に既存 Run の取り込みを追加。MLflow 3.14.0 に対応。
- ログを tail & grep する小ツールを追加。

ビルド・CI・リリース:

- GitHub Actions のリリースパッケージングを新設。tag push で実行用 zip と Doxygen zip を Release へ添付(PRD043, ADR0017, ADR0020)。
- 配布物に対応する第三者ライセンス表記を補完。Java 依存は THIRD-PARTY 一覧を自動生成。
- libtorch を 2.12.0(cu130)へ更新。GitHub Actions の node24 対応。CI 環境の NOMINMAX ビルドエラーを修正。
- Push 時の自動ビルドを無効化。
- Catch2 と libtorch の CATCH マクロ重複でテスト判定が行われず C4005 が出る問題を修正。
- テストアプリで例外ダイアログが表示され、自動テストが止まる問題を修正。
- bat の改行コード指定(CRLF)を追加。

ドキュメント・開発プロセス:

- 設計文書とユーザーガイド(`docs/design/` 0xx / 1xx / 2xx)の初版を追加。
- Metrics Viewer の実装仕様文書(210)を追加。
- AGENTS.md に commit ルール、Run 分析ルール、Fail-Fast 原則、クリーンブレーク方針、責任境界を追記。
- karpathy-guidelines 等の skill とコミットメッセージテンプレをリポジトリに追加。


## 0.2.0: (2026-07-01 リリース)

v2_work ブランチ(2026-03-29〜)をマージした V2 全面改修。Observation / State を TensorDict 化し、
Env・NN・ReplayBuffer・可視化を Dict 契約で再実装した。

非互換:

- Observation / State の受け渡しを TensorDict 化。V1 の Env・NN 定義・設定とは互換性なし(ADR0002)。
- `batch_size` を `num_envs` へ改名。
- `ANET_PROFILE_RANGE*` を廃止し、計測は `ANET_PROFILE_SCOPE*` / `ANET_PROFILE_FUNC` マクロへ移行。
- ElementwiseAddModule と `truncates` を削除。
- LibTorch を 2.9.1 から 2.11.0 へ更新。

全般:

- TensorDict の基本機能を拡張。TensorSpec を anet 名前空間へ移し、TensorSpecMap に統一。
- EnvSpec の整合チェックを追加。
- Config の Read で書式エラーが起きた場合に警告ログを出すよう修正。
- 設定の include で Root パス相対のファイル読み込みに対応。
- runs ディレクトリの変更機能(`app.runs_dir`)を追加。
- stdout.log / stderr.log の出力に対応。実効設定のダンプ `config_data.txt` を追加。
- `backend.deterministic_algorithms` を追加し、同 seed 再現性を確保(PRD015, ADR0006)。
- `backend.cuda_launch_blocking` を追加。
- Profile 全般をマクロベースへ移行(`ANET_PROFILE_SCOPE_NEXT`、`ANET_PROFILE_FUNC`)。
- Pause 切替時にメトリクスとログを Flush するよう修正。
- AP 終了時の例外対策とエラーログ強化。
- タイムスタンプ無しログに対応。Save 時の書き込みバイト数誤りとバイナリモード未指定を修正。

Agent・学習・評価:

- Ape-X 型の空間探索(Vectorized Exploration)を追加(PRD002)。
- Transformed Bellman Operator を導入(DefaultDQN 限定)(PRD008, ADR0001)。
- Dueling の V/A ストリームを body DAG の Branch で分岐できるよう修正(PRD009)。
- FusedAdamW(ATen `_fused_adamw_` 直接呼び出し)を導入(PRD011, ADR0003)。
- Sample+To プリフェッチで ReplayBuffer サンプリングと H2D を GPU 学習に overlap(PRD013, ADR0005)。
- PrefetchingReplayBuffer を decorator として分離し、再現性を修正(PRD014)。
- Device 転送の共通部品 `transfer.hpp` を追加し、prefetch 対象に Push を加えた(PRD020)。
- PER priority 更新の D2H 境界を Optimize 前後で分離(PRD021)。
- Eval メトリクス取得を強化し、EpisodeEndEvent を追加(PRD003)。
- `eval_panel.model_sync.mode` を追加。
- Learn の最小区間だけ train モードへ切り替える TrainingModeGuard を追加。
- PER 関連メトリクスを追加。
- 画像分類の Env と Agent(ImageCls)を追加。
- 不要な ReplayBuffer::Size() 呼び出しの削減と detach 不足の修正(性能)。

ReplayBuffer:

- Dict 対応・GPU 対応で再実装。
- PER サンプリングで SumTree を正しく使うよう修正。初期優先度が付与されない / 不正になる問題を修正。
- Truncated 時のダミーステップとインデックスの不整合、Next State 抽出のオフセット誤り、N-Step 用ダミーステップがサンプリングされる問題を修正。
- 最新ステップがサンプリングされた時に obs が壊れる問題と、Size() の結果が正しくない問題を修正。
- Stacking で過去方向がエピソード境界を越えた場合は、初回フレームの複製で動的パディングするよう修正。
- エピソード開始から StackFrame 分がサンプリングされない問題を修正。
- バッファ一巡時の Observation 不整合、ダミーステップ選択、負数によるクラッシュを修正。
- FrameStacking 有効時に不正な過去情報がサンプリングされる問題を修正(PRD007)。
- n_step 有効時に未初期化データがサンプリングされる問題を修正。
- stack_keys 指定時に RB と Agent で不整合が起きる問題を修正。
- Stacker 内のメモリ領域重複で落ちる問題を修正。DictFrameStacker::Stack を高速化。
- Push 経路の alloc 関連を最適化。

NN:

- NN を StateDict(Dict 入力)対応。NetworkConfig の Prefix と dummy_input の Device 指定に対応。
- SDPA / FlashAttention 化(CustomTransformerEncoderLayer の self-attention 経路)(PRD012, ADR0004)。
- ResBlock に PreActivation モードを追加。
- DropoutModule、StackMergeModule、HybridSpatialEmbedderModule(V1 互換)、ReshapeModule、GAP2D、MaxPool2d を追加。
- ResBlockConfig / TransformerConfig に Dropout / DropPath 設定を追加(PRD024, ADR0007)。
- Head レス構成に対応。Network::SoftCopyTo を最適化。
- auto_format=false 時に対象 Key 判定の論理が逆だった問題、余計次元の処理、NoGradGuard 漏れを修正。

Env:

- 各 Env を StateDict / ObservationDict 対応。連続系 Observation で Float 系以外のデータ型を許容。
- ImageCls: 画像分類 Env を追加。train 専用データ拡張(PRD022)、GAP2D head(PRD023)、ResNet18 相当設定を追加。
- ImageCls: エピソード境界が伝わらず ImageLog が出ない問題と、Conv2Panel が出ない問題を修正。
- DropMerge: NoLegalDrop 終端 `no_legal_drop_terminal` と `use_no_drop_timeout_gameover` を追加(PRD005)。
- DropMerge: Grid がずれる問題を修正。MaxRank メトリクスを強化。droppath_rate に対応。
- DropMerge: `use_settle_after_drop=false` で端 DROP が即接地判定になる問題と、ep_settle_steps メトリクスの計測誤りを修正。
- LunarLander: wind_stop_mode / contact_mode / landing_detection_mode を追加。地面(崖)に当たり判定がない場合がある問題を修正。
- ObservationNormalizer / FrameStacker を StateDict 対応。RunningStd で正常データ 1 件のとき統計が更新されない問題を修正。

メトリクス・可観測性:

- 可視化層(metrics_image / HeatMap / 状態スイープ)を TensorDict 対応で復旧し、Phase2 で高速化(PRD010)。
- Conv2d 可視化を実 forward トレース化し、Panel と Observation を実データから描画(PRD006)。
- Conv2dVisualizerConfig に layer_margin_y を追加。scale_factor を float 化し縮小指定に対応。
- NN の GraphViz 可視化(dot 出力)を追加(PRD004)。
- BatchActionInfo のメトリクス出力に対応。metrics.image を Dict 対応。
- メトリクスの warn ログを抑止。NN の可視化で落ちる場合がある問題を修正。

Runner:

- QValuePanel: Grid クリックでアクション実行、スクロールで HeatMap 追従。表示乱れを修正し高速化。
- 閉じるボタンを追加。パネルレイアウトを変更。TrainPanel の描画乱れを修正。
- anet-core-test のログが GUI に出る問題を修正。

Metrics Viewer:

- LOG スケール表示を追加(負値表示に対応)。
- Run 切替で選択済み tag が無い Run になっても選択状態を記憶するよう修正。
- mainArea ダブルクリックで RELOAD。AutoReload ON 時にボタン色を変更。AutoScale ボタンを非表示。
- metrics.jsonl のサイズ変化が無い場合はパースをスキップし、長期稼働時の CPU スパイクを低減。
- 追記差分の先頭 1 行を取りこぼす場合がある問題を修正。
- PNG 保存がエラーになる問題を修正。tag 一覧でスワイプが効かない問題を修正。
- ズームで Min / Max / Mean の順番がずれる問題を修正。CTRL+C 終了時にキャッシュを書き込む。

ツール:

- DropMerge の Optuna 対応。run-study の探索パラメータ指定、grid mode、summarize_study、harness.log(PRD016)。

ビルド・CI・リリース:

- Catch2 による単体テスト anet-core-test を追加。
- LibTorch を 2.11.0 へ更新。runner に CONFIGURE_DEPENDS を追加。コンパイラ互換性を向上。
- ブランチでの自動レビューと手動ビルドトリガを有効化。
- 改行コードを LF へ統一。TortoiseGit 向けの .tgitconfig を追加。

ドキュメント・開発プロセス:

- AGENTS.md を追加し、PRD / 実装計画(`docs/memo/`)と ADR(`docs/adr/`)の運用を開始。
- grill-me / implement-prd-with-docs 等の skill を追加。
- 性能測定(ProfileRange)と設定値の扱いに関する規約を追記。


## 0.1.0: (2026-06-26 リリース)

V1 系の初回タグ。コードは 2026-03-28 時点の V1 実装で、このタグでは概要資料
`docs/anet_overview_ja.pdf` を追加した。V1 の到達点:

- 強化学習基本クラス群、Rainbow Agent、QR-DQN、MuZero 試作(Categorical 分布 / PER / Reanalyze / Batched MCTS は除外)。
- 1D-Conv / 2D-Conv、TransformerEncoder、Reward Scaling、Observation Normalization。
- Env: CartPole、LunarLander、DropMerge、GridMaze。
- メトリクス: Metrics Viewer(Java)による scalar のリアルタイム表示、TensorBoard 連携(bridge スクリプト)。
- 可視化: ヒストグラム / ヒートマップの時系列動画出力(ffmpeg)、Conv2D 動画可視化、GraphVizObserver とエピソード録画。
- AP: CUDA 対応、マルチスレッドによる N 環境並列実行、設定による NN 定義とメトリクス定義。


## 将来バージョンの対応候補

対応するissueで状態・優先度・順序を管理するPRDから。新しいPRDは直下に保持し、過去の保管場所は移さない。番号の意味はdocs/memo/README.md。

従来から正式番号を持つ候補:

- TimeHistogram の fine-bin 保持と描画時集約。有効時に学習スループットが 1/3〜1/4 に落ちる問題の解消(PRD056)。
- 捕まらない fatal の可視化(worker catch・set_terminate・エラーログ write-through)。CUDA 非搭載機で無言終了する事象の対策(PRD071)。
- .cpp の内部リンケージ(static / 無名 namespace)と namespace インデント規約の確定、既存 39 ブロックの一括是正(PRD079)。
- EvalPanel のゲームを trace へ出す。鏡写し eval インスタンスを起動時に宣言し、dormant trace だけを束縛(PRD081, ADR0047)。

PRD131 で 9xx から振り直した候補:

- train / eval mode 整理後の network 排他(lock)の精査(PRD134)。
- background eval が参照する network snapshot の順序保証(PRD141)。
- eval 自己ベスト更新時の best checkpoint 自動保存(PRD136)。
- NN ブロック別の勾配メトリクス(PRD135)。
- エージェント選択 action の Observation 注入を env 非依存の共通部品にする(PRD142)。
- Run 全体の Serialize(Save / Load)。optimizer load で Run 側の設定変更が実効値に反映されない件を含む(PRD143)。
- Episode Forensics。特殊挙動エピソードの自動捕捉と事後調査(PRD144)。
- DefaultDQN の staged batch / replay schedule。batch size と replay ratio の段階的変更(PRD145)。
- target network 更新の step 軸正規化。`soft_update_tau` / `hard_update_interval` を replay ratio に追随させる(PRD148)。
- IQN 初回 Learner priority の quantile-pair 方式(PRD147)。
- CI/CD 改善。ビルド・テスト自動化、定点ジョブ、LLM 補助。
- 設定 root の所有整合。`app.run_name` 等の階層ずれを規約で棚卸し。
- `episode_start` を stack / n-step 境界とするかの契約裁定。shouldfail 2 件の解消。
- メトリクス目標値によるハイパラ自律調整。
- Notifier::AttachScoped の Detach 契約修復(wrapper ハンドルを返す)。
- Q 値統計による多環境ハイパラ健全性スクリーニング。
- 標準 Spectral Normalization と residual identity 開始を両立する α ゲート。
- Run キュー(計画スナップショット + exe 固定)。
- スカラー閾値越え率の汎用化(Observation 層 / Viewer 層)。
- 行カバレッジ計測(OpenCppCoverage)。

凍結中(再開条件付き):

- MetricsLogger の null-safe 静的ログ API。再開条件: 排水外の経路で Reset 後ログが実害になったら(PRD026)。
- Train Actor の adaptive policy synchronization。再開条件: adaptive が固定 profile を上回る根拠が得られたら。
- Agent Update Result View。再開条件: MixUp / CutMix の可視化要求が実運用で出たら。
- AsyncLoader の N-deep 化。再開条件: 1-deep prefetch が律速になる実測が出たら。
- C++ SqliteBackend による metrics DB 直接出力。再開条件: PRD041 の安定運用。
- NN 内部表現の 2D 埋め込み可視化(t-SNE / PCA)。再開条件: 需要が出たら。

その他のメモ由来(`docs/*残件.txt`、`atari.txt`):

- Optuna: Study / Trial のダンプ(CSV)対応。
- ImageCls: Food101 以外のデータセット、転移学習(一般 NN モデルの読み込み)、ブランチ別の学習制御(学習有無・LR)、CLS メトリクスと前処理 / Augment の強化。
- Atari: 探索で得た知見の DropMerge へのフィードバック。

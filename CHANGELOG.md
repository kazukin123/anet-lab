# anet-lab Change Log

注記: v0.1.0 以降の変更を記載する。それ以前(V1 系の開発履歴)は git log を参照。
1 行 = PRD・機能・不具合事象の 1 単位。末尾の (PRDnnn) は `docs/memo/` の PRD 番号、
(ADRnnnn) は `docs/adr/` の設計判断。ハイパラ調整・実験記録・Run 設定だけの更新は載せない
(`docs/experiments/` 側で追う)。

更新のタイミング: リリース前にまとめて更新する。拾う先は `git log v0.3.0..main` と、
`docs/memo/done/` へ移した PRD。未リリース節は main 8f683164(2026-09-25)時点。


## v0.3.0 以降の非互換変更

後方互換は既定の要件としない(AGENTS.md「クリーンブレーク方針」)。v0.3.0 の設定・成果物を
次版で使うときに読み替えが必要な変更をまとめる。

設定を書き換えないと起動できない変更:

- 共用の特徴境界を DQN の `learner.final_feature_branch_key`、ImageCls の `final_feature_branch_key` へ改名。過去 dump の再生時は旧 `learner.plasticity.feature_key` / `plasticity.feature_key` を読み替える(PRD093)。
- 設定体系をプロファイル(`@name`)と選択チェーン(`X.$ = A > B`)へ全面移行。旧 `*.baseline.*` の直書きは通らない(PRD059, ADR0030)。
- Run 設定の root を `train.*` から `run.*` へ改名(`run.seed`、`run.train.num_envs`、`run.eval.[tag]`)(PRD061)。
- eval のスケジュールを `run.eval_schedule.[tag].interval` へ分離。interval は必須で、0 が OFF(PRD052, ADR0027)。
- 評価スロットの方策は Actor カタログ `<Agent>.actor.[key]` へ移動。`run.eval.[tag].actor` で名前参照する(PRD061, ADR0038)。
- device 指定を `agent.device` / `env.device` / `run.eval_device` に統一。旧 `*_device_type` / `*_device_index` は無効(PRD070)。
- QR-DQN の `use_qr` / `num_quantiles` は `quantile_mode = none|qr|iqn` / `qr.num_quantiles` へ(PRD001, ADR0018)。
- Run は `apps/runner/workspaces/<ws>/runs` へ、Env 選択は workspace の `config/_main.txt` へ。`app.runs_dir` は指定禁止(PRD046, ADR0021)。
- Runner は引数省略時に workspace 選択ダイアログを出す。bat / CLI からは `run.$=run.@name` で Run プロファイルを選ぶ(PRD046, PRD072)。
- Metrics Viewer は workspace 単位で Runs を読む。Optuna 用の別起動 bat `22_metrics_viewer_java_optuna.bat` は廃止(PRD046)。
- frame stack 用の Reshape ブロック(`ReS4F8` 等)は不要になり、NN 配線に残っていると shape が合わない(PRD057)。
- ビルドに wxWidgets 3.3.3 が必要。DLL を差し替えると 3.3.1 でビルドした旧 exe は起動できない。

黙って挙動・値が変わる変更:

- 既定葉 `?=` を導入。`=` は個別葉としてチェーンの供給値に勝つため、共通ファイルに残した `=` は再確認する(PRD072, ADR0040, ADR0042)。
- eval は `eval_batch_size`(lane 数)×`eval_episodes`(本数)の N 本平均を 1 点として記録する。1 エピソード 1 点ではない(PRD060, ADR0034)。
- interval 系の発火が bucket-crossing 化され、perf メトリクスは時間重み EMA になった(PRD053, ADR0028)。
- EMA メトリクスに初期値のバイアス補正が入り、序盤の値が変わる(PRD045)。
- ReplayBuffer は ring 折り返しで stack 履歴を失った transition を sample 候補から外す(PRD050, ADR0024)。
- 観測履歴は Push 時にエピソード開始から始める。旧実装で混入していた前エピソードのフレームは出ない(PRD078, ADR0044)。
- Atari のメトリクスを `game_score` / `game_len` / `game_frames` / `hns57` / `hns49` に改名。
- 解析用メタデータ `metrics.scalar.defs` / `metrics.trace.defs` を Runner が出力。解析側は設定から再導出しない(PRD054, PRD069, ADR0029)。
- Run 終了時に background 評価セッションの完走を `app.drain_timeout_sec`(既定 3600 秒)まで待つ(PRD076, ADR0045)。


### PRD101 ProfiledValue のキー対応

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

時間変化は `type = linear` を明示し、旧decay長が0ならconstantとbare開始値へ移す。spatialの旧start/endは `spatial.epsilon.first/last`・`spatial.uqe.epsilon.first/last`・`spatial.uqe.tau.first/last` へ別に写す。共有baseのspatial epsilonは0.4/0.01。既存のbare `weight_decay`・`gamma`・Munchausen値・soft update τの行はそのまま定数になる。PER βは更新末尾に次の抽選要求用の値を評価し、適用時機を維持する。（PRD101）

## 0.4.0: (未リリース)

- TransformerEncoderの層平均attentionエントロピーを追加。`run.@attention` / `metrics.scalar.@attention`の購読時だけ更新前probeで測り、`48_agent_attention`へ公開する(PRD093)。
- 実際の更新から勾配ノイズスケールを測る15指標を追加。`metrics.scalar.@gradnoise` の購読で有効化し、BF16とfeature/readoutに対応(PRD092)。

v0.3.0 以降の main の変更。非互換変更は冒頭の節を参照。

全般:

- 設定体系をプロファイル / 選択チェーンモデルへ全面移行し、ConfigResolver による解決基盤を導入(PRD059, ADR0030)。
- 設定継承の既定葉 `?=` と個別葉 `=` を区別し、選択の最終値参照と Run プロファイルの優先段を導入(PRD072, ADR0040, ADR0042)。
- device 指定を 3 キーへ統一し既定 `auto` を導入。CUDA 非搭載機では CPU で起動する(PRD070)。
- Run 設定の root を `train.*` から `run.*` へ改名(PRD061)。
- 設定の bool 値が txt 保存時に true/false ではなく 1/0 になる問題を修正。
- worker スレッドの停止通知の取りこぼしでハングする場合がある問題を修正。
- メモリ確保失敗時に要求サイズを含む診断ログを出すよう強化。
- Tracy 起動時のシンボル一括ロードを止め、PDB ファイルのロックを解放。

Agent・学習・評価:

- IQN(Implicit Quantile Networks)に対応。`quantile_mode`、bind `*` 演算子、IQNHead / IQNLearner(PRD001, ADR0018, ADR0019)。
- IQN の tau 配置方式を 5 方式(stratified / systematic / antithetic を含む)へ拡張(PRD044)。
- Munchausen RL(M-DQN / M-QR / M-IQN)に対応(PRD067, ADR0035, ADR0036)。
- 評価セッション契約を導入。L 並列 / N エピソード集計で 1 点を記録する(PRD060, ADR0034)。
- eval 定義と定期スケジュールを `run.eval.[tag]` / `run.eval_schedule.[tag]` に分離(PRD052, ADR0027)。
- Actor 設定カタログを導入し、評価スロットごとに方策・network・clone を名前で選べるようにした(PRD061, ADR0038)。
- 学習を完全停止した評価専用 Run 向けに `learner.enabled` を追加。
- Run 終了時に background 評価セッションの終了を待つ(または破棄する)排水を追加(PRD076, ADR0045)。
- DefaultDQN の stack 次元をネイティブ軸として扱い、Reshape ハックを廃止(PRD057)。

ReplayBuffer:

- ring 折り返しで stack 履歴を失った transition を sample 候補から除外。ready / sampleable を分離(PRD050, ADR0024)。
- 観測履歴と bootstrap stack の整合性が壊れる問題を修正。履歴は Push 時にエピソード開始から始める(PRD078, ADR0044)。
- sampleable index 列の確保を持続バッファ化(PRD077)。
- RB 履歴整合性テストを追加。`episode_start` 境界の既知失敗は shouldfail 化し、契約裁定を起票。

NN:

- Spectral Normalization(`weight_norm.mode`)に対応。自前実装で buffer 意味論を定義(PRD065, ADR0032)。

Env:

- AtariEnv を追加。ALE を C++ で直結し、前処理は自前、オプショナルビルド(PRD051, ADR0025, ADR0026)。
- Atari: reset / episodic_life を標準実装に準拠させ、`game_*` メトリクスへ改名し HNS(hns57 / hns49)を追加。
- Atari: スコア相転移計測用メトリクス(`game_score.ge.[N]`)とゲーム完了ログを追加。
- Atari: ゲーム固有の進行(面クリア・ボス撃破・ボス命中)を RAM から番号付き汎用キーで取るメトリクスを追加(PRD082, ADR0046)。
- Atari: Atari-5 / Atari-100k のバッチ定義を追加。
- Atari: sound 有効時に SDL_AUDIODRIVER を自動設定し、WASAPI 無音を回避。
- Atari: AtariView でテキスト情報が見切れる問題を修正。
- Atari: 実効値が意図と違っていた `noop_max` / `adam_eps` を修正。

メトリクス・可観測性:

- メトリクス trace チャネル(1 件 = 1 レコード)と SessionEnd イベントを追加(PRD069, ADR0037)。
- メトリクス定義に購読先・eval 条件・clip 指定を追加(PRD069)。
- Runner が解析用メタデータ `metrics.defs` を出力(PRD054, ADR0029)。
- EMA の初期値バイアス補正を導入(PRD045)。
- interval 発火を bucket-crossing 化し、perf メトリクスを時間重み EMA 化(PRD053, ADR0028)。
- 可塑性メトリクス(srank / dormant unit 率)と feature / readout 分割の weight norm を追加(PRD062, PRD063, ADR0031)。
- policy churn メトリクスを追加(PRD066, ADR0033)。
- replay 抽選履歴別の TD・損失メトリクス(replay_fit)を追加(PRD073, ADR0039)。
- IQN 探索 P0 診断メトリクスと分位 tail 診断メトリクスを追加(PRD047, PRD048)。
- 共通のエピソード長指標(episode_steps)と評価セッション・待機ログを追加(PRD075, ADR0043)。

Runner:

- workspace 機構を導入。Run を workspace フォルダで分類し、Runner / Viewer / Optuna を統合(PRD046, ADR0021, ADR0022)。
- ツールバー 4 本(対象軸)、View FPS メニュー、checkpoint 保存を追加(PRD055)。
- AlwaysOnTop に対応。FPS メニューに 15 を追加。
- batchrun 向けにエラーダイアログ抑止・終了コード・bat 側の致命エラー検出を追加(PRD068)。
- Eval pane 表示切替時の window 自動リサイズを廃止。右クリックの Eval resume 時も EvalPanel を表示。
- pane 見出しと workspace 履歴の視認性を改善。
- EvalPanel の FPS を 0 にするとエラーになる問題を修正。
- wxWidgets 3.3.3 移行の影響で TrainPanel が真っ黒になる問題を修正。

Metrics Viewer:

- workspace 切替に対応(PRD046)。
- 放棄された query が permit を握り続け、素早い操作で `Update failed` になる問題を修正(サーバ側 supersede)(PRD049, ADR0023)。
- 不変 Run の定常取り込み I/O を抑制(PRD064)。
- 選択中 Run の色を分離。Recolor ボタンと AutoRecolor を追加(PRD074, ADR0041)。
- グラフのホバー表示を自前描画に置き換え。終了済み系列に別 step の値が混ざる問題を修正。
- タグ別 percentile Y 軸範囲と表示設定の永続化を追加。
- Runs に SelectedOnly トグルを追加。
- スクリーンショットモードでスワイプによる縦スクロールに対応。スクロール位置が飛ぶ問題を修正。
- スマホでグラフをタップしてもポイント情報が出ない問題を修正。
- 統計表示に隠れて Plotly modebar が操作できない問題を修正。

ツール:

- AI 向け Run 解析 CLI `inspect_run.py` を追加。subcommand 分割と trace-csv(PRD054)。
- Optuna ハーネスを workspace へ統合(PRD046)。
- workspace metrics の一括 gzip 圧縮(`70_compress_workspace_metrics.bat`)を追加。
- Atari-5 の 5 本の Run を 1 つのベンチマークとして集計するスクリプトを追加。
- Atari-5 専用 bat を `11_batch_run.bat` / `12_batch_run.bat` へ統合。
- MLflow bridge が config のコメント行で落ちる問題を修正。
- optuna ハーネスの実設定との不整合を修正。

ビルド・CI・リリース:

- wxWidgets を 3.3.3 へ更新。find_package を Config モードへ一本化し、DLL コピー処理を追加。
- runtime DLL コピーのパスを CMAKE_SOURCE_DIR 基準に修正。
- ENV 別 workspace 枠(`<Env>-01/config/_main.txt`)をリリース zip へ同梱。
- docs/ を GitHub Pages へ配信するワークフローを追加。

ドキュメント・開発プロセス:

- 設計文書の英語版(`*.en.md`)を追加。
- アーキテクチャ Atlas(5 図と索引)と生成 skill `anet-archify-atlas` を追加。
- AI ハーネス動線マップと生成 skill `anet-harness-map` を追加(PRD080)。
- 開発支援 skill 群(anet-stats / audit / housekeeping / survey-queue / translate-docs、prepare-commit)を追加。
- grill skill に goal anchor と簡素化パスを追加(PRD058)。
- PRD を状態別フォルダ(done / frozen / dropped)へ振り分け、バックログを Phase 別に採番。
- AGENTS.md にクリーンブレーク方針、責任境界、Fail-Fast 原則、GetScalar 規約、実験記録と実効 config の保存ルール、Run 命名規約を追記。
- Atari / DropMerge / LunarLander の実験記録と、サーベイ(SimPLe、報酬スケールと割引率、target 更新速度、BTR)を追加。


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

`docs/memo/` 直下(未対応 / 対応中)と `frozen/` の PRD から。番号と状態の意味は `docs/memo/README.md`。

正式採番済み(着手中・実装待ち):

- TimeHistogram の fine-bin 保持と描画時集約。有効時に学習スループットが 1/3〜1/4 に落ちる問題の解消(PRD056)。
- 捕まらない fatal の可視化(worker catch・set_terminate・エラーログ write-through)。CUDA 非搭載機で無言終了する事象の対策(PRD071)。
- .cpp の内部リンケージ(static / 無名 namespace)と namespace インデント規約の確定、既存 39 ブロックの一括是正(PRD079)。
- EvalPanel のゲームを trace へ出す。鏡写し eval インスタンスを起動時に宣言し、dormant trace だけを束縛(PRD081, ADR0047)。

バックログ(Phase 別。下 1 桁が推奨着手順):

- train / eval mode 整理後の network 排他(lock)の精査(PRD910)。
- background eval が参照する network snapshot の順序保証(PRD912)。
- eval 自己ベスト更新時の best checkpoint 自動保存(PRD913)。
- NN ブロック別の勾配メトリクス(PRD920)。
- エージェント選択 action の Observation 注入を env 非依存の共通部品にする(PRD921)。
- Run 全体の Serialize(Save / Load)。optimizer load で Run 側の設定変更が実効値に反映されない件を含む(PRD930)。
- Episode Forensics。特殊挙動エピソードの自動捕捉と事後調査(PRD932)。
- DefaultDQN の staged batch / replay schedule。batch size と replay ratio の段階的変更(PRD933)。
- target network 更新の step 軸正規化。`soft_update_tau` / `hard_update_interval` を replay ratio に追随させる(PRD940)。
- IQN 初回 Learner priority の quantile-pair 方式(PRD941)。

バックログ(Phase 未割当 999):

- CI/CD 改善。ビルド・テスト自動化、定点ジョブ、LLM 補助。
- 設定 root の所有整合。`app.run_name` 等の階層ずれを規約で棚卸し。
- `episode_start` を stack / n-step 境界とするかの契約裁定。shouldfail 2 件の解消。
- eval スケジュールの初回発火(learn_step=0)の抑止。
- メトリクス目標値によるハイパラ自律調整。
- NoisyNet(Noisy Networks for Exploration)。
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

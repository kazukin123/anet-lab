---
title: Run実行ユーザーガイド
page_kind: サマリ
material: "[[docs/design/020_user_guide_run.jp.md|020_user_guide_run.jp]]"
material_kind: 設計文書
concepts:
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
updated: 2026-11-26
---
# Run実行ユーザーガイド

## 元の材料

[Run実行ユーザーガイド](../../../design/020_user_guide_run.jp.md) blob:c4ce6ffddd9a60557b04e2d00b7c1e0460cfd12e

## 要点

- 起動前に workspace、Env、Agent、実行モード、評価条件を確認し、目的に合う設定を選ぶ。
- 設定は既定キー、プリセットの合成、個別指定で決まる。解決後の値を Run の設定記録で確認できる。
- 評価スロットごとに Actor と方策を選べる。同じ重みでも選択方策が異なれば別の評価になる。
- Runner の画面では学習を一時停止・再開し、評価を進めたり Evaluation Trace から遷移と行動の経緯を調べたりできる。
- 停止条件、checkpoint の保存・読込、メトリクスの出力先を確認すると、実行後に同じ Run の結果をたどれる。

## 内容

### 2. 実行前の準備

- device指定は`auto`、`cpu`、`cuda`、`cuda:N`を受け付ける。`auto`はCUDAが利用可能ならcurrent CUDA device、そうでなければCPUを選ぶ。指定値は`config/config_data.txt`、実効deviceは`json/agent.json`、`json/env.json`、`json/run.json`の`effective_device`/`effective_eval_device`に記録する。EnvをCPU、AgentとEvalをCUDAに置く構成では、device転送を含めて性能を判断する。
- 探索を定数にする場合は `policy.epsilon.type = constant` とbare値 `policy.epsilon = 0.1` を指定する。時間変化は `type = linear|cosine|cosine_restart|exponential|phased` と対応する欄で指定する。
- 外的価値と内的価値を別々に学ぶ場合、RND の target と predictor を設定し、`run.@rnd` を選んだ DefaultDQN に `DefaultDQNAgent.intrinsic.value_mode = decomposed` を指定する。この構成と skip は併用しない。

### 3. 設定ファイルの書き方

- 同じ選択キー `X.$` を再指定すると、以前のチェーン全体が入れ替わる。
- 同じキーを複数回書いたら、同じ演算子同士は後勝ちである。`$include`や追加ファイルをまたいでも同じ。`=`と`?=`が混在したら順序に関係なく`=`が勝ち、後から書いた既定キーが個別指定を取り消すことはない。
- 短い`@name`は、書いた場所(定義元)の名前空間で解決する。`Env.$ = @a`は`Env.@a`、`Env.@a : $ = @b`は`Env.@b`を指す。`Other.$ = Env.@a`のように別の場所から使っても、`@a`の中の`@b`は`Env.@b`のままである。
- 各項は自分の最終値を持ち込み、左から右へ重ねる。右の項にあるキーは右が勝ち、右に無いキーは左から残る(差分合成)。
- `run.$`は通常の選択より先に展開される。項は左から右へ後勝ちで、同じキーを複数の項が書けば右の項の値になる。
- `config/config_data.txt`は解決後の実効設定である。上書き層(`A1`〜`A3`、`E1`、`M1`/`M2`、`P1`)や選択元の通常prefix(`app.online`等)の定義も同居するが、`@`プロファイルと`.$`は含まれない。どのプロファイルが選ばれたかはこのfileからは分からない。

### 4. Runを開始する

- ウィンドウが表示されたら、Logsで起動のログを確認する。初期化中もウィンドウを移動・サイズ変更できる。初期化中はステータスバーに`Initializing run...`が表示され、終わると`Training started`（`app.train_auto_start=false`では`Run initialized`）に変わる。初期化に失敗した場合は`Run initialization failed`になる。
- `app.train_exit_step`、`app.exp_exit_step`は上限到達時にRunを終了する。`app.train_pause_step`、`app.exp_pause_step`は一度だけ自動pauseする。
- batch実験では、`app.$=app.batchrun`で低FPS表示、`exp_exit_step`、`app.show_error_dialog=false`をまとめて選ぶbatchrun構成が用意されている。人が操作するonline構成は`app.$=app.online`で`app.show_error_dialog=true`を選ぶ。これらはTrain / Evalの`RunMode`とは別概念である。

### 5. AP画面

- 画面上端には対象別の4本のツールバーがあり、右の列の上まで窓の全幅に並ぶ。各バーはgripperをドラッグしてdock、float、再dockでき、`View > Reset Layout`で上端1行の既定位置へ戻る。floatさせたバーのwindow titleにはバー名(`Run Control`、`Steps`、`Run Operations`、`Panels`)が出る。
- `Panel表示`: `Logs`、`Train View`、`Eval View`、`Q-Values`、`Trace`の表示を切り替える。対応する`View` menuとpaneの状態に同期する。
- `Train View`: Train RunnerのEnv固有Viewを見るときは、Panelsの`Train View`または`View > Train View`で表示する。隠すときも同じ項目かpaneの✕を使う。隠している間も学習は続く。

### 6. 操作方法

- Run制御ツールバーの`Train`を選ぶとTrainをpause/resumeする。Train/Eval View上の左クリックと`Shift`も同じ操作として併存する。`app.train_auto_start=false`で起動した場合も同じ操作で開始できる。pause時はmetrics、stdout/stderr、text logを明示的にflushする。Trainが停止して再開不能になった後はツールが無効になる。
- Action数はEnvごとに異なる。QValue paneまたはEnvのActionSpecで有効範囲を確認する。離散Actionの範囲外の番号を指定すると、ステータス欄に指定値と有効範囲が表示され、Evalは進まない。選択中のTrace履歴もそのまま維持される。
- 初期化中に閉じたいときもCloseを1回選び、ログの終了要求を確認してウィンドウが閉じるまで待つ。押し直す必要はない。初期化中にログオフ・シャットダウンを要求した場合も、終了するまで待つ。Windowsに終了を妨げているアプリとして表示された場合、強制終了を選ぶとRunログの末尾が欠ける場合がある。

### 7. 成果物

- 設定ファイルを直接指定して起動した場合、相対の `app.runs_dir` は `data/` を基準に解決します。既定の出力先は `data/runs/` です。
- `metrics.jsonl` — scalar、JSON metadata、動画metadataを追記する主メトリクス。
- `config/config_data.txt` — 解決後の実効設定。deviceの`auto`など利用者の指定値を保持し、`@`プロファイルと`.$`は含まない(3.8)。
- `json/run.json`、`json/env.json`、`json/agent.json` — 評価・Env・Agentの実効deviceを`effective_eval_device` / `effective_device`に記録。
- `agent_close.anet` — 正常なwindow close時に保存されるAgent checkpoint。`app.save_agent_on_close=false`では作られない。
- 比較や再現には、Run ディレクトリに記録した `config/config_data.txt` を使う。

### 8. よくある確認事項

- 起動直後にTrainが進まない: `app.train_auto_start`を確認し、左クリックまたは`Shift`でresumeする。
- Evalが進まない: `Evaluation View`を表示し、`app.eval_panel.auto_start`または`Space`を確認する。
- toolbarのcheck状態が操作と合わない: 最大200ms待って実状態への同期を確認する。配置が崩れた場合は`View > Reset Layout`を使う。
- Train Viewだけ更新されない: `View > Train View FPS`が`0 (Off)`になっていないか確認する。
- Saveに失敗する: error log（online構成ではダイアログも表示）の対象pathと失敗段階を確認する。権限、空き容量、file lockを解消するか別pathを選んで再実行する。Runは継続しているが、失敗した出力fileは不完全な可能性がある。
- Run folderが開かない: error log（online構成ではダイアログも表示）の対象pathとOS側のfolder関連付けを確認する。失敗後もRunは継続する。

## 主要な概念

- [RunnerApp](../../concepts/apps/110_runner_app.md)
- [316 設定管理](../../concepts/framework/316_config.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)

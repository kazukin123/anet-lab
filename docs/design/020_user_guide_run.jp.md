# Run実行ユーザーガイド

> 主たる観点: 行程単位（設定、起動、操作、終了、成果物確認）

## 1. はじめに

### 1.1 目的

本書は、ANET RL Runnerで1つのRunを設定し、学習・評価画面を操作し、終了後の成果物を確認するまでの基本手順を説明する。

### 1.2 対象読者

- 既存のEnvとAgentを設定してRunを実行する利用者
- Train、Eval、可視化パネルの基本操作を知りたい利用者
- Run directoryに保存されるログ、メトリクス、checkpointを確認したい利用者

### 1.3 記載範囲

現行の`AnetRLRunner`、`apps/runner/config`、標準GUI操作、Run成果物を扱う。
新しいEnv、Agent、Observerの実装方法は対象外とし、各設計文書を参照する。

> [!NOTE]
> 本書の実行経路はWindows x64とNVIDIA CUDAを使う構成で検証済みである。CPU-only、Linux、macOS、他GPU backendは未検証であり、同じ操作結果を保証しない。

## 2. 実行前の準備

### 2.1 実行要件

本書は、`apps/runner/bin/Release/AnetRLRunner.exe`がbuild済みで、runnerが必要とするDLLとconfigが配置済みであることを前提とする。検証済み構成ではWindows x64、NVIDIA driver/CUDA runtime、CUDA対応libtorchを使用する。

開発環境の準備、依存関係、CMake preset、build手順は[開発環境](040_development_environment.jp.md)を参照する。

### 2.2 設定ファイルの選択

引数を省略したrunnerはworkspace選択ダイアログを表示し、選択したworkspaceの`config/_main.txt`からEnvを選ぶ。共通の`apps/runner/config/_main.txt`はAgent、Network、metric等だけを読み、Env選択はworkspace側へ分離される。新規workspace、および既存ディレクトリの初回選択時に不足しているworkspace configは、`apps/runner/config/_workspace_template.txt`を`config/_main.txt`へコピーして作成される。

```text
# apps/runner/workspaces/<workspace>/config/_main.txt
#$include <LunarLander.txt>
$include <DropMerge.txt>
```

1回のRunでは、意図したEnv設定だけを有効にする。各Env設定内の`app.$`、`metrics.scalar.$`などは、`>`の左から右へ設定群を重ねる。コマンドラインの`key=value`はmerge後にも再適用されるため、最終overrideとして扱われる。

### 2.3 最初に確認する設定

| キー | 役割 |
|---|---|
| `app.run_name` | Run名。`{t}`は起動時刻へ展開される |
| `app.runs_dir` | workspaceモードでは`<workspace>/runs`へRunnerが導出する。設定やCLIからの変更は禁止 |
| `app.train_auto_start` | `true`ならGUI初期化後に学習を開始する |
| `app.eval_panel.auto_start` | 手動EvalPanelを起動直後から動かすか |
| `train.seed` | Runの基準seed |
| `train.num_envs` | Train用BatchEnvのlane数 |
| `agent.class_id` | 使用するAgent実装 |
| `agent.device_type` / `agent.device_index` | AgentのCPU/CUDA device |
| `env.worker_type` / `env.worker_threads` | Env batchの実行方式とworker数 |
| `train.eval_device_type` / `train.eval_device_index` | configured evalのdevice |
| `backend.deterministic_algorithms` | 決定論的algorithmを要求するか |

`agent.device_type=1`はCUDA、`0`はCPUである。EnvをCPU、AgentとEvalをCUDAに置く構成では、device転送を含めて性能を判断する。

## 3. Runを開始する

### 3.1 標準起動

`apps/runner`から次を実行する。

```powershell
10_run.bat
```

初回は`_default`が新規名として入力済みの選択ダイアログが開く。履歴、`workspaces/`直下の全ディレクトリ一覧、任意パス参照、新規名から選択できる。過去Runだけを移動したフォルダなど`config/_main.txt`が無い既存ディレクトリも一覧に出て、選択時に不足configだけが補完される。新規名は入力中に検証され、不正理由が入力欄の下へ表示されている間はOKを選択できない。`--workspace dm_long`で相対workspaceを直接指定し、`--select-workspace`でスキップ設定に関係なくダイアログを表示できる。相対パスは`apps/runner/workspaces/`基準、絶対パスも使用できる。入力の外側空白は除去され、`#`、`//`、末尾`;`、UNC pathは拒否される。

または、リポジトリルートから実行ファイルとmain configを明示する。

```powershell
apps\runner\bin\Release\AnetRLRunner.exe `
  --config apps\runner\config\_main.txt `
  app.run_name=run_{t}_trial `
  train.seed=12345
```

`--config`はworkspace、履歴、`last_workspace.txt`を一切参照しない完全自己記述モードである。`--workspace`または`--select-workspace`との併用は起動エラーになる。

起動時は概ね次の順に初期化される。

1. workspaceを確定し、共通main config、導出`app.runs_dir`、workspace config、コマンドラインoverrideの順で解決する。
2. Run directory、`metrics.jsonl`、標準出力ログを準備する。
3. libtorch backendと登録済みEnvを初期化する。
4. `RunManager`がEnv、Agent、Train Runner、configured Evalを構築する。
5. Train、Eval、QValue、Logの各パネルを接続する。
6. `RunnerThread`を開始する。`app.train_auto_start=false`の場合はpause状態で待機する。

起動に失敗した場合は、画面のエラーダイアログだけでなく、対象Runの`stderr.log`と`<run_name>.log`も確認する。

### 3.2 自動停止・自動pause

`app.train_exit_step`、`app.exp_exit_step`は上限到達時にRunを終了する。`app.train_pause_step`、`app.exp_pause_step`は一度だけ自動pauseする。
batch実験では、`app.$=app.batchrun`で低FPS表示と`exp_exit_step`をまとめて選ぶ構成が用意されている。

## 4. AP画面

### 4.1 基本構成

Runner画面はwxAUIのpaneで構成される。

画面上端には対象別の4本のツールバーがある。各バーはgripperをドラッグしてdock、float、再dockでき、`View > Reset Layout`で上端1行の既定位置へ戻る。floatさせたバーのwindow titleにはバー名(`Run Control`、`Steps`、`Run Operations`、`Panels`)が出る。

- `Run制御`: Trainのpause/resumeと、separatorを挟んでEvalのpause/resume、Evalの1 step実行を提供する。走行中のtoolは押下表示になり、iconは次の操作を表す一時停止記号へ変わる。停止中は再生記号へ戻る。非表示のEvalをresumeするか1 step実行すると`Evaluation View`も表示する。
- `Step表示`: Trainの`exp_step`と`train_step`を表示する。
- `Run操作`: 任意pathへのcheckpoint保存とRun folder表示を提供する。
- `Panel表示`: `Logs`、`Eval View`、`Q-Values`の表示を切り替える。対応する`View` menuとpaneの状態に同期する。

- `Train View`: Train Runnerから受け取ったEnv固有Viewを表示する。
- `Evaluation View`: Trainとは別のEval Envを手動またはtimerで進める。初期状態では非表示である。
- `Evaluation Q-Values`: Eval Actorの出力を表示し、Actionを手動指定できる。
- `Logs`: 実行ログを表示し、Error、Warn、Info、Verboseを切り替える。
- `HeatMap` / `Conv2d`: `View`メニューから追加する補助pane。

![DropMergeのTrain ViewとEvaluation View](assets/020_runner_dropmerge_train_eval.png)

![LunarLanderの実行画面と可視化pane](assets/020_runner_lunarlander_visualization.png)

paneを閉じたり初期化前のViewを表示した場合は、次のように描画対象が少ない状態になることがある。`View > Reset Layout`で既定配置へ戻せる。

![描画対象がない状態のRunner画面](assets/020_runner_empty_view.png)

## 5. 操作方法

### 5.1 学習のpauseと再開

Run制御ツールバーの`Train`を選ぶとTrainをpause/resumeする。Train/Eval View上の左クリックと`Shift`も同じ操作として併存する。`app.train_auto_start=false`で起動した場合も同じ操作で開始できる。pause時はmetrics、stdout/stderr、text logを明示的にflushする。Trainが停止して再開不能になった後はツールが無効になる。

### 5.2 評価と画面操作

| 操作 | 動作 |
|---|---|
| Run制御ツールバーの`Eval` | EvalPanelをpause/resumeする。resume時にpaneが非表示なら表示する |
| Run制御ツールバーの`Step` | Evalを1 step自動実行する。paneが非表示なら表示する |
| 右クリック、または`Space` | EvalPanelをpause/resumeする |
| `Ctrl` | Evalを1 step自動実行する |
| 矢印キー | LunarLander向けに`0`から`3`のActionを指定してEvalを1 step進める |
| テンキー`0`から`9` | 同じ番号のActionを指定してEvalを1 step進める |
| `View > Evaluation View` | Eval paneの表示を切り替える |
| `View > Evaluation QValue View` | Q値paneの表示を切り替える |
| `View > Log Level` | GUIへ表示するログlevelを切り替える |
| `View > Reset Layout` | pane配置とframe sizeを既定へ戻す |
| Run操作ツールバーの`Save Checkpoint` | Trainが走行中ならまずpauseし、Run directoryと`agent_<exp_step>.anet`を既定に、任意pathへAgentを保存する |
| Run操作ツールバーの`Open Run Folder` | 現在のRun directoryをExplorerで開く |

Action数はEnvごとに異なる。範囲外Actionを前提にせず、QValue paneまたはEnvのActionSpecを確認する。

`app.eval_panel.model_sync.mode`は手動EvalPanelがTrain modelを参照する方法を決める。`shared`はTrain modelを共有し、`frame`、`time`、`episode`は対応するintervalでclone modelを同期する。clone modelを使うmodeではEvalのresume時にも同期する。表示中のEvalが常にTrainの最新parameterと一致するとは限らないため、比較時はmodeとintervalを記録する。

### 5.3 表示FPSと進行状況

`View > Train View FPS`はTrain Viewの描画頻度だけを変更する。`0 (Off)`では描画timerを止めるが、学習は継続する。`View > Eval View FPS`はEvalPanelのtimer周期を変更するため、描画だけでなくEvalの進行速度も変わる。どちらの`Config (N)`も起動時config値へ戻す項目である。これらは実行時UI操作であり、選択結果をRunのconfig dumpへ書き戻さない。

Step表示ツールバーは`exp`と`train`のstep数を別々のread-only text欄へ表示する。値は選択してコピーでき、exp/train間は標準separatorで区切られる。status bar右側は`exp <N> steps/s    train <N> steps/s`と経過時間を表示する。SPSのEMAがまだ初期化されていない間は`-`、最初のTrain snapshotが無い間は両step欄が`-`、経過時間が`--:--:--`になる。経過時間はpause中もwall-clockとして進む。

### 5.4 停止、保存、checkpointからの再開

WindowのCloseまたは`File > Exit`でRunを停止する。終了処理はTrain停止、`agent_close.anet`保存、Run出力のflush、GUI破棄の順に進む。保存中にprocessを強制終了するとcheckpoint、metrics、動画の末尾が不完全になる可能性があるため、windowが閉じるまで待つ。

`Save Checkpoint`は押下時にTrainが走行中なら先にpauseする。これはdialog操作中にstepが進み、既定ファイル名と保存内容がずれるのを防ぐためで、保存やcancelの後もTrainは自動再開しない。再開はRun制御ツールバーの`Train`か`Shift`で行う。保存処理自体はTrain走行中でも安全である。`DefaultDQNAgent`はserialization全体をAgentのshared lockで保護し、Learner更新と排他する。保存先の権限、空き容量、file lockなどで失敗した場合は対象pathと理由をErrorDialogへ表示し、Runは継続する。失敗したfileは不完全な可能性があるが自動削除されないため、内容を確認してから処理する。有効なpathを選べば再度Saveできる。

close時の`agent_close.anet`保存に失敗した場合もErrorDialogで通知し、その後のlog shutdownとGUI cleanupを続行してwindowを閉じる。この場合、`agent_close.anet`は有効なcheckpointとは限らない。AgentがSaveを実装していない場合は0 byteのfileが残り、対象path付きのWARNが出る。保存できたcheckpointもAgent、Network、archive contractが一致することを確認してから再開に使う。

checkpointから再開する場合は、新しいRunの互換設定にAgent固有の`auto_load_file`を指定する。現行例は`DefaultDQNAgent.auto_load_file`のaliasである`R.auto_load_file`、または`ImageClsAgent.auto_load_file`である。Network構成やarchive contractが異なるcheckpointは読み込めない。保存対象はAgentごとに異なり、現行DQN系ではReplayBufferの内容やsampling状態を復元しない。再開後は新しいRun directoryとstep系列を持つため、旧Runの`metrics.jsonl`へ追記する操作ではない。DQN系の保存対象は[DQN系Agent](200_dqn_agents.jp.md)を参照する。

## 6. 成果物

workspaceが`dm_long`、`app.run_name=run_{t}`の場合、成果物は`apps/runner/workspaces/dm_long/runs/<run_name>/`へ保存される。絶対パスworkspaceでも同様に、そのworkspaceの`runs/`配下へ保存される。

| 成果物 | 内容 |
|---|---|
| `metrics.jsonl` | scalar、JSON metadata、動画metadataを追記する主メトリクス |
| `config/config_data.txt` | includeとoverride解決後の全ConfigData |
| `config/*.txt`、`json/*.json` | コンポーネント別の注入済み設定・metadata dump。Envは`config/env.<Env name>.txt` |
| `<run_name>.log` | timestampとlevelを含むrunner text log |
| `stdout.log` / `stderr.log` | process標準出力・標準エラー |
| `agent_close.anet` | 正常なwindow close時に保存されるAgent checkpoint |
| `videos/*.mkv` | image系Observerが生成した動画 |
| `images/<tag>/*.png` | `app.metrics_logger.use_png_dump=true`時の個別frame |
| `dot/**/*.dot` | GraphViz Observerの出力 |

比較や再現では、手元の編集前ConfigではなくRun directory内の`config/config_data.txt`を正とする。グラフ分析は[分析ユーザーガイド](030_user_guide_analysis.jp.md)を参照する。

## 7. よくある確認事項

- 起動直後にTrainが進まない: `app.train_auto_start`を確認し、左クリックまたは`Shift`でresumeする。
- Evalが進まない: `Evaluation View`を表示し、`app.eval_panel.auto_start`または`Space`を確認する。
- toolbarのcheck状態が操作と合わない: 最大200ms待って実状態への同期を確認する。配置が崩れた場合は`View > Reset Layout`を使う。
- Train Viewだけ更新されない: `View > Train View FPS`が`0 (Off)`になっていないか確認する。
- Saveに失敗する: ErrorDialogの対象pathと失敗段階を確認する。権限、空き容量、file lockを解消するか別pathを選んで再実行する。Runは継続しているが、失敗した出力fileは不完全な可能性がある。
- Save結果が0 byteになる: `<run_name>.log`の対象path付きWARNを確認する。利用中AgentがSaveを実装していない可能性がある。
- Run folderが開かない: ErrorDialogに表示される対象pathとOS側のfolder関連付けを確認する。起動失敗後もRunは継続する。
- CUDA初期化に失敗する: libtorch/CUDA/driverの組み合わせ、`agent.device_type`、eval deviceを確認する。
- 期待したEnvでない: 選択workspaceの`config/_main.txt`で有効なEnv includeと、Run内`config/config_data.txt`を確認する。
- workspaceを選び直したい: `--select-workspace`で起動する。履歴は`GetAppDataDir()/history.txt`、ダイアログ選好は`prefs.txt`を削除すると個別にリセットできる。
- Viewが空: Log paneのEnv class ID、View factory、初期化errorを確認し、`Reset Layout`も試す。

## 8. 関連文書

- [分析ユーザーガイド](030_user_guide_analysis.jp.md)
- [開発環境](040_development_environment.jp.md)
- [実行基盤と設定](100_runtime_and_configuration.jp.md)
- [Agentと学習](110_agents_and_learning.jp.md)
- [環境](120_environments.jp.md)
- [可観測性](140_observability.jp.md)
- [ReplayBuffer](150_replay_buffer.jp.md)
- [DQN系Agent](200_dqn_agents.jp.md)
- [アプリケーションとツール](160_applications_and_tools.jp.md)

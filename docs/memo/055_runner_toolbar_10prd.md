# PRD 055: Runner ツールバー(対象軸 4 バー構成)

- 起票日: 2026-08-19
- 状態: implementation ready
- 対象: `apps/runner`(RunnerFrame / RunnerApp / TrainPanel / EvalPanel)、`core/anet-core` は `DefaultDQNAgent::Save` の lock 1 行のみ
- 関連: CONTEXT.md 用語 2 件(ツールバー pane / 実行時 UI 操作。本 PRD 作成時に追加済み)、`docs/memo/999_network_lock_audit_10prd.md`(Save 経路監査項目の部分消込)
- 設計文書: `docs/design/020_user_guide_run.jp.md`、`docs/design/160_applications_and_tools.jp.md`(§7 で実装時更新)
- ADR: 新設なし(可逆な UI 追加であり ADR 3 条件を満たさない。設計判断は §4 に記録)

## Context(背景・目的)

Runner の Run 制御(Train/Eval の pause/resume)は左クリック / `Shift` / `Space` / `Ctrl` という隠し操作しかなく、画面上に操作 UI と状態表示が存在しない。進行度(exp_step)は env 固有 View が個別に描いており(5 View 中 5 つが exp_step のみ、train_step 表示はゼロ)、View を持たない Atari では step が全く見えない。また pause 状態は UI 操作以外(自動 pause、`train_auto_start=false` 起動)でも変わるため、状態の可視化には実状態からの同期表示が必要である。

本 PRD は wxAUI ツールバー(auidemo 型: ドラッグ/アンドック可)を 4 本導入し、操作と状態を env 非依存の 1 箇所へ集約する。バーは「機能」ではなく**対象**(Train / Eval / Run 成果物 / Panel 表示)で分割し、「どのバーのボタンが何に効くか」を説明不要にする(Eval Step ボタンが Eval バーに載ることで帰属が自明になる)。あわせて、走行中 Save を安全化する lock 1 行、View FPS のメニュー変更、未使用ステータスバー欄への SPS/経過時間表示を含める。

## 0. 決定一覧(グリル確定値)

| ID | 決定 |
|---|---|
| D1 | バーは 4 分割: Run 制御(Train / Eval / Eval Step)/ Step 表示 / Run 操作 / Panel 表示。当初は対象軸(Train バーと Eval バーを分離)としたが、実使用で違和感が出たため操作系を 1 本へ統合した。Eval Step の帰属は、Train と Eval 系の間へ置く separator で示す |
| D2 | 全バー auidemo 型: `ToolbarPane()` で Gripper 有効(ドラッグ/アンドック可)、`CloseButton(false)`(×なし)、overflow・カスタマイズ機能なし。`AddPane()`後はwxAUIがpane側gripperを無効化し、`wxAuiToolBar`内蔵gripperだけを表示する |
| D3 | 状態同期は wxUpdateUIEvent(`SetUpdateInterval(200)`)。専用 wxTimer・Notifier への新イベント追加は行わない |
| D4 | StepCounts の UI 直読みは禁止(規格上データレース)。Observer + 既存 `UIDataStore<StepCounts>` 経由で受け渡す |
| D5 | Train pause 状態は `RunnerApp::IsTrainingPaused()`(atomic 直読み)、Eval pause 状態は `EvalPanel::IsPaused()`(UI スレッド完結)を新設して照会 |
| D6 | Step表示は専用バーとし、`exp`+read-only text control、標準separator、`train`+read-only text controlの順とする。数値は3桁区切り、選択・コピー可能、固定MinSizeとする |
| D7 | ツールバーの「▶ Eval」ON 時と「⏭ Step」押下時は、Eval pane が非表示なら自動表示してから実行する。`Space`/`Ctrl`/右クリックの既存経路は従来どおり表示と独立(挙動不変) |
| D8 | 走行中 Save の安全化は `DefaultDQNAgent::Save` 先頭への `std::shared_lock` 1 行(§2.6)。UI 側での pause 強制はしない |
| D9 | Save はワンクリック保存ではなく wxFileDialog(`wxFD_SAVE \| wxFD_OVERWRITE_PROMPT`)。既定ディレクトリ=RunDir、既定ファイル名=`agent_<exp_step>.anet`。**dialog を開く前に Train が走行中なら pause する**(ファイル名の step と保存内容のずれを防ぐ)。保存完了後・cancel 後の自動 resume はしない |
| D10 | Rainbow 等の Save 未実装 Agent(基底 no-op、0 バイト保存)対策として、`SaveAgent` 戻り値 size==0 で `LOG::warn` 1 行 |
| D11 | FPS メニューは View 配下に radio サブメニュー 2 つ。プリセットは Train View: `Config (N)` / `0 (Off)` / 1 / 5 / 10 / 30 / 60、Eval View: `Config (N)` / 1 / 5 / 10 / 30 / 60 / 120(Eval 側に 0 は置かない=▶ Eval トグルと役割重複するため。Eval FPS は実行速度そのものなので上振り側を 1 段広く取る) |
| D12 | ステータスバー フィールド 1=「`exp 1,234 steps/s    train 567 steps/s`」(区切りは記号でなく空白 4 つ。分子は語なので複数形、分母は `/s`)、フィールド 2=経過時間「`12:34:56`」。`SetStatusWidths` を設定(フィールド 0 可変、1/2 固定) |
| D13 | Run 制御系アイコン(▶ / ⏸ / ⏭)は単色 SVG 文字列リテラル + `wxBitmapBundle::FromSVG` + 実行時色置換で自作。Save/フォルダは wxArtProvider 標準。Train / Eval トグルのアイコンは実行状態に連動し、走行中は次の操作を表す ⏸、停止中は ▶ を表示する(メディアプレイヤ慣習)。ラベル文字列は対象識別のため "Train" / "Eval" のまま変えない |
| D14 | Panel 表示トグルはメニュー項目と同一 ID を共有し(wxEVT_TOOL は wxEVT_MENU と同一イベント型)、基底 `AuiLayoutFrame::OnTogglePaneMenu` を無改修で再利用 |
| D15 | Train トグルは `IsRunning()==false`(STOP 後=再開不能)で Disable。「Stop」ボタンは置かない(Stop は事実上の終了操作で `File > Exit` と同義のため) |
| D16 | Reset Layout(`RestoreDefaultPanes`)にツールバー 4 本の既定位置(Top Row0 Position0-3)回収を追加。フロート中も Dock へ戻す |
| D17 | 初期ウィンドウ幅(現行 800)は実装時に実測して必要なら 820〜860 へ調整してよい(幅見積 §5.6 参照) |
| D18 | 初期化完了後の Save / Open Run Folder の環境依存失敗はErrorDialogで通知してRunを継続する。close時のSave失敗も通知後にlog shutdown、Eval detach、AUI破棄を継続する |
| D19 | 左端gripperは`AddPane()`が有効化する`wxAuiToolBar`内蔵の標準gripper 1つだけとし、pane側gripperと重複させない。toolbar/buttonの背景・hover・checked色はwxWidgetsの標準artとsystem colourに任せ、独自色を指定しない |

## 1. 現状の事実(コード確認済み)

2026-08-19 時点、branch `main` で実測済み。

| 事実 | 根拠 |
|---|---|
| ツールバーは存在しない(`wxToolBar` / `wxAuiToolBar` / `CreateToolBar` の出現は apps・core 通じて 0 件)。wxUpdateUIEvent の使用実績も 0 件 | 全文 grep |
| Train 制御の実体は `RunnerThread`(=`ThreadBase`)の `Pause()` / `Resume()` / `IsPaused()` / `IsRunning()`。`paused_` / `running_` は `std::atomic<bool>`、pause はステップ境界 + 10μs スピン | `thread.hpp:26-32,47-48`、`thread.cpp:50-59` |
| `RunnerApp::trainer_thread_` は private で pause 状態の公開 getter が無い。`ToggleTraining()`(RunnerApp.cpp:548-562)内でのみ参照 | `RunnerApp.hpp:62` |
| Eval(EvalPanel)は wxTimer 駆動(UI スレッド)。pause は `EvalPanel::is_pause_`(private、getter 無し)を `OnTimer` 先頭で見て skip するだけ。pane を隠しても timer と eval は走り続ける | `EvalPanel.hpp:70`、`EvalPanel.cpp:53,126-136,201-225` |
| Train pause は UI 操作以外でも変わる: `train_pause_step` / `exp_pause_step` の自動 pause、`train_auto_start=false` 起動 | `RunnerApp.cpp:363-413` |
| `StopTraining()`(=`ThreadBase::Stop()`)は join まで行い再開不能。Stop 後も `IsPaused()` は false のままなので、走行表示の判定には `IsRunning()` の併用が必須 | `thread.cpp:31-40`、`RunnerApp.cpp:564-567` |
| `StepCounts`(train_step / exp_step / learn_step 等)は Runner インスタンスごとに独立。`RunnerBase::step_counts_` は plain uint64_t 群を Train スレッドが更新するため、UI からの `GetCounts()` 直読みは規格上データレース | `rl.hpp:94-127`、`trainer.hpp:44,63` |
| `TrainEvent.counts` に StepCounts が値コピーで載る。Serial は通知後カウント更新(=イベント値は 1 step 前)、Pipeline は prev_counts で 1 step 遅れ。表示用途では実害なし | `rl.hpp:882-906`、`trainer.cpp:549,620,627,675` |
| UI への正規経路は「Observer は Trainer スレッドで `UIDataStore` に書くだけ、UI は timer/イベントで吸い出す」(TrainPanel と同型)。`Notifier::AttachScoped<FunctionTrainObserver>(train_runner, ...)` で対象 runner の通知だけ受けられる | `gui.hpp:171-236`、`TrainPanel.cpp:35-41`、`rl.hpp:1004-1020` |
| SPS(EMA)と経過時間は `TrainRunner::GetScalar` で取得可能: `Runner::EXP_STEP_PER_SEC` / `TRAIN_STEP_PER_SEC`(**EMA 未初期化時 NaN**)、`ELAPSE_HOUR`(hour 単位 double) | `rl.hpp:1085-1092`、`trainer.cpp:385-403` |
| ステータスバーは 3 フィールド作成済みだがフィールド 0 のみ使用("Ready" / "Training paused" 等)。`SetStatusWidths` 呼び出しは 0 件 | `RunnerFrame.cpp:120-125`、`RunnerApp.cpp:383,391,560`、`EvalPanel.cpp:134` |
| `DefaultDQNAgent::Save` は lock を取らない。一方 Learner の `UpdateFromBatch` は `unique_lock<shared_mutex>` 下で parameter/optimizer を in-place 更新するため、走行中 Save は data race(Adam state map の構造変更と `torch::save` 走査が並行するとクラッシュ級)。`ImageClsAgent::Save` は `shared_lock` を取っており、`EvalRunner::Sync` → `Actor::CopySourceNetwork` / `GetScalar` / `GetTensor` も全て shared_lock 準拠。**Save だけが規約(Train スレッド外から Agent 状態に触るなら Agent shared_mutex)から漏れている** | `default_dqn_agent.cpp:255-290,538`、`image_cls_agent.cpp:538-543`、`dqn_based_agent.cpp:1674-1684` |
| Pause→Save は安全化にならない: `Pause()` は ack なしフラグで、Pipeline 構成では pause 後も LearnThread 上の update が走行中でありうる(close 時が安全なのは `Stop()` が join するため) | `thread.hpp:28`、`trainer.cpp:613-615,650-652` |
| `RunnerApp::SaveAgent(file_name)` は `GetRunDir()/file_name` 固定。production の呼び出しは `RunnerFrame::OnClose` の 1 箇所のみ | `RunnerApp.cpp:511-546`、`RunnerFrame.cpp:668` |
| Rainbow は `Save()` を override せず基底 no-op(`return 0`)。0 バイトファイルが作られる(docs/design/200 §記載済みの既知事項) | `rl.hpp:736` |
| 基底 `AuiLayoutFrame` の snapshot 往復(`TakeLayoutSnapshot` / `ApplyLayoutSnapshot`)・`SyncDockSizesToPanes` は全 pane 対象だが ToolbarPane にも安全(標準の wxAuiPaneLayoutInfo 往復)。`RunnerFrame::OnApplyLayoutPolicy` の右 dock 集計は layer/window/名前条件で ToolbarPane(layer=10, Top)を素通りし干渉しない | `gui.cpp:207-249`、`RunnerFrame.cpp:365-371` |
| `SetupPanes` は client size を読むため、メニュー/ステータスバー同様にツールバー生成も SetupPanes より前が必要 | `RunnerFrame.cpp:54` のコメント |
| env View の step 表示は 5 View とも exp_step のみ(`FormatWithCommas` 使用)。本 PRD で env 非依存表示が加わるが、View 側の表示は変更しない | GridMazeView.cpp:118 / ImageClsView.cpp:168 / CartPoleView.cpp:106 / LunarLanderView.cpp:390 / DropMergeView.cpp:522 |
| RunnerApp は `SetAppearance(Appearance::System)` でダークモード対応済み。固定色画像はダークで沈むためテーマ追従が必要 | `RunnerApp.cpp:199` |
| バックグラウンド eval(`EpisodeEvalObserver`)には停止 API 自体が無い(config `interval` で起動時固定)。本 PRD のスコープ外 | `observers.cpp:487-586` |

## 2. 仕様

### 2.1 バー構成(既定レイアウト)

```
[≡ ▶ Train │ ▶ Eval ⏭ Step]  [≡ exp 12,345,678 │ train 3,086,420]  [≡ 💾 📂]  [≡ Logs  Eval View  Q-Values]
   バー1 Run制御                  バー2 Step表示                        バー3 Run操作   バー4 Panel表示
```

4 本とも `wxAuiToolBar`(スタイル `wxAUI_TB_DEFAULT_STYLE | wxAUI_TB_HORZ_TEXT`)を pane として追加する:

```cpp
wxAuiPaneInfo().Name("TrainToolBar").ToolbarPane().Top().Row(0).Position(0).CloseButton(false)
```

- pane 名は `RunControlToolBar` / `StepToolBar` / `RunOpsToolBar` / `PanelToolBar`、Position 0〜3。float 時のミニフレームは pane caption を window title に使う(`floatpane.cpp:120`)ため、caption も与える(`Run Control` / `Steps` / `Run Operations` / `Panels`)。caption は `ToolbarPane()` でも LoadLayout 往復でも保持される(dock 中は `optionCaption` が落ちるので caption bar は出ない)。
- Gripper は `ToolbarPane()` の指定を受けた `AddPane()` が `wxAuiToolBar` 内蔵側だけを有効化する(ドラッグ・アンドック・再ドック可)。Reset Layoutで`ToolbarPane()`を再適用する際は`Gripper(false)`を続け、pane側と内蔵側の二重表示を防ぐ。CloseButton(false) でフロート時ミニフレームからも×を消す。
- overflow(»)・カスタマイズ機能は付けない。
- 生成は RunnerFrame ctor 内、`CreateStatusBar()` の後・`SetupPanes()` の前。
- 全ツールにツールチップを付ける(ショートカット併記。例: "Pause/resume training (Shift / left-click)")。

### 2.2 同期機構(全バー共通)

**wxUpdateUIEvent による宣言的同期**を唯一の同期手段とする。専用 wxTimer は追加しない。

- RunnerFrame 初期化時に `wxUpdateUIEvent::SetUpdateInterval(200);` を設定(現在 EVT_UPDATE_UI の使用箇所はゼロなのでグローバル設定の副作用なし)。
- 既存の TrainPanel(100ms)/EvalPanel(33ms)の wxTimer が常時回っており、タイマー処理後に必ずアイドルへ戻るため、UPDATE_UI は実質 200ms 周期で駆動される。
- 各ツールの `Bind(wxEVT_UPDATE_UI, ..., id)` ハンドラで**あるべき状態を宣言**する(`event.Check(...)` / `event.Enable(...)`)。押下時イベントで表示を切り替える方式は取らない(自動 pause・キー操作・メニュー操作との乖離が生じるため)。

各状態のデータソース:

| 表示 | ソース | スレッド安全性 |
|---|---|---|
| Train トグル check | `!wxGetApp().IsTrainingPaused()`(新設 getter、§3) | `ThreadBase::paused_` は atomic。直読み安全 |
| Train トグル enable | `wxGetApp().IsTrainingRunning()`(新設 getter) | `running_` は atomic。STOP 後 false → Disable(D15) |
| Eval トグル check | `eval_panel_->IsPaused()`(新設 getter)の否定 | `is_pause_` は UI スレッド完結 |
| Step 表示 / Save 既定名の exp_step | `UIDataStore<StepCounts>` スナップショット(§2.3) | mutex 保護(既存部品) |
| Panel トグル check | `aui_mgr_.GetPane(window).IsShown()` | UI スレッド完結 |
| ステータスバー 1/2 | `GetTrainRunner()->GetScalar(...)`(§2.8) | Perf EMA / 経過時間の read。既存 GUI からの GetScalar 参照と同等 |

### 2.3 StepCounts の受け渡し(直読み禁止)

`GetTrainRunner()->GetCounts()` を UI スレッドから直接読むことは**禁止**する(plain uint64_t 群への並行 read = 規格上データレース)。代わりに TrainPanel と同型の既存パターンを使う:

1. RunnerFrame(または toolbar 保持クラス)がメンバに `anet::rl::gui::UIDataStore<anet::rl::StepCounts> step_store_;` を持つ。
2. `RunnerFrame::Initialize` で `notifier->AttachScoped<FunctionTrainObserver>(train_runner, [this](const TrainEvent& e){ if (step_store_.ShouldUpdate()) step_store_.Update(e.counts); }, "RunnerToolBar")` を登録(Trainer スレッドで書くだけ。`ShouldUpdate()` ゲートで書き込みも間引かれる)。
3. Step表示のEVT_UPDATE_UI handlerで`step_store_.Get()`し、値が前回と変わった時だけexp/trainのread-only text controlへ`ChangeValue`する。
4. 一度もイベントが来ていない間(`Get()` が nullopt。`train_auto_start=false` 起動直後など)はexp/train両方の値欄へ`-`を表示する。
5. Observer は panel 群と同様、close 時に `Detach` する。

Serial/Pipeline のイベント値は 1 step 遅れるが表示用途では問題ない(§1)。

### 2.4 バー1「Run 制御」

| ツール | 種別 | 動作 |
|---|---|---|
| ▶ Train | `wxITEM_CHECK` トグル(自作 play/pause アイコン+ラベル "Train") | クリックで `wxGetApp().ToggleTraining()`(既存 API そのまま。ログ・ステータスバー・Flush 込み)。check=走行中、`IsRunning()==false` で Disable。直後に標準separatorを置き、Eval 系と分ける |
| ▶ Eval | `wxITEM_CHECK` トグル(自作 play/pause アイコン+ラベル "Eval") | クリックで: OFF→ON かつ Eval pane 非表示なら pane を表示してから `eval_panel_->TogglePause()`。ON→OFF は `TogglePause()` のみ。check=走行中 |
| ⏭ Step | 通常ボタン(自作 step アイコン+ラベル "Step") | Eval pane 非表示なら pane を表示してから `eval_panel_->DoStep(); eval_panel_->Refresh();`(既存 `Ctrl` キー相当)。pause 中でも押せる(現行 DoStep の挙動どおり) |

- pane 表示は Eval トグルと Step で共通の 1 ヘルパにまとめる(`GetPane(eval_panel_).Show(true)` → `Update()` → `ApplyLayoutPolicy()`。メニュー・toolbar のチェック同期は UPDATE_UI と基底の pane 連動に任せる)。
- 対象は **EvalPanel の Eval**(configured eval tag を鏡写し参照する UI 用インスタンス)のみ。バックグラウンド eval(eval schedule 駆動の EpisodeEvalObserver)には一切関与しない。
- pane 自動表示はツールバー経路のみの挙動。`Space` / `Ctrl` / 右クリックは従来どおり表示と独立(D7)。

### 2.5 バー2「Step 表示」

| ツール | 種別 | 動作 |
|---|---|---|
| step 表示 | `exp`/`train` label + `AddControl`したread-only `wxTextCtrl` 2個 | 各値を`FormatWithCommas`で表示し、値欄を選択・コピー可能にする。expとtrainの間に標準separatorを置く。1G step級が収まる固定MinSizeを与え、値変化でtoolbarが振動しないようにする |

### 2.6 バー3「Run 操作」

| ツール | 種別 | 動作 |
|---|---|---|
| Save Checkpoint | 通常ボタン(`wxArtProvider::GetBitmapBundle(wxART_FILE_SAVE)`、アイコンのみ) | wxFileDialog(`wxFD_SAVE \| wxFD_OVERWRITE_PROMPT`、拡張子フィルタ `*.anet`)を開く。既定ディレクトリ=`GetRunDir()`、既定ファイル名=`agent_<exp_step>.anet`(exp_step は §2.3 のスナップショットから。未取得時は `agent_0.anet`)。OK なら選択されたフルパスへ保存する。失敗時は対象path、理由、stack trace、不完全fileが残り得る旨をErrorDialogへ表示し、Runを継続する |
| Open Run Folder | 通常ボタン(`wxART_FOLDER_OPEN`、アイコンのみ) | `wxLaunchDefaultApplication(GetRunDir())` で Run ディレクトリをエクスプローラで開く。失敗時は対象path付きErrorDialogを表示し、Runを継続する |

**Save の安全化(core 側 1 行)**: `DefaultDQNAgent::Save`(default_dqn_agent.cpp:255 付近)の先頭に

```cpp
std::shared_lock<std::shared_mutex> lock(*mutex_);
```

を追加する(`ImageClsAgent::Save` と同形)。これにより Learner の `unique_lock` と排他され、**走行中 Save が正当な操作になる**(Save 中 learner が数百 ms 待つのみ)。UI 側での pause 強制・ボタン無効化は行わない(D8)。

**SaveAgent のシグネチャ変更(クリーンブレーク)**: `RunnerApp::SaveAgent(const std::string& file_name)` を `SaveAgent(const std::filesystem::path& file_path)`(フルパス受け取り)へ変更する。既存呼び出しは `RunnerFrame::OnClose` の 1 箇所のみで、`GetRunDir() / "agent_close.anet"` を渡す形に同時移行する。旧シグネチャの overload は残さない。

`RunnerApp::SaveAgent`は例外通知契約を維持し、open、serialization、flush、closeの各失敗を対象pathと失敗段階付きで送出する。`RunnerFrame`は対話的Saveの境界でこれを捕捉し、自動retry、fallback、不完全fileの自動削除を行わない。close時の`agent_close.anet`保存失敗も同じ方法で通知するが、その後のlog shutdownとGUI cleanupは必ず続行する(D18)。設定値や初期化時の契約違反は従来どおりfail-fastとする。

**Save 未実装 Agent の警告(D10)**: `SaveAgent` の戻り値 size が 0 の場合、`LOG::warn` で「Agent が保存を実装していない(0 バイト)」旨を対象ファイルパス付きで 1 行出す(Rainbow / MuZeroProto が該当)。

### 2.7 バー4「Panel 表示」

| ツール | 種別 | ID |
|---|---|---|
| Logs | `wxITEM_CHECK` トグル(テキストのみ) | `ID_LogView`(既存) |
| Eval View | 同上 | `ID_EvalPanel`(既存) |
| Q-Values | 同上 | `ID_QValuePanel`(既存) |

- **メニュー項目と同一 ID を共有する**。wxEVT_TOOL は wxEVT_MENU と同一イベント型のため、基底 `AuiLayoutFrame::OnTogglePaneMenu`(`RegisterPaneMenu` で Bind 済み)がツールバークリックでもそのまま発火する。基底クラスの改修は不要(D14)。
- ツール側のチェック同期は EVT_UPDATE_UI で `aui_mgr_.GetPane(window).IsShown()` を宣言する。これによりメニュー・ツールバー・pane の✕ボタンのどの経路で変えても両 UI が揃う(メニュー側は既存の `CheckPaneMenuItem` 同期のまま)。
- Reset Layout / HeatMap / Conv2d はツールバーに置かず、View メニューのみ維持(幅制約と、動的追加系はトグルでないため)。
- ラベルは実行トグル(▶ Eval)との混同を避けるため pane 名に合わせる("Eval View" / "Q-Values")。

### 2.8 ステータスバー(フィールド 1/2)

- `CreateStatusBar` 後に `SetStatusWidths` を設定: フィールド 0 = 可変(-1)、フィールド 1/2 = 内容が収まる固定幅。
- フィールド 1: `exp 1,234 steps/s    train 567 steps/s`(区切りは空白 4 つ)。書式は 1 つのヘルパへ集約し、幅見積もり・初期表示・更新で同じ関数を通す。値は `GetTrainRunner()->GetScalar(Runner::EXP_STEP_PER_SEC)` / `TRAIN_STEP_PER_SEC`(時間重み EMA)。**NaN(EMA 未初期化)の間は `-` を表示**する。整数へ丸めて `FormatWithCommas`。
- フィールド 2: 経過時間 `12:34:56`。`GetScalar(Runner::ELAPSE_HOUR)`(hour 単位 double)を h:mm:ss へ整形。
- 更新はツールバーと同じ EVT_UPDATE_UI 周期だが、Train toggleのtool IDではなくRunnerFrame自身へBindし、値変化時のみ `SetStatusText(text, field)`を呼ぶ。これによりstatus更新をtoolbar toolの存在から独立させる。
- フィールド 0 は操作結果メッセージ専用とし、`SetStatusBarPane(-1)` で wx の help pane 機構を無効化する。wxAuiToolBar の hover は `wxFrame::DoGiveHelp` 経由で help pane を書き換え、**tool から離れるときに hover 直前の文字列へ復元する**(`auibar.cpp:1519-1528`)。これが toolbar 操作で出した pause/resume メッセージを消すため。menu 項目の help 文字列は status bar へ出なくなるが、toolbar の short help は tooltip として従来どおり出る。
- メッセージ文言は対象名を UI 表記へ揃え、句点を付けない: `Training paused` / `Training resumed` / `Training paused automatically` / `Evaluation paused` / `Evaluation resumed`。

### 2.9 FPS メニュー(View 配下、取り急ぎ版)

```
View
├─ Train View FPS ▸  (radio) Config (10) / 0 (Off) / 1 / 5 / 10 / 30 / 60
└─ Eval View FPS ▸   (radio) Config (30) / 1 / 5 / 10 / 30 / 60 / 120
```

- 先頭 **`Config (N)`** は起動時 config 値(`app.train_panel.fps` / `app.eval_panel.fps`)を埋め込んだ動的項目で、起動時の既定チェック。プリセットに無い値(15 等)もこれで表現でき、プリセット選択後に config 値へ戻る経路にもなる。
- 選択時の動作: `TrainPanel::SetFps(float)` / `EvalPanel::SetFps(float)`(新設、§3)を呼ぶ。実装は `update_timer_.Stop()` → fps>0 なら `Start(1000/fps)`、**0 なら停止したまま**(Train View のみ。描画コストゼロ化)。
- **意味の差を UI 利用者向けドキュメントに明記する**: Train View FPS は純粋な表示頻度(学習速度・成績に無関係)。Eval View FPS は EvalPanel の実行速度そのもの(timer が step を回すため steps/sec = fps × step_per_frame が変わる)。
- これは実行時 UI 操作であり config dump(`config/config_data.txt`)には反映されない。Run の比較・再現の根拠にしないこと(CONTEXT.md「実行時 UI 操作」参照)。
- 将来パネル内 UI へ移設する際も `SetFps()` API を再利用する。

### 2.10 アイコン(Run 制御系のみ自作)

- 自作は ▶(play)、⏸(pause)、⏭(step)の 3 種。単色 SVG(viewBox 16×16、`fill="#000000"` 固定)を C++ 文字列リテラルとしてコードに内包し、`wxBitmapBundle::FromSVG(svg, wxSize(16,16))` で生成する(wx3.3 は NanoSVG 内蔵)。リソースファイル(.rc)には追加しない。
- SVG 案(実装時に微調整可):
  - play: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path fill="#000000" d="M4 2 L13 8 L4 14 Z"/></svg>`
  - pause: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect fill="#000000" x="4" y="2" width="3" height="12"/><rect fill="#000000" x="9" y="2" width="3" height="12"/></svg>`
  - step: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path fill="#000000" d="M3 2 L10 8 L3 14 Z"/><rect fill="#000000" x="11.5" y="2" width="2.5" height="12"/></svg>`
- **テーマ追従**: 生成時に `fill="#000000"` を `wxSystemSettings::GetColour(wxSYS_COLOUR_BTNTEXT)` の 16 進表現へ文字列置換してから FromSVG に渡す。`wxSysColourChangedEvent` を受けたら再生成して `SetToolBitmap` で差し替える(ダークモード対応。RunnerApp は `SetAppearance(Appearance::System)` 済み)。
- **トグルのアイコンは実行状態に連動する**(D13): 走行中は ⏸、停止中は ▶。同じ EVT_UPDATE_UI ハンドラ内で `SetToolBitmap` する。押し込み描画(checked)も従来どおり併用し、ラベル文字列は変えない(対象識別の維持とバー幅の固定のため)。
- 差し替えは**表示中の状態と実状態が食い違った時だけ**行い、200ms 周期での毎回 `SetToolBitmap` + 再描画は避ける(チラつき・再描画コスト防止)。tool が無効(Disable)の間の淡色描画は wxAUI が `m_bitmap` から都度生成するため(`auibar.cpp:117-128`)、無効時専用 bitmap の設定は不要。テーマ変更時の再生成も現在の実行状態を保って行う。
- Save / Open Folder は `wxArtProvider::GetBitmapBundle(wxART_FILE_SAVE / wxART_FOLDER_OPEN, wxART_TOOLBAR)`。

## 3. 追加・変更 API 一覧

| API | 種別 | 内容 |
|---|---|---|
| `RunnerApp::IsTrainingPaused() const` | 新設 | `return trainer_thread_->IsPaused();`(atomic load) |
| `RunnerApp::IsTrainingRunning() const` | 新設 | `return trainer_thread_->IsRunning();`(atomic load)。Train トグルの Disable 判定用 |
| `EvalPanel::IsPaused() const` | 新設 | `return is_pause_;` |
| `TrainPanel::SetFps(float fps)` | 新設 | timer 再起動。0 で停止(Off) |
| `EvalPanel::SetFps(float fps)` | 新設 | timer 再起動(0 は渡されない前提。防御は不要 — メニューに 0 項目が無い) |
| `RunnerApp::SaveAgent(const std::filesystem::path& file_path)` | シグネチャ変更 | フルパス受け取りへクリーンブレーク。呼び出し 1 箇所(OnClose)を同時移行。size==0 で WARN(D10) |
| `DefaultDQNAgent::Save` | 変更(core、1 行) | 先頭に `std::shared_lock<std::shared_mutex> lock(*mutex_);` を追加(§2.6) |

RunnerFrame 側の enum(`RunnerFrame.cpp:31-43`)には新規ツール ID(Train トグル / Eval トグル / Eval Step / Save / Open Folder / FPS メニュー項目群)を追加する。Panel トグルは既存 ID を共有(D14)。

## 4. 設計判断の記録

- **wxUpdateUIEvent 採用(タイマー/新イベント不採用)**: pause 状態は UI 操作・自動 pause・起動時設定の複数経路で変わるため、変化イベント駆動は発火点の仕込み漏れリスクがある。現在値を宣言するポーリング同期が堅牢で、wxUpdateUIEvent はそのための wx 標準機構。既存 wxTimer 群が回る本アプリではアイドルが常時発生するため実用上周期駆動になる。Notifier へのイベント追加・専用タイマーはいずれも不要。
- **StepCounts は Observer + UIDataStore 経由(直読み禁止)**: `RunnerBase::step_counts_` は plain uint64_t 群で、UI からの `GetCounts()` 直読みは C++ 規格上のデータレース。x64 実挙動では aligned 64bit load のため実害はまず出ないが、規格準拠かつ既存イディオム(TrainPanel と同型)で解決できるため直読みは採らない。`step_counts_` の atomic 化は hot path への侵襲と `GetByAxis` の参照返し破壊(歪)になるため不採用。
- **Save の安全化は core の lock 1 行(UI 側 pause 強制は不採用)**: このコードベースの規約は「Train スレッド外から Agent 状態に触るなら Agent の shared_mutex を取る」であり(`Sync` / `GetScalar` / `GetTensor` / ImageCls Save は準拠済み)、`DefaultDQNAgent::Save` だけが漏れていた。UI 側の「pause 中のみ Save 可」は Pipeline の残存 learn task に穴があり(Pause は ack なし)、規約準拠の lock が正解。`docs/memo/999_network_lock_audit_10prd.md` の監査項目「`Save()` / `torch::save` 経路が Actor/Learner と並行し得るか」「`Save()` は runtime 中にも呼ばれ得るか」への回答が本 PRD で確定する(呼ばれ得る、lock で守る)。
- **対象軸分割(案 A)**: 当初の機能軸(Run 制御バー+Step 表示バー)では Eval Step ボタンの帰属(何をコマ送りするか)が見た目から読めない。バー=対象にすることで「同一バー内のボタンは同一対象」という慣習(メディアプレイヤ/Unity/SUMO と同型)に載せた。
- **一般事例との対応**: Play/Pause/Step の 3 つ組は Unity Editor / SUMO の定番。カウンタ常時表示は NetLogo tick counter / SUMO Time 表示。Save は BizHawk の Save State に対応。

## 5. 実装ノート

### 5.1 生成順序

RunnerFrame ctor: `SetupMenuBar()` → `CreateStatusBar()`(+`SetStatusWidths`)→ **`SetupToolBars()`(新設)** → `SetupPanes()`。SetupPanes は client size を読むため(RunnerFrame.cpp:54 コメント)、ツールバー pane 追加で client 領域が確定した後に呼ぶ。ツールバーの `Realize()` を忘れない。

### 5.2 Reset Layout

`RestoreDefaultPanes()` に 4 本のツールバー pane を既定(Top Row0 Position0-3、Dock、Show)へ戻す処理を追加。フロート中も `Dock().Top()` で回収する(D16)。`ToolbarPane()`の再適用後はpane側に`Gripper(false)`を指定し、既に有効なtoolbar内蔵gripperと重複させない。

### 5.3 EvalPanel 初期化前のガード

ツールバーは ctor で生成されるが、`eval_panel_` の runner 接続や `UIDataStore` の Observer は `Initialize()` 後に有効になる。EVT_UPDATE_UI ハンドラは各ポインタ/`Get()` の null/nullopt を許容し、初期化前は Disable または `-` 表示にする。

### 5.4 wxEVT_TOOL = wxEVT_MENU の同一視(D14)

Panel トグルはメニューと同一 ID なので `Bind(wxEVT_MENU, OnTogglePaneMenu, id)`(基底が実施済み)がツールバーからも発火する。トグルツールのクリック時は wxAuiToolBar が内部チェック状態を反転してからイベントを出すため、`event.IsChecked()` は正しい目標状態を返す。EVT_UPDATE_UI が毎周期 `pane.IsShown()` で上書きするため、途中の不整合も自己修復される。

### 5.5 表示更新の作法

トグルの `SetToolBitmap`、step値欄の`wxTextCtrl::ChangeValue`、ステータスバー`SetStatusText`は**値が変わった時のみ**呼ぶ(毎周期呼ぶと再描画コストとチラつき)。step値欄には1G step級が収まる固定MinSizeを与える。

### 5.6 幅見積(実測して調整)

概算(Segoe UI 9pt、gripper+ボーダー 14px/本): バー1 ≈ 370px(トグル 71 + テキスト 280)、バー2 ≈ 142px、バー3 ≈ 62px、バー4 ≈ 210px、計 ≈ 785px。初期ウィンドウ幅 800 にギリギリのため、**実装時に実測し、必要なら初期幅を 820〜860 へ調整してよい**(D17)。溢れた場合ユーザーはドラッグで Row 1 へ落とせる(Reset Layout で Row 0 一列へ復帰)。

### 5.7 テスト観点

GUI 主体のため自動テストは要求しない。ビルド(Debug)+以下の手動確認を受入とする(§6)。`DefaultDQNAgent::Save` の lock は既存テスト(`anet-core-test`)の緑維持で確認。

## 6. 受入条件

1. 4 本のツールバーが既定で Top 1 行に並び、ドラッグ/アンドック/再ドックでき、×ボタンが無い。Reset Layout で既定位置へ戻る(フロート中も)。
2. Train トグル: クリックで pause/resume が切り替わり、`Shift`/左クリック/自動 pause(`train_pause_step`)/`train_auto_start=false` 起動のいずれで状態が変わっても押下表示とアイコン(走行中 ⏸ / 停止中 ▶)が追従する。`train_exit_step` 到達(STOP)後は Disable される。
3. Eval トグル: クリックで EvalPanel の pause/resume が切り替わり、アイコンも ⏸/▶ へ切り替わる。`Space`/右クリックとも同期する。pane 非表示で ON にすると pane が表示される。⏭ Step は pane 非表示でも pane を表示してから 1 step 進む。
4. Step表示がTrain Runnerのexp_step/train_stepを別々のread-only text controlへカンマ区切りで表示し、走行中に更新される。値は選択・コピーでき、`train_auto_start=false`起動直後は両方`-`表示となる。Train直後とexp/train間は同じ標準separatorを使う。
5. Save Checkpoint: ダイアログが RunDir・`agent_<exp_step>.anet` 既定で開き、**Train 走行中に保存してもクラッシュ・破損しない**。保存した checkpoint が `auto_load_file` で読み込める。Rainbow 構成では WARN が出る。書込み不能pathへの保存はErrorDialog表示後もRunが進行し、有効pathへ再Saveできる。close時の`agent_close.anet`保存失敗も通知後にlog shutdownとwindow closeが完了する。
6. Open Run Folder で Run ディレクトリが開く。起動失敗時は対象path付きErrorDialogを表示し、main loopへ例外を漏らさない。
7. Panel 表示トグル 3 個がメニュー・✕ボタンと双方向同期する。
8. FPS メニュー: Train View 0(Off) で描画が止まり Train は走り続ける。Eval View FPS 変更で Eval の進行速度が変わる。`Config (N)` で起動時値へ戻る。
9. ステータスバーに SPS(NaN 時 `-`)と経過時間が表示・更新される。
10. ライト/ダークテーマ切替でアイコンが追従する。
11. `x64-Debug` ビルドが通り、`anet-core-test` が緑(既知の失敗を除く)。

## 7. ドキュメント更新(実装と同一変更内で実施)

| 文書 | 更新内容 |
|---|---|
| `docs/design/020_user_guide_run.jp.md` | §4.1 画面構成にツールバー 4 本を追記。§5 操作方法にツールバー操作(トグル/Step/Save/Open Folder/FPS メニュー)を追記し、キー操作は併存として維持。§7 の「起動直後に Train が進まない」等にツールバーでの確認方法を追記 |
| `docs/design/160_applications_and_tools.jp.md` | RunnerFrame の説明(§ コンポーネント表)にツールバー pane と wxUpdateUIEvent 同期を追記 |
| `docs/memo/999_network_lock_audit_10prd.md` | 監査項目「`Save()` / `torch::save` 経路が Actor/Learner と並行し得るか」と Open Question「`Save()` は runtime 中にも呼ばれ得るか」へ「PRD 055 で確定: runtime 中に呼ばれ得る。DefaultDQNAgent::Save に shared_lock 追加済み」の注記 |
| `CONTEXT.md` | 「ツールバー pane」「実行時 UI 操作」の 2 語(本 PRD 作成時に追加済み) |

## 8. スコープ外

- バックグラウンド eval(EpisodeEvalObserver)の一時停止・制御(停止機構自体が無い。必要になったら別 PRD)
- Train の 1 step 実行(core に API 無し。`pre_step_func` の `ControlSignal::BREAK` パターンで将来実装可能)
- パネル内 FPS UI(第 2 弾。本 PRD の `SetFps()` を再利用する)
- Step 表示のクリックでの表示切替(exp/train ↔ episode/learn)、Eval Step の長押し連続実行
- ツールバーのカスタマイズ機能・overflow(»)
- Stop(終了)ボタン(`File > Exit` と同義のため置かない)
- env 固有 View 内の既存 step 表示の変更・削除
- `metrics-viewer` 側の変更

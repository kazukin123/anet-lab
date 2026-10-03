# PRD 055 Runner ツールバー実装計画

## 概要

- 本計画を実装開始時の正本とし、以後の実装判断は本書に従う。
- Runner 上端へ Train / Eval / Run 操作 / Panel 表示の4本の `wxAuiToolBar` を追加する。
- Step、SPS、経過時間は Trainer スレッドで生成する snapshot を `UIDataStore` 経由で表示し、UI から `TrainRunner` の可変状態を直接読まない。
- 既存の `CONTEXT.md` 用語を維持し、新規 ADR は作成しない。

## 主な変更

### ツールバーと状態同期

- `RunnerFrame` の初期幅を 860px とし、status bar 後・既存 pane 前に4本の toolbar pane(Run制御 / Step表示 / Run操作 / Panel表示)を生成する。Run制御は Train toggle、separator、Eval toggle、Eval Step の順で 1 本にまとめる。全 toolbar は上端 Row 0、Position 0〜3、`AddPane()`が有効化する`wxAuiToolBar`内蔵の標準gripperのみ有効、close/overflow無効とする。独自gripperや独自button色は追加せず、wxWidgets標準art/system colourを使う。
- `wxUpdateUIEvent` を200ms周期に設定し、Train/Eval状態、Panel表示、Step値欄、status barを実状態から同期する。初期化前は Train/Eval/Save を disable し、exp/trainの値欄は`-`、SPSは `-`、経過時間は `--:--:--` とする。
- Train toggle直後とexp/train間には同じ標準separatorを置く。exp/trainの数値は固定幅のread-only `wxTextCtrl`へ分け、選択・コピー可能にする。
- status snapshot更新はRunnerFrame自身の`EVT_UPDATE_UI` handlerへBindし、Train toggleのenable/check更新から分離する。
- Train snapshot は counts、SPS、経過時間、取得時刻をまとめる。Trainer-thread callback内で `GetScalar()` し、`UIDataStore` は request-driven modeとして強制更新間隔0で使用する。経過時間はUI側で取得時刻から補間し、pause中も進める。
- `RunnerScopedTrainObserver` wrapper自体を保持してclose時にdetachする。`AttachScoped()` の戻り値を直接detachする既存の不一致は踏襲せず、汎用Notifierの改修は行わない。
- play/pause/step SVGはsystem text色で生成し、`wxEVT_SYS_COLOUR_CHANGED` で差し替える。Train/Evalトグルのbitmapは実行状態へ連動させ(走行中⏸・停止中▶)、反映済み状態と差がある時だけ`SetToolBitmap`する。theme再生成も現在の実行状態を保つ。Save/Open Folderは標準artを使う。文字列とstatus barは値が変化した場合だけ更新する。

### 操作とFPS

- Train toggleは既存 `ToggleTraining()` を呼び、実行中をcheck、停止後をdisableとする。新設getterは未初期化時も安全な値を返す。
- Eval toggleはresume時だけ非表示paneを表示してから `TogglePause()` する。Stepも同じヘルパでpaneを表示してから既存 `DoStep()` 経路を使う。どちらも初期化前はdisableする。
- Panel toggleは既存menu IDを共有する。同じ `EVT_UPDATE_UI` handlerでmenuとtoolbarをpane状態へ同期する。
- View menuへTrain/Eval FPS radio submenuを追加する。`Config (N)` は起動時値、presetはPRD記載値とし、選択内容はconfig dumpへ書き戻さない。
- Train FPSは有限な0〜1000、Eval FPSは有限な0超〜1000を契約とする。configと`SetFps()`の双方で検証し、違反はキー・値・期待範囲付きでfail-fastする。Trainの0のみtimer停止、正値は丸めた1ms以上のintervalで再起動する。
- Reset Layoutで4本を上端1行へdockし直す。`ToolbarPane()`の再適用後はpane側を`Gripper(false)`とし、既に有効なtoolbar内蔵gripperと重複させない。Open Run Folder失敗は対象path付きErrorDialogで通知してreturnし、main loopへ例外を漏らさない。

### API・保存処理

- `RunnerApp` にnull-safeな `IsTrainingPaused() const` / `IsTrainingRunning() const` を追加する。
- `TrainPanel::SetFps(float)`、`EvalPanel::SetFps(float)`、`EvalPanel::IsPaused() const` と、各panel configの値域検証を追加する。
- `RunnerApp::SaveAgent` を `const std::filesystem::path&` のフルパス契約へクリーンブレークし、旧signatureと不要になる `GetOutputStream()` を削除する。close時は `GetRunDir() / "agent_close.anet"` を渡す。
- Save Checkpoint dialogはRunDirと `agent_<exp_step>.anet` を既定にする。dialog表示前に走行中なら`RunnerApp::PauseTraining()`でpauseし、保存後・cancel後も自動resumeしない。Unicode pathを保持して直接binary streamを開き、log/archive名はUTF-8化する。保存sizeが0なら対象path付きで未実装の可能性をWARNする。
- `RunnerApp::SaveAgent`はbusy cursorをRAII化し、open、serialization、flush、closeのstream状態を検証する。失敗は対象pathと段階付き例外として送出し、自動retry・fallbackは行わない。
- `RunnerFrame::TrySaveAgent(path)`で`AnetException`、`std::exception`、未知例外を捕捉し、対象path・理由・stack traceと不完全fileが残り得る旨をErrorDialogへ表示する。toolbar Save失敗後もRunを継続し、close時Save失敗後もlog shutdown、Eval detach、AUI破棄を続行する。不完全fileは自動削除しない。
- `DefaultDQNAgent::Save` のserialization全体をAgentのshared mutexで保護し、走行中SaveをLearner更新と排他する。pause強制や自動fallbackは追加しない。
- 新設helperは各`.cpp`の`runner_app_detail`、`runner_frame_detail`、`train_panel_detail`、`eval_panel_detail`へ置く。pathのUTF-8変換は`RunnerApp.cpp`だけに保持し、`RunnerFrame`では`wxString`へ直接変換する。

### ドキュメント

- Runユーザーガイドへ4本のtoolbar、既存shortcutとの併存、Save/Open Folder、FPS、status表示、トラブルシュートを追加する。
- アプリ設計文書へtoolbar pane、Observer snapshot、`wxUpdateUIEvent`、Train/Eval FPSの意味の差を反映する。
- network lock監査memoのSave項目とOpen Questionへ、PRD 055でruntime Saveとshared lockが確定した旨を追記する。
- `CONTEXT.md` の既存「ツールバー pane」「実行時 UI 操作」は変更せず保持する。

## テスト・検証

- 合意した例外として新しいRunner GUI test harnessやtest-only APIは追加しない。GUI部分は手動受入、Saveは既存DefaultDQN checkpoint roundtripと全core testで回帰確認する。
- 手動確認では、toolbarのdock/float/reset、shortcut・自動pauseとの同期、Eval pane自動表示とStep、menu/pane/toolbar双方向同期、FPS Off/変更、status表示、theme切替を確認する。gripperは起動直後、Reset Layout後、Reset後のアンドック→再ドック、複数回Resetのすべてで1つだけ表示されることを確認する。加えて、書込み不能pathへのSave後もRunが進行し再Saveできること、close時Save失敗後もcleanupとwindow closeが完了すること、Open Run Folder失敗が非致命であることを確認する。
- DefaultDQNを走行させたまま任意pathへ保存し、新Runの`auto_load_file`で読めることを確認する。Rainbowでは0-byte WARN、close時は従来の`agent_close.anet`を確認する。
- 検証コマンド:

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target AnetRLRunner anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

## 前提

- background eval、Train 1-step、panel内FPS UI、toolbar customization、metrics-viewerは対象外とする。
- 汎用 `UIDataStore` の強制更新時刻と `AttachScoped()` detach契約の既存問題は別課題とし、PRD 055では局所的に回避する。
- 無関係な未コミット変更を保持し、Git staging・commit・pushは行わない。

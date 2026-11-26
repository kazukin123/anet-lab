# Runner ツールバー(対象軸 4 バー構成)

## 元の材料

[Runner ツールバー(対象軸 4 バー構成)](../../../memo/done/0055_runner_toolbar_10rfd.md) blob:231fd03cdcfa36da9f8f8e8510801fa40f875db0

## 要点

- Runner の操作を、Run 制御、step 表示、Run 操作、Panel 表示の4つのツールバーに整理する。
- 学習・評価・評価1ステップは、操作する人が同じ場所から扱えるよう Run 制御に集める。
- 各バーはドラッグとアンドックができ、閉じるボタンは置かない。
- 既存の TrainPanel・EvalPanel の実行状態と、ツールバーからの操作を同期する。
- Run の保存経路も含め、実行中に利用者が操作するときの状態と排他を確認する。

## 内容

### 仕様

wxUpdateUIEvent による宣言的同期を唯一の同期手段とする。専用 wxTimer は追加しない。
既存の TrainPanel(100ms)/EvalPanel(33ms)の wxTimer が常時回っており、タイマー処理後に必ずアイドルへ戻るため、UPDATE_UI は実質 200ms 周期で駆動される。
Step表示のEVT_UPDATE_UI handlerで`step_store_.Get()`し、値が前回と変わった時だけexp/trainのread-only text controlへ`ChangeValue`する。
ツール側のチェック同期は EVT_UPDATE_UI で `aui_mgr_.GetPane(window).IsShown()` を宣言する。これによりメニュー・ツールバー・pane の✕ボタンのどの経路で変えても両 UI が揃う(メニュー側は既存の `CheckPaneMenuItem` 同期のまま)。
先頭 `Config (N)` は起動時 config 値(`app.train_panel.fps` / `app.eval_panel.fps`)を埋め込んだ動的項目で、起動時の既定チェック。プリセットに無い値(15 等)もこれで表現でき、プリセット選択後に config 値へ戻る経路にもなる。

### 設計判断の記録

wxUpdateUIEvent 採用(タイマー/新イベント不採用): pause 状態は UI 操作・自動 pause・起動時設定の複数経路で変わるため、変化イベント駆動は発火点の仕込み漏れリスクがある。現在値を宣言するポーリング同期が堅牢で、wxUpdateUIEvent はそのための wx 標準機構。既存 wxTimer 群が回る本アプリではアイドルが常時発生するため実用上周期駆動になる。Notifier へのイベント追加・専用タイマーはいずれも不要。
対象軸分割(案 A): 当初の機能軸(Run 制御バー+Step 表示バー)では Eval Step ボタンの帰属(何をコマ送りするか)が見た目から読めない。バー=対象にすることで「同一バー内のボタンは同一対象」という慣習(メディアプレイヤ/Unity/SUMO と同型)に載せた。
一般事例との対応: Play/Pause/Step の 3 つ組は Unity Editor / SUMO の定番。カウンタ常時表示は NetLogo tick counter / SUMO Time 表示。Save は BizHawk の Save State に対応。

### 実装ノート

ツールバーは ctor で生成されるが、`eval_panel_` の runner 接続や `UIDataStore` の Observer は `Initialize()` 後に有効になる。EVT_UPDATE_UI ハンドラは各ポインタ/`Get()` の null/nullopt を許容し、初期化前は Disable または `-` 表示にする。
トグルの `SetToolBitmap`、step値欄の`wxTextCtrl::ChangeValue`、ステータスバー`SetStatusText`は値が変わった時のみ呼ぶ(毎周期呼ぶと再描画コストとチラつき)。step値欄には1G step級が収まる固定MinSizeを与える。

### 受入条件

Eval トグル: クリックで EvalPanel の pause/resume が切り替わり、アイコンも ⏸/▶ へ切り替わる。`Space`/右クリックとも同期する。pane 非表示で ON にすると pane が表示される。⏭ Step は pane 非表示でも pane を表示してから 1 step 進む。
Step表示がTrain Runnerのexp_step/train_stepを別々のread-only text controlへカンマ区切りで表示し、走行中に更新される。値は選択・コピーでき、`train_auto_start=false`起動直後は両方`-`表示となる。Train直後とexp/train間は同じ標準separatorを使う。
Open Run Folder で Run ディレクトリが開く。起動失敗時は対象path付きErrorDialogを表示し、main loopへ例外を漏らさない。
Panel 表示トグル 3 個がメニュー・✕ボタンと双方向同期する。

### スコープ外

バックグラウンド eval(EpisodeEvalObserver)の一時停止・制御(停止機構自体が無い。必要になったら別 PRD)。
Train の 1 step 実行(core に API 無し。`pre_step_func` の `ControlSignal::BREAK` パターンで将来実装可能)。
Stop(終了)ボタン(`File > Exit` と同義のため置かない)。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)

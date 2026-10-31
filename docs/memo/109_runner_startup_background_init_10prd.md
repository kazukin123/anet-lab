# Runner の起動で RunManager を別スレッドで構築する PRD

> 2026-10-31 起票（Claude Code、`anet-create-prd`。グリルあり）。
> 2026-10-31 別モデル（Fable）の検討のあと（同じ枠）: 初期化中に閉じる要求があったとき、構築のあと後半を行わずに閉じるよう D7 を改めた（後半を行う理由にした「close 手順は後半を前提にする」がコードと合わなかった。§1.4 の 11。ユーザーの裁定）。初期化中の OS のログオフ・シャットダウンは拒否せず、構築の終わりを待って閉じると D8 に決めた（wx は OS の終了をまず拒否できる閉じる要求として届ける。§1.4 の 13。ユーザーの裁定）。RunManager が RUNNING にならなかったときに close の保存と終了後の後始末で落ちる経路を D4 で塞ぎ、拒否できない閉じ方で構築が失敗していたときの扱いを D8 に、構築の結果を 1 回だけ受け取ることを D2 に、HeatMap のハンドラ側の確かめを D6 に足した。§4.1 に案の代価の比較を足し、受入 3・4 を直した。
> 2026-10-31 実装側（Codex）の指摘のあと: wxWidgets 3.3.3 のソースで、OS の終了の確定の通知は Frame の閉じる要求にならず、終了処理（`OnExit`）を直接呼んでプロセスを終えると分かったので、§1.4 の 13 を訂正した。D8 から「OS の終了が決まったあとの通知」を外し、その通知は wx の扱いのまま D10 の待ちだけを通すと決めた（待機と Close の道は足さない。ユーザーの裁定）。問い合わせは Frame の側で ✕ と見分けられないので、RunnerApp が Frame の閉じる操作と分けて受けることを D8 に書いた。§5.2・§5.4・§5.5・§9 を合わせた。改訂のレビューの指摘で、確定の通知の道では `OnExit` の前に Frame とパネルが閉じる要求を経ずに削除されることを §1.4 の 13・D10・§5.2 に書き、その道で構築が失敗していたときの終了コード（1）を D10・§5.3 に、表示を省くことを G4 に書いた。160 §7.1 の「`OnExit()` は cleanup 専用」を直す対象に D13・受入 8 で加えた。
> 起点: 2026-10-31、Runner の起動直後にウィンドウの中が描かれず、前の画面の残りが見えたまま乱れ、しばらくすると「応答なし」になる。ログ欄だけは起動中もログが流れて見える（§1.3）。
> 関連: [071](071_uncaught_fatal_visibility_10prd.md)（捕まらない fatal の可視化。OnInit の冒頭と worker の例外に触るが、関心は別。NG5）、[068](done/068_batchrun_fatal_error_handling_10prd.md)（batchrun の致命エラー処理。本 PRD は §7.1 の fatal 経路をそのまま使う）、[999_runner_theme_switch](999_runner_theme_switch_10prd.md)（RunnerFrame の外観の決め直し。触る箇所は重ならない）、[160 アプリケーションとツール](../design/160_applications_and_tools.jp.md) §5.1・§7.1。

## 1. 問題

### 1.1 今の起動の流れ（2026-10-31 にコードで確かめた）

`RunnerApp::OnInit`（[RunnerApp.cpp:255-492](../../apps/runner/src/RunnerApp.cpp)）は、次をすべてメインスレッドで順に行い、最後に戻る。wxWidgets のイベントループは OnInit が戻ってから回り始める。

| 順 | 処理 | 位置 |
|---|---|---|
| 1 | workspace の確定、設定の解決、MetricsLogger の初期化、InitRL、Env の登録 | :284-376 |
| 2 | RunnerFrame の生成、ログの初期化、**Frame の表示** | :379-401 |
| 3 | `RunnerApp config=` のログと MetricsLogger への記録 | :404-405 |
| 4 | **RunManager の構築**（学習用 BatchEnv、Agent とネットワーク、Learner、定期評価の Runner） | :408 |
| 5 | DefaultViewFactory、InitTrainer、ImageProviderManager | :416-427 |
| 6 | `frame_->Initialize`（TrainPanel の View、EvalPanel 用 EvalRunner の生成、QValuePanel、Evaluation Trace） | :430、[RunnerFrame.cpp:493-518](../../apps/runner/src/RunnerFrame.cpp) |
| 7 | `Notifier:` の一覧のログ、学習スレッド（`TrainThread`）の生成と開始 | :433-489 |

### 1.2 実測（2026-10-30 21:55:54 開始の qbert の Run `run_20261030-215554_a5_qb_aln_t98c30_s1`、Atari5-01）

Run ログの時刻から、表示から OnInit の終わりまでを区切った。

| 区間 | 時刻 | 長さ |
|---|---|---|
| 表示の直後（順 3 の `RunnerApp config=` の行） | 21:55:55.073 | — |
| 学習用 env 64 本の生成（`BatchEnvBuilder config=` から `batch_env_spec=` まで） | 21:55:55.097 → 21:55:58.191 | 3.09 秒 |
| 最初のネットワーク構築（1 つ目の `NetworkBuilder` から 2 つ目まで。CUDA と cuDNN の初期化を含むと推定） | 21:55:58.312 → 21:56:03.262 | 4.95 秒 |
| Learner と定期評価の Runner（greedy・eval・greedy_single）の構築 | 21:56:03.262 → 21:56:06.198 | 2.94 秒 |
| 順 5・6（EvalPanel 用 EvalRunner のネットワーク構築 2 回を含む。境目の 06.198 は RunManager の最後のログ行と推定） | 21:56:06.198 → 21:56:06.577 | 0.38 秒 |
| 順 7 の `Notifier:` の一覧の先頭 | 21:56:06.577 | 合計 11.50 秒 |

約 11.5 秒のうち、順 4 の RunManager の構築が約 11.1 秒を占める。順 1〜3 は、Run 名の時刻（MetricsLogger の初期化。21:55:54 台で、秒までしか分からない）から順 3 の行（21:55:55.073）までで、長くても約 1.07 秒である。

### 1.3 何が起きるか

- 順 4〜7 の間はイベントループが回らないため、描画のメッセージが処理されない。表示した Frame の中は描かれず、前の画面の残りや白・黒の領域が残る。ほかのウィンドウを重ねたり Frame を動かしたりすると、乱れが増える。
- Windows は、5 秒を超えてメッセージを処理しないウィンドウを「応答なし」と判定し、タイトルに「(応答なし)」を付けた代わりの窓に差し替える。§1.2 の構成では約 11.5 秒続くので、毎回この状態になる。
- ログ欄は RichEdit（`wxTE_RICH2`、[160 §3.5](../design/160_applications_and_tools.jp.md)）で、追記のたびにその窓だけを自分で描き直すため、起動中もログが流れて見えると推定している（確かめていない）。そのため「ログだけ流れて、周りは乱れている」状態になる。
- サイズ変更で予約されるレイアウト方針の適用（`AuiLayoutFrame::SchedulePolicy`、[gui.cpp:209-218](../../core/anet-core/src/gui.cpp)）も、OnInit が戻るまで実行されない。

### 1.4 関わる今の事実（2026-10-31 にコードで確かめた）

| # | 事実 | 位置 |
|---|---|---|
| 1 | `LOG` は wxLog を通る。メインスレッドで出したログは、その場で LogPanel と Run ログファイル（C の入出力のバッファ経由）へ渡る。メインスレッド以外で出したログは wx が溜め、メインスレッドのアイドル時と、`app.log_flush_interval_ms`（既定 500 ms）ごとの書き出しのタイマーで出す。学習スレッドのログは今この経路で出ている | [log.hpp:117-121](../../core/anet-core/include/anet/log.hpp)、:226-264、[RunnerApp.cpp:517-546](../../apps/runner/src/RunnerApp.cpp) |
| 2 | 学習スレッドで出た例外は、wx の `StoreCurrentException` で控え、`CallAfter` でメインスレッドで投げ直す。メインループ中の例外は `OnExceptionInMainLoop` が fatal の印を立ててエラーを表示し、ループを終える。`OnRun` は、印が立っていれば終了コード 1 を返す | [RunnerApp.cpp:475-478](../../apps/runner/src/RunnerApp.cpp)、:712-728、:776-787、[160 §7.1](../design/160_applications_and_tools.jp.md) |
| 3 | エラーの表示は `app.show_error_dialog` に従う。online 構成はダイアログ、batchrun 構成は stderr と Run ログ | [160 §7.1](../design/160_applications_and_tools.jp.md)、[068](done/068_batchrun_fatal_error_handling_10prd.md) |
| 4 | launcher は、Runner の終了コードが 0 以外なら失敗として扱う | [11_batch_run.bat:65-67](../../apps/11_batch_run.bat) |
| 5 | RunManager の状態が RUNNING でないとき、OnInit は `Failed to initialize RunManager.` を出して、学習を始めずに true を返す（ウィンドウは残る） | [RunnerApp.cpp:409-413](../../apps/runner/src/RunnerApp.cpp) |
| 6 | RunnerFrame は `frame_->Initialize` が済んだかを覚えており、Eval の切替・Eval Step・Save・Train FPS・Eval FPS・Eval Steps per Frame のメニューとツールバーを、済むまで Update UI で無効にする。Eval の切替・Eval Step・Save は、ハンドラ側でも済んだかを見て戻る。Train の切替は学習スレッドが動くまで無効。Env の View への入力は、View が `frame_->Initialize` で作られるまで届かない | [RunnerFrame.cpp:733-806](../../apps/runner/src/RunnerFrame.cpp)、:1221-1226、:1251-1261、:1290-1292 |
| 7 | `RunnerFrame::OnClose` は、学習の停止、評価の排水の確認、保存（`TrySaveAgent`）、排水、ログの片付け、EvalPanel の切り離し、AUI の破棄の順に進む。保存は RunManager の Agent を使う | [RunnerFrame.cpp:1364-1406](../../apps/runner/src/RunnerFrame.cpp)、[RunnerApp.cpp:581-631](../../apps/runner/src/RunnerApp.cpp) |
| 8 | RunManager の構築の中に、メインスレッドを待つ処理は無い。ffmpeg の起動をメインスレッドへ頼んで待つ仕組み（`ExecuteStarter`）は、動画を最初に記録するときだけ使う。InitRL が決める torch の設定（スレッド数、TF32、cuDNN、決定的アルゴリズム）は、どれもプロセス全体に効く。スレッドごとの torch の状態のうち autocast は、使う側が RAII で元に戻す | [metrics_logger.cpp:359-364](../../core/anet-core/src/metrics_logger.cpp)、:770-773、[init.cpp:122-160](../../core/anet-core/src/init.cpp)、[nn_util.cpp:205-209](../../core/anet-core/src/nn_util.cpp) |
| 9 | 実 RunnerApp を起動する GUI テストが 2 本あり、どちらも `OnInit()` が戻った直後に RunManager と Frame の子を使う。後始末では `Close(true)`（拒否できない閉じ方）を使う | [EvalTracePanel_test.cpp:876-910](../../apps/runner/src/EvalTracePanel_test.cpp)（`[eval_steps_menu]`）、:1044-1090（`[prd105_ui_keys]`） |
| 10 | 今は `frame_->Initialize` の前にイベントループが回らないので、初期化前のパネルの描画・タイマー・サイズ変更と、初期化前のメニューの操作は、製品では一度も動いたことがない。読んだ範囲では、TrainPanel と EvalPanel はタイマーを `Initialize` で始め、Evaluation Trace のタイマーは準備ができていなければ何もせずに戻り、Conv2d のメニューは EvalPanel 用 EvalRunner が無ければ何もしない。**HeatMap のメニューだけは無効にされておらず、ハンドラも確かめずに RunManager を使う**（初期化前に押すと落ちる） | [TrainPanel.cpp:40-83](../../apps/runner/src/TrainPanel.cpp)、[EvalPanel.cpp:147-165](../../apps/runner/src/EvalPanel.cpp)、[EvalTracePanel.cpp:159](../../apps/runner/src/EvalTracePanel.cpp)・:711-717、[RunnerFrame.cpp:703](../../apps/runner/src/RunnerFrame.cpp)・:1143-1146・:1174-1176 |
| 11 | 今の close 手順は、`frame_->Initialize` が済んでいなくても通る。学習の停止・排水は学習スレッドと RunManager が無ければ何もせず、Train の状態の observer の切り離しと EvalPanel の切り離しは、`Initialize` で登録したものがあるときだけ RunManager を使う。保存に要るのは RunManager の Agent だけである | [RunnerApp.cpp:634-710](../../apps/runner/src/RunnerApp.cpp)、[RunnerFrame.cpp:544-551](../../apps/runner/src/RunnerFrame.cpp)、[EvalPanel.cpp:84-101](../../apps/runner/src/EvalPanel.cpp) |
| 12 | RunManager は、学習用 env か Agent を作れないと、`Failed to create env.` か `Failed to create agent.` を出して途中で戻る。状態が RUNNING になるのは構築の最後だけで、途中で戻ると Agent（env のときは学習 Runner も）が無い。この状態で close の保存（Agent の保存）と、ループのあとの学習 Runner の後始末（`OnRun`）へ進むと、無いものを使って落ちる。今も、`Failed to initialize RunManager.` のあと人が閉じれば踏む | [trainer.cpp:1018-1023](../../core/anet-core/src/trainer.cpp)、:1041-1044、:1177-1178、[RunnerApp.cpp:596-606](../../apps/runner/src/RunnerApp.cpp)、:716-719 |
| 13 | wx（3.3.3）は OS のログオフ・シャットダウンの問い合わせを、`wxApp::OnQueryEndSession` で全トップレベルウィンドウへの拒否できる閉じる要求として届け、1 つでも拒否されると OS の終了そのものを止める。Frame には ✕ と同じ形の閉じる要求として届くので、Frame の側では見分けられない。終了が決まったあとの通知は、`wxApp::OnEndSession` が `CallOnExit()` を呼ぶ。その中で、まず `DoDelayedCleanup()` から `DeleteAllTLWs()` がトップレベルのウィンドウ（RunnerFrame とその子のパネル）を閉じる要求を経ずに削除し、そのあとで終了処理（`OnExit`）を呼ぶ。最後に `OnExit` の戻り値で `exit()` する。close 手順（§1.4 の 7）は通らない。RunnerApp はどちらも上書きしていない | wxWidgets 3.3.3 の `src/msw/app.cpp` の 727-768 行、`src/common/appbase.cpp` の 303-310 行、`src/common/appcmn.cpp` の 129-146 行（repo の外。2026-10-31 に実装側とレビューが見つけ、起票側で読んで確かめた）、[RunnerApp.hpp](../../apps/runner/src/RunnerApp.hpp) に該当の上書きが無い |

## 2. ゴールと非ゴール

**解く問題:** Runner の起動で Frame を表示したあと、RunManager の構築がメインスレッドで約 11 秒走り、その間は画面が描かれずに乱れ、「応答なし」になること。

**ゴール**

- G1: 起動直後からウィンドウ全体が描かれ、動かしても、重ねても、大きさを変えても乱れない。「応答なし」にならない。
- G2: 構築中のログが、LogPanel と Run ログファイルへ、書き出しの周期（`app.log_flush_interval_ms`）程度の遅れで流れる。
- G3: 初期化中に閉じる操作やメニュー・ツールバーの操作をしても、落ちたり状態が壊れたりしない。
- G4: 構築の中で例外が出て止まったときの見え方を、今と同じにする。online ではエラーダイアログ、batchrun では stderr と Run ログに出て、終了コードは 0 以外になる。ただし、初期化中に OS の終了の確定の通知が届いた道（実際には起きない。D10）では、エラーの表示を省き、終了コードだけを 0 以外にする。RunManager が例外を出さずに RUNNING でない状態で戻るとき（§1.4 の 12）も、今と同じく `Failed to initialize RunManager.` をログに出してウィンドウを残す（D4。エラーダイアログは出さず、人が閉じれば終了コードは 0）。
- G5: 学習の結果を変えない。同じ seed・決定的モードで、変更前後の metrics が一致する。

**非ゴール**

- NG1: 構築そのものを速くする（env 生成の並列化、CUDA の初期化の前倒しなど）。本当の制約ではなく、別の関心なので外す。
- NG2: 構築を途中で止める口。RunManager の構築には途中で取り消せる点が無く、足すと core 側の構築の契約を変えることになる。本当の制約ではない（クリーンブレークで足せる）が、閉じる要求は構築の終わりを待てば満たせる（D7・D8）ので外す。
- NG3: 進み具合の表示（進捗バー、スプラッシュ画面）。構築中のログが見えればゴールに足りる。本当の制約ではない。
- NG4: 表示より前の段（§1.1 の順 1〜3）の順序。§1.2 のとおり長くても約 1.07 秒で、「応答なし」の 5 秒に遠く、乱れの原因ではない。本当の制約ではない。
- NG5: [071](071_uncaught_fatal_visibility_10prd.md) の範囲（PinnedThreadPool の worker の例外、`set_terminate`、エラーログの write-through）。別の PRD の関心で、制約ではない。

## 3. 成功の測り方

- G1: 実機の Atari5 構成で、表示から `Run initialized.` が出るまでの間にウィンドウを動かし、ほかのウィンドウを重ねて外しても乱れが残らず、タイトルに「(応答なし)」が出ない（§7 の受入 6）。構築を遅らせたテストで、構築中もメインスレッドのタイマーが動く（受入 3）。
- G2: 構築を遅らせたテストで、構築中に出したログが、初期化が終わる前に LogPanel に出る（受入 3）。実機の Run ログで、`Initializing run in background.` と `Run initialized.` の間に構築のログが並ぶ（受入 6）。
- G3: 初期化中の閉じる操作・OS の終了の問い合わせ・メニューの操作で落ちず、閉じる要求は構築が終わってから閉じる。RunManager が RUNNING にならない設定で閉じても落ちない（受入 3・4）。
- G4: RunManager の構築の中で例外が出る設定で、終了コードが 1 になり、エラーが表示方針どおりに出る（受入 4・5・6）。RUNNING にならない設定では、今と同じくウィンドウが残り、閉じても落ちない（受入 4）。
- G5: 4 環境の短い Run で、変更前後の scalar の点が完全に一致する（受入 7）。
- 起動にかかる時間を、Run ログの `Run initialized. elapsed_sec=` で読めるようになる。

## 4. 確定事項

| # | 決定 | 理由 |
|---|---|---|
| D1 | OnInit は、Frame を表示して `RunnerApp config=` を記録したあと（§1.1 の順 3 まで）、RunManager の構築だけを別スレッド（構築スレッド）で始めて、すぐに true を返す。構築スレッドを作れないときは、OnInit の中の失敗として今と同じ経路で終わる | G1・G2。メインスレッドのイベントループを構築の間も回すため。§1.2 のとおり、乱れの時間のほぼ全部が RunManager の構築である。構築スレッドで動かしてよいのは、構築の中にメインスレッドを待つ処理が無く、InitRL の設定がプロセス全体に効き、スレッドごとの torch の状態（autocast）が使う側で元に戻るため（§1.4 の 8）。この前提が崩れると D1 は成り立たない |
| D2 | 構築が終わったら、構築スレッドは結果（構築した RunManager か、失敗）をメインスレッドへ知らせて終わる。メインスレッドは結果を 1 回だけ受け取り、RunManager を持つのもメインスレッドだけにする。D8 で先に受け取ったあとに届いた知らせは何もしない。受け取ったら構築スレッドを片付け、閉じる要求（D7）が無ければ、残り（DefaultViewFactory、InitTrainer、ImageProviderManager、`frame_->Initialize`、`Notifier:` の一覧のログ、学習スレッドの生成と開始）を今の順で行う（初期化の後半）。EvalPanel 用 EvalRunner の生成も後半に残す | GUI の部品（View、パネル）はメインスレッドで作る必要がある。RunManager はメインスレッドの close 手順と排水が読むので、持ち主をメインスレッドに限る。結果の受け口が知らせと D8 の 2 つあるので、二重に受け取らないようにする。EvalPanel 用 EvalRunner の生成は §1.2 で約 0.38 秒と「応答なし」の 5 秒から遠く、後半に残せば `frame_->Initialize` の受け渡しを変えずに済む（§6 の 1） |
| D3 | 構築スレッドで出た例外は、種類によらずすべて構築スレッドの中で捕まえ、学習スレッドと同じ経路（§1.4 の 2）でメインスレッドへ渡して投げ直す。fatal の印、エラーの表示、終了コード 1 は既存の経路に任せ、新しい表示の経路を作らない | G4。今の学習スレッドの失敗と同じ見え方にそろえる。捕まえ漏らすと、プロセスが痕跡なく落ちる（071 の関心）。終了コードは 1 になり、launcher の判定（§1.4 の 4）は変わらない |
| D4 | 構築した RunManager の状態が RUNNING でないときは、今と同じく `Failed to initialize RunManager.` を出し、後半を行わず、学習を始めずにウィンドウを残す。結果を受け取った時点で初期化中は終わり（D7）、そのあとの ✕・`File > Exit` は今の close 手順で閉じる。結果を受け取る前に閉じる要求（D7）があれば、受け取ったらそのまま閉じる。RUNNING でない RunManager では、close 手順の保存を行わず、ループのあとの学習 Runner の後始末も行わない。メニューは `frame_->Initialize` をしないので、今と同じく無効のまま残る | G3・G4。表示と学習を始めない振る舞いは今のまま（§1.4 の 5。G4 の「今と同じ」）。保存と後始末は、無い Agent と学習 Runner を使って落ちる（§1.4 の 12）。今も人が閉じれば踏む経路だが、本 PRD の D7・D8 では閉じる要求があれば自動で踏むので、ここで塞ぐ |
| D5 | 後半で出た例外は、メインループ中の例外として既存の fatal 経路（§1.4 の 2）に入る | 後半はイベントの処理の中で動くので、既存の経路がそのまま効く。新しい経路を作らない |
| D6 | 初期化が済む（D9）までのメニュー・ツールバーの無効化は、既存の仕組み（§1.4 の 6）を使う。今は無効にしていない HeatMap のメニューを同じ仕組みに入れ、初期化が済むまで無効にする。HeatMap のハンドラも、Eval の切替などと同じく初期化が済んだかを見て、済んでいなければ何もしない。初期化前のパネルは、RunManager が無くても描画・タイマー・サイズ変更・pane の開閉で落ちない | G3。本 PRD で初期化前にもイベントループが回るようになり、今まで動いたことのない初期化前の経路が動く（§1.4 の 10）。HeatMap は RunManager を使うので、ほかのメニューと同じ流儀（無効にし、ハンドラでも確かめる）で塞ぐ。メニューのイベントは Update UI の無効を通らずに届きうるので、ハンドラ側でも確かめる。後半の `frame_->Initialize` が済んだ時点で有効になる |
| D7 | 初期化中は、構築スレッドを始めてから、メインスレッドが構築の結果を受け取るまでとする。後半は結果を受け取った 1 回のイベントの処理の中で続けて行うので、後半の間に閉じる操作のイベントは処理されない。初期化中に拒否できる閉じる操作（✕、`File > Exit`）が来たら、拒否して要求として覚え、ログに `Close requested during initialization. The window will close when RunManager construction finishes.` を 1 回だけ出す（2 回目以降の操作は同じ要求として扱い、ログを出さない）。構築が終わって結果を受け取ったら、後半を行わず、学習も始めずに、今の close 手順（[160 §7.1](../design/160_applications_and_tools.jp.md)）で閉じる。保存は、RUNNING なら `app.save_agent_on_close` に従い（学習前の Agent が保存される）、RUNNING でなければ行わない（D4）。構築が失敗したときは D3 の経路で終わる。後半を行わないので `Run initialized.` は出ない | G3。初期化中の閉じる操作を受け付けて終わってから閉じるのは、ユーザーの裁定（2026-10-31）。閉じたいという操作を 1 回で叶え、後始末は今の close 手順をそのまま通す。後半を行わないのも、Fable の検討のあとのユーザーの裁定（2026-10-31）で、今の close 手順は後半が無くても通り（§1.4 の 11）、閉じる道を D8 と共通の 1 本にでき、閉じたいだけの利用者に後半の仕事（EvalPanel 用 EvalRunner の生成、評価 step）を挟まない。構築を途中で止めない（NG2）ことに依存する |
| D8 | 初期化中に、拒否できない閉じる操作（`Close(true)`）か、OS のログオフ・シャットダウンの問い合わせ（§1.4 の 13）が来たら、拒否せず、構築スレッドが終わるのをその場で待つ。問い合わせは Frame には ✕ と同じ形で届き、Frame の側では見分けられないので、RunnerApp が Frame の閉じる操作（D7 で拒否される）とは分けて、OS の終了の問い合わせとして受ける（wx の `OnQueryEndSession` の上書き）。受け取った結果が成功なら、D7 と同じく後半を行わずに今の close 手順で閉じる（保存も D7 と同じ）。失敗なら、その場で fatal として扱い（表示方針どおりのエラー表示、fatal の印、終了コード 1）、保存を行わずに閉じる。待つ間は描画が止まり、Windows が「終了を妨げているアプリ」の画面で強制終了を選ばせうる。強制終了されたときは Run ログの末尾が欠けうることを許容する。OS の終了が決まったあとの通知（§1.4 の 13）は本項の対象にせず、wx の扱いのまま終了処理へ進む（D10） | 構築を止める口が無く（NG2）、待たずに後始末すると、構築スレッドが MetricsLogger とログを使っている最中に落ちる。OS の終了の問い合わせを拒否しないのは、ユーザーの裁定（2026-10-31）で、✕ と同じく閉じたい操作を 1 回で叶える（拒否すると OS の終了が止まり、利用者がもう一度シャットダウンする必要がある）。失敗をその場で扱うのは G4 のためで、投げ直しのイベントに任せると、Frame が先に消えてイベントが処理されず、終了コードが 0 になりうる。構築の中にメインスレッドを待つ処理が無い（§1.4 の 8）ので、メインスレッドが待っても行き詰まらない。問い合わせを RunnerApp で受けるのは、Frame の閉じる操作の中では ✕ と区別できず、D7 のまま拒否すると OS の終了を止めてしまうため（§1.4 の 13） |
| D9 | RunnerApp は「初期化の後半まで済んだか」を答える。構築中、構築の失敗で終わる途中、D7・D8 で閉じる途中、RUNNING でないとき（D4。初期化中は終わっているが、後半は行っていない）は「済んでいない」と答える | 実 RunnerApp を起動する GUI テスト（§1.4 の 9）が、RunManager を使う前に初期化を待てるようにするため。受入 3・4 の確かめにも使う |
| D10 | 終了処理（`OnExit`）は、構築スレッドが残っていれば終わるのを待ってから、MetricsLogger とログの片付けへ進む。OS の終了が決まったあとの通知で wx が終了処理を呼んだとき（§1.4 の 13）も同じである。この道では、wx が Frame とパネルを閉じる要求を経ずに削除したあとで終了処理が呼ばれ、その中で構築スレッドを待つ。構築スレッドは GUI の部品に触れない（D2）ので、Frame が先に消えても構築は続けられる。この道では保存とエラーの表示を行わず、構築が失敗で終わっていれば終了処理は 1 を返し、終了コードを 0 以外にする（G4）。メインスレッドが構築スレッドの終わりを待つのは、D2 の片付け（構築は終わっているのですぐ戻る）、D8、本項の 3 か所だけにする。ループのあとの処理（`OnRun`）は、まだ受け取っていない構築の結果に触らない | 構築スレッドは MetricsLogger とログを使うので、片付けの前に終わっている必要がある。それ以外の場所で待つと、構築の間メインスレッドが止まり、G1 に反する。`OnRun` のループのあとの処理は `OnExit` より先に走るので、受け取っていない結果に触ると片付けの順が崩れる。OS の終了の確定に待機と Close の道を足さないのは、ユーザーの裁定（2026-10-31、実装側の指摘のあと）で、Windows は確定の前に必ず問い合わせを送り、D8 がその中で構築の終わりを待って Frame を閉じるので、確定の時点で初期化中ということは実際には起きないため。D8 の待ちの途中で利用者が強制終了を選べば、確定の通知は処理されずにプロセスが終わる。確定のあとは Windows がすぐプロセスを終えるので、Close を通す道は壊れやすく、保存とエラーの表示を見る人もいない。終了コードは、wx が終了処理の戻り値で `exit()` するので、戻り値を 1 にするだけで G4 を保てる |
| D11 | 構築スレッドの名前は `RunInitThread` とする。プロファイラとログに出るスレッド名である | 学習スレッドの `TrainThread`（[RunnerApp.cpp:436-437](../../apps/runner/src/RunnerApp.cpp)）と同じ流儀で、Tracy で構築の区間を見分けられる |
| D12 | ログ（英語、INFO）を 3 つ足す。構築スレッドを始める直前に `Initializing run in background.`。後半が終わったとき（学習スレッドを開始する直前）に `Run initialized. elapsed_sec=<秒>`。秒は `Initializing run in background.` を出した時点からの実時間で、小数点以下 2 桁で書く（例 `elapsed_sec=11.48`）。閉じる要求の文言は D7。D7・D8 で後半を行わずに閉じるときと、D4 のときは `Run initialized.` を出さない | 起動にかかる時間を Run ログで追え、閉じる要求を受けたことを利用者に見せる。秒の桁はユーザーの指定（2026-10-31）。後半を行わないときに出すと、初期化が済んだと読み違える |
| D13 | 設計文書を同じ変更の中で直す。[160](../design/160_applications_and_tools.jp.md) §5.1 の起動の図と本文を、構築スレッドと後半の流れにする。160 §7.1 に、初期化中の閉じる操作（D7）、OS の終了（D8）、構築の失敗の扱い（D3・D8）、RUNNING でないときは保存を行わないこと（D4）、`OnExit` が構築スレッドの終わりを待つことと、OS の終了の確定の道では `OnExit` の戻り値（構築が失敗で終わっていれば 1）が終了コードになること（D10）を足す。あわせて、160 §7.1 の「`OnExit()` は cleanup 専用」を、確定の道の戻り値を含む形に直す。[140](../design/140_observability.jp.md) §5.2 の Run 終了時の図の保存の条件に、RunManager が RUNNING であることを足す。[020](../design/020_user_guide_run.jp.md) の Close の段落（§6.4）に、初期化中に閉じたときとログオフしたときの振る舞いと、RunManager を初期化できなかった Run では保存を省くことを足す | 現行ドキュメントを同じ変更で新しい流れへ移す（[AGENTS.md](../../AGENTS.md) のクリーンブレーク方針）。保存の条件は、今は `app.save_agent_on_close` だけで書かれており、D4 で合わなくなる。「`OnExit()` は cleanup 専用」は、D10 で確定の道の終了コードを `OnExit` の戻り値で決めるので合わなくなる。D1〜D10 に依存する |

新しい ADR は起こさない。D1〜D13 は、起動の中の処理をどのスレッドで行うかと、初期化中の UI の扱いであり、後から元の形へ戻しやすい。ADR の 3 条件（戻しにくい、文脈なしでは意外、本当のトレードオフ）のうち「戻しにくい」を満たさない。`CONTEXT.md` も変えない（新しいドメインの語を足さない）。

### 4.1 退けた案

| 案 | 退けた理由 |
|---|---|
| Frame の表示を OnInit の最後（学習スレッドの開始の直前）へ移す | 変更は 1 行と文書で済み、乱れも「応答なし」も出ず、初期化中に操作する UI も無いので G1・G3〜G5 は満たす。Run ログの流れ方も今と同じ。しかし約 11 秒ウィンドウが出ず、構築中のログは LogPanel では表示したときにまとめて見えるだけで、G2 を満たせない。D1〜D12 の仕組みは、初期化中もウィンドウを生かしてログを見せることの代価である |
| 表示の直後に 1 回だけ同期で描かせ（`Update()`）、残りを `CallAfter` で後回しにする | 最初の 1 枚は正しく描けるが、後回しにした処理の中でも 5 秒を超えてイベントループが止まり、「応答なし」になる。動かしたり重ねたりすると再び乱れる。G1 を満たせない |
| 起動中に少しずつメッセージを処理させる（ログを出すたびに `wxSafeYield` など） | ログが出ない区間（§1.2 の最初のネットワーク構築の約 5 秒）で「応答なし」になる。処理の途中で別の処理が割り込む危険が増える |
| 構築スレッドを使い、初期化中は Frame を不活性にする（`Enable(false)`） | 初期化中のメニュー・閉じる操作の個別の扱い（D6・D7）が要らなくなるが、不活性の窓は ✕ も押せず動かせもしない。G1 の「動かしても」と、閉じる操作を受け付ける裁定（D7）に反する |
| 初期化中は閉じる操作を拒否する | 終わってから押し直す必要がある。ユーザーは D7 を選んだ（2026-10-31） |
| 初期化中の閉じる操作で、構築を待たずに閉じる | 構築スレッドが MetricsLogger とログを使っている最中に後始末が走り、落ちるかログが欠ける |
| 初期化中に閉じる要求があったとき、後半を済ませてから閉じる | 閉じる道が D8 と分かれて 2 本になり、閉じたいだけの利用者に後半の仕事を挟む。今の close 手順は後半が無くても通る（§1.4 の 11）。ユーザーは D7 の形を選んだ（2026-10-31） |
| 初期化中の OS のログオフ・シャットダウンを、✕ と同じく拒否する | OS の終了が止まり、利用者がもう一度シャットダウンする必要がある。ユーザーは D8 を選んだ（2026-10-31） |

## 5. 実装契約

### 5.1 起動の流れ

| 段 | スレッド | 処理 |
|---|---|---|
| 前半 | メイン | §1.1 の順 1〜3 を今のまま行う。`Initializing run in background.` を出して構築スレッドを始め、OnInit から true を返す |
| 構築 | `RunInitThread` | RunManager を構築する。成功なら RunManager を、失敗なら例外を、メインスレッドへ知らせて終わる（D2・D3） |
| 後半 | メイン | 知らせを受けたら結果を 1 回だけ受け取り、構築スレッドを片付ける。失敗なら D3 の経路で投げ直す。閉じる要求があれば、後半を行わずに close 手順へ（D7）。RUNNING でなければ D4。そうでなければ §1.1 の順 5〜7 を今の順で行い、`Run initialized. elapsed_sec=<秒>` を出してから学習スレッドを開始する（`app.train_auto_start=false` なら今と同じく一時停止の状態で開始） |

### 5.2 初期化中と初期化後の操作

初期化中は、構築スレッドを始めてから構築の結果を受け取るまで（D7）。RunManager が RUNNING でなかったとき（D4）は、結果を受け取ったあとを初期化後として扱う。

| 操作 | 初期化中 | 初期化後（RUNNING でないときを含む） |
|---|---|---|
| ✕・`File > Exit` | 拒否して要求として覚え、ログを 1 回出す。構築が終わったら後半を行わずに閉じる（D7） | 今と同じ（RUNNING でないときは保存を行わない。D4） |
| OS のログオフ・シャットダウンの問い合わせ | RunnerApp が受け、拒否せず、構築の終わりを待ってから閉じる（D8） | 今と同じ（RUNNING でないときは保存を行わない。D4） |
| 拒否できない閉じ方（`Close(true)`） | 構築の終わりを待ってから閉じる（D8） | 今と同じ（RUNNING でないときは保存を行わない。D4） |
| OS の終了の確定 | wx が Frame とパネルを閉じる要求を経ずに削除してから終了処理を呼ぶ。終了処理は構築スレッドの終わりを待ってから片付ける。保存とエラーの表示は行わない（D10）。問い合わせの段で構築を待つので、実際には起きない | 今と同じ（wx が Frame を削除してから終了処理を呼ぶ） |
| Train の切替 | 無効（学習スレッドが無い） | 今と同じ |
| Eval の切替・Eval Step・Save・Train FPS・Eval FPS・Eval Steps per Frame | 無効（既存の Update UI） | 今と同じ（RUNNING でないときは無効のまま） |
| HeatMap | 無効にし、ハンドラでも何もしない（D6 で足す） | 今と同じ（RUNNING でないときは無効のまま） |
| Conv2d | 今と同じく、押されても何もしない（EvalPanel 用 EvalRunner が無い） | 今と同じ |
| pane の開閉（Logs・Train View・Eval View・Q-Values・Trace）、Reset Layout、Always on Top、Open Run Folder | 使える。初期化前のパネルは中身が空のまま表示される（D6） | 今と同じ |
| Env の View へのマウス・キー | View がまだ無いので届かない | 今と同じ |

### 5.3 失敗の扱い

| 場面 | 振る舞い | 終了コード |
|---|---|---|
| 構築スレッドを作れない | OnInit の中の失敗として今と同じ経路（D1） | 今と同じ（OnInit の外へ抜けた例外を受けた wx の入口が決める値） |
| 構築の中の例外 | 構築スレッドで捕まえてメインスレッドで投げ直し、fatal の印、表示方針どおりのエラー表示、ループの終了（D3） | 1（今は、OnInit の外へ抜けた例外を受けた wx の入口が決める値） |
| D8 で待って受け取った結果が失敗 | その場で fatal として扱い、保存を行わずに閉じる（D8） | 1 |
| RunManager が RUNNING でない | `Failed to initialize RunManager.`、ウィンドウを残す。閉じるときは保存と学習 Runner の後始末を行わない（D4） | 今と同じ（人が閉じれば 0） |
| 後半の例外 | メインループ中の例外として fatal 経路（D5） | 1 |
| 初期化中に OS の終了の確定が届き、構築が失敗していた | 保存もエラーの表示も行わず、終了処理が構築スレッドを待ってから終える（D10。実際には起きない） | 1（終了処理の戻り値。wx がその値で `exit()` する） |

### 5.4 移行の対象

- コード: [RunnerApp.cpp](../../apps/runner/src/RunnerApp.cpp)・[RunnerApp.hpp](../../apps/runner/src/RunnerApp.hpp)（OnInit を前半・構築・後半に分ける、結果の受け取り、初期化が済んだかを答える口、閉じる要求の受け口、OS の終了の問い合わせの初期化中の扱い、RUNNING でないときの保存と `OnRun` の後始末の省略、`OnExit` での待ち）、[RunnerFrame.cpp](../../apps/runner/src/RunnerFrame.cpp)・[RunnerFrame.hpp](../../apps/runner/src/RunnerFrame.hpp)（`OnClose` の初期化中の扱い、HeatMap のメニューとハンドラ）。初期化前の経路で落ちるパネルが見つかれば、そのパネルも直す（D6）。
- テスト: [EvalTracePanel_test.cpp](../../apps/runner/src/EvalTracePanel_test.cpp) の `[eval_steps_menu]`（:876-）と `[prd105_ui_keys]`（:1044-）は、`OnInit()` のあと、イベントループを回しながら初期化が済むのを待ってから RunManager と Frame の子を使う（§6 の 6）。新しいテスト `[.][prd109_startup]`（§7 の受入 3・4）と、構築を遅らせるテスト用の Agent を足す。
- 文書: D13 のとおり、[160](../design/160_applications_and_tools.jp.md) §5.1・§7.1、[140](../design/140_observability.jp.md) §5.2 の図、[020](../design/020_user_guide_run.jp.md) §6.4 の Close の段落。
- wx の OS の終了の届き方: §1.4 の 13 は wxWidgets 3.3.3 のソースで確かめた（2026-10-31）。使う wx の版を変えるときは確かめ直す。確定の道では LogPanel を含む Frame が `OnExit` より先に削除されるので、`OnExit` の中のログの片付けで、削除済みの LogPanel へ書き出さないことを実装で確かめる（初期化後の今の道でも同じ順になる）。受入 4 は RunnerApp へ問い合わせを直接送るので、wx が OS の通知をどう届けるかは受入では確かめられない。
- 互換の扱い: 起動を同期に戻す設定や切替は作らない。
- 直さないもの: 過去の Run の記録と、[PRD105 の検証資材](../../core/anet-core/testdata/prd105/README.md)の記録。
- 実装の参考（決めるのは実装計画）: core の `ThreadBase`（[thread.cpp:25-68](../../core/anet-core/src/thread.cpp)）は、スレッド名と、種類によらない例外の捕捉と、例外の受け口を持ち、学習スレッドの例外の経路も同じ形である。構築スレッドはこれに乗せられる。

### 5.5 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `RunnerApp` | 変更 | 起動を前半・構築スレッド・後半に分け、構築の結果を 1 回だけ受け取って後半か close 手順へ進む。初期化中の閉じる要求を覚え、OS の終了の問い合わせを Frame の閉じる操作と分けて受けて（`OnQueryEndSession` の上書き）、初期化中は待ってから閉じる。初期化が済んだか（`IsRunInitialized()`）を答える。RUNNING でないときは保存と学習 Runner の後始末を行わない。終了処理の前に構築スレッドの終わりを待ち、OS の終了の確定の道で構築が失敗で終わっていれば終了処理が 1 を返す | 110 RunnerApp |
| `RunnerFrame` | 変更 | 初期化中の拒否できる閉じる操作を拒否して `RunnerApp` へ要求として渡す。拒否できない閉じ方では構築の終わりを待たせてから今の close 手順へ進む。HeatMap のメニューを初期化が済むまで無効にし、ハンドラでも確かめる | 110 RunnerApp |
| `SlowInitTestAgent` | 新規 | 設定した時間だけ構築に時間をかけ、その間にログを出す。設定で、待ったあと構築の最後に例外を投げられる（設定は §5.6）。gui-test だけで登録するテスト用の Agent | 110 RunnerApp |

### 5.6 テスト用の Agent の設定

`SlowInitTestAgent` だけが読む、gui-test の設定である。製品の設定ファイルには置かない。

| キー | 型・値域・既定 | 振る舞い | 反したとき |
|---|---|---|---|
| `agent.construction_delay_ms` | 整数、0 以上、既定 0 | 構築の初めにこの時間だけ待つ。待つ間に 200 ms ごとに 1 行、INFO のログを出す（`SlowInitTestAgent waiting. elapsed_ms=<ms>`） | 負の値や整数でない値は、構築の初めに設定の不正として例外を投げる（fail-fast） |
| `agent.construction_fails` | 真偽値、既定 false | true なら、待ったあと構築の最後に、英語のメッセージ（`SlowInitTestAgent failed by agent.construction_fails.`）で例外を投げる | 真偽値でない値は、構築の初めに設定の不正として例外を投げる |

## 6. 細部

2026-10-31 のグリルで、推奨の既定で埋めた（3 の秒の桁だけユーザーの指定で直した）。2 と 4 は、Fable の検討のあとのユーザーの裁定で改めた。

1. 構築スレッドで行う範囲: RunManager の構築だけ。EvalPanel 用 EvalRunner の生成（約 0.38 秒）、ImageProviderManager、View の生成は後半でメインスレッドに残す（理由: `frame_->Initialize` の受け渡しを変えずに済み、0.38 秒は「応答なし」の 5 秒から遠い）。
2. 閉じる要求があったとき: 構築が終わったら後半を行わず、学習も始めずに close 手順で閉じる（D7。理由: 今の close 手順は後半が無くても通り、閉じる道を D8 と共通にできる）。
3. ログの文言: D12 と D7 のとおり。`elapsed_sec` は小数点以下 2 桁（理由: 起動の所要を Run ログで追え、閉じる要求を受けたことが見える。桁はユーザーの指定）。
4. 拒否できない閉じ方と OS の終了の問い合わせ: 構築スレッドの終わりをその場で待ってから close 手順へ進む（D8。理由: 構築を止める口が無く、待たずに後始末すると落ちる。OS の終了は 1 回で通す）。
5. 構築スレッドの名前: `RunInitThread`（D11）。
6. GUI テストの待ち方: テストの側でイベントループを有効にし（gui-test は既定ではループを持たず、`Yield` ではタイマーが配られない）、初期化が済んだかを、ループを回しながら見る。120 秒を超えたら失敗にする（理由: 回しながら待たないと構築の知らせが処理されず、上限が無いと失敗したときにテストが止まらない）。
7. 自動の確かめ: gui-test に構築を数秒遅らせるテスト用の Agent を足し、構築中のメインスレッドのタイマー、完了前の LogPanel のログ、初期化中の閉じる操作と OS の終了の問い合わせを確かめる。隠しタグ `[.][prd109_startup]` で、RunnerApp を起動するケースごとに別プロセスで実行する（理由: 実 Run の長い構築は実機でしか出ないが、仕組みは短い遅延で確かめられる。RunnerApp を起動するテストは別プロセスで動かす慣行）。
8. 実機の確かめ: Atari5 構成で人が目で確かめる（受入 6。理由: OS の判定と描画は実機でしか見られない）。
9. 学習値の確かめ: PRD098 の段 0 の短縮条件で、変更前後の scalar の点を比べる（受入 7。理由: [PRD105](105_eval_trace_arrow_key_forced_action_10prd.md) で使った既存の比べ方で、構築するスレッドが変わっても学習の値が変わらないことを直接見られる）。

## 7. 受入基準

1. ビルドと既存のテスト: Debug 全体と `AnetRLRunner-gui-test` がビルドでき、`anet-core-test` と `AnetRLRunner-gui-test` の既定のスイートで、落ちるテストの集合が変更前と同じである（変更前から落ちているテストは増えも減りもしない）。
2. 移したテスト: `AnetRLRunner-gui-test "[eval_steps_menu]"` と `AnetRLRunner-gui-test "[prd105_ui_keys]"` が、それぞれ別プロセスで通る。
3. 構築中の描画とログ（`[.][prd109_startup]`、1 プロセス）: `agent.construction_delay_ms = 3000` の `SlowInitTestAgent` を使う CartPole の設定で RunnerApp を起動すると、次がすべて成り立つ。
   - `OnInit()` が true を返した時点では初期化が済んでおらず、そこから初期化が済むまでに 2 秒以上かかる。
   - 初期化が済むまでの間に、メインスレッドの 100 ms 周期のタイマーが 10 回以上発火する。
   - `SlowInitTestAgent` が構築の途中で出したログ行（`SlowInitTestAgent waiting.`）が、初期化が済む前に LogPanel のログ欄の文字列に現れる。
   - 初期化が済んだあと、Run ログに `Initializing run in background.` と `Run initialized. elapsed_sec=` の行が 1 件ずつあり、秒が小数点以下 2 桁で 3.00 以上である。
   - 構築の間に、View メニューから Logs・Train View・Eval View・Q-Values・Trace の pane を 1 回ずつ開いて閉じ、Frame の大きさを 2 回変え、HeatMap と Conv2d のメニューのイベントを 1 回ずつ送っても、プロセスが落ちず、HeatMap と Conv2d の pane が増えない。その間、HeatMap のメニューは無効である。
4. 初期化中の閉じる操作（`[.][prd109_startup]`、ケースごとに 1 プロセス。どれも `app.save_agent_on_close = true` と `app.show_error_dialog = false`（テストの中でダイアログを出して止まらないため）を使う。RUNNING にならない 2 ケースのほかは、`agent.construction_delay_ms = 3000` の `SlowInitTestAgent` を使う）:
   - 拒否できる閉じ方: 構築中に Frame へ拒否できる閉じる操作を 2 回出すと、その時点では Frame が残り、`Close requested during initialization.` で始まる行が Run ログに 1 件だけ出る。構築が終わると、学習スレッドは開始されず（学習中かの問い合わせが偽）、EvalPanel 用 EvalRunner は作られず、`Run initialized.` の行は出ずに Frame が閉じる。Run フォルダに `agent_close.anet` ができる。
   - OS の終了の問い合わせ: 構築中に RunnerApp へ OS のログオフの問い合わせ（拒否できる形）を送ると、問い合わせは拒否されず、構築が終わってから Frame が閉じ、プロセスが落ちない。
   - 拒否できない閉じ方: 構築中に `Close(true)` を出すと、構築が終わってから Frame が閉じ、プロセスが落ちない。`Run initialized.` の行は出ない。
   - 拒否できない閉じ方で構築が失敗: `agent.construction_fails = true` を足し、構築中に `Close(true)` を出すと、プロセスが落ちず、`SlowInitTestAgent failed by agent.construction_fails.` を含む英語のエラー行が Run ログに出て、Run フォルダに `agent_close.anet` ができない。
   - RUNNING にならない、構築中に閉じる: 学習用 env を作れない設定（登録されていない `env.class_id` など。Run ログに `Failed to create env.` と `Failed to initialize RunManager.` が出るもの。`SlowInitTestAgent` は使わない）で、`OnInit()` が戻った直後、イベントを処理する前に拒否できる閉じる操作を出すと（構築の結果はイベントで届くので、この操作は必ず初期化中に当たる）、Frame はその時点では残り、構築が終わってから閉じ、プロセスが落ちず、`agent_close.anet` ができない。
   - RUNNING にならない、構築のあとに閉じる: 同じ設定で、閉じる操作を出さずに `Failed to initialize RunManager.` が出るまで待つと、Frame が残る。そのあと拒否できる閉じる操作を出すと、Frame が閉じ、プロセスが落ちず、`agent_close.anet` ができない。
5. 構築の失敗: `run.eval_schedule.[nosuch].interval = 1`（RunManager の構築の中で止まる設定。[trainer.cpp:961-964](../../core/anet-core/src/trainer.cpp)）を CartPole の `--config` に足し、`app.show_error_dialog = false` で Runner の exe を起動すると、終了コードが 1 で、Run ログと `stderr.log` に `Unknown run.eval_schedule tag 'nosuch'` を含む英語のエラー行が出る。
6. 実機の確かめ（人、Atari5 構成の qbert、online）:
   - 起動してから `Run initialized.` が出るまでの間に、Frame をドラッグで動かし、別のウィンドウを Frame の上に重ねて外す。乱れが残らず、タイトルに「(応答なし)」が出ない。
   - その間、構築のログ（`BatchEnvBuilder`・`NetworkBuilder` など）が LogPanel に流れる。
   - 別の起動で、`Run initialized.` が出る前に ✕ を押すと、ログに閉じる要求の行が出て、構築が終わってから閉じる。
   - 受入 5 と同じ失敗の設定を `app.show_error_dialog = true` で起動すると、描かれた Frame の上にエラーダイアログが出る。
7. 学習値（Run は環境ごとに変更前 1 本と変更後 1 本）: PRD098 の段 0 の短縮条件（GridMaze・LunarLander・Atari qbert・DropMerge、seed=1、8 lane、batch=16、warmup=256、replay ratio=1、8,704 experience、評価停止、FP32、決定的 backend。[PRD105 の検証資材](../../core/anet-core/testdata/prd105/README.md)と同じ）で、`90_perf/`・`91_resource/`・`92_resource/` を除く全 scalar の点が、tag・step・値とも一致する。
8. 文書: D13 のとおりに直っている。160 §5.1 の図に構築スレッドと後半がある。160 §7.1 に、初期化中の閉じる操作、OS の終了、構築の失敗の扱い、RUNNING でないときは保存を行わないこと、`OnExit` が構築スレッドを待つことと確定の道では `OnExit` の戻り値が終了コードになることがあり、「`OnExit()` は cleanup 専用」の記述がそれと合っている。140 §5.2 の図の保存の条件に、RunManager が RUNNING であることがある。020 §6.4 の Close の段落に、初期化中に閉じたときとログオフしたときの振る舞いと、RunManager を初期化できなかった Run では保存を省くことがある。

## 8. 範囲外・先送り

| 項目 | 裁定 | 理由と再開条件 |
|---|---|---|
| 構築を速くする（env 生成の並列化など） | 範囲外 | NG1。再開条件: 起動の時間そのものを縮めたい要求が出たとき。`elapsed_sec` で今の所要を読める |
| 構築を途中で止める口 | 範囲外 | NG2。再開条件: 構築中に閉じたときやログオフのときに、終わりを待つのが困る例が出たとき。RunManager の構築の契約とあわせて決める |
| 進み具合の表示 | 範囲外 | NG3。再開条件: ログだけでは起動の進みが読めないと分かったとき |
| EvalPanel 用 EvalRunner も構築スレッドで作る | 先送り | 約 0.38 秒で「応答なし」に遠い（§6 の 1）。`run.eval_device` が学習側と別の device のときは、その device の初回の初期化が後半に乗りうる。再開条件: 後半の止まりが目に見えて困る例が出たとき。`frame_->Initialize` の受け渡しを変える |
| PinnedThreadPool の worker の例外、`set_terminate` | 範囲外 | NG5。[071](071_uncaught_fatal_visibility_10prd.md) の関心 |

## 9. 未決

なし。2026-10-31 のグリルと、Fable の検討のあとの質疑と、実装側の指摘のあとの質疑で次のとおり閉じた。

| # | 論点 | 閉じ方 |
|---|---|---|
| 1 | ゴールと非ゴール | G1〜G5・NG1〜NG5 で合意 |
| 2 | 足す仕組み | 構築スレッド、初期化の後半、初期化中の閉じる要求、構築の失敗の受け渡し、初期化が済んだかを答える口の 5 つで合意 |
| 3 | 初期化中の閉じる操作 | 受け付けて、構築が終わってから閉じる（D7。ユーザーの裁定） |
| 4 | 細部 9 件 | 推奨の既定で埋め、`elapsed_sec` の桁だけユーザーの指定で小数点以下 2 桁にした（§6） |
| 5 | 閉じる要求があったとき、後半を行うか | 行わない（D7。Fable の指摘を受けたユーザーの裁定） |
| 6 | 初期化中の OS のログオフ・シャットダウン | 拒否せず、構築の終わりを待って閉じる（D8。Fable の指摘を受けたユーザーの裁定） |
| 7 | 初期化中に OS の終了の確定が届いたとき | 待機と Close の道を足さず、wx の扱いのまま D10 の待ちだけを通す（実装側の指摘を受けたユーザーの裁定） |

## 10. 関連

- [160 アプリケーションとツール](../design/160_applications_and_tools.jp.md) §3.5（Logs）、§5.1（Runner 起動）、§7.1（Runner の Lifetime・エラー）。
- [020 Run 実行ガイド](../design/020_user_guide_run.jp.md) の Close の段落。
- [068](done/068_batchrun_fatal_error_handling_10prd.md)（fatal 経路と終了コード）、[071](071_uncaught_fatal_visibility_10prd.md)（捕まらない fatal の可視化）。
- [PRD105 の検証資材](../../core/anet-core/testdata/prd105/README.md)（学習値の比べ方）。

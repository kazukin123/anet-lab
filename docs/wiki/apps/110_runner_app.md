# RunnerApp（110）

## 概要・責務

- RunnerAppはconfig、RunManager、RunnerThread、GUI、loggingのlifecycleを統括し、RunnerThreadが学習RunnerをUI threadから分離する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§2.1・§5.1）。
- RunnerFrameは画面と操作を、TrainPanelは学習表示を、EvalPanelは専用EvalRunnerとmodel同期を受け持つ。QValuePanelは行動候補を可視化する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§2.1・§3.1）。
- apps/runner/toolsのOptuna harnessはRunner子processとRun artifactの境界を扱い、workspace共通基盤と補助launcherを接続する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§2.3・§3.3・§6）。

- EvalTracePanelはEvaluation Traceの行・表示履歴を保持し、EvalPanel・QValuePanel・Conv2dを選択した遷移へ追従させる（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§3.4）。

## 成立の経緯

- 2026-07-18: ImageCls batch-native env の seam：EnvRepository を single/batch variant 化＋Factory/Builder 再レイヤ（出典: [ADR 0009](../../adr/0009-imagecls-batch-env-seam.md)）。
- 2026-07-20: ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、題・目的）。
- 2026-08-10: Run 分類は物理 workspace フォルダで行い、workspace を自己完結の箱とする（出典: [ADR 0021](../../adr/0021-run-classification-by-workspace-folder.md)）。
- 2026-08-12: MLflow DB は選択 workspace の runs 配下に置く（出典: [ADR 0022](../../adr/0022-mlflow-database-lives-with-workspace-runs.md)）。
- 2026-08-12: Workspace 機構(Run 分類の物理フォルダ化と Runner/Viewer/optuna 統合)（出典: [PRD046](../../memo/done/046_workspace_10prd.md)、題・目的）。
- 2026-08-14: configured eval の定義と定期駆動を別 namespace に分離し、生成は schedule エントリが駆動する（出典: [ADR 0027](../../adr/0027-eval-definition-schedule-separation.md)）。
- 2026-08-19: PRD 055: Runner ツールバー(対象軸 4 バー構成)（出典: [PRD055](../../memo/done/055_runner_toolbar_10prd.md)、題・目的）。
- 2026-09-08: Actor 設定は名前付きカタログ参照とし、Agent インタフェースから RunMode を外す（出典: [ADR 0038](../../adr/0038-actor-config-catalog-without-runmode.md)）。
- 2026-09-19: Actor 設定カタログと eval スロットごとの方策 / network 指定（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、題・目的）。
- 2026-09-22: 終了時の背景評価セッションは完走を既定とし、設定または上限時間による協調キャンセルで抜けられるようにして、部分集計は設けない（出典: [ADR 0045](../../adr/0045-eval-session-drain-on-exit-wait-or-discard.md)）。
- 2026-09-22: 終了時の評価セッション排水（出典: [PRD076](../../memo/done/076_eval_session_drain_on_exit_10prd.md)、題・目的）。
- 2026-09-24: device 指定の auto 対応とキー体系統一（出典: [PRD070](../../memo/070_device_spec_auto_10prd.md)、題・目的）。
- 2026-10-22: Evaluation Traceの表示・操作を、内容に応じた列幅・システム色・再生／一時停止トグルを持つアイコン操作・Trace中の操作バー赤枠、低速倍率の小数表示、操作バーと行動列までを守る最小幅、行選択に依存しない列見出し配色、truncated列の短い見出しtrunc、action直後のreward列へ調整する仕様を定めた（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、D10〜D13・§8のUI受入）。
- 2026-10-26: Evaluation Traceの列番号を、百の位が分類・十の位が分類の中の順・一の位が差し込み用の3桁にし、TempoRLのdecision・skipをrewardの右に、既定のdone・truncをQ_sa・V_sの前に置いた（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、冒頭の2026-10-26の改訂・D13）。
- 2026-10-29: Evaluation TraceのTraceモードの間は矢印キーを強制行動にせず表の行の移動に使い、強制したstepを見分ける列とステータス欄の通知、列見出しでの表の並べ替え、行のダブルクリックでの通知の内容のログへのダンプを足す仕様を定めた（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、題・§2）。
- 2026-10-31: 起動中にOSのテーマ（ライト／ダーク）を切り替えたとき、通知のその場で塗らず、RunnerFrameの1か所で通知が落ち着いてから外観を1回だけ決め直し、QValueの表とEvaluation Traceの表の行の無い部分も対象にする案を暫定で起票した（出典: [999_runner_theme_switch](../../memo/999_runner_theme_switch_10prd.md)、題・§3のゴール・§4 D1・D2・D4）。
- 2026-10-31: Runnerの起動でFrameを表示したあと、RunManagerの構築だけを別スレッド（RunInitThread）で行ってメインのイベントループを止めず、構築中のログを流し、初期化中の閉じる操作は構築が終わってから後半を行わずに行う仕様を定めた（出典: [PRD109](../../memo/109_runner_startup_background_init_10prd.md)、題・§2・D1・D2・D7）。

## 選ばなかった形

- workspace選択のCancelで `_default` へ進む案 — 不採用: 選んでいないworkspaceの箱が作られる副作用がある（出典: [PRD046](../../memo/done/046_workspace_10prd.md)、§2.1）。
- Runner GUIで実行中にworkspaceを切り替える — 範囲外: 現状は再起動で切り替え、後続PRDへ分ける（出典: [PRD046](../../memo/done/046_workspace_10prd.md)、§7）。
- `RunManager` が `EpisodeEvalObserver` を直接保持して排水する案 — 不採用: 背景スレッドを持つobserver一般の終了の口を優先した（出典: [PRD076](../../memo/done/076_eval_session_drain_on_exit_10prd.md)、Complexity audit・9）。
- `RunnerApp::DrainBackgroundObservers()` で例外を捕捉する案 — 不採用: 既存の再送出契約とfatal経路でexit code 1へ到達でき、捕捉層を足さない（出典: [PRD076](../../memo/done/076_eval_session_drain_on_exit_10prd.md)、Complexity audit・11）。
- Evaluation Traceに学習側の優先度を表示する — 不採用: EvalPanelは1つのnetworkのforwardしか持たず、replayも無い（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§6）。
- Evaluation TraceでAgentの入力を表示する — 先送り: 画像Tensorの部分表示は読みにくい。再開条件: 必要と分かったとき、選択行の入力を文字列でログへ出す形から起こす（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§6）。→ PRD105 で採用（行のダブルクリックで通知の内容をログへ出す形。出るのはEnvの観測で、Actorが積み重ねた入力ではない。出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、§1.4・D9）
- Evaluation Traceの列を折れ線で表示する — 先送り: まず表と再生で足りるかを見る。再開条件: 近似優先度の推移を一目で見たくなったとき（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§6）。
- Evaluation Traceで条件に合う行を自動で一時停止する — 先送り: まず再生で場面を探す。再開条件: その手間が大きいと分かったとき（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§6）。
- Evaluation Traceで強制した行を色で目立たせる — 先送り: まず列と通知で足りるかを見る。再開条件: 列があっても強制した行を見落とす例が出たとき（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、§8）。
- 矢印キーの割り当てをEnvごとに変える — 範囲外: 操作マッピングの設定化の関心。再開条件: [999_input_mapping_joystick](../../memo/999_input_mapping_joystick_10prd.md)のグリル（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、§2のNG1・§8）。
- Evaluation Traceのpaneが開いている間はLiveモードでも矢印を行の移動にする — 不採用: 明示的にTraceモードに入る操作でモードを意識させる考え（操作バーの赤枠と同じ）と、Liveモードの矢印の操縦を残すユーザーの裁定（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、D1）。
- Traceモードの間の←→をコマ送り・コマ戻しにする — 不採用: ↑↓と同じ行へ動く入口が2つになる（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、D2）。
- 表を並べ替えても再生・コマ送りは時系列のまま進める — 不採用: 並べ替えるのは何かの軸で分析するときで、時系列にこだわる理由がないというユーザーの裁定（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、D7）。
- 並べ替えをクリックした時点の並びに留め、新しい行は先頭へ足す — 不採用: 再生が表の並びに沿うので、表が並んでいないと再生の順も意味を失う（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、D8）。
- ダンプの文字列を行を作る時点で作っておく — 不採用: 観測1つを数値の文字列にするだけで数十KB・数十msかかり、stepごとには払えない（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、D9）。
- ダンプのVERBOSEを改行を含めない1行に平らにする — 不採用: 見るのはAIかコピーして整形する人で、grepで取り出す使い方は想定しない。既存の文字列化で足りる（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、D10）。
- step traceやダンプをRunの記録・ファイルへ保存する — 範囲外: [999_eval_session_step_trace](../../memo/999_eval_session_step_trace_10prd.md)と[PRD081](../../memo/081_evalpanel_episode_trace_10prd.md)の関心（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、§2のNG8・§8）。
- Actorが積み重ねた入力をダンプに出す — 範囲外: 通知に無く、Actorの内部を読む口が要る。再開条件: 積み重ねた入力を見ないと判断を読めない例が出たとき（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、§2のNG9・§8）。
- ツールバーの状態同期に専用の wxTimer や Notifier への新しいイベントを足す形 — 不採用（行わない）: wxUpdateUIEvent による宣言的な同期を唯一の手段とし、既存の TrainPanel・EvalPanel の timer の処理後にアイドルへ戻るので、UPDATE_UI が実質 200ms 周期で回る（出典: [PRD055](../../memo/done/055_runner_toolbar_10prd.md)、0. 決定一覧 D3・2.2 同期機構）。
- 走行中の Save を UI 側で pause を強制して安全にする形 — 不採用（しない）: Pause() は ack の無いフラグで、Pipeline 構成では pause の後も LearnThread の update が走りうるので安全化にならない（出典: [PRD055](../../memo/done/055_runner_toolbar_10prd.md)、0. 決定一覧 D8・1. 現状の事実）。
- 各paneがOSのテーマの変更の通知を自分で受け、自分で遅らせて塗り直す形 — 不採用: まとめ方と遅らせ方をpaneごとに持つことになり、frameの部分（paneの見出し・ツールバー）には持ち主がいない（出典: [999_runner_theme_switch](../../memo/999_runner_theme_switch_10prd.md)、§8）。
- テーマの変更の通知のその場で塗り直し、遅らせてもう一度塗る形 — 不採用: 1回目は切替の前の色で塗りうる（出典: [999_runner_theme_switch](../../memo/999_runner_theme_switch_10prd.md)、§2の3・§8）。
- OSのテーマの切替でRunnerFrameを作り直す形 — 不採用: RunManager・各pane・Traceの状態の付け替えが大きい（出典: [999_runner_theme_switch](../../memo/999_runner_theme_switch_10prd.md)、§8）。
- テーマに合わせない固定の色（Train Viewの枠・QValueのヒートマップの地・HeatMapのstepの文字・Atariの重ね書き）をテーマに合わせる — 範囲外: 起動時から合わせていない部分で、別の関心。再開条件: 固定の色が見づらいという要求が出たとき（出典: [999_runner_theme_switch](../../memo/999_runner_theme_switch_10prd.md)、§2の7・§8）。
- Runnerの設定でライト／ダークを選ぶ — 範囲外: 別の関心。再開条件: OSと違うテーマでRunnerを使いたい要求が出たとき（出典: [999_runner_theme_switch](../../memo/999_runner_theme_switch_10prd.md)、§8）。
- wxWidgetsのソースを手元で直してテーマの切替の不具合を避ける — 範囲外: wxの版を上げるたびに当て直しが要る。再開条件: wxの不具合と分かった部分が、上流の更新でも直らないとき（出典: [999_runner_theme_switch](../../memo/999_runner_theme_switch_10prd.md)、§3の非ゴール・§8）。
- 表示のDPI・フォントの変更も、テーマの切替と同じ入口で受ける — 範囲外: 今は別の経路で追従していて、別の関心である。再開条件: DPIかフォントの変更で、半端な反映が見つかったとき（出典: [999_runner_theme_switch](../../memo/999_runner_theme_switch_10prd.md)、§3の非ゴール・§8）。
- 起動でFrameの表示をOnInitの最後へ移す — 不採用: 初期化中はウィンドウが出ず、構築中のログが見えない（出典: [PRD109](../../memo/109_runner_startup_background_init_10prd.md)、§4.1）。
- 表示の直後に1回だけ同期で描かせ、残りをCallAfterで後回しにする — 不採用: 後回しにした処理の中でも5秒を超えてイベントループが止まり「応答なし」になる（出典: [PRD109](../../memo/109_runner_startup_background_init_10prd.md)、§4.1）。
- 起動中にログを出すたびにメッセージを処理させる — 不採用: ログが出ない区間（最初のネットワーク構築の約5秒）で「応答なし」になり、処理の途中の割り込みの危険が増える（出典: [PRD109](../../memo/109_runner_startup_background_init_10prd.md)、§4.1）。
- 初期化中は閉じる操作を拒否する — 不採用: 終わってから押し直す必要があり、ユーザーは構築が終わってから閉じる形を選んだ（出典: [PRD109](../../memo/109_runner_startup_background_init_10prd.md)、§4.1・D7）。
- 初期化中の閉じる操作で構築を待たずに閉じる — 不採用: 構築スレッドがMetricsLoggerとログを使っている最中に後始末が走り、落ちるかログが欠ける（出典: [PRD109](../../memo/109_runner_startup_background_init_10prd.md)、§4.1）。
- 構築スレッドを使い、初期化中はFrameを不活性にする — 不採用: 不活性の窓は✕も押せず動かせもせず、「動かしても乱れない」ゴールと閉じる操作を受け付ける裁定に反する（出典: [PRD109](../../memo/109_runner_startup_background_init_10prd.md)、§4.1）。
- 初期化中に閉じる要求があったとき、後半を済ませてから閉じる — 不採用: 閉じる道が2本になり、閉じたいだけの利用者に後半の仕事を挟む。今のclose手順は後半が無くても通る（出典: [PRD109](../../memo/109_runner_startup_background_init_10prd.md)、§4.1・D7）。
- 初期化中のOSのログオフ・シャットダウンを✕と同じく拒否する — 不採用: OSの終了が止まり、利用者がもう一度シャットダウンする必要がある（出典: [PRD109](../../memo/109_runner_startup_background_init_10prd.md)、§4.1・D8）。
- 初期化中にOSの終了の確定が届いたとき、構築を待ってCloseの道に通す — 不採用: 確定の前に必ず問い合わせが来て、その中で構築の終わりを待ってFrameを閉じるので実際には起きず、確定のあとはWindowsがすぐプロセスを終える（出典: [PRD109](../../memo/109_runner_startup_background_init_10prd.md)、D10・§9の7）。
- EvalPanel用EvalRunnerも構築スレッドで作る — 先送り: 約0.38秒で「応答なし」に遠い。再開条件: 後半の止まりが目に見えて困る例が出たとき（出典: [PRD109](../../memo/109_runner_startup_background_init_10prd.md)、§8）。
- RunManagerの構築を途中で止める口 — 範囲外: 取り消せる点が無く、足すとcore側の構築の契約を変える。再開条件: 構築中に閉じたときに終わりを待つのが困る例が出たとき（出典: [PRD109](../../memo/109_runner_startup_background_init_10prd.md)、§2のNG2・§8）。
- 起動の進み具合の表示（進捗バー・スプラッシュ画面） — 範囲外: 構築中のログが見えればゴールに足りる。再開条件: ログだけでは起動の進みが読めないと分かったとき（出典: [PRD109](../../memo/109_runner_startup_background_init_10prd.md)、§2のNG3・§8）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- ImageCls batch入力の変更前後で、trainのmax_steps episode、episode_start、Conv2d表示のcadenceを維持する（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、§1.3）。
- close時のTrain停止、checkpoint保存、log/metricsのflush、EvalPanel detach、AUI破棄の順序を維持する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§7.1）。
- Train panelのFPSが0でも学習は続け、実行時のFPS選択をconfig dumpへ書き戻さない（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§7.1）。
- Actorを追加しても既存Actorの行動列を変えず、同一checkpoint・同一seedで全スロットの結果がEvalPanel起動中でも一致することを確かめる（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、§7・§8の受入9）。
- メトリクス配線の整理前後で、同seed・決定的モードの短いRunのmetrics checksum、scalar/traceの定義、WARN行を一致させる（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、D3・§8の受入1）。
- ActorのQヒントのヘルパー切り出し前後で、actor_approxの同seed・決定的モードの短いRunのmetrics checksumを一致させる（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§8の受入3）。
- EvalRunnerと学習側のカウンタを数える時機はそろえず、既存のmetricsのstepをずらさない（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、D9）。
- Traceモードでないとき（Liveモードの間とEvaluation Traceを閉じているとき）の矢印キーと、テンキー・QValuePanelのクリック・Ctrl・Spaceの、行動の割り当てとstepの進み方を変えない（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、§2のNG5）。
- Evaluation Traceの既定の並び（stepの降順）では、再生・コマ送り・スライダー・端の通知を今と同じにする（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、D7・§7の受入8）。
- 表に残す行の範囲（stepの新しい順にmax_rows件）を、並べ替えで変えない（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、D8）。
- EvalRunnerに強制の値を足す前後で、PRD098の検証資材の段0の短縮条件のmetrics checksumを4環境とも一致させる（出典: [PRD105](../../memo/105_eval_trace_arrow_key_forced_action_10prd.md)、§7の受入6）。
- OSのテーマの切替での外観の決め直しで、表示の状態を変えない。Evaluation TraceのLive／Traceモード・選択行・並べ替え・再生・手で変えた列幅・保持行と百分位、Logsの表示の位置・Auto-scroll・Wrap・Indent・未読の行数・未確認の警告・検索の語、paneの配置とframeの大きさ、Train・Evalの実行状態とトグルの表示である（出典: [999_runner_theme_switch](../../memo/999_runner_theme_switch_10prd.md)、§4 D6・§5.5・§7の受入3）。
- wxMSWは作ったあとのウィンドウの外観をSetAppearanceで変えられないので、テーマごとの確かめは別processで行う。RunnerAppを起動するテストのライトのprocessは、OSのアプリのモードがライトの機械でだけ作れる。開いたままのOSのテーマの切替は人が設定で切り替え、同じテーマで起動し直したスクリーンショットと領域ごとに比べる（出典: [999_runner_theme_switch](../../memo/999_runner_theme_switch_10prd.md)、§2の4・§7の受入の前の段落・受入1・4）。
- RunManagerの構築を別スレッドへ移す前後で、PRD098の段0の短縮条件（4環境、seed=1、決定的backend）の90_perf/・91_resource/・92_resource/を除く全scalarの点を、tag・step・値とも一致させる（出典: [PRD109](../../memo/109_runner_startup_background_init_10prd.md)、§7の受入7）。
- 表示より前の起動の段（workspaceの確定から設定の解決、MetricsLogger、InitRL、Env登録、Frameの生成と表示まで）の順序と、RunManagerがRUNNINGでないときにウィンドウを残して学習を始めない振る舞いを変えない（出典: [PRD109](../../memo/109_runner_startup_background_init_10prd.md)、§2のNG4・D4）。

## 関係

- 設計: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md) §2.1・§3.1・§5.1・§7.1・§8。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md) §2・§3。
- 含む機能カテゴリ: [311 開発ツール](../categories/311_dev_tools.md)。

## ファイル

点検が作り直す。手で直さない。

- [Conv2dPanel.cpp](../../../apps/runner/src/Conv2dPanel.cpp) / [Conv2dPanel.hpp](../../../apps/runner/src/Conv2dPanel.hpp)
- [ErrorDialog.cpp](../../../apps/runner/src/ErrorDialog.cpp) / [ErrorDialog.hpp](../../../apps/runner/src/ErrorDialog.hpp)
- [EvalPanel.cpp](../../../apps/runner/src/EvalPanel.cpp) / [EvalPanel.hpp](../../../apps/runner/src/EvalPanel.hpp)
- [EvalTracePanel.cpp](../../../apps/runner/src/EvalTracePanel.cpp) / [EvalTracePanel.hpp](../../../apps/runner/src/EvalTracePanel.hpp) / [EvalTracePanel_test.cpp](../../../apps/runner/src/EvalTracePanel_test.cpp)
- [HeatMapPanel.cpp](../../../apps/runner/src/HeatMapPanel.cpp) / [HeatMapPanel.hpp](../../../apps/runner/src/HeatMapPanel.hpp)
- [LogPanel.cpp](../../../apps/runner/src/LogPanel.cpp) / [LogPanel.hpp](../../../apps/runner/src/LogPanel.hpp)
- [PlotPanel.cpp](../../../apps/runner/src/PlotPanel.cpp) / [PlotPanel.hpp](../../../apps/runner/src/PlotPanel.hpp)
- [QValuePanel.cpp](../../../apps/runner/src/QValuePanel.cpp) / [QValuePanel.hpp](../../../apps/runner/src/QValuePanel.hpp)
- [RunnerApp.cpp](../../../apps/runner/src/RunnerApp.cpp) / [RunnerApp.hpp](../../../apps/runner/src/RunnerApp.hpp)
- [RunnerFrame.cpp](../../../apps/runner/src/RunnerFrame.cpp) / [RunnerFrame.hpp](../../../apps/runner/src/RunnerFrame.hpp)
- [TrainPanel.cpp](../../../apps/runner/src/TrainPanel.cpp) / [TrainPanel.hpp](../../../apps/runner/src/TrainPanel.hpp)
- [WorkspaceDialog.cpp](../../../apps/runner/src/WorkspaceDialog.cpp) / [WorkspaceDialog.hpp](../../../apps/runner/src/WorkspaceDialog.hpp)
- [pch.hpp](../../../apps/runner/src/pch.hpp)
- [runner.ico](../../../apps/runner/src/runner.ico) / [runner.manifest](../../../apps/runner/src/runner.manifest) / [runner.rc](../../../apps/runner/src/runner.rc)

### 311 開発プロセス/開発ツール/テスト基盤

- [batchrun_fatal_error_handling_test.ps1](../../../apps/runner/tools/batchrun_fatal_error_handling_test.ps1)
- [compress_workspace_metrics.py](../../../apps/runner/tools/compress_workspace_metrics.py) / [compress_workspace_metrics_test.py](../../../apps/runner/tools/compress_workspace_metrics_test.py)
- [dropmerge_optuna.py](../../../apps/runner/tools/dropmerge_optuna.py)
- [optuna_common.py](../../../apps/runner/tools/optuna_common.py)
- [optuna_metrics_gzip_test.py](../../../apps/runner/tools/optuna_metrics_gzip_test.py)
- [optuna_workspace_test.py](../../../apps/runner/tools/optuna_workspace_test.py)
- [resolve_workspace.bat](../../../apps/runner/tools/resolve_workspace.bat) / [resolve_workspace_test.ps1](../../../apps/runner/tools/resolve_workspace_test.ps1)
- [symbolize.py](../../../apps/runner/tools/symbolize.py)
- [watch_log.ps1](../../../apps/runner/tools/watch_log.ps1)

### 機能カテゴリなし

- [CMakeLists.txt](../../../apps/runner/CMakeLists.txt)
- [Atari.txt](../../../apps/runner/config/Atari.txt)
- [CartPole.txt](../../../apps/runner/config/CartPole.txt)
- [DropMerge.txt](../../../apps/runner/config/DropMerge.txt)
- [DropMerge_optuna.txt](../../../apps/runner/config/DropMerge_optuna.txt)
- [GridMaze.txt](../../../apps/runner/config/GridMaze.txt)
- [GridMaze_muzero.txt](../../../apps/runner/config/GridMaze_muzero.txt)
- [ImageCls.txt](../../../apps/runner/config/ImageCls.txt)
- [LunarLander.txt](../../../apps/runner/config/LunarLander.txt)
- [LunarLander_cpu.txt](../../../apps/runner/config/LunarLander_cpu.txt)
- [_main.txt](../../../apps/runner/config/_main.txt)
- [_workspace_template.txt](../../../apps/runner/config/_workspace_template.txt)
- [agent.txt](../../../apps/runner/config/agent.txt)
- [common.txt](../../../apps/runner/config/common.txt)
- [metrics_image.txt](../../../apps/runner/config/metrics_image.txt)
- [metrics_scalar.txt](../../../apps/runner/config/metrics_scalar.txt)
- [nn.txt](../../../apps/runner/config/nn.txt)
- [nn_cnx.txt](../../../apps/runner/config/nn_cnx.txt)
- [_main.txt](../../../apps/runner/workspaces/Atari-01/config/_main.txt)
- [_main.txt](../../../apps/runner/workspaces/CartPole-01/config/_main.txt)
- [_main.txt](../../../apps/runner/workspaces/DropMerge-01/config/_main.txt)
- [_main.txt](../../../apps/runner/workspaces/GridMaze-01/config/_main.txt)
- [_main.txt](../../../apps/runner/workspaces/GridMaze_muzero-01/config/_main.txt)
- [_main.txt](../../../apps/runner/workspaces/ImageCls-01/config/_main.txt)
- [_main.txt](../../../apps/runner/workspaces/LunarLander-01/config/_main.txt)

## PRD の一覧

点検が作り直す。手で直さない。

- [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [PRD046](../../memo/done/046_workspace_10prd.md) — Workspace 機構(Run 分類の物理フォルダ化と Runner/Viewer/optuna 統合)
- [PRD052](../../memo/done/052_eval_schedule_separation_10prd.md) — PRD 052: eval 定義とスケジュールの namespace 分離
- [PRD053](../../memo/done/053_interval_gate_perf_ema_10prd.md) — PRD 053: interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化
- [PRD057](../../memo/done/057_native_stack_dim_handling_10prd.md) — Stack 次元のネイティブ対応(Reshape ハックを廃する根本対応)
- [PRD060](../../memo/done/060_eval_batch_episodes_10prd.md) — 評価セッション契約: eval_batch_size（lane 数）と eval_episodes（本数）で N 本平均を 1 点として記録する
- [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [PRD062](../../memo/done/062_plasticity_metrics_10prd.md) — 可塑性メトリクス（srank / dormant unit 率）PRD
- [PRD063](../../memo/done/063_plasticity_weight_norm_10prd.md) — 可塑性 weight norm（feature/readout 分割）PRD
- [PRD065](../../memo/done/065_nn_spectral_norm_10prd.md) — NN Spectral Normalization（weight_norm.mode）PRD
- [PRD066](../../memo/done/066_policy_churn_metrics_10prd.md) — Policy churn メトリクス（35_agent_churn）PRD
- [PRD067](../../memo/done/067_MunchausenRL_10prd.md) — Munchausen RL（M-DQN / M-QR / M-IQN）PRD
- [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md) — メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す
- [PRD070](../../memo/070_device_spec_auto_10prd.md) — PRD 070: device 指定の auto 対応とキー体系統一
- [PRD073](../../memo/done/073_replay_fit_metrics_10prd.md) — PRD 073: replay 抽選履歴別の TD・損失メトリクス
- [PRD075](../../memo/done/075_episode_steps_and_eval_session_log_10prd.md) — PRD 075: エピソード長の汎用メトリクス化と評価セッションの実行ログ
- [PRD076](../../memo/done/076_eval_session_drain_on_exit_10prd.md) — PRD 076: 終了時の評価セッション排水
- [PRD082](../../memo/082_atari_ram_progress_metrics_10prd.md) — Atari のゲーム固有イベントを RAM メトリクスとして取る — 番号付き汎用キーへゲーム別に紐づける PRD
- [PRD083](../../memo/083_action_context_retire_10prd.md) — ActionContext の廃止（Actor が RNG と frame stacker を直接持つ）
- [PRD084](../../memo/084_noisynet_10prd.md) — NoisyNet (Noisy Networks for Exploration) PRD
- [PRD092](../../memo/092_gradient_noise_scale_10prd.md) — 勾配ノイズの大きさ（gradient noise scale）の計測 PRD
- [PRD093](../../memo/093_attention_entropy_diagnostics_10prd.md) — TransformerEncoder の attention エントロピー診断 PRD
- [PRD094](../../memo/094_evalpanel_step_history_10prd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD
- [PRD097](../../memo/097_env_auto_reset_after_notification_10prd.md) — 遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD

## 台帳の指摘

台帳のツールが作り直す。手で直さない。

<!-- registry:begin -->
- F-0018 P2 refactor — QValuePanel::Updateが356行で、統計・Grid更新・HeatMap加工を1関数で行う
- F-0024 P2 drift — nn.txtのbodyのサンプル2つ（@ResNet・@MLP_RES）が、選ぶとエラーになる
- F-0022 P3 drift — metrics_scalar.txt冒頭のscalar記法のミニ仕様が古い（&step、@learn/@trainだけ、属性の欠落）
- F-0023 P3 drift — nn.txtの対応typeの一覧とinit.modeのコメントが実装とずれている

[台帳の一覧](../../../reports/registry/findings.md)
<!-- registry:end -->

# RunnerApp（110）

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [アプリケーション](../index.md#アプリケーション)

## 概要・責務

- RunnerAppはconfig、RunManager、RunnerThread、GUI、loggingのlifecycleを統括し、RunnerThreadが学習RunnerをUI threadから分離する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§2.1・§5.1）。
- RunnerFrameは画面と操作を、TrainPanelは学習表示を、EvalPanelは専用EvalRunnerとmodel同期を受け持つ。QValuePanelは行動候補を可視化する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§2.1・§3.1）。
- apps/toolsのOptuna harnessはRunner子processとRun artifactの境界を扱い、workspace共通基盤と補助launcherを接続する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§2.3・§3.3・§6）。

- EvalTracePanelはEvaluation Traceの行・表示履歴を保持し、EvalPanel・QValuePanel・Conv2dを選択した遷移へ追従させる（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§3.4）。
- EvalTracePanelは行の並べ替え・再生・通知時点のデータのダンプを、LogPanelは警告の確認・表示・検索を受け持つ。終了時は背景評価の排水の後に状態表示用observerを外す（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§3.4・§3.5・§7.1；[可観測性](../../design/140_observability.jp.md)、§5.2）。
- Evaluation Traceのpaneを閉じた間はstep診断を休ませ、開いたときに採点用の写しが無ければEvalPanelを通してActorを同期する。表は左端の1列を固定し、ほかの列は見出しのドラッグで表示の並びを入れ替えられ、選べる行はTraceの1行だけにする（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§3.4）。
- TrainPanelのView dataは、timerが断面を取得した後の最初のTrain eventと、前回から200ms経ったTrain eventだけで作り、Train Viewを隠している間とFPSが0の間も作り続ける（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§5.1）。
- Runnerのソース（ビルドの入力）は`src/apps/runner/`に置き、Runnerは基盤、Agentの各系統、Envの公開の初期化関数を呼ぶ。実行体、設定、起動launcherは`apps/`に置き、Runnerの実行体とGUIのテストは`apps/runner/bin/<構成>/`に出る（出典: [ソース配置とライブラリの境界](../../design/011_source_layout.jp.md)、「ライブラリとアプリ」「出力の置き場」；[アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§1.3）。

## 成立の経緯

- 2026-07-18: ImageCls batch-native env の seam：EnvRepository を single/batch variant 化＋Factory/Builder 再レイヤ（出典: [ADR 0009](../../adr/0009-imagecls-batch-env-seam.md)）。
- 2026-07-20: ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter（出典: [RFD0034](../../memo/done/0034_imagecls_batch_input_10rfd.md)、題・目的）。
- 2026-08-10: Run 分類は物理 workspace フォルダで行い、workspace を自己完結の箱とする（出典: [ADR 0021](../../adr/0021-run-classification-by-workspace-folder.md)）。
- 2026-08-12: MLflow DB は選択 workspace の runs 配下に置く（出典: [ADR 0022](../../adr/0022-mlflow-database-lives-with-workspace-runs.md)）。
- 2026-08-12: Workspace 機構(Run 分類の物理フォルダ化と Runner/Viewer/optuna 統合)（出典: [RFD0046](../../memo/done/0046_workspace_10rfd.md)、題・目的）。
- 2026-08-14: configured eval の定義と定期駆動を別 namespace に分離し、生成は schedule エントリが駆動する（出典: [ADR 0027](../../adr/0027-eval-definition-schedule-separation.md)）。
- 2026-08-19: RFD0055: Runner ツールバー(対象軸 4 バー構成)（出典: [RFD0055](../../memo/done/0055_runner_toolbar_10rfd.md)、題・目的）。
- 2026-09-08: Actor 設定は名前付きカタログ参照とし、Agent インタフェースから RunMode を外す（出典: [ADR 0038](../../adr/0038-actor-config-catalog-without-runmode.md)）。
- 2026-09-19: Actor 設定カタログと eval スロットごとの方策 / network 指定（出典: [RFD0061](../../memo/done/0061_eval_slot_policy_override_10rfd.md)、題・目的）。
- 2026-09-22: 終了時の背景評価セッションは完走を既定とし、設定または上限時間による協調キャンセルで抜けられるようにして、部分集計は設けない（出典: [ADR 0045](../../adr/0045-eval-session-drain-on-exit-wait-or-discard.md)）。
- 2026-09-22: 終了時の評価セッション排水（出典: [RFD0076](../../memo/done/0076_eval_session_drain_on_exit_10rfd.md)、題・目的）。
- 2026-09-24: device 指定の auto 対応とキー体系統一（出典: [RFD0070](../../memo/done/0070_device_spec_auto_10rfd.md)、題・目的）。
- 2026-10-20: Runnerの通知とGUIが終端でもリセット前の遷移結果を読む境界へ移した（出典: [RFD0097](../../memo/done/0097_env_auto_reset_after_notification_10rfd.md)、§4.3.1〜§4.3.4・§9）。
- 2026-10-22: Evaluation Traceの表示・操作を、内容に応じた列幅・システム色・再生／一時停止トグルを持つアイコン操作・Trace中の操作バー赤枠、低速倍率の小数表示、操作バーと行動列までを守る最小幅、行選択に依存しない列見出し配色、truncated列の短い見出しtrunc、action直後のreward列へ調整する仕様を定めた（出典: [RFD0094](../../memo/done/0094_evalpanel_step_history_10rfd.md)、D10〜D13・§8のUI受入）。
- 2026-10-26: Evaluation Traceの列番号を、百の位が分類・十の位が分類の中の順・一の位が差し込み用の3桁にし、TempoRLのdecision・skipをrewardの右に、既定のdone・truncをQ_sa・V_sの前に置いた（出典: [RFD0094](../../memo/done/0094_evalpanel_step_history_10rfd.md)、冒頭の2026-10-26の改訂・D13）。
- 2026-10-29: Evaluation TraceのTraceモードの間は矢印キーを強制行動にせず表の行の移動に使い、強制したstepを見分ける列とステータス欄の通知、列見出しでの表の並べ替え、行のダブルクリックでの通知の内容のログへのダンプを足す仕様を定めた（出典: [RFD0105](../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、題・§2）。
- 2026-10-31: 起動中にOSのテーマ（ライト／ダーク）を切り替えたとき、通知のその場で塗らず、RunnerFrameの1か所で通知が落ち着いてから外観を1回だけ決め直し、QValueの表とEvaluation Traceの表の行の無い部分も対象にする案を暫定で起票した（出典: [0186_runner_theme_switch](../../memo/0186_runner_theme_switch_10rfd.md)、題・§3のゴール・§4 D1・D2・D4）。
- 2026-10-31: Runnerの起動でFrameを表示したあと、RunManagerの構築だけを別スレッド（RunInitThread）で行ってメインのイベントループを止めず、構築中のログを流し、初期化中の閉じる操作は構築が終わってから後半を行わずに行う仕様を定めた（出典: [RFD0109](../../memo/done/0109_runner_startup_background_init_10rfd.md)、題・§2・D1・D2・D7）。
- 2026-10-31: Train Viewを隠せるようにし、Centreを占めるpaneが隠れるかfloatすると表示中でdockedの相方がCentreを占めて戻れば取り戻し、主paneを隠す・floatできるのは相方の主paneが表示中でdockedのときだけ（最後の1つは✕を持たずトグルは無効で動かせない。✕とFloatableは200msのUPDATE_UIで状態から宣言し、離れてよいかの判定を基底の1つの口にまとめる）仕様と、起動時の主paneを選ぶ`app.main_view`（既定はEvalだけ）、昇格前のdock位置を覚えて戻すこと、知らない経路でCentreが空いたときにTrainを出してWARNを出す保険を定めた。2026-11-01の再グリルで、初版の「Centreを占めるpaneはfloat不可、dockedのTrainが常に占有者」をこの対称の規則に改め、floatしたpaneは縁の帯へ落とした場所に留まることにした。同日の2回目の再グリルで、wxAUIがCentreのpaneのcaptionドラッグを始めない（captionの押下をドラッグの判定の前に打ち切る）ことが実機と調査で分かり、Centreを占めるpaneはcaptionのpinボタンでfloatし、pinはFloatableと同じ値で描き、宣言が追いつく前のpinの押下は判定で止め、pinボタンを全paneに付けることにした。同日の3回目の再グリルで、等分の条件を「Evalが Right の layer 0、Trainが Centre」の1配置から「一方がCentre、他方がLeftかRightのdock」に広げ、主paneの置き場が変わったときは200msのUPDATE_UIで幅のポリシーを当て、Centreを占める主paneのcaptionの末尾に`[Centre]`を付けることにした。同日の4回目の再グリルで、縦に並んだとき（一方がCentre、他方がCentreと同じ幅のTopかBottomの帯）も高さを等分し、帯がCentreより広いときと同居は対象外、高さ不足ではLogsを縮めないことにし、「等幅」を「等分」に改めた（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、題・§1.1の6・26〜29・§1.2・§2・D1〜D3・D5・D7・D13〜D16）。
- 2026-10-31: Runner の主 pane は Centre を動的に占有し、Centre を空にする操作を導線に持たない（出典: [ADR 0066](../../adr/0066-main-panes-occupy-centre-dynamically-and-no-flow-empties-centre.md)）。
- 2026-11-02: manifestのgdiScalingを外してRunnerをPer-Monitor V2のDPI対応で動かし、画面の大きさを100%の大きさ（DIP）で書いて窓のDPIで直し、DPIの変更は画素で値を覚える部品が決め直し、Logsの字下げの幅をログ欄が描いた位置から測る案を暫定で起票した（出典: [RFD0113](../../memo/0113_runner_per_monitor_dpi_10rfd.md)、題・§3のゴール・§4 D1〜D4）。
- 2026-11-10: eval scheduleが既定で学習直後の評価を止めるのに合わせ、RunnerAppの閉じる手順のテスト（hiddenの`[runner_close_drain]`）を、最初の評価がlearn step 1へ移る形に合わせた（出典: [RFD0158](../../memo/0158_eval_schedule_first_fire_10rfd.md)、§5.4）。
- 2026-11-11: `QValuePanel::Update`を、Gridの更新、ヒートマップの加工、描画に分け、加工を画面の部品に依らない関数に切り出して単体テストを足した（出典: [RFD0122](../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§4 D10・D11・§5.2）。
- 2026-11-12: RunnerAppがアプリの側の構築（画像のmetricsの登録とRunnerFrameのpanelの初期化）を終えてTrainを始める直前に、読まれない設定キーの一覧を1回出すようにし、Viewのmenuで開くpanel（Conv2dPanel）の設定をRunの構築時に読んで検証するようにした（出典: [RFD0117](../../memo/0117_warn_policy_and_unread_config_10rfd.md)、§4 D5・§5.4）。
- 2026-11-15: RNDを選んだRun（`run.@rnd`）のEvaluation Traceに、内的報酬の3列を足した。`rnd_bonus`は235番（230 `reward`の直後）、誤差2列の`rnd_err`・`rnd_norm`は600・610番に置く（出典: [RFD0127](../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md)、§4 D12・§5.5）。
- 2026-11-15: Evaluation Traceのpaneを閉じている間はstep診断の部品を休ませ、開いたときに写しが無ければEvalPanelがその場でActorを同期するようにした（出典: [RFD0127](../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md)、§4 D20・§5.4）。
- 2026-11-17: EvalPanelがConv2dのpaneが開いている間だけstepにnn_traceの要求を渡し、RunnerAppが録画する設定（intervalが1以上）のConv2dの画像記録を登録したRunで学習側のstepに渡すようにした。Traceのダンプは要求しない（出典: [RFD0170](../../memo/0170_nn_trace_capture_unification_10rfd.md)、§4 D1・D13・D14・§5.4）。
- 2026-11-19: Runnerのソースをsrc/apps/runnerへ移し、workspacesとappdataをトップのdata/へ移した。実行体の出力先とbatの作業ディレクトリは変えていない（出典: [RFD0182](../../memo/0182_source_layout_10rfd.md)、§4.1 D11・D17・D18）。
- 2026-11-20: `QValuePanel.smooth_radius`を1以上の整数として設定の読み込みで検証し、ヒートマップの加工への入力が平滑化の指定を重み1組で持つようにした（出典: [RFD0192](../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5 D12・D13）。
- 2026-11-20: RunnerのPanelとRunnerFrameがObserverを付ける8か所に、そのObserverが動くスレッドと、PanelやFrameが先に壊れない理由をコメントで書いた（出典: [RFD0192](../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§1.2・§5 D7）。
- 2026-11-21: RunnerAppの設定の読み口から作った動画の記録で、codecのエラーが利用者の書く完全キー`app.metrics_logger.video_codec`を示すようにした（出典: [RFD0194](../../memo/0194_config_key_full_names_in_messages_10rfd.md)、§1.2のeeb2c3の行・§7のA3）。
- 2026-11-23: 文書の呼び名をRFDに改めるのに合わせ、Runnerのテストのタグを`[prdNNN…]`から`[rfdNNNN…]`へ改めた（出典: [RFD0195](../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md)、§5.2・§6 D1・D9）。

## 選ばなかった形

- workspace選択のCancelで `_default` へ進む案 — 不採用: 選んでいないworkspaceの箱が作られる副作用がある（出典: [RFD0046](../../memo/done/0046_workspace_10rfd.md)、§2.1）。
- Runner GUIで実行中にworkspaceを切り替える — 範囲外: 現状は再起動で切り替え、後続RFDへ分ける（出典: [RFD0046](../../memo/done/0046_workspace_10rfd.md)、§7）。
- `RunManager` が `EpisodeEvalObserver` を直接保持して排水する案 — 不採用: 背景スレッドを持つobserver一般の終了の口を優先した（出典: [RFD0076](../../memo/done/0076_eval_session_drain_on_exit_10rfd.md)、Complexity audit・9）。
- `RunnerApp::DrainBackgroundObservers()` で例外を捕捉する案 — 不採用: 既存の再送出契約とfatal経路でexit code 1へ到達でき、捕捉層を足さない（出典: [RFD0076](../../memo/done/0076_eval_session_drain_on_exit_10rfd.md)、Complexity audit・11）。
- Evaluation Traceに学習側の優先度を表示する — 不採用: EvalPanelは1つのnetworkのforwardしか持たず、replayも無い（出典: [RFD0094](../../memo/done/0094_evalpanel_step_history_10rfd.md)、§6）。
- Evaluation TraceでAgentの入力を表示する — 先送り: 画像Tensorの部分表示は読みにくい。再開条件: 必要と分かったとき、選択行の入力を文字列でログへ出す形から起こす（出典: [RFD0094](../../memo/done/0094_evalpanel_step_history_10rfd.md)、§6）。→ RFD0105 で採用（行のダブルクリックで通知の内容をログへ出す形。出るのはEnvの観測で、Actorが積み重ねた入力ではない。出典: [RFD0105](../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、§1.4・D9）
- Evaluation Traceの列を折れ線で表示する — 先送り: まず表と再生で足りるかを見る。再開条件: 近似優先度の推移を一目で見たくなったとき（出典: [RFD0094](../../memo/done/0094_evalpanel_step_history_10rfd.md)、§6）。
- Evaluation Traceで条件に合う行を自動で一時停止する — 先送り: まず再生で場面を探す。再開条件: その手間が大きいと分かったとき（出典: [RFD0094](../../memo/done/0094_evalpanel_step_history_10rfd.md)、§6）。
- Evaluation Traceで強制した行を色で目立たせる — 先送り: まず列と通知で足りるかを見る。再開条件: 列があっても強制した行を見落とす例が出たとき（出典: [RFD0105](../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、§8）。
- 矢印キーの割り当てをEnvごとに変える — 範囲外: 操作マッピングの設定化の関心。再開条件: [0159_input_mapping_joystick](../../memo/0159_input_mapping_joystick_10rfd.md)のグリル（出典: [RFD0105](../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、§2のNG1・§8）。
- Evaluation Traceのpaneが開いている間はLiveモードでも矢印を行の移動にする — 不採用: 明示的にTraceモードに入る操作でモードを意識させる考え（操作バーの赤枠と同じ）と、Liveモードの矢印の操縦を残すユーザーの裁定（出典: [RFD0105](../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、D1）。
- Traceモードの間の←→をコマ送り・コマ戻しにする — 不採用: ↑↓と同じ行へ動く入口が2つになる（出典: [RFD0105](../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、D2）。
- 表を並べ替えても再生・コマ送りは時系列のまま進める — 不採用: 並べ替えるのは何かの軸で分析するときで、時系列にこだわる理由がないというユーザーの裁定（出典: [RFD0105](../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、D7）。
- 並べ替えをクリックした時点の並びに留め、新しい行は先頭へ足す — 不採用: 再生が表の並びに沿うので、表が並んでいないと再生の順も意味を失う（出典: [RFD0105](../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、D8）。
- ダンプの文字列を行を作る時点で作っておく — 不採用: 観測1つを数値の文字列にするだけで数十KB・数十msかかり、stepごとには払えない（出典: [RFD0105](../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、D9）。
- ダンプのVERBOSEを改行を含めない1行に平らにする — 不採用: 見るのはAIかコピーして整形する人で、grepで取り出す使い方は想定しない。既存の文字列化で足りる（出典: [RFD0105](../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、D10）。
- step traceやダンプをRunの記録・ファイルへ保存する — 範囲外: [0184_eval_session_step_trace](../../memo/0184_eval_session_step_trace_10rfd.md)と[RFD0081](../../memo/0081_evalpanel_episode_trace_10rfd.md)の関心（出典: [RFD0105](../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、§2のNG8・§8）。
- Actorが積み重ねた入力をダンプに出す — 範囲外: 通知に無く、Actorの内部を読む口が要る。再開条件: 積み重ねた入力を見ないと判断を読めない例が出たとき（出典: [RFD0105](../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、§2のNG9・§8）。
- ツールバーの状態同期に専用の wxTimer や Notifier への新しいイベントを足す形 — 不採用（行わない）: wxUpdateUIEvent による宣言的な同期を唯一の手段とし、既存の TrainPanel・EvalPanel の timer の処理後にアイドルへ戻るので、UPDATE_UI が実質 200ms 周期で回る（出典: [RFD0055](../../memo/done/0055_runner_toolbar_10rfd.md)、0. 決定一覧 D3・2.2 同期機構）。
- 走行中の Save を UI 側で pause を強制して安全にする形 — 不採用（しない）: Pause() は ack の無いフラグで、Pipeline 構成では pause の後も LearnThread の update が走りうるので安全化にならない（出典: [RFD0055](../../memo/done/0055_runner_toolbar_10rfd.md)、0. 決定一覧 D8・1. 現状の事実）。
- 各paneがOSのテーマの変更の通知を自分で受け、自分で遅らせて塗り直す形 — 不採用: まとめ方と遅らせ方をpaneごとに持つことになり、frameの部分（paneの見出し・ツールバー）には持ち主がいない（出典: [0186_runner_theme_switch](../../memo/0186_runner_theme_switch_10rfd.md)、§8）。
- テーマの変更の通知のその場で塗り直し、遅らせてもう一度塗る形 — 不採用: 1回目は切替の前の色で塗りうる（出典: [0186_runner_theme_switch](../../memo/0186_runner_theme_switch_10rfd.md)、§2の3・§8）。
- OSのテーマの切替でRunnerFrameを作り直す形 — 不採用: RunManager・各pane・Traceの状態の付け替えが大きい（出典: [0186_runner_theme_switch](../../memo/0186_runner_theme_switch_10rfd.md)、§8）。
- テーマに合わせない固定の色（Train Viewの枠・QValueのヒートマップの地・HeatMapのstepの文字・Atariの重ね書き）をテーマに合わせる — 範囲外: 起動時から合わせていない部分で、別の関心。再開条件: 固定の色が見づらいという要求が出たとき（出典: [0186_runner_theme_switch](../../memo/0186_runner_theme_switch_10rfd.md)、§2の7・§8）。
- Runnerの設定でライト／ダークを選ぶ — 範囲外: 別の関心。再開条件: OSと違うテーマでRunnerを使いたい要求が出たとき（出典: [0186_runner_theme_switch](../../memo/0186_runner_theme_switch_10rfd.md)、§8）。
- wxWidgetsのソースを手元で直してテーマの切替の不具合を避ける — 範囲外: wxの版を上げるたびに当て直しが要る。再開条件: wxの不具合と分かった部分が、上流の更新でも直らないとき（出典: [0186_runner_theme_switch](../../memo/0186_runner_theme_switch_10rfd.md)、§3の非ゴール・§8）。
- 表示のDPI・フォントの変更も、テーマの切替と同じ入口で受ける — 範囲外: 今は別の経路で追従していて、別の関心である。再開条件: DPIかフォントの変更で、半端な反映が見つかったとき（出典: [0186_runner_theme_switch](../../memo/0186_runner_theme_switch_10rfd.md)、§3の非ゴール・§8）。
- 起動でFrameの表示をOnInitの最後へ移す — 不採用: 初期化中はウィンドウが出ず、構築中のログが見えない（出典: [RFD0109](../../memo/done/0109_runner_startup_background_init_10rfd.md)、§4.1）。
- 表示の直後に1回だけ同期で描かせ、残りをCallAfterで後回しにする — 不採用: 後回しにした処理の中でも5秒を超えてイベントループが止まり「応答なし」になる（出典: [RFD0109](../../memo/done/0109_runner_startup_background_init_10rfd.md)、§4.1）。
- 起動中にログを出すたびにメッセージを処理させる — 不採用: ログが出ない区間（最初のネットワーク構築の約5秒）で「応答なし」になり、処理の途中の割り込みの危険が増える（出典: [RFD0109](../../memo/done/0109_runner_startup_background_init_10rfd.md)、§4.1）。
- 初期化中は閉じる操作を拒否する — 不採用: 終わってから押し直す必要があり、ユーザーは構築が終わってから閉じる形を選んだ（出典: [RFD0109](../../memo/done/0109_runner_startup_background_init_10rfd.md)、§4.1・D7）。
- 初期化中の閉じる操作で構築を待たずに閉じる — 不採用: 構築スレッドがMetricsLoggerとログを使っている最中に後始末が走り、落ちるかログが欠ける（出典: [RFD0109](../../memo/done/0109_runner_startup_background_init_10rfd.md)、§4.1）。
- 構築スレッドを使い、初期化中はFrameを不活性にする — 不採用: 不活性の窓は✕も押せず動かせもせず、「動かしても乱れない」ゴールと閉じる操作を受け付ける裁定に反する（出典: [RFD0109](../../memo/done/0109_runner_startup_background_init_10rfd.md)、§4.1）。
- 初期化中に閉じる要求があったとき、後半を済ませてから閉じる — 不採用: 閉じる道が2本になり、閉じたいだけの利用者に後半の仕事を挟む。今のclose手順は後半が無くても通る（出典: [RFD0109](../../memo/done/0109_runner_startup_background_init_10rfd.md)、§4.1・D7）。
- 初期化中のOSのログオフ・シャットダウンを✕と同じく拒否する — 不採用: OSの終了が止まり、利用者がもう一度シャットダウンする必要がある（出典: [RFD0109](../../memo/done/0109_runner_startup_background_init_10rfd.md)、§4.1・D8）。
- 初期化中にOSの終了の確定が届いたとき、構築を待ってCloseの道に通す — 不採用: 確定の前に必ず問い合わせが来て、その中で構築の終わりを待ってFrameを閉じるので実際には起きず、確定のあとはWindowsがすぐプロセスを終える（出典: [RFD0109](../../memo/done/0109_runner_startup_background_init_10rfd.md)、D10・§9の7）。
- EvalPanel用EvalRunnerも構築スレッドで作る — 先送り: 約0.38秒で「応答なし」に遠い。再開条件: 後半の止まりが目に見えて困る例が出たとき（出典: [RFD0109](../../memo/done/0109_runner_startup_background_init_10rfd.md)、§8）。
- RunManagerの構築を途中で止める口 — 範囲外: 取り消せる点が無く、足すとcore側の構築の契約を変える。再開条件: 構築中に閉じたときに終わりを待つのが困る例が出たとき（出典: [RFD0109](../../memo/done/0109_runner_startup_background_init_10rfd.md)、§2のNG2・§8）。
- 起動の進み具合の表示（進捗バー・スプラッシュ画面） — 範囲外: 構築中のログが見えればゴールに足りる。再開条件: ログだけでは起動の進みが読めないと分かったとき（出典: [RFD0109](../../memo/done/0109_runner_startup_background_init_10rfd.md)、§2のNG3・§8）。
- Train ViewとEvaluation Viewを1つのCentre paneにまとめwxSplitterWindowで分ける — 不採用: Evalのfloat・単独maximize・✕を失い、表示切替がpane系とsplitter系の2系統になる（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§2のNG1・§4.1）。
- TrainとEvalの両方をCentre dockに置く — 不採用: 別のrowでは横に並ぶがsashが無く（今の50:50も遷移で比率を戻すので損は小さい）、floatから戻したEvalは側面のdockにしか落ちないので結局昇格が要り、幅の縮小・記憶・テストを組み替える割に単純にならない（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§1.1の3・6・§4.1）。
- Evalのmaximizeで足りるとする — 不採用: Logs・Q-Values・Evaluation Traceも消え、起動時に選べず、Trainの描画timerは回り続ける（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§1.1の23・§4.1）。
- Train Viewを細い帯に畳む見かけの非表示 — 不採用: 右dockは85%までで、帯が空白のまま残る（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§4.1）。
- 最後の主paneの✕を描いたまま残し、closeのVetoだけで守る — 不採用: 押しても何も起きないボタンが常に残る。✕は状態から描き分け、Vetoは見え方が追いつく前の最大200msの守りに限る（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§4.1・D3）。
- ✕とFloatableをポリシーの適用の時機だけで付け外す — 不採用: ドラッグのfloat・dockはwxAUIがイベントを出さず、EvalをfloatしたあともTrainの✕が残って押せる（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§1.1の7・§4.1）。
- wxEVT_AUI_RENDERでドラッグを監視する／wxAuiManagerを派生させてfloat・dockの瞬間を掴む — 不採用: 前者は描画のたびに走りADR 0016で全廃した方式に戻る。後者は掴める時機がUpdateの中かドラッグ中のヒント計算で、合図に使えない。派生でCentreのpaneのcaptionドラッグを始める形も、wxのprotectedの状態と内部の順に依存する歪なので採らず、Centreのpaneはcaptionのpinボタンで外す（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§4.1・D2。2026-11-01）。
- float中のEvalをTrainの✕でCentreへ引き込む／Centreが空いたら隠れている方を出すことを導線の意味にする — 不採用: アンドックは明示的な操作で、別の操作の副作用で覆さないというユーザーの裁定。知らない経路で空いたときの保険（WARN付き）としてだけ置く（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§2のNG4・§4.1・D13）。
- 主paneを常にfloat不可にする／Trainだけfloat不可にする — 不採用: 前者はTrain表示中のEvalの自由を失い、後者はEvalがCentreのときにfloatで空く。Floatableは✕と同じ「相方が表示中でdocked」の判定で決める（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§4.1・D2）。
- Centreを占めるpaneを常にfloat不可にしdockedのTrainを常に占有者にする（初版）／占有者をTrain優先のまま相方がいればfloatだけ許す — 不採用: 前者はTrainだけが表示中は常にfloatできない。後者はfloatしたTrainを縁の帯へ落とすと最大200msでCentreへ跳び、側面に置き続けられない。占有者は今いる方が保ち、floatしたpaneは落とした場所に留まる（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§1.2・§4.1・D1・D2。2026-11-01）。
- Centreの占有を先着順にして入れ替えは隠して出し直す／TrainとEvalを入れ替えるコマンドを足す／占有をTrain優先へ戻す／wxWidgetsのソースのCentreの判定を外す／Centreにダミーのpaneを置く — 不採用: 先着順は既定のevalからTrainを出すと左右が入れ替わって50:50と主領域を位置で定め直す必要があり、Centreのpaneを外す操作も無いまま。入れ替えのコマンドは導線と機構が1つ増える。Train優先は常に動かせない非対称に戻る。wxのソースはforkの保守が要る。ダミーのCentreは主paneの伸縮がwxの配分に戻る。Centreのpaneはcaptionのpinボタンで外す（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§4.1・D2。2026-11-01）。
- 主paneのpinを常に描き最後の1つでは押しても何も起きないままにする／pinの押下で相方の昇格を先回りする — 不採用: 前者はwxがFloatableでないpaneのpinを無視するので押しても何も起きないボタンが残る。後者は昇格の時機がドラッグとpinの2つになる。pinは✕と同じく状態から描き分け、押下は判定が否のときに止めるだけに使う（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§4.1・D2・D3）。
- 降格先を常に定位置にする／定位置のEvalだけを昇格の候補にする — 不採用: 前者は定位置以外に置いたEvalの位置をTrainの表示切替で失い、後者はEvalを左などに置いた人がTrainを隠す前に戻す手間が要る。昇格前の位置を覚えて戻す（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§4.1・D5）。
- Trainの描画timerをOnPaneHiding／OnPaneShowingで伝える／timerを回し続けてtickでIsShownOnScreenを見る — 不採用: 前者はReset Layoutがこの2つを通らず漏れる。後者は最小だがG3の「timerが止まる」を変え、止まっていることを外から読みにくい（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§1.1の18・§4.1・D6）。
- TrainとEvalの表示を別々の真偽値の設定にする — 不採用: 「両方非表示」の不正な組を検証で弾く必要が出る（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§4.1・D7）。
- Train Viewを隠している間にTrain側のobserverも外す — 範囲外: FPS 0でも残る既存の挙動で、AttachScopedのDetachバグの修復が先。再開条件: その修復が入り、Trainスレッド側のView更新のコストを消したい要求が出たとき（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§2のNG2・§8）。
- Eval paneを隠したときにEvalのtimerを止める — 範囲外: Trainの定期停止の増幅要因としての対策と一緒に決める。再開条件: 定期停止の対策のRFDを起こすとき（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§2のNG3・§8）。
- 等分の対象をCentreに隣接する側面のdockに限る／自動の等分をやめて明示の操作とReset Layoutだけにする／側面に落とした主paneを定位置へ引き戻す — 不採用: 隣接に限ると間にQ-Values列があると効かず判定も増える。明示の操作はTrain Viewを出したとき必ず等分になる既定の見え方とADR 0016の50:50の意味を変える。引き戻しはD1の「落とした場所に留まる」を覆す。等分は一方がCentre、他方がLeftかRightのdockのときに当てる（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§4.1・D15。2026-11-01）。
- 縁へ落とした直後の等分を次の遷移まで待つ — 不採用: ドラッグで落とす操作は遷移に当たらず、落とした直後に「効かない」と見える場面が残る。主paneの置き場が変わったときだけ200msのUPDATE_UIで当て、サッシの操作は覆さない（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§4.1・D15・§1.1の27。2026-11-01）。
- Centreの印をステータスバー・Panelsのボタン・captionの色で出す／文言を`(Centre)`・記号・中点にする — 不採用: ステータスバーは離れた配置で目線が合わず実行状態の欄に配置が混ざる。ボタンは押下表示と印の2状態が乗る。色はdock artの派生とテーマ切替が絡む。文言は見た目の差で角括弧を選んだ（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§4.1・D16。2026-11-01）。
- 縦に並んだときの高さ不足でLogsの帯を縮めて確保する／帯の幅がCentreより広いTop・Bottom（windowの縁に落とした最外層）も等分の対象にする／縦の等分の予算にLogsの帯を含める — 不採用: Logsを縮める挙動は既存に無く小さい窓でLogsが勝手に縮む。左右の列の下まで広がる帯はCentreと組になっていない。横でQ-Valuesの列を予算から除くのと同じく組になる2枚だけを等分する（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§4.1・D15。2026-11-01）。
- サッシで変えた主paneの幅の比率の記憶 — 範囲外: 別の関心。再開条件: 遷移で等分に戻る動きが邪魔だという要求が出たとき（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§2のNG7・§8）。
- TrainとEvalの左右の入れ替え（設定で定位置を選ぶ形） — 範囲外: 要求が無い。Trainをpinで外して側面へ落としEvalをCentreにすることはできる。再開条件: Evalを左に置きたい要求が出たとき（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§8・D1）。
- manifestのgdiScalingを残す（今のまま） — 不採用: DPIの宣言が上書きされてDPI非対応で動き、字幅の食い違いとぼやけが残り、manifestと実際の動きが違う（出典: [RFD0113](../../memo/0113_runner_per_monitor_dpi_10rfd.md)、§2の2〜4・§8）。
- システムのDPIだけに対応する宣言（System aware）／Per-Monitor（V1）の宣言 — 不採用: 前者は起動時の1つのDPIでしかくっきり描けず150%と175%の間で動かすとぼやけ、後者は子のウィンドウにDPIの変更が通知されない（出典: [RFD0113](../../memo/0113_runner_per_monitor_dpi_10rfd.md)、§2の3・§8）。
- gdiScalingを外すだけにし、Logsの字下げはGDIで測ったままにする — 不採用: 測る側と描く側が別の仕組みのままで、フォントの代替などで再び食い違いうる（出典: [RFD0113](../../memo/0113_runner_per_monitor_dpi_10rfd.md)、D4・§8）。
- DPIの変更をテーマの切替と同じ入口（RunnerFrameの1か所）で受ける — 範囲外: 0186_runner_theme_switchが分けると決めた。再開条件: 両方を実装したあとで、DPIの変更の反映が半端になる例が出たとき（出典: [RFD0113](../../memo/0113_runner_per_monitor_dpi_10rfd.md)、§2の8・§8）。
- アイコン・Viewの絵を高い解像度向けに描き直す — 範囲外: 別の関心。再開条件: PMv2にしたあとで、ぼやけて見える絵が見つかったとき（出典: [RFD0113](../../memo/0113_runner_per_monitor_dpi_10rfd.md)、§3の非ゴール・§8）。
- px直書きの型（大きさ・枠と間隔・余白と線の太さの6つの型）に当たらない、Viewの絵の中の長さと位置（矢印の長さ、Atariの待ちの表示の文字の位置など）をDIPにする — 範囲外: 画面の大きさのゴールは6つの型の大きさまでとした。再開条件: PMv2にしたあとで、小さく見える絵の部品が見つかったとき（出典: [RFD0113](../../memo/0113_runner_per_monitor_dpi_10rfd.md)、§2の5・§8）。
- `QValuePanel::Update`を、メンバを直接読む関数に分けるだけにする — 不採用: ヒートマップの加工を壊したときに気づく手段が人の目のままになる（出典: [RFD0122](../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§4 D10・§7）。
- Runnerを起動してQ-ValuesのGridのセルとハイライトを読むGUIのテスト — 不採用: Gridの更新は中身を変えずに関数へ移すだけで、Runnerを起動するテストの保守に見合わない（出典: [RFD0122](../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§4 D11・§7）。
- Q-Valuesのヒートマップの加工をcoreのGUI共通へ移し、既定のテストスイートで流す — 不採用: 加工を使うのはRunnerのQ-Valuesの表示だけで、テストを既定のスイートに入れるためにアプリケーションの部品をcoreへ置くことになる（出典: [RFD0122](../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§4 D11・§7）。
- QValuePanelのGridの死にコード（無効化されたAdvantageの引き算、コメントアウト）を消す — 範囲外: もとからある死にコードは消さずに報告する決まり。再開条件: 片づけを別の課題として起こすとき（出典: [RFD0122](../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§7）。
- Traceの記録中はTraceのpaneもnn_traceを要求する形 — 不採用: 1行のダンプに出るnn_traceの項目は形・型・置き場所だけで、EvalPanelの費用が今と変わらなくなる（出典: [RFD0170](../../memo/0170_nn_trace_capture_unification_10rfd.md)、§4 D1・§4.1）。
- 録画するエピソードの間だけ学習側のstepにnn_traceを要求する形 — 範囲外: 要求したときの費用を下げる話で関心が別（出典: [RFD0170](../../memo/0170_nn_trace_capture_unification_10rfd.md)、§3.2・§4 D14）。
- アプリのソースをappsに残す形 — 不採用: 置き場の規則が2つになる（出典: [RFD0182](../../memo/0182_source_layout_10rfd.md)、§4.3）。
- 設定をソースの側に置き、ビルドでappsへ写す形 — 不採用: 設定を直してすぐ走らせる回し方に、写しの手順が挟まる（出典: [RFD0182](../../memo/0182_source_layout_10rfd.md)、§4.3；[ADR 0078](../../adr/0078-top-level-layout-separates-build-inputs-runtime-layout-and-runtime-data.md)、Considered Options）。
- dataの根を探して見つける形、見つからないときの代替 — 不採用: 黙った代替を作らない（出典: [RFD0182](../../memo/0182_source_layout_10rfd.md)、§4.1 D18）。
- `smooth_radius`の0以下を「平滑化なし」として扱う形 — 不採用: 特定の値に別の意味を持たせることになり、する・しないは`use_hist_smooth`が表す（出典: [RFD0192](../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5.1）。
- Logs paneとPanels toolbarの表示を変える — 範囲外: 既にある未確認の警告の表示に意味を持たせるのがRFD0117で、表示を変える必要は見つかっていない。必要が出たら別のRFDで変える（出典: [RFD0117](../../memo/0117_warn_policy_and_unread_config_10rfd.md)、§3.2）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- ImageCls batch入力の変更前後で、trainのmax_steps episode、episode_start、Conv2d表示のcadenceを維持する（出典: [RFD0034](../../memo/done/0034_imagecls_batch_input_10rfd.md)、§1.3）。
- close時のTrain停止、checkpoint保存、log/metricsのflush、EvalPanel detach、AUI破棄の順序を維持する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§7.1）。
- Train panelのFPSが0でも学習は続け、実行時のFPS選択をconfig dumpへ書き戻さない（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§7.1）。
- Actorを追加しても既存Actorの行動列を変えず、同一checkpoint・同一seedで全スロットの結果がEvalPanel起動中でも一致することを確かめる（出典: [RFD0061](../../memo/done/0061_eval_slot_policy_override_10rfd.md)、§7・§8の受入9）。
- メトリクス配線の整理前後で、同seed・決定的モードの短いRunのmetrics checksum、scalar/traceの定義、WARN行を一致させる（出典: [RFD0094](../../memo/done/0094_evalpanel_step_history_10rfd.md)、D3・§8の受入1）。
- ActorのQヒントのヘルパー切り出し前後で、actor_approxの同seed・決定的モードの短いRunのmetrics checksumを一致させる（出典: [RFD0094](../../memo/done/0094_evalpanel_step_history_10rfd.md)、§8の受入3）。
- EvalRunnerと学習側のカウンタを数える時機はそろえず、既存のmetricsのstepをずらさない（出典: [RFD0094](../../memo/done/0094_evalpanel_step_history_10rfd.md)、D9）。
- Traceモードでないとき（Liveモードの間とEvaluation Traceを閉じているとき）の矢印キーと、テンキー・QValuePanelのクリック・Ctrl・Spaceの、行動の割り当てとstepの進み方を変えない（出典: [RFD0105](../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、§2のNG5）。
- Evaluation Traceの既定の並び（stepの降順）では、再生・コマ送り・スライダー・端の通知を今と同じにする（出典: [RFD0105](../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、D7・§7の受入8）。
- 表に残す行の範囲（stepの新しい順にmax_rows件）を、並べ替えで変えない（出典: [RFD0105](../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、D8）。
- EvalRunnerに強制の値を足す前後で、RFD0098の検証資材の段0の短縮条件のmetrics checksumを4環境とも一致させる（出典: [RFD0105](../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md)、§7の受入6）。
- OSのテーマの切替での外観の決め直しで、表示の状態を変えない。Evaluation TraceのLive／Traceモード・選択行・並べ替え・再生・手で変えた列幅・保持行と百分位、Logsの表示の位置・Auto-scroll・Wrap・Indent・未読の行数・未確認の警告・検索の語、paneの配置とframeの大きさ、Train・Evalの実行状態とトグルの表示である（出典: [0186_runner_theme_switch](../../memo/0186_runner_theme_switch_10rfd.md)、§4 D6・§5.5・§7の受入3）。
- wxMSWは作ったあとのウィンドウの外観をSetAppearanceで変えられないので、テーマごとの確かめは別processで行う。RunnerAppを起動するテストのライトのprocessは、OSのアプリのモードがライトの機械でだけ作れる。開いたままのOSのテーマの切替は人が設定で切り替え、同じテーマで起動し直したスクリーンショットと領域ごとに比べる（出典: [0186_runner_theme_switch](../../memo/0186_runner_theme_switch_10rfd.md)、§2の4・§7の受入の前の段落・受入1・4）。
- RunManagerの構築を別スレッドへ移す前後で、RFD0098の段0の短縮条件（4環境、seed=1、決定的backend）の90_perf/・91_resource/・92_resource/を除く全scalarの点を、tag・step・値とも一致させる（出典: [RFD0109](../../memo/done/0109_runner_startup_background_init_10rfd.md)、§7の受入7）。
- 表示より前の起動の段（workspaceの確定から設定の解決、MetricsLogger、InitRL、Env登録、Frameの生成と表示まで）の順序と、RunManagerがRUNNINGでないときにウィンドウを残して学習を始めない振る舞いを変えない（出典: [RFD0109](../../memo/done/0109_runner_startup_background_init_10rfd.md)、§2のNG4・D4）。
- Train Viewが表示中の既存の振る舞い（Evalのfloat・maximize・✕、Train:Eval=50:50、幅の記憶、Evalのstep・resumeでEvalを出す動き、Logs・Q-Values・補助paneの配置）を変えず、既存のGUIテスト（`[eval_steps_menu]`・`[rfd0105_ui_keys]`・`[rfd0109_ui]`）が通ることを確かめる。意図して変えるのはTrain ViewもEval Viewと同じ条件でfloatできること（Centreのpaneはcaptionのpinボタンで）、floatしたpaneは縁の帯へ落とした場所に留まること、maximize中のpaneをメニューで隠すと先に解けること、起動直後の配置がEvalだけになること、Logs以外のpaneのcaptionにもpinボタンが付くこと、等分の条件を「一方がCentre、他方がLeftかRightのdock」に広げて主paneの置き場が変わったときは200ms以内に当てることの6つ（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、§2のG5・§7の受入1）。
- 等分は、両主paneが表示中でdocked、一方がCentre、他方がLeftかRightのdock（幅）か、Centreと同じ幅のTopかBottomの帯（高さ。表示中のLeft・Rightのdockがすべてその帯より外側のlayer）のときだけ当て、帯がCentreより広いときと補助paneやLogsが同居する列・帯はwxの配分に任せる。縦の予算はCentreの高さと主paneの帯の高さの和で、Logs・toolbarは入れず縮めない。当てる時機は既存の遷移と、主paneの置き場（方向・layer・row・位置・float・表示）が変わった200msの宣言で、サッシで変えた幅は置き場が変わらないので次の遷移まで保つ。Centreを占める主paneのcaptionには`[Centre]`が付き、float中・側面・非表示の主paneとほかのpaneには付かない（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、D15・D16・§7の受入2のo〜r）。
- paneの表示切替のチェックと有効・無効の同期、✕・pin・Floatableの宣言、ドラッグやpinでfloatした直後の占有の規則の当て直しは、既存のwxUpdateUIEventの束だけで行い、専用のtimerやイベントを足さない（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、D3）。
- 最後の主paneのcaptionにはpinも✕も無く、宣言が追いつく前（最大200ms）の✕・pin・トグルは判定で止める。pinの押下はframeが先に受け取り、判定が否ならwxAUIへ渡さない（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、D3・§5.4・§6の21）。
- Train Viewを隠している間も学習とTrain側のobserverは止めず、止めるのは描画のtimerだけである（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、D6・§2のNG2）。
- 導線の操作ではRunログに保険のWARN（`No main pane occupies the centre.`と`Both main panes occupy the centre.`）が出ないことを確かめる（出典: [RFD0110](../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md)、D13・§7の受入6・8）。
- DPI対応の見直しで、100%の表示の寸法（すべて今と同じ画素のまま）・配置・振る舞い、RFD0110の主paneの規則とそのGUIテストの判定（テストの中の固定の寸法をFromDIPで直すことは判定を変えないので行ってよい）、Closeの順序、設定のキー、学習と評価を変えない（出典: [RFD0113](../../memo/0113_runner_per_monitor_dpi_10rfd.md)、§5.5）。
- `QValuePanel::Update`を分けるときは、Gridの差分更新の2つの条件（選択が変わった行だけを塗り直す、値が広がった列だけ幅を当て直す）を残し、加工の前に値を写す回数を増やさない。切り出す前のコードで控えた加工後の値と、切り出したあとの単体テストの期待値が一致することを確かめる。加工の単体テストは`AnetRLRunner-gui-test`に置き、既定のテストスイートに入らないので、加工を変えたら流す（出典: [RFD0122](../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§4 D10〜D12・§6 A8）。
- Conv2dのpaneが「開いている」の判定は、Traceが追従するpaneの判定と同じにする。別にすると、控えを取る行にnn_traceが無いという食い違いが起きる（出典: [RFD0170](../../memo/0170_nn_trace_capture_unification_10rfd.md)、§4 D13）。
- workspace と portable appdata は配布位置を基準に `data/` に置く。指定した workspace の解決と相対 Run の配置は同じ根を使う（出典: [RFD0182](../../memo/0182_source_layout_10rfd.md)、§4.1 D22・§6 のS4-2・C3）。
- `smooth_radius`の読み方を変えても、半径が1以上の設定では、平滑化あり・なしの加工の結果を今と同じ値に保ち、加工の今の単体テストが期待値を変えずに通ることを確かめる（出典: [RFD0192](../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§6.6・§7のA11）。
- 学習のRunnerの通知を受ける3つのObserver（TrainPanel、EvalPanelが学習側のstep数を受けるObserver、RunnerFrameの状態表示用のObserver）は、学習スレッドからPanelかFrameを触る。窓を閉じる処理で、学習の停止と背景評価の排水より前に、これらを外したりPanelを壊したりしない（出典: [RFD0192](../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§1.2・§5 D6）。
- WARNの線引きを変えても、ログのレベルの名前、Runのログの`[W]`・`[I]`の書式、Logs paneの未確認の警告の数え方は変えない（出典: [RFD0117](../../memo/0117_warn_policy_and_unread_config_10rfd.md)、§5.1）。

## 関係

- 設計: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md) §2.1・§3.1・§3.4・§3.5・§5.1・§7.1・§8。
- ソースと出力の置き場: [ソース配置とライブラリの境界](../../design/011_source_layout.jp.md)、「ライブラリとアプリ」「出力の置き場」。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md) §2〜§6（起動、画面、操作、Evaluation Trace）。
- 含む機能カテゴリ: なし。

## ファイル

点検が作り直す。手で直さない。

- [Conv2dPanel.cpp](../../../src/apps/runner/src/Conv2dPanel.cpp) / [Conv2dPanel.hpp](../../../src/apps/runner/src/Conv2dPanel.hpp)
- [ErrorDialog.cpp](../../../src/apps/runner/src/ErrorDialog.cpp) / [ErrorDialog.hpp](../../../src/apps/runner/src/ErrorDialog.hpp)
- [EvalPanel.cpp](../../../src/apps/runner/src/EvalPanel.cpp) / [EvalPanel.hpp](../../../src/apps/runner/src/EvalPanel.hpp)
- [EvalTracePanel.cpp](../../../src/apps/runner/src/EvalTracePanel.cpp) / [EvalTracePanel.hpp](../../../src/apps/runner/src/EvalTracePanel.hpp) / [EvalTracePanel_test.cpp](../../../src/apps/runner/src/EvalTracePanel_test.cpp)
- [HeatMapPanel.cpp](../../../src/apps/runner/src/HeatMapPanel.cpp) / [HeatMapPanel.hpp](../../../src/apps/runner/src/HeatMapPanel.hpp)
- [LogPanel.cpp](../../../src/apps/runner/src/LogPanel.cpp) / [LogPanel.hpp](../../../src/apps/runner/src/LogPanel.hpp) / [LogPanel_test.cpp](../../../src/apps/runner/src/LogPanel_test.cpp)
- [Panel_key_test.cpp](../../../src/apps/runner/src/Panel_key_test.cpp)
- [PlotPanel.cpp](../../../src/apps/runner/src/PlotPanel.cpp) / [PlotPanel.hpp](../../../src/apps/runner/src/PlotPanel.hpp)
- [QValuePanel.cpp](../../../src/apps/runner/src/QValuePanel.cpp) / [QValuePanel.hpp](../../../src/apps/runner/src/QValuePanel.hpp) / [QValuePanel_test.cpp](../../../src/apps/runner/src/QValuePanel_test.cpp)
- [RunnerApp.cpp](../../../src/apps/runner/src/RunnerApp.cpp) / [RunnerApp.hpp](../../../src/apps/runner/src/RunnerApp.hpp) / [RunnerApp_test.cpp](../../../src/apps/runner/src/RunnerApp_test.cpp)
- [RunnerFrame.cpp](../../../src/apps/runner/src/RunnerFrame.cpp) / [RunnerFrame.hpp](../../../src/apps/runner/src/RunnerFrame.hpp) / [RunnerFrame_test.cpp](../../../src/apps/runner/src/RunnerFrame_test.cpp)
- [TrainPanel.cpp](../../../src/apps/runner/src/TrainPanel.cpp) / [TrainPanel.hpp](../../../src/apps/runner/src/TrainPanel.hpp)
- [WorkspaceDialog.cpp](../../../src/apps/runner/src/WorkspaceDialog.cpp) / [WorkspaceDialog.hpp](../../../src/apps/runner/src/WorkspaceDialog.hpp)
- [pch.hpp](../../../src/apps/runner/src/pch.hpp)
- [runner.ico](../../../src/apps/runner/src/runner.ico) / [runner.manifest](../../../src/apps/runner/src/runner.manifest) / [runner.rc](../../../src/apps/runner/src/runner.rc)

### 機能カテゴリなし

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
- [CMakeLists.txt](../../../src/apps/runner/CMakeLists.txt)

## RFD の一覧

点検が作り直す。手で直さない。

- [RFD0034](../../memo/done/0034_imagecls_batch_input_10rfd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [RFD0052](../../memo/done/0052_eval_schedule_separation_10rfd.md) — PRD 052: eval 定義とスケジュールの namespace 分離
- [RFD0053](../../memo/done/0053_interval_gate_perf_ema_10rfd.md) — PRD 053: interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化
- [RFD0057](../../memo/done/0057_native_stack_dim_handling_10rfd.md) — Stack 次元のネイティブ対応(Reshape ハックを廃する根本対応)
- [RFD0060](../../memo/done/0060_eval_batch_episodes_10rfd.md) — 評価セッション契約: eval_batch_size（lane 数）と eval_episodes（本数）で N 本平均を 1 点として記録する
- [RFD0061](../../memo/done/0061_eval_slot_policy_override_10rfd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [RFD0062](../../memo/done/0062_plasticity_metrics_10rfd.md) — 可塑性メトリクス（srank / dormant unit 率）PRD
- [RFD0063](../../memo/done/0063_plasticity_weight_norm_10rfd.md) — 可塑性 weight norm（feature/readout 分割）PRD
- [RFD0065](../../memo/done/0065_nn_spectral_norm_10rfd.md) — NN Spectral Normalization（weight_norm.mode）PRD
- [RFD0066](../../memo/done/0066_policy_churn_metrics_10rfd.md) — Policy churn メトリクス（35_agent_churn）PRD
- [RFD0067](../../memo/done/0067_MunchausenRL_10rfd.md) — Munchausen RL（M-DQN / M-QR / M-IQN）PRD
- [RFD0069](../../memo/done/0069_metrics_trace_channel_10rfd.md) — メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す
- [RFD0070](../../memo/done/0070_device_spec_auto_10rfd.md) — PRD 070: device 指定の auto 対応とキー体系統一
- [RFD0073](../../memo/done/0073_replay_fit_metrics_10rfd.md) — PRD 073: replay 抽選履歴別の TD・損失メトリクス
- [RFD0075](../../memo/done/0075_episode_steps_and_eval_session_log_10rfd.md) — PRD 075: エピソード長の汎用メトリクス化と評価セッションの実行ログ
- [RFD0076](../../memo/done/0076_eval_session_drain_on_exit_10rfd.md) — PRD 076: 終了時の評価セッション排水
- [RFD0082](../../memo/done/0082_atari_ram_progress_metrics_10rfd.md) — Atari のゲーム固有イベントを RAM メトリクスとして取る — 番号付き汎用キーへゲーム別に紐づける PRD
- [RFD0083](../../memo/done/0083_action_context_retire_10rfd.md) — ActionContext の廃止（Actor が RNG と frame stacker を直接持つ）
- [RFD0084](../../memo/done/0084_noisynet_10rfd.md) — NoisyNet (Noisy Networks for Exploration) PRD
- [RFD0092](../../memo/done/0092_gradient_noise_scale_10rfd.md) — 勾配ノイズの大きさ（gradient noise scale）の計測 PRD
- [RFD0093](../../memo/done/0093_attention_entropy_diagnostics_10rfd.md) — TransformerEncoder の attention エントロピー診断 PRD
- [RFD0094](../../memo/done/0094_evalpanel_step_history_10rfd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD
- [RFD0097](../../memo/done/0097_env_auto_reset_after_notification_10rfd.md) — 遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD
- [RFD0098](../../memo/0098_temporl_skip_policy_10rfd.md) — TempoRL 対応（スキップ: 選んだ行動を続ける長さの学習）RFD
- [RFD0099](../../memo/done/0099_atari_resize_mode_10rfd.md) — Atari の観測の縮小方式を設定で切り替える PRD
- [RFD0100](../../memo/done/0100_gridmaze_solvable_maze_10rfd.md) — GridMaze の迷路を必ず解ける形にする — 解けない seed は派生 seed で決定的に引き直す PRD
- [RFD0101](../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md) — 探索と学習のパラメータを ProfiledValue で時間変化させる PRD
- [RFD0103](../../memo/done/0103_temporl_reference_alignment_10rfd.md) — TempoRL 第二弾（著者の実装との差の洗い出しと是正）PRD
- [RFD0104](../../memo/done/0104_temporl_skip_auxiliary_only_10rfd.md) — TempoRL の補助学習のみモード（毎 step 判断し、スキップの出力は補助タスクとしてだけ学ぶ）PRD
- [RFD0105](../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md) — Evaluation Trace の矢印キーと強制した step の見分け、表の並べ替えと行のダンプ PRD
- [RFD0106](../../memo/done/0106_temporl_skip_forward_noise_key_10rfd.md) — TempoRL の skip 側の forward のノイズを Q 側と分けて指定する（`full` でだけノイズが効く仕組みの切り分け）PRD
- [RFD0109](../../memo/done/0109_runner_startup_background_init_10rfd.md) — Runner の起動で RunManager を別スレッドで構築する PRD
- [RFD0116](../../memo/0116_rnd_lifelong_novelty_10rfd.md) — RND による一生の新しさだけの腕（内的報酬の第一の腕）暫定 RFD
- [RFD0117](../../memo/0117_warn_policy_and_unread_config_10rfd.md) — RFD0117: WARN の意味の線引きと、読まれない設定の検出 RFD
- [RFD0119](../../memo/0119_small_doc_comment_config_fixes_10rfd.md) — RFD0119: 文書・コメント・設定の小さな食い違いをまとめて直す
- [RFD0121](../../memo/0121_extrinsic_intrinsic_value_decomposition_10rfd.md) — 外的と内的の価値の分解（Q を外的報酬と内的報酬の 2 つの Head で学ぶ） RFD
- [RFD0122](../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md) — RFD0122: 小さなリファクタリングとテストの空白をまとめて片づける
- [RFD0123](../../memo/0123_rnd_intrinsic_reward_scale_10rfd.md) — RND の内的報酬の尺度（誤差の統計の取り方・warm-up・平均を引かない形・終端の mask） RFD
- [RFD0127](../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md) — Evaluation Trace で RND の内的報酬を見る（統計を変えない採点と step 診断の鍵）RFD
- [RFD0158](../../memo/0158_eval_schedule_first_fire_10rfd.md) — RFD0158: eval スケジュールの発火の列 — 学習直後の評価を既定で止め、発火の時点をずらせるようにする
- [RFD0170](../../memo/0170_nn_trace_capture_unification_10rfd.md) — RFD0170: forward の名前付き出力を capture に一本化し、nn_trace を要求した step だけ作る RFD
- [RFD0182](../../memo/0182_source_layout_10rfd.md) — ソース配置の見直し（置き場の規則、Agent の系統ごとのライブラリ、DQN のファイル分割、テストの参照範囲）RFD
- [RFD0191](../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md) — RFD0191: RND の設定を段階ごとに組み替え、誤差で割らない内的報酬の式と、ためてからエポックで学ぶ predictor の補助学習を足す（Burda らの形へ寄せる） RFD
- [RFD0192](../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md) — RFD0192: 監査で見つかった不具合 5 件をまとめて直す（Notifier の同時実行、動画の画像の大きさ、強制行動の device、重複登録、QValuePanel の平滑化）
- [RFD0193](../../memo/0193_variance_scaling_weight_init_10rfd.md) — RFD0193: 重みの初期化に、分散を倍率÷fan_in にする一様分布の方式を足し、初期化の設定の読み取りを 1 つの口にまとめる（Dopamine の IQN・Rainbow と同じ初期値） RFD
- [RFD0194](../../memo/0194_config_key_full_names_in_messages_10rfd.md) — RFD0194: エラー・WARN・INFO が示す設定キーを完全キーにする（device の二重の fatal、指標名、用語の不整合 6 件を含む）
- [RFD0195](../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md) — RFD0195: 文書の呼び名を RFD に改め、番号を 0 埋めの 4 桁にし、今使っている場所をすべて移す PRD

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | RFD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261111-61ad60](../../management/issues/issue-20261111-61ad60.md) | 調査中 | 2.7 | — | コード | 実装 | 110_runner_app | 開発 | 調査 | — | 未確認 | GUI既定スイートのLogs折り返し整列テストが変更前から失敗する |
| [issue-20261117-eedd0c](../../management/issues/issue-20261117-eedd0c.md) | 調査中 | 2.6 | — | コード | 実装 | 110_runner_app | UI | 不具合 | — | 未確認 | Evaluation Trace の単一行選択テストが合成ドラッグ後に失敗する |
| [issue-20261103-955e50](../../management/issues/issue-20261103-955e50.md) | 調査中 | 1.2 | — | コード | RFD | 318_rl_core, 317_metrics, 110_runner_app, 311_dev_tools, 120_metrics_viewer | UI | 機能追加 | [RFD0081](../../memo/0081_evalpanel_episode_trace_10rfd.md) | 残件有り | EvalPanel のゲームを trace へ出す — dormant タグの trace を鏡写しインスタンスへ結び付け、trace 行の時刻で振り返る |
| [issue-20261111-9e6e27](../../management/issues/issue-20261111-9e6e27.md) | 未着手 | 3.6 | — | コード | RFD | 110_runner_app | UI | 機能改善 | — | 未確認 | 無人で走るRun（番号付きbat）でWARNに気づけない |
| [issue-20261120-1979f6](../../management/issues/issue-20261120-1979f6.md) | 未着手 | 3.5 | — | コード | 実装 | 110_runner_app | UI | 不具合 | — | 未確認 | QValuePanel.hist_bins=0 が設定検証を通り、描画処理で空のヒストグラムを参照しうる |
| [issue-20261123-c455b5](../../management/issues/issue-20261123-c455b5.md) | 未着手 | 3.3 | — | コード | 実装 | 311_dev_tools, 110_runner_app | 開発 | 不具合 | — | 未確認 | 既定のスイートの batchrun_fatal_error_handling_test が、前の Run の重みを読む段を持つキューを既定にした bat で、重みが無い段を失敗に数えて落ちる |
| [issue-20261103-c8fb06](../../management/issues/issue-20261103-c8fb06.md) | 未着手 | 3.0 | — | コード | RFD | 312_util, 110_runner_app | 基盤 | 機能改善 | [RFD0071](../../memo/0071_uncaught_fatal_visibility_10rfd.md) | 残件有り | RFD0071: 捕まらない fatal の可視化（worker catch・set_terminate・エラーログ write-through） |
| [issue-20261111-585e60](../../management/issues/issue-20261111-585e60.md) | 未着手 | 3.0 | — | コード | 監査 | 318_rl_core, 321_agent_core, 315_nn, 110_runner_app | 基盤 | リファクタリング | — | 残件有り | 名前やコメントが実際の動きと合わない関数・クラス・設定項目がある（BatchState の IsDone、DefaultAgentFactory、NetworkGraphVizConfig の 2 項目、Runner の 4 関数） |
| [issue-20261103-9df5c5](../../management/issues/issue-20261103-9df5c5.md) | 未着手 | 2.7 | — | コード | RFD | 318_rl_core, 110_runner_app, 314_tensor | 学習基盤 | 不具合 | [RFD0146](../../memo/0146_notifier_scoped_detach_10rfd.md) | 残件有り | Notifier::AttachScoped の Detach 契約修復 — wrapper ハンドルを返す |
| [issue-20261107-16ffd8](../../management/issues/issue-20261107-16ffd8.md) | 未着手 | 2.7 | — | コード | RFD | 318_rl_core, 322_env_individual, 110_runner_app | 学習基盤 | 不整合 | — | 残件有り | ERROR のログを出したあとも Run を止めずに続ける箇所（Env・Agent の作成失敗、DropMerge の果物の範囲外）が Fail-Fast 原則と合わない |
| [issue-20261111-15ba39](../../management/issues/issue-20261111-15ba39.md) | 未着手 | 2.7 | — | コード | 監査 | 110_runner_app | UI | リファクタリング | — | 未確認 | EvalPanel のモデル同期が mode を文字列のまま step ごとに解釈し直し、EvalTracePanel は pane の開閉から導ける active_ を 4 通りの問い合わせで合わせている |
| [issue-20261118-dc5de3](../../management/issues/issue-20261118-dc5de3.md) | 未着手 | 2.4 | — | コード | その他 | 317_metrics, 110_runner_app | 基盤 | 機能改善 | — | 未確認 | Atari の Run の列ごとにメトリクスの組と tag がずれ、本線・Taiga 寄せ・易しいゲームの Run どうしを Viewer で重ねて比べられない |
| [issue-20261120-7ac5fa](../../management/issues/issue-20261120-7ac5fa.md) | 未着手 | 2.4 | — | コード | 実装 | 110_runner_app, 324_dqn | UI | 不具合 | — | 未確認 | IQN の学習中に HeatMap を既定の選択で開くと taus の不足で Run が停止する |
| [issue-20261111-360278](../../management/issues/issue-20261111-360278.md) | 未着手 | 2.3 | — | コード | 監査 | 110_runner_app | UI | リファクタリング | — | 未確認 | Runner のパネルに描画と表の写しがある（GL キャンバスへ画像を貼る一式が 2 パネルに同文、EvalTracePanel の行の探索 8 か所と再生ツールの並び 3 か所） |
| [issue-20261103-5addea](../../management/issues/issue-20261103-5addea.md) | 未着手 | 2.2 | — | コード | RFD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [RFD0079](../../memo/0079_cpp_internal_linkage_convention_10rfd.md) | 残件有り | RFD0079: .cpp の内部リンケージと namespace インデント規約 |
| [issue-20261103-785e68](../../management/issues/issue-20261103-785e68.md) | 未着手 | 2.0 | — | コード | RFD | 315_nn, 318_rl_core, 321_agent_core, 324_dqn, 319_gui, 320_env_core, 314_tensor, 312_util, 110_runner_app, 322_env_individual, 325_muzero | 開発 | リファクタリング | [RFD0169](../../memo/0169_outside_core_extension_layout_10rfd.md) | 残件無し | anet-core の外で実装する拡張点とコード配置の規約 |
| [issue-20261111-54c7e1](../../management/issues/issue-20261111-54c7e1.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core, 321_agent_core, 315_nn, 319_gui, 110_runner_app | 基盤 | リファクタリング | — | 残件有り | 呼ばれない関数・型・マクロと読まれないメンバ・届かない分岐が、core と Runner に合わせて 1,100 行以上残っている |
| [issue-20261111-90fa5c](../../management/issues/issue-20261111-90fa5c.md) | 未着手 | 2.0 | — | コード | 監査 | 110_runner_app | UI | リファクタリング | — | 未確認 | QValuePanel の ComputeQValueHeatMap が 218 行で値の変換・範囲の更新・投票・色の正規化を並べ、データの受け渡しにスレッド越し用の部品と保護の無いメンバが混ざっている |
| [issue-20261114-9468cb](../../management/issues/issue-20261114-9468cb.md) | 未着手 | 2.0 | — | コード | 実装 | 324_dqn, 110_runner_app | 学習基盤 | 性能改善 | — | 残件有り | Evaluation Traceを開いている間のActor同期で、RNDの推定器の写しが1回約170 msかかる |
| [issue-20261103-0ef7b8](../../management/issues/issue-20261103-0ef7b8.md) | 未着手 | 1.6 | — | コード | RFD | 317_metrics, 311_dev_tools, 316_config, 318_rl_core, 321_agent_core, 120_metrics_viewer, 110_runner_app | UI | 機能改善 | [RFD0108](../../memo/0108_metrics_tag_numbering_10rfd.md) | 残件有り | メトリクスの tag の番号を全体で振り直す（群の番号と群の中の番号） |
| [issue-20261113-d38960](../../management/issues/issue-20261113-d38960.md) | 未着手 | 1.6 | — | コード | 実装 | 110_runner_app, 324_dqn, 318_rl_core | UI | 性能改善 | — | 残件有り | EvalPanel のモデル同期が UI スレッドで学習ロックを待ち、画面の更新や操作応答を止め得る |
| [issue-20261103-50644a](../../management/issues/issue-20261103-50644a.md) | 未着手 | 1.4 | — | コード | RFD | 317_metrics, 320_env_core, 321_agent_core, 318_rl_core, 110_runner_app | UI | 機能追加 | [RFD0184](../../memo/0184_eval_session_step_trace_10rfd.md) | 残件有り | 評価 session の step trace を Run の記録に残す |
| [issue-20261103-66f453](../../management/issues/issue-20261103-66f453.md) | 未着手 | 1.4 | — | コード | RFD | 110_runner_app, 316_config | 学習基盤 | 機能追加 | [RFD0151](../../memo/0151_run_queue_snapshot_10rfd.md) | 残件有り | Run キュー（計画スナップショット + exe 固定） |
| [issue-20261107-2d4932](../../management/issues/issue-20261107-2d4932.md) | 未着手 | 1.4 | — | コード | その他 | 110_runner_app, 324_dqn | UI | 機能追加 | — | 残件有り | EvalPanelで行動選択のτをランダムと固定で切り替えられず、学んだ行動と学習した価値の判断を見比べられない |
| [issue-20261115-c89660](../../management/issues/issue-20261115-c89660.md) | 未着手 | 1.4 | — | コード | その他 | 110_runner_app, 321_agent_core, 318_rl_core | UI | 機能追加 | — | 残件有り | Runner の Conv2dPanel と metrics.image の conv2d に、RND の target と predictor の畳み込みの出力が出ない |
| [issue-20261103-b4e5c9](../../management/issues/issue-20261103-b4e5c9.md) | 未着手 | 1.3（引継ぎ） | — | コード | RFD | 316_config, 110_runner_app | 基盤 | リファクタリング | [RFD0155](../../memo/0155_config_root_ownership_alignment_10rfd.md) | 残件有り | 設定 root の所有整合(`train.` / `app.run_name` 等の階層ずれ棚卸し) |
| [issue-20261123-b1205d](../../management/issues/issue-20261123-b1205d.md) | 未着手 | 1.3 | — | コード | 実装 | 110_runner_app, 322_env_individual | 開発 | リファクタリング | — | 未確認 | Runner と Atari のテストが同じ文字列を重ねて書いている（環境変数の名前 RFD0094_UI_DARK を 3 か所、検証資材のパス rfd0099/frames_210x160.u8 を 2 か所） |
| [issue-20261103-07629c](../../management/issues/issue-20261103-07629c.md) | 未着手 | 1.2 | — | コード | RFD | 110_runner_app, 318_rl_core, 319_gui | UI | 不具合 | [RFD0176](../../memo/0176_qvaluepanel_manual_action_state_alignment_10rfd.md) | 残件有り | QValuePanel の Q の表示と、手動で選んだ行動が当たる状態のずれ |
| [issue-20261103-15f115](../../management/issues/issue-20261103-15f115.md) | 未着手 | 1.1 | — | コード | RFD | 110_runner_app, 319_gui | UI | 機能改善 | [RFD0113](../../memo/0113_runner_per_monitor_dpi_10rfd.md) | 残件有り | Runner を Per-Monitor V2 の DPI 対応で動かす RFD（manifest の gdiScaling を外す） |
| [issue-20261103-206933](../../management/issues/issue-20261103-206933.md) | 未着手 | 1.1 | — | コード | RFD | 110_runner_app | 学習基盤 | 機能追加 | [RFD0175](../../memo/0175_seed_population_early_stop_10rfd.md) | 残件有り | 多数の seed で打席を稼ぐ（条件付きの打ち切りで、火が付かなかった Run を早く止める） |
| [issue-20261111-d1d12b](../../management/issues/issue-20261111-d1d12b.md) | 未着手 | 1.0 | — | コード | 監査 | 321_agent_core, 315_nn, 318_rl_core, 317_metrics, 316_config, 312_util, 313_profiling, 314_tensor, 322_env_individual, 110_runner_app | 学習基盤 | リファクタリング | — | 残件有り | 今は起こさないリファクタリング候補 16 群を暫定で 1 件にまとめた（手を付けるときに群ごとに分けて起票し直す） |
| [issue-20261103-53ed7a](../../management/issues/issue-20261103-53ed7a.md) | 未着手 | 0.8 | — | コード | RFD | 110_runner_app, 319_gui | UI | 機能追加 | [RFD0186](../../memo/0186_runner_theme_switch_10rfd.md) | 残件有り | 起動中の Runner を OS のテーマ（ライト／ダーク）の切替に追従させる |
| [issue-20261103-736e39](../../management/issues/issue-20261103-736e39.md) | 未着手 | 1.4 | [issue-20261103-b754ff](../../management/issues/issue-20261103-b754ff.md), [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md), [issue-20261103-9bd71e](../../management/issues/issue-20261103-9bd71e.md), [issue-20261103-75b8da](../../management/issues/issue-20261103-75b8da.md), [issue-20261107-949e8a](../../management/issues/issue-20261107-949e8a.md) | コード | RFD | 324_dqn, 321_agent_core, 318_rl_core, 322_env_individual, 110_runner_app | アルゴリズム | 機能追加 | [RFD0173](../../memo/0173_intrinsic_reward_novelty_10rfd.md) | 残件有り | 内的報酬（NGU・E3B のエピソード内の新しさと RND）と方策の組 |
| [issue-20261103-6661d2](../../management/issues/issue-20261103-6661d2.md) | 未着手 | 1.3 | [issue-20261103-f39d10](../../management/issues/issue-20261103-f39d10.md), [issue-20261103-2d66d2](../../management/issues/issue-20261103-2d66d2.md) | コード | RFD | 318_rl_core, 324_dqn, 110_runner_app | 学習基盤 | 機能追加 | [RFD0136](../../memo/0136_best_checkpoint_10rfd.md) | 残件有り | best checkpoint の自動保存（eval 自己ベスト更新時の重み保持） |
| [issue-20261103-b68e49](../../management/issues/issue-20261103-b68e49.md) | 未着手 | 1.3 | [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md) | コード | RFD | 110_runner_app, 318_rl_core, 322_env_individual | UI | 機能追加 | [RFD0159](../../memo/0159_input_mapping_joystick_10rfd.md) | 残件有り | 操作入力の Joystick 対応と操作マッピングの設定化（EvalPanel） |
| [issue-20261103-578986](../../management/issues/issue-20261103-578986.md) | 未着手 | 0.6 | [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md), [issue-20261103-b68e49](../../management/issues/issue-20261103-b68e49.md) | コード | RFD | 318_rl_core, 319_gui, 110_runner_app | 学習基盤 | 機能追加 | [RFD0160](../../memo/0160_input_target_train_forced_action_10rfd.md) | 残件有り | 操作入力の入力先切替と Train 側の強制行動（学習への反映） |

着手できる 32 / 待ち 4 / 保留 0 / 次版 0 / 解決 0
- 関連する課題: [issue-20261103-b754ff](../../management/issues/issue-20261103-b754ff.md) — stucker.stack_keysに存在しないキー名を書いても検出されない
- 関連する課題: [issue-20261107-949e8a](../../management/issues/issue-20261107-949e8a.md) — AtariのRAMメトリクスに値の種類数とビットを数える畳み方が無く、Montezumaの訪れた部屋の数や持ち物を記録できない
- 関連する課題: [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md) — 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化
- 関連する課題: [issue-20261103-2d66d2](../../management/issues/issue-20261103-2d66d2.md) — 999: train/eval mode 整理後の排他精査 RFD（草案）
- 関連する課題: [issue-20261103-75b8da](../../management/issues/issue-20261103-75b8da.md) — n-step の目標値の補正（窓の途中の Munchausen の項・方策オフの補正）
- 関連する課題: [issue-20261103-f39d10](../../management/issues/issue-20261103-f39d10.md) — Background eval の network snapshot 順序保証
- 関連する課題: [issue-20261103-9bd71e](../../management/issues/issue-20261103-9bd71e.md) — 直前行動観測の共通部品（`prev_action` フレーム段）
<!-- issues:end -->

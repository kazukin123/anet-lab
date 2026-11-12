# Runner の主 pane が Centre を動的に占有し、Train View を隠せるようにする PRD

> 2026-10-31 起票（Claude Code、`anet-create-prd`。グリルあり）。
> 2026-10-31 別モデル（Opus）の検討のあと（同じ枠）: G2 の守りの穴 3 つ（イベントを出さないドラッグ、maximize 中のガード、UPDATE_UI を通らないメニューイベント）を塞ぐ作りに改めた。✕ と Floatable は既存の 200 ms の UPDATE_UI で状態から宣言し（D3）、「隠してよいか」の判定を基底の 1 つの口にまとめて UPDATE_UI・ハンドラ・✕ の Veto が同じ判定を読み（D3）、メニューで maximize 中の pane を隠すときは先に maximize を解く（D9）。Train の描画 timer はポリシーの入口で表示状態を渡す 1 か所に畳み（D6。ユーザーの裁定）、昇格前の Eval の dock 位置を覚えて降格で戻し（D5。ユーザーの裁定）、知らない経路で Centre が空いたときの保険を WARN 付きで置き（D13。ユーザーの裁定）、`app.main_view` の既定を `eval` に変えた（D7。ユーザーの裁定: Train View は最近ほとんど見ておらず、学習の進みは Step 表示とログで足りる）。あわせて「両方 Centre」の却下理由、Eval の maximize が回避策として足りない理由、D7 の理由、D2 の範囲、D5 の守れない約束、✕ のテストの送り方、行番号を直した。
> 2026-10-31 レビュー（1 往復目）のあと: `CONTEXT.md` に「主領域」が既にあったので、冒頭の語の説明と D12 を直し、用語集の定義を Centre の占有が切り替わることに合わせて改めた。§5.5 の実装の語（アクセス指定子）を契約の言い方に替えた。再確認ポイントから、D13 の Fail-Fast 上の区分、設定に無いときの既定（§5.3）、受入 2 の f の幅の時機、D3 と ADR 0016 の関係を本文に書いた。
> 2026-11-01 再グリル（Claude Code、`anet-create-prd`。初版の実装のあと、ユーザーの求め）: Train View が表示中は常に float できなくなっていた。初版の D1（docked の Train は常に占有者）と D2（Centre を占める pane は float 不可）の帰結で、初版のグリルでは独立の問いにしていなかった。float 可否を D3 と同じ判定（相方の主 pane が表示中で docked）に揃え（D2）、Centre の占有者を「隠れるか float するまで保ち、離れたら表示中で docked の相方が昇格し、元の占有者が docked のまま戻れば取り戻す」対称の規則に改め（D1。ユーザーの裁定）、float した pane を縁の帯へ落とすとその場所に留まることにし、昇格前の位置の記憶を両 pane に広げた（D5）。ドラッグで float した直後の昇格は D3 の 200 ms の UPDATE_UI で当て、保険の対象に「両方が Centre」を足した（D13）。あわせて §1.1 の 24・25、§1.2、G1・G2・G5、§3、D7・D8・D11・D12、§4.1、§5.1・§5.2・§5.4・§5.5・§5.6、§6 の 14・16〜18、§7 の受入 2・3・6・8、§8、§9 を直した。
> 2026-11-01 レビュー（1 往復目）のあと: 移行対象に 160・020 の en 版と検証資材の README、160 §2.1・§7.1 の占有と位置の記憶の文を足した（D12・§5.5・受入 9）。§5.2 の Reset Layout の記憶の単位、G5 の件数、受入 3 の 4 つ目の手順、§5.1 の列の意味、冒頭の語の列挙に「占有者」を直した。2 往復目のあと: D12 の README の対応先を受入 8 の 3 つ目にし、受入 9 の検索条件を初版の文に固有の形に、§5.1 の 5 行目の記憶の条件を直した。3 往復目のあと: §6 の 13 の文書の更新先を D12 の列挙に揃えた。
> 2026-11-01 再グリル（2 回目。Claude Code、`anet-create-prd`。改訂の実装のあと、実機で Centre の Train View を caption ドラッグで float できないとユーザーが報告し、Codex の調査で wxAUI が Centre の pane の caption 押下をドラッグの前に打ち切ることが分かった）: §1.1 の 6 を訂正し（Centre を占める pane は caption ドラッグで float できない。初版と 1 回目の再グリルは `OnMotion` だけを見ていた）、pin ボタンの経路の事実を 26 に足した。Centre を占める主 pane の float は caption の pin ボタンで行い、pin は Floatable と同じ値で描き、宣言が追いつく前の pin の押下は判定で止める（D2・D3。ユーザーの裁定: wx の判定を迂回する実装の歪は入れず、操作して確かめられる形で直す）。pin ボタンを全 pane に付ける（D14。ユーザーの裁定: 一部だけ付けるのは変）。あわせて §1.2、G2・G5・NG5、§3、D1・D8・D9・D12、§4.1、§5.1・§5.2・§5.4・§5.5・§5.6、§6 の 3・13・14・15・17 と 19〜21、§7 の前文と受入 2・8・9、§8、§9 を直した。レビュー（1 往復目）のあと: §6 の 15 の理由の経路の件数、§1.1 の 6 の `RunnerFrame.cpp` の行番号、§5.5 の HeatMap・Conv2d の pane の作られる場所、§6 の前文と本行の列挙、§9 の 8 を直した。2 往復目のあと: §9 の前文の列挙に 8 を入れた。
> 2026-11-01 再グリル（3 回目。Claude Code、`anet-create-prd`。2 回目の改訂の実装のあと、実機で、Train を左の縁へ落として Eval の隣に置いても等分にならない、位置が逆でも効かないとユーザーが報告）: 等分の条件を「Eval が Right の layer 0 で Train が Centre」の 1 配置から「一方が Centre、他方が Left か Right の dock」に広げ（D15。ユーザーの裁定: 案 1）、主 pane の置き場が変わったときは 200 ms の UPDATE_UI で幅のポリシーを当て（D15。ユーザーの裁定: A）、Centre を占める主 pane の caption の末尾に `[Centre]` を付ける（D16。ユーザーの裁定: caption に印、文言は `[Centre]`）。ADR 0016 の先頭に注記を足し、CONTEXT.md の「主領域」を直した。あわせて §1.1 の 10・27、§1.2、G5・G6・NG7、§3、D3・D5・D11・D12、§4.1、§5.1・§5.2・§5.5・§5.6、§6 の 7・11・13 と 22〜27、§7 の前文と受入 2・8・9、§8、§9 を直した。レビュー（1 往復目）のあと: §5.5 のテストに D15 で断定が変わる既存テストを足し、受入 2 の o〜q を置き場の準備の形にしてサッシの手順を外し（§7 の前文に直接操作の扱いを書いた）、r に時機、§3 の G5 の参照、§9 の 27 の件数、受入 9 の 020 の印の置き場、D11 の理由、§1.1 の 27 の右の縁の layer を直した。2 往復目のあと: §7 の前文の送れる操作に maximize と restore を足し、受入 2 の m と受入 4 の A〜C を「押したことにする」形にした。
> 2026-11-01 再グリル（4 回目。Claude Code、`anet-create-prd`。Plan Mode。3 回目の改訂の実装のあと、実機で、Eval の下の帯に Train を落として縦に並べたときも高さを等分したいとユーザーが求めた）: 等分の条件に「一方が Centre、他方が Centre と同じ幅の Top か Bottom の帯」を足し、高さを等分する（D15。ユーザーの求め）。帯が Centre より広い（左右の列の下まで広がる最外層など）ときと同居は対象外。高さが足りないときは Logs を縮めず最小高で clamp する（ユーザーの裁定）。「等幅」を「等分」に改めた。あわせて §1.1 の 6・24・27 と 28・29、§1.2、G5、§3、D5・D12・D15、§4.1、§5.1・§5.2・§5.5、§6 の 7・23 と 28〜31、§7 の前文と受入 2・8・9、§9 を直した。レビュー（1 往復目）のあと: 置き換えで壊れた「等幅」の引用 3 か所、D5 の括弧、D15 の理由の文、Centre pane の縁の帯の厚み（40 DIP）、§5.5 の既存テスト、§7 の前文の列挙、D12 の 020 の同居、§1.1 の 24・27 の window の縁、§6 の 23 を直した。2 往復目のあと: §1.1 の 24 の位置列（window の縁の帯の行）と §6 の前文の記録（23）を直した。
> 起点: 2026-10-31、Train を裏で任せて Eval だけ見たいとき、Train を流さず Eval だけ流したいときに、Train View を隠せない（§1）。
> 関連: [ADR 0016](../../adr/0016-runnerframe-aui-layout-roundtrip.md)（dock サイズの往復と遷移時同期。本 PRD の昇格・降格はこの枠の中で行う）、[ADR 0066](../../adr/0066-main-panes-occupy-centre-dynamically-and-no-flow-empties-centre.md)（本 PRD で起こした）、[PRD055](055_runner_toolbar_10prd.md)（Panels ツールバーと UPDATE_UI による同期）、[PRD109](109_runner_startup_background_init_10prd.md)（起動の別スレッド化。実装済み。D6 の「初期化前の pane の開閉で落ちない」に本 PRD も従う）、[999_runner_theme_switch](../953_runner_theme_switch_10prd.md)（外観の決め直し。pane の配置を変えない条件を共有する）、[999_notifier_scoped_detach](../942_notifier_scoped_detach_10prd.md)（NG2 の前提）、[160 アプリケーションとツール](../../design/160_applications_and_tools.jp.md) §2.1・§7.1、[020 Run 実行ガイド](../../design/020_user_guide_run.jp.md) §6 のツールバーと pane。

本書で「主 pane」は Train View（`TrainPanel`）と Evaluation View（`EvalPanel`）の 2 つ、「docked」は frame の中の dock にいること、「float」は wxAUI の floating frame にいること、「占有者」は Centre を占めている主 pane。この 4 つは本書の中だけの言い方で、既存の語（pane、dock、Centre）の組み合わせで言い表せるので用語集には載せない。「主領域」は [CONTEXT.md](../../../CONTEXT.md) の「Runner GUI」節にある語（Runner 画面中央で Train pane・Eval pane が占める領域）で、本 PRD で Centre の占有が切り替わるようになるので、用語集の定義もあわせて直す（D12）。

## 1. 問題

### 1.1 今の事実（2026-10-31 にコードと wxWidgets 3.3.3 のソースで確かめた）

| # | 事実 | 位置 |
|---|---|---|
| 1 | Train pane は `Centre()` で、`CloseButton(false)`。Floatable は既定（true）のまま。View メニューの Train View の項目はコメントアウト、Panels ツールバーに Train のボタンは無い。`ID_TrainPanel` は pane 登録（`RegisterPaneMenu`）だけされている | [RunnerFrame.cpp:261](../../../apps/runner/src/RunnerFrame.cpp)、:404-407、:450-456、:726 |
| 2 | wxAUI は Centre dock が 1 つも無いと、中央に背景の伸縮枠（`typeBackground`）を置く。側面の dock は固定幅のまま広がらない | wxWidgets 3.3.3 `src/aui/framemanager.cpp` 3004〜3024 行（`LayoutAll`） |
| 3 | Centre dock を 2 つ（別の row）置くと横に並び、それぞれが伸縮するが、間に sash が無く比率を変えられない。同じ dock に 2 つ置くと縦に積まれる | 同 2660 行（`LayoutAddDock`）、`include/wx/aui/framemanager.h` 814〜819 行（`IsVertical`） |
| 4 | caption のボタン（Minimize・Maximize・Pin・Close）は pane のフラグだけで決まり、dock の方向は見ない。Centre の特別扱いは Minimize の禁止だけ。Centre pane にも ✕ は出せる | 同 2398〜2447 行（`LayoutAddPane`）、1645〜1650 行 |
| 5 | ✕ の経路: `OnPaneButton` が `wxEVT_AUI_PANE_CLOSE` を出し、Veto されなければ `ClosePane` と `Update` を行う。`ClosePane` は隠す前に、その pane が maximize 中なら解く（`DoHidePaneWindow`）。`OnPaneButton` は ✕ が描かれているかを見ないので、✕ の無い pane にも「押した」のイベントを作れる | 同 5388〜5410 行、1447〜1453 行 |
| 6 | caption を押したとき、その pane の dock が Centre なら wxAUI はドラッグの判定に入る前に処理を打ち切るので、Centre を占める pane は caption ドラッグで float できない（2026-11-01 の 2 回目の再グリルで訂正。初版と 1 回目の再グリルは `OnMotion` だけを見て「Centre の判定は無い」としていた）。Centre でない pane のドラッグの開始条件は「toolbar でない」「manager に `wxAUI_MGR_ALLOW_FLOATING`」「pane が Floatable」の 3 つで、Runner はフラグを立てている。float した pane を Centre pane の上に落とすと、Centre pane の縁の帯（厚みは 40 DIP と pane の幅・高さの 20% の小さい方）ならその方向の layer 0 の新しい row、真ん中なら何もせず float のままで、Centre へは戻れない（縁の帯の layer は 2026-11-01 の 4 回目の再グリルで確かめた。§1.1 の 29） | 同 4767〜4770 行（`OnLeftDown`）、5164〜5182 行（`OnMotion`）、3800〜3836 行（`DoDrop`）、[RunnerFrame.cpp:212](../../../apps/runner/src/RunnerFrame.cpp)（`SetFlags`。2026-11-01 の行番号） |
| 7 | ドラッグで float するとき、落として dock するときの両方で、wxAUI は frame にイベントを出さない。wxAUI のイベントは BUTTON・CLOSE・MINIMIZE・MAXIMIZE・RESTORE・ACTIVATED・RENDER・FIND_MANAGER だけで、ACTIVATED には `wxAUI_MGR_ALLOW_ACTIVE_PANE` が要るが Runner は立てていない | 同 5176〜5207 行、4288〜4347 行（`OnFloatingPaneMoved`）、`framemanager.h` 868〜875 行 |
| 8 | `Movable(false)` は `src/aui` のどこでも読まれない。toolbar でない pane のドラッグは「float して引きずる」の 1 経路だけなので、`Floatable(false)` を付けると位置の移動もできなくなる | 同 `src/aui` 全体の `IsMovable`・`optionMovable` の参照 0 件 |
| 9 | 右 dock の合計幅は `SetDockSizeConstraint(0.85, 0.3)` で frame の 85% まで | [RunnerFrame.cpp:189](../../../apps/runner/src/RunnerFrame.cpp) |
| 10 | Train:Eval=50:50 のポリシーは、Eval が Right(layer 0) にあり Train が Centre に表示中のときだけ適用され、遷移のたびに Eval の幅を半分へ戻す。幅不足時は補助列→Q-Values 列の順で縮め、主領域の最小幅は Train と Eval の最小幅（200+200）の和で数える。maximize 中は何もしない（LoadLayout の往復が `savedHiddenState` を壊すため）。集計するのは Right の縦の列（Eval の dock、Q-Values 列、補助列）だけで、Left の dock は見ない。条件の layer は 0 で、ほかの layer にいる Eval は定位置と見なさない（2026-11-01 の 3 回目の再グリルで確かめた） | [RunnerFrame.cpp:1017-1153](../../../apps/runner/src/RunnerFrame.cpp)（`ApplyMainPaneWidths`。条件は :1027-1028、集計は :1047-1080。2026-11-01 の行番号）、[ADR 0016](../../adr/0016-runnerframe-aui-layout-roundtrip.md) |
| 11 | 隠す経路はメニュー OFF（`OnTogglePaneMenu`）と ✕（`OnAuiPaneClose`）の 2 つで、どちらも dock 幅を `pane.dock_size` へ退避し、`savedHiddenState` を同期し、派生の `OnPaneHiding` を呼んでからポリシーを適用する。メニュー OFF は `pane.Show(false)` だけで、maximize 中の pane を解かない。表示は `OnPaneShowing`。チェックと有効・無効は UPDATE_UI の束が `IsPaneOpen` で決め、束の有効条件は `pane.IsOk()` だけ。メニューイベントは UPDATE_UI の無効を通らずにハンドラへ届く | [gui.cpp:295-342](../../../core/anet-core/src/gui.cpp)、[gui.hpp:118-122](../../../core/anet-core/include/anet/gui.hpp)、[RunnerFrame.cpp:756-768](../../../apps/runner/src/RunnerFrame.cpp)、[PRD109](109_runner_startup_background_init_10prd.md) D6 |
| 12 | 基底の `SyncDockSizesToPanes` は、隠す遷移のたびに表示中の pane 全部の dock の実サイズを `pane.dock_size` へ書き戻す（Centre の pane も含む） | [gui.cpp:221-234](../../../core/anet-core/src/gui.cpp) |
| 13 | frame 自身に Bind した 200 ms 周期の UPDATE_UI（`OnUpdateTrainStatus`）があり、Always on Top の反映もここで行う。idle の外で `Update` を呼ぶ前例は `UpdateLogAlertBadge` の `CallAfter` にある | [RunnerFrame.cpp:697](../../../apps/runner/src/RunnerFrame.cpp)、:1328、:607-642、:205 |
| 14 | `TrainPanel` は `Initialize` で View を作り、Train の observer を付け、timer を始める。`SetFps` が周期を決め、0 で止め、呼ばれた時点で timer を始める。実行時の FPS は View メニューのラジオ（Config・0・1・5・10・15・30・60）が持ち、`SetFps` を呼ぶ。`OnClose` はどこにも Bind されていない | [TrainPanel.cpp:40-83](../../../apps/runner/src/TrainPanel.cpp)、[RunnerFrame.cpp:271-280](../../../apps/runner/src/RunnerFrame.cpp)、:775-783 |
| 15 | `app.eval_panel.auto_start` は Eval の pause 状態だけを決める。Eval pane の起動時の表示は `.Hide()` の固定で、設定は無い。`app.eval_panel.model_sync.mode` は前後の空白を除いて小文字にしてから比べる | [EvalPanel.cpp:81](../../../apps/runner/src/EvalPanel.cpp)、:40-50、[RunnerFrame.cpp:468](../../../apps/runner/src/RunnerFrame.cpp) |
| 16 | app の設定は `app.$ = app.online > P1` のような選択で構成別（online／batchrun）の束を重ねる。`RunnerApp::Config` が `app.*` を読んで検証する。Atari の online は `app.online.train_panel.fps = 0` で、Train View は描かれないまま主領域を占める | [Atari.txt:5](../../../apps/runner/config/Atari.txt)、:212、[common.txt:131-144](../../../apps/runner/config/common.txt)、[RunnerApp.cpp:55-110](../../../apps/runner/src/RunnerApp.cpp) |
| 17 | Eval の step と resume は、Eval pane が隠れていれば先に表示する（`ShowEvalPaneIfHidden`） | [RunnerFrame.cpp:1232-1242](../../../apps/runner/src/RunnerFrame.cpp) |
| 18 | Reset Layout（`RestoreDefaultPanes`）は Train・Eval・Logs・Q-Values の pane を直接書き換え、`OnPaneHiding`／`OnPaneShowing` を呼ばない（補助 pane だけは `HideAuxPanes` が `OnPaneHiding` を呼ぶ）。frame の大きさは変えない | [RunnerFrame.cpp:1013-1058](../../../apps/runner/src/RunnerFrame.cpp)、:1194-1218 |
| 19 | PRD109 は実装済み。初期化前のパネルが pane の開閉・timer・サイズ変更で落ちないことを求め（D6）、初期化中に使える操作の表に Train View の開閉を含める。その GUI テスト `[prd109_ui]` は「Train は現行 UI では menu が無い」として、AUI を直接操作して Train を隠して出している | [PRD109](109_runner_startup_background_init_10prd.md) D6・§5.2、[RunnerApp_test.cpp:184-217](../../../apps/runner/src/RunnerApp_test.cpp) |
| 20 | 既存の GUI テストは実 RunnerApp を起動して Frame の子を操作し、RunnerApp を起動するケースは別プロセスで動かす。`[eval_steps_menu]` は `RunnerFrame` を直接作るケースも持つ。`[prd105_ui_keys]` は Frame の子から `TrainPanel` を探す | [EvalTracePanel_test.cpp:876-935](../../../apps/runner/src/EvalTracePanel_test.cpp)、:1098-1110 |
| 21 | `TrainPanel` の observer は `AttachScoped` の戻り値で、`Detach` へ渡しても一致しない既存バグの対象 | [999_notifier_scoped_detach](../942_notifier_scoped_detach_10prd.md) Context |
| 22 | Eval pane を隠しても Eval の timer は止まらない（`OnPaneHiding` は Evaluation Trace だけを止める）。Train の定期停止の増幅要因として記録がある | [RunnerFrame.cpp:1168-1176](../../../apps/runner/src/RunnerFrame.cpp)、[999_eval_schedule_first_fire](../918_eval_schedule_first_fire_10prd.md) Context |
| 23 | Eval を maximize すると、その dock が伸びて Eval だけが主領域を占める。ただし Logs・Q-Values・Evaluation Trace も隠れ、起動時には選べず、Train の timer は回り続ける。Eval を maximize したまま Eval View を OFF にすると、maximize の印を持ったまま隠れ、ほかの pane も隠れたままで画面にツールバーだけが残る | `framemanager.cpp` 2660 行、[gui.cpp:313](../../../core/anet-core/src/gui.cpp)、§1.1 の 10・11 |
| 24 | caption ドラッグで float した直後は、wxAUI がメインフレームのマウスキャプチャで floating frame を動かす（`actionDragFloatingPane`）ので、wx のイベントループは回り、idle と UPDATE_UI は届く。floating frame 自身の title bar で動かすときは OS の移動ループで、落とした時点の `OnFloatingPaneMoved` が dock を決める。window の縁の帯へ落とすと、その方向の新しい列に dock する（2026-11-01 に確かめた。Centre pane の縁の帯は §1.1 の 29） | 同 5164〜5207 行（`OnMotion`）、5228〜5243 行、4288〜4347 行、3533〜3602 行（`DoDrop` の window の縁の帯） |
| 25 | caption ドラッグで float する前に、その pane が maximize 中なら wxAUI が restore する（2026-11-01 に確かめた） | 同 5191〜5192 行 |
| 26 | caption の pin ボタンの経路（`OnPaneButton`）は、manager に `wxAUI_MGR_ALLOW_FLOATING` があり pane が Floatable であることだけを見て、Centre の判定が無い。maximize 中なら `wxEVT_AUI_PANE_RESTORE` を出し、Veto されなければ restore してから `Float()` と `Update()` を行う。caption のボタンの押下は `wxEVT_AUI_PANE_BUTTON` として frame、manager の順に届く（✕ と同じ。§1.1 の 5）。pin で float した floating frame の位置は、pane の `floating_pos` が未設定なら OS 既定で、動かすと `floating_pos` に同期され、次からはそこに出る。Runner で pin があるのは Logs だけで、Train・Eval・Q-Values は `PinButton(false)`、補助 pane（Evaluation Trace・HeatMap・Conv2d）は指定が無く pin 無し。toolbar pane は caption を持たない（2026-11-01 に確かめた） | 同 5452〜5476 行（`OnPaneButton`）、5061〜5065 行（`OnLeftUp`）、954〜964 行（`ProcessMgrEvent`）、`src/aui/floatpane.cpp` 45 行・215〜219 行、[RunnerFrame.cpp:472](../../../apps/runner/src/RunnerFrame.cpp)、:483、:496、:511、:520、:1320、:1354 |
| 27 | window の縁の帯へ落とした pane を wxAUI は最外層の新しい dock に置く（Centre pane の縁の帯は §1.1 の 29）。layer は、その方向と隣り合う 2 方向の fixed でない dock の最大 layer に 1 を足した値で、toolbar の dock は fixed なので数えない。Runner の配置では、左の縁なら Left の layer 1（Bottom の Logs の layer 0 の外側で、全高になる）、右の縁なら表示中の最も外側の Right の dock の layer に 1 を足した値（補助列が出ていれば 21、Q-Values だけなら 11、どちらも無ければ 1）。既存の dock の上に落とすとその dock に入る。幅のポリシーは遷移（表示トグル・✕・restore・補助 pane の追加・リサイズ・Reset）でだけ走り、落とした直後は走らない（2026-11-01 に確かめた） | 同 3533〜3602 行（`DoDrop`）、123〜133 行（`GetMaxLayer`）、[ADR 0016](../../adr/0016-runnerframe-aui-layout-roundtrip.md) の Consequences |
| 28 | `LayoutAll` は layer ごとに「Top の dock ／ Left の dock・内側・Right の dock ／ Bottom の dock」の入れ子を内側から作る。Top・Bottom の dock の横幅は、同じ layer 以下の Left・Right の dock を含む幅になる。したがって Top・Bottom の dock が Centre と同じ幅になるのは、表示中の Left・Right の dock がすべてその dock より外側の layer にあるとき（2026-11-01 に確かめた） | 同 2968〜3051 行（`LayoutAll`） |
| 29 | Centre pane の縁の帯（厚みは `auiNewRowPixels` の 40 DIP と pane の幅・高さの 20% の小さい方）に落とすと、その方向の layer 0 の新しい row に入る。実機の画像で確かめた配置（Eval が Centre、Train がその下、右に Q-Values と Trace の列）は、Eval の下の帯に落とした Bottom の layer 0 で、右の列（layer 10・20）の内側にあるので Centre と同じ幅（2026-11-01 に確かめた） | 同 3503 行（`auiNewRowPixels`）、3800〜3836 行（`DoDrop`） |

### 1.2 何が起きるか

- Train View を隠す手段が無い。`View > Train View FPS` を `0 (Off)` にしても pane は主領域を占めたままで、Eval だけを見たいときに Eval は右半分に留まる。Atari の online では Train View は常に描かれないまま主領域を占めている（§1.1 の 16）。
- 既存の回避策は Eval の maximize だが、Logs・Q-Values・Evaluation Trace も消え、起動時に選べず、Train の timer は回り続ける（§1.1 の 23）。
- 仮に Centre の Train を隠しても、wxAUI は中央を背景で埋め、右の Eval は固定幅のまま広がらない（§1.1 の 2）。Centre を占める pane を切り替えないかぎり、隠す意味が無い。
- Train View は Centre にいるので caption を引いても動かず、pin ボタンも無いので float できない（§1.1 の 6・26。初版と 1 回目の再グリルは「caption を引くと float でき、Centre が空いて背景が見える」と書いていたが、2026-11-01 の実機で起きないことが分かった）。
- 初版の実装（2026-10-31）では、Train View は表示中は常に float できない。Centre を占める pane を float 不可にし（初版の D2）、docked の Train を常に占有者にした（初版の D1）帰結で、Centre の占有者は切り替わるのに、動かせない側が Train に固定されていた。利用者が思い描いた規則は「float すると Centre が空く（表示中で docked の相方がいない）ときだけ float 不可」で、Train か Eval かは問わないものだった（2026-11-01）。
- 1 回目の再グリルの改訂（2026-11-01。Codex が実装）は、両方の主 pane が docked なら両方を Floatable にしたが、Centre を占める pane は caption ドラッグが始まらない（§1.1 の 6）ので、実機では Centre の Train View を外せなかった。相方がいれば Centre の pane も外せる、という利用者の規則を満たす操作が無かった（2026-11-01、2 回目の再グリル）。
- 2 回目の改訂の実装（2026-11-01。Codex）のあとの実機で、Train を pin で外して左の縁へ落とし Eval の隣に置くと、見た目は定位置と同じなのに等分にならない。Eval を外して右の縁へ戻しても、位置を逆にしても同じ。等分の条件が「Eval が Right の layer 0、Train が Centre」の 1 配置で、縁へ落とした pane は最外層の dock に入るため（§1.1 の 10・27）。また、どの主 pane が Centre かが画面で分からない（2026-11-01、3 回目の再グリル）。
- 3 回目の改訂の実装（2026-11-01。Codex）のあとの実機で、Eval の下の帯に Train を落として縦に並べると、横と違って高さが等分されない。横の等分は Left・Right の dock だけを対象にしていた（D15）。利用者の求めは、帯の幅が Centre と同じときだけ高さも等分すること（2026-11-01、4 回目の再グリル）。

## 2. ゴールと非ゴール

**解く問題:** Train View が wxAUI の Centre pane に固定されていて隠せず、Train を裏で任せて Eval だけ見たいとき、Train を流さず Eval だけ流したいときに、Train View が主領域を占めたままになること。初版の実装のあとは、Train View だけが常に float できないこと（§1.2）。

**ゴール**

- G1: Train View を隠せる。Centre を占めていた pane が隠れるか float すると、表示中で docked の相方が Centre を占め、元の占有者が docked のまま戻れば相方は昇格前の位置へ戻る。
- G2: Train と Eval のどちらか 1 つは必ず Centre にある。✕・Panels ツールバー・View メニューの導線から Centre を空にする操作ができず、ドラッグでも pin でも最後の 1 つ（相方が表示中で docked でない主 pane）は動かせない。相方がいれば Centre を占める pane も caption の pin ボタンで float でき（caption ドラッグは wxAUI が Centre では始めない。§1.1 の 6）、空いた Centre は相方が埋める。導線の見え方が追いつく前（最大 200 ms）の操作は判定で止める。知らない経路で空いたときは保険が埋め、WARN で気づける。
- G3: Train View を隠している間は描画の timer が止まる。学習は続く。再表示で、そのとき選ばれている FPS で再開する。
- G4: 起動時にどの主 pane を見せるかを設定で選べ、既定は Eval だけ。Reset Layout はその配置へ戻る。
- G5: Train が表示中の既存の振る舞いを変えない。Eval の float・maximize・✕、Train:Eval=50:50、幅の記憶、Eval の step と resume で Eval を出す動き、Logs・Q-Values・補助 pane の配置規則である。意図して変える点は 6 つで、Train View も Eval View と同じ条件で float できる（D2。Centre を占める pane は caption の pin ボタンで）、float した pane を縁の帯へ落とすとその場所に留まって Centre は相方が占めたまま（D1）、maximize 中の pane をメニューで隠すと先に maximize が解ける（D9。今の Eval の空画面の癖も直る）、起動直後の配置が Eval だけになる（D7）、Logs 以外の pane の caption にも pin ボタンが付く（D14）、等分の条件を「一方が Centre、他方が Left か Right の dock、または Centre と同じ幅の Top か Bottom の帯」に広げて主 pane の置き場が変わったときは 200 ms 以内に当てる（D15）。
- G6: どの主 pane が Centre を占めているかが、caption の印で分かる。印は目立たなくてよい。

**非ゴール**

- NG1: Train と Eval を 1 つの pane にまとめて wxSplitterWindow で分ける形。消える案は「主領域 1 pane 化」で、Eval の float・単独 maximize・✕ を失い、表示切替が pane 系と splitter 系の 2 系統になるので採らない。本当の制約ではなく、設計の選択（§4.1）。
- NG2: Train View を隠している間の Train 側 observer の切り離し。observer のコストは `Train View FPS` が 0 でも残る既存の挙動で、`AttachScoped` の Detach バグ（§1.1 の 21）の修復が先に要る。別の関心として外す。本当の制約ではない。
- NG3: Eval pane を隠したときの Eval の timer 停止（§1.1 の 22）。Train の定期停止の増幅要因としての対策は別件で、本 PRD の「隠す」とは別の関心。本当の制約ではない。
- NG4: 利用者の操作（ドラッグ・✕）の意味として、隠れている主 pane を出したり float を解いたりすること。消える案は「float 中の相方を ✕ で Centre へ引き込む」「Centre が空いたら隠れている方を出す」で、利用者が明示的に決めた配置を別の操作の副作用で覆すので採らない（ユーザーの裁定）。知らない経路で空いたときの保険（D13）は、導線の意味ではなく最後の砦として置く。本当の制約ではない。
- NG5: Logs・Q-Values・補助 pane（HeatMap・Conv2d・Evaluation Trace）の配置規則と、[PRD109](109_runner_startup_background_init_10prd.md) の起動の流れ。別の PRD の関心で、制約ではない。caption に pin ボタンを足すこと（D14）はボタンの有無であって配置規則ではないので、この非ゴールに当たらない。
- NG6: Atari の online で Train View FPS を 0 にしている設定の見直し。描画コストと「Train の進みを目で見るか」の両方が絡む別の関心。既定が `eval` になるので、Train View が空のまま主領域を占める不便は消える。本当の制約ではない。
- NG7: サッシで変えた主 pane の幅の比率の記憶。今と同じく、遷移と主 pane の置き場の変化で等分に戻す。消える案は「比率を覚えて保つ」で、別の関心として外す。本当の制約ではない。

## 3. 成功の測り方

- G1: GUI テストで、Train View を OFF にすると Eval の dock の方向が Centre になり、ON に戻すと Eval が昇格前の位置に戻る（Left か Right の dock なら等分）。Train を側面に置いて Eval を OFF にすると Train が Centre へ移り、Eval を ON にすると Eval が Centre を取り戻して Train は側面に戻る（§7 の受入 2・3）。実機で、Train を隠した Eval が主領域の全幅を使う（受入 8）。
- G2: GUI テストで、相方が隠れているか float 中のときに ✕ も pin も無く、メニューとツールバーのトグルが無効で、Floatable が false である。相方が表示中で docked なら、Centre を占める pane でも Floatable が true で pin があり、pin を押したことにすると float して相方が Centre へ移り、縁の帯へ落として dock するとその場所に留まる。Eval を float した直後の最大 200 ms の間に ✕ を押したことにしても Train は隠れず、pin を押したことにしても Train は docked のまま。maximize 中の手順でも主 pane のどちらかが Centre にある。両方の主 pane を AUI の直接操作で隠すと、保険が Train を出して WARN が 1 行出る（受入 2〜4・6）。
- G3: GUI テストで、隠している間は `TrainPanel` の timer が止まり、再表示で選ばれている FPS の周期で再開する。隠したまま FPS を変えても止まったまま。Reset Layout で出しても再開する（受入 2 の h・3）。
- G4: 既定で起動すると Eval が Centre に表示され Train は隠れ、Reset Layout でも同じ配置になる。`train` では Train だけ、`both` では Train が Centre、Eval が Right に 50:50 で出る（受入 2・3）。
- G5: 既存の GUI テストが通り、Train 表示中の Eval の float・maximize・✕ が今と同じに動く（受入 1・2 の f）。Logs・Q-Values・補助 pane は pin が付くだけで配置が変わらない（受入 2 の n・8）。側面の主 pane が Left の dock にいても Right の外側の layer にいても、置き場が変わったあとの宣言で dock の実幅が主領域の半分になり、Centre と同じ幅の Bottom の帯では高さが Centre と等しくなり、Centre より広い Bottom の帯と同居では変わらない（受入 2 の o〜q・s〜v）。縁へ落とした直後に等分になることと、サッシで変えた幅が次の遷移まで保たれることは実機で確かめる（受入 8）。
- G6: GUI テストで、Centre を占める主 pane の caption が `[Centre]` で終わり、もう一方の主 pane（float 中・側面・非表示）と Logs・Q-Values・補助 pane の caption には付かない（受入 2 の r）。実機で、Centre が入れ替わると印が付け替わる（受入 8）。

## 4. 確定事項

| # | 決定 | 理由 |
|---|---|---|
| D1 | **Centre の占有者は 1 つの対称な規則で決める。** Centre を占める主 pane（占有者）は、隠れるか float するまで占有者のまま。Train か Eval かは問わない。占有者が隠れるか float したら、表示中で docked の相方を Centre へ昇格させ、その直前の位置を覚える（D5）。相方が定位置にいても別の dock にいても昇格させる。隠れていた元の占有者が docked のまま（Centre の情報を持って）表示に戻れば Centre を取り戻し、昇格していた相方は覚えた位置へ降格する。float のまま隠れた pane が戻るときは float のまま。float した pane を縁の帯へ落として dock すると、落とした場所に留まり、占有者は変わらない。float 中の pane は昇格の候補に数えず、ポリシーが float を解くことはしない。昇格・降格は遷移の前（基底の `OnPaneHiding`／`OnPaneShowing`、Reset Layout の復帰）で live pane の dock 情報を書き換えて基底の 1 回の Update で正しい配置にし、ドラッグや pin の float のように遷移の前で決めない経路は D3 の 200 ms の UPDATE_UI で同じ規則を当て、ポリシー（ADR 0016 の適用の時機）でも冪等に当て直す | G1・G2。wxAUI は Centre が無いと背景で埋め、側面の dock は広がらない（§1.1 の 2）ので、占有者が離れるには相方が Centre を占める必要がある。占有者を「docked の Train」ではなく「今 Centre にいる方」にするのは、float 可否を Train と Eval で同じにし（D2）、float した pane を落とした場所に留めるため（ユーザーの裁定、2026-11-01。初版の Train 優先は、Train だけが常に float できない帰結を生んだ。§4.1）。隠れていた占有者が取り戻すのは、相方の表示切替で利用者の置いた位置を失わないという裁定（D5）を両 pane に広げたもので、隠れている pane が Centre の情報を持つのは占有者のまま隠れたときと起動時だけ、そのとき相方は最後の 1 つで Centre を離れられない（D3）ので、取り戻す判定は Centre の情報だけで足りる。float を候補にせず float を解かないのは、アンドックは利用者の明示的な操作で、✕ の副作用でそれを覆してはいけないというユーザーの裁定（2026-10-31）。別の dock にいる相方を候補にするのは、docked なら利用者が「frame の中に置く」と決めているため。遷移の前に決めるのは、✕ の経路ではポリシーが `CallAfter` で遅れ、Centre が背景のまま一度描かれうるため（§1.1 の 5・11）。ドラッグの float は wxAUI がイベントを出さず（§1.1 の 7）、200 ms の UPDATE_UI が最も早い時機（§1.1 の 24）。pin の float も、昇格の時機を 1 つに保つために同じ時機で当てる（D2）。依存: D2・D3・D5・D13 |
| D2 | **主 pane は、相方の主 pane が表示中で docked のときだけ float でき、Centre を占める pane は caption の pin ボタンで float する。** Floatable は D3 の「離れてよいか」の判定（相方の主 pane が表示中で docked）と同じ値にし、Centre を占めているかは見ない。両方が表示中で docked なら両方 true、相方が非表示か float なら false。pin ボタン（`PinButton`）も Floatable と同じ値で描く。付け替えは D3 と同じく、状態から宣言する。側面の主 pane はドラッグでも pin でも float でき、Centre を占める主 pane は pin だけで float できる（caption ドラッグは wxAUI が始めない。§1.1 の 6）。float 中の pane の相方は必ず表示中で docked なので、float 中の値に意味は無い。pin の押下は、判定が否のときに止める（D3）ほかは wxAUI に任せ、float のあとの昇格はドラッグと同じ D3 の 200 ms の UPDATE_UI で当てる | G2。caption ドラッグは Centre の pane では始まらず（§1.1 の 6）、pin の経路には Centre の判定が無く Floatable だけを見る（§1.1 の 26）ので、Centre の pane を float する操作は pin ボタンで足りる。`Movable(false)` は効かない（§1.1 の 8）ので、Centre を守るフラグは `Floatable(false)` の 1 つで足りる。pin を Floatable と同じ値で描くのは、wx が Floatable でない pane の pin を無視する（§1.1 の 26）ので、描いたまま残すと押しても何も起きないボタンになり、§4.1 で退けた「✕ を描いたまま Veto だけで守る」と同じになるため。✕ と同じ判定にするのは、「最後の 1 つは Centre を離れられない」という 1 つの規則で ✕・ドラッグ・pin を決め、Train と Eval で条件を分けないため（ユーザーの裁定、2026-11-01）。`wxAuiManager` を派生させて Centre の pane の caption ドラッグを始める形は、wx の判定を迂回する実装の歪なので入れない（ユーザーの裁定、2026-11-01。§4.1）。pin の押下で昇格を先回りしないのは、昇格の時機を D3 の 1 つに保つため。初版の「Centre を占める pane は float 不可」は、docked の Train を常に占有者にする初版の D1 と合わさって Train だけが常に float できない帰結を生み（§1.2、§4.1）、1 回目の再グリルの改訂は Floatable を付けるだけで Centre の pane を外す操作が無かった（§1.2）。依存: D1・D3・D14 |
| D3 | **隠す導線と float は、相方の主 pane が表示中で docked のときだけ有効にし、4 か所が同じ判定を読む。** 「この主 pane を今 Centre から離してよいか（隠す・float）」の判定（相方の主 pane が表示中で docked。maximize 中は `IsPaneOpen` の値で見る）を基底 `AuiLayoutFrame` の 1 つの口（派生が上書きする。既定は常に可）にまとめる。(1) UPDATE_UI: Train View・Eval View のメニューとツールバーは、表示中の pane を隠す向きのとき、判定が否なら `Enable(false)`（チェックは付いたまま）。隠れている主 pane を出す向きは常に有効。(2) トグルのハンドラ: 隠す向きで判定が否なら何もしない。(3) ✕: `wxEVT_AUI_PANE_CLOSE` を Veto する。(4) pin: `wxEVT_AUI_PANE_BUTTON` の pin の押下は、判定が否なら frame が受け取って wxAUI へ渡さない（float しない）。✕ を描くか（`CloseButton`）、pin を描くか（`PinButton`）、Floatable（D2）、caption の印（D16）は、frame 自身の既存の 200 ms の UPDATE_UI で、主 pane のあるべき値を今の状態から求め、変わったときだけ `CallAfter` で `Update` して描き直す。同じ時機で D1 の占有の規則も当て直し（ドラッグや pin の float で空いた Centre を相方が埋める）、主 pane の置き場が前回と変わっていれば幅のポリシーも当てる（D15）。ポリシーの適用の時機でも同じ値を当てる | G2。ドラッグの float・dock は wxAUI がイベントを出さず（§1.1 の 7）、ポリシーの時機だけで付け外すと、Eval を float したあとも Train の ✕ が残って押せる穴が開く。メニューイベントは UPDATE_UI の無効を通らずに届く（§1.1 の 11。PRD109 D6 も同じ理由でハンドラ側でも確かめている）。pin の押下は `wxEVT_AUI_PANE_BUTTON` として frame に先に届く（§1.1 の 26）ので、✕ の Veto と同じ守りを同じ判定で置ける。状態から宣言する形は Always on Top と同じ流儀で、専用の timer やイベントを足さないので PRD055 D3 と合う。✕ や pin を残して止めるだけで守る形と違い、描かれた ✕ や pin を押しても何も起きない時間は最大 200 ms に限られる。ADR 0016 が許容した「pane ドラッグ直後の自動補正は次の遷移まで遅れる」は、幅のポリシーについてはそのまま残し、✕・pin・Floatable だけが 200 ms 以内に追従する。dock 幅の制御の仕組み（往復と遷移時同期）は変えないので、0016 の判断と矛盾しない。依存: D1・D2 |
| D4 | **Train View の導線を足す。** Train pane を `CloseButton(true)` にし（描くかは D3 が決める）、View メニューに `&Train View` のチェック項目を復活し、Panels ツールバーに「Train View」のチェックボタンを足す。`ID_TrainPanel` の UPDATE_UI を Eval などと同じ束に入れ、有効条件は D3 の判定にする | G1。Eval View と同じ 3 つの導線にそろえる。pane 登録（`RegisterPaneMenu`）は済んでいる（§1.1 の 1） |
| D5 | **降格先は昇格前の位置。** 相方を昇格させるとき、その直前の dock の位置（方向・layer・row・位置）を frame が覚え、元の占有者が戻ったらそこへ降格させる。Train・Eval のどちらが昇格しても同じ。覚えた位置が無いとき（`eval` で起動したときなど）は Eval の定位置 Right(layer 0, row 0, 位置 0)。Train が昇格するのは側面の dock にいるときだけなので、Train の降格先には常に覚えた位置がある。Reset Layout は覚えた位置を消す。降格後の幅は D15 の等分の規則が決める（降格先が Left か Right の dock なら幅を、Centre と同じ幅の Top・Bottom の帯なら高さを等分し、それ以外は触らない）。占有者が 1 つだけ docked の間の幅不足時の縮小は、主領域の最小幅を Centre を占める pane の最小幅（200）で数え、縮める順（補助列→Q-Values 列）は変えない。側面の主 pane の列は縮小の対象にしない | 利用者が置いた dock の位置を、相方の表示切替の副作用で失わない（D1 の理由と同じ理屈。ユーザーの裁定、2026-10-31。両 pane に広げたのは 2026-11-01）。位置の記憶を frame に持つのは、昇格で pane の dock 情報を書き換えるので pane 自身には残らないため（幅は ADR 0016 のとおり `pane.dock_size` に残る）。縮小の順を変えないのは G5。側面の主 pane の列の幅は、2026-11-01 の 3 回目の再グリルで「定位置以外は wx の配分に任せる」決めを改め、D15 が決める。依存: D1・D15 |
| D6 | **Train View の描画 timer は、表示状態と FPS から 1 か所で決める。** ポリシーの入口（maximize のガードより前）で、Train pane の表示状態（`IsShown`）を `TrainPanel` へ渡す。`TrainPanel` は「表示中か」と「選ばれている FPS」の 2 つから timer を回すかを 1 か所で決め、`SetFps`・`Initialize`・表示状態の変化のどれからも同じ判定を通す。隠れている間は timer を止め、`SetFps` は値を覚えるだけにする。表示に戻ったら、その時点の FPS で始める（0 なら始めない）。`Initialize` 前は表示状態を覚えるだけで timer に触らず、`Initialize` のときにその状態で決める。Train 側の observer は外さない（NG2） | G3。経路ごとに呼ぶ形（`OnPaneHiding`／`OnPaneShowing`）では、Reset Layout（§1.1 の 18）と今後足す経路で漏れる。ポリシーは表示トグル・✕・Reset・maximize／restore・リサイズ・起動のすべてで走るので、ここで渡せば全遷移を拾う（ユーザーの裁定、2026-10-31）。`Initialize` 前の安全は PRD109 D6。observer を外さないのは §1.1 の 21 と、FPS 0 でも残る既存の挙動との整合。依存: D4 |
| D7 | **起動時の主 pane を `app.main_view` で選び、既定は `eval`。** 値は `train`（Train だけ）、`eval`（Eval だけ。既定）、`both`（Train を Centre、Eval を Right に 50:50）。`eval` では Train を Centre の情報を持った非表示で作り、Eval が Centre を占める（Train View を ON にすると D1 で Train が Centre を取り戻す）。`both` では Eval を表示して作る。比べ方は `app.eval_panel.model_sync.mode` と同じ（前後の空白を除いて小文字にする）。それ以外の値は起動時に止める（§5.3）。読みと検証は `RunnerApp::Config` | G4。既定を `eval` にするのはユーザーの裁定（2026-10-31。Train View は最近ほとんど見ておらず、学習の進みは Step 表示とログで足りる）。1 つのモード値にするのは、Train と Eval の表示を別々の真偽値にすると「両方非表示」という不正な組が生まれ、検証で弾く必要が出るため（ユーザーの裁定）。キーを app 直下の葉にするのは、Frame 全体の選択で `train_panel`・`eval_panel` のどちらにも属さず、Frame 用の階層も無いため。値の語は `app.train_panel`・`app.eval_panel` と利用者に見える「Train View」「Evaluation View」に揃えた（impl.md 名前の 3・4。ユーザーの裁定）。比べ方を `model_sync.mode` に揃えるのは、同じ app の列挙の設定で扱いを分けないため。依存: D1 |
| D8 | **Reset Layout は `app.main_view` の配置へ戻す。** `RestoreDefaultPanes` は起動時と同じ表示（`train`／`eval`／`both`）に戻し、覚えた位置（D5）を消す。float 中の pane も dock に戻す。占有者・Floatable・✕・pin は D1〜D3 で決まり直し、Train の timer は D6 で決まり直す | Reset Layout を「起動直後の配置へ戻す」という 1 つの意味に保つ。固定の既定へ戻すと、設定を変えた Run で Reset を押すと起動時と違う配置になる（ユーザーの裁定、2026-10-31）。依存: D5・D6・D7 |
| D9 | **maximize 中の pane をメニューで隠すときは、先に maximize を解く。** 基底の `OnTogglePaneMenu` が、隠す向きで当の pane が maximize 中なら restore してから隠す（wx の ✕ と同じ意味）。✕・pin・Floatable の宣言（D3）と Train の timer（D6）は maximize のガードの外で行う。pin で maximize 中の pane を外すときは wxAUI が先に解く（§1.1 の 26）。LoadLayout の往復を使う幅のポリシーは、今と同じく maximize 中は何もしない | G2。今の経路では、Train を maximize したまま Train View を OFF にすると、maximize の印を持ったまま隠れてポリシーも動かず、画面にツールバーだけが残る。同じ癖が今の Eval にもあり（§1.1 の 23）、この決定で直る（意図した変化。G5）。フラグの付け外しと Update には LoadLayout の往復が要らない |
| D10 | **ログは保険の WARN だけ。** pane の開閉・昇格・降格でログを出さない。保険（D13）が発動したときだけ WARN を 1 行出す | 既存の pane の開閉もログを出していない。保険は起きないはずの事態なので、気づける形にする（ユーザーの裁定、2026-10-31） |
| D11 | **ADR を起こす。** [ADR 0066](../../adr/0066-main-panes-occupy-centre-dynamically-and-no-flow-empties-centre.md)「Runner の主 pane は Centre を動的に占有し、Centre を空にする操作を導線に持たない」。2026-11-01 の 3 回目の再グリルでは新しい ADR を起こさず、[ADR 0016](../../adr/0016-runnerframe-aui-layout-roundtrip.md) の先頭に、主 pane の置き場の変化だけは 200 ms の UPDATE_UI で幅のポリシーを当てる旨の注記を足す（本文は書き換えない。D15） | 導線と GUI テストがこの不変条件に依存して戻しにくく、wxAUI の Centre の制約を知らないと「なぜ最後の 1 つは float も ✕ もできないのか」「なぜ Eval が中央へ移るのか」が読めず、1 pane 化・自動で埋める・常に固定を退けた取捨がある（ユーザーの裁定、2026-10-31）。3 回目の再グリルで新しい ADR を起こさないのは、dock 幅の制御の 2 機構（往復と遷移時同期）を変えず、ADR 0016 の判断と矛盾しないため（D3 の理由と同じ）。本文を書き換えずに先頭の注記にするのは、コミット済みの ADR だから（§6 の 27）。注記で狭めるのは Consequences の許容の範囲（主 pane の置き場の変化だけ）で、判断そのものではない |
| D12 | **文書を直す。** [160](../../design/160_applications_and_tools.jp.md) §2.1 の RunnerFrame の行（レイアウトポリシーに Centre の占有の規則、✕・Floatable の状態からの宣言、保険を足す）と §7.1（Train View を隠している間は描画 timer を止め、学習は続くこと。maximize 中の pane をメニューで隠すと先に解くこと）。[020](../../design/020_user_guide_run.jp.md) の Panels ツールバーの一覧に Train View、Train View と Evaluation View の項（既定は Eval だけが表示、Train は Panels か View メニューから出す、最後の 1 つは隠せない）、Reset Layout の説明（`app.main_view` の配置へ戻す。あわせて「frame size を既定へ戻す」という今の記述を、frame の大きさは変えないという実装（§1.1 の 18）に合わせて直す）、`app.main_view` の説明。「描画対象が少ない状態」の段落に、主 pane は両方とも隠せないことを添える。初版の実装が書いた記述のうち、初版の D1・D2・D5 に依るものは次のとおり直す（2026-11-01）: 160 §2.1 の RunnerFrame の行「表示中で docked の Train があれば Train、それ以外は Eval が Centre を占め」は、占有者は隠れるか float するまで保ち、離れたら表示中で docked の相方が占め、戻れば取り戻す、に。160 §7.1 の「Eval の Centre 昇格前の dock 位置は Frame が保持し、Train 再表示時に戻す」は、昇格した pane の昇格前の位置を Frame が 1 組保持し、元の占有者の再表示で戻す、に。同 §7.1 の「Centre を占める pane は float 不可で、相方が開いて docked のときだけ隠せる」と 020 の Evaluation View の項の「中央を占める View はドラッグで移動できない」は、最後の 1 つは動かせない・float した View を縁へ落とすとそこに留まり中央は相方のまま、に。160・020 の en 版（[160 en](../../design/160_applications_and_tools.en.md) §2.1・§7.1、[020 en](../../design/020_user_guide_run.en.md) の Evaluation View の項）の同じ文も同じ内容に直す。検証資材 [testdata/prd110/README.md](../../../core/anet-core/testdata/prd110/README.md) の人が確かめる項の「Train と、Train 非表示時の Eval は caption をドラッグしても動かない」は、受入 8 の 3 つ目の項目（Train の caption をドラッグして float、左の縁へ落とす、Eval View の隠す・出す、Train を隠した状態の Eval は動かない）と同じ手順に直す。[CONTEXT.md](../../../CONTEXT.md) の「主領域」の定義を、Centre を占める pane が隠れるか浮動すると表示中で docked の相方が Centre へ移り、戻れば取り戻すことに合わせて直す（1〜2 文、実装の詳細なし。起草で直し、2026-11-01 の再グリルで改めた）。[wiki 110](../../wiki/apps/110_runner_app.md) は起草と再グリルで書き戻した。2 回目の再グリル（2026-11-01）で直す記述は次のとおり: 160 §2.1 の RunnerFrame の行の「✕・Floatable と占有規則は 200ms の UPDATE_UI で状態から適用する」に pin を足す。160 §7.1 の「両方 docked なら両方を動かせる。側面へ dock した pane はそこに留まり、最後の 1 つは動かせない」は、両方 docked なら両方を float でき、Centre の pane は wxAUI が caption ドラッグを始めないので caption の pin ボタンで、側面の pane はドラッグでも pin でも float し、pin は Floatable と同じ値で描き、宣言が追いつく前の pin は止める、側面へ dock した pane はそこに留まり、最後の 1 つは pin も ✕ も無く動かせない、に。020 の Evaluation View の項の「両方の View をウィンドウ内に表示した状態なら、どちらもドラッグで float できる。Train を float すると Eval が中央に広がる。Train を左の縁へ落とすと左の列に留まる。最後の 1 つはドラッグでも移動できない」は、中央の View は caption をドラッグしても動かず外すには caption の pin ボタンを押す、側面の View はドラッグでも pin でも外せる、外した View を左の縁へ落とすと左の列に留まり中央は相方のまま、最後の 1 つには pin も ✕ も無い、に。020 の pane の一覧の前に、各 pane の caption に pin ボタンがあり押すと floating window になる旨を 1 文足す。160・020 の en 版の同じ文も同じ内容に直す。検証資材 README の人が確かめる項の 2・3 は受入 8 の 2・3 つ目（pin で外す）と同じ手順にし、pin の有無の項（受入 8 の 5 つ目）を足す。[CONTEXT.md](../../../CONTEXT.md) の「主領域」は float の手段を書いていないので変えない。wiki 110 は 2 回目の再グリルで書き戻した。3 回目の再グリル（2026-11-01）で直す記述は次のとおり: 160 §2.1 の RunnerFrame の行の「定位置の両 pane だけを 50:50 にする」は、一方が Centre、他方が Left か Right の dock のとき等分にし、主 pane の置き場が変わったときは 200 ms の UPDATE_UI でも当て、Centre の主 pane の caption に `[Centre]` を付ける、に。160 §7.1 の段落に、同じ条件と時機、サッシの幅は次の遷移か置き場の変化まで保つこと、印を足す。020 の Evaluation View の項の「Train View も表示すると定位置では左右 50:50 になる」は、両方を表示すると、一方が中央で他方が左右どちらかの列にあるとき、間の列を除いた幅を等分する、落とした直後にも等分する、に。020 の pane の一覧の前の pin の文の後に「中央の View の caption には [Centre] が付く」を足す。160・020 の en 版の同じ文も同じ内容に直す。検証資材 README の人が確かめる項に受入 8 の 6〜8 つ目（等分と印）を足し、1 つ目の「50:50」の言い方を等分に揃える。[CONTEXT.md](../../../CONTEXT.md) の「主領域」は、一方が Centre、他方が左右の dock に表示されている間だけ等分の対象、に直す（1〜2 文、実装の詳細なし。起草で直した）。[ADR 0016](../../adr/0016-runnerframe-aui-layout-roundtrip.md) の先頭の注記は起草で足した（D11）。wiki 110 は 3 回目の再グリルで書き戻した。4 回目の再グリル（2026-11-01）で直す記述は次のとおり: 160 §2.1 の RunnerFrame の行と §7.1 の段落の等分の条件に、Centre と同じ幅の上下の帯なら高さを等分し、帯が Centre より広いときと同居は対象外、高さ不足では Logs を縮めない、を足し、「等幅」を「等分」に。020 の Evaluation View の項の等分の文に、上下に並べたときも帯の幅が中央と同じなら高さを等分し、窓の縁に落として左右の列の下まで広がる帯と、Logs と同じ帯に入れたときは等分しない、を足す。160・020 の en 版の同じ文も同じ内容に。検証資材 README の人が確かめる項に受入 8 の 9〜11 つ目（縦の等分）を足す。[CONTEXT.md](../../../CONTEXT.md) の「主領域」に上下の帯を足し「等幅」を「等分」に（起草で直した）。wiki 110 は 4 回目の再グリルで書き戻した | 現在の仕様は設計文書を正本とし、利用者の手順は 020 が持つ。020 の Reset Layout の記述は今の実装と合っていないので、同じ段落を直すときに合わせる。用語集は、本 PRD で「主領域」の占有の決まり方が変わるので、同じ語に 2 つの定義が立たないように直す |
| D13 | **保険。** ポリシーの最後に、表示中で docked の主 pane が 1 つも無ければ Train を表示して Centre に dock し（float 中なら dock する）、`LOG::warn()` で英語の WARN を発動ごとに 1 行出す（`No main pane occupies the centre. Showing Train View as a fallback.`）。表示中で docked の主 pane が両方とも Centre の情報を持つときも保険とし、Train を Centre に残して Eval を覚えた位置（無ければ定位置）へ降格し、`Both main panes occupy the centre. Moving Evaluation View aside as a fallback.` を 1 行出す。D1〜D3・D9 で既知の経路は塞ぎ、両方が Centre になる状態は隠れていた占有者が戻る遷移のフックが解消するので、保険は知らない経路（wx の更新、今後足す導線、テストの直接操作）でだけ動く | G2。既知の穴は塞いで受入で確かめるが、Centre が空く状態を残さない最後の砦を 1 か所に持ち、動いたら気づけるようにする（ユーザーの裁定、2026-10-31）。AGENTS.md §Fail-Fast 原則の表では、主 pane が両方 docked でなくなる状態は「契約違反の内部状態」に当たるが、画面の配置の不整合で学習の結果にも Run の記録にも影響せず、Run を落とすより Centre を埋めて知らせる方が利用者に有利なので、「自動選択の不履行」と同じ「WARN して実行可能な形へ fallback」の扱いにする。発動は起きないはずの事態なので、発動ごとに 1 行出しても spam にならない。導線の意味としての自動表示（NG4）とは別。依存: D1・D10 |
| D14 | **pin ボタンを全 pane に付ける。** Logs は今のまま。Train・Eval は D2 の値で描く。Q-Values と補助 pane（Evaluation Trace・HeatMap・Conv2d）は常に描く。toolbar pane は caption が無いので対象外。pin で外した floating frame の位置は wxAUI に任せる（初回は OS 既定の位置、動かせば次からはその位置） | ユーザーの裁定（2026-11-01: 一部だけ付けるのは変。実機で操作して確かめたい）。pin は wxAUI 既定のボタンで、float できる pane にはどれも同じ意味を持つ（§1.1 の 26）。位置を決めないのは Logs の pin と同じ扱いで、専用の処理を足さないため。配置規則（NG5）は変えない。依存: D2 |
| D15 | **等分の条件は向きで決め、置き場が変わったら 200 ms で当てる。** 両主 pane が表示中で docked で、(1) 一方が Centre、他方が Left か Right の dock（layer・row は問わない）のとき、側面の主 pane の dock の幅を、主領域（client 幅から、その dock 以外の表示中の縦の dock（Left・Right）の幅を除いた幅）の半分にする。(2) 一方が Centre、他方が Top か Bottom の dock で、表示中の Left・Right の dock がすべてその dock より外側の layer にある（帯が Centre と同じ幅。§1.1 の 28）とき、その dock の高さを「Centre の高さとその dock の高さの和」の半分にする。ほかの Top・Bottom の帯（Logs・toolbar）の高さは予算に入れず、縮めない。高さが Train と Eval の最小高（200+200）に足りないときは等分せず、最小高で clamp する。(1)(2) とも、主 pane と同じ dock に補助 pane や Logs が同居する列・帯は対象外で、wx の配分に任せる。Left・Right の dock がその dock と同じか内側の layer にある Top・Bottom の帯（Centre より広い。window の縁に落とした最外層など）も対象外。(1) の幅不足時の最小幅（200+200）と縮小順（補助列→Q-Values 列）は変えない。当てる時機は、今の遷移（表示トグル・✕・restore・補助 pane の追加・リサイズ・Reset）に加えて、D3 の 200 ms の UPDATE_UI で主 pane 2 つの置き場（方向・layer・row・位置・docked か float か・表示中か）が前回と変わったとき 1 回。サッシで変えた幅や高さは置き場が変わらないので触らず、次の遷移か置き場の変化で等分に戻す（今と同じ）。maximize 中は当てない | G5（意図した変化）。縦の帯を足すのは、Eval の下の帯に落として縦に並べたときも等分したいというユーザーの求め（§1.2、2026-11-01）。帯の幅が Centre と同じときに限るのは、左右の列の下まで広がる帯（最外層）は Centre と組になっていないため（ユーザーの求め）。高さ不足で Logs を縮めないのは、Logs を縮める挙動が既存に無く G5 を広げるため（ユーザーの裁定、2026-11-01。§4.1）。等分の条件を「Eval が Right の layer 0、Train が Centre」に固定すると、縁へ落とした主 pane は最外層の dock に入り（§1.1 の 27）、見た目が同じでも等分にならず、位置が逆でも効かない（§1.2。ユーザーの報告、2026-11-01）。「Centre に隣接する側面の dock に限る」案は間に Q-Values 列があると効かず、「明示の操作にする」案は Train View を出した既定の見え方を変え、「定位置へ引き戻す」案は D1 を覆す（§4.1。ユーザーの裁定: 案 1、2026-11-01）。落とした直後に当てるのは、ドラッグで落とす操作が遷移に当たらず（§1.1 の 27）、次の遷移まで「効かない」と見えるため。置き場の変化だけを合図にするのは、サッシの操作を覆さず、往復を置き場が変わったときに限るため（ユーザーの裁定: A、2026-11-01）。ADR 0016 の「ドラッグ直後の補正は次の遷移まで遅れる」の許容は、主 pane 以外（補助 pane の列、サッシ）に残る（D11 の注記）。(1) の幅の等分から Top・Bottom を外すのは、dock の size が高さになり幅の等分に意味が無いためで、Top・Bottom の帯は (2) で高さを等分する。同居列を外すのは今と同じ扱い（§1.1 の 10）。依存: D1・D3・D5 |
| D16 | **Centre を占める主 pane の caption の末尾に `[Centre]` を付ける。** caption の本体（"Train View"、"Evaluation View"）は変えず、Centre を占めている間だけ半角スペースと `[Centre]` を足す。float 中・側面・非表示の主 pane と、Logs・Q-Values・補助 pane には付けない。付け外しは D3 の 200 ms の UPDATE_UI で状態から決め、変わったときだけ描き直す | G6。どの主 pane が Centre かは配置を見ても分からず（§1.2）、ADR 0066 の「なぜ Eval が中央へ移るのか」が画面で読めない。pane 自身の caption に付けるのは、離れた配置でも目線の先にあり、✕・pin・Floatable と同じ宣言で決まって新しい仕組みが要らないため（ユーザーの裁定: caption に印、2026-11-01。ステータスバー・Panels のボタン・caption の色は §4.1）。文言は文書と wxAUI の語に合わせた `[Centre]`（ユーザーの裁定、2026-11-01。`(Centre)`・`(centre)`・記号・中点は §4.1。`(Main)` は用語集が避ける語）。目立たなくてよい（ユーザーの求め）。依存: D1・D3 |

### 4.1 退けた案

| 案 | 退けた理由 |
|---|---|
| Train と Eval を 1 つの Centre pane にまとめ、wxSplitterWindow で分ける（以前の案） | 比率はリサイズに自動で追従し、Centre が構造上空かないが、Eval の float・単独 maximize・✕ を失い、表示切替が pane 系（基底の `AuiLayoutFrame`）と splitter 系の 2 系統になる。性能は同等（効くのは timer の停止で、どちらでも同じ）。G5 を保てる分だけ動的 Centre が有利 |
| 両方を Centre dock に置く（別の row で横に並べる） | Train を隠したときの Eval の全幅と 50:50 は wx の伸縮だけで決まるが、sash が無く比率を変えられない（今の 50:50 も遷移のたびに比率を戻すので、この損は小さい）。float から戻した Eval は側面の dock にしか落ちない（§1.1 の 6）ので結局昇格が要り、Q-Values・補助列の縮小、幅の記憶、既存のテストを組み替える割に単純にならない |
| Eval の maximize で足りるとする | Logs・Q-Values・Evaluation Trace も消え、起動時に選べず、Train の timer は回り続ける（§1.1 の 23） |
| Train を細い帯に畳む見かけの非表示 | 右 dock は 85% まで（§1.1 の 9）で Eval が広がりきらず、帯が空白のまま残る |
| ✕ を描いたまま残し、close の Veto だけで守る | 押しても何も起きないボタンが常に残る。D3 は ✕ を状態から描き分け、Veto は見え方が追いつく前の最大 200 ms の保険に限る |
| ✕ と Floatable をポリシーの適用の時機だけで付け外す（最初の草案） | ドラッグの float・dock は wxAUI がイベントを出さず（§1.1 の 7）、Eval を float したあとも Train の ✕ が残って押せる |
| `wxEVT_AUI_RENDER` でドラッグを監視する | 描画のたびに走り、ADR 0016 で全廃した方式に戻る |
| `wxAuiManager` を派生させて float・dock の瞬間を掴む | `CreateFloatingFrame` は `Update` の中で呼ばれ、`ProcessDockResult` はドラッグ中のヒント計算でも呼ばれるので合図に使えず、基底の `aui_mgr_` の型と wx 内部の呼び順に依存する。ドラッグで float した直後の昇格を 200 ms より早くする目的（§6 の 17）でも同じ理由で退ける |
| float 中の相方を、Train の ✕ で Centre へ引き込む | 一度推奨したが、ユーザーが撤回。アンドックは明示的な操作で、✕ がそれを覆してはいけない（D1・D3） |
| Centre が空いたら隠れている方を出すことを、導線の意味にする | 利用者が決めた配置を別の操作の副作用で覆す。知らない経路で空いたときの保険（D13）としてだけ置く |
| 主 pane を常に `Floatable(false)` にする | Train 表示中の Eval の float・移動の自由を失う。相方の有無で決めれば足りる（D2） |
| Train だけ `Floatable(false)` にする | Train が隠れて Eval が Centre のとき、Eval を float すると Centre が空く |
| Centre を占める pane を常に `Floatable(false)` にし、docked の Train を常に占有者にする（初版の D1・D2） | 2 つが合わさると Train だけが表示中は常に float できず、Centre の占有者が切り替わる設計なのに動かせない側が Train に固定される（§1.2）。Centre を守る目的は「最後の 1 つは離れられない」で足りる（D2。ユーザーの求めによる 2026-11-01 の再グリル） |
| 占有者を Train 優先のままにし、相方がいれば float だけ許す（再グリルの案 (a)） | float した Train を縁の帯へ落とすと、最大 200 ms で Centre へ跳んで Eval が覚えた位置へ戻り、Train を側面に置き続けられない。占有者を「今いる方」にすれば落とした場所に留まり、Train と Eval の条件分岐が対称になる（D1。ユーザーの裁定、2026-11-01） |
| `wxAuiManager` を派生させ、Centre の pane の caption 押下を掴んでドラッグを始める（2 回目の再グリルの案） | wx の protected の状態（押下の種類と開始点）と `OnLeftDown` の内部の順に依存する実装の歪で、wx の更新で壊れうる。利用者の求める操作は pin ボタンで足りる（D2。ユーザーの裁定、2026-11-01: wx の仕様を回避する歪を入れてまで達成しない） |
| Centre の占有を先着順にし、位置を入れ替えたいときは隠して出し直す（2 回目の再グリルの案） | Floatable の付け替えは消えるが、既定の `eval` で Train View を出すと Train が右・Eval が中央になって今と左右が入れ替わり、50:50 と CONTEXT.md の主領域を位置で定め直す必要がある。Centre の pane を外す操作が無いままで、入れ替えの手順も遠回り（2026-11-01） |
| Train と Eval を入れ替えるコマンドを View メニューと Panels に足す（2 回目の再グリルの案） | 導線が 1 つ増え、入れ替えだけが別の機構になる。pin で外して縁へ落とせば同じ配置が作れる（2026-11-01） |
| 占有を Train 優先に戻し、Train は動かさない（2 回目の再グリルの案） | Train が常に動かせない非対称に戻る。1 回目の再グリルで退けた理由がそのまま当たる（2026-11-01） |
| wxWidgets のソースの Centre の判定を外す | fork の保守が要り、wxAUI のほかの利用者への影響を負う（2026-11-01） |
| Centre にダミーの pane を置き、主 pane を両方とも側面に置く | Centre の pane が無いので主 pane の伸縮が wx の配分に戻り、50:50・列の規則・幅の記憶を組み替える割に得るものが無い（2026-11-01） |
| 主 pane の pin を常に描き、最後の 1 つでは押しても何も起きないままにする | wx が Floatable でない pane の pin を無視する（§1.1 の 26）ので、押しても何も起きないボタンが残る。✕ と同じく状態から描き分ける（D2） |
| pin の押下（`wxEVT_AUI_PANE_BUTTON`）で相方の昇格を先回りする | 昇格の時機がドラッグ（200 ms の UPDATE_UI）と pin（押下の時点）の 2 つになる。pin の押下は判定が否のときに止めるだけに使う（D2・D3） |
| 等分の対象を Centre に隣接する側面の dock に限る（3 回目の再グリルの案 2） | 間に Q-Values 列や補助列があると効かず、右の縁へ落とした主 pane が対象外になる。「隣接」の判定（同じ側で layer が最小か）も増える（D15。2026-11-01） |
| 自動の等分をやめ、View メニューの操作と Reset Layout だけで等分にする（3 回目の再グリルの案 3） | サッシの幅が遷移で戻らなくなる利点はあるが、Train View を出したとき必ず等分になる既定の見え方（G5）が変わり、ADR 0016 の 50:50 ポリシーの意味も変わる（2026-11-01） |
| 側面に落とした主 pane を定位置へ引き戻す（3 回目の再グリルの案 4） | D1 の「落とした場所に留まる」を覆す。必要性も薄い（ユーザー、2026-11-01） |
| 落とした直後は次の遷移まで待つ（3 回目の再グリルの問い 1 の B） | ドラッグで落とす操作は遷移に当たらず（§1.1 の 27）、落とした直後に「効かない」と見える場面が残る（D15。ユーザーの裁定: A、2026-11-01） |
| Centre の印をステータスバーの欄に出す | pane から離れた場所で、離れた配置だと目線が合わない。実行状態の欄に配置の情報が混ざる（D16） |
| Centre の印を Panels ツールバーのボタンに出す | ラベルの幅が変わり、押下表示（表示中）と印（Centre）の 2 つの状態が 1 つのボタンに乗る（D16） |
| Centre の印を caption の色で出す | dock art の派生が要り、テーマ切替（[999_runner_theme_switch](../953_runner_theme_switch_10prd.md)）と絡む（D16） |
| 印の文言を `(Centre)`・`(centre)`・記号 `◆`・中点 `· Centre` にする | 意味は同じで見た目の差。角括弧は状態の印として慣れがある（D16。ユーザーの裁定、2026-11-01）。`(Main)` は用語集が避ける語、`(Primary)`・`(Anchor)` は新語、`(Pinned)` は pin ボタンと紛れる |
| 縦に並んだときの高さ不足で Logs の帯を縮めて確保する（4 回目の再グリルの案） | Logs を縮める挙動は既存に無く、小さい窓で Logs が勝手に縮む。等分せず最小高で clamp する（D15。ユーザーの裁定、2026-11-01） |
| 帯の幅が Centre より広い Top・Bottom（window の縁に落とした最外層）も等分の対象にする | 左右の列の下まで広がる帯は Centre と組になっておらず、等分の意味が無い。ユーザーの求めでも対象外（D15。2026-11-01） |
| 縦の等分の予算に Logs の帯を含める（Centre・主 pane の帯・Logs の 3 等分や、Logs を除いた高さの等分） | 横で Q-Values の列を予算から除くのと同じく、組になる 2 枚だけを等分する（D15） |
| 降格先を常に定位置にする（最初の草案） | 定位置以外に置いた Eval の位置を、Train の表示切替で失う（D5） |
| 定位置の Eval だけを昇格の候補にする | 配置は保つが、Eval を Left などに置いた人は Train を隠す前に Eval を右へ戻す手間が要る。位置を覚えて戻す形をユーザーが選んだ（D5） |
| Train の timer を `OnPaneHiding`／`OnPaneShowing` で伝える（最初の草案） | Reset Layout はこの 2 つを通らず（§1.1 の 18）、timer が止まったまま、または回ったまま残る（D6） |
| Train の timer を回し続け、tick で `IsShownOnScreen()` を見て描かない | 伝達も記憶も要らず最小だが、G3 の「timer が止まる」を変えることになり、止まっていることを外から読みにくい（D6） |
| Train と Eval の表示を別々の真偽値の設定にする | 「両方非表示」の不正な組を検証で弾く必要が出る。1 つのモード値なら不正な組が無い（D7） |

## 5. 実装契約

### 5.1 状態と見え方

主 pane の状態ごとに、Centre の占有者と各導線の状態を決める。表は宣言が追いついたあと（最大 200 ms）の見え方で、追いつく前の操作は D3 の判定が止める。Train と Eval は対称で、Eval が占有者の行は Train と Eval を入れ替えて読む。pin ボタンは Floatable と同じ値で描く（D2）ので、Floatable の列は pin の有無も表す。Centre の列にある主 pane の caption には `[Centre]` が付く（D16）。

| Train | Eval | Centre | 占有者でない主 pane の位置 | Floatable（pin も同じ） | Train の ✕ | Eval の ✕ | Train のトグル | Eval のトグル |
|---|---|---|---|---|---|---|---|---|
| 表示・docked（占有者） | 非表示 | Train | （非表示） | Train false | 無し | — | 無効（チェック付き） | 有効 |
| 表示・docked（占有者） | 表示・docked | Train | Left か Right の dock なら幅を等分、Centre と同じ幅の Top か Bottom の帯なら高さを等分（D15）。それ以外はそのまま | Train true、Eval true | 有り | 有り | 有効 | 有効 |
| 表示・docked（占有者） | float | Train | float | Train false、Eval true（意味なし） | 無し | 有り（floating frame の ✕） | 無効（チェック付き） | 有効 |
| 非表示 | 表示・docked（占有者） | Eval | （非表示。Train が占有者のまま隠れたなら、Eval の昇格前の位置を覚えている） | Eval false | — | 無し | 有効 | 無効（チェック付き） |
| float | 表示・docked（占有者） | Eval | float（Train が占有者から float したなら、Eval の昇格前の位置を覚えている） | Train true（意味なし）、Eval false | 有り（floating frame の ✕） | 無し | 有効 | 無効（チェック付き） |
| 表示・docked（側面） | 表示・docked（占有者） | Eval | 落とした dock。Left か Right なら幅を等分、Centre と同じ幅の Top・Bottom なら高さを等分（D15）、それ以外は wx の配分 | Train true、Eval true | 有り | 有り | 有効 | 有効 |
| 非表示 | 非表示 | 保険が Train を出す（D13） | | | | | | |
| 非表示 | float | 保険が Train を出す（D13） | | | | | | |
| float | float | 保険が Train を出す（D13） | | | | | | |
| 表示・docked（Centre） | 表示・docked（Centre） | 保険が Eval を降格する（D13） | | | | | | |

下の 4 行は導線からは到達しない。占有者を隠す・float できるのは相方が表示中で docked のときだけで、そのとき相方は Centre へ移って離れられなくなる（D1〜D3）。両方が Centre の情報を持つ状態は、隠れていた占有者が戻る遷移のフックで解消する。到達するのは知らない経路だけで、そのときは保険が埋めて WARN を出す。

### 5.2 操作ごとの振る舞い

| 操作 | 振る舞い |
|---|---|
| View メニューの Train View を OFF、Panels の Train View を OFF、Train の ✕（docked・float とも） | Eval が表示中で docked のときだけ効く（否ならハンドラは何もせず、✕ は Veto）。Train を隠し、timer を止める（D6）。Train が占有者なら、Eval の今の位置を覚えて（D5）Centre へ移し、`Floatable(false)`、✕ 無し、Eval のトグルが無効になる。Train が側面の dock か float なら Eval は動かない |
| Train View を ON | Train を隠す前の状態に表示する。Centre の情報を持つ（占有者のまま隠れた、または起動時から非表示）なら Centre を取り戻し、Eval は覚えた位置へ戻り（無ければ定位置）、`Floatable(true)`、✕ 有り、Eval の位置が Left か Right の dock か Centre と同じ幅の Top・Bottom の帯なら等分（D15）。側面の dock で隠れたならそこに出て Eval は Centre のまま。float のまま隠れたなら float のまま出る。timer は選ばれている FPS で始める（0 なら始めない） |
| Eval View を OFF、Eval の ✕（docked・float とも） | Train が表示中で docked のときだけ効く。Eval を隠す。Eval が占有者なら、Train の今の位置を覚えて（D5）Centre へ移す。Train の ✕ が消え、Train のトグルが無効になる |
| Eval View を ON | Eval を隠す前の状態に表示する。Centre の情報を持つ（占有者のまま隠れた）なら Centre を取り戻し、Train は覚えた位置へ降格する。それ以外（docked なら前の dock、float なら float）は今と同じ |
| Eval の step・resume（ツールバー、右クリック） | 隠れていれば Eval を表示する（§1.1 の 17）。今と同じ |
| Eval を float（側面にいれば caption のドラッグか pin、Centre にいれば pin）、縁の帯へドロップ | Train が表示中で docked のときだけできる（Eval が `Floatable(true)` で pin が描かれる。宣言が追いつく前の pin は止める）。wxAUI は float・dock のイベントを出さないので、次の 200 ms の UPDATE_UI で Train の ✕・pin と Floatable が消える（float）か戻る（dock）。Eval が占有者なら同じ時機で Train が Centre へ昇格し、Train の位置を覚える（それまで最大 200 ms は Centre が背景）。帯へ落とすと落とした場所に留まり、占有者は変わらない。落とした先が Left か Right の dock か、Centre と同じ幅の Top・Bottom の帯なら、次の 200 ms の宣言で置き場の変化を検知し、Train と等分になる（D15）。その前に Train の ✕ を押しても Veto され、pin を押しても float しない。pin で外した floating frame の位置は wxAUI に任せる（D14） |
| Train を float（Centre にいれば pin、側面にいれば caption のドラッグか pin）、縁の帯へドロップ | Eval が表示中で docked のときだけできる（Train が `Floatable(true)` で pin が描かれる）。Centre の Train は caption をドラッグしても動かない（§1.1 の 6）。Train が占有者なら、次の 200 ms の UPDATE_UI で Eval が Centre へ昇格し、Eval の位置を覚える（それまで最大 200 ms は Centre が背景）。帯へ落とすと落とした場所に留まり、Eval は Centre のまま。落とした先が Left か Right の dock なら最大 200 ms で Eval と幅が等分され、Centre と同じ幅の Top・Bottom の帯（Eval の上下の帯に落とした layer 0）なら高さが等分され、それ以外（window の縁の最外層の帯、同居）は wx の配分（D15）。float 中も Train は表示中なので timer は回る（D6） |
| Logs・Q-Values・補助 pane の pin | 押すと float する（Logs は今と同じ）。主 pane の占有・✕・Floatable には関与しない（D14） |
| 主 pane のサッシで幅や高さを動かす | 置き場は変わらないので触らない。次の遷移（表示トグル・✕・restore・補助 pane の追加・リサイズ・Reset）か、主 pane の置き場の変化で等分に戻る（D15。今と同じ） |
| Centre が切り替わる（昇格・降格・取り戻し） | 次の 200 ms の宣言で、Centre を占める主 pane の caption に `[Centre]` が付き、離れた方から消える（D16） |
| Reset Layout | `app.main_view` の配置へ戻し、覚えた昇格前の位置（Train・Eval 共通の 1 組。§6 の 16）を消し、float 中の主 pane も dock に戻す（D8）。toolbar・Logs・Q-Values・補助 pane の戻し方は今と同じ |
| `View > Train View FPS` | Train が隠れている間も選べる。値は覚え、timer は再表示のときにその FPS で始める。表示中の動きは今と同じ |
| maximize・restore | maximize 中の pane をメニューで隠すと先に maximize が解ける（D9）。✕ と pin の経路は今と同じく wx が解く（§1.1 の 5・26）。✕・pin・Floatable・timer は maximize 中も状態に従う。幅のポリシーは maximize 中は動かない |
| 起動 | `app.main_view` に従う（§5.3）。`Initialize` 前の pane の開閉で落ちない（PRD109 D6） |
| フレームのリサイズ | 今と同じ（ポリシーの coalesce）。占有者だけが docked のときは主領域の最小幅を Centre の pane 1 つ分で数える（D5） |
| 知らない経路で主 pane が両方 docked でなくなる、または両方が Centre の情報を持つ | 保険が Train を Centre に出す、または Eval を覚えた位置へ降格し、WARN を 1 行出す（D13） |

### 5.3 設定

| キー | 型・値域・既定 | 振る舞い | 反したとき |
|---|---|---|---|
| `app.main_view` | 列挙 `train`｜`eval`｜`both`、既定 `eval`。前後の空白を除き小文字にして比べる | 起動時の主 pane の表示。`train` は Train を Centre に表示し Eval は非表示。`eval` は Train を非表示にして Eval を Centre に表示。`both` は Train を Centre、Eval を Right に表示して 50:50。Reset Layout もこの配置へ戻す | 3 つ以外の値は起動時に止める。英語のメッセージ `Invalid app.main_view: value=<値> (expected: train, eval, both)`。経路は `app.train_panel.fps` の不正値と同じ（`RunnerApp::Config` の検証） |

共通の既定は [common.txt](../../../apps/runner/config/common.txt) の app の節に `app.main_view ?= eval` で置く。設定に無いときの `RunnerApp::Config` の既定も `eval` で、common.txt の値と同じにする（`app.train_panel.fps` などの既存の既定と同じ持ち方）。構成別に変えるときは `app.online.main_view`／`app.batchrun.main_view` の選択で上書きできる（§1.1 の 16）。Train だけで起動したい Run の設定例は次のとおり。

```text
run.@trainview : app.main_view = train
```

`app.eval_panel.auto_start` との関係は無い（§1.1 の 15）。`eval` で起動しても Eval を進めるかは `auto_start` が決め、進めていない間の Eval View は初期化後の最初の表示のまま止まっている。

### 5.4 失敗の扱い

| 場面 | 振る舞い | 終了コード |
|---|---|---|
| `app.main_view` が 3 つ以外 | 起動時に止める（§5.3） | 今の `app.train_panel.fps` の不正値と同じ |
| 導線の見え方が追いつく前（最大 200 ms）に、隠せない pane の ✕・pin・トグルを操作する | 判定が止める（D3）。何も起きない | — |
| 知らない経路で主 pane が両方 docked でなくなる、または両方が Centre の情報を持つ | 保険が Train を出す、または Eval を降格し、WARN を 1 行出す（D13） | — |

### 5.5 移行の対象

- コード: [RunnerFrame.cpp](../../../apps/runner/src/RunnerFrame.cpp)・[RunnerFrame.hpp](../../../apps/runner/src/RunnerFrame.hpp)（`SetupPanes` の Train の `CloseButton(true)` と `app.main_view` による初期表示、`SetupMenuBar` の Train View の復活、`SetupToolBars` の Panels の Train View、`SetupEvents` の UPDATE_UI の束と有効条件、`OnUpdateTrainStatus` での ✕・pin・Floatable の宣言と占有の規則の当て直し、`SetupPanes` の Train・Eval・Q-Values・Evaluation Trace の pane の pin ボタンと、`OnHeatMap`・`OnConv2d` が作る HeatMap・Conv2d の pane の pin ボタン（D14。§1.1 の 26）、「離れてよいか」の判定の上書き、`ApplyMainPaneWidths` の条件と集計の一般化（側面の主 pane の dock を向きで特定し、Left・Right の縦の dock を集計する。D15 の (1)）と、Top・Bottom の帯の「Centre と同じ幅」の判定と高さの等分（D15 の (2)）、`OnUpdateTrainStatus` での主 pane の置き場の比較と幅のポリシーの適用、caption の印の宣言（D15・D16）、`OnPaneHiding`／`OnPaneShowing` と `RestoreDefaultPanes` での昇格・降格と位置の記憶（Train・Eval とも）、`OnApplyLayoutPolicy` の入口での Train の表示状態の受け渡しと、占有の規則の当て直し、Eval が Centre のときの最小幅、保険）、[gui.hpp](../../../core/anet-core/include/anet/gui.hpp)・[gui.cpp](../../../core/anet-core/src/gui.cpp)（「隠してよいか」の口、`OnTogglePaneMenu` でのハンドラ側の確かめと maximize の解除、`OnAuiPaneClose` の Veto、pin の押下の `wxEVT_AUI_PANE_BUTTON` を判定が否なら wxAUI へ渡さない受け取り）、[TrainPanel.cpp](../../../apps/runner/src/TrainPanel.cpp)・[TrainPanel.hpp](../../../apps/runner/src/TrainPanel.hpp)（表示状態を受け取り、表示状態と FPS から timer を 1 か所で決める。`Initialize` 前は状態を覚えるだけ。timer の状態をテスト用の派生クラスから読めるようにする）、[RunnerApp.cpp](../../../apps/runner/src/RunnerApp.cpp)（`Config` に `app.main_view` の読みと検証、RunnerFrame への受け渡し）。
- 設定: [common.txt](../../../apps/runner/config/common.txt) に `app.main_view ?= eval`。`[shipped_config]` の検査を通す。`run.@evalonly`（[Atari.txt:425](../../../apps/runner/config/Atari.txt)）には足さない（既定が `eval` のため）。
- テスト: GUI テスト `[prd110_main_pane]`（§7 の受入 2〜6。RunnerApp を起動するケースは別プロセス）。初版の実装が書いた [RunnerFrame_test.cpp:246](../../../apps/runner/src/RunnerFrame_test.cpp) の、両方 docked のときの Train の `IsFloatable()` false の断定は true に直す。相方が非表示のときの断定（[:340](../../../apps/runner/src/RunnerFrame_test.cpp)・[:512](../../../apps/runner/src/RunnerFrame_test.cpp)）はそのまま。受入 2 の j〜l と受入 6 の両方 Centre を足す。2 回目の再グリルで、受入 2 の a〜c・f・j に `HasPinButton()` の断定と pin の押下を足し、m・n を足す。pin の押下は ✕ と同じく `wxEVT_AUI_PANE_BUTTON`（button は `wxAUI_BUTTON_PIN`）を frame へ送る。3 回目の再グリルで、受入 2 の o〜r（等分と印）を足す。4 回目の再グリルで、受入 2 の s〜v（縦の等分）を足し、q を Centre より広い帯の場合に限定する。既存の `[prd110_width_exclusions]`（[RunnerFrame_test.cpp:1028-1041](../../../apps/runner/src/RunnerFrame_test.cpp)）は、`both` で起動して Train を pin で外し Bottom の layer 1 に置いて寸法が変わらないことを断定しているが、表示中の Left・Right の dock が無いので D15 の (2) が当たる。Q-Values を表示して Q-Values の列より外側の layer（Centre より広い帯）に置く配置に替え、変わらない断定を保つ（受入 2 の q と同じ）。既存の `[prd110_main_pane]` のうち、側面に置いた主 pane の幅の保持を断定する箇所は D15 で側面の主 pane の幅が主領域の半分になるので、等分の断定（主領域の半分と 2 px 以内）に替える: [RunnerFrame_test.cpp:285-298](../../../apps/runner/src/RunnerFrame_test.cpp) の Eval を Left の layer 2 に置いて Train View を OFF→ON したあとの `DockSize == before_width` と [:308](../../../apps/runner/src/RunnerFrame_test.cpp) の `dock_size == before_width`、[:588-612](../../../apps/runner/src/RunnerFrame_test.cpp) `[prd110_side_owner]` の Train を Left の layer 2 に `dock_size = 420` で置いたあとの `DockSize == width`。置き場（方向・layer・row・位置）の断定はそのまま。既存の `[eval_steps_menu]`・`[prd105_ui_keys]`・`[prd109_ui]` は、Train の表示を前提にしている箇所があれば設定に `app.main_view = train` を書くか前提を直し、`RunnerFrame` を直接作る箇所（§1.1 の 20）は初期表示の指定を渡す。`[prd109_ui]` の「Train は現行 UI では menu が無い」のコメントと AUI の直接操作は、View メニューの Train View での往復に直す。
- 文書: D12 のとおり。直す現用の文書は [160 jp](../../design/160_applications_and_tools.jp.md) と [160 en](../../design/160_applications_and_tools.en.md) の §2.1 の RunnerFrame の行と §7.1 の段落、[020 jp](../../design/020_user_guide_run.jp.md) と [020 en](../../design/020_user_guide_run.en.md) の Evaluation View の項と pane の一覧の前の pin の 1 文、検証資材 [testdata/prd110/README.md](../../../core/anet-core/testdata/prd110/README.md) の人が確かめる項。en 版も現用の範囲に入る（[実装 品質確認観点](../../quality/impl.md) 項番 20・71）。[CONTEXT.md](../../../CONTEXT.md) の「主領域」は起草で直し、再グリルで改め、3 回目の再グリルで等分の条件を向きに改め、4 回目の再グリルで上下の帯を足した。[ADR 0016](../../adr/0016-runnerframe-aui-layout-roundtrip.md) の先頭の注記は 3 回目の再グリルで足した。
- 互換の扱い: Train を隠せない今の形へ戻す設定や切替は作らない。既定が `eval` に変わることへの案内や互換の既定は置かない。
- 直さないもの: 過去の Run の記録と実験記録の config の写し。[PRD109](109_runner_startup_background_init_10prd.md) の本文（§5.2 の「pane の開閉」に Train View を含める記述は、既定の `eval` では Eval が表示中で docked なので、Train View の開閉がそのまま成り立つ）。

### 5.6 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `AuiLayoutFrame` | 変更 | 「この pane を今 Centre から離してよいか」の口を持ち（派生が上書き、既定は可）、メニューの隠す向きのハンドラでその判定を確かめ、maximize 中の pane は先に解いてから隠す。✕ の close イベントは判定が否なら Veto し、pin の押下は判定が否なら wxAUI へ渡さない | 319 GUI 共通基盤 |
| `RunnerFrame` | 変更 | Centre の占有者の規則（占有者の保持、昇格・降格、昇格前の位置の記憶を Train・Eval とも）、「離れてよいか」の判定（相方が表示中で docked）に揃えた Floatable・✕・pin の状態からの宣言と同じ時機での占有の規則の当て直し、全 pane の pin ボタン（D14）、等分の条件と置き場の変化での適用（D15）、caption の印（D16）、Train View の 3 つの導線、`app.main_view` による初期表示と Reset Layout の復帰、Eval が Centre のときの最小幅、Train の表示状態の `TrainPanel` への受け渡し、保険と WARN | 110 RunnerApp |
| `TrainPanel` | 変更 | 表示状態を受け取り、表示状態と FPS から描画 timer を回すかを 1 か所で決める。隠れている間は timer を止め、`SetFps` は値を覚えるだけ。`Initialize` 前は状態を覚えるだけ。observer は変えない | 110 RunnerApp |
| `RunnerApp` | 変更 | `Config` で `app.main_view` を読み、正規化して 3 値以外を起動時に止め、RunnerFrame へ渡す | 110 RunnerApp |

## 6. 細部

2026-10-31 のグリルで、推奨の既定で埋めた。5・8 は Opus の検討のあとのユーザーの裁定（既定 `eval`、保険の WARN）で改め、14〜16 を足した。2026-11-01 の再グリルで 14・16 を改め、17・18 を足した。同日の 2 回目の再グリルで 3・13・14・15・17 を改め、19〜21 を足した。同日の 3 回目の再グリルで 7・11・13 を改め、22〜27 を足した。同日の 4 回目の再グリルで 7・23 を改め、28〜31 を足した。

1. View メニューの項目: `&Train View` のチェック項目を `&Evaluation View` の上に復活する（理由: コメントアウトされていた位置そのまま）。
2. Panels ツールバー: 「Train View」ボタンを「Eval View」の左に足し、説明は "Show/hide Train View"（理由: 画面の左右の並びと同じ順）。
3. Train pane の caption とボタン: caption は "Train View"、MaximizeButton は true、PinButton は Floatable と同じ値で描く（D2）。足すのは ✕ と pin（理由: 隠せること・外せることに要るボタンだけ。2026-11-01 の 2 回目の再グリルで pin を足した）。
4. 隠している間の Train View FPS メニュー: 操作でき、値は覚え、再表示のときにその FPS で timer を始める（理由: FPS の選択は pane の表示とは別の状態）。
5. 既定の `eval` で起動したとき: Train の timer は `Initialize` でも始めない（理由: 隠れている間は止める規則の起動時の形）。
6. maximize 中の有効・無効: `IsPaneOpen` の値で判定する（理由: maximize で一時的に隠れた pane は利用者が開いている pane である）。
7. 昇格・降格の幅: 降格先が Left か Right の dock なら幅を、Centre と同じ幅の Top・Bottom の帯なら高さを等分し（D15）、それ以外は触らない（理由: D15。3 回目の再グリルで「定位置だけ 50:50」から改め、4 回目で上下の帯を足した）。
8. ログ: pane の開閉・昇格・降格でログを出さず、保険の発動だけ WARN（理由: 既存の pane の開閉も出していない。保険は起きないはずの事態）。
9. 無効値の止め方: `app.main_view` の検証は `RunnerApp::Config`、メッセージは既存の形 `Invalid app.main_view: value=<値> (expected: train, eval, both)`（理由: TrainPanel の検証と同じ形）。
10. 無効なトグルの見え方: メニューとボタンは灰色でチェック付き、✕ は描かない（理由: 規則 D3 の見え方の確定）。
11. 主 pane の caption: 本体は "Train View"・"Evaluation View" のままで、Centre を占めている間だけ末尾に `[Centre]`（理由: 識別子は同じに保ち、役は印で示す。D16。3 回目の再グリルで改めた）。
12. GUI テストのタグ: `[prd110_main_pane]`。既存の RunnerApp を起動するテストの形に沿う（理由: `[eval_steps_menu]` と同じ作り）。
13. 文書の更新先: D12 の列挙が正本。160 の jp・en の §2.1 の RunnerFrame 行と §7.1、020 の jp・en の Panels・Train View・Evaluation View・Reset Layout と設定キーと pane の一覧の前の pin の 1 文、検証資材 `core/anet-core/testdata/prd110/README.md` の人が確かめる項、CONTEXT.md の「主領域」、ADR 0016 の先頭の注記、wiki 110（理由: 今の記載がある場所。列挙を 1 か所に保つため、§5.5 と受入 9 も D12 に従う）。
14. ✕・pin・Floatable の宣言と占有の規則の当て直しの置き場: frame 自身の既存の 200 ms の UPDATE_UI（`OnUpdateTrainStatus`）。変わったときだけ `CallAfter` で `Update`（理由: Always on Top と同じ流儀。idle の外の `Update` は `UpdateLogAlertBadge` の前例。ドラッグの float は遷移の前が無いので、ここが最も早い時機。pin の押下は先に frame へ届くが、昇格には使わず判定で止めるだけにして時機を 1 つに保つ。§1.1 の 24・26）。
15. 「離れてよいか」の口の置き場: 基底 `AuiLayoutFrame` の仮想の口 1 つ（理由: メニューのハンドラ・✕ の Veto・pin の押下の 3 つの経路が基底にあり、同じ判定を読ませる（D3 の (2)〜(4)）。派生の UPDATE_UI での Floatable・✕・pin の宣言も同じ口を読む）。
16. 昇格前の位置の記憶: 方向・layer・row・位置の 4 値を frame が持つ。Train・Eval のどちらが昇格したときも同じ 1 組で足りる（同時に昇格しているのは 1 つ）。Reset Layout で消す（理由: 昇格で pane の dock 情報を書き換えるので pane 自身には残らない）。
17. ドラッグや pin で float した直後の Centre: 相方が昇格するまで最大 200 ms は背景が見える（理由: wxAUI はドラッグのイベントを出さず、pin の押下で先回りする形は時機を 2 つにするので採らない。`wxAuiManager` の派生は §4.1 で退けた。ADR 0016 が許容したドラッグ直後の遅れと同じ種類）。
18. 両方 Centre の保険の文言: `Both main panes occupy the centre. Moving Evaluation View aside as a fallback.`（理由: D13 の既存の文と同じ形）。
19. pin の付け先と値: Logs はそのまま、Train・Eval は Floatable と同じ値、Q-Values・Evaluation Trace・HeatMap・Conv2d は常に有り。toolbar pane は無し（理由: D14。toolbar pane は caption を持たない）。
20. pin で外した floating frame の位置と大きさ: wxAUI に任せる（理由: Logs の pin と同じ。位置は初回 OS 既定で動かすと覚え、大きさは今のドラッグの float と同じ。§1.1 の 26）。
21. 宣言が追いつく前の pin の止め方: 判定が否なら frame が `wxEVT_AUI_PANE_BUTTON` を受け取って wxAUI へ渡さない（理由: ✕ の Veto と同じ守りを同じ判定で置き、最大 200 ms の間の pin が保険（D13）に届かないようにする）。
22. 置き場の比較: 200 ms の宣言で、主 pane 2 つの方向・layer・row・位置・docked か float か・表示中かを前回と比べ、変わったときだけ幅のポリシーを 1 回当てる。比較の初期値は起動と Reset Layout で現在の置き場（理由: D5 の位置の記憶と同じ 4 値に float と表示を足した最小の組。最初の宣言で無駄な往復をしない）。
23. 幅・高さの書き込み先: 側面の主 pane の dock（(2) では帯の dock）に属する pane 全部の dock_size（横は幅、縦は高さ）（理由: 今の Right の列と同じ書き方。同居列・同居帯は対象外なので主 pane だけになる）。
24. maximize 中の幅のポリシー: 当てない（理由: ADR 0016 の既存のガード）。
25. 印の文言と付け方: 半角スペースと `[Centre]` を末尾に。float 中・側面・非表示は無印。Logs・Q-Values・補助 pane は常に無印（理由: D16）。
26. 「定位置」の語: Reset Layout と D5 の降格先（Train=Centre、Eval=Right の layer 0）の意味で残し、等分の条件からは外す（理由: 既定の配置の語としてはまだ要る）。
27. ADR 0016 の注記: 先頭に、主 pane の置き場の変化だけは 200 ms の UPDATE_UI で幅のポリシーを当てる（PRD110、2026-11-01）旨の 1 段落。本文は書き換えない（理由: コミット済みの ADR）。
28. 縦の等分の予算: Centre の pane の実際の高さ（rect）と、主 pane の dock の高さの和。Logs・toolbar の帯は入れない（理由: 横で Q-Values の列を除くのと同じ。組になる 2 枚だけを等分する）。
29. 高さ不足: 最小高（200+200）で clamp し、Logs・toolbar の帯は縮めない（理由: ユーザーの裁定、2026-11-01。Logs を縮める挙動は既存に無い）。
30. 同じ幅の帯の判定: 表示中の Left・Right の dock の layer がすべてその帯の layer より大きい（理由: §1.1 の 28。実際の rect の幅を比べても同じだが、判定は layer で書ける）。
31. Top の帯: Bottom と同じ規則（理由: 上下は対称。利用者の求めは「上下」）。

## 7. 受入基準

受入 2〜6 の pane の状態は `wxAuiManager::GetPane` の dock の方向・表示・`IsFloatable()`・`HasCloseButton()` で、トグルの状態はメニュー項目とツールバーのツールの有効・チェックで読む。「宣言が追いついたあと」は、UPDATE_UI を 1 回処理して `CallAfter` を流したあとを指す。✕ を押したことにするには、`wxEVT_AUI_PANE_BUTTON`（button は `wxAUI_BUTTON_CLOSE`、pane はその pane）を frame へ送る（§1.1 の 5）。pin を押したことにするには、同じイベントで button を `wxAUI_BUTTON_PIN` にする（§1.1 の 26）。pin の有無は `HasPinButton()` で読む。caption の印は `wxAuiPaneInfo` の caption で読む。dock の実幅は受入 2 の b と同じ読み方で、主領域は client の幅から、その dock 以外の表示中の縦の列を除いた幅。pane を `Float()`・`Dock()` して Update する手順（f・g・k・o〜q・s・u）は、利用者の操作（pin・ドラッグ・縁へ落とす・サッシ）を模すものではなく、規則の入力となる置き場を作る準備で、受入 6 の AUI の直接操作と同じ扱い。操作そのものは受入 8 で人が確かめる。入力から送れる操作（✕・pin・maximize と restore・メニュー・ツールバー・リサイズ）はイベントで送る。maximize と restore を押したことにするには、✕・pin と同じイベントで button を `wxAUI_BUTTON_MAXIMIZE_RESTORE` にする。Train の timer は、`TrainPanel` のテスト用の派生クラスから読む。

1. ビルドと既存のテスト: Debug 全体と `AnetRLRunner-gui-test` がビルドでき、`anet-core-test` と `AnetRLRunner-gui-test` の既定のスイートで、落ちるテストの集合が変更前と同じである（変更前から落ちているテストは増えも減りもしない）。`anet-core-test "[shipped_config]"`、`AnetRLRunner-gui-test "[eval_steps_menu]"`、`AnetRLRunner-gui-test "[prd105_ui_keys]"`、`AnetRLRunner-gui-test "[prd109_ui]"` が、それぞれ別プロセスで通る。
2. 主 pane の遷移（`[prd110_main_pane]`、1 プロセス。CartPole の設定、`app.main_view` は既定の `eval`、`app.show_error_dialog = false`）: RunnerApp を起動し初期化が済んだあと、次の順で操作してすべて成り立つ。
   - a. 起動直後: Eval は Centre に表示、`IsFloatable()` が false、`HasCloseButton()` と `HasPinButton()` が false。Train は非表示で、timer は止まっている。Eval View のメニュー項目とツールバーのツールはチェック付きで無効。Train View は有効でチェック無し。
   - b. Train View を ON（メニュー）: Train が Centre に表示。Eval が Right、layer 0 に表示。宣言が追いついたあと、Train と Eval の `IsFloatable()`・`HasCloseButton()`・`HasPinButton()` が true。両トグルが有効でチェック付き。Eval の dock の幅が、主領域（client の幅から右の列を除いた幅）の半分と 2 px 以内で一致する。Train の timer が、選ばれている FPS の周期で動いている。
   - c. Train View を OFF（ツールバー）: Train は非表示で timer が止まっている。Eval の dock の方向が Centre、`IsFloatable()` が false、宣言が追いついたあと `HasCloseButton()` と `HasPinButton()` が false。Eval のトグルはチェック付きで無効、Train のトグルは有効でチェック無し。
   - d. Train View を ON: b と同じ状態に戻る。
   - e. Train の ✕（押したことにするイベント）: c と同じ状態になる。Train View を ON で b に戻る。
   - f. Eval を float（pane を `Float()` して Update）: 宣言が追いつく前に Train の ✕ を押したことにしても Train は表示のまま、pin を押したことにしても Train は docked のまま。宣言が追いついたあと、Train の `HasCloseButton()`・`HasPinButton()`・`IsFloatable()` が false、Train のトグルが無効、Eval のトグルは有効。Train は Centre のまま（Eval は占有者ではなかったので昇格は起きない）。Train View のメニューのイベント（OFF の向き）を送っても Train は表示のまま。Eval を Right へ Dock して Update すると、宣言が追いついたあと ✕・Floatable・トグルは b と同じになり、Frame の大きさを変えて幅のポリシーを走らせると幅も b と同じになる（プログラムからの Dock と Update はポリシーの時機に当たらないため）。
   - g. Eval を Left の dock へ Dock して Update し、Train View を OFF、ON: Eval は Centre へ移ってから Left の dock に戻る（方向が Left）。
   - h. Train View FPS を `0 (Off)` にして Train View を OFF、ON: timer は止まったまま。Train を OFF にしたまま FPS を 10 にしても timer は止まったまま。Train View を ON にすると timer が 100 ms の周期で動く。
   - i. Reset Layout: a と同じ状態（Eval だけ）に戻る。g で覚えた Left の位置は消え、Reset のあと Train View を ON にすると Eval は Right、layer 0 に出る。
   - j. 両方を表示した状態で Train の pin を押したことにする: Train が float する。宣言が追いつく前に Eval の ✕ を押したことにしても Eval は表示のまま、pin を押したことにしても Eval は docked のまま。宣言が追いついたあと、Eval の dock の方向が Centre、`IsFloatable()`・`HasCloseButton()`・`HasPinButton()` が false、Eval のトグルが無効。Train は float のまま表示で、timer は選ばれている FPS の周期で動いている。Train View のメニューのイベント（OFF の向き）で Train が隠れ、ON で float のまま出る（Eval は Centre のまま）。
   - k. float 中の Train を Left へ Dock して Update: 宣言が追いついたあと、Train の方向が Left、Eval の方向は Centre のまま。両方の `IsFloatable()` と `HasCloseButton()` が true、両トグルが有効。Eval View を OFF: Train の方向が Centre、`IsFloatable()` が false。Eval View を ON: Eval の方向が Centre、Train の方向が Left に戻る。
   - l. k の最後の状態（Train が Left、Eval が Centre）で Train View を OFF、ON: Eval は Centre のまま、Train は Left に出る。Reset Layout: a と同じ状態（Eval だけ）。そのあと Train View を ON にすると、Train が Centre、Eval が Right、layer 0 に出る。
   - m. a の状態（Eval だけ）で Eval の pin を押したことにしても、Eval は docked のまま Centre にいる。両方を表示して Train の maximize を押したことにし、Train の pin を押したことにすると、maximize が解けて Train が float し、宣言が追いついたあと j と同じ状態になる。
   - n. 両方を表示した状態で、Logs と、表示した Q-Values・Evaluation Trace の pane の `HasPinButton()` が true。Logs の pin を押したことにすると Logs が float し、Train・Eval の方向・Floatable・✕・pin・トグルは b のまま。
   - o. 両方を表示した状態で Train の pin を押したことにして float させ、AUI の直接操作で Left の layer 1 に置いて Update（置き場の準備。縁へ落とす操作は受入 8）: 宣言が追いついたあと、Train の方向が Left、Eval は Centre、Train の dock の実幅が主領域の半分と 2 px 以内で一致する。Frame の大きさを変えても半分のまま。
   - p. Q-Values を表示し、AUI の直接操作で Train を Right の、Q-Values の列より外側の layer（Q-Values だけ表示なら 11）に置いて Update: 宣言が追いついたあと、Eval（Centre）と Train の実幅が主領域（Q-Values の列を除いた幅）の半分と 2 px 以内で一致し、Q-Values の列の幅は変わらない。
   - q. Q-Values を表示し、AUI の直接操作で Train を Bottom の、Q-Values の列より外側の layer（Q-Values だけ表示なら 11。window の下の縁に落とした帯に相当し、Centre より広い）に置いて Update: 宣言が追いついたあとも、Train の dock の実幅と実高は置いた直後から変わらない。
   - s. Logs を隠し、両方を表示した状態で、AUI の直接操作で Train を Bottom の layer 0 の row 0 に置いて Update（Eval の下の帯に落とした配置に相当）: 宣言が追いついたあと、Train の dock の実高が「Eval の実高と Train の dock の実高の和」の半分と 2 px 以内で一致し、Train の実幅は Eval と同じ。Frame の高さを変えても半分のまま。
   - t. s の状態で Q-Values と Evaluation Trace を表示: 等分は保たれ（右の列は Bottom の layer 0 より外側）、右の列の幅は変わらない。
   - u. Logs を表示した状態で、AUI の直接操作で Train を Logs と同じ dock（Bottom の layer 0 の row 0）に置いて Update: 宣言が追いついたあとも Train の dock の実幅・実高は置いた直後から変わらない（同居）。Train を Bottom の layer 0 の row 1 に置き直して Update すると、宣言が追いついたあと Train の実高が「Eval の実高と Train の dock の実高の和」の半分と 2 px 以内で一致し、Logs の実高は変わらない。
   - v. s の状態で Eval View を OFF、ON: OFF で Train が Centre へ昇格し、ON で Eval が Centre を取り戻して Train は Bottom の layer 0 に戻り、宣言が追いついたあと高さが再び等分される。
   - r. a〜v の各状態で、宣言が追いついたあと（a は初期化が済んで最初の宣言が追いついたあと）、Centre を占める主 pane の caption が `[Centre]` で終わり、もう一方の主 pane（float 中・側面・非表示）の caption は本体だけ。Logs・Q-Values・Evaluation Trace の caption にも付かない。
3. 初期表示（`[prd110_main_pane]`、設定ごとに 1 プロセス）:
   - `app.main_view = train`: 初期化が済んだあと、Train が Centre に表示、`IsFloatable()` が false、`HasCloseButton()` が false。Eval は非表示。Train View のトグルはチェック付きで無効、Eval View のトグルは有効。Reset Layout を押しても同じ状態。Train View を OFF にしようとメニューのイベントを送っても Train は表示のまま。
   - `app.main_view = both`: Train が Centre、Eval が Right、layer 0 に表示され、幅は受入 2 の b と同じ。
   - `app.main_view = EVAL `（前後の空白と大文字）: 既定の `eval` と同じ配置で起動する。
   - `app.main_view = train` で起動し、Eval View を ON にしてから Train View を OFF、ON、Reset Layout（train）: Reset のあと Train が Centre に表示され、Eval は非表示で、Train の timer が選ばれている FPS の周期で動いている。
4. maximize（`[prd110_main_pane]`、受入 2 のプロセスの続きでよい）:
   - A: Train View を ON にして両方を表示し、Train の maximize を押したことにする。Train View を OFF（メニュー）にすると、maximize が解けてから Train が隠れ、Eval が Centre に表示される。宣言が追いついたあと、Eval の `HasCloseButton()` が false。
   - B: 両方を表示して Train の maximize を押したことにし、Eval View を OFF（メニュー）にする。宣言が追いついたあと Train の `HasCloseButton()` が false で、Train の ✕ を押したことにしても Train は表示のまま。Train の restore を押したことにしても Train が Centre に表示されている。
   - C: 両方を表示して Eval の maximize を押したことにし、Eval View を OFF にすると、maximize が解けてから Eval が隠れ、Train が Centre に表示される（今の空画面の癖が直っている）。
5. 初期化前の安全（`[prd110_main_pane]`）: `Initialize` を呼んでいない `TrainPanel` に対して、隠す・表示するの表示状態を渡し、FPS を変えても落ちず、timer は動いていない。そのあと `Initialize` を呼ぶと、表示中なら timer が始まり、非表示なら始まらない。
6. 保険（`[prd110_main_pane]`、1 プロセス）: 両方を表示したあと、AUI の直接操作で Train と Eval の両方の pane を Hide して Update し、Frame の大きさを変えてポリシーを走らせると、Train が Centre に表示され、Run ログに `No main pane occupies the centre. Showing Train View as a fallback.` を含む WARN が 1 行出る。両方を表示したあと、AUI の直接操作で Eval の pane を `Centre()` にして Update し、Frame の大きさを変えてポリシーを走らせると、Train が Centre に残り Eval が Right、layer 0 に戻り、Run ログに `Both main panes occupy the centre. Moving Evaluation View aside as a fallback.` を含む WARN が 1 行出る。受入 2〜5 の手順ではどちらの WARN も 0 件である。
7. 不正値（Runner の exe、`app.show_error_dialog = false`）: `app.main_view = center` を CartPole の `--config` に足して起動すると、終了コードが 0 以外で、ログに `Invalid app.main_view: value=center (expected: train, eval, both)` を含む英語のエラー行が出る。
8. 実機の確かめ（人、Atari5 構成の qbert、online）:
   - 既定で起動すると Eval だけが中央に出て、Train View は Panels と View メニューから出せる。Train View を出すと 50:50 になり、Q-Values と Evaluation Trace を開いても右の列は今と同じ位置に出る。Train View を隠すと Eval が主領域の全幅に戻る。
   - Train View を出した状態で Eval を float（caption のドラッグか pin）すると、Train View の ✕ と pin が消え、Panels の Train View が灰色になる。Eval を縁の帯へ落として dock すると ✕ と pin が戻る。
   - Train View を出した状態で、Train View の caption をドラッグしても動かない。caption の pin を押すと Train が float し、Eval が中央の全幅に広がる。float した Train を左の縁へ落とすと左の列に入り、Eval は中央のまま。左の Train は caption のドラッグでも pin でも外せる。そこで Eval View を隠すと Train が中央へ移り、Eval View を出すと Eval が中央に戻って Train は左に戻る。Train を隠した状態の Eval は caption に pin も ✕ も無く、ドラッグしても動かない。
   - Eval を Left の dock に置いて Train View を隠し、出すと、Eval は Left に戻る。
   - Logs・Q-Values・Evaluation Trace・HeatMap・Conv2d の caption に pin があり、押すと float する。縁の帯へ落とすと dock に戻る。
   - Train View を出した状態で Train を pin で外し、左の縁へ落とす: 落として間もなく Train と Eval が等分になる。そこから Eval を pin で外して右の縁へ戻すと、Train が中央、Eval が右で等分。Q-Values を開いたまま Train を右の縁へ落とすと、Eval、Q-Values、Train の並びで Eval と Train が等分になり、Q-Values の幅はそのまま。Train を下の縁へ落とすと幅は変わらない。
   - 等分のあと、サッシで幅を変えても戻らず、Train View を隠して出すと等分に戻る。
   - 中央の View の caption に [Centre] が付き、Centre が入れ替わると付け替わる。float 中の View の window title には付かない。
   - Train View を出した状態で Train を pin で外し、Eval の下の帯（Eval の下端から 40 DIP まで。Eval が 200 DIP より低いときは高さの 2 割まで）に落とす: 落として間もなく Train と Eval の高さが等しくなり、Train の幅は Eval と同じ。Q-Values や Trace の列はそのまま。Logs を出しても Logs の高さは変わらない。
   - Train を window の下の縁の帯に落とす（左右の列の下まで広がる）: 高さは変わらない。
   - 縦に並んだ状態でサッシで高さを変えても戻らず、Train View を隠して出すと等分に戻る。
   - Run ログに保険の WARN が出ない。
9. 文書: D12 のとおりに直っている。160 §2.1 の RunnerFrame の行に Centre の占有の規則と宣言と保険があり、§7.1 に Train View を隠している間の timer の停止と maximize の解除がある。020 の Panels ツールバーの一覧に Train View があり、pane の一覧の前に pin ボタンの文があり、Train View と Evaluation View の項に既定の表示、中央の View は pin で外すこと、`app.main_view` の説明、Reset Layout の戻り先がある。160 §7.1 に、Centre の pane は pin で float し、pin は Floatable と同じ値で描くことがある。160 §2.1・§7.1 と 020 の Evaluation View の項に等分の条件（一方が中央、他方が左右どちらかの列）と置き場が変わったときの時機があり、160 §2.1・§7.1 と 020 の pane の一覧の前（pin の文の後）に `[Centre]` の印の記述があり、README の人が確かめる項に受入 8 の 6〜8 つ目があり、CONTEXT.md の「主領域」に「定位置」の語が無く、ADR 0016 の先頭に注記がある。160・020 の jp・en と `core/anet-core/testdata/prd110/README.md` に、初版の記述が残っていない。次の文字列の検索がすべて 0 件である（D12 が名指しした初版の文に固有で、新しい記述には現れない形）: 「TrainがあればTrain」「Train があれば Train」「otherwise Eval does」「Train再表示時に戻す」「Train 再表示時に戻す」「restored when Train returns」「Centreを占めるpaneはfloat不可」「Centre を占める pane は float 不可」「Centre pane cannot float」「中央を占めるViewはドラッグで移動できない」「中央を占める View はドラッグで移動できない」「occupying Centre cannot be dragged」「Train 非表示時の Eval」「Train非表示時のEval」。3 回目の改訂で直す文に固有の文字列も 0 件である: 「定位置の両paneだけを50:50」「定位置では左右50:50」「home Train/Eval layout uses 50:50」「the home layout a 50:50 split」とその空白違い。4 回目の改訂で、160 §2.1・§7.1 と 020 の Evaluation View の項に上下の帯の等分の条件（帯の幅が中央と同じとき）と対象外（窓の縁の帯、同居）があり、README の人が確かめる項に受入 8 の 9〜11 つ目があり、CONTEXT.md の「主領域」に上下の帯があり、160・020 の jp と README の現用の文に「等幅」が残っていない（0 件。CONTEXT.md の「主領域」も同じ）。

## 8. 範囲外・先送り

| 項目 | 裁定 | 理由と再開条件 |
|---|---|---|
| 隠している間の Train 側 observer の切り離し | 範囲外 | NG2。FPS 0 と同じ関心。再開条件: [999_notifier_scoped_detach](../942_notifier_scoped_detach_10prd.md) の修復が入り、Train スレッド側の View 更新のコストを消したい要求が出たとき。隠す・FPS 0 の両方で外す形にする |
| Eval pane を隠したときの Eval の timer 停止 | 範囲外 | NG3。Train の定期停止の増幅要因としての対策（[999_eval_schedule_first_fire](../918_eval_schedule_first_fire_10prd.md) Context）と一緒に決める。再開条件: 定期停止の対策の PRD を起こすとき |
| 利用者の操作の意味として隠れている pane を出す・float を解く | 範囲外 | NG4。保険（D13）は知らない経路でだけ動く。再開条件: 保険の WARN が実機で出たとき（まず規則の穴として本 PRD を改訂する） |
| Logs・Q-Values・補助 pane の配置規則 | 範囲外 | NG5。別の関心 |
| Atari の online の Train View FPS 0 の見直し | 範囲外 | NG6。再開条件: Train の進みを目で見たい要求が出たとき |
| Train と Eval の左右の入れ替え（設定で定位置を選ぶ形） | 範囲外 | 要求が無い。Train を pin で外して側面へ落とし、Eval を Centre にすることは D1・D2 でできる（2026-11-01）。再開条件: Eval を左に置きたい要求が出たとき。D5 の覚えた位置の仕組みの上で、定位置を設定にする形から起こす |
| サッシで変えた主 pane の幅の比率の記憶 | 範囲外 | NG7。再開条件: 遷移で等分に戻る動きが邪魔だという要求が出たとき。D15 の置き場の比較の上で、比率を覚える形から起こす（2026-11-01） |

## 9. 未決

なし。2026-10-31 のグリルと、Opus の検討のあとの質疑で次のとおり閉じた。2026-11-01 の再グリル（ユーザーの求め）で 3・14 を改め、17〜19 を足した。同日の 2 回目の再グリル（実機の報告）で 3・8・19 を改め、20〜22 を足した。同日の 3 回目の再グリル（実機の報告）で 7・8 を改め、23〜27 を足した。同日の 4 回目の再グリル（実機の求め）で 23 を改め、28・29 を足した。

| # | 論点 | 閉じ方 |
|---|---|---|
| 1 | 土台（動的 Centre か、主領域 1 pane 化か） | 動的 Centre。1 pane 化は Eval の float・maximize・✕ を失い、表示切替が 2 系統になる（§4.1） |
| 2 | Centre を空にできる操作 | 導線（✕・ツールバー・メニュー）では「どちらか 1 つは必ず表示」を守る（D3。ユーザーの提案） |
| 3 | Centre pane のアンドック | 初版は「Centre を占める pane は float できない。役に追従させる」（ユーザーの提案）。2026-11-01 の再グリルで、相方が表示中で docked のときだけ float でき、Centre か否かは見ない形に改めた（D2。ユーザーの求め）。同日の 2 回目の再グリルで、Centre を占める pane は caption の pin ボタンで float することにした（D2。wxAUI は Centre の caption ドラッグを始めない。§1.1 の 6） |
| 4 | float 中の相方の扱い | 候補に数えず、✕ を無効にする。一度推奨した「引き込む」はユーザーが撤回（D1・§4.1） |
| 5 | 隠している間の observer | timer だけ止める（D6・NG2） |
| 6 | 初期表示の設定 | モード値 1 つ、`app.main_view`（D7） |
| 7 | ゴールと非ゴール | G1〜G5・NG1〜NG5 で合意。Opus の検討のあとに G2・G4・G5・NG4 を書き改め、NG6 を足した。2026-11-01 の 3 回目の再グリルで G6・NG7 を足し、G5 の意図して変える点を 6 つにした |
| 8 | 足す仕組み | 占有の規則、役に追従する Floatable、隠す導線の守り、Train View の導線、timer の停止・再開、初期表示のモード値の 6 つで合意。Opus の検討のあとに保険（D13）を足した。2026-11-01 の 2 回目の再グリルで全 pane の pin ボタン（D14）を足し、隠す導線の守りに pin の押下を含めた（D3）。3 回目の再グリルで等分の条件と時機（D15）と Centre の印（D16）を足した |
| 9 | Reset Layout の戻り先 | 設定の初期表示へ（D8） |
| 10 | キーの名前 | `app.main_view`（D7） |
| 11 | ADR | 起こす（D11） |
| 12 | 細部 13 件 | 推奨の既定で埋めた（§6） |
| 13 | Train の描画 timer の伝え方（Opus の指摘） | ポリシーの入口で表示状態を渡し、TrainPanel が 1 か所で決める（D6。ユーザーの裁定） |
| 14 | 降格先（Opus の指摘） | 昇格前の位置を覚えて戻し、Reset Layout で消す（D5。ユーザーの裁定）。2026-11-01 に Train・Eval の両方へ広げた |
| 15 | 保険（Opus の指摘） | 置き、発動したら WARN を出す（D13。ユーザーの裁定） |
| 16 | 既定の `app.main_view`（Opus の指摘から） | `eval`。Train View は最近ほとんど見ておらず、学習の進みは Step 表示とログで足りる（D7。ユーザーの裁定） |
| 17 | 占有の規則（再グリル） | Train 優先から「占有者は隠れるか float するまで保ち、離れたら相方が昇格し、戻れば取り戻す」の対称の規則へ。ゴールの変更をユーザーが確認した（D1。2026-11-01） |
| 18 | float した pane を縁の帯へ落としたときの置き場（再グリル） | 落とした場所に留まり、占有者は変わらない。Train 優先のまま Centre へ戻す案 (a) は、Train を側面に置き続けられないので採らなかった（D1・§4.1。ユーザーの裁定、2026-11-01） |
| 19 | ドラッグや pin で float した直後の昇格の時機（再グリル） | D3 の 200 ms の UPDATE_UI。最大 200 ms は Centre が背景（§6 の 17。コードと wx の事実から導いた。pin の押下で先回りしない） |
| 20 | Centre を占める pane を外す操作（2 回目の再グリル） | caption の pin ボタン（D2）。`wxAuiManager` の派生で wx の判定を迂回する案、占有を先着順にして隠して出し直す案、入れ替えのコマンドを足す案、Train 優先へ戻す案、wx のソースを直す案、ダミーの Centre は §4.1（ユーザーの裁定、2026-11-01: 実装の歪を入れず、操作して確かめられる形で直す） |
| 21 | pin の付け先（2 回目の再グリル） | 全 pane。主 pane は Floatable と同じ値で描く（D14。ユーザーの裁定: 一部だけ付けるのは変） |
| 22 | 宣言が追いつく前の pin（2 回目の再グリル） | 判定で止める。frame が pin の押下を受け取って wxAUI へ渡さない（D3・§6 の 21。✕ の Veto と同じ守り。コードと wx の事実から導いた） |
| 23 | 等分の条件（3 回目の再グリル） | 一方が Centre、他方が Left か Right の dock（案 1）。隣接に限る案、明示の操作にする案、定位置へ引き戻す案は §4.1（D15。ユーザーの裁定、2026-11-01）。4 回目の再グリルで Centre と同じ幅の Top・Bottom の帯の高さの等分を足した（28） |
| 24 | 落とした直後の等分の時機（3 回目の再グリル） | 200 ms の宣言で主 pane の置き場が変わったときに 1 回当てる（A）。次の遷移まで待つ案は §4.1（D15。ユーザーの裁定、2026-11-01） |
| 25 | Centre の印の示し方（3 回目の再グリル） | caption の末尾。ステータスバー・Panels のボタン・caption の色は §4.1（D16。ユーザーの裁定、2026-11-01） |
| 26 | 印の文言（3 回目の再グリル） | `[Centre]`。`(Centre)`・`(centre)`・記号・中点は §4.1（D16。ユーザーの裁定、2026-11-01） |
| 27 | 細部（3 回目の再グリル） | 推奨の既定で埋めた。質疑の一覧の 13 項目を §6 の 7・11・13・22〜27 の 9 件にまとめた（ユーザーの裁定、2026-11-01） |
| 28 | 縦に並んだときの高さの等分（4 回目の再グリル） | 一方が Centre、他方が Centre と同じ幅の Top か Bottom の帯なら高さを等分。帯が Centre より広いときと同居は対象外（D15。ユーザーの求め、2026-11-01） |
| 29 | 縦の高さ不足（4 回目の再グリル） | Logs・toolbar を縮めず、最小高で clamp（D15・§6 の 29。ユーザーの裁定、2026-11-01） |

## 10. 関連

- [160 アプリケーションとツール](../../design/160_applications_and_tools.jp.md) §2.1（RunnerFrame）、§7.1（Runner の Lifetime）。
- [020 Run 実行ガイド](../../design/020_user_guide_run.jp.md) のツールバーと pane の節。
- [ADR 0016](../../adr/0016-runnerframe-aui-layout-roundtrip.md)、[ADR 0066](../../adr/0066-main-panes-occupy-centre-dynamically-and-no-flow-empties-centre.md)。
- [PRD055](055_runner_toolbar_10prd.md)（ツールバーと UPDATE_UI）、[PRD109](109_runner_startup_background_init_10prd.md)（起動の別スレッド化）。
- [wiki 110 RunnerApp](../../wiki/apps/110_runner_app.md)、[wiki 319 GUI 共通基盤](../../wiki/categories/319_gui.md)。

# PRD110 主 pane の Centre 切替 実装メモ

## 概要

[PRD110](110_runner_main_pane_dynamic_centre_10prd.md) の D1〜D13 を実装する。Train View を隠して Eval を Centre へ昇格し、再表示で元の dock 位置へ戻す。起動と Reset Layout は `app.main_view = train | eval | both`（既定 `eval`）へそろえる。

基準コミット: `36cc56ebb506c2fae78fb0ee46a519ef3b39e407`。2026-10-31 の承認済み計画を正本として保存した。未決事項監査のユーザー判断ブロッカーは 0 件。

## 主な変更

- RunnerApp::Config で main_view を正規化・検証し、RunnerFrame の初期表示の列挙型へ変換して明示的に渡す。省略時と common.txt は eval。既存の直接生成も移行し、互換 overload は置かない。
- AuiLayoutFrame に既定は許可する CanHidePane(const wxAuiPaneInfo&) を追加し、UPDATE_UI・menu handler・close Veto を同じ判定へ接続する。拒否時は無変更。最大化された対象を menu から隠す前に RestorePane する。
- RunnerFrame は Train が開いて docked なら Train、それ以外は Eval を Centre にする。開閉の前に遷移後の状態を適用し、Eval の方向・layer・row・位置を記憶して復元する。Reset で記憶を消し、設定の初期表示へ戻す。
- Train View の View menu と Panels tool を追加する。200 ms の既存 UPDATE_UI で CloseButton / Floatable を宣言し、変化時のみ CallAfter で Update する。幅の往復は最大化中に行わず、50:50、dock_size による幅の記憶、補助列→Q-Values 列の縮小順を維持する。Eval Centre の最小幅は 200。
- Centreの実寸法は側面dockの記憶幅ではないため、共通SyncDockSizesToPanesはCentreを同期対象から除く。TakeLayoutSnapshotもCentreにはliveのdock_sizeを使う。これで昇格中の別paneの開閉や幅の往復でも記憶幅を保つ。位置の記憶は4値のままとし、Frameに別の幅stashは足さない。ADR0016の側面dockの同期・構造化往復の方式は維持し、Centreの記憶値を上書きしない扱いを加える。
- TrainPanel::SetViewVisible(bool) を追加し、初期化・表示状態・選択 FPS から timer を一か所で決める。Initialize 前の指定も保持する。timer は protected に置き、test subclass で参照する。ポリシー入口は実際の IsShown を渡し、maximize 後も遅延適用する。
- 通常のポリシー処理の最後で docked の主 pane 不在を修復し、PRD 指定の英語 WARN を発動ごとに一行出す。次の適用で重複しない。
- 対象: RunnerFrame / RunnerApp / TrainPanel、共通 gui、common.txt、既存 GUI test と core config test、新規 RunnerFrame_test.cpp、PRD110 検証資材、設計文書160とガイド020。新規GUIテストファイルは既存の巨大な Trace テストから関心を分けるため。公開 gui ヘッダの変更は派生 frame の開閉判定フックだけとする。

## テスト

Public interface / surface: app.main_view、実 RunnerApp の menu / toolbar / AUI button、wxAuiManager の pane 状態と dock 幅、TrainPanel::Initialize / SetFps / SetViewVisible、Run ログと製品 Runner の終了値。

TDD は次の優先 behavior ごとに一つのテスト→RED→最小実装→GREENの順で進める。全テストを先に書かない。

1. tracer bullet: CartPole の実 RunnerApp を既定で起動し Eval Centre / Train 非表示を確認。
2. 開閉の三導線、最後の主 pane の保護、50:50 と非定位置からの位置復帰。
3. timer の非表示停止・FPS 保持・0・再表示と Reset・初期化前の安全。
4. train / both / 空白付き大文字 / 省略 / 不正値、構成別上書きと共通既定。
5. maximize 中の開閉、float 直後の古い ✕、UPDATE_UI を経由しない menu。
6. 直接 Hide 後の保険と WARN 件数、既存 UI の回帰。

GUI tag は `[.][prd110_main_pane]` と case 別 tag。RunnerApp は一 process 一 case。test main の起動判定も更新する。既存 prd109_ui は Train の直接操作を menu へ移す。

## 検証

- AGENTS.md の Windows/MSVC 手順（VsDevCmd.bat 経由）で変更前後の Debug 全体と AnetRLRunner-gui-test を build。
- repo root で core / GUI の既定 suite を JUnit 採取し、失敗 case の集合を前後比較。既存失敗も成功に読み替えない。
- shipped_config、eval_steps_menu、prd105_ui_keys、prd109_ui、新規 case を別 process で実行。
- 製品 Runner を app.main_view=center / show_error_dialog=false で起動し、非0終了と指定診断を確認。
- core/anet-core/testdata/prd110/ に手順と検証 driver を置き、結果は新規 .scratch/prd110/ 配下へ保存する。
- 設計160 §2.1・§7.1・テスト章、ガイド020の Panels・Train/Eval・Reset・設定説明を更新。既存 CONTEXT / ADR0066 / wiki は保持して整合確認。
- 全品質観点の自己点検後、独立 an-review に計画と実装を出し、指摘を直して同じ agent に再依頼する。
- Atari5 qbert online の実機受入は人が行う。未実施なら完了としない。

## 前提

- PRD110 の既定・診断・状態表を採用する。追加のユーザー判断は不要。
- observer detach、Eval 非表示時 timer 停止、Atari FPS 設定変更は範囲外。
- 最大化中の一時非表示は IsPaneOpen で開閉判定し、timer は実際の表示で停止する。幅往復と保険は最大化ガードの後で行う。
- 毎 tick の新しいロック待ち・device 転送・全レイアウト再構築は足さない。主 pane 2 件のフラグ比較のみ行い、実変更時だけ Update を予約する。
- 無関係な staged / unstaged / untracked の変更を保持し、commit / push はしない。

## 経過と結果

- 2026-10-31: 承認済み計画を保存。これから変更前の build / suite を採取する。
- 変更前 build: VsDevCmd.bat 経由の `cmake --build --preset x64-Debug` と `--target AnetRLRunner-gui-test` は成功（no work）。基準 GUI は `.scratch/prd110/before-gui.xml`、失敗は `Logs indent wrapped lines to the message on demand` の1件。core は `.scratch/prd110/before-suites/` で採取中。
- TDD 1 RED: `run_validation.py gui --output .scratch/prd110/red-default --tag prd110_default` は初期表示の3不一致で失敗（Trainが表示、Evalが非表示、Rightのまま）。tracer bullet が実 RunnerApp の起動経路を通ることを確認した。
- TDD 1 GREEN: `.scratch/prd110/green-default/` の実 RunnerApp は成功。
- 変更前既定 suite: `.scratch/prd110/before-suites/suites.json`。core は失敗0（456.53秒）、GUI は既存のLogs indentの1件。別採取した変更前GUIの `.scratch/prd110/before-gui.xml` も同じ1件。
- TDD 2 RED: `.scratch/prd110/red-transitions/` は Train View menu が存在せず失敗。共通CanHidePane、三導線、位置復帰、Resetを実装。初回の再検証ではtoolbar更新の8不一致が残り、wxAuiToolBarが `wxUPDATE_UI_FROMIDLE` を必要とすることを使用中のwxソースで確認し、testの公開更新経路を合わせた。
- ビルド途中で空引数の計測マクロを使ってコンパイルが失敗した。`ANET_PROFILE_SCOPE(layout)` に修正して再ビルドは成功。新規の戻り値警告も修正した。
- TDD 2 GREEN: `.scratch/prd110/green-transitions-native/` は144 assertionsで成功。menuは明示更新、AUI toolbarはidle由来の公開UpdateWindowUIでそれぞれ更新する必要があった。
- 設計160・ガイド020はjp/enの対応段落を更新する。英語版のほかの古い差分は今回の対象に含めない。
- TDD 3 RED: `.scratch/prd110/red-timer/` は非表示Trainのtimerが稼働しているため失敗。GREEN: `.scratch/prd110/green-timer/` のdefault / transitions / timerはすべて成功。初期化・表示・FPSをUpdateTimerStateへ統一した。
- TDD 4 RED: `.scratch/prd110/red-config/config.xml` は共通既定の欠落1件だけで失敗（構成選択の合成テストは成功）。common.txtに既定を追加し、`.scratch/prd110/config-delta/` の全キー・文字列値比較は成功。
- 出荷設定: `.scratch/prd110/shipped-config/` は終了0（67.2秒）。`check_default_leaves.py` は既存Atari.txtの1525 / 1529行のFPSの`?=`を2件報告した。Atari.txtはHEADとの差分なしで、今回変更しない（NG6）。追加したmain_viewに指摘なし。
- 製品不正値: `.scratch/prd110/fatal-main-view/` は終了255（非0）、process.logに指定診断あり。Config検証がMetricsLogger初期化より前なのでRunログではなく非modalのprocess出力で検査した。
- Debug全体の変更後buildは成功。GUI受入case追加後のリンク時に、こちらのcase一覧確認と競合して実行体がロックされ、LNK1104で失敗した。確認processの終了後にリンクを再実行する。
- 上記リンクの再実行は成功。GUIは `.scratch/prd110/gui-acceptance/` で初期表示・遷移・timer・train・both・正規化・初期化前が成功。float caseの追加検査で、浮動frameの✕をdock用のAUI buttonで送ったためwx内部のsizer assertionに当たった。使用中のwxソースでnative closeが後始末する経路を確認し、実際の浮動frameのCloseへ修正した（production変更は不要）。`.scratch/prd110/gui-native-float/` でfloatまで成功し、残るcaseを検証中。
- 機械点検: 既存16ファイルのBOM変更0件、production追加行の重複識別子リテラル0件、差分の空白違反0件、新規ファイルはLF。CMakeLists.txt / common.txtはGitの既存属性により「次回Git操作でCRLFへ変換」の警告があるが、作業ツリーはLF。
- `.scratch/prd110/gui-native-float/` は新規10caseと既存eval_steps_menu / prd105_ui_keys / prd109_uiの計13processが全件成功。最大化の3パターン、Logs最大化による一時非表示、保険1件・通常0件のWARNも含む。
- `.scratch/prd110/gui-columns/` は成功。Q-ValuesとTraceを開いた状態の50:50、Eval単独で主領域300pxを残したときの列幅保持、pane開閉とResetでFrameサイズを変えないことを追加確認した。これで新規11case＋既存3case。
- `.scratch/prd110/after-suites/suites.json` の比較は成功。coreは変更前後とも失敗0（変更後406.76秒）、GUIは前後とも既存Logs indentの1件だけ（変更後14.92秒）。追加テストの後のproduction変更は説明コメントのみで、既定suiteは繰り返さない。
- 自己点検後の Debug全体とGUI target buildは成功。新たな警告は残らず、MSVCの既存D9025（/Zi→/Z7）は継続している。

## 実行した検証コマンド

以下は実行記録。同じ出力先には上書きできないので、再実行では新しい出力先を使う。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 4 && cmake --build --preset x64-Debug --target AnetRLRunner-gui-test --parallel 4'
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd110/run_validation.py suites --output .scratch/prd110/after-suites --baseline .scratch/prd110/before-suites --baseline-gui .scratch/prd110/before-gui.xml
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd110/run_validation.py gui --output .scratch/prd110/gui-native-float
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd110/run_validation.py gui --output .scratch/prd110/gui-columns --tag prd110_columns
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd110/run_validation.py config --output .scratch/prd110/config-delta --common-baseline .scratch/prd110/input/common.txt
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd110/run_validation.py fatal --output .scratch/prd110/fatal-main-view
```

```bash
core/anet-core/bin/Debug/anet-core-test.exe "[shipped_config]"
```

shipped_configは上の実行体と引数を検証driverのexecute経由で実行し、JUnitと終了値を`.scratch/prd110/shipped-config/`へ保存した。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd072/check_default_leaves.py
```

## 自己点検

実装品質確認観点の全項目を適用範囲に従って確認した。

- 実装計画: 前提、公開面、behaviorごとのTDD、PRDの確定事項、MSVC検証手順を確認。
- 契約・失敗・設定: 旧コンストラクタの現用呼出しは全移行。main_viewはConfig境界で検証し、互換経路を追加しない。保険のWARNはD13が定める例外。設定差分は1葉だけで、構成選択は合成入力、出荷設定テストは共通既定のみ固定する。
- 名前・C++: 公開guiの追加は汎用CanHidePaneのみ。Apply / Update / Scheduleの動詞は既存ApplyLayoutPolicy / UpdateToolBarBitmaps / SchedulePolicyと同じ用途。Frameが位置のState、TrainPanelがFPSとtimerのStateを持つ。新規GUIテストはFrameの関心として独立ファイルに置いた。
- コメント・計測・ログ: 新規コメントは日本語、診断は英語。幅処理の計測を追加。200ms更新では2paneのフラグを比べ、変わった場合だけ遅延Updateを行う。新しいロック待ち・device転送・毎tickのレイアウト再構築はない。
- テスト・文書・文字コード: 実RunnerAppの公開導線・実AUI状態で検証し、protected timer以外のtest-only APIは追加しない。設計160とガイド020をjp/enで追随。BOM保持・LF・差分空白・重複リテラルを確認。skill、ヘッダオンリー追加、GetScalar、Agent所有権、改名、列番号、metricsの連鎖の変更は対象外。
- 新たな仕様判断の未決はなし。未実施の人による受入と、変更範囲外のAtari既定葉2件を既知としてレビューへ渡す。

## レビュー

- 実機確認中の追加指摘: 表示中の主paneのViewボタンが無効になると選択枠が消える。状態のcheckは維持されており、標準AUIのdisabled描画が枠を省いていた。公開のtoolbar描画を使う `[prd110_toolbar]` を先に追加し、Evalのみ・Trainのみの両方で無効のまま選択枠が出ることをRED→修正→GREENで確認する。Panelsの描画だけに選択枠を補い、disabledの文字色と操作拒否を維持する。ユーザーの手動確認は進行中として、受入8全体は未完了のまま残す。
- 枠のREDは `.scratch/prd110/red-disabled-toolbar/`。両Viewで上辺・下辺の強調色pixelが0になり、2アサーションが失敗した。RunnerFrame.cppの無名namespaceのPanelToolBarArtProviderで標準描画の後にdisabledかつcheckedの枠だけを補う。標準の文字色、状態、イベント判定を変えず、別toolbarへは適用しない。ArtProviderは既存wxの命名・Clone/DrawButtonの契約に従う。公開API・設定キーは追加しない。設計160のjp/enへ同期した。
- 枠のGREEN: VsDevCmd.bat経由のDebug全体・GUI target成功。`run_validation.py gui --output .scratch/prd110/green-disabled-toolbar` は追加描画caseを含む新規12＋既存3の15processがすべて終了0、JUnit失敗0。描画caseは27アサーション成功。文字のdisabled状態と操作制限は既存の開閉caseでも継続して確認した。今回core・設定値は変えていないため、既定suite・shipped_configには前掲の結果を用い、繰り返さない。変更した既存3ファイルのBOM保持、変更6ファイルのLF、差分空白を確認した。全品質観点の自己点検で新たな未決はなく、同じan-reviewへ継続レビューを依頼する。
- 選択枠修正の独立レビューも計画・実装ともOK、修正要求0件。原文 `.scratch/prd110/review-20261031-2336.md`。通常の無効文字色・状態・操作判定を維持する描画拡張、描画によるRED/GREENとGUI15process、文書同期を確認した。Debugには反映済みだが、ユーザーが起動中のRelWithDebInfo版はまだ更新していない。
- ユーザーは「今のRunnerを維持し、今回はDebug版までにする」を選択した。起動中のRunnerを終了せず、RelWithDebInfoの再ビルドは行わない。
- cc-review（2026-10-31 23:02）: Codexは計画・実装OK、Claudeは計画OK、実装は全値比較を恒久テストへ置いた点で差し戻し。原文 `.scratch/prd110/review-20261031-2302.md`、`.scratch/prd110/review-20261031-2302-claude.md`。ユーザーの指摘対応指示により、`[prd110_config_compare]` と専用includeを削除する。全値比較はPRD110のPython検証資材へ移し、既存の任意入力JSON採取器 `[prd103-resolve]` を利用する。共通既定の `[shipped_config]` と合成入力の構成選択テストは維持する。
- 今回は製品挙動を変えず、検査の配置だけを修正する。新たな恒久テストは追加せず、移動した比較器で正しい変更前入力の成功と、別の葉を変えた合成入力の拒否を確認する。Debug全体・GUI target、関連coreテストを再実行し、製品コードとGUIテストに変更がないため既定suiteと14 GUI processには前掲の保存済み結果を用いる。検証後に同じCodex・Claudeへ継続レビューを依頼する。
- 指摘対応後: Debug全体・GUI target成功。`.scratch/prd110/config-asset-valid/` は終了0（11.40秒）、全キー比較でapp.main_view=evalの追加だけ。`.scratch/prd110/config-asset-reject/` はapp.run_nameだけを変えた合成入力を終了1（12.16秒）で拒否し、comparison.jsonに差分を保存した。`.scratch/prd110/core-review-config/` の `[shipped_config],[prd110]` は15case・失敗0（95.20秒）。全16既存対象のBOM保持、LF、差分空白とPython構文も成功。自己点検の全観点を再走査し、変更はテスト削除・移行検査・説明だけで、恒久テストが固定する葉はmain_viewの共通既定だけであることを確認した。手順と結果の補足は `.scratch/prd110/fix-validation-20261031-2318.md`。
- cc-review継続（依頼23:19、完了23:22）: 同一依頼 `.scratch/prd110/request-20261031-2319.md` を前回と同じCodex担当・Claudeセッションへ出し、両方とも計画・実装OK、修正要求0件。原文 `.scratch/prd110/review-20261031-2319.md`、`.scratch/prd110/review-20261031-2319-claude.md`。Claude表示費用は約2.89 USD。恒久テストから全値比較を除き、公開ConfigManagerの採取とPythonの比較へ移した点は両モデルが解消を確認した。Claudeは検査スクリプトを実行できず、親の保存済み記録で確認した旨を記載した。人による受入8は未実施を維持する。

- 1往復目: 計画OK、実装は2件で差し戻し。原文 `.scratch/prd110/review-20261031-2238.md`。Evalの非定位置の幅が昇格中のLogs開閉で失われること、str_util.hppのinclude配置を指摘。
- 幅の再現テスト `.scratch/prd110/red-width-memory/` は失敗を確認。共通同期とsnapshotがCentre寸法で側面幅を上書きしないように修正し、Frameの4値の位置記憶とpane.dock_sizeという契約を維持する。includeを基礎ヘッダ側へ移した。
- 差分外のdynamic_cast3件は既存台帳F-0020（reports/registry/findings.jsonl）と同一。今回の新規の課題ではなく、修正範囲外として保持する。
- 追加修正後の検証と、同じan-reviewへの再依頼をこれから実行する。人による実機受入は未実施。
- 再現REDは元幅504pxが336pxへ変わることを確認した。Centreが以前のTrainのdockを引き継ぐため、Evalの元幅と一致するとは限らなかった。SyncだけでなくTakeLayoutSnapshotの補完もCentreの記憶値を保つ必要がある。
- 修正後のDebug全体・GUI target buildは成功。`.scratch/prd110/gui-review-width/` は全14process成功。遷移caseは154 assertionsで成功し、Logs非表示による同期、Centre昇格中にQ-Valuesを500→450pxへ縮める構造化往復、その後のLeftへの幅504px復帰を含む。
- `.scratch/prd110/after-review-suites/suites.json` の比較は成功。coreは失敗0（451.18秒）、GUIは既存Logs indentの1件（14.80秒）で変更前と一致。Config差分・shipped_config・不正値診断はその後設定も検証コードも変えていないため、前掲の成功記録を使う。
- 差し戻し対応の自己点検では、共通同期の責務（方向と記憶サイズだけで判定）、Frameに幅の別stateを持たないこと、side dockの同期とCentreの補完、追加includeの順、BOM/LFと差分空白を再確認。新たな判断待ちはなく、同じan-reviewへ2往復目を依頼する。
- 2往復目: 計画・実装ともOK。原文 `.scratch/prd110/review-20261031-2253.md`。1往復目の2指摘は解消し、新規の範囲外検出もなし。人による受入8の未実施は維持する。

追加修正後の実行記録:

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd110/run_validation.py gui --output .scratch/prd110/green-disabled-toolbar
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd110/run_validation.py config --output .scratch/prd110/config-asset-valid/capture --common-baseline .scratch/prd110/input/common.txt
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd110/run_validation.py config --output .scratch/prd110/config-asset-reject/capture --common-baseline .scratch/prd110/config-wrong-baseline.txt
```

```bash
core/anet-core/bin/Debug/anet-core-test.exe "[shipped_config],[prd110]" --reporter junit --out C:/dev/anet-lab/.scratch/prd110/core-review-config/config.xml
```

上記の比較器は親driverから実行し、process出力・終了値・実時間を各driver配下へ保存した。coreテストも同じexecute経由でprocess配下へ保存した。正例とcoreは0、負例は意図した1である。出力先は新規専用のため、再実行時は別の名前を使う。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd110/run_validation.py gui --output .scratch/prd110/gui-review-width
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd110/run_validation.py suites --output .scratch/prd110/after-review-suites --baseline .scratch/prd110/before-suites --baseline-gui .scratch/prd110/before-gui.xml
```

## 決まらなかったこと

- 人による実機受入は確認途中。追加指摘の選択枠の欠落を修正し、自動検証と独立レビューを完了した。受入全体の完了報告はまだ受けていない。
- 既定葉の静的検査が報告したAtari.txtのFPS代入2件は今回の変更範囲外。自動修正や課題の起票は行わない。

## 自動作業の完了報告（実機受入待ち）

実施内容: 承認済み計画に沿う実装、TDD、Debugビルド、自動検証、品質自己点検、独立レビューと差し戻し対応を実施した。主paneのCentre切替、三導線の保護、timer、初期表示とReset、幅の記憶を実装した。

変更ファイル: RunnerApp / RunnerFrame / TrainPanel、共通gui、common.txt、core設定テスト・GUIテストとCMake登録、PRD110検証資材、設計160・ガイド020のjp/en、本メモ（既存16ファイル＋新規4ファイル）。既存のCONTEXT / ADR0066 / wiki / 他のユーザー変更は保持した。

レビュー結果: 先行an-reviewは2往復で計画・実装ともOK。幅の保持とinclude配置の指摘を修正した。追加cc-reviewはCodexが両方OK、Claudeが全値比較の恒久テスト配置を指摘し、ユーザー指示により検証資材へ移動した。継続レビューでは両モデルが計画・実装ともOKで、修正要求はすべて解消した。重点確認した幅の復元は、元504pxが別操作で336pxになるREDから、同期・往復の両方を含む154 assertions成功へ進めた。レビューで挙がった既存dynamic_cast3件は台帳F-0020であり、新規起票しない。

検証結果: Debug全体・GUI target成功。追加の選択枠修正後は新規12case＋既存3caseの15process成功で、独立レビューもOK。既定suiteはcore失敗0、GUI既存Logs indent1件で前後の失敗集合が一致する前掲記録を保持。shipped_config、共通設定の全値差分、不正値の製品診断も成功。既定葉の静的検査には今回変更していないAtari.txtの2件が残る。

決まらなかったこと: 仕様判断の未決はない。PRD受入8（人によるAtari5/qbert/onlineの確認）は確認途中で、最初の指摘として無効な表示中ボタンの枠の欠落を修正した。全受入の達成は未確認で、PRD全体を完了扱いにしない。

判断してほしいこと: 人による表示・幅・ドラッグの受入結果。手順はPRD110受入8とPRD110検証READMEにある。

次にできること: 実機確認の結果を受け取って、受入状態を本メモへ追記する。今回の手戻りから改善策を検討する場合は `$anet-improve` を利用できる。commit / pushは行っていない。

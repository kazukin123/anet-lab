# PRD110 主 pane の Centre 切替の検証

等分と共通Panelのresize再描画の正本は [24impl](../../../../docs/memo/done/110_runner_main_pane_dynamic_centre_24impl.md)。過去の改訂記録は [23impl](../../../../docs/memo/done/110_runner_main_pane_dynamic_centre_23impl.md)・[22impl](../../../../docs/memo/done/110_runner_main_pane_dynamic_centre_22impl.md)・[21impl](../../../../docs/memo/done/110_runner_main_pane_dynamic_centre_21impl.md)、初版は [20impl](../../../../docs/memo/done/110_runner_main_pane_dynamic_centre_20impl.md)。コマンド、結果、未実施と理由、レビュー結果は実装メモに記録する。

## ビルドと既定スイート

AGENTS.md の Windows/MSVC 手順で VsDevCmd.bat を経由し、Debug 全体と `AnetRLRunner-gui-test` を明示的にビルドする。GUI test は全体 build に含まれない。

変更前後で `run_validation.py suites --output 出力先` を実行する。変更後は `--baseline 変更前の出力先` を付ける。作業ディレクトリは repo root。JUnit の失敗 case 集合が完全一致することを要求し、既存失敗を成功扱いしない。出力先は毎回新規にし、既存結果を上書きしない。

## GUI と設定の検証

`run_validation.py gui --output 出力先` は `[prd110_main_pane]` の case 別 tag を列挙し、1 process 1 case で実行する。`--tag prd110_default` のように case を指定できる。既存の `eval_steps_menu`・`prd105_ui_keys`・`prd109_ui` も別 process で実行する。RunnerApp の再初期化を避けるため `[prd110_main_pane]` 全体を一度に実行しない。

- CartPole / CPU、Train 自動開始なし、Eval FPS 0、エラーダイアログなしの実 RunnerApp を使う。
- pane 状態は `wxAuiManager::GetPane`、menu / toolbar の enable・check、実 dock 幅で読む。✕ とpinは公開の `wxEVT_AUI_PANE_BUTTON` を送り、実際のcloseとVeto、pinの拒否とwxのfloat処理を通す。pinの有無は `HasPinButton()` で読む。実画面のpinをマウスで押す確認は手動受入で行う。
- Train timer は test subclass が protected timer の公開メンバポインタを提供して読む。製品に test-only getter や panel 生成の差し替え口は足さない。
- 初期化待ちは30秒、GUI process の上限は60秒。UPDATE_UI と CallAfter は実イベントループで配送する。
- 初期表示、三導線、最後の主 pane、float 直後の古い表示、最大化、位置復帰、幅、FPS と初期化前の安全を確認する。
- `[prd110_toolbar]` は実toolbarの公開描画経路で、Evalのみ・Trainのみの無効な表示中ボタンにも強調色の選択枠が残ることを確認する。
- 左右の等分は、主paneだけが入るLeft/Rightのdockの実幅を、ほかの表示中の縦dockを除いた主領域の半分と比べる。上下の等分は、単独のTop/Bottom帯の実高を、Centreのrect高とそのdock_sizeの和の半分と比べる。表示中のLeft/Rightのlayerがすべて帯より大きいときだけ対象とし、同居帯や左右の列まで含む広い帯は対象外。Float/Dock/LoadLayoutは規則への入力を準備するためで、drag・drop・サッシ操作の代用とは扱わない。置き場が変わらない幅・高さの変更は保つ。
- Centreの主paneだけがcaptionの末尾に`[Centre]`を持つことを、開閉・pin・左右・非表示・Resetの各宣言後に確認する。
- `[prd110_resize_paint]`はCPU LunarLanderのEvalを1step表示して停止し、Eval単独/両表示と縦/横/両方向の6条件で、resize後と全面再描画後の画素が一致することを検査する。画像はprocessごとの一時Runの隣へ保存する。Viewを覆わないデスクトップ環境で実行する。
- `[prd110_panel_repaint]`は共通Panelの既定styleと明示styleのそれぞれで、停止した描画の縦/横/両方向resizeを全面再描画後と比較する。各Viewは共通Panelからこの保証を受け、別windowのGL canvasには既存の指定を残す。
- `[prd110_cartpole_resize]`は実CartPoleにもLunarLanderと同じ停止中resizeの6条件を適用する。GridMaze・ImageCls・DropMergeは共通Panelの継承とstyleを後から上書きしないことをコードで確認し、個別の学習Runは追加しない。
- `[prd110_height_split]`はBottomとTop・frame高さ変更、`[prd110_height_columns]`は外側のQ-Values/Trace幅保持、`[prd110_height_shared]`はLogs同居の除外・row分離・同layer左右列の除外、`[prd110_height_restore]`はEval OFF/ON・手動の高さ保持・maximize/restore・最小高とLogs非縮小を確認する。AUIのdock最小高には、client最小高200にcaptionと枠の実寸が加わる。
- 直接Hide・両Centreと冗長ONの保険caseでは指定WARNを発動1件要求し、通常のcaseでは両保険WARN0件を要求する。

`run_validation.py fatal --output 出力先` は製品 Runner に `app.main_view=center` を渡し、非0終了と `Invalid app.main_view: value=center (expected: train, eval, both)` を確認する。Config の検証は Run directory の作成前なので、この診断は非modalの process 出力から読む。

`run_validation.py config --output 出力先 --common-baseline 変更前common.txt` は移行時の検証資材として全キー・文字列値を比較する。既存の任意入力採取器 `[prd103-resolve]` で公開の ConfigManager の解決結果を出力し、比較はPython側だけで行う。入力パス・SHA256、変更前後のJSON、JUnit、追加・削除・変更の一覧を出力先へ保存し、`app.main_view=eval` の追加以外は失敗にする。全値を固定する恒久テストは置かず、`[shipped_config]` ではPRDが宣言した共通既定だけを固定する。GUI tagが一件も一致しない場合も検証失敗にする。

## 自動検証結果（2026-10-31）

- Debug全体・GUI targetのビルド成功。新規11caseと既存3caseの計14processで成功。
- 既定suiteの失敗集合は前後一致。coreは0件、GUIは既存のLogs indentの1件。
- 共通設定の全値比較は`app.main_view=eval`の追加だけ。shipped_configは成功。不正値は終了255で指定診断あり。
- 既定葉の静的検査には、変更していないAtari.txtのFPS代入2件が残る。
- 個別のRED / GREEN、コマンド、実行結果の保存先は実装メモを参照する。
- レビュー修正後も全14processと既定suiteの前後比較が成功。LeftのEvalをCentreへ昇格し、Logsの非表示と右列の幅縮小による構造化往復を挟んでも、元の504pxへ戻ることを追加確認した。
- 独立an-reviewは2往復で計画・実装ともOK。人による受入8の未実施は継続している。
- 追加cc-reviewの指摘で全値比較をPython検証資材へ移動した後も、正例は成功、app.run_nameを変えた負例は意図どおり拒否した。`[shipped_config],[prd110]` は15case成功。詳細は実装メモに記録する。
- 指摘対応後の継続cc-reviewはCodex・Claudeとも計画・実装OK、修正要求0件。人による実機受入は未実施のまま残す。
- その後の手動受入で見つかった選択枠の欠落は、追加描画caseでRED（2失敗）を確認し、修正後は27アサーション成功。新規12＋既存3の15processが全成功した。実機受入全体は確認途中。
- 選択枠の修正についても独立an-reviewで計画・実装ともOK。Debugへ反映済みで、使用中のRelWithDebInfo版の更新は別途必要。

## 改訂版の検証結果（2026-11-01）

- Train/Evalのfloat・占有維持、側面位置の復帰、両Centreの保険、単独Trainの幅、Step/Resumeの再表示を追加・更新した。
- `.scratch/prd110/revision-review-gui/` は新規17case＋既存3caseの20processがすべて成功。通常15caseは両保険WARN0件、保険caseは指定WARNがそれぞれ発動1件で、再適用では増えない。レビューで見つかった既表示paneへの冗長ONも、両Centreの保険へ渡りWARN1件となる。
- Debug全体とGUI targetのbuild成功。既定suiteは変更前後ともcore・GUIの失敗0件で、集合が一致した。結果は `.scratch/prd110/revision-before-suites/` と `revision-after-suites/`。
- レビュー修正後のGUI既定28caseも失敗0件（`revision-review-suites/`）。coreは修正前後の実行体SHA256が同じため、`revision-after-suites/` の1069case・失敗0件を再利用した。
- 改訂分の独立an-reviewは2回目で計画・実装ともOK。既表示paneへの冗長ONを隠れていた占有者の復帰と区別する指摘1件を修正し、再確認した。原文は `.scratch/prd110/review-20261101-0349.md`。
- 現用5文書の初版固有文は空白違いも含めて0件。製品Runnerの起動と実機受入は実施せず、人による確認を残す。詳細・RED/GREENは21implを参照する。

## pin改訂の検証結果（2026-11-01、2回目）

- Trainのpin表示、更新前のpin連打の拒否、補助paneのpinでそれぞれREDを確認し、最小修正後にGREENを確認した。
- `pin-gui/` は新規20case＋既存3caseの23processが全成功。最後のEvalへのpin、maximizeからのpin、Logs・Q-Values・Traceのpinも公開ボタンイベントで確認した。通常caseは両保険WARN0件。
- `[shipped_config]` の14caseも別processで成功。既定suiteの前後比較とレビューの詳細は22implへ記録する。
- 既定suiteは前後ともcore1069caseの失敗0、GUI28case中の既存Logs indentの失敗1で、失敗集合が完全一致した。保存先は `pin-before-suites/` と `pin-after-suites/`。既定GUI全成功を意味するものではない。
- 独立an-reviewは1回目で計画・実装ともOK。原文は `.scratch/prd110/review-20261101-1609.md`。pin以外の既存文書不整合1件は別の課題記録候補として22implへ残した。
- 実画面でpinをマウスで押す確認、側面へのdrop、Atari5/qbert/online、HeatMap・Conv2dの動的生成とpinは人が行う受入として残す。Debugと、Runner未起動を確認したあとの追加Release buildは成功。製品Runnerは起動していない。RelWithDebInfoは更新していない。

設定変更後は `check_default_leaves.py` と `[shipped_config]` を通す。共通設定の全キー・文字列値は、追加前の common.txt と追加後を公開の ConfigManager で解決して比較し、`app.main_view=eval` の追加だけを許す。構成別上書きは合成設定で確かめる。

## 等分・Centre caption改訂の検証結果（2026-11-01、3回目）

- Debug全体・GUI targetとReleaseのRunner targetはbuild成功。
- 初回レビュー前の`.scratch/prd110/width-gui-final/`は28process / 28caseすべて成功。Left、外側Right、左右の別列、Bottom・同居列の除外、幅だけの変更の保持、maximize、captionを含む。
- `.scratch/prd110/width-after-suites/suites.json`で既定suiteの失敗集合が変更前と一致。coreは1069case成功、GUIは既存Logs indentの1件だけ。`[shipped_config]`は14case成功。
- 実機受入は人が行うため未実施。製品Runnerの起動・終了は行っていない。
- レビューで見つかったTrace行移動時の全体更新は、captionだけなら再描画へ限定して修正。追加の早期float closeで見つかったsizer例外は、許可済みnative closeで旧sizerの所有を解放して修正。`width-green-real-trace/`で実Trace連続移動の全体更新0回、caption・浮動タイトル、早期closeが成功。修正後の全回帰は`width-gui-reviewed/`、既定suiteは`width-after-review-suites/`、出荷設定は`width-shipped-final/`へ保存する。
- 修正後の最終結果はGUI30process / 30caseが全成功、出荷設定14case成功。既定suiteはcore1069case失敗0、GUIは同じLogs indent1件で、失敗集合の前後一致を確認した。

## 停止中LunarLander Viewのリサイズ回帰（2026-11-01）

- LunarLanderPanelのリサイズ時の部分再描画で古い地形・Wind文字が残る既存不具合を、構築時の`wxFULL_REPAINT_ON_RESIZE`で修正した。等分や配置のポリシーは変更していない。
- `.scratch/prd110/resize-repaint/runner-red/`は6条件中4条件で画素差が出て失敗。Eval単独の縦/横/両方向と、両View表示の縦で再現し、等分補正が働く両表示の横/両方向では差0だった。
- `runner-green/`は6条件すべて画素差0。resize・width_left・width_stability・trace_captionの4processが全成功。同directoryの`captures/`に前後画像と条件別の画素差を保存した。
- VsDevCmd経由・1並列のDebug GUI target buildが成功（Debug製品Runnerも依存targetとしてリンク）。今回の修正では既定suite全体、Release/RelWithDebInfoのbuild、利用者のRunの操作は行っていない。起動中Runを妨げず、局所的な描画修正と直接のGUI回帰に検証を限定した。

## 高さ等分・共通Panel改訂の検証結果（2026-11-02、4回目）

- 共通Panelの既定/明示style×縦/横/両方向の6条件で残像を再現し、コンストラクタへ全面再描画を追加して全成功。LunarLanderの個別指定は共通処理へ移した。実CartPole・LunarLanderも各6条件が画素差0。
- Debug全体・GUI targetのbuild成功。`.scratch/prd110/height-gui-final/`の37process / 37caseが全成功。Bottom/Top、外側Q-Values/Trace、Logs同居の除外とrow分離、昇格・復帰、任意の高さ保持、maximize、最小高とLogs非縮小を含む。
- 既定suiteは`height-before-suites/`と`height-after-suites/`で失敗集合が一致。core1069case失敗0、GUI28case中の既知Logs indent1件。出荷設定14case成功。RED/GREEN、自己点検とレビューは24implへ記録する。
- エージェントによる製品Runnerの起動・終了とRelease/RelWithDebInfoのbuildは行っていない。人による実機受入は下記の2026-11-02のユーザー報告で確認済み。

## 実機での受入

人が Atari5 / qbert / online を起動し、次を確認する。自動検証では置き換えない。

1. 既定でEvalだけが中央に出る。Trainをメニュー・Panelsから出すと、ほかの列を除いた主領域を等分し、隠すとEvalが主領域の全幅を使う。Q-Values / Evaluation Traceの列配置は維持される。
2. Train Viewを出した状態でEvalをcaptionのドラッグかpinでfloatすると、Trainの✕とpinが消え、PanelsのTrain Viewが灰色になる。Evalを縁の帯へ落としてdockすると✕とpinが戻る。
3. Train Viewを出した状態でcaptionをドラッグしても動かない。captionのpinを押すとTrainがfloatし、Evalが中央の全幅に広がる。Trainを左の縁へ落とすと左の列に入り、Evalは中央のまま。左のTrainはcaptionのドラッグでもpinでも外せる。Evalを隠すとTrainが中央へ移り、Evalを出すとEvalが中央へ戻ってTrainは左へ戻る。Trainを隠した状態のEvalはcaptionにpinも✕も無く、ドラッグしても動かない。
4. Eval を Left に置いて Train を隠し、出すと Left に戻る。
5. Logs・Q-Values・Evaluation Traceと、追加したHeatMap・Conv2dのcaptionにpinがあり、押すとfloatする。縁の帯へ落とすとdockに戻る。
6. Trainをpinで外して左の縁へ落とすと、間もなくTrainとEvalが等分になる。Evalをpinで外して右の縁へ戻すと、Trainが中央、Evalが右で等分になる。Q-Valuesを開いたままTrainを右の縁へ落とすと、Eval・Q-Values・Trainの並びで主Viewが等分となり、Q-Valuesの幅はそのまま。Trainを下の縁へ落とすと幅は変わらない。
7. 等分のあとサッシで幅を変えても戻らず、Train Viewを隠して出すと等分へ戻る。
8. 中央のViewのcaptionに`[Centre]`が付き、Centreの交代で付け替わる。float中のViewのwindow titleには付かない。
9. Trainをpinで外し、Evalの下の帯（下端から40 DIPまで、Evalが200 DIPより低い場合は高さの20%まで）へ落とすと、間もなく高さが等分になり、幅はEvalと同じになる。Q-Values・Traceの列と、表示したLogsの高さは変わらない。
10. Trainをwindowの下の縁へ落とし、左右の列の下まで広がる帯にすると、高さは等分しない。
11. 縦に並んだ状態でサッシで高さを変えても戻らず、Train Viewを隠して出すと等分へ戻る。
12. 通常操作のRunログに保険のWARNが出ない。

### 受入結果（2026-11-02）

ユーザーから「実機動作確認、受入確認OK」と報告を受けた。上記の人による受入とPRD110の受入8を確認済みとして、実機受入待ちを解消し、PRD全体を完了とする。詳細は[24implの受入記録](../../../../docs/memo/done/110_runner_main_pane_dynamic_centre_24impl.md#人による実機受入2026-11-02)を参照する。

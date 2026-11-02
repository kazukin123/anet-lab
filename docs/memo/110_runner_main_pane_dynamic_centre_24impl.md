# PRD110 上下の等分と共通Panelのresize再描画 実装メモ

## 概要

PRD110の4回目の改訂（D15(2)、細部28〜31、受入2q・s〜v・8・9）を実装する。Centreと同じ幅のTop/Bottomの主paneを高さで等分し、左右の等分と既存の開閉・pin・最大化・timerの規則を保つ。

併せてユーザーの明示依頼により、停止中のresizeで残像が出る問題への対策を`anet::rl::gui::Panel`へ移す。既定styleを使うViewだけでなく明示styleを渡すViewにも適用し、追加Viewが個別に再描画指定を意識しなくて済むようにする。20〜23implは以前の記録として編集しない。

## 主な変更

- 共通Panelのコンストラクタで、渡されたstyleへ`wxFULL_REPAINT_ON_RESIZE`を加えてからwxPanelを構築する。公開API・設定・timer・新しいイベントは追加しない。前回LunarLanderへ足した個別の指定を共通処理へ移し、他の描画処理は変えない。Atari/HeatMap/Conv2dのwxGLCanvasは別windowなので既存の指定とOnSizeを保つ。既存TrainPanelの明示指定やRefresh処理は今回の目的に不要な整理として触らない。
- 高さの規則は既存の構造化SaveLayout/LoadLayout往復に載せる。主paneが一つだけCentreを占め、相方が表示中・dockedのTop/Bottomで、同じ方向/layer/rowに他paneがいないときに候補とする。表示中でdockedのLeft/Rightのpaneはすべて、その帯より外側（大きいlayer）であることを確認する。
- 高さの予算はCentreのpane.rectの高さと側の帯の実dock_sizeの和。候補の帯だけを半分へ変更し、Train/Evalの最小高200ずつでclampする。Logs・toolbar・Q-Values・補助列の寸法を予算へ混ぜず、縦の配置では幅不足の縮小処理を行わない。Top/Bottomの同居帯とCentreより広い帯は変更しない。
- 既存の200msの配置6値比較と遅延適用、明示遷移、resize、restore、Resetをそのまま使う。幅・高さだけの変更は比較しない。最大化中は往復せず、LoadLayoutの後に以前のpane参照を使わない。高さの別stashは作らない。
- 高さの最初のRED→GREENは既存関数内で最小実装する。その後のGREENで`ApplyMainPaneWidths`を`ApplyMainPaneSizes`へ改名する。理由は、幅と高さの両方を決める処理になるため。呼出しと私有宣言を同じ変更で移行し、現用の旧名を残さない。過去のPRD・実装記録の記述は保持する。
- テストの`SetDockWidth`も`SetDockSize`へ改名する。同じ公開serializerが幅・高さの両方の入力を準備するためで、本体APIは増やさない。
- D12に従い設計160日英§2.1・§7.1、ガイド020日英のEvaluation View、検証READMEを更新する。現用の160 jp・020 jp・READMEでは「等幅」を「等分」へ移す。共通Panelのresize保証も設計160へ記す。PRD・CONTEXT・ADR0016/0066・wikiは起票側の変更を保護して編集しない。

## テスト

- Public interface / surface: RunnerApp、Viewメニューとtoolbar、pin/maximize/restoreイベント、wxAuiManagerの公開pane・rect・構造化serializer、実際のframe resizeと描画画像。内部の関数・比較stateを直接呼ばず、本体にtest-only APIを追加しない。
- 優先behaviorとTDD順序は一つずつRED→最小実装→GREEN:
  1. Logsを隠し、TrainをBottom layer0 row0へ置く入力を公開AUIで準備する。宣言後・frame高さ変更後の半高、Centreと同じ幅、captionを最初のtracer bulletにする。Topにも同じ規則が働くことを回帰する。
  2. Q-Values/Traceの外側列の幅保持、Logs同居の除外とrow1への移動、Eval OFF/ONでの昇格・復帰を一caseずつ回帰する。既存Bottom除外caseはQ-Valuesを表示し、その列より外側のBottomへ移す。最小高・Logs非縮小、同じlayerのLeft/Right除外、最大化、サッシ相当の高さ変更の保持も確認する。不足があればそのcaseのREDから直す。
  3. 共通Panelへ描画をBindした最小canvasで、停止中の縦/横/両方向resize後の画素と明示的な全面再描画後の画素を比較する。既定引数と明示styleの両方を検査してREDを確認し、共通コンストラクタだけでGREENにする。
  4. 実CartPoleと既存LunarLanderの停止中resizeを回帰する。他のEnv Viewは同じPanelから継承しstyleを後から上書きしないことを確認する。ImageCls/DropMerge/GridMazeのdataset・学習・GPUを新規起動せず、共通の描画保証と実Runnerの代表経路で確かめる。AtariなどGLの実機操作は人の受入へ残す。
- 新規caseは`[.][prd110_main_pane]`とcase別tagを付け、RunnerApp起動caseは別processで実行する。Dock/LoadLayoutによる準備をdrag/drop/サッシの実操作の代用とはしない。実操作は受入8で人が行う。

## 検証

- 変更前にDebug全体とGUI targetをVsDevCmd経由でbuildする。ソース10ファイル・保護対象9ファイル・同名の隔離core/GUI実行体とhashを`.scratch/prd110/height-before-suites/`へ保存し、既定suiteの失敗集合を新規採取する。古い23impl時の結果を今回のbaselineへ読み替えない。
- buildは起動中Runに配慮して1並列で進め、RelWithDebInfoのRunnerは操作・上書きしない。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 1 && cmake --build --preset x64-Debug --target AnetRLRunner-gui-test --parallel 1'
```

- 各RED/GREENと最終の全GUI caseは新規`.scratch/prd110/height-*/`へ保存する。新旧の既定suite失敗集合を比較し、shipped_configとeval_steps_menu/prd105_ui_keys/prd109_uiも別processで確認する。再リンク中の実行体は実行せず、baselineは隔離コピーを使う。
- 改名の旧名、現用文書の「等幅」と受入9の旧18表現、LF/BOM/空白、追加productionの禁止パターン、保護ファイルとindexの不変を点検する。実行した検証と未実施・理由をここへ記録する。
- 全品質確認観点の自己点検後、新しい24implと今回の共同変更を独立an-reviewへ提出する。指摘は直して同じレビュアーへ出し直し、原文と結果を保存する。追加の有料cc-reviewは明示依頼がないため実行しない。

## 前提

- ユーザー判断が必要な未決は0。D15が高さの対象・予算・最小高・除外を確定し、共通Panelでの修正は今回のユーザーの明示依頼である。
- 共通Panelの全面再描画はresize時だけの保証で、データ更新・学習・通常のtimerの頻度を変えない。styleを渡した場合も必ず加える方法を採用する。
- 主paneの横方向の規則、pin/closeの守り、timer、caption、fallbackを保つ。幅の不足を縦の配置のときに縮小で埋める挙動は加えない。
- ユーザーの既存Runと無関係な未コミット変更を保持し、git add/commit/pushはしない。製品Runnerの実機起動はしない。受入8のAtari5/qbert/online・実drag/drop/サッシは人が行い、未実施ならPRD全体を完了扱いにしない。
- 既知の別件は、起票済みのMetrics Viewer LOD不整合、ユーザーが見送った設計160英§5.1の旧起動フロー、共通gui.cppの既存dynamic_cast3箇所。新しい問題として積み直さない。

## 経過と結果

- 2026-11-02: heightハンドオフ、D15(2)と受入、現在の幅処理とPanelを照合し、両変更の計画を本書へ保存した。共通GUIの実装の地図とADR0016を確認。次は変更前buildと基準記録を採取する。
- 変更前Debug全体・GUI target build成功。ソース10ファイル・保護9ファイル・同名隔離実行体を`height-before-suites/`へ保存し、core/GUIの既定suite採取を開始した。隔離実行体のため、この先の再リンクと競合しない。
- TDD1 RED: `height-red-split/`はBottom・frame高さ変更・Topの半高で3失敗、エラー0。現行の幅処理の前へ、同幅の単独Top/Bottom帯だけの高さの往復を最小追加した。Logs等の高さを予算へ入れず、帯以外の寸法を変更しない。
- TDD1 GREEN: `height-green-split/`は終了0。Bottom、resize、Topの半高と幅・captionが成功。次は外側列の幅保持を回帰する。既存Bottom除外caseはQ-Values表示・その外側layer11へ移す入力に改め、D15(2)との前提の衝突を解消する。
- 幅の回帰: `height-green-columns/`の外側Q-Values/Trace列とBottom/同居列の除外は全終了0。変更前の既定suiteも採取完了し、coreは終了0・失敗0（710.95秒）、GUIは既存Logs indent1件だけ（終了42、21.13秒）。次はLogs同居とrow分離を回帰する。
- `height-green-shared/`は終了0。Logs同居帯の寸法保持、row1への分離後の半高とLogs高保持、同layerのQ-Valuesを含む広い帯の除外が成功。次は昇格・復帰、任意の高さ、最大化と高さ不足を回帰する。
- 共通PanelのREDは`panel-red/`で6条件すべて画素不一致（既定/明示style×縦/横/両方向）。構築時に全面再描画を加え、LunarLanderの個別指定を除いた後の`panel-green/`は共通Panelと実LunarLanderが両方終了0。各6条件が成功した。CartPole/GridMaze/ImageCls/DropMergeのPanel継承とstyleの後続上書きなしをソースでも確認した。
- 高さ不足の最初の期待値200は、AUIのcaption・枠を含む実dock最小高222と一致せず失敗した。art metricを足す写しにも差が残ったため、本体の高さ計算は変更せず、公開LoadLayoutへ最小未満を渡してAUI自身の実最小高を得る検査へ修正した。これらの失敗記録は`height-green-restore/`、`height-panel-red/`、`height-panel-green/`へ保持し、成功結果へ読み替えない。
- `height-restore-final/`は終了0。Eval OFF/ONの占有とBottom復帰、任意の高さ250の保持、maximize中の非変更とrestore後の半高、Train OFF/ON、高さ不足時のwx実最小高とLogs高さ保持を確認した。GREEN後に幅・高さを扱う私有関数とテスト準備ヘルパーを上記の名前へ移行し、実CartPoleの停止中resizeを追加した。
- 作業中に利用者側の別commitでHEADが`74ebade9`から`d8ee7501`へ進んだ。差分はcreate-prd/improve skill日常コピー4ファイルとPRD品質観点だけで、今回の対象10ファイルのcommit差分は0。基準は保存済みの`74ebade9`とし、並行変更を保持する。indexは空のまま。

## 最終検証と自己点検

- VsDevCmd経由・1並列のDebug全体とGUI targetのbuildが成功。上のbuildコマンドを変更前後に実行した。今回Release/RelWithDebInfoはbuildせず、稼働中の製品Runnerは操作していない。
- `.scratch/prd110/height-after-suites/suites.json`で変更前後の既定suiteの失敗集合一致を確認した。coreは1069case成功（610.40秒）、GUIは28case中の既知の`Logs indent wrapped lines to the message on demand`1件だけ（終了42、17.03秒）。GUI既定の全成功とは報告しない。
- `.scratch/prd110/height-gui-final/`の37process / 37caseはすべて終了0。同じ最終GUI実行体のSHA256を照合した。内訳はPRD110の34caseとeval_steps_menu/prd105_ui_keys/prd109_ui。高さの4case、共通Panelの6描画条件、実CartPoleとLunarLander各6条件も成功した。通常のcaseで保険WARN0、保険caseは所定の発動1件を従来どおり確認した。
- `[shipped_config]`は14case成功。設定変更がないためcommon.txtの全値比較と既定葉検査は今回は再実行しない。不正modeの製品Runner起動検証も設定処理を変えていないため再実行せず、稼働Runの操作を避けた。
- 最終検証に使ったコマンド（出力先は当時新規）:

```bash
.\.venv\Scripts\python.exe -X utf8 -B core/anet-core/testdata/prd110/run_validation.py gui --output .scratch/prd110/height-gui-final
```

```bash
.\.venv\Scripts\python.exe -X utf8 -B .scratch/prd110/height_after.py core
```

```bash
.\.venv\Scripts\python.exe -X utf8 -B .scratch/prd110/height_after.py gui
```

```bash
.\.venv\Scripts\python.exe -X utf8 -B .scratch/prd110/height_after.py shipped
```

```bash
.\.venv\Scripts\python.exe -X utf8 -B .scratch/prd110/height_after.py compare
```

- 全品質確認観点の自己点検を`.scratch/prd110/height-selfcheck.md`へ記録。LF11ファイル、既存BOM8ファイル保持、保護9ファイル不変、禁止旧18表現0、現用3文書の「等幅」0、現用の旧関数名0、追加productionのdynamic_cast/TODO/ヘッダusing namespace0、index空を確認した。初回結果は`height-checks-initial.json`、レビュー前の最終結果は`height-checks-review.json`へ保存する。
- 自動検証終了時点では、人による受入8のAtari5/qbert/online、nativeマウスでのpin・drop・サッシ、動的HeatMap/Conv2dは未実施だった。公開AUIで作った配置やserializer入力を実操作の検証の代用とはせず、人の確認を残した。その後の受入結果は末尾の「人による実機受入」に記録する。
- 自己点検後の未決と、新たな範囲外の課題候補はなし。次に今回の共同変更を新しい独立an-reviewへ提出する。

## レビューと指摘対応

- 初回の独立an-reviewは計画OK、実装は文書2点の修正要求。原文は`.scratch/prd110/review-20261102-0144.md`。範囲外の検出はなし。
- 160日本語版のLogsのフォント説明まで用語を置換していた誤りを「固定幅フォント」へ修正し、元の意味を戻した。日英020のEvaluation Viewには、主Viewだけの上下帯を等分し、Logs・補助paneと同居する帯は除外する条件を明記した。
- 修正は上記3文書と本書の記録のみ。production・test sourceと検証済み実行体のhashは初回提出時から不変なのでbuild・自動テストは繰り返さず、文書の整合・LF/BOM・禁止旧表現・保護対象を再確認して同じレビュアーへ出し直す。文書の誤った「等分フォント」が残っていないことも検索する。
- 実CartPole/LunarLanderの最終actual/expected画像は各12枚、計24枚を`.scratch/prd110/height-paint-captures/`へコピーして保存した。元の一時Runは保持し、manifestへ出典とhashを記録。縦resize後のactualでsceneが描かれ、残像がないことも目視した。
- 2回目の同じan-reviewで計画・実装ともOK。原文は`.scratch/prd110/review-20261102-0148.md`。文書の2指摘は解消し、範囲外の検出・新たな課題候補はなし。再確認は固定幅フォントの意味、日英の同居帯の除外、LF/BOM保持。コード・検証結果は変更なしとして初回の確認を維持した。根拠なく見送った指摘はない。

## 完了報告

実施内容: 共通Panelの構築時にresizeの全面再描画を保証し、既定style・明示styleと他のCPU Viewへ適用した。LunarLanderの個別指定を共通へ移し、PRD4回目の上下帯の高さ等分も実装した。今回の実装・自動検証・自己点検・独立レビューを完了した。

変更ファイル: 共通gui.cpp、LunarLanderView.cpp、RunnerFrame.cpp/.hpp、RunnerFrame_test.cpp、検証README、160/020日英と本書の計11ファイル。PRD・CONTEXT・ADR・wiki・20〜23implと無関係な変更は保持した。

レビュー結果: 独立an-reviewは2往復で計画・実装ともOK。文書2点を修正した。Debug全体/GUI、GUI37case、core1069case、出荷設定14caseが成功。GUI既定は既知Logs indent1件で、変更前後の失敗集合一致を確認した。

決まらなかったこと: なし。新しい範囲外の起票候補もなし。

判断してほしいこと: なし。2026-11-02のユーザー報告で実機動作・受入確認OKとなり、受入待ちを解消した。エージェントがRelease/RelWithDebInfoをbuildせず、製品Runnerを操作していないという実施履歴は変わらない。

次にできること: ユーザーが指定した26ファイルをステージした後、コミットは人が行う。

## 人による実機受入（2026-11-02）

- ユーザーから「実機動作確認、受入確認OKで文書更新してからステージ実行」と報告・指示を受けた。これを人による受入8の確認結果として記録し、PRD110の実機受入待ちを解消して完了とする。エージェント自身が実機操作を行ったという記録ではない。
- 過去の20〜23implと、本書の自動検証当時の未実施履歴は保持する。検証READMEの現在の受入結果も同じユーザー報告へそろえる。
- 文書更新後のステージは、直前に提示した全体26ファイル・一部0ファイルの承認範囲で行う。コミット候補7ファイルは含めず、commit・pushは行わない。実装時のgit add禁止に対して、今回のユーザー指示でステージのみ明示的に承認された。

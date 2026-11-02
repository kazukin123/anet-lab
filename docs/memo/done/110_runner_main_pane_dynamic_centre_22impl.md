# PRD110 captionのpinによるfloat 実装メモ

## 概要

[PRD110](110_runner_main_pane_dynamic_centre_10prd.md)の2026-11-01の2回目の再グリルと `.scratch/prd110/codex-handoff-20261101-pin.md` に沿い、Centreの主paneをcaptionのpinボタンでfloatできるようにする。20impl・21implは過去の記録として保持し、今回は本22implを正本とする。初期表示の既定evalと既存の占有・幅・timerの規則は維持する。

基準HEADは `8c9fb3fdacff523262b5f71537a65344765f87df`。開始時indexのSHA256は `168d88849a3b3e179e4dd20c2cc5b820fd6c4b41fc4a69209f41305f5cd917a3`。初版のステージ内容、1回目の改訂差分、他の作業を保持し、git add・commit・pushは行わない。

## 主な変更

- RunnerFrame::UpdateMainPaneControlsで、CanHidePaneの同じ値をCloseButton・Floatable・PinButtonへ宣言する。HasPinButtonの差も変更検出へ含め、既存のCallAfterだけでUpdateする。
- AuiLayoutFrameはSetManagedWindowの後でwxEVT_AUI_PANE_BUTTONをBindする。pinでCanHidePaneが否ならイベントを消費し、他のボタン・許可されたpinはSkipしてwxの既存処理へ渡す。非pinイベントで判定や副作用を増やさない。短いhandlerは既存のBindと並べたlambdaで実装し、ヘッダ・公開APIは増やさない。
- pinでのfloat・maximize解除・floating windowの位置と大きさはwxへ任せる。占有の昇格は先回りせず、既存200ms UPDATE_UIへ任せる。拒否時はsavedHiddenState・配置・幅・timerを変えない。
- Q-Values・Evaluation Trace・HeatMap・Conv2dの登録元でPinButton(true)を指定する。Logsの既存pinを保ち、toolbarは対象外とする。
- 設計160の日英§2.1と§7.1、ガイド020の日英Evaluation Viewとpane一覧前、検証READMEの手動受入をpinの操作へ同期する。PRD・ADR0066・CONTEXT・wikiは起票側の改訂を保持して編集しない。

## テスト

- Public interface / surface: 実RunnerApp、Viewメニュー、captionボタンと同じwxEVT_AUI_PANE_BUTTONをframeへ送る既存Button helper、wxAuiManagerの公開pane状態、UPDATE_UI、実Runログ。pinではFloat()を直接呼ぶ代用をしない。新しいtest-only本体APIは追加しない。
- 優先behaviorとTDD順序。各段階を1テスト→RED→最小実装→GREENで進める:
  1. 既存Train-float caseをpinの押下へ移行し、両方dockedならpinがあり、TrainがfloatしEvalがCentreへ昇格する経路を通す。まずpin表示のREDを取り、同じ場所での宣言でGREENにする。
  2. 相方のfloat直後、UPDATE_UI前に残ったpinを押しても最後のdocked paneを外せないことを1caseでRED→GREEN。既存Eval-float caseの逆方向、初期Evalのみへのpin、maximize中のTrainのpinも同じ公開イベントで確認する。
  3. Logs・Q-Values・Evaluation Traceにpinがあり、公開pinイベントで外しても主paneの配置・操作可否が変わらないことを新caseでRED→GREEN。HeatMap・Conv2dは生成箇所の指定を照合し、人による受入も残す。
- 既存の起動・開閉a〜c・f・jにはHasPinButtonの確認を足す。全caseに従来の `[.][prd110_main_pane]` とcase tagを付け、RunnerAppを起動するcaseは1process1case。通常操作の保険WARN0も確認する。
- 実captionのpinのマウス操作、側面へのdrop、Atari5/qbert/onlineと動的HeatMap・Conv2dは人が行う受入8として残す。イベント経路の自動検証と、実画面での確認を同一視しない。

## 検証

作業ディレクトリはrepo root。bash fenceはAGENTS指定の実行ボタン用で、中身はPowerShell。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 4 && cmake --build --preset x64-Debug --target AnetRLRunner-gui-test --parallel 4'
```

- 変更前にDebug全体・GUI targetをbuildし、実行体を同じbasenameで `.scratch/prd110/pin-before-suites/binaries/` へコピーする。元DLLの場所をPATHへ設定して既定suiteを採取する。採取中の実行体を再リンクせず、source編集後も変更前の証跡を保持する。
- 各RED/GREENは新規 `.scratch/prd110/pin-*/` に保存し、前の結果を上書きしない。

```bash
.\.venv\Scripts\python.exe -X utf8 -B core/anet-core/testdata/prd110/run_validation.py gui --output .scratch/prd110/pin-gui
```

```bash
.\.venv\Scripts\python.exe -X utf8 -B core/anet-core/testdata/prd110/run_validation.py suites --output .scratch/prd110/pin-after-suites --baseline .scratch/prd110/pin-before-suites
```

- `[shipped_config]`も別processで確認する。不正値の製品Runner起動は設定を変更しない今回では繰り返さず、20implの保存結果を参照する。製品Runnerは人の承認なしに起動・終了しない。当初はDebugだけを対象としたが、Runnerが未起動と確認できたため、下記の追加Release buildも実施した。RelWithDebInfoは更新しない。
- 現用5文書で受入9の禁止文字列を空白を除いて検索し0件を保つ。改訂で追加されたドラッグについての旧記述も除去する。既存BOM・LF、diff --check、共有契約、旧名と追加コード、index不変を確認する。
- 全品質確認観点で自己点検後、独立したan-reviewへ本計画と今回の変更を依頼し、直せる不足は直して同じレビュアーへ再提出する。原文・結果・未実施と理由を本書へ記録する。

## 前提

- ユーザー判断が必要な未決は0。D2・D3・D14と§6の19〜21がpinの導線・可否・宣言・配置を確定している。wx側はframeへOnPaneButtonを先にBindし、あとからBindした基底handlerが先に受け取れる。許可されたpinはwxのFloatable確認・RestorePane・Float・Updateへ進む。
- 21impl末尾で候補にしたcaptionドラッグ開始の迂回は採用しない。wxのprotected状態・独自のmouse経路・新timer・新しい占有者stateは追加しない。
- CanHidePaneは主pane以外には既定許可を返す既存契約で、基底にTrain/Eval固有名を持ち込まない。新しいhandlerは基底のイベント処理の追加であり、設定・公開API・ヘッダは変更しない。
- observer、Eval非表示timer、AtariFPS、他の環境や学習、無関係なLunarLanderViewの変更は対象外。手動受入の未実施を理由なく完了へ読み替えない。

## 経過と結果

- 2026-11-01: pinのハンドオフ、改訂PRD、wxのOnPaneButtonと基底のBind順を照合。計画を保存し、変更前buildから進める。
- 変更前Debug全体・GUI targetのbuild成功。同名の隔離実行体と今回の変更起点のソースを `pin-before-suites/` へ保存して既定suiteの採取を開始した。
- TDD1 RED: `pin-red-train/` はTrainのHasPinButtonがfalseで1アサーション失敗。既存のUpdateMainPaneControlsでpinも同じ値に宣言し、差分検出へ含める最小修正を入れた。
- TDD1 GREEN: `pin-green-train/` は終了0。pinの公開イベントからwxのfloatとUpdate、既存UPDATE_UIによるEval昇格、timer継続とfloatのままの再表示まで成功した。
- TDD2 RED: `pin-red-guard/` は更新前に残る相方のpinで最後のdocked paneもfloatし、3アサーション失敗。基底のSetManagedWindow後にpinイベントのhandlerをBindし、CanHidePaneが否のpinだけを消費する最小修正を入れた。
- TDD2 GREEN: `pin-green-guard/` は更新前のpinの保護、Trainのpin、既存maximizeの3processが全終了0。初期の最後のEvalへのpinとmaximize中のpinも、既存wxの経路を確認する回帰caseとして足す。通常のa〜c・f・jへpinの有無と拒否も揃える。
- 変更前suite: coreは終了0・失敗0（720.94秒）、GUIは終了42で既存の `Logs indent wrapped lines to the message on demand` の1件失敗（17.30秒）。今回の比較ではこの実際の失敗集合を使い、成功へ読み替えない。
- TDD3 RED: `pin-red-aux/` はLogsのpinによるfloatは通り、次のQ-ValuesのHasPinButtonがfalseで1アサーション失敗。Q-Values・Evaluation Trace・HeatMap・Conv2dの登録元へPinButton(true)を指定する最小修正を入れた。
- Debug全体とGUI targetの最終build成功。変更後の既定suiteと、全pin GUI回帰の採取を開始した。既存20impl・21implは編集していない。
- 自己点検の機械検査: `pin_checks.py` が成功し、変更起点のソースとの比較で今回の改訂は8ファイルだけ、計画を含め9ファイルがLF、既存6ファイルのBOM保持、追加productionのdynamic_cast/TODOなし、共有契約の依存語0件、現用5文書の禁止14文字列0件、旧ドラッグ説明0件、index SHA256不変を確認した。保存結果は `pin-mechanical.json`。
- TDD3 GREENと全GUI回帰: `pin-gui/` の新規20case＋既存3caseの23processが全終了0、JUnitの失敗・errorなし。補助paneのpin、最後のEvalへのpin、最大化からのpin、更新前のpin連打、既存の位置復帰・幅・timer・選択枠が成功した。すべて同じ実行体SHA256 `40785a9c2750a7b6975a993a06466407be543ab9634dd27eebbcbc0ebe97afaf`。
- `[shipped_config]` は別processで14case・終了0（106.97秒）。JUnit・実行体ハッシュは `pin-shipped/` に保存した。
- 変更後の既定suiteも採取完了。`pin-after-suites/` はcore1069case・終了0・失敗0（633.76秒）、GUI28case・終了42で既存Logs indentの同じ1件だけ失敗（17.64秒）。変更前と失敗caseの集合が完全一致し、比較driverは終了0。全23GUIと既定GUI、shipped_configと既定coreの実行体SHA256もそれぞれ一致した。既定GUIを全成功とは報告しない。
- 文書の結果追記後の検査は `pin-mechanical-final.json` へ別名で保存する。検証済みproductionコードは追加変更せず、独立レビューへ進む。
- 追加Release build: レビュー待ちの間にRunnerが未起動と確認できた。起動中の実行体を更新しないという当初の前提の制約がなくなったため、起動・終了操作をせず、同じ検証済みソースで下記を実行し成功した。Release実行体の更新時刻は16:19:36、サイズ6,394,368 bytes。Debugに加えて次回のRelease起動にもpin修正が入る。実機受入を実施した意味ではない。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Release --target AnetRLRunner --parallel 4'
```

## 自己点検

- 実装品質確認観点の全項目を適用範囲ごとに走査した。変更は既存の機能グループ内の8ファイルと新規22impl。設定・公開API・ヘッダ・新しいclass・互換層・改名は追加していない。20impl・21impl、PRD・ADR0066・CONTEXT・wiki、LunarLanderView等の別作業は保持した。
- 基底のpin処理はイベントとpaneの情報だけで判断し、Train/Evalの名前を持たない。SetManagedWindow後のBindでwxより先に受け取り、拒否時には何も変更せず、許可・非pinはSkipする。拒否は意図された操作制限なのでエラーやWARNを足さない。CanHidePaneと最大化中のIsPaneOpenの既存契約を利用する。
- pinの表示は既存5Hzの2paneのフラグ確認・宣言へ比較を1個ずつ加えるだけで、配置の往復や割り当て・ロックは増やさない。変更時だけ既存のCallAfterを使い、昇格の先回り・独自mouse経路・新timer・別の占有者stateは足していない。
- 基底lambdaのthisは所有するframe自身、paneはイベント処理の間だけの借用で、借用メンバやSet/Clearは増やしていない。Agent・GetScalar・計測・device転送の変更はない。コメントは日本語、新しい実行ログはない。
- a〜c・f・jのpin有無とイベント、mの最後のpaneとmaximize、nのLogs/Q-Values/Traceを公開イベントで確認した。HasPinButtonとwxの実処理を読み、Float()だけでpin操作の代用をしない。HeatMap・Conv2dは登録箇所のPinButton(true)を照合し、実際の生成・pin操作は受入8へ残した。製品にtest-only APIを追加していない。
- 日英の設計160とガイド020、READMEをpinへ同期した。020本文は利用者の手順だけを記載し、wxの制約と理由は160へ置いた。禁止14文字列と旧ドラッグ説明は現用5文書で0件。保存後の本文とハッシュ、LF・BOM・差分空白を確認した。
- core gui.cppの既存dynamic_cast3箇所（既知のF-0020）、既定GUIのLogs indentの既存失敗は今回修正しない。追加したproduction行にdynamic_cast/TODOはない。実画面のpinとdrop、Atari5/qbert/online、動的HeatMap/Conv2dは未実施とし、PRD全体の受入完了扱いにしない。

## レビュー

- 独立an-reviewは1回目で計画・実装ともOK、修正要求0件、開発者への確認事項0件。原文は `.scratch/prd110/review-20261101-1609.md`。Bindの順と拒否・Skipの配送、同じ判定でのpin宣言、wxの最大化解除、RED/GREENと23process、既定suiteの失敗集合、LF・BOM・共有契約・重複識別子を再確認した。根拠がなく見送った指摘はない。
- 判定の依頼時点ではReleaseは未更新だった。提出後のソース変更はなく、上記追加buildでReleaseも更新した。レビュー後は本書とREADMEに結果・状態の記録だけを追記している。
- 範囲外の検出1件: 設計160jpのMetrics ViewerのLOD記述が、§2.1のfactor 4と§7.1の完成した16子で一致していない。変更前ソースにも同じ記述がある。追加の読み取りでLodBucket.LOD_FACTOR=4とLodIngestWriter.combineの4子チェックを確認した。ユーザーが別件の記録を採用したため、[LOD記述整合の暫定PRD](../999_metrics_viewer_lod_documentation_consistency_10prd.md)へ記録した。pin修正とは切り離し、記述の修正には着手していない。

## 決まらなかったこと

- pinの仕様・実装に未決はない。実マウスとAtari5/qbert/online、動的HeatMap・Conv2dの受入8は人が行う未実施項目のまま。
- 記録済みの別件: [999_metrics_viewer_lod_documentation_consistency_10prd.md](../999_metrics_viewer_lod_documentation_consistency_10prd.md)。ユーザーが「課題として記録する」を選択したため起票した。根拠は独立レビューの範囲外指摘と現行Java実装の4子チェック。現用の暫定PRDに同じ記述不整合はなく、過去のfrozenのSQLite移行案とは別件。未処理の起票候補はない。

## 完了報告

- 実施内容: 改訂PRD110のpin表示・押下の保護と全paneへのpin指定を実装し、Debug検証・自己点検・独立レビューを終了した。Runner未起動を確認して追加Release buildも成功した。製品Runnerの起動と実機受入は行っていない。
- 変更ファイル: 今回のpin改訂8ファイルと本22impl。レビューで見つかった別件は、ユーザーの採用により上記999暫定PRDへ記録した。20impl・21implと無関係な作業は保持した。
- レビュー結果: 1往復で計画・実装ともOK。pinの配送・宣言と拒否、wxの最大化解除、23 GUI process、既定suiteの失敗集合一致を確認済み。見送った指摘はなく、範囲外の1件は別件として記録した。
- 決まらなかったこと: pin仕様の未決なし。未処理の課題記録候補なし。実マウス操作、Atari5/qbert/online、動的HeatMap・Conv2dは人が行う受入待ちとして残るため、PRD全体は受入完了扱いにしない。
- 判断してほしいこと: なし。追加差分は未ステージで、git add・commit・pushは行っていない。
- 次にできること:
  1. 人が検証READMEの実機受入手順を実施し、結果を本書へ記録する。

# PRD110 左右の主paneの等幅とCentre表示 実装メモ

## 概要

[PRD110](110_runner_main_pane_dynamic_centre_10prd.md)の2026-11-01の3回目の再グリルと `.scratch/prd110/codex-handoff-20261101-width.md` を実装する。D15の等幅条件・配置変化時の再適用、D16のcaptionの印を本書の正本として進め、20〜22implは履歴のまま保持する。起動の既定eval、pin、閉じる制限、timer、占有と位置復帰は維持する。

開始時HEADは `74ebade925c53a9ffcb2486416d67d8aea891098`、ステージ差分のSHA256は空の差分の `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。ユーザーが前回からGit状態を変更しているため、過去のindex状態を復元しない。git add・commit・pushは行わない。

## 主な変更

- RunnerFrame::ApplyMainPaneWidthsを左右の縦dockの集計へ一般化する。方向・layer・rowをdockのキーにし、一方がCentreでもう一方がLeft/Rightなら、その側面dockを主領域の半幅にする。主領域はclient幅から、それ以外の表示中の縦dockを除いた幅。Train/Evalの名前やlayer・rowで分岐しない。
- Top/Bottomの主paneと、主pane以外が同じ側面dockに同居する列の等幅は対象外とし、wxの配分を保つ。その列や既知の補助列以外の縦dockも予算には数えるが、主paneを含む同居列や未知の列を縮めない。既存の最小幅200+200、単独Centre200、補助列→Q-Values列の縮小順を保つ。
- 幅の書き込みと記憶は既存のSaveLayout/LoadLayoutとpane.dock_sizeを使い、Frameに幅の別stashを足さない。
- 主paneの方向・layer・row・位置・float・表示の6値を2pane分だけ比較する私有の値型を置く。幅・geometry・maximizeは比較値へ入れない。起動・Reset・通常ポリシー適用後は現在の置き場へ同期する。
- OnUpdateTrainStatusは、最大化の外で占有を当てたあとに置き場の変化も確認し、既存のCallAfterへ集約する。宣言・占有の変更だけがUpdateを要求し、配置比較だけの変更では不要なUpdateを行わない。callbackで最新の置き場を再確認し、変化しているときだけ幅を1回当てて比較値を更新する。最大化中は幅を当てず、restoreの既存ポリシーに任せる。
- UpdateMainPaneControlsでcaptionも状態から宣言する。表示中・docked・Centreの主paneだけ本体の末尾に半角スペースと`[Centre]`を付け、それ以外は本体に戻す。既存のpin・✕・Floatableの変更はUpdate、captionだけの変化は再描画と浮動タイトルの同期で反映する。
- EvalTracePanelは登録時のcaptionを保持し、Traceのstep注記を付ける。captionを固定文字列で上書きせず、Centreの宣言では自分の印だけを除去して必要な場合だけ末尾へ付け直す。既存Trace注記と本体を保ち、浮動frameのタイトルも同期する。新しい公開APIは足さない。
- 設計160日英§2.1・§7.1、ガイド020日英のEvaluation Viewとpinの文の後、検証READMEをD12へ同期する。PRD、CONTEXT、ADR0016/0066、wikiは起票側の変更を保持して編集しない。

## テスト

- Public interface / surface: 実RunnerApp、公開のpin/menu/toolbar/resize/maximizeイベント、wxAuiManagerの公開pane状態・構造化serializer、UPDATE_UIとCallAfter。Float/Dock/LoadLayoutによる準備は規則への入力であり、利用者のdrag・drop・サッシ操作の再現とは扱わない。製品にtest-only APIを追加しない。
- TDDの順序は1behaviorずつ、テスト→RED確認→最小実装→GREEN:
  1. Trainをpinで外し、Left layer1へDockしてUpdateした置き場を準備する。宣言後とリサイズ後に半幅になるcaseを最初のtracer bulletにする。幅集計と置き場の変化を契機にする経路を最小実装する。
  2. Rightの外側layerにTrainを置き、間のQ-Values列を除いた主領域で半幅となりQ幅を保つこと、左右の補助列の予算、Bottomと同居列の除外、最大化中の幅を回帰する。既に満たす条件はそのまま回帰し、不足があれば1caseのRED→最小修正→GREENで直す。
  3. 公開SaveLayout/LoadLayoutで置き場を変えず幅だけを準備し、宣言の繰り返しでは幅を戻さず、次の置き場の変化・遷移で等幅へ戻ることを確認する。サッシの実操作は受入8へ残す。
  4. captionの初期状態・開閉・pin・左右・非表示・Reset・maximize/restoreで印を検査するcaseをRED→最小宣言→GREENにする。主pane以外には付かないことも確認する。
- 幅の期待値は公開serializerで実dockサイズを読み、表示中の縦dockを重複なく数えて主領域を求め、半分と2px以内で比較する。
- 既存の非定位置への復帰テストは位置4値を保ち、再表示後の幅は半幅へ移す。Centre中の記憶幅を確かめる箇所では、2回目の昇格の直前に実際の等幅後の幅を捕捉し、その値の保持を確かめる。過去の任意幅が降格後も残るという断定はしない。
- 全caseは従来の `[.][prd110_main_pane]` とcase別tagを持ち、RunnerAppを起動するcaseは別process。既存のpin・timer・位置・保険・選択枠と3つの旧GUI caseも回帰する。drag/drop/sashとAtari5/qbert/onlineの実画面は人の受入として残す。

## 検証

repo rootから実行する。bash fenceはAGENTSの実行ボタン用で、中身はPowerShell。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 4 && cmake --build --preset x64-Debug --target AnetRLRunner-gui-test --parallel 4'
```

- 変更前build後、同名のcore/GUI実行体と今回の起点ソースを `.scratch/prd110/width-before-suites/` へ保存し、元のDLLの場所をPATHへ設定して既定suiteを採取する。採取中に実行体を再リンクしない。
- 各RED/GREENは新規 `.scratch/prd110/width-*/` へ保存し、既存結果を上書きしない。

```bash
.\.venv\Scripts\python.exe -X utf8 -B core/anet-core/testdata/prd110/run_validation.py gui --output .scratch/prd110/width-gui
```

```bash
.\.venv\Scripts\python.exe -X utf8 -B core/anet-core/testdata/prd110/run_validation.py suites --output .scratch/prd110/width-after-suites --baseline .scratch/prd110/width-before-suites
```

- shipped_configを別processで確認する。製品Runnerは人の承認なしに起動・終了しない。Debugで必要な確認を済ませ、Releaseへの追加buildの可否は実行中のRunnerを妨げない条件で判断し、行った場合は記録する。
- 現用5文書の受入9の旧記述を空白違いも含めて検索し0件にする。LF・既存BOM・差分空白・追加productionコード・共有契約・index不変を点検する。
- 全品質確認観点の自己点検後、独立an-reviewへ本計画と今回の改訂を依頼し、指摘を直して同じレビュアーへ再提出する。原文・結果・未実施の理由を本書へ記録する。

## 前提

- ユーザー判断が必要な未決は0。D15・D16と§6の22〜27が幅の対象と時機、captionの文言、サッシの扱いを確定している。
- 配置比較の記憶は幅の記憶ではなく、2paneの6値だけ。占有者や浮動windowの位置を新しいstateで持たず、既存CallAfter・最大化ガード・構造化往復を使う。取得したpane参照をLoadLayout後へ持ち越さない。
- pin・timer・最後の主pane・未知経路の保険は既存の規則を保つ。新しい設定・公開API・timer・イベント・mouseの迂回は追加しない。
- 追加の早期close再現で、wxのfloating frameの遅延破棄前に再表示すると、旧sizerの所有が残ることを確認した。共通AuiLayoutFrameの許可済みnative closeで、移動するwindowだけを公開sizer APIでDetachしてからwxのClosePaneへ進める。Vetoとメニューhideには適用しない。共通GUI機構の寿命の問題なのでRunner固有の分岐を足さず、`core/anet-core/src/gui.cpp`も今回の変更対象へ追加する。追加直前のソースは`width-before-gui-close/`へ保存した。
- LOD記述の別件は既存の999暫定PRDで管理し、幅修正へ混ぜない。無関係な未コミット変更を保持する。

## 経過と結果

- GUI全回帰で既存`prd110_float`がRED（幅の不一致0・wx sizer例外1）。`width-red-float-path/`の単独再実行でも再現。配置変化だけのcallbackが無条件にAUI Updateを呼び、閉じたfloatのframeが破棄待ちの間に非表示frameを再生成していた。宣言・占有の変更で必要なUpdateだけを予約に集約し、配置比較だけのときは幅の処理へ進める。新しい公開APIやwx内部への介入は加えない。Debug全体・GUIを再buildし、`width-green-float-path/`はexit 0。

- 2026-11-01: widthハンドオフ、改訂PRD、既存の幅往復とUPDATE_UI、位置復帰テストを照合して本計画を保存。変更前buildから開始する。
- 変更前Debug全体・GUI targetはbuild成功（no work）。`width-before-suites/` へ起点ソース・保護対象のhash・同名の隔離実行体を保存し、既定suiteの採取を開始した。
- TDD1 RED: `width-red-left/` は配置変更後・リサイズ後の半幅の不一致で2アサーション失敗。左右dock集計、方向による等幅判定、置き場6値の比較と既存CallAfterでの再適用を最小実装した。
- TDD1 GREEN: `width-green-left/` は終了0。次は外側Rightと左右の別列、Bottom・同居列の除外、幅だけの変更と最大化を回帰し、非定位置復帰の旧断定をD15の半幅へ移す。
- 変更前の既定suiteはcore終了0・失敗0（679.32秒）、GUIは既存Logs indentの1件失敗（終了42、33.28秒）。今回の前後比較はこの失敗集合を用いる。
- 幅の回帰 `width-regression/` は外側Right・左右の列・Bottom/同居列・幅だけの変更/最大化・旧位置復帰・列の縮小順・Leftの8processが全終了0。
- 追加自己点検: 保険で両Centreを修復した場合、修復後の置き場まで比較値へ記憶すると、修復前に適用できなかった等幅が次の宣言で当たらない可能性がある。保険caseに任意幅からの修復後の半幅を追加し、REDから確認する。
- 追加RED: `width-red-fallback/` は保険修復後の半幅で失敗。比較値は幅ポリシーを実際に当てた直後、最後の保険より前で同期する最小修正を入れた。保険で変わった置き場は次の宣言が検出し、再修復やWARNなしで幅を当てる。
- 追加GREEN: `width-green-fallback/` の保険・幅だけの変更/最大化・外側Rightは全終了0。保険のWARNは発動1件のまま、修復後の半幅も成功した。次はCentreのcaptionを1caseのRED→GREENで実装する。
- caption RED: `width-red-caption/` は印の未表示と既存Trace注記との順序・浮動タイトルで失敗。UpdateMainPaneControlsの同じ宣言で自分の印だけを正規化し、浮動frameのlabelも同期する最小修正を入れた。印がすでに正しい場合は文字列を組み立て直さない。
- caption GREEN: `width-green-caption/`は終了0。最終版はfloat回帰修正後にDebug全体とGUI targetを再buildし、ReleaseのAnetRLRunner targetもVsDevCmd経由で成功した。製品Runnerの起動・終了は行っていない。
- 初回レビュー前GUI: `.venv/Scripts/python.exe -X utf8 -B core/anet-core/testdata/prd110/run_validation.py gui --output .scratch/prd110/width-gui-final`は28process / 28caseが全終了0。新規5case、既存20case、`eval_steps_menu`・`prd105_ui_keys`・`prd109_ui`を含む。28processのbinary SHA256はすべて当時のDebug GUI実行体の`c5dcf6b988c3f3b0c032f94a35e083af6719a34148dcf0dc089a9fe008dc5491`と一致。
- 既定suite: 再リンクとの競合を避けるため、`.scratch/prd110/width_after_suites.py core`と`gui`で個別採取し、同scriptの`compare`で比較した。`width-after-suites/suites.json`で失敗集合一致。coreは1069case・失敗0・終了0（651.82秒）、GUIは既存の`Logs indent wrapped lines to the message on demand`1件だけ（終了42、21.67秒）。coreの実行体hashは変更前・後・shipped_configのすべてで一致。
- `.scratch/prd110/width-shipped/`の`[shipped_config]`は14case・終了0（102.45秒）。本改訂は設定の読み取り・既定・診断を変えていないため、製品Runnerを起動する不正設定の再検証は行っていない。人による実機受入8（drag/drop、サッシ、Atari5・qbert・online）は未実施。ユーザーの指定に従い人が行い、PRD全体を完了扱いにしない。

## 自己点検

実装品質確認観点の全73項目を適用範囲に従って走査した。実装判断の未決はなし。初回レビューで見つかった別件の起票候補1件は「レビュー」に記録する。

- 1〜8・62・71・72: 追加productionにdynamic_cast・TODO・header using namespaceなし、共有契約の非依存語0、23implの名前と対PRDを確認。初回レビュー時はLF9ファイル・既存BOM6・改訂8ファイル、共通GUI修正追加後はLF10ファイル・既存BOM7・改訂9ファイル。保護8ファイルをhashで照合。重複識別子リテラル検査は出力0。改名・skill変更なしの項目は対象外。HEADと空のstaged差分を保持。
- 9〜22: D15/D16と受入2o〜r・8・9に対応。既存の任意幅保持の断定だけを改訂し、位置4値は保持。幅の別stash・公開API・新設定・互換コードを足していない。過去の20〜22impl、起票側のPRD・用語・ADR・wikiは保持。人の受入を自動検証で代用していない。
- 23〜29・54・64・67・68・70・73: 今回は新設定、入力の失敗処理、背景処理、本番設定を固定するテストを足しておらず対象外。既存の保険WARNは1発動1行を回帰。
- 30〜41・55〜61・63・65・66・69: 幅と印は既存RunnerFrame内、native closeの所有解放は共通AuiLayoutFrameに置く。配置比較は私有の値型、借用を保持せず、LoadLayout後に参照を使わない。sizerの所有・破棄はwxの規約に従う。追加includeは所定位置、初期化は指示付き、captionの印は単一定数。UIに学習ロック・device同期を追加していない。初回自己点検はTraceがcaptionを繰り返し上書きする経路を見落としていた。指摘対応後はcaptionだけなら再描画に限定し、実Traceの連続行移動で全体更新0回を確認。200 msごとは2paneの配置値とcaptionの比較だけで、配置・ボタンの変更があるときだけcallbackを1回へ集約する。幅処理の既存`ANET_PROFILE_SCOPE(layout)`を維持。コメントは日本語、未対応の分岐はなし。
- 42〜44: GetScalarとAgentのState/Resource・依存関係を変えておらず対象外。
- 45〜53: ログは追加せず、publicなGUI経路・serializerで検証。Debug全体・GUI・Releaseと関連検証を実施し、未実施の実機受入を理由つきで記録。設計160・ガイド020の日英と検証READMEを同期。PRD資材の既存位置と設計からの参照を維持し、AGENTSは変更なし。設定・skill変更の項目は対象外。
- 受入9: 現用5文書について初版14表現と今回追加4表現を空白違いも含めて検索し、18表現すべて0件。詳細は`.scratch/prd110/width-mechanical-pre-green.json`と最終再検査記録に置く。

## レビュー

- 自己点検後、今回の23implと改訂8ファイルを独立an-reviewへ提出する。原文と判定は受領後ここへ記録する。
- レビュー待ちの追加自己点検で、Centreをpinした直後、200 msのcaption宣言より先にnative floating frameを閉じる条件が前のfloat caseにはないことに気づいた。`prd110_float_before_declaration`を追加し、REDから確認する。レビュアーへ対象変更を通知し、変更後に差分と検証を再提出する。
- `width-red-early-close/`はsizer例外1件でRED。captionだけを再描画へ変えても`width-early-close-caption-only/`で同じ例外が残ったため、native closeの共通処理で旧sizerからのDetachを最小追加した。共通GUI変更後はcore既定suiteとshipped_configも新しい実行体で再採取する。
- 独立レビュー1往復目は計画OK・実装差し戻し1件。実Traceの行移動が登録時captionを上書きし、印の正規化が200 msごとのAUI全体更新になる点を指摘された。原文は`.scratch/prd110/review-20261101-2220.md`。captionだけの変化は既存Traceと同じ`Refresh(false)`と浮動frameのlabel更新へ限定し、配置・ボタンの変更だけをUpdateの合図にする。実Traceの`DoStep`→`MoveSelection`→UPDATE_UIで、Centre時とfloat時に連続6行ずつ移動する回帰caseを追加。公開root sizerをtest-onlyのsizerで包み、その破棄回数から不要な全体更新を観測する。本体APIは追加しない。
- 範囲外の新規指摘は、設計160英§5.1に古い同期起動フローが残ること1件。今回の幅改訂前からの記述で、対象章は§2.1・§7.1のため編集せず、工程末に起票候補として提示する。
- 共通GUI修正後のGREEN: `width-green-early-close/`で早期close・通常Eval float・Train floatが全終了0。`width-green-real-trace/`で実Trace連続行移動（Centre/floatとも全体更新0回、captionと浮動タイトル正常）・早期closeが終了0。
- 追加修正後のDebug全体・GUI targetとRelease RunnerはVsDevCmd経由で再build成功。製品Runnerの起動・終了なし。`width-shipped-final/`は更新後core実行体で14case・終了0（119.48秒）。coreとGUIの既定suite、およびGUI全30processを新規の結果へ採取し、完了後に最終比較と再レビューを行う。
- 最終結果: `.venv/Scripts/python.exe -X utf8 -B core/anet-core/testdata/prd110/run_validation.py gui --output .scratch/prd110/width-gui-reviewed`は30process / 30caseが全終了0。28caseに早期closeと実Trace captionの2caseを追加した。実行体SHA256は全30processと既定GUIで`fc5c51ac48387c12849a9688e6fbe858bd54f5cfea25bb6a9ffb813bc7dcd1c5`に一致。
- 最終既定suite: `.scratch/prd110/width_after_suites.py core .scratch/prd110/width-after-review-suites`と`gui`で採取し、同scriptの`compare`が成功。core1069case・失敗0・終了0（626.62秒）、GUIは変更前と同じLogs indent1件・終了42（22.27秒）。`width-after-review-suites/suites.json`で前後失敗集合一致。coreのSHA256は`f4da81adadf9355c170479452ec6bbe42dfde9b14b21bc8a33fc30e81af73fd8`で最終shipped_configの実行体と一致する。
- 再自己点検: 実Traceでの全体更新0回と早期close GREENを含め、品質確認観点の全73項目を再走査。`.scratch/prd110/width-mechanical-review2.json`で改訂9ファイル・LF10・既存BOM7・保護8・旧表現18種類0件・index不変を確認。追加productionは共通GUIの許可済みnative closeの6行とcaption再描画への限定だけで、公開API・設定・新イベント・timerは追加していない。独立レビューは同じレビュアーへ再提出する。
- 再レビュー2往復目は計画・実装ともOK、修正要求0件。原文は`.scratch/prd110/review-20261101-2252.md`。Traceの連続行移動で全体更新をしないこと、早期closeの解放が許可済みnative closeだけに作用すること、最終検証と実機受入未実施の区別が再確認された。根拠がなく見送った修正要求はなし。
- 再レビューの範囲外に出た共通GUIの既存dynamic_cast3箇所は、同じ会話のpin改訂レビュー`.scratch/prd110/review-20261101-1609.md`ですでに「既知の3箇所」と報告済み。追加修正前の原本にも同じ3箇所があり、増加なし。既報のため新しい起票候補には重複追加しない。

## 完了報告

実施内容: D15の左右の任意dockでの等幅と配置変更時の適用、D16のCentre印を実装。レビュー指摘のTrace行移動時の全体更新と、追加で再現したpin直後のclose例外も修正した。

変更ファイル: RunnerFrame.cpp/.hpp、RunnerFrame_test.cpp、共通gui.cpp、設計160日英、ガイド020日英、検証READMEの9ファイルと本計画。起票側のPRD・用語・ADR・wikiと20〜22impl、無関係な変更は保持。

レビュー結果: 独立an-reviewは2往復で計画・実装OK。Debug全体・GUI target・Release Runnerのbuild成功。最終GUI30caseと出荷設定14case成功、既定suiteの失敗集合は変更前と一致。詳細と未実施は上記のとおり。

決まらなかったこと: なし。別件の設計160英§5.1の古い同期起動フロー1件は、999暫定PRDとPRD109の記録で同じ内容の既存課題が見つからず、新規候補として提示した。ユーザーが「今回は見送る」と回答したため、起票せず既存文書も変更しない。

判断してほしいこと: 実装判断・起票判断の待ちはなし。人による受入8は指定どおり未実施で残し、PRD全体を完了扱いにしない。

次にできること: 人がRelease版で受入8の左右の縁へのdrop、等幅、サッシ、Centreの印、pinとclose、Atari5・qbert・onlineを確認し、結果を記録する。製品Runnerの起動・終了、add・commit・pushは行っていない。

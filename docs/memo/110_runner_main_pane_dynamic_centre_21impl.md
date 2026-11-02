# PRD110 改訂版の対称なCentre占有 実装メモ

## 概要

[PRD110](110_runner_main_pane_dynamic_centre_10prd.md) の2026-11-01改訂と `.scratch/prd110/codex-handoff-20261101.md` を実装する。初版の[20impl](110_runner_main_pane_dynamic_centre_20impl.md)を保持し、今回の正本を本書とする。設定の既定eval、描画timer、無効な表示中ボタンの選択枠は維持する。

最新状態（2026-11-01 08:03）: 人による確認で、左右表示中のTrainのcaptionをドラッグしてもアンドックできない不具合が判明した。末尾の調査記録のとおり未修正であり、以下の自動検証・レビュー成功は実captionドラッグの成立を証明していない。

基準HEADは `8c9fb3fdacff523262b5f71537a65344765f87df`。初版の20ファイルはユーザー指示でステージ済みのままであり、今回git add・commit・pushは行わない。今回の差分はその初版への追加変更として記録する。

## 主な変更

- FloatableをCanHidePaneと同じ「相方が開いてdocked」の判定へ統一する。Centreという方向やTrain/Evalの名前で許可を分けない。最大化中は既存IsPaneOpenを使う。
- ApplyMainPaneLayoutは、現在のCentre占有者を維持し、不在のときだけ表示中でdockedの相方を昇格する。floatしたpaneが側面にdockされても、その場所を変更しない。
- 表示フックの遷移後状態と明示した表示対象を使い、Centreのpaneの再表示を正当な復帰として扱う。最大化で相方も一時非表示になっている場合も表示対象を取り違えず、相方を降格してから基底のUpdateへ進む。単なる通常ポリシーの両方Centreは静かに正常化せず、最後の保険へ渡す。
- Eval専用の位置記憶を、昇格した主paneの方向・layer・row・位置を持つ1組へ移す。昇格時に上書き、元の占有者の再表示で復元、未記憶はRight/layer0/row0/position0。Resetで消す。幅は既存pane.dock_sizeに持たせる。
- OnUpdateTrainStatusの既存200ms UPDATE_UIで、最大化ガードを尊重しながら占有規則も再適用する。変更時だけ既存CallAfterでUpdateし、毎tickの幅往復・新timer・イベントは追加しない。
- 既存のEval Step/Resumeの自動表示も表示前フックへ接続する。元の占有者の再表示を単なる両Centreの直接操作と取り違えず、G5の既存導線を維持するために必要である。
- 通常ポリシーの最後の保険へ、両方が表示中・docked・Centreの場合を追加する。Trainを残してEvalを記憶位置かRight定位置へ降格し、指定の英語WARNを発動ごとに1行出す。再適用では出さない。両Centreのまま幅の往復は行わない。
- 単独Centreの最小幅200をTrain/Evalともに適用する。Trainが側面にいる列は補助列/Q-Values列の縮小対象から外し、wxの配分へ任せる。定位置の50:50と補助列→Q-Values列の順を保つ。
- 設計160の日英§2.1・§7.1、ガイド020の日英Evaluation View、PRD110検証READMEを改訂仕様へ同期し、受入9の初版固有文を現用範囲からなくす。起票側のPRD・ADR0066・CONTEXT・wikiは編集しない。

## テスト

- Public interface / surface: 実RunnerAppのViewメニュー・Panels toolbar、wxAuiManagerの公開pane操作と実状態、FrameのUPDATE_UI、protected timerを読む既存test subclass、Runログ。製品にtest-only APIを追加しない。
- 優先behaviorとTDD順序（各項目を1テストのRED→最小実装→GREENまで進めてから次へ）:
  1. 既存の開閉往復caseで、両方dockedならTrainもFloatableになることをtracer bulletとして確認する。
  2. TrainをFloat＋Updateし、UPDATE_UIでEvalがCentreへ昇格する。更新前のEvalの✕は拒否され、Trainのtimerは継続し、Trainを隠して再表示してもfloatを保つ（受入2j）。
  3. floatしたTrainをLeftにdockしてもEvalが占有者のまま。EvalのOFF/ONでTrainの昇格・元位置復帰、TrainのOFF/ONで側面位置維持、Resetで記憶消去を確認する（2k〜l）。
  4. 既存の保険caseへ、直接操作で両Centreにした場合の修復・指定WARN1件・再適用で増えないことを追加する（6）。通常遷移では両保険WARN0件。
  5. 単独Train Centreでも最小幅200と列の縮小順を保ち、側面Trainの列へ縮小を強制しないことを確認する。
  6. 自己点検で見つかったStep/Resumeの自動表示も、公開ツールバー操作で正当なCentre復帰となり、保険WARNが出ないことを確認する。
  7. レビューで見つかった既表示paneへの冗長なONを、公開メニューイベントで確認する。直接操作で両Centreにした状態は正当な再表示とせず、Trainを残してEvalを降格し、指定WARNが1件だけ出ることを確認する。
- 新caseも `[.][prd110_main_pane]` とcase別tagを付け、1process1case。既存の起動・timer・幅・maximize・選択枠と3つの旧GUI caseも回帰する。

## 検証

作業ディレクトリはrepo root。AGENTSの指定によりPowerShellのコマンドもbash fenceで記録する。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 4 && cmake --build --preset x64-Debug --target AnetRLRunner-gui-test --parallel 4'
```

- 変更前buildは成功（両方no work）。実行体を同名で `.scratch/prd110/revision-before-suites/binaries/` へコピーし、元のDLLの場所をPATHへ設定して既定suiteを採取する。source変更・再リンクが採取中の実行体へ影響しないようにする。JUnit、終了値、実時間、実行体SHA256、失敗case集合を同ディレクトリへ保存する。
- 各RED/GREENは新規 `.scratch/prd110/revision-*/` 配下へ保存し、上書きしない。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd110/run_validation.py gui --output .scratch/prd110/revision-gui
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd110/run_validation.py suites --output .scratch/prd110/revision-after-suites --baseline .scratch/prd110/revision-before-suites
```

- 製品Runnerを起動するfatalモードと実機受入8は、人の承認なしに実行しない。今回は設定・RunnerAppの検証を変えないので不正値・shipped_configには20implの保存結果を使う。実機受入8は人による確認として残す。
- 文書の初版固有文を受入9に従い検索し、指定された現用文書で0件を確認する。差分空白、LF、既存BOM、旧位置記憶名の移行漏れ、ステージ済み差分の不変を確認する。
- 全品質観点の自己点検後、独立an-reviewへ計画と実装を依頼し、指摘を直して再レビューする。結果・未実施・理由を本書へ記録する。

## 前提

- 未決事項監査のユーザー判断ブロッカーは0。D1〜D3・D5・D13と受入2j〜l・6は改訂PRDが確定済み。現在のTrain優先・Centreのfloat不可との食い違いは今回の修正対象である。
- 基底AuiLayoutFrameのCanHidePane、IsPaneOpen、Updateの予約、dock_sizeの保持は既存機構を利用する。公開API・設定キー・既定値は変更しない。
- 位置の記憶はFrameの1組の4値だけ。占有者の別stateを増やさず、live paneの表示・float・Centre情報から判断する。元の占有者の再表示は表示前フックで処理する。
- 両方が側面にありCentreが不在という未規定の直接操作では、表示中でdockedの候補からTrain、なければEvalを選ぶ。不正な両Centreとは分け、現在の有効な占有者を置き換える用途には使わない。
- observer切り離し、Eval非表示時timer停止、AtariFPS、起動の非同期化、LogPanelやRAMメトリクス等は範囲外。すべての無関係な作業・ステージ内容を保持する。

## 経過と結果

- 2026-11-01: ハンドオフと改訂PRDを照合し、本計画を保存。Debug全体・GUI targetの変更前build成功。隔離した実行体で変更前の既定suiteを採取中。
- TDD1 RED: `.scratch/prd110/revision-red-floatable/`。既存の往復caseでTrainのFloatableがfalseのため失敗。両方の導線で使うCanHidePaneとFloatableの判定を一致させる最小修正を入れ、GREENを確認する。
- TDD1 GREEN: `.scratch/prd110/revision-green-floatable/` は終了0。開閉・位置・幅の既存往復が成功した。次はTrainをfloatしたあとのUPDATE_UIによるEval昇格を検査する。
- 変更前の隔離suiteはcore終了0・失敗0（570.01秒）、GUI終了0・失敗0（16.17秒）。前回の20implの一部採取では既存Logs indentが1件だったが、今回の変更前採取では0件であり、今回の失敗集合は両方0件を基準にする。
- 今回はindexを変更しない。作業開始時の `git diff --cached --binary` SHA256は `168d88849a3b3e179e4dd20c2cc5b820fd6c4b41fc4a69209f41305f5cd917a3`。
- TDD2 RED: `.scratch/prd110/revision-red-train-float/` は、Trainをfloatした直後のEvalがRightのままで失敗。既存のOnUpdateTrainStatusで最大化の外だけ占有規則を当て、変化時のUpdateを既存CallAfterへまとめる最小修正を入れた。
- 文書160・020の日英と検証READMEを改訂仕様へ同期。空白を除いて受入9の初版固有文を検索し、5ファイルで0件。差分空白と保存後のハッシュを確認した。
- TDD2 GREEN: `.scratch/prd110/revision-green-train-float/` の新Train-float caseと既存往復caseはともに終了0。次はTrainをLeftの非定位置へdockした後の占有維持と、EvalのOFF/ONによる対称な位置復元を検査する。
- TDD3 RED: `.scratch/prd110/revision-red-side-owner/` は側面へdockしたTrainがCentreへ戻ることで失敗。現在のCentreを保持し、表示前フックが表示対象を明示して元の占有者を復帰させる実装へ変更した。位置の記憶はMainPaneDockPositionの4値1組へ移行し、既存の幅記憶には触らない。
- TDD3の最初の修正後実行 `.scratch/prd110/revision-green-side-owner/` は、位置・幅・状態の復帰は通り、Resetの1件だけ失敗。テストをboth起動にしていたのにReset後の既定eval配置を期待していた前提ミスだった。PRD受入2のとおり、既定eval起動からTrainをONにするsetupへ直して再実行する。
- TDD3 GREEN: `.scratch/prd110/revision-green-side-owner-2/` は側面位置・Train float・従来往復・最大化の4processとも終了0。次は直接操作の両Centreを既存保険caseへ追加し、通常の表示復帰と区別した修復・WARN1件を確認する。
- TDD4 RED: `.scratch/prd110/revision-red-double-centre/` は直接操作の両CentreでEvalがRightへ戻らず、指定WARNも無いことで失敗。通常ポリシーの最後でEvalを記憶位置か定位置へ降格し、指定WARNを1行出す保険を追加した。修復前の幅往復は行わない。
- TDD4 GREEN: `.scratch/prd110/revision-green-double-centre/` は保険・側面位置・従来往復の3processとも終了0。両保険のWARNは発動ごとに1件で、再適用では増えない。次は単独Trainの幅とTrainを含む側面列を検査する。
- TDD5 RED: `.scratch/prd110/revision-red-train-columns/` は単独Trainのとき補助列が512のまま縮まず、必要な462以下にならないことで失敗。単独Centreの幅制御をTrainにも適用し、Trainが側面にいる場合は同じ列の補助paneも含めて縮小対象から除く最小修正を入れた。
- TDD5 GREEN: `.scratch/prd110/revision-green-train-columns/` はTrainの列幅・既存列幅・側面位置・往復の4processが終了0。通常14caseへ両保険WARN0件の確認を揃え、補助pane最大化中の元占有者の再表示も既存maximize caseへ追加した。
- 追加自己点検: ShowEvalPaneIfHiddenは従来pane.Showを直接呼んでいた。改訂後はEvalが元の占有者の場合に両Centreの保険と取り違えるため、G5のStep/Resume導線も表示前フックへ接続する。公開Run ControlのStep/Eval操作のテストから進める。
- TDD6 RED: `.scratch/prd110/revision-red-eval-show/` は公開Step/Resume操作で元のEvalがCentreを取り戻せず、通常操作に両Centreの保険WARNが出ることで失敗。ShowEvalPaneIfHiddenでOnPaneShowingを先に呼ぶ最小修正を入れた。
- 最終GREEN: Debug全体・GUI targetのbuild成功。`.scratch/prd110/revision-gui/` の新規16case＋既存3caseの19processは全終了0。TDD6のStep/Resume、補助pane最大化中の再表示、通常15caseの両保険WARN0件も成功した。保険caseは両警告それぞれ発動1件、再適用で増えないことを確認した。
- 既定suiteの前後比較も成功。変更前core失敗0・GUI失敗0に対し、変更後 `.scratch/prd110/revision-after-suites/` はcore失敗0（570.04秒）・GUI失敗0（17.05秒）。変更後のGUI既定suiteと19processの実行体SHA256は一致し、core実行体は変更前後で同じだった。
- 最終機械点検: 既存6ファイルのBOM保持、変更9ファイルのLF、差分空白、ヘッダusing namespace、今回productionのdynamic_cast/TODO、共有契約の非依存、旧位置記憶名、受入9の旧文言を確認し、違反0件。index SHA256は開始時の値と一致した。今回の8ファイルの改訂差分は未ステージのままで、新規21implもステージしていない。
- TDD7 RED: `.scratch/prd110/revision-red-redundant-show/` は公開メニューの冗長ONでTrainを誤降格し、指定WARNも出ないため3アサーション失敗。表示前フックの対象が実際に非表示だった場合だけ正当な復帰とする最小修正を入れた。
- TDD7 GREENと最終回帰: Debug全体・GUI targetの再build成功。`.scratch/prd110/revision_review_validation.py` により `revision-review-gui/` の新規17case＋既存3caseの20processが全終了0。通常15caseは両保険WARN0、保険caseは各WARNが発動1件、冗長ON caseもTrainを残してEvalを降格しWARN1件となった。最大化とStep/Resumeの通常復帰も成功した。
- レビュー修正後の既定GUIは `revision-review-suites/` に保存し、28case・失敗0（15.90秒）。20processと同一の実行体SHA256で、変更前の空の失敗集合と一致した。coreは改訂後の1069case・失敗0の採取から実行体SHA256が変わっていないことを照合し、570.04秒の実結果を再利用した。結果のsuites.jsonへ元記録の参照を明記した。GUIだけの修正でcoreを再実行していない。
- レビュー修正後も9ファイルのLFと差分空白に違反はなく、index SHA256とHEADは開始時から不変。修正前の証跡は上書きせず保持した。

## 自己点検

- 全品質確認観点を適用範囲ごとに走査した。D1〜D3・D5・D13は実GUIの公開導線・AUI状態で確認し、対称な位置復帰・最大化・幅・timer・設定既定・選択枠の回帰も20processで成功した。レビュー指摘の冗長ONは真の再表示と区別し、直接操作の両Centreを保険へ渡すことも確認した。
- 公開API・設定キー・既定値は追加せず、初版のCanHidePaneとtimer制御を利用する。位置記憶の旧名は現用コードから除去し、互換層・別の占有者state・幅の別stashを追加していない。表示対象は私有関数の一時的な借用引数で渡し、最大化の一時非表示と区別する。
- Frameの既存機能グループの私有関数・型に収めた。200ms更新は2paneの状態を読むだけで、幅の往復・新timer・新ロック・device転送を足していない。幅の既存計測範囲を維持し、Updateは実変更時だけ予約する。
- コメントは日本語、追加WARNは指定の英語で発動ごと1行。通常のStep/Resume表示を保険へ流さず、入力・起動・学習・observerの契約を変更しない。
- 合成設定のGUIテストを使い、出荷設定の全値固定やtest-only本体APIは追加しない。日英の設計・ガイドとREADMEは改訂仕様に同期し、旧文言は空白違いも含めて5ファイルで0件。ステージ済み初版・他者の変更・PRD・ADR・CONTEXT・wikiは保持した。
- GetScalar、Agentの所有権、設定ファイル、skill、公開ヘッダ、ヘッダオンリー追加は今回の改訂では変更なし。未決の仕様判断は0。製品Runner起動・実機受入8は人の承認がないため実施せず、RelWithDebInfoも更新していない。

## レビュー

- 初回: 計画OK、実装は1件差し戻し。原文は `.scratch/prd110/review-20261101-0328.md`。表示中のEvalへ冗長なONイベントが届くと、直接操作で作られた両Centreを正当な復帰と取り違え、Trainを降格して保険WARNを出していなかった。D1の隠れていた占有者の復帰とD13の保険を区別するテストから修正する。
- 2回目: 計画・実装ともOK、修正要求0件、範囲外の検出0件、開発者への確認事項0件。原文は `.scratch/prd110/review-20261101-0349.md`。冗長ONの修正、公開メニューのRED3失敗からGREEN、20processのJUnitと既定GUIとの実行体一致、coreのハッシュ一致による結果再利用、Step/Resume・最大化の通常復帰を再確認した。根拠がなく見送った指摘はない。
- レビュー後は本書とREADMEに判定の記録だけを追記し、検証済みコードは変更していない。受入9の旧文言0件と既存6ファイルのBOM保持も最終確認した。

## 完了報告

- 実施内容: 改訂PRDのFloatable・Centre占有維持と対称な位置復帰、200ms更新、両Centreの保険、単独Trainの幅と側面列の除外を実装した。Step/Resumeと冗長ONも公開導線のテストで確認し、日英の設計・ガイドを同期した。改訂分の実装・Debug検証・自己点検・独立レビューは終了。PRD全体は実機受入待ちである。
- 変更ファイル: RunnerFrame.cpp/.hpp、RunnerFrame_test.cpp、設計160の日英、ガイド020の日英、PRD110検証READMEの8ファイルと新規の本21impl。既存の20impl、PRD・ADR0066・CONTEXT・wikiと無関係な作業は保持した。
- レビュー結果: 2往復、最終は計画・実装ともOK。初回の冗長ONに関する1件を修正して再確認した。レビュー原文・全コマンド・RED/GREEN・未実施の理由は上記へ記録した。
- 決まらなかったこと: なし。新たな起票候補なし。人が担当する実機受入8は未実施として明記している。
- 判断してほしいこと: なし。製品Runnerは起動せず、RelWithDebInfoは更新していない。git add・commit・pushは行わず、今回の追加差分は未ステージである。
- 次にできること:
  1. 人がAtari5・qbert・onlineでREADMEの実機受入手順を実施し、結果を本書へ記録する。

## 実機受入で判明した不具合（2026-11-01 08:03）

- ユーザー報告: TrainとEvalの左右表示でTrainをアンドックできず、従来どおり固定されているように見える。意図した動作はTrainのアンドック後にEvalがCentreへ昇格すること。
- 実行体の確認: 起動中のPID47524は `apps/runner/bin/Release/AnetRLRunner.exe`、起動07:57:12、実行体更新07:51:43。Releaseのバイナリに改訂で追加した両Centreの診断文字列が含まれることを確認した。未更新のRelWithDebInfoを使っただけの事象ではない。
- 原因: `C:/dev/wxWidgets-3.3.3/src/aui/framemanager.cpp:4767` のOnLeftDownはcaptionのdockがCentreならその場でreturnし、actionClickCaptionへ進まない。その後のOnMotion（5164〜）のFloatable判定へ到達しないため、Floatable(true)だけでは実際のcaptionドラッグは始まらない。
- 上流の誤り: PRD110 §1.1の6とD2の理由はOnMotionだけを見て「Centreの判定は無い」「Floatableの1つで足りる」としている。これは前段のOnLeftDownと食い違う。Train固有の制限ではなく、Centreの占有者全般に当たる。求める利用者向けの動作自体は変わらない。
- 検証の不足: 既存のTrain-float等は公開pane APIのFloat()から昇格を検査しており、caption押下からドラッグ開始までを通していない。20process成功・独立レビューOKを実ドラッグ成立の根拠にはできない。ユーザー報告と実コードの入口条件は一致しているが、今回新たにGUI入力の再現テストやbuildは実行していない。
- 必要な対応: PRDの誤った技術的前提を訂正し、Centreでも許可条件を満たせばcaptionドラッグを開始できる経路を設計・実装する。実際のcaption入力でRED→GREENを取り、最後の主paneの移動禁止と、相方の昇格・側面へのdrop・maximizeも回帰する。
- 08:03停止: implement-prd-with-docsが従うプロセス契約5節9の、上流の誤りを見つけた時点で止める規定に従い、PRDの訂正を含む対応の扱いをユーザーへ返す。課題はまだ起票せず候補とする。今回の変更は本書の調査記録だけで、実装・PRD・wxWidgets・実行中のRunner・Git indexは変更していない。

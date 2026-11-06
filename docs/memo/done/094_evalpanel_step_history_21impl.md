# PRD094 表示・操作調整 実装メモ

## 概要

2026-10-22改訂の094_evalpanel_step_history_10prd.mdを実装する。既存20implを保存し、最新のLiveコマ戻し・送りとメインStepアイコンも対象にする。PRD097、学習・診断値の計算、再生速度の段階、履歴保持の契約は変更しない。基準コミットと既存差分・対象ファイルは `.scratch/prd094/ui-impl-20261022/baseline/` に保存する。

## 未決事項監査

- ユーザー判断が必要なブロッカー: 0。
- repo evidenceで解決済み: EvalTracePanelがwxGridと操作UI、TracePlayerが履歴選択と再生状態を所有する。最新PRD D10〜D13・受入9〜12を今回の変更境界とする。既存通知の時点テストを再実行する。
- 2ximplへ固定: 幅は新規行の列と遅延確定列だけを計測する。テーマ・フォント・DPI変更時だけ保持行を再計測できる。幅の操作はUIスレッドで行う。システム色への強調色の混合は薄くし、選択色はwxGridへ委ねる。
- PRD範囲外: 既存の汎用GUIのdynamic_cast、既存Agent等の関連課題、PRD097の性能測定。今回の変更に混ぜない。

## 主な変更

1. TracePlayerのLive時のStepBackは最新の1つ前、1件ならその行でstartとする。StepForwardは最新行でendを返す。それ以外の再生契約を維持する。
2. EvalTracePanelの操作をwxAuiToolBarの6個のアイコン（Play/Pause、Stop、Step back、Step forward、Slower、Faster）へ移す。Play/Pauseの状態・英語tooltip・無効化を同期する。操作バー全体を囲む2 DIPの赤枠の領域を常に予約し、Traceの切替で配置を動かさない。
3. メインツールバーとTraceのStepアイコンを同じ縦棒＋右三角へそろえる。共通の描画が必要なら既存gui機能群にPlaybackIconとbitmap生成を置く。2つのUIから利用するための最小の公開GUI口であり、Agentの意味を持たせない。新しい単機能グループは作らない。
4. 列幅は見出し＋余白から開始し、新規行C列と遅延確定K列だけを調べて必要な列を拡張する。古い行の削除で縮めない。手動調整列を自動変更せず、閉じて開くと初期化する。計測範囲と日本語段階コメントを付ける。
5. 通常・見出し・文字・罫線・選択をシステム色に合わせる。報酬は薄い緑、終端は薄い橙を通常背景へ重ね、終端を優先する。テーマ・フォント・DPIの変更に追従する。色付けの設定は増やさない。
6. common.txtとAtari.txtの列4・5へlabel:Q_sa、label:V_sだけを追加する。
7. 日本語の設計160・ガイド020を最新実装に同期する。対象ファイルの既存の無関係な編集を保持する。wikiに食い違いがあれば正本に合わせ、logへ記録する。

8. 現用の設定比較器の既定列ラベル期待値を同期し、検証資材READMEへ今回のUI検証経路を追記する。全キー比較はConfigManagerを使い、変更前の設定の写しと現設定を比較して2ラベル以外の差がないことを確認する。

## テスト

- Public interface / surface: TracePlayerの公開操作・選択状態、列設定の解析、Runnerの通知、実wxGridとtoolbarの公開wx API・通常イベントを通した表示と操作。
- 最初の縦スライス: 複数行の履歴を渡し、LiveからStepBackして1つ前の行へ入り、停止・送りで最新行へ入る公開経路を1テストでRED→GREENにする。
- 続くbehavior: 1件・0件・既存再生境界、6操作とtoggle、幅の拡張・遅延値・手動幅・開き直し、配色・赤枠・アイコン。1つのbehaviorごとにテスト→RED確認→最小実装→GREEN→整理を行う。
- UI検証のためだけの本体friendやtest-only APIは追加しない。必要ならRunner側に実wxWidgetのテスト実行体を追加し、production source収集からtestを除外する。既存core GUIテストも回帰する。

## 検証

- C++はAGENTS.mdのVsDevCmd.bat経由でDebug全体ビルドと関連テスト、隠し以外のcore全テスト、Release Runnerビルドを行う。構成・終了コード・ログは `.scratch/prd094/ui-impl-20261022/` へ保存する。
- 稼働中のRunnerの実行体・DLLを保護するため、CMakeのproject includeで実行体の出力先を今回のscratch/binへ分離し、終了時に注入したcache設定を外す。通常binのハッシュを前後確認する。
- 列の設定差分がlabel追加のみであること、時点と再生の既存テスト、再生・停止・slider・Live復帰・開き直しを確認する。
- 幅処理は実wxGridでC=12、新規行と遅延確定K=2を測定する。PRD D10の2.5%基準を超える場合はUIだけの試作で是正してから採用する。UI変更の回帰に必要な範囲を確認し、初期実装の大規模学習Runは繰り返さない。
- 検証・自己点検のあと独立したan-reviewへimpl-planとimplを依頼する。同じレビュー担当へ修正をまとめて再依頼し、応答原文と結果を保存する。

## 前提

- 最新のユーザー回答「Pauseした。検証も実行。」により負荷のある検証も実行可。既存2本のRunnerは停止・再起動しない。
- コミット・stage・pushは実行しない。
- 既存20impl、PRD本文の追加調整、他作業のdirty変更、過去artifactを保持する。
- 実測・目視未実施は未実施と記録し、未確認を合格扱いにしない。

## 実施記録

- 2026-10-22: 計画保存。実装・検証はこれから。

## 実装・検証結果（2026-10-22）

### 実装した内容

- Liveのコマ戻し・コマ送りを最新PRDへ移行した。GUIは6個のwxAuiToolBar操作とPlay/Pauseトグルへ変更した。メインStepとTraceの送りは共通のPlaybackIcon::StepForwardを使う。
- 列幅は新規行C列と遅延確定K列だけを測り、実際に幅が変わる場合だけGridの寸法を更新する。手動列・削除後の幅・開き直し・フォント変更を検証した。SetFontは既存wx APIのoverrideであり、DPIによる子windowの自動拡大後へ再計測を延期して二重拡大を避ける。
- システム配色へ移行し、標準背景に16%の緑／橙を重ねる。選択はnativeの配色を優先する。操作バーは常時3 DIPの余白を持ち、Trace中だけ2 DIPの赤枠を描く。
- 設定変更はcommon.txtの2ラベルだけ。Atari.txtは共通設定を継承し、追加の上書きは不要だった。値・取得キー・番号・順は変更していない。現用比較器の期待値も同期した。
- 設計160・ガイド020・検証READMEを同期した。wikiの概要・成立の経緯を照合し、今回の正本と食い違いはなかった。PRD094本文・20impl・PRD097は変更していない。

### TDDと回帰

| 検証 | 結果・記録 |
|---|---|
| Liveの公開操作の縦スライス | 旧版で選択行とend通知の2項目が失敗（red-live-test.log）、最小修正後にGREEN。0件・1件の追加回帰も成功 |
| 実wx操作列 | 旧版にtoolbarが存在しないREDを確認。修正後6操作・tooltip・無効化・bitmapの26項目が成功 |
| 列幅 | 一律95pxで初期幅のRED（red-width-test.log）を確認し、見出し幅・長い値・非縮小をGREENへ移行 |
| 配色 | 罫線がシステム色に一致しないRED（red-palette-test.log）を確認し、修正後GREEN。手動幅・遅延値・toggle等は追加回帰として確認 |
| 最終GUI通常テスト | Debug/Releaseのlight/darkそれぞれ7件・68項目成功。native-manifest/final-verification/final-*-suite.log |
| 最終GUI描画テスト | 同4条件で各1件・38項目成功。12列の実画面PNG、赤枠・再生/停止/一時停止/終了・slider・Esc・開き直し、狭幅での列幅保持・横スクロールを確認。native-manifest/final-verification/final-*-visual.log、同Debug-visual/・Release-visual/ |
| coreのPRD094回帰 | 25件・1793項目成功。core-prd094-final.log |
| 設定の公開解決・比較 | ConfigManagerで6環境・5834キーの文字列値を比較し、2ラベルだけが差分（5860項目成功）。config-v2-test.log、config-equivalence.json |
| 設定規約・Python | check_default_leaves.py: audited 3474、errors 0。比較器のpy_compile成功 |
| Debug全体・Release | VsDevCmd経由で成功。debug-final-all.log、debug-final-ui.log、release-final-ui.log |

実行したC++コマンドは、VsDevCmd.batをcallしてから`cmake --build --preset x64-Debug --parallel 2`、同Debugの`--target AnetRLRunner-gui-test`、Releaseの`--target AnetRLRunner-gui-test`。configureでは今回のisolate.cmakeをCMAKE_PROJECT_anet-lab_INCLUDEへ注入した。GUIテスト実行体はscratch/bin/{Debug,Release}/AnetRLRunner-gui-test.exe。通常テストは`[prd094]~[.]`、描画は`[trace_ui_visual]`、計測は`[trace_ui_perf]`を明示し、`--reporter compact`で実行した。設定比較は`[trace_ui_config]`を使った。最終実行引数と終了コードはnative-manifest/final-verification/final-tests.jsonに保存した。

テストのbitmap確認がwindowを渡さずwxのDebug Alertを出したため、テストをGetBitmapFor(toolbar)へ訂正し、以後はwx assertを例外として扱って自動実行を止めないようにした。wxログtargetの寿命と描画テストのtimer配送も検証側で修正した。失敗した初回ログは保持し、最終成功と区別する。

### core全体の既存失敗（修正前の記録）

この失敗は、末尾の追加対応でfixtureを修正し、全体再実行の成功を確認して解消した。以下は修正前の検証履歴である。

`anet-core-test.exe --reporter compact`は741件中739件成功・1件失敗・1件skip、150009項目中150008成功（core-all.log）。失敗はepisode_end_test.cppの「EvalRunner forced action keeps derived action-info scalars」。開始前からあるtrainer.cppの強制行動の範囲検査に、行動数1のfixtureが行動5を渡している。trainer.cppは開始時の保存版とSHA256が一致（501610880346F0024125802AC0C396F96A9CEA4A10F05E2DE39D81E4E9E0BCD8）し、今回の変更対象外なので修正していない。全体テストの成功とは扱わない。skipはCUDAが利用可能なための、CUDA不在時のエラー検査のskipである。

### UI更新処理の測定

実wxGrid・StepTraceBuilder・Notifierを通し、C=12、K=2、保持1000行で1000件を採取した。Notify内の診断・表更新を含む時間であり、幅処理だけの時間より広い範囲である。

| 構成 | 中央値 | 95%点 | PRDの2.5%予算（14.5 step/s） |
|---|---:|---:|---:|
| Debug | 0.5727 ms | 1.4473 ms | 1.72414 ms |
| Release | 0.1996 ms | 0.8280 ms | 1.72414 ms |

最終ログはnative-manifest/final-verification/final-perf-Debug.log、同final-perf-Release.log。以前の測定ログは元の保存先へ保持した。これは合成通知によるUI単独の測定であり、実AtariのNN・View描画を含む実速度や学習速度の測定値ではない。初期実装の性能・メモリの未確認を、この値で合格へ変更しない。

### 未実施・未確認

- 実OSのテーマを切り替えたままの確認、実モニタ間のDPI移動は未実施。wxMSW 3.3.3のSetAppearanceは既存windowがあるとCannotChangeを返すため、ライト／ダークを別processで確認し、表示中の更新はsystem-colour通知経路で検証した。OS全体の設定は変更していない。
- 元の受入8の実Atariの1分間のEval速度、学習速度の1分×3の基準、1000件の実Viewを持つメモリ増分は今回再測定していない。今回のUI回帰は受入12が指定する操作範囲と、上記UI費用の測定である。20implの未確認事項は残る。
- core全体の既存失敗1件は、後続のfixture修正と全体再実行で解消した（末尾の追加対応を参照）。

### 自己点検

docs/quality/impl.mdの全項目を適用範囲で確認した。詳細はscratchのself-check.md。変更14ファイルはBOM保持・LF・diff checkが成功。新しいproduction dynamic_cast、ヘッダusing namespace、大文字TODOは0件。gui.cppの既存dynamic_cast 3件は20implで範囲外として記録されたまま、追加・変更していない。GetScalar・Agent所有権・skillの変更はなく、設定・共有GUI・コメント・計測・公開口・検証資材の同期を確認した。

通常RunnerのDebug/Release実行体は開始前と同じSHA256。開始時に既存dirtyだった対象外ファイル14件は同一で、作業中に別途更新された095とwiki/logは最新を保持した。PRD094本文も開始時と同一。stage・commit・pushは行っていない。

### 独立レビュー

独立したan-reviewへimpl-planとimplを一括依頼し、読取りレビュー中。途中指摘としてガイド020のFPS=0時の記述がPlay/Pauseトグルと不整合と分かり、再生中はPauseが有効である説明へ修正した。最終判定は計画・実装ともOK（差し戻し0）。正式依頼は1往復で、途中指摘の修正を同じ担当が確認した。応答原文: `.scratch/prd094/review-20261022-2125.md`。

- 追加自己点検: GUIテストへ本体と同じWin32 manifest/resource（PerMonitorV2・Common Controls v6）を付けた。mt.exeで埋込み内容を確認（native-manifest/manifest-checks.json）。Debug/Releaseを再ビルドし、ライト／ダーク双方で上記の最終GUI検証がすべて成功した（native-final2-*-build.log）。
- 速度のFaster/Slowerも実toolイベントで確認した。狭幅テストで最初はwxGrid本体のGetVirtualSizeを使って1項目が失敗したが、wxGridは子のgrid windowをscroll targetとするため、GetGridWindow()->GetVirtualSizeへ検査を訂正した。最終結果は4条件とも38項目成功。最初の失敗ログ・PNGはnative-manifest/直下へ保持し、検証コードの誤りと本体のREDを区別した。

### 検証後の後片付け

Debug/Releaseとも`VsDevCmd.bat`経由の`cmake --preset x64-Debug -U CMAKE_PROJECT_anet-lab_INCLUDE`（Releaseも同様）で今回注入した出力分離を解除した。再ビルドは行わず、通常RunnerのDebug/Release実行体が開始前と同じSHA256であることを確認した（cleanup-checks.json、cleanup-x64-*.log）。隔離した新しい実行体・DLL・PNG・失敗を含む各世代のログはscratchへ保持した。

最終静的再監査はBOM差0・CRLF0・追加規約違反0・diff check終了0。設定5834キーも2ラベル以外が一致し、対象外baseline14件は同一、並行更新された095とwiki/logは最新を保持した。別作業で追加されたAGENTS/skill/wiki等の変更にも触れていない。

### UI調整時点の独立レビュー記録・残作業

独立an-reviewの全62観点による最終判定はimpl-plan/implともOK。不足点・修正要求・新しい範囲外検出はなし。途中のガイド020 FPS=0指摘は修正済みとして確認された。適用外の所有権・skill等の節は原文の再確認ポイントに件数を記録している。根拠がなく見送った指摘はない。

初回レビュー時点の再確認ポイントは、実OSテーマ切替・モニタ間DPI移動、元の受入8の実Atari速度・学習速度・メモリが未確認であることと、core全体の既存1失敗だった。実機確認は未確認のまま保持し、coreの1失敗だけを末尾の追加対応で解消した。UIレビューのOKに実機受入の合格は含めない。

初回の工程末に提示した起票候補は次の2件。後続のユーザー指示により起票は行わず、2だけをこの場で修正することになった（末尾の追加対応）。課題ファイルやPRDの未決表への追加は行わない。

1. PRD094の残る実機受入（OSテーマ/DPI、元の受入8の実Atari速度・学習速度・メモリ）。20implからの保留と今回の実機未確認をまとめた候補。質問していない（独立レビューの確認事項として最終提示）。
2. 強制行動の範囲検査とepisode_end_test.cppのfixtureの不整合。今回開始前からの実装・テストで1件失敗が残る。質問していない（範囲外のため未修正）。

今回のUI実装・自動回帰・独立レビュー・検証後の後片付けは終了。PRD全受入の完了とは報告しない。stage・commit・pushは実行していない。

## 追加対応: 強制行動テストのfixture不整合（2026-10-22）

ユーザー指示「起票無し。#2はこの場で修正」と承認済み計画に従い、課題ファイルを作らず本テストを修正する。候補1の実機確認は未確認のまま保持する。

### 保存した追加計画

- TestBatchEnvのコンストラクタへEnvSpecを値で渡せる省略可能引数を追加し、既定はMakeEnvSpec()の1行動とする。受けた仕様をfixture内へ所有し、GetSpecで返す。他のfixture利用側は変更しない。
- 該当テストだけnoop/forcedの2行動を宣言し、通常Actorの0に対してDoStep(1)を実行する。環境が1を受け取り、派生scalar 12.5が通知されることを確認する。
- 修正前の範囲外失敗を再確認し、Debugのanet-core-testをVsDevCmd経由でビルド。該当テスト、関連episode_end/eval_runner/action_infoと、隠し以外のcore全体を再実行する。不正な強制行動の副作用前拒否の既存テストも維持する。
- 同じ独立an-reviewへ追加差分と検証結果を渡す。結果は本節、ログは.scratch/prd094/forced-action-fixture-20261022/へ保存する。
- productionのRunner・公開API・範囲検査は変更しない。無関係な変更と既存Runを保持し、stage・commit・push・起票は行わない。

### 追加対応の検証結果

- RED: 修正前の通常Debug実行体で該当テストが行動5・期待範囲[0,0]の例外により失敗（red.log、exit 42）。
- GREEN: fixtureだけを修正し、VsDevCmd.bat経由で`cmake --build --preset x64-Debug --parallel 2 --target anet-core-test`成功（debug-build.log）。該当テスト1件3項目、関連`[episode_end],[eval_runner],[action_info]`の18件647項目が成功。範囲外の強制行動の副作用前拒否も含む。
- 隠し以外の全体テストは終了0、741件中740件成功・既存1件skip、150011項目すべて成功。所要420.564秒。今回のfixture不整合による1失敗は解消した。コマンド・実行体hash・終了コードはforced-action-fixture-20261022/results.jsonへ保存。
- 追加差分の全品質観点による自己点検を実施（同self-check.md/json）。production・通常Runnerのハッシュを今回の開始時と照合した。

### 追加対応の最終検証

Debugビルド、単独GREEN、関連18件647項目、隠し以外のcore全体がすべて成功した。テスト実行体SHA256はcf8642b1ca0a45ce0f233ed00f83b60fc03dd183cb6eaddc13cd8f9a7a322adb。fullの741件中1件skipは既存の条件付きskipであり、失敗0である。以前のcore-all.logは修正前の記録として保持する。

TestBatchEnvの既定1行動は維持し、該当ケースだけ2行動・強制1へ変更した。Envへの到達action1と派生scalar12.5をpublic経路で検証した。productionの範囲検査を緩和せず、既存の不正な強制行動の副作用前拒否の回帰も成功した。

同じ独立an-reviewの追加レビュー1往復で、計画・実装ともOK（差し戻し0）。途中指摘の旧「1失敗が残る」記述は、修正前の履歴と現在の解消済み状態を区別するよう修正し確認された。不足点・新規の範囲外検出・開発者への確認事項はなし。応答原文は`.scratch/prd094/review-20261022-2149.md`。起票はユーザー指示により行わず、候補1の実機確認は未確認のまま保持する。追加対応は完了し、stage・commit・pushは行っていない。

## 追加対応: GUIテストと再生アイコンの指摘4点（2026-10-22）

ユーザー指摘に従い、次の局所修正を行う。起票・stage・commit・pushは行わず、既存の実機未確認事項は維持する。

- GUIテストのMetricsLoggerと画像の既定出力先をOSの一時ディレクトリ/anet-eval-trace-<PID>/配下へ移す。画像の明示的なPRD094_UI_CAPTURE_DIR指定は維持し、実際の保存先をテストの標準出力へ表示する。カレントディレクトリ配下へ作業時の日付フォルダを作らない。
- 変更時だけの設定snapshot比較であるtrace_ui_configを削除し、不要になるincludeとREADMEの実行説明を除く。保存済み比較結果は当時の検証記録として保持する。
- Runnerのsource収集から除く条件を_test.cpp接尾辞へ限定し、Backtest.cppのようなproduction候補を除外しない。
- Train/Evalの生成・状態変更・テーマ反映も既存MakePlaybackBitmapへ統一する。自前のPlay/Pause SVG、使われなくなるMakeSystemTextSvg、残った余分な空行を削除する。
- Debug/Releaseを既存Runnerとは別の出力先へビルドし、ライト/ダークの通常GUIと描画を別のcwdから確認する。既定の画像出力と明示指定の双方、CMakeの除外対象、coreのPRD094関連回帰を確認する。内部表現の整理は既存GREENから進め、REDを作ったことにはしない。
- 自己点検後、同じ独立レビュー担当へ追加差分・結果を渡す。ログは.scratch/prd094/ui-review-fixes-20261022/、結果は本節へ記録する。出力分離のCMake注入は終了時に解除する。

### Release検証で見つかったPCH/PDB配置の補正

Debugの通常GUI7件68項目・描画1件38項目は両テーマで成功し、core PRD094の25件1793項目も成功した。旧版が別cwdへ作業フォルダを作ることと、修正後は作らずtempへ出力することをbefore/after-path-check.jsonで確認した。

Releaseの初回buildはC2859で失敗した。生成Ninjaを確認すると、PCH所有者のコンパイルPDBはout/build/x64-Release/apps/runner/AnetRLRunner.dir/AnetRLRunner.pdbだが、GUI再利用先の/Fdは実行体隣のリンクPDBを指していた。単なるソースのコンパイルエラーではなく、元のGUI targetのPCH再利用設定が既定のPDB名・配置へ依存する問題だった。CMakeのインストール済みpropertyヘルプで両PDBの区別を確認し、PCH所有者のCOMPILE_PDB_NAMEとCOMPILE_PDB_OUTPUT_DIRECTORYを明示する補正を同じCMakeへ追加した。

初回Releaseの終了コード確認前にGUI実行を試みたため、未生成exeへの起動は失敗した。成功扱いせず、修正後のビルド完了を確認してから再実行する。初回ログは保存し、補正後の結果は別ログへ分離する。

PDB名・配置の明示だけでは、再configure後もGUI再利用先がリンクPDBを指す状態が変わらなかったため、その明示設定は撤去した。GUIテストは本体と同じpch.hppから専用のPCHを作る形へ変更し、コンパイルPDBをtargetごとに独立させる。本体のDLL配置への依存はadd_dependenciesで維持する。productionのPCH・コンパイル設定は変更せず、テストtargetだけでC2859を解消する。補正後の生成Ninjaとビルド・回帰で確認する。

### 指摘4点の最終検証結果

- 4点をすべて修正。GUIテストの既定出力はOSの一時ディレクトリ/anet-eval-trace-<PID>/{metrics,captures}。明示capture先は維持し、保存先をstdoutへ表示する。READMEも同期した。one-off configテストと専用includeを除き、登録一覧でも消えたことを確認した。
- CMakeの実filter文を実行し、EvalTracePanel_test.cppだけが除外され、Backtest.cpp/RunnerFrame.cppは残ることを確認した。Train/Evalの初期生成・状態変更・テーマ更新はMakePlaybackBitmapへそろえ、旧SVG定数・専用helperと指定された余分な空行を削除した。
- ReleaseのC2859はGUIテスト専用PCH/PDBへの分離で解消した。両構成の生成Ninjaを確認し、GUI自身のPCHとcompile PDBが本体と独立している（pch-layout.json）。Debug/ReleaseともVsDevCmd経由のbuild成功（debug-pch-result.json、release-pch-result.json）。通常の全体buildにGUI testを追加せず、DLL配置のための本体依存は維持する。
- 最終のDebug/Release×light/darkで、通常7件68項目・描画1件38項目が各成功。全processを別のtemp cwdから起動し、cwd下に.scratchを作らず、metricsをOS tempへ出すこと、画像の既定tempと明示指定の双方、保存先表示を確認した。final-gui-results.jsonにコマンド/cwd/pid/終了コード/出力先を保存。初回成功・失敗のログも上書きせず保持した。
- core PRD094は25件1793項目成功（final-core-prd094.log）。今回の修正はテスト出力・ビルド・SVG共通化なので、前回全体150011項目の成功は以前の結果として保持し、今回は全core再実行や性能反復Runを行っていない。実OSテーマ/DPI等の未確認を合格へ変更しない。
- 自己点検の最終差分はBOM/LF保持・diff check成功。通常実行体を保護した分離buildのcache復元は工程末に行い、独立レビューの最終結果とともに追記する。

### 指摘4点の後片付け

Debug/ReleaseのCMAKE_PROJECT_anet-lab_INCLUDEは開始時の状態へ復元済み（どちらも未設定）。再ビルドは行わず、通常Runnerの両実行体、trainer.cpp、前回修正したepisode_end_test.cppが今回の開始時と同じhashであることを確認した（cleanup-checks.json）。コミット・stage・push・起票は行っていない。同じ独立an-reviewの追加レビュー1往復で計画・実装ともOK（差し戻し0）。不足点・新しい範囲外検出・開発者への確認事項はなし。応答原文は`.scratch/prd094/review-20261022-2222.md`。今回の4点と検証に必要だったテスト専用PCH補正は完了し、未確認の実機受入は従来どおり保持する。

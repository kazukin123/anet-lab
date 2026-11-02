# PRD109 Runner 起動の非同期化 実装メモ

## 概要

訂正済み [PRD109](109_runner_startup_background_init_10prd.md) と、2026-10-31 にユーザーが承認した計画を正本とする。Frame 表示後の RunManager 構築だけを RunInitThread へ移し、構築中も描画とログ表示を行う。

## 主な変更

- 単発の std::thread と ProfileThreadName を使う。worker は設定をコピーして構築し、結果を専用領域へ置く。例外はすべて StoreCurrentException で保存する。CallAfter は通知だけを行い、main が join 後に一度だけ結果を受け取る。正常時だけ既存順序で後半を行う。
- RunnerApp に IsRunInitialized() const と PrepareRunClose(bool can_veto) を追加する。後者の true は Close 続行、false は veto を意味する。HeatMap は Update UI とハンドラで未初期化を抑止する。
- 拒否可能な Close は一度だけ記憶・ログ出力し、構築後に後半を省いて閉じる。Close(true) と初期化中の OnQueryEndSession は構築を待ってから閉じる。構築失敗は既存 fatal 経路で報告する。
- OS の終了確定通知は上書きせず、Close の道を足さない。OnExit は worker を待ち、未処理の構築失敗は表示・保存せず戻り値1とする。OnRun は未受領の結果に触れない。RUNNING でない RunManager では保存・学習 Runner の Shutdown を省く。
- LogPanel が先に消えた道ではログチェーンの UI 参照を切り離し、ファイル側へ保留ログを排出する。worker が終わる前にログと MetricsLogger を片付けない。
- 開始・完了・終了要求の INFO ログは PRD の文言とし、所要秒は steady_clock で小数点以下2桁にする。
- 設計文書160の起動図・終了処理、140の保存条件、020の Close 説明を追随する。固有の検証資材は core/anet-core/testdata/prd109 に置き、160のテスト章から参照する。

## テスト

- Public interface / surface: RunnerApp::OnInit / IsRunInitialized / PrepareRunClose / OnQueryEndSession / OnRun / OnExit、Frame の menu/close、LogPanel の公開 UI、Run log・stderr・終了コード・保存ファイル。
- 優先 behavior と TDD 順序: 1件ずつ RED→GREEN を行う。最初は SlowInitTestAgent（3秒）の構築中に100ms timerが10回以上動き、ログが完了前に現れ、最後に初期化完了となる tracer bullet。続いて構築中の pane・resize・HeatMap/Conv2d、終了要求の重複・OS問い合わせ・強制Close・構築失敗、RUNNINGでない状態の前後のClose、CallOnExitでのウィンドウ先行破棄と戻り値を通す。
- SlowInitTestAgent は gui-test だけで登録し、agent.construction_delay_ms（整数0以上、既定0）、agent.construction_fails（bool、既定false）を PRDどおり検証する。
- RunnerApp 起動ケースは隠しタグ [.][prd109_startup] と個別タグで1プロセス1ケース。既存 [eval_steps_menu] と [prd105_ui_keys] はイベントを処理しながら最大30秒初期化を待つ。
- 変更前後の既定スイートの失敗集合を一致させる。実 Runner の未知 eval schedule で終了コード1、Runログに診断あり、stderr.logに診断なしを確認する。
- GridMaze/LunarLander/qbert/DropMergeをseed=1、8lane、batch=16、warmup=256、replay ratio=1、8704experience、評価停止、FP32、決定的backendで前後各1本逐次実行する。90_perf/・91_resource/・92_resource/を除く全scalarのtag/step/値を完全一致で比較する。PRD098条件生成と既存正規化を再利用し、PRD105の記録は変えない。

## 検証

変更前に次のビルドと既定スイートを行い、exe・DLL・configとhash、失敗集合を .scratch/prd109 の新規ディレクトリへ保存する。同じ検証を実装後に繰り返す。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug && cmake --build --preset x64-Debug --target AnetRLRunner-gui-test'
```

```powershell
core/anet-core/bin/Debug/anet-core-test.exe
apps/runner/bin/Debug/AnetRLRunner-gui-test.exe
```

実機の Atari5 qbert の描画、移動・重ね合わせ、「応答なし」、ログ表示、終了要求、onlineエラーダイアログは人が確認する。未実施なら受入完了とは報告しない。

## 前提

- GUIテストの待機上限は、承認済み計画と改訂PRD §6の6に従い30秒にする。3秒の制御した遅延を使う起動試験と既存2ケースが対象で、外側のprocess上限60秒には通常の起動・後始末の余裕を含める。改訂前PRDの120秒との相違と当時の判断は、下記のレビュー履歴に残す。改訂PRDはユーザー側で修正済みであり、今回編集しない。
- ユーザー判断待ちのブロッカー0。PRD冒頭の2026-10-31訂正を確認済み。wxWidgetsは3.3.3。終了確定は CallOnExit→ウィンドウ削除→OnExit→戻り値でexitである。
- 製品設定、同期起動への切替、構築の取消、進捗表示、新しいADRと用語集項目は足さない。coreのThreadBaseは変更しない。
- 無関係な未コミット変更と過去のRun記録を保持する。コミット・pushは行わない。
- 自己点検後に独立した an-review へ計画と実装を依頼し、修正・再レビューを行う。必須検証、レビュー、未実施と理由を本書へ記録する。

## 経過

- 2026-10-31: 承認済み計画を保存。これ以降は本書に結果を積む。
- 変更前 Debug 全体・AnetRLRunner-gui-test ビルド成功。exe/DLL/config/hash を `.scratch/prd109/baseline-20261031/` へ保存した。
- tracer bullet の RED: `IsRunInitialized` が未実装のため GUI test のコンパイルが C2039 で失敗。最小の非同期構築と結果受領の実装へ進んだ（GREENは下記に記録）。
- スイート採取ドライバの初回は cwd を apps/runner にしていたため停止した（`.scratch/prd109/before-suites/`、比較基準にしない）。cwd を repo root に直し `.scratch/prd109/before-suites-repo/` へ採取し直す。coreのコード・バイナリは変更していない。
- tracer bullet GREEN（`[prd109_ready]`）、構築中UI RED→HeatMapのUpdate UI/handler追加→GREEN（`[prd109_ui]`）、終了要求 RED（PrepareRunClose未実装のC2039）→実装→GREEN（`.scratch/prd109/close.xml`）。
- OS問い合わせのREDは `.scratch/prd109/query-red.xml`。wx既定handlerが通常Closeへ渡し、初期化中のvetoでOS問い合わせもvetoされることを確認した。wx3.3.3のOnQueryEndSessionは非virtualなので、同名handlerを明示Bindする。終了確定のhandlerは変えない。
- 変更前coreの既定スイート成功（438.38秒、失敗集合0）。保存済み変更前GUIのJUnitは `.scratch/prd109/before-gui.xml`、既知失敗は `Logs indent wrapped lines to the message on demand` 1件。採取中に作った新GUI側では、出力先のprd109も起動タグと誤認してEnv登録を省くテスト入口の判定不足があり、`.scratch/prd109/before-suites-repo/gui.xml` は比較に使わない。判定を `[prd109_` のタグ表記に限定して修正した。前後比較には保存済み旧GUIのJUnitを明示する。
- Train Viewには現行UIの表示切替menuが無い（中央pane）。構築中の往復だけ公開wxAUI操作で検証し、新しいmenuは足さない。
- OS問い合わせGREEN（`.scratch/prd109/query.xml`）。強制Close失敗RED→既存reporterでその場で報告し保存を抑止→GREEN（`close-failure.xml`）。RUNNINGでない結果のClose/OnRun RED→Shutdownのstatus判定追加→GREEN（`incomplete-after.xml`）。
- CallOnExitのRED（`exit-red.xml`）で末尾の構築ログがRunログから欠けた。LogPanel削除後はchainの旧UI参照をdetachし、一時的にfile側だけへ切り替えて排出・closeする形でGREEN（`exit.xml`）。ウィンドウ先行削除後の構築失敗の戻り値と、受領前の不完全な構築のCloseも追加回帰の対象にした。
- 旧 `[eval_steps_menu]` は非同期化後にアクセス違反でRED（`.scratch/prd109/old-menu-red/`）。既存の2ケースへ30秒上限のイベント処理付き待機を追加した。
- Debug全体とgui-testビルド成功。`.scratch/prd109/gui-validation/` の新規10件＋既存2件は全件成功。
- 製品fatalの受入5は終了コード1・Runログ成功だがstderr.logに診断が無くRED（`.scratch/prd109/fatal-validation/`）。共通ReportErrorはwxLogへしか書かず、Frame表示後はstderr targetがLogPanelへ置き換わるため。当時の受入5の記載に従い ErrorDialog.cpp の既存ReportErrorの非モーダル分岐にstderr出力を追加した。新しいreporterや公開APIは作らなかった。この追加はPRD068のD5と矛盾すると判明し、改訂PRDに合わせて撤去する（下記「追加指摘への対応」）。

## 検証結果

- Debug全体・AnetRLRunner-gui-testビルド成功（VsDevCmd経由、2026-10-31）。
- `.venv/Scripts/python.exe core/anet-core/testdata/prd109/run_validation.py gui --output .scratch/prd109/gui-validation`: 新規10件と既存2件が全件成功。各JUnitとexecution.jsonを保存した。
- GREEN後の自己点検で、SlowInitTestAgentの設定をprivate Config DTO＋ANET_READ_CONFIGへ揃え、繰り返しの識別子とファイル名・環境変数・CLIを定数化した。RunnerApp_test.cppは既存RunnerApp機能グループに対応する起動試験として分け、EvalTracePanelの試験へ起動責務を混ぜない。`.scratch/prd109/gui-validation-final/` の12件は全件成功。`after-gui-final.xml` でも既定GUIの失敗集合が変更前と同じことを確認した。
- 同ドライバの `fatal --output .scratch/prd109/fatal-validation-fixed`: 当時の検証条件では成功。製品Runnerの終了コード1、Runログとstderr.logのUnknown run.eval_schedule tag 'nosuch'を確認した。改訂PRDの受入5はstderr非出力なので、この結果は現行条件の合格根拠に使わない。
- 同ドライバの `suites --output .scratch/prd109/after-suites --baseline .scratch/prd109/before-suites-repo --baseline-gui .scratch/prd109/before-gui.xml`: 成功。core失敗0（378.37秒）、GUIは変更前と同じインデント折り返し1ケースのみ。GUIのtests=380/failures=3/errors=0も前後同じで、ケース集合を完全一致で比較した。
- 同ドライバの `compare --baseline .scratch/prd109/baseline-20261031 --output .scratch/prd109/learning-comparison`: 成功。GridMaze 26,946点、LunarLander 24,399点、qbert 23,185点、DropMerge 31,152点、合計105,682点がtag・step・値とも完全一致。各実行条件・exe hash・config hash・正規化checksumは同ディレクトリへ保存した。
- 数値比較に使用した変更後exeは `.scratch/prd109/validated-after.exe`（SHA256 `febb213228562c49845afee073a569060fcc0cb5cc1fecd855e80cf29b015d59`）に保存した。この時点の比較後のproduction修正は段階コメントだけで、学習処理は同じだった。レビュー修正後もVsDevCmd経由のDebug全体・GUIビルドは成功し、`.scratch/prd109/gui-review-fixed/` のGUI12件が全件成功、`fatal-review-fixed/` も当時のexit1・Runログ・stderrの条件に成功した。当時のexeのSHA256は `6876799487eabafa6871b324a402ca796726c0256c5345ab595c2291b683a61a`。stderr追加の撤去後の検証は下記へ追記し、これらの過去の成果物は保持する。
- `python -m py_compile`相当をリポジトリ.venvで実行し成功。git diff --check、LF、BOM保持、productionのdynamic_castとTODOとヘッダusing namespaceの不在を確認した。
- 実機受入（PRD受入6）: 2026-10-31、ユーザーから「もろもろ実動作確認してoｋ」と報告を受領した。人による実機確認待ちは解消した。自動テストの結果とは分けて、ユーザーの確認結果として記録する。

## 自己点検

- 実装品質確認観点の全項目を適用範囲に従って点検。契約・公開口・英語ログ・結果の同期と一度だけの受領、3か所の待機、テスト対象面・TDD順序、文書追随を確認した。
- 設定キー置換、汎用層変更、skill、ADR、用語集、ホットパスの変更は無い。対応する観点は対象外。テスト用Agentには既知のscalar keyが無いためGetScalarはnullopt。検証の設定採取は等値期待値を固定する出荷設定テストではない。
- ユーザー判断待ちの設計未決はなし。既定スイート・学習比較・レビュー対応後の再検証に加え、修正後の再レビューとユーザーによる実機確認も完了した（末尾の「再レビューと実機確認の完了」）。

## レビュー結果

自己点検後、長い既定スイート・学習比較の進行と並行して独立したan-reviewへ初回依頼する。進行中の検証と人の実機確認未実施を明記し、結果がそろった後に本書を更新する。コード変更が必要なら関連検証を再実施して同じreviewerへ出し直す。

- 1往復目: impl-plan/implとも差し戻し。原文 `.scratch/prd109/review-20261031-1910.md`。待機上限の相違、include順、共有識別子の定数化、段階コメント、ユーザーガイドの本文と仕組みの分離を指摘された。30秒はユーザーの直近の承認済み計画を優先する根拠と正本の扱いを前提へ明記した。残りはコード・文書で修正した。範囲外の検出・判断待ちはなし。実機確認の保留は継続する。
- 2往復目: impl-plan/implともOK（2件）。原文 `.scratch/prd109/review-20261031-1920.md`。前回指摘への対応、待機上限の正本、全GUI/fatal/既定スイート/4環境全scalarの結果を確認された。範囲外の検出、根拠なく見送った指摘はなし。再確認ポイントは、承認済み計画の30秒の採用、既定GUIの失敗集合が同じこと、人の受入6はこのOKに含まれないことである。

## 完了報告

実施内容:

- RunManagerの構築だけをRunInitThreadへ移し、結果の一度だけの受領、初期化中のCloseとOS終了問い合わせ、OnExitの待機と終了値、未初期化UI、RUNNINGでない場合の保存・後始末を実装した。エラーの出力先は既存loggerの起動段階に従い、Frame表示後の構築失敗をLogPanelとRunログへ出す。
- Debug全体・GUIビルド、GUI12ケース、製品fatalの検証に成功。既定スイートの失敗集合は前後同じ。4環境の105,682点は全点完全一致。

変更ファイル（12件。ユーザー側のPRD改訂を除く）:

- apps/runner/CMakeLists.txt
- apps/runner/src/RunnerApp.cpp
- apps/runner/src/RunnerApp.hpp
- apps/runner/src/RunnerFrame.cpp
- apps/runner/src/EvalTracePanel_test.cpp
- apps/runner/src/RunnerApp_test.cpp
- core/anet-core/testdata/prd109/README.md
- core/anet-core/testdata/prd109/run_validation.py
- docs/design/160_applications_and_tools.jp.md
- docs/design/140_observability.jp.md
- docs/design/020_user_guide_run.jp.md
- docs/memo/109_runner_startup_background_init_20impl.md

レビュー結果: an-review 2往復でimpl-plan/implともOK。その後のcc-reviewの指摘を修正し、修正後の再レビューでもCodex・Claudeとも計画・実装がOK。原文は `.scratch/prd109/review-20261031-1951.md` と `.scratch/prd109/review-20261031-1951-claude.md`。見送った指摘なし。

決まらなかったこと: なし。起票候補なし。

判断してほしいこと: なし。修正後の再レビューは両モデルでOK、PRD受入6の実機確認もユーザーからOKの報告を受領した。既存の自動検証結果と合わせ、PRD109の受入完了とする。

次にできること: コミット対象を人が確認する。ユーザーの選択に従いステージは行っていない。コミット・pushも行っていない。

## cc-review後の文書修正（2026-10-31）

以下と「追加指摘への対応」は、それぞれの修正時点の履歴である。最新の状態は末尾の「再レビューと実機確認の完了」を参照する。

- cc-review原文: `.scratch/prd109/review-20261031-1931.md`（Codex: 計画・実装ともOK）、`.scratch/prd109/review-20261031-1931-claude.md`（Claude: 計画OK、実装の文書1件を差し戻し）。
- Claudeの指摘: 020 §4.1の初期化順序の一覧で第4項だけが利用者の操作になっており、背景でのRunManager構築の段が抜けていた。
- ユーザーの「指摘を修正。再レビューはまだ。」に従い、本文へログ確認・移動/resizeの操作案内を置き、初期化順序の6項目をNOTEへ移した。第4項に表示後の背景構築を戻し、第5項は構築後のメインスレッドでのパネル接続とした。Logsは表示時点で使えるため第5項から外し、160 §5.1への参照を追加した。
- 自己点検: 実装品質確認観点の全項目を適用範囲に従って点検し、020 §4.1・§6.4のNOTEと160 §5.1の順序、本文と仕組みの分離を確認した。文書のみの修正で、ビルド・テストは対象外。NOTEの6項目・リンク先の見出し・LF・BOM・git diff --checkの確認は成功した。
- 再レビューは未実施（ユーザー指示）。上記のOKは修正前の判定であり、今回の文書変更の再レビュー結果ではない。人の実機受入6も引き続き未実施。

## 追加指摘への対応（2026-10-31）

- ユーザーの「これも修正」により `.scratch/prd109/handoff-20261031-review-fixes.md` の残件を修正する。改訂PRDの§1.4の3・G4・受入5とPRD068のD5を確認した。Frame表示後の構築失敗はLogPanelとRunログへ出し、stderrへ直接追加しない。
- 手順: fatalドライバを「終了コード1、Runログに診断あり、stderr.logに診断なし」へ直してREDを確認し、ErrorDialog.cppの追加出力を撤去する。VsDevCmd経由のDebug全体・GUIテストのビルド後、新規出力先でfatalとGUI12ケースを検証し、本書とREADMEへ結果を記録する。
- 020の修正済み箇所と改訂PRDは編集しない。学習処理への変更は無いため4環境の比較は再実行しない。再レビューはユーザー指示に従い未実施のままにし、人の実機受入6も保留する。
- RED: `.venv/Scripts/python.exe core/anet-core/testdata/prd109/run_validation.py fatal --output .scratch/prd109/fatal-routing-red-20261031-1947` は `Unexpected fatal diagnostic: .../stderr.log` で失敗した。Runner自体は終了コード1で、追加のstderr出力が現行条件に反することを確認した。
- 修正: ErrorDialog.cppの`<cstdio>`と`fwrite`/`fflush`を撤去し、HEADの内容へ完全に戻した。既存reporterと起動段階ごとのloggerをそのまま使う。fatalドライバはRunログの診断の存在とstderrの診断の不在をそれぞれ検証する。READMEと本書の現行条件・完了報告・ファイル一覧を更新し、旧条件での結果は当時の記録と明記して保持した。
- ビルド: 本書「検証」のVsDevCmd経由コマンドでDebug全体とAnetRLRunner-gui-testを再ビルドし、終了値0。既存のD9025（`/Zi`から`/Z7`への上書き）警告のみ。
- GREEN: `.venv/Scripts/python.exe core/anet-core/testdata/prd109/run_validation.py fatal --output .scratch/prd109/fatal-routing-fixed-20261031-1947` は成功。Runnerの終了コード1、Runログに `Unknown run.eval_schedule tag 'nosuch'` あり、stderr.logは0バイトで診断なし。修正後exeのSHA256は `dba23eec44da6cdf542828f1ba910e149fa8eddb5f1bc6dc5945e3eba07a2438`。各実行のarguments・hash・終了値・所要時間とRun出力を新規ディレクトリへ保存した。
- GUI: `.venv/Scripts/python.exe core/anet-core/testdata/prd109/run_validation.py gui --output .scratch/prd109/gui-routing-fixed-20261031-1947` は新規10件と既存2件の全12プロセスで終了値0。`[prd109_close_failure]` を含め、JUnit12件もfailures/errorsが0だった。
- 自己点検: 実装品質確認観点の全項目を適用範囲に従って確認した。今回の修正に公開API・設定・学習経路・所有権・計測の追加は無い。ErrorDialog.cppはHEADとのバイト一致、LF・BOM保持、Python構文、git diff --checkを確認した。020のSHA256は修正前後で `dee840abed8aeb104ca8f2d52843d9345b1d2938969ac45f259c0d42f7a67171` のまま。文書の再読・hash・mtimeと差分も確認した。
- 未実施: 既定スイートと4環境scalar比較は今回の変更がエラー出力先の復帰だけなので再実行せず、上記の前回結果を保持する。再レビューはユーザー指示で未実施、人の実機受入6も未実施であり、受入完了とはしない。コミット・pushは行っていない。

## 再レビューと実機確認の完了（2026-10-31）

- 修正後のcc-reviewはCodex・Claudeとも計画・実装がOKで、追加指摘なし。原文は `.scratch/prd109/review-20261031-1951.md` と `.scratch/prd109/review-20261031-1951-claude.md`。レビュー時点では人の実機受入6だけが未実施だった。
- その後、ユーザーから「もろもろ実動作確認してoｋ」と報告を受領した。この報告を人による実機確認の結果として記録し、受入6の保留を解消した。実装・自動検証・修正後再レビューと合わせ、PRD109の受入完了とする。
- 今回は結果記録だけを更新した。ビルド・テスト・再レビューは再実行していない。ユーザーの「しない」の選択に従いステージせず、コミット・pushも行っていない。

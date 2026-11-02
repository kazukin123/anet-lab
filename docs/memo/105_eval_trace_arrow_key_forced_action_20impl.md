# Evaluation Trace 操作・並べ替え・ダンプ 実装メモ

## 概要

[PRD105](105_eval_trace_arrow_key_forced_action_10prd.md) の D1〜D12 と受入基準1〜10を実装する。Traceモードの転送矢印を行移動へ回し、強制stepの識別、数値による並べ替え、保持行の通知内容のログへのダンプを追加する。

## 未決事項監査

- ユーザー判断が必要なブロッカー: 0。
- repo evidenceで解決済み: EvalRunnerの通知前にforcedを確定できる（trainer.cpp）。列は通知中に同期に読む（metrics.cpp、ADR0057）。TracePlayerは受け取った行IDの順だけで再生する（gui.cpp）。CONTEXT.mdの強制行動は追加済み。
- 本計画へ固定: 通知オブジェクトを保持せず、状態のTensor参照・共有のaction_infoとstep_result・counts・名前・forcedを行へ保持する。転送と文字列化はダンプ時だけ行う。
- PRD範囲外: Env別キー割り当て、train側の強制行動、Actor内部状態とframe stack、ファイルへのダンプ保存、色による強制行動の強調。

## 主な変更

- EvalRunner::GetScalarにlane指定のforcedを追加する。初期値0、強制stepで1、次の通常stepで0。lane省略はnullopt、範囲外はfail-fast。他のキーはRunnerBaseへ委譲する。BatchActionInfoとWithActionは変更しない。
- EvalTracePanelへTraceモードの照会と行移動の公開口を追加し、表の↑↓と共有する。RunnerFrameはTrace中の転送↑↓を行移動へ回し、←→を消費する。最大化で隠れている間も同じとする。
- EvalPanelは有効な強制stepの成功後にstatus field 0へ `Forced action: 番号:名前` を出す。common.txtの260へ `$runner forced label:forced` を?=で足し、既存列の値を維持する。
- gui.hpp/cppへTraceRowSorterを追加する。既定は実step降順、列クリックは初回降順・以後反転、空欄/pendingは常に下端、同値はstep降順。百分位は保持した百分位、行動は番号で比較する。追加と遅延確定を並びへ反映し、保持上限では表示順によらず最古stepを消す。既定の追加は先頭挿入を維持する。
- 選択を行IDで維持し、並べ替え後は選択行へスクロールする。Live/Traceと再生状態を保持する。TracePlayerへ表の下端から上端のIDを渡し、既存ロジックは維持して引数名と説明を再生順へ改める。見出しの印とsliderも表の順に従う。閉じて開き直すと並べ替えを初期化する。
- gui.hpp/cppへTraceDumpFormatterとその保持データを追加する。行は遷移前状態のTensor参照、action_info、step_result、counts、Runner/Env/Agent名、forcedを保持する。forcedは列の有無によらず同期通知中に採取する。行削除とpane閉鎖で解放する。
- ダブルクリックで行を選び、PRD§5.5のINFO1件とVERBOSE1件を出す。INFOは改行なし、全列と通知項目を指定順で出す。複数要素/空Tensorは形・型・元deviceのみ、単一要素は値、未定義はundefined。VERBOSEはCPUへ写して既存の文字列化をつなぎ、infoと不透明なReplay初期優先度ヒントも出す。ログレベル処理を維持する。
- 日本語設計文書020/100/140/160とPRD094の冒頭注記を追随させる。既定列の記載順と強制行動の表記をそろえる。英訳はPRD指定どおり翻訳工程へ残す。新しいADRと用語は作らない。

公開ヘッダの変更はinclude/anet/trainer.hpp（forcedの読出し）とinclude/anet/gui.hpp（GUI非依存の順序・ダンプ部品と再生順の説明）である。新しいproductionファイルは作らず、既存の機能グループへ置く。CreateTraceSnapshot/TracePlayer::Select等の既存の動詞の意味に合わせる。

## テスト

- Public interface / surface: EvalRunner::DoStep/GetScalar、TrainEventからのStepTraceBuilder、common.txtの解決済み設定、TraceRowSorter/TraceDumpFormatter/TracePlayer、実wxイベント経由のRunnerFrame/EvalTracePanel。
- 最初のtracer bullet: 実EvalRunnerのDoStep(action)→TrainEvent→StepTraceBuilderでforced=1の行ができる。
- TDD順序: 各behaviorを1テスト→RED確認→最小実装→GREENとする。全テストの先書きは行わない。次に通常step復帰/初期値/lane/TrainRunner、既定列、数値順と空欄/同値/トグル、挿入/遅延値/上限、任意の再生順、ダンプの各書式、実wxの転送矢印/選択と再生の維持/見出し/ダブルクリックを確認する。
- 保持した通知内容は後続stepで変わらず、行削除とpane閉鎖で解放されることも確認する。

## 検証

1. MSVCのVsDevCmd.bat経由で変更前のDebug全体をビルドし、現在の未コミット変更を含むバイナリと設定をPRD105専用保存先へ保存する。過去のPRD098 baselineは流用しない。
2. Debug全体とAnetRLRunner-gui-testをビルドし、追加テストとTrainer/Trace/GUIの関連回帰を実行する。
3. core/anet-core/testdata/prd105/へ専用検証資材を追加する。PRD098段0の4環境、seed1、8lane、batch16、warmup256、replay ratio1、8,704experience、評価停止、FP32、決定論的backendを引き継ぎ、時間・資源の90_perf/91_resource/92_resource以外の全scalar点を完全比較する。過去のドライバ・記録を変更しない。
4. qbert/LunarLanderでPRD§7の操作確認を実施する。qbertのverboseダンプがログ欄へ出るまでの実時間を記録する。並べ替え/保持/文字列化/転送の主要境界を計測する。
5. 実装品質確認観点の全項目を自己点検し、独立したan-reviewへ今回の差分と本計画を依頼する。差し戻しは修正して同じレビューエージェントへ再依頼する。

頻繁な経路の見積もり: forcedはCPUのlane数分の更新。通知保持はTensorとshared_ptr等の参照コピーのみで転送・文字列化を足さない。既定順の挿入は先頭へ追加し、任意順はmax_rows内の比較と移動（既定1000行）。百分位窓は並べ替えで再投入しない。ダンプ時の同期転送は明示ダブルクリック時のみで毎step/tickには足さない。Atari1000行の追加保持はPRD見積もり約108MB、既存snapshotと合わせ約240〜410MB。実測は上記4で記録する。

## 前提

- PRDの確定事項を採用し、旧挙動用の設定や互換層は作らない。
- 既存の未コミット変更と過去のRun/検証記録を保持する。コミット・pushは行わない。
- 比較中に別作業のDQN変更が進んだ場合、その状態ではPRD105前後の比較を成立とせず基準を揃え直す。

## 経過と結果

- 2026-10-30: 合意済み計画で実装開始。基準HEADは4f2e2645c82a3202c02ee0d63781ba30e5ffb652。既存変更はDQN/TempoRL、wiki、bat等で、予定productionファイルとの直接重複なし。
- ユーザーはコピー版の評価Runを実行中。ビルド・単体テストの並行実行を許可した。4環境比較と追加の実Runは評価終了後に行う。
- 未実施: baseline採取、ビルド、テスト、4環境比較、実GUI確認、自己点検、独立レビュー。まだ実装完了ではない。
- baseline: VsDevCmd経由の `cmake --build --preset x64-Debug` は成功（ninja: no work to do）。exe・DLL・configを `.scratch/prd105/baseline-20261030-0138/` へ保存した。
- TDD forced_trace: `cmake --build --preset x64-Debug --target anet-core-test` 後に `anet-core-test.exe "[forced_trace]"` を実行。REDはcolumn=260、target=$runner、key=forcedの未認識で1件失敗。最小実装後のGREENは1テスト・3assertion成功。初期値・lane境界・通常step復帰を追加して次の検証へ進んでいる。
- TDD forced_columns: 既定列260が空文字でREDを確認。common.txtへ1行追加後、forced_trace/forced_columnsは計22assertion・2テスト成功。
- sorterの受け渡し: TraceRowSorterは行ID・実step・表示用数値だけを所有し、挿入位置、表示順、保持上限で除くIDを返す。画像・通知内容はEvalTracePanelが所有する。列変更時だけ全行を並べ、通常の追加/遅延確定は対象行だけ入れ直す。
- 設定全キー比較: `[prd093-config-snapshot]` を保存済みconfigと現用configに対して別の出力先で実行（各15assertion成功）。Atari/CartPole/DropMerge/GridMaze/GridMaze_muzero/ImageCls/LunarLanderの7環境とも、差分は `app.eval_trace_panel.columns.[260]` の追加だけだった。結果は `.scratch/prd105/config-before-20261030-0138/` と `config-after-20261030-0138/`。
- `check_default_leaves.py`: audited=3720、errors=2で終了1。今回触らないAtari.txtの `app.online.eval_panel.fps` と `app.online.train_panel.fps` の既存の?=が検査対象。PRD105の新規common列は指摘なし。今回の変更範囲外のため保持する。
- TDD trace_sort: 未宣言のTraceRowSorterでREDを確認し、数値順と列トグルのGREENは7assertion。次にTrim未実装でREDを確認し、空欄/同値/挿入/遅延確定/最古削除を追加後は2テスト・17assertion成功。
- TDD trace_dump: 未宣言のTraceDumpFormatterでREDを確認。CPUの通知部品を使うINFO/VERBOSEと任意の表順の再生を実装・確認し、sorterを含む4テスト・47assertion成功。
- GUIテストtargetの初回ビルドはコンパイラの同時起動数が大きかったため、自分のビルドを中断し、`cmake --build --preset x64-Debug --target AnetRLRunner-gui-test --parallel 2` で再開した。裏のコピー版評価Runは操作していない。
- PRD094の冒頭注記だけを、成果物キーprd・品質観点docs/quality/prd.mdで独立したan-reviewへ依頼した（1往復目）。実装のレビューは別に行う。
- PRD094注記の独立レビューはOK。不足・修正要求・範囲外の検出・ユーザー確認事項はすべてなし。原文: `.scratch/prd105/review-20261030-0210-prd.md`。
- TDD trace_ui_sort: IsTraceMode未実装でREDを確認。パネルへsorterと再生順を接続後、実wxの数値順/選択行/再生状態/保持上限/開き直しは18assertion成功。
- GREEN後の小調整としてsorterの数値保持をdequeにし、既定の先頭挿入で全行の入れ物を移動しない形へ変えた。再検証は `[prd105],[trace_player],[trace_live_step],[trace_live_boundary]` の9テスト・114assertion成功。
- 既存CPU回帰: `[trace_player],[trace_display],[trace_play_available],[trace_live_step],[trace_live_boundary],[trace_speed_label],[trace_timing],[step_trace],[manual_action],[trace_snapshot]` は11テスト・212assertion成功。ログ: `.scratch/prd105/core-regression-20261030-0215.log`。
- 現在位置: `trace_ui_dump` のテストだけを追加し、RED確認用のGUI targetを並列2でビルド中。通知の保持/セルダブルクリック接続、RunnerFrameの矢印ガード、EvalPanelの成功通知はまだ未実装。`forced_scope`（複数lane初期値とTrainRunner非対応）の追加テストは次のcoreビルドで実行する。4環境比較とqbert/LunarLander実画面は裏の評価Run終了後。
- TDD trace_ui_dump: action_infoのweak参照が即失効するREDを確認。通知内容の保持とセルダブルクリックを接続後、繰り返しのINFO/VERBOSE、eviction/閉鎖時の解放とsortの2テスト・36assertion成功。名前は初期化時に評価Envの実効class_id（取得できなければEnv名）とAgentのclass_idをキャッシュし、通知時のRunner名と合わせて行へ写す。
- TDD prd105_ui_keys: 初回はテストmainの単独起動タグ未登録によるEnv二重登録だったため、既存のeval_steps_menuと同じ分岐へタグを追加。再試行で転送↓20回後も選択が0行目のままのREDを確認。矢印ガード/共通MoveSelection/成功通知を接続後、単独CPU RunnerAppでGREEN（詳細 `.scratch/prd105/keys-green.log`）。最大化で隠れたTrace、Live、テンキー、Ctrl、閉鎖後も確認した。
- 実装コードの接続は完了。これからDebug全体、core/GUI回帰、自己点検、impl/impl-planの独立レビューを行う。4環境比較と実qbert/LunarLanderは追加Runのため、評価Run終了のユーザー回答待ち（質問済み）。
- 自己点検で、stepを百分位表示する列を既定の生step列として扱う境界ケースを検出した。実EvalRunnerの3stepを通知して見出しと百分位順を確認する `[trace_ui_step_pct]` は5assertion失敗のRED。既定のstep列の検出を `display:value` に限る最小修正を行い、Debug全体/GUI targetを再ビルドしている。
- 最終core回帰: `core/anet-core/bin/Debug/anet-core-test.exe "[prd105],[trace_player],[trace_display],[trace_play_available],[trace_live_step],[trace_live_boundary],[trace_speed_label],[trace_timing],[step_trace],[manual_action],[trace_snapshot]" --out .scratch/prd105/core-final.log` は18テスト・287assertion成功。複数lane初期値、範囲外、TrainRunner非対応を含む。
- 差分の自己点検: 今回の18既存ファイルはすべてLF、UTF-8 BOM有無を基準HEADから保持。`git diff --check` は指摘なし。ヘッダのusing namespace・TODO形式・旧引数名oldest_to_newest・共有プロセス文書のプロジェクト依存語は該当なし。gui.cpp冒頭のdynamic_cast3箇所は基準HEADからある変更範囲外で、今回の追加は0箇所。新規の同期Tensor転送はダンプの明示要求時だけで、通知保持に追加していない。
- 最終Debugビルド: `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 2 && cmake --build --preset x64-Debug --target AnetRLRunner-gui-test --parallel 2'` は終了0。裏の評価へ配慮して並列2を維持した。
- 最終GUI回帰: `apps/runner/bin/Debug/AnetRLRunner-gui-test.exe --out .scratch/prd105/gui-final.log` は17テスト・166assertion成功。step百分位列の5assertionもGREEN。単独の `apps/runner/bin/Debug/AnetRLRunner-gui-test.exe "[prd105_ui_keys]" --out .scratch/prd105/keys-final.log` は1テスト・30assertion成功。
- impl/impl-planの独立レビューへ初回依頼済み。差分は `.scratch/prd105/diff-20261030-0308/`、対象は今回の17追跡ファイルと新規検証README・新規計画だけ。PRD094注記と無関係なdirtyは別扱い。4環境比較、qbert/LunarLander実画面とダンプ表示時間はユーザーの評価終了の返答待ちで未実施。
- CPU合成計測: `apps/runner/bin/Debug/AnetRLRunner-gui-test.exe "[trace_ui_perf]" --out .scratch/prd105/gui-perf-final.log` は1assertion成功、1000行・12列・2遅延列・1000sampleでmedian=1.0331ms / p95=3.8646ms。既存目安1.72414msを超過した。コピー版の評価Runと並行という条件で、実Run速度の判定には使わない。CompleteRowで表示位置が不変のときの再生順/選択/controls再構築を省き、再ビルド/回帰・再計測へ進む。レビュー側にも変更を通知済み。評価終了後の同条件再測定を残す。
- impl/impl-planレビュー1往復目: 計画はOK、実装はテスト2点で差し戻し。後続step後の保持値不変の確認が足りないことと、RestoreLogの集成体初期化が位置指定だったこと。原文は `.scratch/prd105/review-20261030-0314-impl.md`。保持上限2でstep1のreward=7とstep2のreward=8を残し、step1のダブルクリックを2回行って列/通知/INFOとVERBOSEの不変を確認する形へ改めた。step3でstep1が解放され、pane閉鎖で残りも解放される確認を維持した。初期化も指示付きへ修正。
- CompleteRow調整後、同じVsDevCmd経由のDebug全体/GUIビルドは終了0。調整後のCPU合成計測は `.scratch/prd105/gui-perf-adjusted.log`、median=0.7574ms / p95=2.4544ms、保持行数の1assertion成功。前回より低いが、p95は既存目安を超えており合格とはしない。追加の調整や実行は評価終了後の同条件再測定を踏まえて決める。レビュー対応はテストのみで、テストtargetを再ビルド中。
- レビュー対応後のGUI targetビルドは終了0。最終GUI通常スイート `apps/runner/bin/Debug/AnetRLRunner-gui-test.exe --out .scratch/prd105/gui-reviewed-final.log` は17テスト・173assertion成功。単独キー `apps/runner/bin/Debug/AnetRLRunner-gui-test.exe "[prd105_ui_keys]" --out .scratch/prd105/keys-reviewed-final.log` は1テスト・30assertion成功。production最終調整の回帰と、保持値不変の補強を含む。
- 独立レビュー2往復目: impl-plan/implともOK。修正要求、範囲外の検出、ユーザー確認事項なし。前回の2点は解消済み。原文 `.scratch/prd105/review-20261030-0317-impl.md`。この判定は未実施の実Run/実画面/性能再測定の合格を示すものではない。

## 03:17 JST時点の到達点

- 実施内容: PRD105の矢印ガード、forced列と成功通知、数値ソートと表の順の再生、過去行の通知保持とINFO/VERBOSEダンプを実装した。CPUでのDebugビルド・関連回帰と自己点検・独立レビューを完了した。
- 変更ファイル: Runnerの3パネル/フレーム、coreのtrainer/gui、対応テスト、commonの1列、日本語設計4文書、PRD094の注記、PRD105の計画と検証README。無関係な未コミット変更とコピー版の評価Runを保持し、コミットしていない。
- レビュー結果: PRD094注記は1往復でOK、計画/実装は2往復でOK。見送った指摘なし。残る再確認は4環境の数値等価性、qbert/LunarLanderの実画面、dump表示時間と性能の再測定。
- 決まらなかったこと: 設計判断の未決と課題の起票候補はなし。受入検証は一部未実施であり、全受入の完了とはしない。
- 判断してほしいこと: 裏の評価Runが終了したかの回答待ち。ユーザーの回答まで追加Runと実画面検証へ進まない。
- 次にできること: 評価終了後、4環境のbefore/after比較、実画面とダンプ表示時間、同条件の性能測定を実施し、結果を本計画へ追記する。英語設計文書はPRD指定どおり別の翻訳工程へ残す。

## 追加の受入検証

- 2026-10-30 08:11 JST: ユーザーの「まだだけど開始」により、コピー版の評価Runを継続したまま、残る4環境比較と実画面/性能測定を並行開始する。上の評価終了待ちを解除する。速度と実時間には他の評価と並行した条件を明記する。
- 専用 `core/anet-core/testdata/prd105/run_validation.py` を追加。PRD098の段0の条件生成とPRD084の正規化だけを呼び、PRD105専用出力へ8本を逐次実行する。beforeは保存済みbin/config、afterは現Debug exeと採取時点のconfigコピーを使う。出力先は新規だけを許し、過去のRunとドライバは変更しない。
- 最初の保存先 `validation-20261030-0811/` は、条件生成器が読むbaseline-configのコピー不足でRun起動前に停止した。コピーを追加し、別の新規保存先 `.scratch/prd105/validation-20261030-0811b/` で実行した。GridMazeは26,946点/79タグ、LunarLanderは24,399点/84タグでbefore/afterの完全一致。AtariとDropMergeは実行中。
- computer-useのnode runtimeと統合CUAの両方が、MXC launcher `CreateProcessSecurityEnvironment ... HRESULT = 0x80070020` で起動不能。再初期化後も同じだった。外部UI操作による実キー/フォーカス/見た目の確認は未実施として残す。
- 代わりに既存の実wx `[prd105_ui_keys]` へ `PRD105_UI_CONFIG` 入力を追加し、qbert/LunarLanderの実Env/Agentを初期化して同じ公開イベントの受入を行う。ソートと表順の再生、QValue行クリック、全4矢印の強制行動、ログ欄のinfo/verboseとdump反映時間も確認する。production APIは変更しない。専用ドライバの `--prepare-ui` は停止状態の設定だけを生成する。`.scratch/prd105/ui-20261030-0815/` の2入力を作成済み。
- Atari(qbert)も23,185点/80タグで完全一致。DropMergeの比較を継続中。今回のafter比較はexe SHA256 `bbcd364da7d26bfba6c15f1e9cded5d9c6abf07fedd6f8946c2796fd27df3502` で開始し、GUI専用テストtargetだけを並行ビルドしている。
- LunarLander実wxは `.scratch/prd105/real-ui-lunar.log` で71assertion成功。実ログ欄にinfo 1,510文字/381.089ms、verbose 63,562文字/364.457msで反映した（評価および比較Runと並行、Eval actorのTensorはCUDA）。Run出力は `ui-20261030-0815/runs/run_20261030-081927_prd105_ui_lunarlander/`。
- 実ログでenvがEvalPanelと出ていたため確認したところ、GetScopedConfigDataは完全キーを返すのにclass_idを相対キーで読んでいた。`trace_ui_dump` にenv=CartPoleEnvの確認を追加して1assertion失敗のREDを確認（`.scratch/prd105/env-name-red.log`）。読むキーをenv.class_idへ直した。実wx入力側にも実効Env名の確認を足して再ビルド中。変更はダンプ名の初期化だけで、学習経路と共通設定に追加変更はない。
- 4環境比較は全8本が終了0、全105,682点（片側の合計）で完全一致。環境別の点数/タグ数はGridMaze 26,946/79、LunarLander 24,399/84、qbert 23,185/80、DropMerge 31,152/129。出力 `.scratch/prd105/validation-20261030-0811b/comparison.json` と各normalized.summary.jsonが詳細を持つ。最初の3環境のafterはbbcd364d...、最後のDropMergeのafterはEnv表示名だけを修正した `0af0e8efb75b174f8960d3a2bd07c70ef83c4fbe9ccc3ddaea668655263f80aa`。各実行のSHAと入力hashをexecution.jsonへ残した。比較中の追加変更はGUI初期化時に表示名を読むキー1箇所だけで、学習コード/設定は不変。
- Env名修正後のDebug全体はVsDevCmd経由で成功（ninja: no work to do）。GUI通常回帰 `.scratch/prd105/gui-acceptance-final.log` は17テスト・174assertion、通常の単独キー `.scratch/prd105/keys-acceptance-final.log` は30assertion成功。LunarLanderの実wx再検証 `.scratch/prd105/real-ui-lunar-final.log` は73assertion成功、env=LunarLanderEnvも確認した。
- 修正後LunarLanderのdump反映時間はinfo 1,515文字/412.938ms、verbose 63,567文字/456.963ms。元ログは `ui-20261030-0815/runs/run_20261030-082535_prd105_ui_lunarlander/` のstdout.log。qbertの実wxも `.scratch/prd105/real-ui-qbert-final.log` で73assertion成功、env=AtariEnvとaction=2:UPを確認した。
- qbertのdump反映はinfo 1,975文字/15,062.6ms、verbose 1,747,083文字/13,931.7ms。元ログ `ui-20261030-0815/runs/run_20261030-082645_prd105_ui_atari/stdout.log`。既存のTensor文字列化で全要素を出す現在の契約による重い明示操作で、ホットパスの指標とは分ける。infoでもVERBOSE生成とRunテキストログへの出力を行うため、ログ欄のlevelを下げても処理時間の軽減にはならなかった。画面の外部操作による計測ではなく、実wxダブルクリックイベントの配送から、Flush/Yield後のログ欄のテキストを取得するまでの実時間である。
- 追加の比較Run/ビルド/実Envテストが終わってからCPU合成計測を3回実施した。コピー版の評価Runは並行継続中という条件。各回とも保持行数の1assertion成功で、median/p95は順に0.6172/1.2922ms、0.6354/1.2832ms、0.6706/1.8753ms。目安1.72414ms内が2回、超過が1回であり、性能を一律に合格とはしない。ログ `.scratch/prd105/perf-acceptance-1.log` から `-3.log`。評価終了後の安定した条件での再測定は未実施として残す。
- 追加検証後の独立レビュー3往復目は計画/実装ともOK。追加指摘・範囲外検出・ユーザー確認事項なし。原文 `.scratch/prd105/review-20261030-0829-impl.md`。外部UI確認と安定条件での性能確認の合格は示さない。

## 08:29 JST時点の到達点

- 実施内容: 4環境の数値比較とqbert/LunarLanderの実Env/実wx検証、ログ欄反映時間とCPU合成性能の測定を追加した。ダンプのEnv名の取得キーを修正し、RED/GREENと回帰で確認した。Debug全体は成功。
- 変更ファイル: 追加分はEvalTracePanel.cpp、EvalTracePanel_test.cpp、PRD105検証README/新規run_validation.pyと本計画。その他の変更は03:17の記録どおり。無関係なdirtyを保持し、コミットしていない。
- レビュー結果: 計画/実装は3往復目でOK、PRD094注記は1往復でOK。見送った指摘なし。数値比較は全105,682点一致、最終GUI回帰174assertion、通常キー30assertion、実Env2環境は各73assertion成功。
- 決まらなかったこと: 設計判断の未決と起票した課題はなし。外部ツールでの実キー/フォーカス/外観確認はツール起動不能で未実施。背景評価終了後の性能確認も未実施。全受入が済んだとはしない。
- 判断してほしいこと: 追加の設計判断はなし。qbertの全要素ダンプは約14〜15秒かかり、info表示だけでは軽減しないという実測上の制約を残す。性能p95も3回中1回が目安を超えた。
- 次にできること: UI操作環境の復旧と背景評価終了後に残る確認を行う。ダンプ応答の高速化を扱う場合は、全要素/既存書式/Runログへの出力の契約を維持する前提で検討する。英訳は別工程へ残す。

## ソート順の印の表示修正

- 2026-10-30 22:35 JST: ユーザーから既定STEP降順と見出しクリック後の昇降順を三角で示すよう依頼された。ソート状態は既にSetSortingColumnへ渡していたが、wxWidgets 3.3.3の通常のGrid見出しではUpdateColumnSortingIndicatorの描画が未実装で、印が見えていなかった。
- SetUseNativeColLabelsを有効にして、既存のソート列/昇降順をシステム標準の見出し描画へ渡す。既定のstep降順は下向き、昇順は上向きの三角となる。ラベルとソートの処理は変えない。
- 実wxの見出し画像で、既定の降順・印なし・昇順・降順へ復帰を比較するtrace_ui_sort_iconを追加した。既存のソート属性だけを検証するテストでは描画の不足を検出できなかったため、実描画を確認する。ステージ済み21ファイルのindexは保持し、今回の追加分は作業ツリーだけへ編集する。
- REDは「印なしと既定降順の描画が同じ」で失敗した（`.scratch/prd105/sort-icon-red.log`）。失敗詳細でCatch2が大きい画素配列を文字列化すると内部assertになったため、配列の比較結果をboolとして渡す形へ修正した。`PRD105_SORT_CAPTURE_DIR` 指定時は比較する見出し画像4枚も保存する。
- ネイティブ描画後のGREENは10assertion、通常GUI回帰は18テスト・180assertion成功した。ただし保存画像を確認すると短いstep見出しでラベル末尾と右側の印が近く重なっていたため、見出しの初期幅とフォント変更時の幅で使う余白を28 DIPに統一した。手動指定済みの幅を保持する処理は維持する。三角のための幅と描画を再確認する。
- 最終Debug全体/GUI target: VsDevCmd経由の `cmake --build --preset x64-Debug --target all AnetRLRunner-gui-test --parallel 2` は終了0。描画と幅の確認は `.scratch/prd105/sort-icon-spacing-green.log` で11assertion成功。画像 `.scratch/prd105/sort-icons-20261030-224635/sort-header-0.png` が既定降順、-1が印なし、-2が昇順、-3が降順復帰で、三角と文字の重なりが解消したことを確認した。
- 最終GUI通常回帰 `AnetRLRunner-gui-test.exe --out .scratch/prd105/gui-sort-icon-spacing-final.log` は18テスト・181assertion成功。単独の `[prd105_ui_keys]` は `.scratch/prd105/keys-sort-icon-final.log` で30assertion成功。今回の描画修正4ファイルは未ステージのまま、既存indexの21ファイルは保持する。
- 追加修正後の独立レビュー（通算4往復目）は計画/実装ともOK。追加指摘・範囲外検出・確認事項なし。原文 `.scratch/prd105/review-20261030-2247-impl.md`。

### 表示修正の完了報告

- 実施内容: 既定STEP降順とクリック後の現在のソート列に、降順は下向き・昇順は上向きの三角を表示する。文字と印のための余白を確保した。
- 変更ファイル: EvalTracePanel.cpp、EvalTracePanel_test.cpp、020_user_guide_run.jp.md、本計画の4ファイル。
- レビュー結果: 追加修正は1往復でOK（PRD105全体では通算4往復）。Debug全体/GUI target、描画/幅、GUI回帰、単独キーも成功。
- 決まらなかったこと: 今回の表示修正の未決と課題の起票候補はなし。以前の検証留保は継続する。
- 判断してほしいこと: なし。
- 次にできること: 今回の追加4ファイルを、ユーザーの指示があれば既存のPRD105ステージ内容へ反映する。

## ソート印のテーマ対応

- 2026-10-30: ユーザーのLight/Dark画像で、OS標準の三角が暗い背景では黒く見えにくいことを確認した。ユーザー指定どおり、印はシステムの選択ハイライト色（wxSYS_COLOUR_HIGHLIGHT）で描画する。
- EvalTracePanel.cppの内部にTraceGridを置く。枠と文字はwxGridの標準描画、印だけはDIPで寸法を決めた三角形を描き、OS標準の印を重ねない。ソート変更/初期化/外観変更時に見出しを再描画する。色は描画時に取得し、キャッシュしない。
- 三角はフォントやUnicode文字を使わないため、日本語フォント・Windowsの表示言語・コードページに依存しない。英語版Windowsの実機確認は行っていない。
- 実wx描画テストにハイライト色の画素がソート列で現れ、ソートなしでは消えることを追加した。LightとDarkを別processで確認する。直前にstageされた21ファイルは保持し、今回の追加は未ステージとする。
- DarkのRED `.scratch/prd105/sort-highlight-red.log` は、ソート印のハイライト色の画素数が0で1assertion失敗（9assertion中8成功）を確認した。標準の三角を別の三角で覆うのではなく、wxGridの見出し描画からその印を外し、色を指定した三角だけを追加する。
- Debug全体/GUI targetはVsDevCmd経由の `cmake --build --preset x64-Debug --target all AnetRLRunner-gui-test --parallel 2` で終了0。Light/Darkは `[trace_ui_sort_icon]` を別processで実行し、各13assertion成功（`.scratch/prd105/sort-highlight-light.log` / `sort-highlight-dark.log`）。Darkは `PRD094_UI_DARK=1`、Lightは同変数を除いて起動した。
- 最終画像は `.scratch/prd105/sort-highlight-light-20261030-2306/` と `sort-highlight-dark-20261030-2306/` の `sort-header-0.png`（降順）〜`-3.png`（降順復帰）。Light/Darkの下向きとDarkの上向き三角を目視し、ハイライト色で見え、文字と重ならないことを確認した。
- 通常GUI回帰 `.scratch/prd105/gui-sort-highlight-final.log` は18テスト・183assertion成功。単独キー `.scratch/prd105/keys-sort-highlight-final.log` は30assertion成功。MSVCのビルド設定もCMakeLists.txtで /utf-8 を明示しているが、英語Windows実機の検証を代替したとはしない。
- 独立レビュー5往復目は計画/実装ともOK。追加指摘・範囲外検出・確認事項なし。原文 `.scratch/prd105/review-20261030-2308-impl.md`。今回の色と描画の修正は完了、英語Windows実機と以前の検証留保は継続する。

### テーマ対応の完了報告

- 実施内容: 三角をシステムのハイライト色で図形として描き、Light/Darkの見やすさと表示言語非依存を確認した。
- 変更ファイル: EvalTracePanel.cpp、同_test.cpp、020_user_guide_run.jp.md、本計画の4ファイル。
- レビュー結果: 追加修正は1往復でOK（通算5往復）。Debugビルド、Light/Dark描画、GUI回帰、単独キーも成功。
- 決まらなかったこと: 今回の実装判断の未決はなし。英語Windows実機は未検証で、以前の検証留保も継続する。
- 判断してほしいこと: なし。
- 次にできること: 今回の追加4ファイルは未ステージ。指示があれば既存のPRD105ステージ内容へ反映する。

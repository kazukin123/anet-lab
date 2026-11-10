# PRD122 小さなリファクタリングとテストの空白 実装メモ

## 概要

[PRD122](122_small_refactor_and_test_gap_fixes_10prd.md) の D1〜D15 を実装する。既存の振る舞いを固定するテストを先に置き、core 6件、GUI 2件の順で進める。基準コミットは `9bd1f17dfd912f08a02314ba1ad25c4c5b859d18`。着手時の無関係な未コミット変更は保持する。実装・検証の記録は本書を正本とする。

## 未決事項監査

- ユーザー判断が必要なブロッカー: 0。
- repo evidenceで解決済み: dqn_based_agent.hpp の BatchUpdateResult は直接比較43キーと8群を持ち、補助4関数も同ヘッダにある。init.cpp は3キーと既定値の表を持つ。gui.cpp は型照会3箇所を持つ。QValuePanel::Update の加工と Grid 更新は PRD の記述に一致する。
- 2ximplへ固定: 直接キーの表は constexpr 配列と重複検査の static_assert で守る。8群は現行の順で問い合わせる。既知だが未成立の値を含め現行結果を保持する。
- PRD範囲外: PRD117・PRD948・PRD922・PRD108 と、MuZero truncated 補い値、平滑化半径0以下、既存 Grid の死にコード。

## 主な変更

### ステージ1

1. replay_buffer_test.cpp: DefaultReplayBuffer のコンストラクタの priority store/completer の片方のみ2通りの例外と文言、両方あり/なしの構築をテストする。
2. scaler_test.cpp: auto post scale（gamma≠0.5）、robust update の warmup 境界と行の除外、正の非単位scaleでのSymLogを公開APIで確認する。
3. prd101_muzero_test.cpp: builder の done/truncated、sampler の終端またぎ/記録末尾、ReplayBuffer::Push のlane分離を手計算で確認する。
4. dqn_based_agent.hpp/.cpp: GetScalar の全キーの成立/未成立、未知キー、到達可能な異常経路の特性テストを先に dqn_based_agent_scalar_test.cpp に置く。巨大な既存テストと独立してキー全体を照合するために分ける。43個は表の行を名前付き定数として扱い、8群は専用関数へ分ける。GetPlasticityScalar・ValidateSpectralNormSentinel・TransQToCpu・TransRealQToCpuも.cppへ移し、先頭コメントとpragma onceを直す。
5. init.cpp/init_test.cpp: キー名3個だけを残し、config_data.Hasで存在を確認してからGetする。指定された不正deviceと未指定のテストを通す。既定のauto/cpuと明示cudaの一括エラーは維持する。

### ステージ2

1. apps/runner/src/QValuePanel.hpp/.cpp: PRDのQValueHeatMapRequest・QValueHeatMapRangeState・QValueHeatMapResultを既存のQValueDataと同じファイルに置く。ComputeQValueHeatMapはQValueDataをconst参照で受け、vvだけを1回コピーする。表示指定はUpdateで各1回読む。平滑化の重みは既存のPanelコンストラクタで作りrequestで同期呼び出し中だけ借用する。範囲状態はPanelが保持し、既存ResetRangeからリセットする。Gridと描画は既存コードの内容を保って別関数へ移す。既存ComputePlasticityMetricsと同じComputeの語義を使う。UpdateGridのUpdateは既存QValuePanel::Update（QValuePanel.cpp:639）と同じく表示値の更新、RenderHeatMapのRenderは既存Conv2dPanel::RenderActivations（Conv2dPanel.cpp:214）およびSweepHeatMapPanel::Render（HeatMapPanel.cpp:563）と同じく描画の意味で使う。
2. QValuePanel_test.cppを新設し、全Histogram/Advantage/Log/Auto Range組み合わせ、平滑化、累積範囲、一定値を手計算する。windowを作らず加工関数を呼ぶ。切り出し前のUpdateで同じ入力を通して一時出力を採取してから本体を移す。
3. Panel_key_test.cppを新設し、入れ子の子window/内側Panel自身/直下/Panel外からwxEVT_CHAR_HOOKを送る特性テストを先に通す。Panel::OnKeyは転送の内容を保ち、SkipとStopPropagationを使う。型照会を削除する。
4. GUIテスト2ファイルはapps/runner/CMakeLists.txtのGUI専用実行体に追加する。core/gui.hppのAPI変更はしない。設計文書160の日英のPanelの行にイベント名と伝播の停止を記す。翻訳元情報を正しく更新するため、レビューで見つかった既存のstage_issue_memos.py説明1行の未訳も英語側へ追随させる。
5. 対象8課題の対応内容、AI判定を更新する。statusは対応中のままとし、解決は選ばない。

## テスト

Public interface / surface: ReplayBuffer構築、RewardScaler::Scale、ObservationNormalizer::NormalizeAndUpdateStats/Normalize/GetScalar、MuZeroReplayBuilder::ProcessQueue・MuZeroUniformSampler::Sample・MuZeroReplayBuffer::Push、BatchUpdateResult::GetScalar、InitRL、ComputeQValueHeatMap、wxEVT_CHAR_HOOKからEVT_FORWARDED_KEYへの配送。


優先 behavior（各項を1つずつGREENまで閉じる順）:

1. Replayの構築: priority store/completerの片方だけ2通りは指定の文言で例外、両方あり/なしは構築成功（tracer bullet、A1）。
2. Scaler: gammaが0.5でないauto post scale、robustのwarmup到達直後と行単位の除外、正の非単位scaleのSymLogが手計算値になる（A2）。
3. MuZero: done/truncatedで残りの割引価値とキューの排出、終端またぎ/記録末尾のmaskと価値0、lane別Push/Sampleの分離を確認する（A3）。
4. BatchUpdateResult: 43直接キーと8群の成立/未成立・未知キー・8異常経路の現行値を移動前後で保持し、キー重複をcompile時、綴り違いをテストで検出する。ヘッダは`#pragma once`が1件、旧先頭名`rainbow_agent_impl`が0件で、Debugビルドが成功することを確認する（A4/A5）。
5. InitRL: 3キーそれぞれの不正な明示deviceを拒否し、未指定は通る。CUDAを無効にした別processで明示cudaの一括報告も通す（A6）。
6. ステージ1のRunner: 4環境それぞれの変更前後Runで時間/資源以外の全scalarのtag/step/valueが一致する（A7）。
7. Q-Values: 元Updateの28条件を先に手計算と照合し、window無しの加工1ケースから全表示組合せ・平滑化・一定値・累積範囲へ広げる。共有入力不変、コピー1回、Grid差分条件を保持する（A8）。
8. キー配送: 入れ子の子window/内側Panel自身/直下/Panel外で転送元と回数1/0、元イベント未処理を移動前後で保持する。指定3GUIタグと人による表示/キー/フォーカスの確認も行う（A9、A8の人による確認を含む）。

TDD順序: 構築の不変条件1件をtracer bulletとする。テスト不足の解消では現行実装でGREENを確認し、PRD指定の一時的な破壊でRED、その破壊だけを戻してGREENを記録する。次の挙動へ進む前にその1件のサイクルを閉じる。GetScalarとキー配送は変更前に特性テストを通し、本体変更後も同じテストを通す。加工は元Updateの出力を先に採取し、1ケースの関数テストのRED→最小の切り出し→GREENから広げる。RED中はリファクタリングしない。

## 検証

- MSVCはAGENTS.mdのVsDevCmd.bat経由でDebug全体、GUI実行体をビルドする。
- ステージ1変更前に現在ソースの全体Debugビルドを通し、RunnerとDLL・configを `.scratch/prd122/baseline/` へ保存する。旧Runと既存PRDの基準は上書きしない。
- 変更前/後の4環境（GridMaze/LunarLanderはCPU、Atari qbert/DropMergeはCUDA）を逐次実行。PRD098/PRD109の既存ドライバを調べ、現行configで動く同じ条件の手順をPRD122の検証資材に置く。seed1・8lane・batch16・warmup256・RR1・8704experience・評価停止・FP32・決定的backendを固定する。時間/資源3prefix以外の全scalarをtag/step/valueで完全照合し、点数・checksumを残す。
- 既定core suiteと[shipped_config]の着手前/後の失敗集合を比較する。既知失敗も実測して記録する。
- CUDA_VISIBLE_DEVICES=-1で[prd070]を別processで流す。環境変数はprocess内で閉じる。
- 各破壊検査は1箇所ずつ行い、REDのテスト名・変更行・復元とGREENを記録する。既存変更は復元対象にしない。
- GUI既定suiteと[prd105_ui_keys]・[eval_steps_menu]・[prd109_ui]をそれぞれ独立processで流す。
- A8/A9の人によるRunner確認は自動検証と別に残す。未実施なら受入済みとはしない。
- 品質確認観点impl.mdの全項目を適用範囲で自己点検し、独立コンテキストのanet-review-agentへ計画と実装を依頼する。差し戻しは修正と再レビューを行う。


### 実行コマンド

リポジトリルートのPowerShellから実行する。コードフェンスはAGENTS.mdの実行ボタンの指定に従いbashを使うが、中身はPowerShellである。以下は今回使った実行体・引数を具体化したもの。標準出力と標準エラーを日時付きの別ログへ保存し、既存の証拠を上書きしない。実行後の終了値は`$LASTEXITCODE`、通常ログは`.scratch/prd122/`に残る。長い既定suiteとRun比較は、保存済み結果があり今回のinclude順だけの修正では繰り返さない。

Debug全体（MSVC初期化経由）:

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug' *> (Join-Path '.scratch/prd122' ('debug-build-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'.log'))
```

既定では作られないGUIテスト実行体:

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target AnetRLRunner-gui-test' *> (Join-Path '.scratch/prd122' ('gui-build-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'.log'))
```

coreのPRD122関連テストと既定suite（各別process）:

```bash
& .\core\anet-core\bin\Debug\anet-core-test.exe '[prd122]' --reporter compact *> (Join-Path '.scratch/prd122' ('core-prd122-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'.log'))
```

```bash
& .\core\anet-core\bin\Debug\anet-core-test.exe --reporter compact *> (Join-Path '.scratch/prd122' ('core-default-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'.log'))
```

出荷設定の既定環境は既知のoneDNN assertionで停止するため、その結果も記録した。全件を通した再検証はprocess内だけOMP/MKLスレッド数を1にした次の条件である:

```bash
cmd /s /c 'set "OMP_NUM_THREADS=1" && set "MKL_NUM_THREADS=1" && core\anet-core\bin\Debug\anet-core-test.exe "[shipped_config]" --reporter compact' *> (Join-Path '.scratch/prd122' ('shipped-threads1-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'.log'))
```

CUDAを無効にした別processのdevice検査:

```bash
cmd /s /c 'set "CUDA_VISIBLE_DEVICES=-1" && core\anet-core\bin\Debug\anet-core-test.exe "[prd070]" --reporter compact' *> (Join-Path '.scratch/prd122' ('device-cuda-hidden-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'.log'))
```

GUIのPRD122関連テスト、既定suite、指定3タグ（各別process）:

```bash
& .\apps\runner\bin\Debug\AnetRLRunner-gui-test.exe '[prd122]' --reporter compact *> (Join-Path '.scratch/prd122' ('gui-prd122-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'.log'))
```

```bash
& .\apps\runner\bin\Debug\AnetRLRunner-gui-test.exe --reporter compact *> (Join-Path '.scratch/prd122' ('gui-default-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'.log'))
```

```bash
& .\apps\runner\bin\Debug\AnetRLRunner-gui-test.exe '[prd105_ui_keys]' --reporter compact *> (Join-Path '.scratch/prd122' ('gui-prd105-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'.log'))
```

```bash
& .\apps\runner\bin\Debug\AnetRLRunner-gui-test.exe '[eval_steps_menu]' --reporter compact *> (Join-Path '.scratch/prd122' ('gui-menu-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'.log'))
```

```bash
& .\apps\runner\bin\Debug\AnetRLRunner-gui-test.exe '[prd109_ui]' --reporter compact *> (Join-Path '.scratch/prd122' ('gui-prd109-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'.log'))
```

4環境比較は着手前に凍結したbaselineを使い、日時付きの新しい出力先へ保存する:

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd122/run_validation.py --baseline .scratch/prd122/baseline --output (Join-Path '.scratch/prd122' ('compare-recheck-'+(Get-Date -Format 'yyyyMMdd-HHmmss'))) *> (Join-Path '.scratch/prd122' ('compare-driver-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'.log'))
```

## 前提

- 挙動、設定、ログ文言、メトリクスの識別子を変えない。例外はD13の上位Panelのデバッグログだけ。
- 既知キーの一部が現行で0等を返すことはG2に従って維持し、今回のリファクタリングで規約への変更を混ぜない。
- 新規CONTEXT/ADRの判断は不要。API名と型名はPRDのクラス一覧に従う。
- DLLは基準実行体とセットで保存する。比較Run名にはtmpを入れ、既存artifactを編集しない。
- コミット・ステージ・pushは行わない。

## 経過

- 2026-11-10: PRD、プロセス契約・レジストリ、関連コードとテストを照合。未決ブロッカー0。着手前Debug全体ビルド成功（ninja: no work to do）。基準コミットを固定し本計画を保存。

## 検証・レビュー結果

- 変更前Debug全体: VsDevCmd.bat経由のcmake --build --preset x64-Debugが終了値0（no work）。
- 基準保存: PRD109のsnapshotを--output .scratch/prd122/baselineへ実行し終了値0。53個のexe/DLL（約3.18GB）と設定・hashを保存。
- 基準suite: PRD109のsuitesを--output .scratch/prd122/suites-beforeで実行中。通常ログはcore/process.log、JUnitはcore.xmlへ保存し、完了時に失敗集合を採る。
- 不変条件テスト追加後のビルド: replay_buffer_test.cppのコンパイル成功。基準coreテストとリンクを重ねたため実行体のロックでLNK1168。基準suite終了後にリンクをやり直す。これはテストのREDではなく検証手順の失敗として扱う。
- GetScalar全キー・8異常経路の特性テストを本体変更前に追加した。本体はまだ未変更で、ビルドと実行は基準suite後に行う。
- PRD122固有の比較資材run_validation.pyとREADME.mdを追加。PRD098/109/084の公開された検証関数を再利用し、Run名にtmpを含める。

- 2026-11-11: 基準既定suite終了。core終了値0・失敗0（737.45秒）。GUI終了値42、既存の `Logs indent wrapped lines to the message on demand` だけが失敗（35.27秒）。`.scratch/prd122/suites-before/suites.json` が正本。PRDが挙げたattentionの既知失敗は今回は発生しなかった。
- 変更前の追加特性テスト: `[prd122] --reporter compact` が3ケース・524assertionすべて成功（`.scratch/prd122/characterization-before.log`）。GetScalarの43直接キーと8群、8つのsystem error経路をすべて通し、到達不能として省いた経路は0。
- 破壊検査の初回はPythonからcmdへの引用符の受け渡しでビルド起動に失敗。ソースはハッシュ一致で復元された。ドライバの起動方法を修正し、`replay_pair_guard_r2` で再検証中。初回の失敗はREDに数えない。

- A1破壊検査: replay_buffer_impl.cpp:1556の対の検査を無効化すると2ケースとも例外不在で失敗（終了値42）。検査を復元し再ビルド後、4通りの全assertionが成功。復元前後のSHA-256は一致。`.scratch/prd122/mutations/replay_pair_guard_r2/result.json`。
- GetScalarの本体・補助の移動と43行の表、8群の関数、ヘッダガードを実装した。固定値の表のキー重複はstatic_assertで止める。全キーの変更後の特性テストはビルド終了後に実行する。

- A2 auto_post_scale: scaler_impl.cpp:894の(1-gamma)をgammaに替えると変換値のテストが失敗（42）、式を復元すると成功（0）。前後ハッシュ一致。`mutations/reward_auto_post_scale/result.json`。
- A2 robust update: scaler_impl.cpp:495の外れ値の>を<に替えると正常行の採用件数と外れ値率が失敗（42）。復元して7assertion成功。前後ハッシュ一致。`mutations/robust_outlier_direction/result.json`。
- InitRLの変更前の特性テスト成功後、既定値の表を消してHasでキーの有無を問う形へ変更した。変更後の3キーの異常値テストは成功。
- 追加テストの集計でMuZeroの補助assertが2件失敗した。各step自身のdoneを返す実装なのに、先頭stepまでtrueと期待していたため。価値ターゲットの期待値はすべて一致。先頭の補助assertをfalseへ訂正し、本体は変えていない。この失敗は破壊検査のREDに数えない。
- GUI準備: Panel_key_test.cppに本体変更前の伝播特性テストを、QValuePanel_test.cppに元Updateの28条件の一時採取fixtureを準備。まだGUIターゲットへ追加しておらず、GUI本体は未変更。採取fixtureは元の値を控えたあと、windowを作らない関数テストへ置き換える。

- coreの追加テストは10ケース・1522assertionが成功（`.scratch/prd122/core-characterization-complete.log`）。done/truncatedの値、窓のmask/価値、lane別Push/Sampleも通った。
- A6: CUDA_VISIBLE_DEVICES=-1の別processで[prd070]が4ケース・37assertion成功、skip0（`device-cuda-hidden.log`）。
- [shipped_config]初回はoneDNN内部のnthr_==nthr assertionで終了値3。これはCatchの失敗ではなく完走失敗として残す。OMP_NUM_THREADS=1/MKL_NUM_THREADS=1をprocess内だけに置いた再実行で12ケース・2604assertion成功（`shipped-config-threads1.log`）。環境の条件を変えているため、既定環境の成功とは書かない。
- A2のwarmup境界（463行 < を <=）とSymLogの掛け戻し（404行 *scale を除去）でも、Catchの失敗42→復元後0を確認。復元後のscaler_impl.cppは元とハッシュ一致。各`mutations/*/result.json`に行・コマンド・終了値とhashがある。scalerのproduction差分は0。
- GUI専用実行体へ2つの新規テストを追加するCMakeの行を更新した。GUI本体の加工とPanelはまだ変えていない。

- A3打ち切り長: muzero_rb.cpp:104を1短くするとdone/truncatedの価値4チェックが失敗（42）、復元すると16assertion成功（0）。A3mask: 232行のpadding maskを1にすると170チェックが失敗（42）、復元すると645assertion成功（0）。どちらもMuZero productionのハッシュが元と一致し、production差分は0。
- 検証中: `.scratch/prd122/run_core_remaining.py` が残るGetScalarの綴り/重複を逐次実行している。完了前に同じ出力先で再起動しない。
- GUIの28条件の手計算を `.scratch/prd122/qvalue_expected.py` とwindowを作らないテスト案 `.scratch/prd122/QValuePanel_unit_test.cpp` に準備。原Updateの一時出力の挿入/除去は `qvalue_capture_instrument.py`、切り出しは `extract_qvalue.py`。まだこれらのGUI本体変更は適用していない。採取→手計算照合→1ケースのRED→切り出し→GREEN→全条件の順で進める。

- coreの全9箇所の破壊検査を完了。`core-mutation-summary.json`に一覧を保存。各箇所は元とハッシュ一致で復元し、復元後のビルド/関連テストはすべて0。綴り違いはCatch 42、重複キーはstatic_assertのC2338（指定の診断文を確認、build 2）を検出した。
- A4/A5/A6静的検査も成功（`core-static-checks.json`）。pragma once 1、旧先頭名0、ヘッダの文字列比較1、GetScalar宣言5、表43キーすべて一意、43キーの直接比較0、device既定値の対0/Hasあり。MuZero/scaler production差分0、core変更ファイルは全てLF。
- 設計160の日英のPanel行とPRD122の検証資材リンクを追記した。今回の対応する2箇所とリンクを照合し英語側のblob/dateを更新した。最終本文とblobの現物照合は後述。

- ステージ1のDebug全体とGUI専用targetをVsDevCmd.bat経由でビルドし終了値0（`stage1-build.log`）。Runnerと5つのEnvテスト実行体も再リンク済み。
- 復元後のcore追加テストは10ケース・1523assertion成功（`core-stage1-final.log`）。成立するReplay2通りはPER/uniformの完全なfactory構成で確認し、片方だけの2通りの直接コンストラクタ検査は維持した。robust updateは平均(0+10)/2とsqrt(50)+epsilonも明示して確認した。
- A9の変更前特性テストは23assertion成功（`panel-keys-before.log`）。まだPanel本体は変更前。
- A7進行中: `.scratch/prd122/compare-stage1/`。GridMazeは前後26946点・79tagが完全一致しSHA-256は2278b59c28b93e85d1adbfe203a1548a60589715fbdf576b029b1c9b3d800f98。実行体・入力・終了値は各sideのexecution/inputs.json、全scalarと除外一覧はnormalized.summary.jsonに記録。
- GridMaze変更前のmodule dumpで確認した実効設定を `gridmaze-effective-config-all.json` に保存。Agent=DefaultDQNAgent、agent/env=cpu、torch threads=1、seed=1、8 lane、batch16、warmup256、RR1、AMP=false、exp_exit_step8704、deterministic_algorithms=true/warn_only=falseを確認。

- A7中間結果: LunarLanderは24399点・84tag（SHA-256 941fa28b97b6a5833504c7a7b314bbc9966e77d6f66ff5edb8e4487e2a5a5a6c）、Atari qbertは23185点・80tag（5c7854af8296044d7aa52433138632ed187af926398a46c0e63e128f645f2e12）が前後完全一致。DropMergeは現在実行中。

- A7完了: DropMergeも31152点・129tag（6cd959f2df2406a3c32d8371eaed17fddef5e50bcd8e19dec976fa8f2da672c9）が完全一致。4環境合計105682点の全点が前後一致し、ドライバ終了値0。8本のmodule dumpをinspect_run.py configで採り、学習条件・AMP無効・device・qbertを確認。評価停止は各Runのconfig/config_data.txtの解決済みeval_schedule 3件が全てinterval=0で確認。effective-input-audit.jsonに保存。
- ステージ1既定GUI suiteは変更前と同じLogsの1ケースだけが失敗（42）。suites-stage1/gui.xml。core suiteは実行中。GUI本体変更前の採取用一時計測を挿入して専用targetをビルド中。

- A8切り出し前採取: 元Updateへ同じ入力を与えるhidden fixtureは126assertion成功。qvalue-before.jsonlの28条件（Histogramの全8組合せ×平滑化2、raw2、一定値4、累積範囲2×3）を手計算と照合し全一致、最大絶対誤差3.4177393892775854e-7。qvalue-hand-check.jsonに各期待値を保存。一時計測の除去後は元QValuePanel.cppのSHA-256 b912e398f6e0d7b52034915069992341bc86a3ecd8b5aa737d9787d35988afe9とバイト単位で一致。採取fixtureもscratchへ控え、window無しの最初の関数テスト1件へ置換した。

- ステージ1既定core suiteは終了値0、失敗0、705.60秒。GUIは基準と同じLogsの1件のみで、new-failures.jsonはcore/guiとも空。suites-stage1/suites.jsonと各JUnit・通常ログに保存。

- shipped_configの停止は既存の[issue-20261109-fa2f69](../management/issues/issue-20261109-fa2f69.md)と同一のテスト・nthr_==nthr assertion・終了値3であることを照合した。既存課題にはPRD119変更前からの再現が記録されている。今回のOMP/MKLスレッド数1での成功は回避条件の観測であり、原因や修正を確定したものではない。
- GUI既定suiteの変更前失敗はLogPanel_test.cpp:375の2行目先頭位置のalignedチェック。起票済み課題に同じ事象は見つからなかった。範囲外の候補として保持し、今回は修正しない。

- A8 tracer: 新API未定義のコンパイルRED（qvalue-tracer-red-build.log、終了値2、QValueHeatMapRangeState/ComputeQValueHeatMap未定義）→元の式の切り出し→window無し1ケース13assertionのGREEN（qvalue-tracer-green.log、終了値0）。次に全条件のテストへ広げた。
- GUI限定の反復検証では、通常のNinjaグラフをscratchへ写し、GUIテストからRunner本体へのorder-only依存2本だけを除いて使う。コンパイル・リンク規則、テスト対象のソース・ライブラリ依存は変更しない。outのグラフは編集しない。最終の検証では通常のCMake presetによる全体/GUIビルドも行う。
- A8静的照合: 元のGridブロック全体が移動先に1箇所そのまま残ることを照合。vvのコピーはQValuePanel.cpp:426のresult初期化1箇所。選択行の差分更新は708行、列幅の再適用条件は731行（最終行番号は後で照合する）。

- A8全条件のwindow無しテストは4ケース・422assertion成功（qvalue-full-green.log）。採取した元Updateと同じ28条件の配列・guide・累積範囲を確認。続いてA9のOnKeyの型照会を除去し、SkipとStopPropagationへ置き換えた。

- A9変更後: 変更前と同じPanel_key_test.cppの4場面・23assertionが全成功（panel-keys-after.log）。gui.cpp/gui.hppのproduction dynamic_castは0。転送イベントの型・中身、OnMouse、RunnerFrame::OnKeyは変更していない。
- PRD §5.3を再照合。wiki4ページの成立の経緯・不採用案・変更条件はPRD改訂時に書き戻し済みで、実装の移行対象ではない。本文は変更せず、課題判定後に派生区画だけ作り直す。

- 英訳元blobの最終検査で、記録値が最終の日本語本文へ追随していないことを検出した。対応する本文2箇所とリンクを再照合し、anet-translate-docsが指定するgit hash-objectの現物値0719ffdc1327535860f5bfce57eccd58ad50a0e8へ翻訳元情報を訂正した。既存のBOMの有無は保持した。

- A8破壊検査2件完了: Advantageの引き算を外すと80assertion失敗、両端binの除外を外すと44assertion失敗（いずれも42）。それぞれ復元後422assertionが成功（0）、前後のソースSHA-256が一致。gui-mutation-summary.jsonとmutations/qvalue_advantage・qvalue_edge_binsに記録。
- A9のStopPropagationを外すと、入れ子で転送が2回になり、8assertion失敗（42）。復元後のビルドとGREENを実行中であり、完了結果は後述。

- A9破壊検査完了: gui.cpp:101のStopPropagationを外すと転送2回・8assertion失敗（42）、復元後23assertion成功（0）。ソースの前後SHA-256が一致。これでGUI3箇所、core9箇所の全12件を完了（gui-mutation-summary.json、core-mutation-summary.json）。
- 自己点検の直せる不足を整理: テストのLinear識別子の共有を名前付き定数へ、CMake変更ファイルをLFへ揃えた。検証ドライバの説明コメントも日本語へ統一し、--helpが成功。挙動と設定は変えていない。
- 対象8課題すべての対応内容へ、今回の実装・確認結果・本記録へのリンクを追記した。GUI2件にはA8/A9の人による確認が未実施と明記した。statusは対応中のまま。

## 最終の受入確認

| 受入 | 結果と証拠 |
| --- | --- |
| A1 | 対の4通りと指定の文言を確認。pair guard破壊のRED→復元GREEN、hash一致 |
| A2 | auto post scale・warmup境界・行単位の除外・SymLogを確認。指定4箇所のRED→復元GREEN |
| A3 | done/truncated価値、終端/記録末尾のmask、lane別Push/Sampleを確認。指定2箇所のRED→復元GREEN |
| A4 | pragma once 1、旧先頭名0、Debugビルド（最終結果は後述） |
| A5 | 全認識キーの成立/未成立と8異常経路、43行一意、header比較1、宣言5。綴り/重複のRED→復元GREEN |
| A6 | Hasで明示値のみ。CUDA無効の別processで4ケース37assertion成功・skip0 |
| A7 | 4環境8本、105682点がtag/step/value完全一致。各checksumと実効入力は前記 |
| A8 | 元28条件と手計算一致、window無し422assertion、2破壊を検出・復元。コピー426行、行選択708行、列幅731行。2026-11-11に開発者が表示・切替・選択・クリックの実機確認を「問題なし」と回答 |
| A9 | 同じ特性テストが前後23assertion成功、StopPropagation破壊で転送2回・復元GREEN、production RTTI0。3GUIタグは別processで全成功。2026-11-11に開発者が実キー・フォーカス・クリック・Tabの実機確認を「問題なし」と回答 |
| A10 | ステージ1の全体Debug・core既定成功、GUIは基準と同じ1件。ステージ2の通常全体Debug/GUIビルドは成功。既定core失敗0、GUIは基準と同じLogs1件、3GUIタグとprocessスレッド数1のshipped_configは全成功。コメント修正後の再ビルド/関連テストは後述 |
| A11 | 8課題の対応内容を追記済み。A8/A9の実機確認結果を受け、AI判定は全8件が残件無し（2026-11-11）。statusは全て対応中のまま。派生区画の再生成結果は末尾に記録 |

- 最終静的確認はfinal-contract-checks.json、改行/BOM/禁止記法/重複リテラルはquality-mechanical.json。直訳の避ける語は全変更の追加行で0、共有契約のanet/PRD検索も0。課題8件の追記は再読込・Git差分・hash/mtimeで保存を確認した。

## 品質確認観点の自己点検

impl.mdの全項目を、その適用範囲に従って通した。必要な自動検証と、人によるA8/A9の実機確認が完了した。

| 項番 | 自己点検 |
| --- | --- |
| 1〜7・62・71・72・76 | header using namespace、production dynamic_cast、uppercase TODO、共有契約の検索は0。計画名/対のPRDは適合。BOM差異0、CRLF0、重複リテラル候補0、直訳の避ける語0。外部契約の改名は無い |
| 8・53 | skill変更無しのため対象外 |
| 9〜13 | 前提・公開API・TDDの順・PRDとの対応・MSVC経由の検証を計画へ固定。追加資材の理由も主な変更/検証へ記録 |
| 14〜17・74 | A1〜A11の対応は上表。自動検証と人によるA8/A9の確認結果を区別して記録。8課題へ対応を追記済み。範囲外の修正は混ぜていない |
| 18〜22 | 既存の設定/API/メトリクス/保存形式の置換無し。新しい加工関数と型はPRDの指定範囲。互換層無し、過去のRun/履歴を変更していない |
| 23〜28 | 既存の値域・エラー・WARN・例外を保持。deviceの明示値検査とcuda一括報告も保持。加工の契約外入力と既知の不具合へ検証/fallbackを足していない |
| 29・52・54・67・68・70 | 現用設定ファイルとbatの変更無し。検証資材が既存のPRD098設定生成を再利用する理由と実行入力はA7へ記録 |
| 30〜35・55〜60・63〜66・69・75 | 既存の機能グループへ配置。新規は指定のGUIテスト2本、全キー照合を独立させるScalarテスト、PRD検証資材。補助型は関数内、集成体は指示付き初期化、既存includeの順を維持。spanは同期呼び出しの間だけcallerの重みを借りる旨を行末に記載。可変global/新UIロック/不要な取り出し口無し。名前はPRD/既存Computeの語義、初期化順は保持 |
| 36〜41・45・61 | 新コメントは日本語、移動元の意図は移動先に保持。Update/加工/Grid/描画とGetScalarに計測。既存の同期転送/レイアウトの頻度を増やさず、コピーは1回。ログ文言は保持、D13が許す親Panelのdebugの減少だけ |
| 42〜44 | GetScalarは特性テストでNaN/nulloptとPRD G2の現行例外を保持。遅延cache/Resourceの所有と依存方向は保持。範囲StateはPanelが所有 |
| 46・47・73 | 公開API・runner-facingイベントを通して確認。テスト入力/回数/拒否条件とA1〜A9の対応を照合。加工はwindow無し。本番設定の新しい固定値テストは足していない |
| 48〜51 | 通常MSVCの最終全体/GUI Debugビルド成功（stage2-final-build.log、終了値0）。関連テストと12破壊検査、A7比較は完了。ステージ2既定/隠しGUI/出荷設定suiteも完了。通常ログとJUnit・実行情報をscratchへ保存。設計160日英の対応2箇所とリンク、最終source blobを照合。PRD固有資材をtestdata/prd122へ置き設計160のテスト章から参照 |

- 人による確認: 2026-11-11の開発者回答「問題なし」でA8/A9の実機確認も完了。自動テストによる代替ではなく、開発者の確認結果を受入証拠として記録した。

- 独立レビューを開始した（担当prd122_review）。対象は計画と今回の実装18ファイル、観点impl.md。追跡済み13ファイルの差分は.scratch/prd122/diff-20261111-0233/、新規5ファイルと計画は全文。無関係なdirty変更は対象外。最終スイート実行中のため、結果を送るまで最終判定を保留してもらい、読み込みとコード検査を先に進めている。

- 独立レビューの途中確認で、観点38（移動コメント）と50（翻訳元情報）の不足2種を受領。PERのCV/IS ESS/priority ESSの3箇所へ元の意味・対策・理想範囲のコメントをそのまま復元。C++のコメント以外の行が修正前と完全一致することを機械照合した。JP既存のstage_issue_memos.py説明1行がENに無いことも確認し、JPは変えず対応する英文を追加した。progress:doneと現物blobの整合を保つための最小追随である。
- コメント修正中も最終suiteの実行体は変更していない。完走後に通常Debug全体+GUIを再ビルドし、再リンク後の関連テストも確認する。実行中のexeは再リンクしない。

- ステージ2最終suite完走（ドライバ終了値0）。core既定は0/失敗0/704.30秒、GUI既定は42/基準と同じLogs整列1件/33.36秒。新規失敗集合は両方とも空。prd105_ui_keys（7.51秒）、eval_steps_menu（7.87秒）、prd109_ui（10.62秒）は各別processで0。OMP_NUM_THREADS=1/MKL_NUM_THREADS=1をprocess内だけに置いたshipped_configも0（112.08秒）。suites-stage2の各XML・通常ログ・suites.json/new-failures.jsonに保存。既定環境のoneDNN停止は既存課題に対応する未解消事象として保持する。
- 全suite終了を確認してからreview_final_validation.ps1を開始。コメント以外のC++行は変えていないため、重い全suiteを同じ内容で繰り返さず、通常Debug全体/GUIのビルドと関連するcore/GUIのprd122テストを再リンク後に確認する。

- 最終JUnitの詳細照合: GUIのLogs失敗はbefore/stage1/stage2とも同じcase・同じaligned(second_row_x("first"), x_of(ui.StartOf("first")))・LogPanel_test.cpp:375で、suiteの失敗数属性も3のまま。既存case内の悪化も無い。coreのskip1はCUDA可用時のPRD070明示cuda検査で、別process CUDA_VISIBLE_DEVICES=-1の4ケース37assertion/skip0で実行済み。

- コメント後の通常Debug全体+GUI再ビルドは成功（review-final-build.log、終了値0）。初回の関連テスト起動はPowerShellのErrorActionPreference=Stopがnative stderrの情報ログをNativeCommandErrorとして停止させた。CatchのREDに数えず、部分ログを保持して既存のPython executeへ変更した。
- 再リンク後の関連テスト: coreは10ケース1523assertion、GUIは5ケース445assertionが全成功（終了値はいずれも0）。各review-final-core-r2/gui-r2のexecution.jsonとprocess.log、review-final-results.jsonへ保存。GUIの集計を447と見込んだ検証補助のassertが止まったが、正しくは422+23=445であり、GUI本体のテストは0で成功している。期待件数を訂正して保存済み出力を再照合し、同じテストを不要に再実行していない。

## 最終レビュー結果

- 独立担当1名へ計画と実装を依頼し、途中の指摘2種を1回の差分更新で直して再確認した。最終判定は計画・実装ともOK（OK2、差し戻し0）。最終応答原文は.scratch/prd122/review-20261111-0257.md、途中応答も同じ置き場へ原文保存。
- 解消した指摘: 観点38のPERコメント3箇所の意図、観点50の既存未訳1行を含めた英訳元情報。根拠が無く見送った指摘は無し。追加の範囲外検出も無し。
- 再確認ポイント: GetScalarの現行例外はPRD G2/D7に沿う保持。レビュー時点では人によるA8/A9確認が未了だったため区別して記録し、その後の開発者回答で受入を完了した。
- レビュー直後の判定: core6課題は残件有り→残件無し、GUI2課題は残件有りを維持。ai_judgedはいずれも2026-11-11。status/監理の項目は変更せず対応中を保持。issue-judgments.jsonに前後値とhash/mtimeを保存。maps成功、16地図の手書き区画とwikiログが変わらないことをmap-handwritten-preservation.jsonへ保存。

## 最終の受入と課題記録

- 2026-11-11: 開発者がA8/A9の実機確認結果を「問題なし」と回答。Q-Valuesの表示と切替・選択・クリック、Liveの実キー、各paneのフォーカス、Tab移動の受入を完了した。
- GUIのissue-20261103-58bf49とissue-20261103-e088ebの対応内容へ人による確認を追記し、AI判定を残件有りから残件無しへ更新。これでPRD122の対象8課題すべてが残件無し。statusは対応中を維持し、解決へは変更していない。
- 工程末の候補から開発者が選んだLogs整列テストの事象を、[issue-20261111-61ad60](../management/issues/issue-20261111-61ad60.md)として起票した。原因未特定のため調査中とし、変更前・ステージ1・ステージ2の同じ失敗の証拠を記録した。PRD122が直す課題とは分けている。
- コミットの準備は「今回は準備しない」との回答に従い、実行しない。stage・commit・pushも行っていない。

## 完了報告

- 実施内容: core6件とGUI2件の実装、自動検証、自己点検、独立レビュー、人によるA8/A9の受入確認を完了した。
- 変更ファイル: core/runnerのコードとテスト、PRD122検証資材、設計160の日英、対象8課題と派生区画、本実装記録、新規のLogs調査課題。無関係なdirty変更を保持。
- レビュー結果: 計画・実装OK。途中の指摘2種を修正し、再確認済み。受入結果の追記と課題記録以外に、レビュー後のコード変更はない。
- 決まらなかったこと: PRD122の受入に残件なし。範囲外のLogs事象はissue-20261111-61ad60へ記録済み。
- 判断してほしいこと: なし。
- 次にできること: 今回はコミットの準備を行わず、ここで完了する。

- 受入反映後のmapsも終了値0。INDEX・RunnerApp・GUI共通基盤の派生区画を更新し、16地図の手書き部分とwikiログが変わらないことを再確認した。記録と課題を再読込し、hash/mtime、Git差分、index未変更を確認して完了。

## cc-reviewの追加確認

- 2026-11-11: 開発者の「指摘あれば修正して再レビュー」の指示で、同じ依頼をCodexとClaudeへ送った。Codexは計画/実装ともOK、Claudeは計画2件（優先behaviorの項と具体的コマンド）、実装2件（新規テストのinclude順とUpdate/Renderの既存用例）を指摘した。
- 計画に順序付きの優先behavior、実行コマンド、既存動詞の実物参照を追記。新規Scalarテストのincludeをnn.hpp→nn_impl.hpp→dqn_based_agent.hppへ並べ替えた。計算・分岐・テストの入力とassertion・GUIの実装は変更していない。
- 修正後の検証: 通常のVsDevCmd経由Debug全体と明示GUI targetのビルドは終了0。core[prd122]は10ケース1523assertion、GUI[prd122]は5ケース445assertionで成功。計画内12コマンドのPowerShell構文解析と実行体・baselineの存在確認も成功した。保存先は`.scratch/prd122/cc-review-20261111-0328/`のfix-tests.jsonと各ログ。
- 自己点検: impl.mdの全項目を適用範囲で再確認し、追加したbehavior順・具体コマンド・既存動詞の参照・基礎からのinclude順が一致することを確認した。BOM差異・CRLF・追加C++の重複リテラルは0。今回の修正は計算や分岐に触れないため、既定suiteと4環境比較には保存済みの実測結果を用いる。
- 計測の再確認: profile.hppのANET_PROFILE_FUNCはProfileRangeへ接続し、profile.cppはTracy/NVTXのzone/rangeを出す。scalarのtagを登録・出力する経路は無く、GetScalarのキー契約は変わらない。
- 追加の再レビュー原文は同じ保存先のreview-20261111-0328.md（Codex）とreview-20261111-0328-claude.md（Claude）へ保存する。

- 第2回レビュー: Claudeは計画/実装ともOK。Codexが優先behavior 4の重複include成功という記載に対応する検証が無いと指摘したため、PRD A4と実施済みの静的確認・Debugビルドに合わせて記述を修正した。コード/テスト条件は変更せず、既存のビルドとテスト結果を引き継ぐ。

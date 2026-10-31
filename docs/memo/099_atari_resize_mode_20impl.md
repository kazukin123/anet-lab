# Atari縮小方式の選択 実装メモ

## 概要

正本は[PRD099](099_atari_resize_mode_10prd.md)。AtariEnv.resize_mode=adaptive_average（既定）/pixel_areaを実装し、Reset/Stepと評価への伝播、実効設定・構築ログを揃える。通常の実行体を保持し、今回の作業ツリーを変更前の基準とする。基準HEADは790afd45a874a2c2990f4424ad5e2aaf85b54917で、関連ファイルの編集前コピーも.scratch/prd099/implementation-20261023/beforeへ保存する。既存のAtari.txtの未コミット変更を保持する。

## 未決事項監査

- ユーザー判断が必要なブロッカー: 0。
- repo evidenceで解決: AtariEnvConfigの通常のReadConfigが共通値→評価個別値の順で読み取る。AtariEnvFactoryが同じ設定型を構築し、CaptureObservationとStepがCPUで縮小してdeviceへ渡す。ConfigDataの整数読み取りは既存の厳密変換を使う。
- 2ximplへ固定: 下記の縮小器、局所検証、TDD順序、短い決定的Runの比較条件。内部APIの置換は現用呼び出しもまとめて移行する。
- PRD範囲外: OpenCVリンク、既定切替、拡大対応、他の前処理・Agent・Runnerの変更。性能受入5・表示を含む受入6・長いQbert学習比較は測定可能な環境で行い、未実施を明記する。

## 主な変更

### 縮小器とEnv

- AtariPreprocessへGrayscaleResizerを追加。コンストラクタは入力高さ・幅、出力サイズ、方式を受け、Resizeは借用したuint8入力から所有する[1,S,S] uint8 Tensorを返す。設定と重みは生涯固定であり、途中で差し替える入口は設けない。
- 現行のResizeGrayscale free関数は縮小器の現行方式へ移し、interpolate(float32, area)→squeeze→round→clamp→uint8→contiguousを保持する。古い関数と呼び出しは削除し、互換overloadは残さない。
- pixel_areaは各軸で重なる元画素と整数の重なり幅を構築時に準備する。出力の境界をS倍した整数格子では、出力区間は[d×source_size, (d+1)×source_size]、元画素は[i×S, (i+1)×S]となり、重なりを誤差なく表せる。重みはintで保持し、横の重み付き和→縦の重み付き和をint64_tで積算する。途中の除算・丸めは行わず、最後にsource_height×source_widthで割る。各軸の重みの和がsource_sizeなので、この除算が正規化を兼ねる。Atariの分子の最大値は255×210×160=8,568,000である。入力画像を一時Tensorへコピーしない。作業領域はResize呼び出しごとに所有して共有可変状態を増やさない。
- 最後の偶数丸めは整数の商・余りと商の偶奇から明示的に行い、浮動小数誤差と環境の丸めモードに依存させない。非負の重みで正規化するため出力は0〜255の範囲になる。
- AtariEnvConfigへresize_modeを追加し、設定読み取り時に既知の2値を検証する。評価個別指定のキーもエラーで識別できるようにする。screen_sizeの既存の正値検証を維持し、pixel_areaにだけ1〜160の制限を加える。
- AtariEnvが縮小器を所有する。既存のValidateConfigを通してから構築時に一度生成し、Resetの前に準備する。CaptureObservationとStepの両方で使用する。各Env構築時に名前のprefixを伴うresize_modeの英語verboseログを1回出す。

### 設定・文書・資材

- Atari.txtへAtariEnv.resize_mode ?= adaptive_averageとrun.@pixel_area : AtariEnv.resize_mode = pixel_areaを局所追加する。
- PRD099のクラス一覧どおり、AtariEnvConfig/AtariEnvを変更し、GrayscaleResizerをAtariPreprocess機能グループへ置く。ADR・用語集は変更しない。
- core/anet-core/testdata/prd099/へOpenCV正解、生成スクリプト、検証harnessとREADMEを置く。Qbert/Breakout/Battle Zone各8枚。入力と期待画像のchecksum、版・seed・採取条件を記録する。
- Atari設計書の日本語・英語の設定、前処理、テスト章を実装と同時に更新する。AGENTS.mdや過去Run・goldenは変更しない。

## テスト

Public interface / surface: AtariEnvFactory/AtariEnvConfig、EnvのReset/Step/GetConfigData、GrayscaleResizerの観測生成、ConfigResolverによるRun設定。

優先behaviorとTDD順序（各項目でRED→最小実装→GREENを完了してから次へ）:

1. tracer bullet: 工場からEnvを構築し、縮小方式が実効設定に現れ、Reset観測が既存のshape/dtypeを持つ。まず既定の方式の記録を確認する。
2. 非整数比の合成画像でpixel_areaが面積平均になる。既存free関数を縮小器へ移し、現行方式の既存goldenも同時に移行して維持する。
3. 偶数丸めと途中丸めなし、210×160入力のS=1/84で厳密に半整数になるケース、定数画像・整数比・画素境界、S=1/84/160、OpenCV固定24枚との各画像の許容差を検証する。重みの非負・和1は、privateの検査口を増やさず、定数保持と入力範囲保持・独立な参照計算で確認する。
4. 未知・空の方式名、不正型・screen_sizeの範囲外で失敗し、キー・値・期待値が出る。評価個別上書きと実効dumpを検証する。
5. 通常の設定継承、Runプロファイル、工場のReset/Stepで選択方式が実際に適用されることを確認する。テスト用のproduction APIを増やさない。

既存Atariテスト一式を回帰確認する。追加behaviorのために1件ずつテストを足し、GREEN後に整理する。

## 検証

1. 変更前の関連ファイル、HEAD、通常の実行体のhashを記録する。既存の.scratch/prd094/isolate.cmakeと同じDEFER方式で、CMAKE_PROJECT_INCLUDEを一時注入し、実行体の出力だけ.scratch/prd099/bin/<Config>へ向ける。CMakeの既存出力を上書きしない。
2. VsDevCmd.batをcallしたcmd内からcmake --preset、cmake --build --preset x64-Debug/x64-Release --parallel 1 --target AtariEnv-test AnetRLRunnerを使う。まずdry-runで差分範囲を確認する。新規ビルド木・clean・全再ビルド・ninja直接実行はしない。
3. 変更前に隔離Releaseの決定的な短いAtari Runを2回採取する。同じseed、pipeline Runner、CPUの小さい学習設定、同期または停止した定期評価、一定の経験数を使用し、再現性を確認する。設定一式と実行体を固定する。予算は16,384経験を既定とし、学習更新を含むことを確認する。
4. TDDのRED/GREENは隔離DebugのAtariEnv-testで実行し、ログと終了コードを.scratch/prd099へ保存する。サイクルごとの対象テストを明記する。
5. OpenCV正解は既存のWSL BTR venv（cv2 4.8.1）で生成し、入力と期待値を同じ固定ファイルとしてC++テストから読む。版が異なる場合は生成を止め、事実を記録する。
6. Release更新後にキー省略・adaptive_average明示の短いRunを採取し、既存のscalar/trace比較ロジックで変更前との全点checksumを比較する。除外は時刻・90_perf/・資源使用量だけに固定し、学習値の除外を追加しない。
7. 設定の変更前後で全キー・文字列値を比較する。通常設定では追加のresize_modeだけ、run.@pixel_areaではその方式差だけを許す。check_default_leaves.pyと既存[config]テストを使い、意図しない既存値の変化を確認する。
8. CMAKE_PROJECT_INCLUDEの一時指定は必ず外し、構成だけを戻す。戻した後に通常出力先へbuildはしない。通常の実行体hashが不変であることを確認する。
9. PRDの60,480寄与/lane・64laneで3,870,720寄与はCPU総仕事量の見積もりで、elapsedへ直接置き換えない。時間単価が不明なので内部方式の採用確認として隔離された短い縮小単体のA/B試作測定を行い、計算量と測定結果を記録する。Run全体の受入5は競合する重い処理があるため判定保留にする。

## 前提

- PRD099の名前・対応範囲・偶数丸め・各画像差1以下かつ99.9%以上一致・速度低下2%の契約を変えない。
- 実装の内部方式は数値テストを満たすまで確定しない。許容差を自動的に緩和しない。
- ユーザーが承認したPRDに沿う内部API・検証順の具体化であり、新たなユーザー判断はない。
- TDD・短い受入のみを逐次実行し、既存Runを停止・再起動しない。全再ビルドをせずMSVCはparallel 1とする。
- git add/commit/pushは行わない。

## 経過と結果

- 2026-10-23: 監査ゲートを通過し、本計画を保存。以下へ実行コマンド、RED/GREEN、検証結果、未実施事項とレビュー結果を追記する。
- 2026-10-23: 変更前の関連8ファイル・設定一式・通常の実行体hashをimplementation-20261023/beforeへ固定。Debug/ReleaseをCMAKE_PROJECT_INCLUDEで隔離し、既存ビルド木の差分リンクのみで変更前実行体を準備した。
- TDD 1 RED: red1-test.logで工場経由の実効設定にAtariEnv.resize_modeがないことを確認（1 test case失敗、4 assertions中1失敗）。設定読み取りと構築ログだけを追加し、green1-test.logで1 test case・5 assertions合格。
- golden-generation.log: WSL BTR venvのOpenCV 4.8.1・ALE 0.9.0で3ゲーム各8枚を生成。画像は975,744 byte、checksumと環境情報をmeta.jsonへ固定した。
- TDD 2 RED: red2-build.logでGrayscaleResizer未宣言を確認。縮小器とEnv所有を実装し、green2-test.logで合成画像・既存前処理・既定記録の7 test cases・28 assertions合格。
- golden-test.log: 固定24枚・169,344画素が完全一致（max_error=0、mismatches=0、53 assertions合格）。許容を緩めていない。
- baseline-runs.log: CPU・1 torch thread・16,384経験の変更前Runが2本完了（209.1秒、198.0秒）。baseline-comparison.jsonでscalar/trace 16,662点・定義・WARN・既存設定値が一致した。
- TDD 3 RED: 未知・空の共通/評価方式がConfig読み取りで拒否されない4件の失敗を確認。読み取る両宣言と直接組み立てたEnv設定の方式を検証し、green3-test.logで4 cases・66 assertions合格。
- TDD 4 RED: screen_size=161と評価個別の0/-1/161の診断キーが不足する4件の失敗を確認。型検証は既存ConfigDataを使用し、正値・pixel_area上限の診断を追加。green4-test.logで5 cases・74 assertions合格。
- 設定編集前の既定葉監査: default-leaves-before.log、28,053 assignments・3,491 audited・errors=0。
- TDD 5 RED: red5-test.logでrun.@pixel_area未定義を確認。Atari.txtへ弱い既定とRunプロファイルだけを追加し、green5-test.logで6 assertions合格。評価個別上書きと未指定時の継承も確認した。default-leaves-after.logはerrors=0。
- boundary-test.log: 半整数・途中丸めなし、独立な整数格子の面積計算、S=1〜160の全サイズ、定数・入力範囲、既存方式のS=161、整数倍平均と結果の所有の548 assertions合格。
- golden-reproduction.log: 別出力先への再生成で固定入力・期待画像のSHA256が一致。生成器は既存固定ファイルの上書きを拒否する。
- env-test.log: 公開factory/Reset/Stepで同seed・同行動のRGBと終端は一致し、学習と評価個別設定で観測が変わることを確認（177 assertions合格）。
- final-build-debug.log / final-build-release.log: VsDevCmd経由のcmake --build --preset x64-Debug/x64-Release --target AtariEnv-test AnetRLRunner --parallel 1に成功。既存ビルド木の差分のみを使用し、実行体は.scratch/prd099/binへ出力。
- atari-debug-all.log: DebugのAtari全体47 cases・1,831 assertions合格。prd099-release.log: ReleaseのPRD099 8 cases・805 assertions合格。両方のgolden比較でmax_error=0、169,344画素が完全一致。
- core-config-tests.log: 変更していない設定基盤の既存Debugテストを上書きせず実行し、122 cases・2,628 assertions合格。
- resize-probe-release.log: 縮小単体の試作測定は5組×200回。重み準備36.3µs、速度比中央値1.17328。組ごとの揺れがある競合中の試作値であり、64 envの性能受入の合格とは扱わない。2段階計算の実際の寄与回数は横50,400＋縦21,168＝71,568/lane、64 laneなら4,580,352。PRDの60,480は直接2次元計算のモデルであり、変更しない。
- 自己点検: 既存8ファイルのBOM保持、テキスト13ファイルのUTF-8/LF、Markdownリンク、goldenのSHA256、git diff --checkを確認し問題なし。Pythonの構文検査にも成功。生成したpycacheは作業ファイルとして除去した。

## 検証結果のまとめ

| 項目 | 結果 | 記録（.scratch/prd099/以下） |
| --- | --- | --- |
| Debug差分ビルド（追加指摘対応後） | AtariEnv-test・AnetRLRunner成功 | followup-20261023/build-debug.log |
| Debug Atari全体（追加指摘対応後） | 47 cases・1,851 assertions合格 | followup-20261023/test-debug.log |
| Release差分ビルド（整数版） | AtariEnv-test・AnetRLRunner成功 | review-green-build-release.log |
| Release PRD099（整数版） | 8 cases・825 assertions合格 | review-green-release.log |
| 固定OpenCV画像（整数版） | Debug/Releaseとも全24枚合格、最大差1・不一致2/169,344、各画像99.9%以上一致 | review-green-debug-all.log、review-green-release.log |
| 変更前の再現性 | 2 Run・16,662点一致、checksum一致 | acceptance/baseline-comparison.json |
| 変更後・キー省略（最終整数版） | 16,662点・全学習系列・定義・WARN・既存設定値一致 | acceptance/after-default-integer-comparison.json |
| 変更後・現行方式明示（最終整数版） | 同じく16,662点一致 | acceptance/after-explicit-integer-comparison.json |
| 学習を含む確認 | baselineにloss・loss EMA各125点あり | acceptance/baseline-comparison.json |
| 設定基盤 | 既存[config] 122 cases・2,628 assertions合格 | core-config-tests.log |
| 既定葉監査 | 変更前後ともerrors=0 | default-leaves-before.log、default-leaves-after.log |
| 正解再生成 | 入力・期待画像のSHA256一致 | golden-reproduction.log、testdata/prd099/meta.json |
| 隔離と後始末（最終確認） | 通常の実行体4本のhash不変、Debug/ReleaseのCMAKE_PROJECT_INCLUDE解除済み | review-isolation-check.json、review-restore-debug.log、review-restore-release.log |

学習系列のchecksumは変更前・キー省略・明示とも `b57edc3df186f07890253fd4081e6b6d97bf4bccf828201e336c3b168c9a4d61`。比較はPRD094の既存ロジックを再利用し、時刻・性能・資源以外の学習値を除外していない。全キー比較では追加のAtariEnv.resize_mode=adaptive_averageだけを認めた。出力先・Run名は各Run固有のため別扱いとした。

実行した主要コマンド（CMAKE_PROJECT_INCLUDEによる隔離を設定した状態で、VsDevCmd.batをcallするcmd経由）:

- `cmake --build --preset x64-Debug --target AtariEnv-test AnetRLRunner --parallel 1`
- `cmake --build --preset x64-Release --target AtariEnv-test AnetRLRunner --parallel 1`
- `.scratch/prd099/bin/Debug/AtariEnv-test.exe "[atari]"`
- `.scratch/prd099/bin/Release/AtariEnv-test.exe "[prd099]"`
- `.scratch/prd099/bin/Release/AtariEnv-test.exe "[prd099-resize-probe]"`
- `.venv/Scripts/python.exe core/anet-core/testdata/prd099/run_validation.py before` / `after`
- `.venv/Scripts/python.exe core/anet-core/testdata/prd072/check_default_leaves.py`
- 変更していない設定基盤の既存実行体: `core/anet-core/bin/Debug/anet-core-test.exe "[config]"`（再ビルド・上書きなし）

`Resize`の動詞は、置き換えた`ResizeGrayscale`と同じ画像サイズ変換の意味で用いる。内部APIの移行は全現用呼び出しへ適用し、旧関数や互換入口は残していない。

## 残件・実行しなかった検証

- PRD受入5: Qbert・64 envでの正式な交互5組の速度測定。別の重い処理と競合するため未実施・判定保留。縮小単体の試作値を合格の根拠にしない。
- PRD受入6のGUI実表示: 学習/定期評価/EvalPanelの実効継承とReset/Step適用はテスト済みだが、実際のEvaluation Traceプレビューを使う確認は未実施。稼働中の通常実行体を使わず、機械が空いてから隔離実行体で行う。
- Qbertの50M学習効果A/Bは後続実験。今回の短い等価性Runを学習効果の判定に使わない。
- 仕様上の未決と新規起票候補はなし。性能・GUIの残件は本メモとPRDの受入へ保持し、別PRDを自動起票しない。
- ビルド設定は元へ戻した。戻した後に通常出力先へbuildしていない。隔離実行体は `.scratch/prd099/bin/Debug/AnetRLRunner.exe` と `.scratch/prd099/bin/Release/AnetRLRunner.exe` に残る。

## 独立レビューへの対応

- 初回: an-reviewの独立コンテキストで計画・実装とも差し戻し。原文は.scratch/prd099/review-20261023-1937-impl.md。半整数の数値契約（計画12、実装14/15）とPython処理段階のコメント不足（37）の2点で、範囲外の検出・開発者への確認事項はなし。
- float重みの初回方式を撤回した。210×160の各行の左80画素=128・右80画素=129なら厳密平均128.5だが、float2passでは128.500350952となる。f32の再現をhalf-f32-reproduction.py/.txtに保存し、公開Resizeのネイティブ入力のテストを追加した。
- 追加TDD RED: review-red-build.logで追加テストだけをビルドし、review-red-test.logで568 assertions中4失敗を確認。S=1の0.5/2.5/128.5が奇数側へずれる。S=84の厳密半整数も同時に固定した。
- 計画と実装を整数格子の重なり・int64_tの正確な積算・最後の商/余りによる偶数丸めへ更新した。既定adaptive_averageの式と契約は変更しない。正規化前の各軸の整数重みの和が元辺長となるため、分母は210×160。入力uint8で分子は最大8,568,000、途中の除算・丸めは0回である。
- make_golden_cv2.pyの4反復/最後2枚/終端除外/採取間隔と生成・保存・環境記録、run_validation.pyの設定固定/逐次実行/終了状態確認/基準比較/集計へ日本語コメントを追加。構文検査は合格し、生成と比較のロジックは変更していない。
- 上のfloat版の単体試作・golden完全一致は初回方式の履歴である。以下に整数版の更新結果を記録し、最終結果として参照する。

- 整数版GREEN: review-green-build-debug.logで隔離Debug差分ビルド成功、review-green-debug-all.logで47 cases・1,851 assertions合格。OpenCV固定画像は最大差1、不一致は24枚の計169,344画素中2画素。全画像それぞれで99.9%以上一致を満たし、許容と期待ファイルは変更していない。初回float版の完全一致と区別する。

- 整数版Release: review-green-build-release.logで隔離差分ビルド成功。review-green-release.logで8 cases・825 assertions合格。goldenはDebugと同じ最大差1、不一致2/169,344、各画像の99.9%以上一致を満たす。
- 整数版の試作測定: review-resize-probe.log、5組×200回、準備26.3µs、速度比中央値1.3122（各組1.16516〜2.17945）。画素寄与回数は2passの71,568/laneで変わらない。競合下の短い縮小単体の値であり、正式なEnv Stepの受入5は引き続き保留。
- 修正後自己点検: review-self-check.jsonで既存8ファイルのBOM保持・テキスト13ファイルのUTF-8/LF・Python構文・Markdownリンク・固定goldenのSHA256すべてOK。全品質項目を適用範囲に従って再走査し、整数型・積算上限・正規化・最終丸め・既定式の保持とコメントを確認した。
- 最終実行体での既定の再比較を開始。既存のbefore1と、after-default-integer / after-explicit-integerを同じrun_validation.pyの処理で比較する。入口は.scratch/prd099/run_integer_acceptance.py、ログはreview-acceptance.log。前回Run・比較結果は上書きせず保持。初回の既定式から変更はないが、最終実行体の証拠として更新する。完了結果は以下に追記する。
- 再レビュー時点では数値修正後のDebug/Releaseビルドとテストは合格。Debug/ReleaseのCMAKE_PROJECT_INCLUDEは検証のため再設定中であり、最後の検証後に再び解除して通常実行体のhashを再確認する。

- 2回目レビュー: 計画・実装ともOK（OK 2件／差し戻し0件）。原文は.scratch/prd099/review-20261023-1953-impl.md。数値契約とPythonコメントの修正を確認し、範囲外の検出・開発者への確認事項はなし。指摘の見送りはなし。再確認ポイントは正式性能・GUI・長期A/Bの保留、最終実行体の追加再比較と一時設定の再解除・hash確認であり、後者2点は以下で完了した。
- 最終実行体の学習比較: review-acceptance.logでafter-default-integer（218.8秒）・after-explicit-integer（195.4秒）がともに終了値0。変更前の16,662点と全点checksumが一致。定義・WARN・既存設定値も一致。acceptance/summary-integer.jsonはerrors=[]。基準のloss・loss EMA各125点を含む。前回と同じchecksum `b57edc3df186f07890253fd4081e6b6d97bf4bccf828201e336c3b168c9a4d61` である。
- 最終の隔離確認: review-restore-debug.log / review-restore-release.logでCMAKE_PROJECT_INCLUDEを解除し、構成だけ復元。その後の通常出力先へのbuildは0回。review-isolation-check.jsonで通常のRunner/test計4本のhashが開始時から不変、両cacheから一時includeが除かれたことを確認した。稼働中のRunや過去のRun artifactは変更していない。

## 完了報告

実施内容: PRD099の縮小方式選択を実装。既定adaptive_averageを保持し、pixel_areaは整数の面積比の平均を最後に偶数丸めする。Reset/Step・通常の評価設定継承・実効設定・構築ログ・fail-fastを揃えた。Debug/Releaseビルドと関連テスト、固定24画像、既定の学習系列の前後比較、自己点検を完了した。

変更ファイル:

- core/envs/atari1/src/AtariPreprocess.hpp・AtariPreprocess.cpp: GrayscaleResizerと重み再利用、数値計算。
- core/envs/atari1/src/AtariEnv.hpp・AtariEnv.cpp: 設定・検証・所有・Reset/Step・ログ。
- core/envs/atari1/src/AtariEnv_test.cpp: 既定/数値/設定/Env経路の回帰と単体試作測定。
- apps/runner/config/Atari.txt: 弱い既定とrun.@pixel_areaだけ追加。既存の未コミット変更を保持。
- core/anet-core/testdata/prd099/のREADME.md・make_golden_cv2.py・run_validation.py・meta.json・frames_210x160.u8・expected_84x84.u8: 固定入力と参照、再生成・比較手順。
- docs/design/220_atari_env.jp.md・220_atari_env.en.md: 設定・前処理・検証の現行仕様。
- 本実装メモ: 計画・TDD・検証・レビュー・未実施事項。

レビュー結果: 独立an-reviewと3往復し、計画・実装ともOK。初回の2点は修正済み。見送った指摘・範囲外の検出はなし。再確認ポイントのうち正式64 env性能・実GUI・長期学習は上記の保留として残す。最終の追加再比較と構成復元・hash確認は完了。

決まらなかったこと: なし。新規の起票候補は0件。

判断してほしいこと: なし。

次にできること: 機械が空いてからPRD受入5の正式64 env性能測定と受入6のEvaluation Trace実GUI確認を行う。Qbert長期A/Bは後続実験として、方式だけを変えて第8面突破と25M時点の進行を記録し、50M比較を行う。今回の短いRunと単体試作をそれらの合格や学習効果の根拠にはしない。staging・commit・pushは行わず停止する。

## 追加指摘への対応（2026-10-23）

- ファイル内helperは既存のstatic関数へ合わせる。ValidateResizeModeの無名namespaceを除き、サイズのネイティブ設定検証もstatic helperへ集約する。設定読取と、直接組み立て・変更した設定を受けるEnv構築で同じhelperを使う。
- GrayscaleResizerには、直接利用時の方式と入力寸法・拡大禁止の検証を残す。設定キーの診断を担うAtari側と、画像処理部品の入力条件を担う境界を分ける。設定側の期待上限160は元画像寸法の最小値から決まり、別の設定定数を増やさない。
- 各Envの構築時の記録はverboseへ下げる。PRDの名前・方式の追跡は維持し、既定infoのLogPanelへ64 lane分の行を追加しない。ファイルログとverbose表示には各Envの記録を残す。グローバルな出力済み集合・once状態を増やさない。日英設計書へ粒度を反映する。
- 数値計算・設定値・Runプロファイルは変更しない。通常実行体を保護した既存Debugビルドの差分とAtari全体テストで検証し、CMake一時指定を解除する。学習系列・golden生成・性能測定の再実行は不要。検証結果と再レビューを以下へ記録する。

- 追加検証: VsDevCmd経由でcmake --build --preset x64-Debug --target AtariEnv-test AnetRLRunner --parallel 1に成功。隔離DebugのAtariEnv-test.exe "[atari]"で47 cases・1,851 assertions合格。test-debug.logを検索し構築infoログ0件・verboseログ29件を確認した。今回のRelease再ビルド・学習Run・性能測定は、数値計算や設定が変わらない局所リファクタリングとログレベル変更のため再実行していない。直前までの整数版の検証結果は履歴として保持する。
- 追加レビュー: 同じ独立an-reviewによる3回目は計画・実装ともOK。原文.scratch/prd099/review-20261023-2043-impl.md。指摘・範囲外の検出・開発者への確認事項・見送りはなし。
- 後始末: followup-20261023/restore-debug.logでDebugの一時includeを解除。verification.jsonでDebug/Releaseとも一時指定なし、通常のRunner/test4本のhash不変を確認。解除後のビルドは行っていない。BOM3ファイル保持、UTF-8/LFとgit diff --checkも合格した。
- 追加対応の完了: 変更はAtariEnv.cpp・日英Atari設計書・本メモ。仕様上の未決、起票候補、追加の判断依頼はなし。正式64 env性能・実GUI・長期A/Bは従来どおり未実施として残し、コミットせず停止する。

## 高速化（2026-10-24）

### 固定した前提と計画

- 基準は今回開始時の作業ツリー。ソース・設計・検証README・本メモの開始時コピーとhashを.scratch/prd099/perf-20261024/beforeとbaseline.jsonへ保存した。実験用.scratch/prd099/bin/Releaseの全45ファイルのsize/mtimeと、通常・実験用計6実行体のSHA256をprotected.jsonへ固定。変更とビルド出力はこのディレクトリへ書き込まない。
- 出力先は.scratch/prd099/bin-perf/<Config>。既存ビルド木へperf-20261024/isolate.cmakeを一時注入し、VsDevCmd経由のcmake --build --preset x64-Debug/x64-Release --target AtariEnv-test AnetRLRunner --parallel 1で差分のみビルドする。最初に-- -nで範囲を確認し、最後に両cacheから一時includeを外す。全再ビルド・ninja直接実行・通常実行体の上書きはしない。
- 外部の数値挙動を変えない最適化である。変更前のGrayscaleResizerをPixelAreaReferenceResizerとしてテスト内へ固定し、最初に既存実装との一致を確認する。参照計算は高速化対象にしない。
- 優先する変更は、縮小器が作業領域を持って再利用、収まる画像寸法では32bit整数積算、軸ごとの重みを平坦化すること。同一縮小器のpixel_area Resizeを同時実行しない契約を明記する。AtariEnvは独自の縮小器を持ち、同じEnvのReset/Stepは直列なので呼び出し側を変更しない。shared/TLS状態は増やさない。
- 32bitの上限は横255×160=40,800、全体255×210×160=8,568,000。任意の大きい入力寸法を狭い整数へ黙って落とさず、積算上限がint32を超える場合は64bitを維持する。
- 固定4要素は最大カーネルが4以下の場合だけに使い、S=1などの大きいカーネルは必要数の平坦配列で扱う。0重みのpaddingも実在する画素だけを参照し、範囲外の読み取りを生まない。
- 商/余りと偶数丸めの式は維持する。手書きの逆数乗算・シフトは初回採用せず、独自除算の全数テストは対象外。adaptive_averageの実行経路は文字列としても保持する。設定・検証・ログ・Runは変更しない。

### 検証手順

1. 隠し[prd099-perf-exact]で、固定24枚・乱数1000枚・定数/縞/市松の5種類とS=1..160の直積を旧参照と全画素比較する。さらに3000×3000の64bit必要例を確認。基準固定後と最終Releaseで実行し、比較数・画素数・終了値を記録する。
2. Debug [atari]とRelease [prd099]で既存テストを実行。OpenCV結果が最大差1・不一致2/169,344のままであることを確認。固定goldenと生成器は変更しない。
3. [prd099-resize-probe]を、同じプロセス内の変更前参照・現在実装・adaptive_averageの5巡ローテーションへ更新。各方式50回準備、1試行2000回、各方式5値と中央値を保存。負荷競合中の参考値とし、正式なStep区間受入は行わない。
4. 変更前後で既定経路の文字列と、保護対象実行体SHA256・Release全ファイルのsize/mtime、一時include除去、BOM/LF・リンク・差分を確認する。既定の学習系列は経路を変えないため再実行しない。自己点検後、結果を本節へ残してClaude Codeの差分レビューへ渡す。

### 実行記録

- 基準Release: baseline-build.logの5差分アクションでbin-perfへ出力。AtariEnv-test-before.exeとして保存し、protectedなRelease実行体は起動・書換しない。baseline-exact.logで164,644 assertions合格、164,640組・1,418,126,640画素の完全一致（38.735秒）。baseline-probe.logの5巡で参照と現行（最適化前）は中央値73.9323µs / 70.1179µs、比1.0544。負荷の大きな揺れがある対照値として記録し、正式性能の判定に使わない。
- 適用: pixel_areaの作業領域再利用、32bitに収まる場合の積算幅縮小、平坦な重み配列と小カーネル4要素の展開。paddingのindex=0は常に実在する入力を指し、重みは0。S=84では横70,560＋縦28,224＝98,784寄与/lane（0重みの寄与を含む）になり、旧71,568より演算数は増える。分岐と64bit演算・再確保の削減が実時間へ効くかは試作値で確認する。作業領域は141,120 byteの毎回確保から、70,560 byte/Envの構築時確保へ変わる。64 Envで4,515,840 byteを保持する。
- 入れないもの: 手書きの除算の逆数/シフト置換。商・余りの式を保持し、全数の除算テストは対象外。32bitに収まらない画像では従来の64bitを維持する。既定のResize関数先頭からadaptive_averageのreturn式までを保存元と文字列比較し、不変を確認した。

- 最適化Release: optimized-build-release.logの7差分アクションでAtariEnv-testとAnetRLRunnerをbin-perf/Releaseへ出力。optimized-release.logで既存8 cases・825 assertions合格、OpenCVは最大差1・不一致2/169,344。optimized-exact.logで164,644 assertions合格、164,640組・1,418,126,640画素すべて完全一致（41.985秒）。固定画像・許容は変更していない。

### 縮小単体の参考測定

同一プロセス内で順序を巡ごとに回し、各方式50回準備・各試行2000回の平均を5値採取した。optimized-probe.logとmeasurements.jsonへ全数値を保存。以下のusは1 Resizeあたりである。

| 巡 | 変更前参照 | 高速化後 | adaptive_average |
| --- | ---: | ---: | ---: |
| 0 | 76.2679 | 64.9244 | 502.967 |
| 1 | 79.0537 | 61.9712 | 410.011 |
| 2 | 78.1923 | 63.9019 | 1014.08 |
| 3 | 72.8379 | 59.9516 | 341.710 |
| 4 | 76.6939 | 61.4964 | 315.882 |
| 中央値 | 76.6939 | 61.9712 | 410.011 |

変更前/変更後の中央値の比は1.23757、縮小単体の所要時間は約19.20%減った。変更前どうしの基準測定でも比1.0544の揺れがあり、既定の補間の時間も大きく動く負荷競合下の参考値である。各方式の値からRun全体や64 envのEnv Step性能を推定せず、正式受入5は未実施として保持する。

### 最終検証と引き渡し

| 検証 | 実行内容 | 結果・ログ（.scratch/prd099/perf-20261024/以下） |
| --- | --- | --- |
| Release差分ビルド | VsDevCmd経由でcmake --build --preset x64-Release --target AtariEnv-test AnetRLRunner --parallel 1 | 成功、optimized-build-release.log |
| Debug差分ビルド | 同じくx64-Debugの2対象 | 成功、optimized-build-debug.log |
| Debug全体 | .scratch/prd099/bin-perf/Debug/AtariEnv-test.exe "[atari]" | 47 cases・1,851 assertions合格、optimized-debug-all.log |
| Release既存 | .scratch/prd099/bin-perf/Release/AtariEnv-test.exe "[prd099]" | 8 cases・825 assertions合格、optimized-release.log |
| 全画素一致 | 同Release実行体 "[prd099-perf-exact]" | 164,644 assertions合格、164,640組・1,418,126,640画素の不一致0、optimized-exact.log |
| OpenCV固定24枚 | Debug全体とRelease既存に含まれる比較 | 両構成とも最大差1・不一致2/169,344、各画像の99.9%以上一致を維持 |
| 除算の全数検査 | 手書きの逆数/シフトを採用しないため対象外 | 商/余りの式は維持、32bit条件は積算上限で決める |
| 参考速度 | 同Release実行体 "[prd099-resize-probe]"、5巡交互 | 17 assertions合格、optimized-probe.log、measurements.json。上表の全値を記録 |
| 既定の経路 | 開始時コピーとResize関数の先頭〜既定returnを文字列比較 | 完全一致、SHA256 b537924a71eb2e4b6c11713588126c5b198837c056c99ddfc8c6072c7f392211。学習Run再比較は再実行しない |
| 保護対象 | protected.jsonとの比較 | 通常・実験用6実行体SHA256不変、実験用Releaseの全45ファイルのsize/mtime不変、verification.json |
| ビルド設定復元 | VsDevCmd経由のcmake --preset x64-Debug/x64-Release -U CMAKE_PROJECT_INCLUDE | 両cacheから一時includeが除去された。restore-debug.log・restore-release.log。復元後のbuildは行わない |

上のビルド行は、一時includeによるbin-perf隔離中の実行履歴である。現在はcacheを通常構成へ復元済みなので、隔離ビルドを再実行する際はperf-20261024/isolate.cmakeを先に設定する。既存ビルド木の差分のみ使用し、clean・全再ビルド・ninja直接実行はしていない。

変更ファイル:

- core/envs/atari1/src/AtariPreprocess.hpp・AtariPreprocess.cpp: 作業領域所有、積算幅の選択、平坦重みと小カーネルの展開。設定・検証・ログ・既定の式は維持。
- core/envs/atari1/src/AtariEnv_test.cpp: 開始時の参照コピー、全サイズの出力一致、5巡交互の隠しprobe。
- docs/design/220_atari_env.jp.md・220_atari_env.en.md: 同一縮小器のResizeを並行実行しないことと、結果の独立所有を明記。
- core/anet-core/testdata/prd099/README.md: 現行のbin-perf検証コマンドと検査範囲。
- 本実装メモ: 本節の計画・上限の証明・検証・測定・保全記録。

新しい実行体:

- .scratch/prd099/bin-perf/Release/AnetRLRunner.exe
- .scratch/prd099/bin-perf/Debug/AnetRLRunner.exe
- .scratch/prd099/bin-perf/Release/AtariEnv-test.exe
- .scratch/prd099/bin-perf/Debug/AtariEnv-test.exe
- .scratch/prd099/bin-perf/Release/AtariEnv-test-before.exe（高速化前の比較テストを固定した実行体）

品質確認観点は全項目を適用条件に従って自己点検した。privateな計算情報を公開せず、作業領域は縮小器が所有し、共有可変状態・ロック・device転送・設定キーを追加していない。BOM/LF・Markdownリンク・固定goldenのSHA256・git diff --checkに問題なし。比較基準からの7ファイルの差分をperf-20261024/*.diffへ保存した。

Claude Codeの差分レビューは未実施であり、自己点検と区別して引き渡す。仕様上の未決はなし。正式な64 env Env Stepの受入と実Runは今回の対象外で、単体の1.23757倍をその合格とは扱わない。実験用Releaseと通常実行体を保持し、staging・commit・pushせず停止する。

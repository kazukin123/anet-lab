# PRD102 GridMaze seed_mode 実装メモ

## 概要

[PRD102](102_gridmaze_seed_mode_10prd.md)に従いseed指定時に固定、seed未指定時にepisode・processを選択し、既定をprocessにする。2026-10-24にユーザーが本計画を承認した。既存indexと無関係な変更、過去Run・baselineは保持し、staging・commit・pushを行わない。

## 主な変更

- Config共通のReadConfigへoptional値の読み取りを追加し、未指定を保持する。既定prefixと上書きprefixの順序・数値変換・記録を既存契約に合わせる。GridMazeEnvConfigのseedをoptionalにして、randomize_seedを2値のseed_modeへ置換する。mode検証は設定読み込みと直接構築で共通化する。
- GridMazeEnv.cpp内部の補助クラスGridMazeProcessSeedStateに初期化状態・採番値・失敗を限定して保持する。初回の採番・INFOを一度だけ行い、失敗を保存して後続へ再送出する。公開ヘッダの名前空間へ補助型を出さない。PRDのクラス一覧に無い補助型だが、プロセス寿命と失敗保持を一か所に閉じるために置く。
- 各Envは構築時にprocessの入力seedを取り出し、その後は個体内の値を使う。seed指定ならmodeによらず固定し、seed未指定のepisodeだけがResetで毎回MakeAutoSeedを呼ぶ。Step/Resetに共有seed用のロックを足さない。既存のfixed引き直し通知の共有記録は維持する。
- 標準設定2種・固定fixture・PRD100の現行検証driver/README・環境設計日英版を移行する。表示・aux・生成/引き直し/観測/報酬の契約は維持する。
- core/anet-core/testdata/prd102/へ独立プロセス試験と短Run/設定比較のdriver・READMEを置く。変更前の実行体・設定・出力は.scratch/prd102に保存する。
- PRDの正式採番とA7の検証補足を反映し、wikiの参照と最終blobを更新する。

## テスト

- Public interface / surface: Configのoptional読み取り、GridMazeEnvConfig、GridMazeEnv構築・GetSpec・Reset・Step・aux/GetScalar、RunManagerの学習・configured eval・CreateEvalRunner、設定ファイルとCLI。
- 優先behavior/TDD順序: まずprocessの構築からReset・aux・共通INFOまでのtracer bulletを1本RED→GREEN。次に並行8個体と寿命、episodeの50 Resetと指定seedの境界・復元、不正設定と通知、RunManagerの用途別共有を、それぞれ1テスト→失敗確認→最小実装→成功確認で進める。既存の固定fixtureはseedだけを明示し、自動モードではseedを未指定にする。
- 共有初期化タグは通常スイートから分離し、fresh processで起動する。2プロセス間のseed相違や50回の一意性は要求しない。各採用seedで盤面を再現し、episodeの呼出経路をレビューする。テスト専用の本体APIは追加しない。
- A1〜A15は末尾の検証結果へテスト名/ログを対応付ける。GUIは実画面の描画変更ではなくRunnerFrameが使うCreateEvalRunnerの公開経路でEnv共有を確認する。

## 検証

1. 編集前にMSVC初期化経由でDebug全体をビルドし、実行体・DLL・設定を凍結する。PRD100のdriverを作業領域の別OUTへ読み込んで変更前の短Run・ログを保存し、既存記録を上書きしない。
2. 各TDDサイクルでGridMazeEnv-testをビルド・対象タグ実行する。RED/GREENの終了値と出力を.scratch/prd102へ保存する。
3. Debug全体、通常GridMazeスイート、PRD100ログ専用タグ、新しい共有状態専用タグを独立プロセスで検証する。
4. check_default_leaves.pyとその回帰テストを実行し、現行の全設定キー/文字列値を前後比較する。許容差分はGridMazeの旧キー削除、seed_mode=processの追加、共有seed既定の削除で、既存の明示的な実験用seedは保持する。別の変更は分けて記録する。PRD072の履歴goldenは変更しない。
5. 変更前の固定Runと変更後seed指定の全scalarをPRD100の正規化規則で完全比較し、件数・被覆・SHA256を保存する。実行体・設定のhashも保存する。
6. 品質観点の全項目を自己点検し、PRDレビューと実装/計画レビューを独立したan-reviewで受ける。原文・結果・残る未決を保存する。

ビルドは常に次のMSVC初期化を使う。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```

## 前提

- 未決事項監査のユーザーブロッカーは0。次番号102と計画20は着手時に衝突なしを確認した。
- 共有範囲・既定process・seed指定の優先/WARNなし・共通INFO1回・入力seedと採用seedの区別は合意済み。
- 追加の設計分岐はない。モード検証はGridMaze設定が所有し、既存の数値検証は共通ConfigDataが所有する。他Envとrun.seed契約は範囲外。
- .scratch/prd102/source-beforeとindex-before.binが今回の編集前の記録。既存のdirty変更を戻さない。BOMの有無を各元ファイルから保持し、改行はLF。

## 経過

- 2026-10-24: 正式採番、監査、計画保存を完了。実装前の基準ビルド・比較を開始する。

### 仕様調整後の確定計画

- 2026-10-24ユーザー変更: seed_modeはprocess/episodeだけ。seedがあれば固定を優先し、0/最大値を含む全整数域を保持する。NaNではなくoptionalで未指定を表す。
- 追加対象: core/anet-core/include/anet/config.hppの共通optional読み取りとcore/anet-core/src/config_test.cppの公開設定テスト。個別Envへ未指定判定用の独自キー解決を重ねないために必要。
- TDDを共通optional読み取りの未指定・0・最大値・prefix・不正値から再開し、その後新仕様のprocess単体→並行共有→episode→seed優先/不正設定→RunManagerを1挙動ずつ進める。旧仕様のRED/GREENは履歴であり新仕様の成功に数えない。
- GridMaze_muzeroの共有seed既定を削除する。GridMaze.txtのユーザー実験用seedは削除せず、seed優先による固定として扱う。自動processのRun検証はseedのない隔離設定を作る。
- 旧fixedモードは通常の未知モードとする。個別検出/警告/互換処理を加えない。設定比較では意図したseed既定削除も差分に含め、既存実験用seedの保持を確認する。

- 2026-10-24: 新仕様のPRD・計画を保存した。実装スキルの計画変更時の確認に従い、コード続行前に合意内容を提示する。
- 共通optionalはConfigDataが読み取る葉値を対象とし、既存ConfigReaderのprefix/変換/記録へ委譲する。複合ConfigReaderをoptionalで包む拡張は要求されておらず、誤って未指定扱いにせず型の段階で拒否する。
- 共通optional設定の追加に合わせ、実行基盤設計100の日英版にも葉値対応・初期値保持・省略記録・不正値の扱いを追記する。
- デバッガで並行REDの異常終了をLogCaptureGuardの記録vectorへの同時追加と特定した。test_util.hppの追加部分だけをmutexで保護し、テスト本体の検証はjoin後に行う。productionのAPI・ログ動作は変えない。
- 2026-10-24: 新仕様の続行をユーザーがYESで承認。共通optionalは未対応RED→15 assertions GREEN。

- PRD再レビューはOK（.scratch/prd102/review-20261024-1905-prd-r3.md）。共通optionalの設定記録と設計100日英の移行対象を本文/A15に明記した。
- episode-redは未知modeで失敗、episode-greenは251 assertions成功。settings-redはDTO直接構築だけ未拒否で失敗、共通検証適用後settings-greenは54 assertions成功。
- 遅延初期化・異なる迷路設定・RunManager・全設定比較は、既存の最小実装に対する公開経路の追加受入検証として行う。不要な本体変更でREDを作らない。
- 既定葉監査はGridMaze.txt:94の既存実験用seed指定1件を検出した。ユーザー承認の明示指定を保持するため変更せず、編集前の同じ行と照合して既存事項として区別する。新規違反は0。監査回帰テストは15件成功。


## 現時点の検証結果

- 変更前: MSVC経由Debug全体の終了値0（build-before.log）、実行体・DLL・全設定をbaseline-bin/baseline-configへ凍結。
- 変更後: MSVC経由Debug全体の終了値0（build-after.log）。統合fixtureのみの不足を修正し、増分Debugを追加する。
- 共通optional: [prd102_optional] は28 assertions/2 cases、共通設定 [config] は2676 assertions/127 cases成功（optional.log、config-regression.log）。
- 通常GridMaze [gridmaze]: 2207 assertions/10 cases成功。0・seed2・golden・最大値、50 Resetと復元、座標・到達・GetScalarを含む。
- 独立プロセス: fixed-log 14、process-1/process-2 各12、concurrent 16、late 9、settings-errors 54 assertions成功。再採番の偶然の一致を失敗にしない。
- 短Run: 変更後131.4秒・終了値0。66 tags/20095 pointsが変更前と完全一致。SHA256は両方54cb0e97abb891ac9799018ee5dec88d7de014e536d24cde7704840e44034ea8。execution.json、normalized.summary.json、noisy-coverage.json、comparison.jsonへ保存。
- RunManagerの最初の試験はfixtureのNN型登録漏れでUnknown module type: Flattenとなった。本体は変えず、通常起動のInitRLを使うよう修正する。初回結果は成功扱いにしない。
- 全設定の最初の試験はPropertiesの通知がMessageダイアログになり停止した。CPU時間不変と作成入力時刻、MainWindowTitleを確認し、自分の[prd102_config_snapshot]プロセスだけ終了。LogCaptureGuardを追加して再実行する。ユーザーのRunには操作していない。
- source-beforeとの比較で22ファイルのLF・既存BOMを保持し、相対リンク切れ0、git diff --check終了値0。途中で別作業のHEAD/indexが更新されたため元へ戻さず、index-observed.binに観測を保存した。この実装からgit add/commit/pushは行っていない。

## 自己点検（実装品質の全項目）

| 観点 | 点検結果 |
|---|---|
| 機械検査1〜8・62 | using namespaceヘッダ0・production dynamic_cast 0・TODO 0。計画名/相対リンク/LFを確認。BOMは編集前snapshotと一致。GridMaze.txtのBOMは今回以前の変更を保持。skill非変更のため同期検査は対象外 |
| 実装計画9〜13 | 前提・公開経路・TDD順・MSVCコマンドを保存。仕様変更後にユーザーYESで続行。最小実装のRED/GREENと追加受入を区別 |
| PRD適合14〜17 | A1〜A15の結果を対応表に記録。共通optional/test capture/検証資材の追加理由を明記。無関係なdirty変更は差分のworking版で分離 |
| 契約18〜22 | 現用の旧bool読み/分岐/設定/検証driverは除去。履歴資料と保存baselineは保持。互換層なし（22対象外） |
| 失敗23〜28 | 既存型変換を再利用し、modeは設定/直接構築で早期検証。初期化失敗はcall_once内で保存して再送出し、自動retryを防ぐ。worker例外はfuture.getで主スレッドへ戻す |
| 設定29・54 | 追加のmode既定は?=、テスト/明示設定は=。既存実験seedの監査検出1件は変更前からあり、ユーザー合意で保持。新規プロファイル定義なし |
| C++30〜35・55〜60・63〜66 | 既存機能ファイル内へ配置。新hppなし。公開ヘッダのoptional対応はPRD/計画に明記。可変共有状態はPRD102の限定例外。新たな参照メンバなし。汎用ConfigはEnv名非依存。新UI更新ロック/取り出し口なし。mode定数はGridMazeEnvConfig内（ユーザー指定のhpp配置修正後）。固定キーはmacro読み、検証をGridMaze共通helperへ集約 |
| コメント36〜39 | 新規コメントは日本語で同期・優先・記録の段階を記載。未対処のコード懸念なし |
| 計測40・41・61 | Resetの既存profileを保持。共有取得Getは構築時だけ計測。通常Reset/Stepへ共有取得ロックを追加しない |
| GetScalar42・所有権43・44 | 当該本体は非変更。既存の公開遷移テストで回帰確認。map/RNGは個体所有 |
| ログ45・テスト46・47 | 新ログは英語。公開Config/構築/Reset/Step/aux/RunManager経路で検証し、本体にtest-only APIを足さない |
| 検証48〜52 | Debug・関連テスト・短Run・設定比較の結果とログを本節/対応表に保存。設計100/120日英と領域READMEを更新。OS採番失敗の実故障注入とGUI画面描画は未実施（下記） |
| skill53 | skillを変更していないため対象外 |

未実施の範囲: OS自動採番失敗の実故障注入は、本体にtest-only APIを作らず行えないため、例外保存・再送出・call_once完了のコード照合で確認する。画面描画の手動試験は非変更部分のため行わず、既存の符号なし%llu表示をコード確認し、GUIと同じCreateEvalRunner生成経路を検証する。独立プロセス間のseed相違や50回の一意性は契約外であり要求しない。

- 設定比較の初回失敗は保存済み入力run.@nature_dqnの定義が編集前時点で無いことが原因だった。PRD072の履歴を変更せず、PRD102のmanifest.jsonへ現用17入力を保存する。旧Atari 2条件を現用default/baseへ置換し、残る15条件を維持。比較は実際のCLI parserを通し、同じ入力を凍結設定/現用設定へ与える。検証資材にmanifest.jsonを追加する理由は、比較入力を固定して廃止済みRunを本体に復活させないため。


## 最終受入結果（A1〜A15）

全ログは.scratch/prd102配下。独立状態を持つタグはそれぞれ新しいプロセスで起動した。

| 受入 | 検証と結果 |
|---|---|
| A1 | process-1/process-2、concurrent: 既定/明示processの反復Resetでseed・map一致、共通INFO1件 |
| A2 | concurrent: latchで同時初回構築8個体→全個体破棄→再構築、16 assertions成功 |
| A3 | runner-fixed: RunManagerで学習2 lane、configured eval、CreateEvalRunner、別run.seedの2 Run、20 assertions成功 |
| A4 | late: 異なる寸法・密度で共通入力seedから各盤面を復元し、片側Step/Reset後の他方も一致 |
| A5 | late: seed指定とepisodeを先に破棄し、process利用前はINFO0件、利用後1件。9 assertions成功 |
| A6 | process-1とprocess-2は独立プロセスで各12 assertions成功。seedの偶然一致は禁止しない。永続化経路なしを照合 |
| A7 | prd102_episode: 50回到達・全採用seed復元、251 assertions成功。既存prd100_randomも9条件×50 Reset成功。重複禁止を除去し、ResetごとのMakeAutoSeed分岐をコード照合 |
| A8 | 通常gridmaze: seed 0/2/6545537233705677626/UINT64_MAX、反復/再構築、25セルgolden・生成後RNG一致 |
| A9 | process/episode/指定seedの公開auxからseedだけで復元。最大値をunsigned decimalで読込。非変更のGridMazeView.cppの%llu/unsigned long long表示も照合 |
| A10 | settings-errors: seed 0/最大値と有効mode併用を3 Resetずつ確認。指定seed優先、共通採番INFO/WARNなし |
| A11 | settings-errors: 空/未知mode、DTO変更後の直接構築、不正seed5種×2 modeをfail-fast、54 assertions成功 |
| A12 | fixed-log: 指定seedの引き直しINFO0/1件と自動modeの追加通知なし、14 assertions成功 |
| A13 | comparison.json: 66 tags/20095 points、時間・資源以外のtag/step/value完全一致。Noisy被覆とSHA256も保存 |
| A14 | config-comparison.json: 現用17入力の全キー/文字列値を実CLI経由で比較し17件成功。差分はGridMazeの旧bool削除・mode=process追加・MuZeroの共有seed既定削除だけ。現用コード/設定/driverの旧boolは0件。履歴は保持 |
| A15 | optional.log: 28 assertions/2 cases成功。未指定記録なし/DTO初期値7の記録、0/最大値、空文字/false/空vectorの再読込、prefix上書き、不正baseの早期失敗。config-regression.logは127 cases/2676 assertions成功 |

最終Debugはbuild-matrix-fix.log（MSVC経由、終了値0）。通常GridMazeは10 cases/2207 assertions成功。構築fixture修正後の共通production変更はないため、既に成功した独立タグ/短Runを繰り返さず、修正fixtureと設定比較だけを再実行した。

実行コマンド:

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
.\.venv\Scripts\python.exe core/anet-core/testdata/prd102/run_validation.py tests
.\.venv\Scripts\python.exe core/anet-core/testdata/prd102/run_validation.py settings
.\core\anet-core\bin\Debug\anet-core-test.exe "[config]" --reporter compact
.\.venv\Scripts\python.exe core/anet-core/testdata/prd102/run_validation.py after
.\.venv\Scripts\python.exe core/anet-core/testdata/prd102/run_validation.py compare
.\.venv\Scripts\python.exe core/anet-core/testdata/prd072/check_default_leaves.py
.\.venv\Scripts\python.exe core/anet-core/testdata/prd072/check_default_leaves_test.py
```

再実行結果はtests.jsonのrunner-fixed.log、settings-final-exit.json、scalar-exits.jsonに保存。driverは同名のログ/JSONを.previous-Nへ退避して再検証記録を保持する。短Runはstageディレクトリがあれば失敗するため、過去stageを退避してから再実行する。

未解決の実装事項: なし。既定葉監査の既存実験seed1件と、実故障注入/画面描画の未実施範囲は上記のとおり。これを新規の成功や未実装扱いへ読み替えない。


## 独立レビューと完了

- 改訂PRDは2往復でOK。初回の共通optionalの記録契約/設計100日英の移行対象という不足2件を修正済み。原文: .scratch/prd102/review-20261024-1855-prd-r2.md、review-20261024-1905-prd-r3.md。
- 実装・実装計画は独立an-reviewの1往復で両方OK、修正要求0件、範囲外の検出0件、開発者の確認事項なし。原文: .scratch/prd102/review-20261024-1921-impl.md。
- レビューの再確認点: process共有はPRDの限定例外、通常Step/Resetの共有取得ロックなし、初期化失敗の保持を照合。GUI共有はCreateEvalRunnerの公開経路で確認し、実描画とOS採番失敗の故障注入は未実施として明記した。
- Noisy被覆は変更前後とも定義11件・観測タグ0件。この固定RunではNoisyが無効なので、動作中のNoisy数値の比較ではない。既存の出力を除外せず、実際に観測された全scalarの一致を確認した。
- PRD最終blob bd35c61814c7d64c3d2b146b9b0e267fb3576420をwiki/logへ記録済み。過去の999ログを保持し、現用リンクは102へ移した。
- 最終検証集計: verification-summary.json。未解決の実装事項・起票候補・ユーザー判断が必要な事項はなし。既定葉監査の既存実験seed1件は承認どおり保持する。こちらからのstaging/commit/pushは行っていない。


## ユーザー指定による配置修正

- GridMazeEnvConfigの設定一覧と読込処理を一緒に見られるよう、コンストラクタ定義をhpp内へ戻す（ユーザーが意図した配置）。DTO内のmode定数とValidateSeedModeを共通に使い、設定読込/直接構築の検証・既定値・seed優先を維持する。
- 検証はMSVC経由Debugと既存GridMaze/seed_modeテストを使う。動作を変えない配置修正のため新しいテストは追加しない。今回の変更前3ファイルを.scratch/prd102/constructor-hppへ保存した。

- 配置修正後の短Run再比較は行わず、既存の短Run記録を保持する。上記の完全一致・実行体hash照合は配置修正前の検証記録であり、今回の配置修正はDebugと既存の迷路/設定/既定processテストで確認する。設定値・生成・乱数消費の処理は変更しない。

- 配置修正後の結果: MSVC経由Debug全体0（constructor-hpp/build.log）。GridMazeEnv-test.exe "[gridmaze],[prd102_settings]"は2261 assertions/11 cases、独立プロセスの"[prd102_process]"は12 assertions/1 caseで全て成功。ログ・終了値はconstructor-hpp/gridmaze.log、process.log、results.json。新規テスト追加なし、動作変更なしを差分で自己点検した。実装品質の全項目を走査し、今回適用される配置/定数/設定読込/検証/コメント/符号化/検証記録を確認、ほかは既存判定を維持する。

- 配置修正の独立再レビューも実装・計画ともOK、修正要求・未決・開発者確認事項なし。原文: .scratch/prd102/constructor-hpp/review-20261024-2012.md。hppの設定一覧とコンストラクタをセットにする意図を今後も維持する。


## 2026-10-25 GridMaze seedの監査分類の修正

- 原因: HEAD版のassignment_roleはGridMaze.txtのGridMazeEnv.seedをenvironment default definitionと分類し、意図的な実験指定「=」に対して「?=」を要求していた。同じ現用設定を使った分類関数の差し替えで、HEAD版はエラー1件・現用版は0件を確認した。以前の「既存実験seed1件」の監査記録は当時の結果として保持し、今回解消した。
- 対応: 調査開始時点で作業ツリーにあったmigrate.pyの分類修正（explicit experiment seed）を採用する。GridMazeのseed値・演算子・Envの実行時契約を変更しない。
- 回帰: 既存test_gridmaze_experiment_seed_remains_an_explicit_choiceを補強。「=」の受理・分類理由・期待演算子、「?=」の拒否、seed_modeの環境既定「?= process」の受理、migrate_textによる値・「=」・コメント・改行の保持を確認する。
- RED/GREEN: 本体ファイルを書き戻さず、メモリ上のHEAD分類関数で補強テストを実行して3 failures/0 errors。現用分類ではPython回帰16 testsが全て成功、全体監査errors=0・終了値0。red-head-rule.log、regression.log、regression-cli.log、audit.logを.scratch/prd102/seed-classification-20261025へ保存した。
- コマンド: .\.venv\Scripts\python.exe core/anet-core/testdata/prd072/check_default_leaves_test.py、および.\.venv\Scripts\python.exe core/anet-core/testdata/prd072/check_default_leaves.py。
- 自己点検: 実装品質の全項目を走査し、今回適用するPythonコメント・ログ・公開監査/移行経路の検証・記録・BOM/LF・差分範囲を確認。C++・設定・公開API・設計契約・skillは非変更のため該当項目は対象外。承認計画どおりC++ビルドは不要で未実施。
- 未決なし。設定ファイルの内容と既存indexを保持し、staging/commit/pushは行わない。

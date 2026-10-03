# 設定選択の最終値参照と Run プロファイル優先順位 実装メモ

## 概要

正本は `072_config_selection_final_value_10prd.md` の 2026-09-12 改訂版と ADR0040。
すべての選択が参照先の最終値・最終キー集合を読み、書き込みは従来の適用位置を維持する。
優先順位は直書き < 選択・上書き層 < Run プロファイル < CLI。
公開 ConfigManager / ConfigData API は維持する。

## 未決事項監査

- ユーザー判断が必要なブロッカー: 0。
- repo evidenceで解決済み: `config_impl.cpp` の Resolve / ApplyTerm は逐次 snapshot コピーであり、CLI 第2相は effective_map のみを更新する。P1 の伝播には内部評価器の変更が必要。
- repo evidenceで解決済み: ConfigManager の実パーサ・include・CLI と GetConfigData / GetResolutionJson から全契約を検証できる。本体に test-only API を追加しない。
- 2ximplへ固定: 宣言の適用位置と最終値の依存評価を分ける。キャッシュによる計算順の変更を後勝ち順位・記録順へ漏らさない。内部実装は config_impl.cpp に閉じる。
- PRD範囲外: Actor API・設定移行、NN 未知キー検出、値参照の多段化、学習性能比較。

## 主な変更

1. Run プロファイルの採用葉をキーごとに畳み、最後の供給元も保持する。trunk 展開後に CLI 解決入力（run.$ 以外）を再注入する。第2相は Run / CLI の順とし、Run の最終値が直前値を変える場合だけ overrides を記録する。
2. 選択を宣言位置を持つコピー指定として扱い、最終値の参照を評価する。プロファイル・上書き層の宣言はコピー先へ展開し、組み立て済みノードの選択命令はコピーしない。未選択素材へ写した宣言は dormant とする。
3. 同じ sub-prefix の選択は最後の宣言のみを採用する。不採用チェーン由来の葉・nested 選択は残さず、独立した直書き・部分選択を保持する。同一 owner のプロファイル継承は term の一部として保持する。
4. 参照に必要な部分について値とキー集合を解決し、Run / CLI の最上位の葉も参照に含める。循環・自己供給・未定義参照・上書き層と owner の兼用を診断する。依存深さは最長経路で10を上限とし、キャッシュでも検査を省略しない。
5. selections は採用したチェーンを適用順に記録する。schema_version=1、references の1段展開を維持する。
6. `docs/design/100_runtime_and_configuration.jp.md` の設定解決・記録を同期する。必要な利用ガイドも同一変更で同期する。用語と ADR はすでに新契約へ更新済み。

## テスト

- Public surface: ConfigManager のファイル / include / CLI 入力、GetConfigData、GetResolutionJson。
- TDD は各 behavior ごとに1テスト -> RED確認 -> 最小実装 -> GREEN確認。RED中に整理目的のリファクタリングをしない。
- 最初は A11（Run の葉が選択に勝つ）を tracer bullet とし、現行で失敗を確認する。次に A12 の層・CLI優先順位を現行との差分として確認する。
- 続いて M14 の参照伝播、M01～M03、M04 の全 CLI 分岐、M05 / M09 の順序、M06 のチェーン差し替えと全独立分岐、M07 / M08、M13 を縦に実装する。
- M10～M12 の循環・自己供給・未定義部分・空層・深さ10/11（宣言順変更とキャッシュ含む）を確認する。
- M06 / M07 / M09 / M14 の記録、同一 owner 継承の両 key=Env.$、Run の4→1→4が overrides に載らないことを確認する。
- 既存 config / resolver テストを維持し、A01～A15 と各分岐の対応・結果を本メモへ追記する。

## 実装前後の比較

- 設定固定 commit: `107a62c8ae01cb758f3cd49d98e8424386160e5d`。この commit の config ツリーを独立した検証ディレクトリに抽出し、入力・CLI・注入値を manifest に固定する。
- リゾルバ変更前に ConfigManager の現行実装で baseline を保存する。実パーサと公開APIを使う検証経路を用意し、production API は拡張しない。
- PRD §7 の Atari 6チェーン、DropMerge 現行IQN32（CLI含む）と stratified / qr51、LunarLander 2入力、ImageCls 3入力、GridMaze / GridMaze_muzero / CartPole 既定をすべて比較する。
- 実効設定のキー・値、selections / references の内容と順序を一致させる。overrides は空。baseline が採れない場合や差分が残る場合は A15 未完了とする。
- Run artifact は編集しない。長時間学習は不要。

## 検証

MSVC を初期化した同一 cmd プロセスで `cmake --build --preset x64-Debug --target anet-core-test` を実行する。
関連する Catch2 `[config]` テストと上記比較を実行し、最後に Debug ビルドを確認する。
実際のコマンド、baseline 保存先、終了値、各 gate の結果は作業中に追記する。

## 前提と進捗

- 通常の実装依頼として本計画から実装を開始する。契約を変更する追加判断が生じた場合は計画へ戻る。
- 無関係な既存 dirty ファイルを保持する。staging / commit / push は行わない。

### 作業記録

- 比較入力は `.scratch/prd072-validation/manifest.json` の17件。設定抽出は同ディレクトリの `prepare.py`。注入値なし、DropMerge 現行IQN32は frozen のファイル既定チェーンに CLI `backend.$=backend.@non-deterministic` を固定した。
- 実パーサを通す隠しテスト `[prd072-baseline]` を `config_test.cpp` に追加した。
- 初回採取先 `baseline/` は無効。採取コードが一時 ConfigData の Map 参照を使い、values が空だった。正本比較に使わず、寿命を修正して `baseline-v2/` へ再採取する。採取時に実効キー数 > 100 を検証する。
- リゾルバ本体変更前の A11 / A12 RED: Debug `[prd072]`、seed 4274950109、exit 1。A11 は `SiLU != ReLU`、A12 は `0.997 != 0.99` の2 assertion失敗を確認した。
- ビルド経路: `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'`。
- `baseline-v2/` の再採取成功: seed 1293936343、69 assertions、exit 0。実効キーを含む17入力を保存済み。
- Run 葉の第2相を追加した最初の GREEN: `[config],[prd072-baseline]`、seed 3384170106、105 test cases / 802 assertions、exit 0。17入力の実効値・selections・references は差分ゼロ、overrides は空。
- この時点では P1 最終値伝播・P3 差し替え・P6 依存検証は未実装であり、PRD全体の完了ではない。
- CLI 解決入力の RED: seed 292330021、`0.997 != 0.9` / `5 != 7`。再注入修正後の GREEN は `.scratch/prd072-validation/cli-input-green.log`（26 test cases / 186 assertions、exit 0、17入力差分ゼロ）。
- M14 最終値伝播の RED は `final-run-red.log`（Other.structure が SiLU、期待 ReLU）。値の供給元キーを記録して最終値を評価する最小実装後、`final-value-first-green.log` は27 test cases / 188 assertions、exit 0、17入力差分ゼロ。
- GREEN後、設定例を `core/anet-core/testdata/config_resolver_prd072.json` へ移し、`config_test.cpp` のデータ駆動テストから実パーサ・CLI・公開APIで検証する形に整理した。以降の例も1件ずつ追加する。baseline専用の隠しテストは通常の `[config]` から外し、明示の `[prd072-baseline]` だけで動かす。
- M01 / M02 は順に追加して GREEN。M03 は親変更が子へ届かず追加 use_amp も欠落する RED（`m03-red.log`）を確認し、組み立て済みノードからの命令コピーを止め、最終キー集合と宣言位置の供給元を確定する処理を追加した。`m03-green.log`: 25 test cases / 266 assertions、17入力差分ゼロ、exit 0。
- M04 は子の層・素材CLI・親CLIと子の層・子の層CLI・子CLI・親CLI伝播・親子CLIの7例を1件ずつ追加して全件 GREEN（`m04-0.log`～`m04-6.log`）。M05 は直書き行の位置を変えた2例とも GREEN（`m05-0.log` / `m05-1.log`）。
- M06 RED（`m06-red.log`）: 旧 eps_end、A2 在処の sync_interval.value、旧チェーン・層在処を含む余分な selections の3件が失敗。宣言の親子・採用状態を保持し、採用されたコピーだけで入力からキー集合を組み直す変更を実装中。上書き層の宣言を root として実行しない処理もこのスライスに含む。
- M06 GREEN（`m06-green.log`）: 25 test cases / 461 assertions、exit 0、17入力差分ゼロ。カタログ・clone後勝ち・上書き層差し替え・独立leaf保持の4追加分岐も順に GREEN（`m06-variant-0.log`～`m06-variant-3.log`）。
- M07 多段・直接宣言、M08 部分継承（カタログ内外）、M09 正順・逆順と記録、M13 生成プロファイル・cautious無し・内側dormantの3分岐を順に追加して GREEN（`m07-*.log`、`m08.log`、`m09-*.log`、`m13-*.log`）。
- M10 RED（`m10-red.log`）: 空のカタログ相互参照が例外にならないことを確認した。空の要求部分に限って生成元のコピーを辿る循環検証を追加し、ビルド中。
- M10 最初の修正は既存の同一 owner 継承で stack overflow を起こした（`m10-green.log` はこの失敗ログであり GREEN 証拠ではない）。同一 owner の全素材範囲へ戻る架空の問い合わせを除外する修正を追加し、再検証する。
- M10 修正後の `m10-green-v2.log`: 25 test cases / 637 assertions、exit 0、17入力差分ゼロ。
- M11 未定義カタログが成功する RED（`m11-2.log`）を確認し、存在検証を採用済み参照の最終キー集合へ移した。`m11-green.log`: 25 test cases / 658 assertions、exit 0、17入力差分ゼロ。素材の部分参照・カタログ外への空部分参照・後段 term が新規作成するプロファイルも1例ずつ GREEN（`m11-parts-*.log`）。
- M12 は10段成功（`m12-10.log`）と11段が誤って成功する RED（`m12-11.log`）を確認した。値の供給元の経路と宣言の生成親をキャッシュに保持し、共有親を重複せず数える深さ検証を追加してビルド中。
- M12 GREEN: `m12-green.log`（714 assertions）。10/11段の宣言逆順、外側nestedを含む深さ10/11、20個の兄弟termも個別に検証した。
- A06 自己供給の RED 後、同一prefix・子prefixへの供給を拒否し、書き戻しのない包含参照・独立部分の相互参照を維持した。`self-supply-green.log`: 809 assertions、17入力差分ゼロ、exit 0。
- A07 上書き層がroot選択も持つ RED 後、兼用を拒否した。`owner-conflict-green.log`: 857 assertions、17入力差分ゼロ、exit 0。
- A09 追加境界: 後段で生成されるプロファイルが`.$`を含むと未定義になる RED（`later-declaration-red.log`）を確認した。採用コピーから素材内のdormant宣言を再構成し、その定義が確定してから元の位置で実体化する。値全体の反復はせず、宣言の反復状態・キー生成経路・参照経路を循環診断に使う。`later-declaration-green.log`: 862 assertions、17入力差分ゼロ、exit 0。
- GREEN後、宣言展開から不要な逐次leafコピーを除き、最終キー集合・値評価へ集約した。記録のschemaと同一ownerのkeyを検証に追加し、設計文書を同期中。
- 整理後の `refactor-green.log`: 902 assertions、17入力差分ゼロ、exit 0。
- M14記録、Run 4→1→4の空overrides、M07の記録順、同一owner継承の基底後勝ち、M15のCLI葉・CLIチェーン・変数、owner兼用、素材宣言の後段差し替えを1件ずつ追加してGREEN（`records-*.log`、`m15-*.log`、`owner-term.log`、`later-declaration-replaced.log`）。
- P6追加境界: `X.$ = X.part > Other`のOtherがX.partへ書く自己供給が成功するRED（`self-supply-sibling-red.log`）を確認した。書き手だけでなく同一チェーンの全sourceを検査する修正を追加した。
- 自己供給修正後の `self-supply-sibling-green.log`: 962 assertions、17入力差分ゼロ、exit 0。
- CLI最優先の追加境界: 後段の層が同じ素材葉を書くとCLIの3が2に戻るRED（`cli-material-final-red.log`）を確認した。第1相で採用したCLI素材値・宣言を、コピーの供給元へ置き換えない形に修正した。`cli-material-final-green.log`: 968 assertions、17入力差分ゼロ、exit 0。
- 最終整理としてnested循環・深さエラーにもselection / term / resolvedを付け、不要な宣言種別の再検査を除いた。公開APIは変更していない。

### 受入条件と検証の対応

| 条件 | 検証箇所 |
|---|---|
| A01 | JSON設定例 M01 / M02 / M03（親値・追加キー・逆流なし） |
| A02 | M04の7分岐（素材、親、子、上書き層のCLI） |
| A03 | M05の行順2分岐、M09の全体・部分の順序2分岐 |
| A04 | M06の5分岐（カタログ、clone後勝ち、層の差し替え、独立葉） |
| A05 | M07の2分岐、M08（多段・部分・カタログ外） |
| A06 | M10、A06の自己供給・独立部分・値循環の各例 |
| A07 | M11の空層・未定義部分、A07の層root選択・owner兼用 |
| A08 | M12の正順・逆順10/11段、nested込み10/11段、20兄弟termと既存nested深さテスト |
| A09 | M13の4分岐、後段生成・変更される素材内宣言、既存相対term・nestedテスト |
| A10 | M08のtarget_policyと既存root順・nestedテスト |
| A11 | A11 / M14（Run葉の優先・Otherへの伝播） |
| A12 | A12 / M15、既存Run項順・run.$のCLI差し替えテスト |
| A13 | M06 / M07 / M09 / M14の完全配列比較、同一owner両key、空overrides・schema_version |
| A14 | 既存[config]のinclude、parser、trunk、CLI、1段値参照と異常系 |
| A15 | [prd072-baseline]でfrozenの17入力の実効key/value・selections/references順・空overridesを比較 |

### 最終結果

- Debug全体ビルド: `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'`、exit 0。Runnerと各Envの実行体までリンク成功。ログ: `.scratch/prd072-validation/final-debug-build.log`。
- 設定全件と固定比較: `& core/anet-core/bin/Debug/anet-core-test.exe '[config],[prd072-baseline]'`、seed 1352737562、105 test cases / 1593 assertions、exit 0。ログ: `.scratch/prd072-validation/final-config-tests.log`。
- JSON設定例は61件。M01〜M15の分岐に加え、後段生成・差し替えの素材内宣言、チェーン内の別termによる自己供給、CLI素材葉・宣言の最優先も検証した。
- A15: `baseline-v2/`を正本とする17入力すべてで、実効キー・値、selections / referencesの内容と配列順序が一致し、overridesは空。比較出力は`.scratch/prd072-validation/actual/`。初回の無効な`baseline/`は使っていない。
- `git diff --check`成功。変更したC++・JSON・MarkdownはUTF-8 / LF。
- 変更対象: `config_impl.cpp`、`config_test.cpp`、`testdata/config_resolver_prd072.json`、`docs/design/100_runtime_and_configuration.jp.md`、本メモ。既存のCONTEXT / ADRは新契約を確認し、今回の実装では変更していない。
- 公開API、Agent / Actor実装、現用設定、既存Run artifactは変更していない。staging / commit / pushは未実施。

### レビュー対応: 比較の再現性とキー順

- manifestと準備スクリプトを`core/anet-core/testdata/prd072/`へ置き、固定commitのresolverをリンクする採取専用`anet-config-baseline`と共有比較テストを追加した。旧`.scratch/prd072-validation/`は当時の記録として保持する。新手順の生成物は`.scratch/prd072-baseline/`で、AGENTS.mdの検証節から再生成できる。
- 値比較のJSON objectはキー順を保証しないため、`Map().Order()`を別配列として採取する。値・解決記録のgateと、`order_comparison.json`による行順診断を区別する。
- 既知の制限: `CollectOverlayRoots`はP3で不採用になるチェーンも事前走査するため、その中だけのtermが上書き層判定や自己供給エラーを起こしうる。現用設定にはない境界として保持する。
- `overrides.to`はRun第2相の値であり、CLIが同じ葉を上書きすると最終実効値とは異なる。最終値は`config_data.txt`を読む旨を設計文書にも追記した。
- 例外終了で検査されないM11の`absent`を削除し、`config_impl.cpp`に`<optional>`を直接includeした。
- 再採取: `prepare.py capture`、seed 160724608、72 assertions、exit 0。再生成した17件の値・解決記録は、実装前に採取した旧`baseline-v2/`とも全件一致した。
- 比較: `prepare.py compare`、seed 839160216、141 assertions、exit 0。値・selections/references・空overridesに加え、今回は17件すべてで`Map().Order()`も一致した。順序一致はこの17入力での実測結果であり、任意の設定について順序不変を保証するものではない。
- Atari 1（manifestの`atari-1`）は前後とも911キー、並びが異なる位置は0件。キーをLFで連結した配列のSHA-256は前後とも`d9c0cb6bc1e5f745aa07f24efe37ab8ba500295abb3b98ca277eeb7b4ac72ef9`。完全な配列は新手順の`baseline/atari-1.json`と`actual/atari-1.json`に再生成できる。
- 通常の`[config]`は104 test cases / 1507 assertions、exit 0（`.scratch/prd072-baseline/config-tests.log`）。既存baselineの再採取拒否、現行resolverでの誤capture拒否、両方の拒否前後でbaseline SHA-256が不変であること、再準備でfrozen sourceのmtimeが変わらないことも検証した。
- 採取専用・現行テストのDebugビルドは最終ヘッダ整理後もexit 0（`.scratch/prd072-baseline/build-verified.log`）。通常設定テストのseedは368293145。AGENTS.mdへ記載したprepare / configure・build / capture / compareの各手順を実際に通した。

### golden方式への簡素化

- PRD固有の採取ビルドを維持する負担をなくすため、採取済み17件を`core/anet-core/testdata/prd072/baseline/`へgoldenとして追加した。最初のコピーでは全件のSHA-256一致を確認した（元データはCRLF、合計1,684,183 bytes）。
- その後、利用者の明示指示によりgoldenだけをLFへ変換した。最終合計は1,652,267 bytes。元データとのJSON内容・配列順は17件すべて一致し、再採取はしていない。変換前後のSHA-256は`.scratch/prd072-baseline/golden-normalization.json`へ記録した。Git属性と通常テストのbinary書き出しでもLFを維持する。
- 既存optionがある間にローカル設定をOFFへ変更し、専用ターゲット・option・固定resolver抽出・互換ソース検査を削除した。専用exeは`core/anet-core/bin/Debug/anet-config-baseline.exe`だけを削除した。通常テストでgoldenを比較し、captureは契約変更時の明示更新だけに使う。
- 元の採取記録は、移動元・移動先の絶対パスが`.scratch`内であることを確認して`.scratch/prd072-baseline-before-golden/`へ退避・保持した。生成物なしからprepare / compareが成功した。17件すべてで値・解決記録が一致、overridesは空、`same_order: true`。LF変換後の再比較も140 assertions、seed 3031725410、exit 0（`golden-lf-compare.log`）。
- VsDevCmd経由の通常preset再構成・Debug全体ビルドはexit 0（`golden-build.log`）。`[config]`は104 test cases / 1507 assertions、seed 2550894907、exit 0（`golden-config-tests.log`）。LF書き出し変更後の通常テスト再ビルドもexit 0（`golden-lf-build.log`）。これらのログは`.scratch/prd072-baseline/`に保存した。
- スクリプトと環境変数付き通常テストの両方で、既存goldenのcapture拒否を確認した。拒否前後でLF変換後の17件のSHA-256は不変。テスト側は既存ファイル検出のassertionで停止した（`golden-lf-capture-refused.log` / `golden-lf-direct-capture-refused.log`）。
- resolver本体とマニュアル例JSONは作業前後のSHA-256が一致。現用比較器・ビルド定義・AGENTS.mdから旧機構の識別子が消えていること、`git diff --check`、追加ファイルのUTF-8 / LFを確認した。旧方式の記述は上記の経緯として保持する。PRD・ADR・CONTEXTの編集とstaging / commit / pushは行っていない。

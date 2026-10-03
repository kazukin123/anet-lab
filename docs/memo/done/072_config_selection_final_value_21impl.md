# 設定継承を差分適用へ統一する実装メモ

## 概要

正本は[PRD072](072_config_selection_final_value_10prd.md)の2026-09-13改訂とADR0042。旧方式の記録は[20impl](072_config_selection_final_value_20impl.md)に保持する。
全体のベース、部分選択、個別葉の順に強くし、別々の選択元は最終値を左から右へ差分合成する。公開APIは変更しない。

## 未決事項監査と前提

- ユーザー判断が必要なブロッカーは0。P1〜P7、M01〜M19、A01〜A15を実装契約とする。
- ConfigManagerの実パーサ、CLI、GetConfigData / GetResolutionJsonで検証できる。テスト専用APIは追加しない。
- 定義の有効化と値の依存解決を区別する。未選択の内側定義を実行せず、参照時に定義位置を保持して解決する。
- Runから供給された宣言は展開先rootを定義位置とする。CLIは指定したそのキーのみ最優先。
- PRD061のActor API移行、学習実験、staging・commit・pushは範囲外。

## 主な変更

1. 名前による上書き層判定、宣言のコピー先再実行、不採用チェーンの由来追跡を除去する。選択元の最終値とキー集合を依存関係から評価する。
2. 相対参照・記録を定義位置に統一する。Run先頭、1段値参照、Runの最終差分の記録、循環・自己供給・深さ10の検証を維持する。
3. PRDの112種・138定義を現用設定で照合し、弱い既定値を通常の@defaultsへ移す。値の文字列表記、コメント、重複定義順を保持する。
4. file・Run・bat・生成ツール・補助テストの全チェーン宣言を機械的に抽出し、該当ownerのdefaultsが先頭に残ることを検査する。
5. 固定commit、manifest、旧goldenは保持する。固定入力に同じ移行を再現し、値全一致、独立した新仕様の記録期待値、行順レポートで比較する。上書き拒否を維持する。
6. docs/designの設定ガイドとAGENTSの比較手順を同期する。不要な旧機構だけを削除し、比較資産・旧記録・無関係な変更は保持する。

## テスト

- Surface: ConfigManagerのファイル/include/CLI入力と公開値・解決記録。
- TDD順序: 個別葉の優先をtracer bulletにしてRED→GREEN。その後、部分指定、別ソースの差分保持、定義元参照とCommon、最終値伝播・内側定義、Run/CLI・診断を1挙動ずつ検証する。
- M02/M05/M06/M07/M09/A07/A13の期待値はPRDの理由と対応づけて明示改訂する。結果の一括採取で期待値を承認しない。
- 既定値のE1/A2上書き、個別指定、CLIの3種類、Run/CLIで選択先が持たないdefaultsの保持を検証する。

## 検証

VsDevCmd経由で通常x64-Debugを再構成・ビルドし、[config]全体と17入力を実行する。全チェーン静的検査、manifest整合性、capture拒否、goldenハッシュ不変、旧識別子残存検索、git diff --check、UTF-8/LF（batはCP932/CRLF）を確認する。
再実装前の退避とログは`.scratch/prd072-differential/`へ保存する。

## 変更・クリーンナップの記録

- `config_impl.cpp`を858行から574行へ整理した。`CollectOverlayRoots`、`IsDeclarationActive`、`FinalizeDeclarations`、`RefreshDeclarationInputs`、上書き層のroot選択禁止、コピー先での宣言再実行・不採用チェーンの由来追跡を除去した。定義位置の参照、必要な定義の有効化、キー集合、値の供給元、依存検証で新契約を実装した。
- 旧実装用の専用ビルドoption・ターゲット・固定resolver抽出処理は現用ソースに存在せず、`anet-config-baseline.exe`も存在しない。今回生成したPythonキャッシュ2ファイルは削除し、専用ディレクトリをignoreへ追加した。
- 比較器・manual fixture・旧golden・manifestは新契約でも必要なため保持した。`.gitattributes`のJSON/LFと`.gitignore`のscratch除外も比較資産の運用に必要なので保持した。20implと過去のscratch採取記録は判断履歴・採取証拠として削除しない。
- 現用設定の移行は112種類・138定義行でPRDの棚卸しと一致した。変換前ファイルと`migrate.py`の変換結果を照合し、右辺・コメント・重複定義順、実験値を保持した。file・Run・bat・Optuna生成・補助テストの選択チェーンへdefaultsを明示した。DropMerge_optuna.txtのbackend選択も検索で拾い、seedの明示指定は保持した。
- 共有agent.txtの初期選択も全経路でdefaultsを明示するため、`DefaultDQNAgent.@defaults : $ =`を通常の空プロファイルとして定義した。env側の既定葉がない入力でも未定義参照にせず、既定葉がある入力では同じプロファイルへ追加する。予約名・暗黙補完はない。138行とは別に、この定義とcommonのbackend/app初期選択を追加した。
- 設定ガイドを実装と同時に同期し、弱い既定値はdefaults、root直書きは強い個別指定、チェーン置換時もdefaultsを先頭に明示する規約を追加した。PRD・ADR・CONTEXTの合意済み契約は変更していない。

## テスト改訂と実行結果

最初に個別葉の`own`が`base`へ戻るREDを確認し、最小修正後にGREENを確認した。次にA2がその在処で解決されないREDから、定義位置の差分合成へ移行した。M01の別の葉を通る外側選択を循環と誤判定した点、同一場所の相対参照で架空の内側定義を辿った点、深さエラーの定義位置不足を、それぞれ実行結果から修正した。最終結果は以下のログで確認する。

| 対象 | 改訂・確認内容 |
|---|---|
| M02 | 外側A2より参照先の個別葉を優先し0.05。親のCLI伝播分岐は保持 |
| M05 | profile内のevalは0.01、root個別指定のeval_targetは0.02。行順逆転でも同じ |
| M06 | eps_end・A2自身の生成葉を保持。右側が持つclone_model等だけを上書き。旧チェーンも元の定義位置で記録 |
| M07 / M09 | 部分指定を全体より優先。M07の直接宣言分岐は0.05、M09逆順もGreedy |
| A07 / A09 / A13 | rootを持つ通常prefixを許容。直接定義したprofile選択が供給された定義より強い。profile自身の個別葉がベースより強く、keyはEnv.@a.$ |
| M11 / M12 | 未定義・深さエラーにコピー先でなく定義位置を要求 |
| M16〜M19 | 定義元相対参照、Common/A2/Library.Common、同一キー再指定と差分合成、別ソースの右側優先を追加 |
| 既定値・CLI | defaultsのE1/A2上書き、同値5キー、file/Run/CLI置換時の不足キー保持、個別葉、CLIの3種類、暗黙補完なしを追加 |
| 既存C++例 | trialの選択キー再指定は個別BaseStructureを変えない。nested記録を@iqn等の定義位置へ変更。値参照記録のsource順を同期 |

- manual fixtureは85入力。`config-final.log`: **104 test cases / 1694 assertions、全成功**。追加の競合WARNは実装していない。
- `debug-build.log`: VsDevCmd経由の`cmake --preset x64-Debug`と通常Debug全体ビルド、exit 0。runnerと各環境のテスト実行体までリンクした。
- `optuna-workspace.log`: 16 tests、OK。`batchrun-fatal.log`: 標準スクリプト成功。
- `chain-audit.json`: 抽出603件、defaults対象98件、違反0。未選択Runとコメントの切替例も対象。親ownerだけにdefaultsがある部分選択、独立fixture、旧goldenは理由付きで除外。過去Run・ローカルworkspaceは現用ソース範囲から除外する。Run・batを含む正例/負例6件でも検査器を確認した。

## 固定17入力・診断・行順

`prepare.py`は元のcommitとmanifestを保持し、原本と移行済み設定を別々に作る。`expected.py`は宣言と旧goldenの参照値から記録期待値を検証する。期待値の再採取・自動更新はしない。`resolution/`へ新仕様の期待値を保存した。

- `compare-final.log`: **17入力の全キー・文字列値、新記録、Run差分が一致、157 assertions成功**。値比較から非@の診断用prefixを除外していない。
- `selections`はdefaultsの追加、共有の空defaults選択、profile内宣言の定義位置に対応して変わる。`run.$`先頭は維持した。
- `references`は参照元キー順に安定化した。参照先・値は旧goldenと同じ。従来の入力順を期待していたC++例は、この診断順の変更として明示改訂した。
- `overrides`は旧実装のRun先行適用済み値との比較から、Run葉を除いた直前値との比較へ改めた。Atari 1〜6は順に6/7/8/15/4/4件、ImageCls convnextは1件、残り10入力は空。Run名、SiLU→ReLU、replay ratio等の従来から存在した変更を記録する。固定入力の同一キー・移行defaults・未定義を根拠に期待値を独立算出した。DropMerge QR51の分位数は@baselineも51を供給するため差分なし。空確認を一律削除せず、10入力は空を、7入力は具体的な記録を検証する。
- 行順は17入力とも変化する。Atari 1は911キー・値差分0、最初の差分位置37、相違位置874。位置のずれを除いた最長共通部分列は862キーで、49キーの移動に相当する。17入力の最長共通部分列は旧キー数の91.9%以上。弱い既定葉の移動と生成順で説明でき、選択の優先順位を行順へ戻す処理は追加しない。
- `validation/comparison_report.md`に入力別の件数、`comparison_summary.json`に全順序差分位置・旧新の選択キー・Run差分を保存する。全キー配列は旧goldenとactualの`map_order`で確認できる。

## 証拠保全・最終確認

- 旧golden17件、合計**1,652,267 bytes**のSHA-256は再実装前から不変。manifestの固定commitも変更していない。再採取はしていない。
- `capture-script-refused.log` / `capture-test-refused.log`: 既存goldenの上書きを両側で拒否（exit 1 / 42）。`manifest-script-refused.log` / `manifest-test-refused.log`: scratchのmanifest不一致を両側で拒否（exit 1 / 42）。manifestはfinallyで復元し、その後の通常compareも成功した。
- 現用ソース・比較器・CMake・AGENTSの旧識別子検索で、上書き層判定・不採用宣言処理・専用採取ビルドの残存なし。比較資産42ファイルはUTF-8 / LF / BOM無し。既存設定のBOMは元のものを保持し、batはCP932 / CRLFを保持した。
- 無関係な実験記録の並行変更と既存untrackedファイルは保持した。staging・commit・pushは行っていない。
- `git diff --check`はexit 0。保存後の21impl実ファイル・関連リンク・LFを再確認した。manifestは旧`.scratch/prd072-baseline/manifest.json`の採取条件とも一致した。

## 部分選択のレビュー確認

`Agent.@base : actor.[eval].policy.$ = Policy.@base`と`Agent.actor.[eval].policy.$ = Policy.@root`を置き、`Agent.$ = @base`から合成する例を追加した。正順・逆順ともroot側の`kind = direct`が勝つ。プロファイル内の宣言は自身の名前空間で値を確定し、Agentへはベースの値として渡る。元の宣言ownerのドット数がroot部分指定との優先を逆転させる現象は、この例では再現しなかった。resolverの処理変更は不要で、同じ場所の部分選択はroot側が強いことを回帰例で固定する。

`anet/json_util.hpp`の直接includeを復元し、Python cacheの除外を`__pycache__/`へ統一した。チェーン静的検査に必要なripgrepをAGENTSへ明記した。PRD P2の補足とP7の参照元キー順は文書担当の修正に委ねる。

共有の空`DefaultDQNAgent.@defaults`宣言は維持する。各envに共通するvalue/advantage feature出力の定義をagent.txtへまとめる整理は、PRD061 P2で同ファイルを変更するときの検討事項とする。

検証: VsDevCmd経由のDebugテストターゲットビルド成功。`config-review.log`は104ケース / 1706 assertions成功（manual fixtureは87入力）。チェーン検査は609宣言を抽出、対象98件、違反0。今回の変更ファイルはUTF-8 / LF、AGENTSの既存BOMは保持した。旧golden・設定・resolverの処理は今回変更していない。

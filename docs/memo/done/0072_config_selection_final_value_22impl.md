# PRD072 既定葉への再実装

## 承認済み計画

最新PRD072・ADR0042を正本とし、作業ツリーの既定値専用プロファイル方式を `?=` に置き換える。20impl・21implと旧golden・manifestは保持し、中間コミット、staging、commit、pushは行わない。

1. パーサからresolverまでの内部入力に強弱を保持する。既定葉、全体ベース、部分ベース、個別葉の順とし、Run・CLIの同一キー優先を維持する。同強度は後勝ち、混在は個別葉優先。Run内・CLI・選択宣言への `?=` はfail-fast。
2. 現用112種類・138定義を元位置の既定葉へ戻し、チェーン・bat・ツールの旧方式変更を除去する。追加されたAtari設定、値・コメント・重複順・BOM・batのCP932/CRLFを保持する。
3. 固定入力の変換と独立の診断期待値を改訂し、全17入力をキーでソートして厳密比較する。旧順序比較を除去し、check_default_leaves.pyで説明のない個別葉を0件にする。
4. 公開経路の失敗例から段階的に実装する。混在順序・include・追加ファイル・注入・プロファイル・Run/CLI・禁止構文・既存依存検証を確認する。設定ガイド・AGENTSを同期し、PRDには選択宣言への禁止だけ補足する。
5. VsDevCmd経由のDebug全体ビルド、設定全体、17入力、静的検査、capture拒否・manifest不一致拒否を確認する。旧goldenハッシュ不変、現用旧識別子、文字コード・改行、git diff --checkを確認する。
6. Atariのrun_20260913-183648_tmp_wiring_btrtrunknopoolを代替goldenとし、保存された全チェーンと400,000予算で新規配線Runを行う。app.run_nameだけを除外して全キー文字列一致を必須にする。HEAD差分と件数・ログ・Atari感度キーの数え方を記録する。

## 保全

編集前の対象ファイルとHEAD差分、旧golden・manifest・Atari代替goldenのSHA-256は `.scratch/prd072-default-leaf/` に保持する。

## 実装とクリーンアップ

PropertiesとConfigManagerの非公開集合で入力の既定葉を保持し、resolverへ渡す。解決後のConfigData、MakeSubConfigData、公開API、JSON形式には強弱を追加しない。個別葉と既定葉の混在は個別葉を採り、include・追加ファイルでも同じ強さだけ後勝ちとした。注入・Run・CLIの個別指定は既定葉属性を消し、既定葉はベースが値を供給しない場合だけ残す。プロファイルの完成値には強弱を付けて転送しない。

現用設定は112種類・138定義行を既定葉にした。appsのHEAD差分から演算子差分を除くと、Atariのユーザー追加2ブロックだけが残る（apps-extra-diff.json）。bat2本、agent.txt、DropMerge_optuna.txt、生成ツール、補助PSテストの旧方式による変更はHEADとの差分がなくなった。追加されたbtrtrunknopoolも保持し、既定値用termだけを外した。

check_chains.pyを削除し、check_default_leaves.pyへ置換した。現用file・Run・bat・生成ツールから宣言を抽出し、設定のスコープに属するowner配下を棚卸しする。通常選択元・プロファイル・意図したRun/CLI・Optuna出力先/seed・独立fixtureは理由付きで区別する。C++診断文字列を現用ownerの発見に使わない。検査器の5回帰例は、新規Runだけで定義されたownerの書き忘れ、個別葉漏れ、既定葉と素材、意図したRun葉、似た名前の兄弟prefixを確認する。

## 検証結果

- 最小の既定葉保持例は旧実行体でEnv.value欠落により失敗し、新実装で成功した。その後、混在順序、同強度後勝ち、プロファイル内ベース、右側の既定葉の完成値、Run/CLI/選択キーの禁止、file/Run/CLIの置換後保持を追加した。
- config-final.log: 設定105ケース / 1804 assertions成功。ファイル境界の例はinclude・追加ファイル・注入の強弱と公開部分Configを確認する。
- debug-build.log: VsDevCmd経由の通常Debug再構成・全体ビルド成功。debug-build-final.logで追加テストを反映した。Runnerも再リンクした。
- compare-final.log: 固定17入力の全キー・文字列値、新解決記録、Run差分が一致、156 assertions成功。行順の比較・WARN・専用レポートは除去した。
- 期待値はexpected.pyが入力宣言から独立算出する。選択記録は元の定義位置を維持し、取り下げた既定値用の選択が消える。参照は参照元キー順、Run差分は同じグラフでRunの当該葉を除いた直前値との比較。Run差分件数はAtari 1〜6が6/7/8/15/4/4、ImageCls convnextが1、他10入力は空で明示検査する。
- capture-script/test.logは既存goldenを拒否（exit 1/42）。manifest-script/test.logはscratchの不一致を拒否（exit 1/42）、finallyで原状復帰後にcompareを再実行した。旧golden17件、manifest、Atari代替goldenのSHA-256は編集前から不変。
- Atariの感度キーは入力1/2/3/4/6で4件、入力5で3件。env全体の和集合はbackend2件・game・[features]の4件であり、3と4は集計範囲の差である。

workspace追加検証では、旧優先順位を前提とする間接runs_dir上書きの例だけが失敗した。注入された個別葉へ選択の値は勝たないため、例外期待を「最終値が注入値に一致する」へ明示改訂した。直接指定とCLIによる変更の拒否は維持する。この改訂は新契約への期待値同期であり、workspaceの最終パス不変条件を緩めていない。

## Atari代替golden

旧run_20260913-183648_tmp_wiring_btrtrunknopoolと、新run_20260913-195441_prd072_default_leaf_wiringの925キーをソート比較した。差分はapp.run_nameだけ、残り924キーは文字列完全一致（atari-comparison.json）。保存されたrun.$の全12項、backendのnon-deterministic、app.batchrun > P1、400,000予算を使う。net・予算を除外しない。ログとdumpはそれぞれのRunフォルダに保持する。

最終の静的検査は22,850候補から619件を理由付きで棚卸しし、違反0。check_default_leaves_test.pyの5例も成功。workspace-final.logは16ケース / 91 assertions成功、debug-build-workspace.logは最終Debug全体ビルド成功。apps以外の無関係な実験記録、既存の.gitignore/.gitattributes変更は保持した。追加JSON/Python/C++テスト/本メモのUTF-8・LF・BOM無しと、goldenの保全を再確認した。

配線Runの正確なargvとworking directoryはwiring-command.jsonへ保存した。旧Runのresolution記録から全12項を確認し、同じapp/backend選択を既定葉の新契約で指定した。dumpの既定葉の位置と値の読みやすさも目視確認した。全HEAD差分の統計はhead-stat.txtへ保存する（未追跡の新規比較資産はgit diffの統計には含まれない）。

配線Runは20:08:39に停止・closeへ到達し、20:08:40にagent_close.anet（78,965,724 bytes）の保存を完了した。400,000予算の停止経路はRunnerAppのexp_step >= exp_exit_stepで発火する。最後の記録済みmetricsは399,872 stepで、停止境界の全stepは記録されない。別シェルから取得したProcessオブジェクトではExitCodeが空だったため、exit 0とは記録せず、自然停止・保存完了・プロセス消滅を終了証拠とする。最終のgoldenハッシュ検査とgit diff --checkも成功。staging・commit・pushは行っていない。

## コミット前レビューへの対応

Optuna追加設定はツールがDropMerge本体の後へincludeするため、DropMerge_optuna.txtの静的監査スコープへDropMerge.txtのownerを加えた。DropMergeEnv.seed_mode / global_seedを意図したseedの個別指定として記録し、存在しないtrain.seed / DefaultDQNAgent.seedの例外は削除した。実際のファイル組み合わせでgrid_cols = 40を加える回帰例は修正前に検出漏れで失敗し、修正後は意図どおり違反を返す。検査器の6例は成功した。

static.logは19:49の中間検査（違反36件）の記録として保持する。前回の619件・違反0件はその後の実行であり、今回の最終実行はstatic-final.logに保存した621件・違反0件。増加した2件はOptunaの実在seedキーで、default-leaf-audit.jsonにも除外理由を残す。ImageClsのコメントアウトされたチェーンは引き続きownerとして有効化しない。

AGENTSのソート比較説明を1文へ統合した。正規化したキーに残る?を不正な代入演算子として拒否し、通常file・プロファイル・CLIの空白入り? =を回帰例に追加した。Propertiesはfriendによる内部参照を廃止し、寿命を明記した読み取り専用DefaultKeys()を返す。公開ConfigDataへの強弱情報の追加はない。

レビュー修正後の検証: review-build.logはVsDevCmd経由のDebug全体ビルド成功。review-config-workspace.logは設定・workspace合計120ケース / 1907 assertions成功。review-compare.logは固定17入力 / 156 assertions成功。旧golden・manifest・Atari代替goldenのハッシュ不変、変更ファイルのUTF-8/LF、git diff --checkを再確認した。今回の修正は有効設定の値を変えないため、Atari学習Runは再実行していない。


## 2026-09-14: 代入演算子を用途別の運用へ統一

共通定義（ベース設定）・デフォルト定義の葉を`?=`、実験で選ぶ値・差分を`=`へ揃えた。コピー先のチェーンを調べないと演算子を決められない運用を避け、共有baseline、NN部品、メトリクス定義、チェーンのないEnvデフォルトも対象とする。動作モード選択、E1／A2、Run、CLI、選択宣言は`=`。resolver・公開API・JSON形式は変更していない。

DropMergeでは先頭の`vector_feature.structure`と`app.run_name`を`=`へ戻した。出力先2行、入力接続1行、ヘッド構造2行はAgentのベースなので`?=`を維持する。他envのRun名・動画codecも`=`に戻す。共有ファイルの全葉を機械的に弱くせず、commonのモードプロファイルやagentのアルゴリズム選択は`=`にする。

### 差分と棚卸し

- 今回の設定差分は13ファイル・1,981演算子。`=`→`?=`が1,971行、`?=`→`=`が10行。値・コメント・配置・重複順は維持した。`git diff --check`で検出された既存の末尾空白2行（GridMaze_muzero:181、nn:187）だけ同時に除去した。
- 現用設定の既定葉は2,000キー種類・2,099定義行。固定commitの移行は1,984種類・2,083定義行。固定commit後のAtari定義を含むため母集団は異なる。以前の112種類・138行の記録は当時の範囲として上に残す。
- `migrate.py`を同じ用途分類に更新。manifest・旧golden17件・解決記録期待値は変更しない。
- `check_default_leaves.py`は既定葉の`=`と実験指定の`?=`を双方向に検査し、ownerのない共有ベースも監査する。分類理由をJSONへ記録し、用途変更時は分類も更新する。これは設定運用の分類であり、resolverの名前による特別扱いではない。
- 設定ガイド・AGENTS・PRD・ADRへ同じ規則とDropMergeの例を反映した。

### 検証

証拠は`.scratch/prd072-assignment-convention/`へ保存した。変更前の全設定は`before-config/`、作業差分は`before.diff`、旧golden・manifestのハッシュは`protected-hashes.json`、全演算子変更は`changes.json`に保持する。

- 現用94入力の変更前後で全キー・文字列値が一致（除外キーなし）。解決記録・Run差分も94件一致。`cases.json`の内訳はmanifest相当17入力、7envの通常設定、全68個のRunプロファイル単独選択、DropMergeのOptuna追加設定、Atariの既存12項配線チェーン。ConfigManagerによる設定解決だけを実行し、学習Runは起動しない。
- 採取は通常の`anet-core-test.exe`を隔離した作業ディレクトリから起動し、`ANET_PRD072_CAPTURE=1`で新しいローカル証拠にだけ出力した。版管理goldenへの採取はしていない。実行スクリプト`capture.py`、比較スクリプト`compare.py`、ログ`before.log`/`after.log`と`comparison-core.json`を保持する。
- 絶対パス検査後、以前の検証生成物を`previous-validation/`へ退避。通常のprepare→compareを実行し、旧golden17入力の値・解決記録・Run差分が一致（156 assertions）。`golden-compare.log`。
- VsDevCmd経由の通常Debugビルド成功（ninja: no work to do）。`[config],[workspace]`は120ケース・1,907 assertionsが成功。`build.log`、`config-workspace.log`。
- 静的検査3,130件・違反0、回帰テスト8件成功。`static.log`、`static-test.log`。追加ケースはownerを持たないベースの見落とし、実験指定の誤った`?=`、実際のDropMerge構造選択を確認する。Optunaの実在seed2キーの許可とgrid_colsの検出も維持した。
- `prepare.py capture`は既存goldenを理由に拒否（exit 1）。旧golden17件とmanifestのSHA-256不変を確認。`capture-refusal.log`、`integrity.json`。
- UTF-8 / LF、設定の既存BOM保持を確認。bat・resolver・テスト実行体のソース・旧実験資料は今回編集していない。staging・commit・pushは実行していない。


## 2026-09-14: 環境別ファイルを`=`中心へ絞り直す

前節の広範な`?=`化は、環境別ファイルを読みやすくする目的と逆になった。共通ファイルのベース定義は維持し、環境別ファイルはデフォルト設定ブロックだけ`?=`、その他は原則`=`に絞り直した。将来の変更可能性だけでは既定葉を増やさない。前節は実施履歴として保持する。

### 変更と残存箇所

環境別7ファイルの1,135行を`?=`から`=`へ戻した。既存の値・コメント・定義順・重複順は維持し、デフォルト設定に開始・終了のコメント見出しを追加した。Atariのヘッド構造2キーと`output.[features]`は、NatureDQNのReLU選択とIQNのiqn_fusion選択を維持するため、理由付きの「Agentのデフォルト設定」ブロックにまとめた。その他の例外は不要だった。

| ファイル | 変更前の`?=`行数 | 変更後 | 残存箇所 |
|---|---:|---:|---|
| Atari.txt | 346 | 16 | Env 13行、Agent 3行 |
| CartPole.txt | 16 | 1 | Envのデフォルト設定のみ |
| DropMerge.txt | 443 | 42 | Envのデフォルト設定のみ |
| GridMaze.txt | 34 | 15 | Envのデフォルト設定のみ |
| GridMaze_muzero.txt | 55 | 15 | Envのデフォルト設定のみ |
| ImageCls.txt | 297 | 11 | Envのデフォルト設定のみ |
| LunarLander.txt | 60 | 16 | Envのデフォルト設定のみ |

共通ファイルを含めた既定葉は944種類・964定義行。固定commitの移行も同じ件数。共通ファイルのバイト列は変更前と一致する。`migrate.py`は同じ規則と見出し追加を固定入力に再現する。静的検査は環境別のNN部品やプロファイルを`=`として検査し、`?=`がデフォルト設定ブロック外にある場合も違反とする。コメント見出しは静的検査の規約であり、設定parserの構文変更ではない。

### 検証

証拠は`.scratch/prd072-env-simple/`に保存した。`before-config/`に設定を保全し、変更前後の94入力を通常テスト実行体のConfigManagerで解決した。隔離した作業ディレクトリのローカル出力を採取し、版管理goldenへ採取していない。学習Runは起動していない。

- `before.log` / `after.log` / `comparison-core.json`: 現用94入力すべてで全キー・文字列値一致。除外キーなし。解決記録・Run差分も94件一致。
- `golden.log`: 絶対パス確認後に旧生成物を`previous-validation/`へ保持して退避し、prepare→compareを実行。旧golden17入力の値・解決記録・Run差分が一致（156 assertions）。
- `config-workspace.log`: 設定・workspaceの120ケース・1,907 assertions成功。
- `static.log` / `static-test.log`: 静的監査3,130件・違反0、回帰テスト11ケース成功。ブロック外の既定葉、ブロック内へ紛れた環境別NN部品の`?=`、閉じていないブロックの検出を追加。Optunaのseed例外とgrid_cols誤指定の検出は維持。
- `integrity.json`: 共通設定不変、既存コメント・有効定義順保持、移行再適用の冪等性、全既定葉がブロック内に収まることを確認。旧golden・解決記録期待値・manifest・config系C++ソースの保全ハッシュ不変。
- UTF-8 / LF、既存BOM保持、`git diff --check`成功。PRD・ADR・設定ガイド・AGENTSを同じ運用へ同期し、保存後の実ファイルを確認した。

resolver・公開API・JSON形式・bat・無関係な実験記録は変更していない。staging・commit・pushは実行していない。


## 2026-09-14: 見出しコメントの強制を撤回

デフォルト設定を区別する運用は維持するが、専用の開始・終了コメントや配置は要求しない。設定ガイド・PRD・ADR・AGENTSの強制記述、静的検査の見出し範囲判定、移行処理のコメント自動挿入を削除した。削除済みの設定コメントを復元せず、全configファイルの変更前後SHA-256一致を確認した。上の見出しを使った検証記録は当時の履歴として残す。

回帰テスト10件成功（コメントなしの既定葉、任意コメント、移行時に見出しを追加しないことを含む）。静的検査3,130件・違反0。生成物を保全・再準備した旧golden17入力比較も成功（156 assertions）。ログとconfigハッシュは`.scratch/prd072-optional-comments/`。UTF-8 / LF、保存後の設計文書、旧見出し処理の残存なし、`git diff --check`を確認した。

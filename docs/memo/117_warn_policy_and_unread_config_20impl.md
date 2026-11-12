# WARN の意味の線引きと、読まれない設定の検出 実装メモ

## 概要

[PRD117](117_warn_policy_and_unread_config_10prd.md) の D1〜D14、A1〜A11 を実装・検証する。新しい設定キーは足さない。未読設定キーの一覧は WARN、実体化したカタログ項目の未読設定キーは構築時の ERROR とする。休止と checkpoint の設定値の置換は INFO、欠けた部分の初期化は WARN とする。

## 未決事項監査

- ユーザー判断が必要なブロッカー: 0。
- repo evidence で解決済み: config.hpp/config.cpp の ConfigData のコピーは値だけを渡し、MakeSubConfigData と NN の走査は相対キーを作る。ここへ元キーの読み取り記録を引き継ぐ。ConfigManager は実効 map を返すため、返却データの記録を共有する。
- repo evidence で解決済み: RunnerApp は RunManager の背景構築のあと画像登録と画面を初期化する。RunManager の末尾では一覧を出さない。
- repo evidence で解決済み: NN の config_profile は定義の走査時に全項目を検証している。参照された項目だけを検証する PRD の契約へ移す。
- 2ximpl へ固定: 記録と通知の共通処理は既存の config.* / log.* に置く。コピー・相対設定には元キーの対応を持たせる。列挙と Has は読み取りに数えない。読み取りと一覧の共有状態を mutex で保護し、step の通常経路には記録処理を加えない。
- 2ximpl へ固定: 通知済みの記録はログ出力先が所有する共通部品 `NotificationFilter` に置く。`warn_once` / `info_once` がソース位置と任意の対象名を wxLog の記録の付加情報へ載せ、FileLogger・Logs pane・テストのログ捕捉先が同じ部品で重複を除く。Run の開始のリセットも同じ経路で両出力先へ届け、制御用の記録は表示・ファイルへ出さない。通常の記録の書式は変えない。新しい可変の global / static は置かない。リポジトリ内の wxLog の出力先を検索し、付加情報の受け口を揃える。出力先が複数あることを理由に警告の表示件数が増えないことは、FileLogger と GUI の両経路で検証する。
- 2ximpl へ固定: カタログの閉じた検証は新しい読み取り範囲で行い、別の構築・別の type の既読結果に検証を依存させない。上書きされた元設定は消費済みとする。
- 2ximpl へ固定: Configの読み口が完全キーを使うカタログは、CreateReadScopeにprefixを渡して、そのprefixの完全キーだけを新しい範囲へ取り出す。default側とoverride側の範囲を合わせ、既存の型付きConfigの読み方と設定記録のキーを保ったまま検証する。Actorのように複数回に分けて読むものは、最後の読み取りまで同じ範囲を保持する。
- 2ximplへ固定: run.evalのenv配下はEnvの型付き設定が読むため、RunManagerの内部では入力全体の新しい読み取り範囲を使い、evalの子設定とBatchEnvBuilderへ同じ出典を渡す。各定義の宣言とEnv設定の読み取りが終わってから、その定義を閉じたスキーマとして検証する。BatchEnvBuilder::ValidateConfigは既にdefinition-onlyのtagにも呼ばれるが、現在はBatchEnvFactoryだけへ届く。SingleDiscreteEnvFactoryにも同じValidateConfigの境界を設け、各Env factoryが既存のConfig DTOだけを読み取る。Env・manifest・描画資源・乱数は作らない。これによりdormantのenv設定も型と未知キーを検証し、env配下を一括で既読や在庫にする抜け道を作らない。
- 2ximpl へ固定: panel とモードで不要な設定の扱いは「前提」の持ち主別の表に固定する。A2 で新しく見つかった持ち主は、読む時機と値域検証・休止の扱いをその表へ追記してから実装する。PRD §9 の 3・5 の結果は検証欄へ積む。
- 2ximpl へ固定: Atari の worker 併用は BatchEnvBuilder が実際に選ぶ方式を確定する境界で判定する。Env 固有の条件を汎用層へ直書きせず、既存 factory の検証境界で渡せる情報を確認して実装する。既存境界で判定できなければ PRD §9 の 9 の WARN 1 回を採る。
- repo evidenceで確定: AtariのSingleDiscreteEnvFactoryにはworkerの方式を受け取る既存の検証境界が無い。CreateSingleEnvには実際に解決したworkerの方式が渡らないため、汎用層へAtariの条件を足さず、display_screenの既存WARNをRunに1回にする。220/040の併用の契約は変えない。
- PRD 範囲外: PER の一様への代替を廃止する判断（issue-20261111-e90762）、無人 Run の WARN 集計（issue-20261111-9e6e27）、値の参照先検証、構築後の ERROR の棚卸し。

## 主な変更

1. 公開ヘッダ config.hpp と config.cpp / config_impl.*: 読んだ元キー、選択元の在庫、相対設定の出典を保持する。カタログ読み取りの検証と未読キーの一覧を同じ汎用機構に置く。ConfigManager、NN、Actor/nn_forward、評価定義・schedule、metrics の読み口を移行する。
2. 公開ヘッダ log.hpp と log.cpp: 呼び出し箇所と任意の対象名で 1 回だけ通知し、Run の開始で記録を空にする。複数スレッドと NN 複製を扱う。PRD §5.5 の通知と個別の重複抑止を移行する。
3. trainer、RunnerApp / RunnerFrame、画像登録、各 panel: 構築の開始と終了をつなぎ、遅れて読む設定を一覧の前へ移す。未使用クラスと休止の在庫は所有者が宣言する。
4. DefaultDQN / RND / Rainbow / scaler / metrics / config: D7・D11 の fail-fast、D8・D10 の INFO、D12 の既知 RND キーの NaN、動画 auto の代替の固定を実装する。閉じた検証で見つかる既存 fixture の誤りを直す。
5. 出荷設定の mergin_x/y を margin_x/y に直す。MuZero の未対応の終了時保存を求めない設定にする。A2 で見つかる設定の誤りを直す。
6. config_baseline_test.cpp と testdata/prd117/README.md: 未使用項目も factory / 読み口へ通す出荷設定検査、fixture の誤りによる失敗、A2 の全構成・停止条件・ログ集計を記録する。block は module factory、profile / nn_forward は公開の設定・Network / Agent 構築経路で検査する。
7. A2で見つかった設定の誤りを、実効値を保って整理する。GridMazeの旧body選択と未使用tau指定を削除する（ユーザー回答済み）。ImageClsの未選択main_const phaseを削除し、ImageClsAgentは教師ラベルを運ぶvectorをNNの入力仕様に含めない。vectorは既にImageClsLearnerが正解ラベルとして読み、NNのbranchはgridを入力としている。Envの観測とLearnerのラベル、NNの重み・構造・出力は変えない。既存の公開Agent構築・学習率更新テストへ、教師ラベルを未使用NN入力としてWARNしない確認を足し、REDからGREENまで検証する。
7. PRD §5.8 の規約・用語集・設計文書 jp/en を追随させる。用語の対応は PRD D14 の表を正本とし、指定の現用範囲で表記だけを言い換える。過去の記録と英語の識別子は保持する。skill の実体コピーを同期する。
8. PRD の対象課題 6 件の状態・対応内容・AI 判定と派生一覧を工程に合わせて更新する。解決への変更・ステージ・コミットはしない。

RNDの既知診断キーは、生成factoryのInitialDiagnosticsが初期値NaNとともに宣言する。推定器とDefaultDQNの両方が同じ宣言を使い、無効化のためだけに推定器やNetworkを作らない。bonusとlearn_countはfacadeが所有する。無効・learner停止時のGetScalarは、checkpointを読んだ後も既知キーをNaNで返す。

登録済みで作らなかったクラスの在庫は、Agent / Single Env / Batch Envのfactoryが所有prefixの集合を宣言し、各repositoryが選択したclass以外のprefixを分類する。既定はclassの設定prefixで、ImageClsEnvはImageDatasetも宣言する。選択したfactoryも所有するprefix以下は分類から除き、共用する設定の誤記を隠さない。ConfigDataのprefix分類に除外prefixを渡す形で、親子のprefixが重なっていてもキーごとに判定する。RunManagerは選択したclass IDを渡すだけとし、個別Agent / Envの名前を汎用層へ書かない。

動画autoの代替の固定は、MetricsLoggerがRunごとの選択結果を所有し、VideoLoggerの構築へ借用させる。既存のResolveVideoCodecによる寸法の判定を使い、autoで選んだh264_nvencの起動に失敗したときだけlibx264への固定を記録する。小さい画像で最初からlibx264を選んだだけでは固定しない。固定後のautoの選択はlibx264、明示したcodecは従来どおりとする。固定の記録はatomicで共有し、wxのメインスレッドを待つffmpeg起動中に共通mutexを保持しない。固定より前に済んだ選択の起動は既存どおり行う。WARNは共通の呼び出し箇所でRunに1回、新しいRunのMetricsLoggerでは選択結果も新しくなる。

この検証にはtestdata/prd117の小さなffmpeg代役を使う。nvencの要求だけ起動失敗させ、libx264はstdinを読み、要求されたcodecを記録する。検証用の子プロセスのPATHにだけ代役を置き、MetricsLoggerの公開LogImageから、小さい画像→大きい画像2タグ→次のRunの大きい画像を通す。代替後にnvencを選び直さないこと、WARN1件、Runを変えるとautoの選択を再開することを実際の起動経路で確認する。製品側へテスト用のcallbackや設定キーは追加しない。

翻訳用語表reports/translate/glossary.jsonも、CONTEXTの正式な語へ追随させる。既定キー・読まれない設定キー・在庫を登録し、廃止した日本語の語の項目は残さない。

profile の構築情報は block と同じく出典を保つ ConfigData を保持し、参照時の factory で型付きの補間規則へ変換する。未参照の定義の不正な数値や未知キーを Run で解析しないためである。旧 DTO の type/start/end/has_* は残さず、内部の補間実装だけが型付き値を持つ。ToJson の profile の値は他の生の設定と同じ文字列で出力し、未参照の不正値も変形せず記録する。使用箇所の検索では nn_impl.cpp と nn_test.cpp だけが旧 DTO のフィールドを使っている。補間の計算結果と実効設定の文字列値は変えない。

出荷カタログの検査は実効キーのカタログ構文からNNのprefixを集める。blockはA9に従い補間印を仮の数値に置き換えてからmodule factoryへ渡し、その読み取り範囲の未読を検査する。profileの有無や参照関係をblock単独の検査の前提にしない。profile自体は別に、検査用Linearのdouble値のinit.constant_valへ参照し、既存のNetworkStructBuilder経路で型付きprofileを作る。branchのprofileは本番と同じく全体の値へ上書きして検査する。NN実行設定の全件検査は、Noisy OFFの検査専用の小さいNetworkのCreateForwardExecutorへ全項目を渡す。ONの必須項目や時計の条件を未使用の項目へ課さず、既知の明示値と未読キーを検査する。ONで実際に使用する構成の必須項目と時計は本番の構築・既存のNNテストで検査する。検査のNetworkとModuleRandomSourceはテストだけが所有し、実際のRunの乱数や資源を使わない。キーの本数・項目名・値のsnapshotや、テスト独自の受理キー一覧は作らない。未知キーを入れた未使用項目で同じ検査が拒否することに加え、branchだけのprofileを使うblockと、OFFで有効なmode=sampleだけの省略形が受理されることを確認する。

名前付きNN実行設定は、NetworkForwardExecutor::Implが新しい読み取り範囲を持つ不変のNetworkForwardConfigを所有し、各具象部品へその参照を渡して準備したあとに未読キーを検査する。一時変数の参照を部品へ渡して寿命を短くしない。既定要求では従来どおり準備・箱・RNGを作らない。Noisy OFFでも明示した既知の値を読み、型・列挙・既知の値域を検証するが、利用しない実行方式の必須項目は要求しない。Noisy ONでは既存の必須項目と時計の契約を保つ。

PRD §5.3 に従って追加するカタログ:

| カタログ | 読み口と扱い | Run で実体化しない項目 |
| --- | --- | --- |
| `metrics.graph.[tag]` | ObserverFactory の graph 定義の解析。scalar / trace と同じく、定義した全項目の未知の値・属性・下位キーを構築時に拒否する | 無い |
| `ImageDataset.[key]` | ImageDatasetConfig::Resolve と ResolveImageDatasetCatalog。ImageCls の構築時に全定義を型付き設定へ解決・登録する境界で閉じたスキーマを検証する。画像の読み込み・cache の確保は前倒ししない | ImageCls が使うときは無い。全定義を RegisterCatalog する既存契約を保つ |
| `AtariEnv.ram_metric.[game]` | AtariEnvConfig が全ゲームの定義と番号の束縛を解析し、全ゲームの既知番号も登録する。値の解析に元キーの Get を使い、既存の厳密なキー文法の検査を保つ | Atariを使うときは無い。既知番号を全ゲームから宣言する現在の契約を保つ |

根拠: observers.cpp の graph 定義の読み口、core/envs/imagecls1/src/ImageData.hpp/.cpp の ResolvedImageDatasetCatalog と ResolveImageDatasetCatalog。どちらも定義した全項目を読むので、D13 の未参照項目を持つ 4 カタログの出荷設定検査へは加えず、各領域の設定読み取りのテストで確かめる。

Atari RAMの追加の根拠はAtariEnv.cppのAtariEnvConfig。全ゲームの定義を解析し、束縛した全番号をknown_ram_metric_numbersへ登録している。生成しないEnvの在庫はfactoryのクラスprefixで扱う。値の解析の追跡を補うだけで、ゲーム選択・RAM採点・乱数は変えない。

## テスト

- Public interface / surface: ConfigManager → ConfigData → 元キーの未読一覧、カタログ factory / Config、RunManager / RunnerApp の構築、既存ログの捕捉、GetScalar、checkpoint の読み込み。
- 優先 behavior / TDD 順序:
  1. tracer bullet: 設定ファイルを解決し、子設定の値を読んだあと、綴り違いだけが元の完全キーの一覧へ残る。RED → 最小実装 → GREEN。
  2. コピー、Has、走査、上書き、選択の入力に残る prefix、CLI / ?=、在庫、一覧 0 件・1 件・構築継続を一つずつ確かめる。
  3. 参照した block の誤った out_channels で停止することを確かめ、コピーと在庫の基本を固める。
  4. 通知の同じ箇所・対象・スレッド・Run 再開始を確かめる。§10 の順序に沿って共通の通知部品を先に用意し、未読キーの一覧から A2 の調査へつなぐ。
  5. ほかのカタログ、未参照の block / profile、dormant、使わないモードのキー、RND の休止・無効時の NaN と学習件数、D7 の各停止入力、checkpoint の INFO / WARN を一つずつ RED → GREEN にする。
  6. 出荷設定検査は内容の本数・名前・値を固定せず、すべての未使用項目に未読キーが無いことを検証する。誤り fixture で検査自体が落ちることを確かめる。
- refactor は GREEN のあとだけ行い、関係するテストを回す。

## 検証

- ビルドは AGENTS.md の VsDevCmd.bat 経由で Debug。標準出力・標準エラーは .scratch/prd117/ のログへ保存し、終了値と失敗箇所だけを返す。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```

- anet-core-test の関連タグ → 既定スイート → [shipped_config]、AnetRLRunner-gui-test [log_panel]。PRD093 の既知失敗以外の失敗を残さない。
- A2: 7 Env の既定、11/12/13_batch_run.bat の run.$ の異なる全構成（goto で飛ぶ構成も含む）、Atari evalonly。学習 Run は learn scalar を1件以上記録（無い構成は learn step 1以上）、評価 Run は最初の session 完了まで。アプリの終了予算を使うため、最初の記録より後まで進むことを許す。正規終了の通知を含めて : [W] を 0 件にする。Atari 既定で保存した checkpoint を evalonly へ渡す。
- 設定変更前後の全キー・文字列値の比較、check_default_leaves.py、D14 の git grep、LF / CP932+CRLF / BOM、skill の同期、英訳の translated-from を検証する。
- 自己点検は docs/quality/impl.md の全項目を適用範囲ごとに通し、独立した anet-review-agent に計画と今回の実装を依頼する。差し戻しは直して同じ相手へ再依頼する。

## 前提

- 2026-11-12のユーザー判断: 11_batch_run.batのNMRND / SERNDは旧exeとfrozen設定を使うため、旧ファイルを保ち、A2専用コピーを現行キーへ移行して検証する。共有intrinsic.rnd.netをtarget.net / predictor.netの両方へ展開し、元の設定値とNN構造を保つ。PRD117の綴り違いの修正もコピーへ適用する。旧exeの動作再現ではなく、現行実装の受入である。製品側の互換コードや旧名aliasは加えない。

- 開始時点に .gitignore、11/12_batch_run.bat、220_atari_env.jp.md、wiki の複数ページ、課題 4 件などの既存変更がある。既存差分を保持し、今回の差分を記録する。
- 実効設定の値は D7・D11 と明示された出荷設定の修正以外では変えない。通知と検証の変更で乱数を消費しない。
- データセット不足等で受入を実行できない場合は未達として記録する。未実施を成功扱いしない。

持ち主別の読み取りと休止:

| 持ち主・prefix | 構築時に固定する扱い |
| --- | --- |
| `Conv2dPanel.conv2d.*` | panel を開く前、RunnerFrame の構築時に DTO の読み取りと検証を行う。描画資源と window は開くときに作る |
| `app.batchrun.*`、`app.online.*` | RunnerApp が構築時に型付き設定として読み、指定値を検証する。モードを選ばなかったことだけで不正値を隠さない。実効側の `app.*` の値は変えない |
| `app.eval_trace_panel.*`、`app.train_panel.*`、`app.eval_panel.*` | 現在のアプリの構築経路で読み、一覧より前に検証する。panel を後で開く経路でも設定の読み取りは先に済ませる |
| `metrics.image.*` の項目以外の共通既定 | ImageProviderManager が、画像の項目の有無にかかわらず構築時に型付きで読み検証する。個々の項目は D3 の検証へ通す |
| `ImageDataset.*` | ImageCls を作るときは Dataset の読み口が構築時に読む。ImageClsEnv を作らない Run では、ImageClsEnv の factory が自分の所有する追加の設定 prefix として宣言し、登録済みで作らなかったクラスの在庫（D4 の 3）にする。汎用層へ ImageDataset という名前の分岐を置かない |
| `DefaultDQNAgent.intrinsic.*`（`intrinsic.enabled=false`） | enabled を読んだあと、残りを休止の prefix として宣言し、prefix と `DefaultDQNAgent.intrinsic.enabled=false` を INFO で 1 回知らせる |
| RND 有効かつ `learner.enabled=false` | checkpoint と推論に必要な RND 設定は読む。補助学習の休止は `DefaultDQNAgent.learner.enabled=false` を理由に INFO。D11 の lane 数との比較は行わない |
| `DefaultDQNAgent.skip.*` と `DefaultDQNAgent.learner.skip.*`（`skip.enabled=false`） | enabled を読んだあと、生成しない skip 機能の prefix を休止として宣言し、`DefaultDQNAgent.skip.enabled=false` を INFO で 1 回知らせる。Actor のカタログ項目の skip 設定にはこの省略を適用せず、D3 に従って読む |
| `DefaultDQNAgent.learner.grad_noise.*`、`learner.plasticity.probe.*`、`learner.policy_churn.*` | 購読が無くても構築時に値を読み、各値の局所的な値域を検証する。購読・AMP・損失の構成に依存する条件は、それらが揃う既存の準備境界で検証する |
| 実体化したカタログ項目の mode で使わないキー | D3 に従って読み、値を検証する。未参照の項目は Run で検証しない。通常 prefix の休止でカタログの検証を回避しない |

表の判断は、新しい設定キーを足さず、既存の読み口と有効フラグを使う。A2 で表に無い持ち主を検出したら、同じ行形式で扱いを固定し、検証結果とともに経過へ積む。

## 経過

- 2026-11-11: PRD117、入口 skill、プロセス、関連コードを読み、未決事項監査を行った。通常サンドボックス内では Git の cwd アクセスが拒否されたため、読み取りを承認済みのサンドボックス外実行で確認した。

## 実施した検証と結果

- 2026-11-11 TDD: tracer は API 未実装によるビルド失敗を確認したあと、Debug ビルド成功、1 テスト・4 assertion 成功（.scratch/prd117/red-tracer-build.log、green-tracer-build.log、green-tracer-test.log）。
- 2026-11-11 TDD: 上書きの元キーの記録は 2 assertion の失敗を確認（red-overwrite-test.log）。初回のリンクは LNK1201 になったため PDB/ILK を .scratch/prd117/failed-overwrite-link.* へ退避して再生成し、ソースを変えず再リンクが成功した（red-overwrite-relink.log）。空き容量は約232GB、残存テスト/リンカー無し。リンク失敗を RED として数えていない。
- 2026-11-11 TDD: 上書きの修正後、2 テスト・9 assertion 成功（green-overwrite-build.log、green-overwrite-test.log）。
- 2026-11-11 TDD: block テストの最初の入力は必須の branch が欠けていたので直した。正しい入力で「例外が出ない」という RED を確認（red-block-fixture-test.log）。構築時の未読キー検出を追加して GREEN、診断と別の読み口の既読状態からの独立も加えて 3 テスト・11 assertion 成功（green-block-diagnostics-build.log、green-block-diagnostics-test.log）。
- 2026-11-11: NN の関連テストは 139 テスト・2,470 assertion 成功（nn-after-block.log）。現用 7 Env の既定の全キー・文字列値を .scratch/prd117/resolved-before/ に採取した（resolved-before-capture.log）。先に試した履歴用の PRD072 固定入力は現用に無いプロファイルを参照して止まったため、現用の 7 Env を読む専用の採取入口を使い直した。
- 2026-11-11: D14 の文書の表記と AGENTS.md の避ける語の表、D1・D2・D6 の規約の文を変更した。コード・設定の用語、翻訳元情報、wiki ログ、全範囲の検証は未了。
- 未実施: A1、A2、A3 の block 以外、A4〜A11 の残り、既定スイートと GUI テスト、設定変更後の比較、全項目の自己点検、実装のレビュー。実装途中のためであり、受入完了とは扱っていない。
- 2026-11-11 計画レビュー 1 回目: 差し戻し。旧用語の記載、持ち主ごとの扱いの未固定、具体的なビルドコマンドの欠落、検証記録の矛盾の 4 件を修正した。原文: .scratch/prd117/review-plan-r1.md。
- 2026-11-11 計画レビュー 2 回目: OK。前回の 4 件は解消。計画だけの判定であり、実装の受入は未完了。原文: .scratch/prd117/review-plan-r2.md。
- 2026-11-11 TDD: 在庫の API 未実装による RED のあと、コピー元と相対設定での分類共有、閉じた読み取り範囲の独立を実装。4 テスト・17 assertion 成功（green-inventory-build.log、green-inventory-test.log）。
- 2026-11-11 計画レビュー 3 回目: 通知済み記録を出力先が所有する方式と、追加カタログの実体化境界も OK。背景スレッド、Run 開始のリセット、File / GUI の件数、制御記録を表示しないことを実装レビューで確かめる。原文: .scratch/prd117/review-plan-r3.md。
- 2026-11-11 TDD: 選択の宣言履歴が失われて 8 件が未読になる RED を確認（red-selection-inventory-test.log）。ファイル・注入・追加ファイル・CLI の宣言の通常 prefix を在庫にし、誤記の 2 件だけを残した。5 テスト・22 assertion 成功（green-selection-inventory-build.log、green-selection-inventory-test.log）。
- 2026-11-11: ImageCls の出荷設定が参照する train/test/classes と images の 4 パスは手元に存在することを確認した。A2 自体は未実施。
- 2026-11-11 TDD: 通知 API 未実装による RED のあと、出力先所有のフィルタと制御記録を実装。テストの thread 固有の出力先設定は未初期化の wx ランタイムでアサートになったため、既存 FileLogger テストと同じ標準経路へ直した。6 テスト・28 assertion 成功（green-notifications-fixture-build.log、green-notifications-fixture-test.log）。
- 2026-11-11: GREEN 後、受け付けるキーの診断の集合は CreateReadScope でだけ作るように整理する。空の共通設定とそのコピーに暗黙の共有履歴を足さないため。次の未読一覧の公開口の宣言を固定し、挙動は次のテストから実装する。
- 2026-11-11: 初期の書き換えで config.hpp / config.cpp / nn_impl.cpp の UTF-8 BOM が失われていたことを HEAD との比較で検出し、復元した。ほかの初期の C++ 3 ファイルは BOM の有無が一致。全変更ファイルの文字コード・改行の検査は工程末にも行う。
- 2026-11-11: 読み取り範囲の整理後、config / log / nn の関連 288 テスト・7,207 assertion が成功（read-scope-refactor-build.log、read-scope-refactor-test.log）。BOM 復元後の再コンパイルは次のサイクルに含める。
- 2026-11-11 TDD: 一覧の未実装によるリンク失敗のあと、未読キーのソート済み一覧・0件の非通知・Run内の重複抑止を実装。7 テスト・39 assertion 成功（green-unread-report-build.log、green-unread-report-test.log）。
- 2026-11-11 TDD: include の欠落・空のパス・空の括弧・閉じない括弧の4入力で、例外が出ない RED を確認した（red-include-test.log）。実装後は設定テスト全体を検証中。
- 2026-11-11: include の実装後、設定の関連155テスト・4,821 assertion成功（green-include-build.log、green-include-test.log）。
- 2026-11-11 計画レビュー4回目: profileの生設定の保持と文字列dumpへの移行の追記もOK。実装完成の判定ではない。原文: .scratch/prd117/review-plan-r4.md。
- 2026-11-11 TDD: 未参照profileの不正な数値で構築が止まるREDを確認（red-unused-nn-test.log）。生設定のまま保持して参照時に検証する変更後、NNの140テスト・2,475 assertion成功（green-unused-nn-build.log、green-unused-nn-test.log）。古い不正profileのテストは必須branchの欠落による例外だけで通っていたため、実際に参照して診断のキーも確認する形へ直した。
- 2026-11-11: 用語変更後の既定キー検査16テスト成功、現用設定の監査エラー0件（d14-default-keys-test.log、d14-default-keys.log）。
- 2026-11-11 TDD: 入れ子カタログの抽出が別prefixの定義も拾うRED（3件、期待2件）を確認した。先頭のprefixとその項目の括弧で切り出す修正後、16テスト・133 assertion成功（green-nested-catalog-build.log、green-nested-catalog-test.log）。
- 2026-11-11 TDD: NNの走査後に6件が未読へ残るRED（期待1件）を確認。共有のblock/profileの在庫、実際のbranch/outputの読み取りを記録し、未参照profile専用のWARNを削除した。17テスト・143 assertion成功（green-nn-inventory-build.log、green-nn-inventory-test.log）。
- 2026-11-11: A2のbatを静的に再集計した。変数を通じて組み立てる宣言も含め19構成あり、既定7 EnvとAtari evalonlyを加えた実行確認は現時点で27構成。集計は .scratch/prd117/batch-chain-inventory.json。実行は未了。
- 2026-11-11 TDD: Actorの未知キーは4 Agentすべてで例外が出ないREDを確認した（red-actor-schema-test.log）。テストのImageCls名前空間の誤記は、その前に直した。構築時の型付き読み取りにdefault/override両方の検証範囲を追加中。OrderedMapにemplaceとoperator[]があると誤って仮定し二度コンパイルを失敗させたため、util.hppの実装を読み、既存のSetへ修正して再検証中（green-actor-schema-set-build.log）。このコンパイル失敗をREDの確認とは数えていない。
- 2026-11-11: Actorの検証範囲の実装後、28テスト・169 assertion成功（green-actor-schema-set-build.log、green-actor-schema-test.log）。GREEN後に自分の変更で未使用になったParseDoubleStrictを削除し、NNの141テスト・2,485 assertion成功（profile-parser-refactor-test.log）。
- 2026-11-11 TDD: RND無効時のキー不在と、learner停止時のlearn_count=0による4件のREDを確認（red-rnd-scalars-test.log）。生成側の診断宣言と休止時のNaNを実装し、RNDの関連テストを検証中。
- 2026-11-11: 100のjp/enを全文で対応確認し、変更を反映。見出し26、表の行43、コードブロック5の一致とリンク先の存在を検査し、translated-fromを確認済みの日本語blobへ更新した。他の英語文書は未了。
- 2026-11-11: RNDの最初の再検証では、新しいテストのepisode_start欠落と、無効時の値をoptionalの等値比較にしていた既存checkpointテストが失敗した。入力を直し、無効時は双方のNaNを確認、読込後のlearner休止も確認する形へ変更。37テスト・2,179 assertion成功（green-rnd-scalars-fixtures-build.log、green-rnd-scalars-fixtures-test.log）。
- 2026-11-11 TDD: RNDの過大batchの8件のREDを確認し、learner有効時の構築時エラーへ変更した。実行時のminによる縮小を除き、テストの共通入力にはlane数以下の2件を明示した。37テスト・2,195 assertion成功（green-rnd-batch-build.log、green-rnd-batch-test.log）。製品の既定32は変更していない。
- 2026-11-11 計画レビュー5回目: dormantを含むeval宣言とEnvの型付き設定を同じ出典で読み、Single側にも既存のValidateConfigの境界を設けてから閉じる判断もOK。Env・manifest・描画資源・乱数を検証で作らないことを実装時に確認する。原文: .scratch/prd117/review-plan-r5.md。
- 2026-11-11 TDD: 未知・空のrunner_typeでEnvを作ってしまう6件のREDを確認（red-runner-type-test.log）。設定読み取りとfactoryで共通検証を行う変更後、RunManagerの5テスト・43 assertion成功（green-runner-type-build.log、green-runner-type-test.log）。大文字小文字を無視するserial/pipelineの選択は保持。

- 2026-11-11 TDD: metrics 宣言の未知 event / step / target / 属性と、scalar / graph の key 欠落の 8 件で例外が出ない RED を確認した（red-metrics-input-fixture-test.log）。event の入力は既存の制御構文 event:unknown とし、従来 key として受け付ける生の @unknown の構文は変更しない。既存の WARN / ERROR 継続箇所を構築時エラーに変更して関連テストを検証中。

- 2026-11-11: metrics の関連31テスト・200 assertion成功（green-metrics-input-build.log、green-metrics-input-test.log）。
- 2026-11-11: 観測正規化の新しいテストではDefaultDQNの必須nn_forward参照が欠けていた。最初の12件の失敗のうちAgent側6件は目的外の診断なのでREDと数えず、既存のWithDefaultForwardConfigを通す入力へ直して確認し直す。factory側の6件は共通検証の接続後に成功。途中でテストファイルをビルド開始後に直したため、red-obs-mode-fixtureは古い入力のまま実行された。red-obs-mode-valid-fixtureで入力を反映して再検証中。
- 2026-11-11: 140・150・212の日英を全文で対応確認し、構造・コードブロック・リンクを検査して翻訳元blobを更新した。140は見出し24・表59行・ブロック7、150は29・58・3、212は30・157・3。150のPER無効時の説明は既知scalarをNaNとし、Tensorの未対応を同じ説明に含めないよう明記した。

- 2026-11-11: 正しい正規化入力で例外が出ない6件のREDを確認（red-obs-mode-valid-fixture-test.log）。Agentの設定読み取りとfactoryへ共通検証を接続し、関連44テスト・2,234 assertion成功（green-obs-mode-valid-fixture-build.log、green-obs-mode-valid-fixture-test.log）。
- 2026-11-11 TDD: Rainbowの不正なQR件数について、設定3条件と手動DTO2条件の5件のREDを確認。共通の件数検証を設定型とAgent構築へ接続して通常Qへの置き換えを削除し、関連21テスト・113 assertion成功（red-rainbow-qr-test.log、green-rainbow-qr-build.log、green-rainbow-qr-test.log）。
- 2026-11-11: 200の日英を全文で対応確認し、RNDの件数・NaN・INFO、休止時の局所検証、用語の表記を追随。見出し53・表77行・コードブロック14とリンクを検査し、翻訳元blobを更新した。NN実行設定の使わないキーとGNSの購読時だけの値検証の説明は、対応する実装を進めるときに改めて追随する。

- 2026-11-11 TDD: 画像providerの重複タグを拒否せず元のproviderを置き換える2件のREDを確認。生成・attachより前のエラーへ変更し、画像の関連8テスト・662 assertion成功（red-image-registry-test.log、green-image-registry-build.log、green-image-registry-test.log）。
- 2026-11-11 TDD: flatなmetrics定義の下位キー6条件と、入れ子のタグ・元キーの既読記録で8件のREDを確認。scalar / trace / graphの括弧を同じ読み口で解析し、値を実際にGetして記録する変更後、関連33テスト・212 assertion成功（red-metric-catalog-test.log、green-metric-catalog-build.log、green-metric-catalog-test.log）。生の未知control tokenの受理範囲は変更していない。
- 2026-11-11: 020の日英を全文で対応確認し、includeの停止、未読一覧・閉じたカタログの診断と構築順序、optimizer差分INFO、用語を追随。見出し48・表117行・ブロック20とリンクを検査して翻訳元blobを更新した。D14で不自然になった「件の代入」は「設定値の代入」へ直し、AGENTSの0xx文書の規約に合わせて既存のADRへの直接リンクを除いた。英語の用語置換は確認時に動詞leavesと重複したconfigurationを検出して修正した。

- 2026-11-12 TDD: 画像カタログの未知フィールド、使わない形式のフィールド、画像0件の共通既定、元キーの追跡で6件のREDを確認。完全キーと出典を保った設定を既存の型付きDTOへ渡し、全既知フィールドを読み、項目の未読キーを検証。関連12テスト・674 assertion成功（red-image-catalog-test.log、green-image-catalog-build.log、green-image-catalog-test.log）。共通既定のDTOは描画資源を作らない。Configの記録キーは既存どおり相対キーのまま。

- 2026-11-12 TDD: eval宣言・schedule・Envの型付き値と元キーの既読記録で13件のREDを確認。RunManager内に新しい読み取り範囲を作り、scheduleは全5欄を読んで閉じ、evalはEnvの型付き検証も通してから閉じる。Singleのfactoryへ既存Batchと同じValidateConfig境界を追加し、5 EnvはDTOだけ構築する。coreと5 EnvライブラリのDebugビルド成功、関連18テスト・138 assertion成功（red-eval-catalog-test.log、green-eval-catalog-build.log、green-eval-catalog-test.log）。Env生成数もdormant1・active2で確認した。
- 2026-11-12 計画レビュー6回目: 動画autoの固定と公開LogImageでの起動検証、Atari RAMカタログの追記もOK。実装の完成判定ではない。原文: .scratch/prd117/review-plan-r6.md。

- 2026-11-12: 名前付きNN実行設定の未知キーとNoisy OFFの既知値検証を実装し、NNの143テスト・2,491 assertion成功（green-forward-catalog-build.log、green-forward-catalog-test.log）。継続確認時には元のテストが実行中で、同じ出力先への再実行はファイル共有違反で開始できなかった。元のテストの完了と成功を確認した。
- 2026-11-12 TDD: NNのカタログ直下の通常キー2件まで在庫にするREDを確認。項目ごとのprefixへ絞ったあと、対応テストの11 assertionが成功（red-nn-inventory-boundary-test.log、catalog-boundaries-core-test.log）。同じ実行で評価トレース列の元キーが未読に残るREDを確認した。
- 2026-11-12 TDD: DatasetとAtari RAMの追加テストは、初回に必須shapeとreducer名が誤っていた。ビルド開始後の修正が実行に反映されなかったため、停止したImageClsテストだけを終了し、ログ捕捉も加えてビルドし直した。正しい入力でDatasetの未知キーを拒否しないREDと、RAM定義の2キーが未読に残るREDを確認（red-dataset-catalog-fixture-test.log、red-ram-catalog-fixture-test.log）。誤った入力の実行はREDと数えない。

- 2026-11-12: カタログの追跡と検証の修正後、coreの3テスト・1,041 assertion、Datasetの2テスト・5 assertion、Atari RAMの1テスト・3 assertion成功（green-catalog-core-test.log、green-dataset-catalog-test.log、green-ram-catalog-test.log）。途中のinit_test.cppのヘッダ不足によるコンパイル失敗は直してから再実行した。
- 2026-11-12 TDD: INFO化・重複抑止の8テストで20件のREDを確認（red-notification-sites-test.log）。checkpointは同じ読み込みを2回行い、値とmomentの検証を保ったまま通知の件数も検査する。
- 2026-11-12 TDD: 公開MetricsLogger::Logからffmpeg代役を起動し、別タグでNVENCを選び直すこととWARNが2件になる2件のREDを確認（video-fallback/test.log、codecs.txt）。代役をコンパイルするPythonのcmd呼び出しは、最初の引用符の誤りを直してから検証した。選択の共有と通知の移行後は検証中。
- 2026-11-12 TDD: factoryが所有prefixを宣言するAPIと、活動中のprefixを除く分類APIの未実装によるコンパイル失敗を確認（red-factory-inventory-build.log）。実装後のDebugビルドと、生成処理を呼ばずに分類するテストを実行中。

- 2026-11-12: 通知の移行後、関連8テスト・969 assertion成功（green-notification-sites-test.log）。動画の実起動経路も5 assertion成功（green-video-fallback-test.log）。起動traceは小画像CPU→NVENC失敗とCPU代替→後続タグCPU→次のRunでNVENC失敗とCPU代替の6行となり、WARNは各Run1件。factoryとprefix除外の3テストも成功した（factory-green-dqn-red-test.log）。
- 2026-11-12 TDD: 同じ実行で、DQNの未使用nn_forwardと休止prefix、射影の元キー、購読の無いGNSの局所値域について3テスト・5 assertionのREDを確認。型付き値を所有するGradNoiseConfigへ局所のValidateを置き、設定読み取りとLearner構築で共用する。batchを割り切る条件とAMPの条件は購読の準備境界へ残す。

- 2026-11-12 計画レビュー7回目: 出荷カタログ検査の追記を差し戻し。補間印を仮数値へ置き換えるA9の手順を明記し、nn_forward全件検査はNoisy OFFへ変更した。branch専用profileとOFFの省略形の受理も確認入力へ加えた。8回目はOK（review-plan-r7.md、review-plan-r8.md）。最終実装の判定ではない。
- 2026-11-12: Conv2dの資源を作らない設定DTOの関連2テスト・9 assertionと、File / Logs paneの同時出力・8スレッド・Runリセットの1テスト・12 assertion成功。アプリの未読一覧の欠落、未選択online/batchrunの負の値を拒否しないことも、実際のOnInit経路でREDを確認した。診断はRunのstdout.logへ転送されるため、red-app-startup-* / red-app-mode-online-* / red-app-mode-batchrun-*へ保存した。
- 2026-11-12: Atariのfactoryは型付きDTOだけでは7つの局所値域を検証していないREDを確認（red-atari-factory-values-early-test.log）。既存の純粋な検証をAtariEnvConfigへ移し、factoryとEnv構築の両方で共用する。ROMは読まない。DQNの射影の追加テストは、Headでは認識しないforce_fp32を誤って使っていた。正しいnoisy.sigma0へ直し、未知キーを既読にする実装は足していない。この入力による最初の失敗は追跡のREDとして扱わない。

## 最終検証の進捗（2026-11-12）

- Debug全体ビルド成功（`acceptance-build.log`）。既定coreスイートは947ケース中941成功・5失敗・1skipで、失敗は契約変更前のfixtureの5ケースだった。通知レベル・Run単位の通知回数・GNSの局所検証時機・手書きNNでの既定実行要求を修正した。関連72ケースの再実行は70成功、その残り2ケースも修正後に50 assertion成功（`fixture-update-test.log`、`fixture-update-final-test.log`）。最終の既定スイート再実行は残っている。
- 出荷設定は15ケース・2,632 assertion成功（`shipped-all-test.log`）。既定キー検査は16テスト成功、静的監査は4,026対象・誤り0（`default-keys-final-*`）。7 Envの全キー・文字列比較では、merginからmargin、使われないhistgram.intervalと古いheatmap設定の削除、MuZeroの保存無効化だけが変わり、残った既存キーの値の差は0（`resolved-comparison.json`）。
- GUIはLogs 11、Conv2d設定2、起動後の未読一覧1、非選択online/batchrunの局所値域各1の計16ケース成功（`gui-acceptance-final.log`）。stderrの非モーダル出力先も同じNotificationFilterで通知を受け、リセット用の記録を出力しない。
- A2のCartPoleは終了コード0、learn scalar 2,815件、WARN 0、通常保存と排水を確認（`acceptance/runs/02-CartPole/acceptance.json`）。残る26構成は実行中で、未了の構成を成功とは扱わない。既定Atariの初回の試行は出力名がRunプロファイルに上書きされた。保存と排水は観測したが、controllerを止めたため終了コードを採取できず、合格には数えない。出力先と終了予算をCLIの最優先指定へ移して再実行する。学習・評価の設定は変えない。
- NMRND/SERNDの受入用コピーは設定解決成功。変換と元/先のSHA-256は`acceptance/legacy-rnd-current/migration.json`。プロファイル境界の初回の変換で、既にコロンを持つ左辺の@varsまで変えた誤りは、既存の境界を保持する変換へ直した。元のfrozen設定とbat・旧exeは変更していない。
- `config_inventory_test.cpp`はAgentとEnvの両repositoryを通す在庫分類の公開境界の検証として分ける。資源生成を呼ばないことと、活動中の子prefixが非活動の親prefixに消されないことを検証するため、個別Agent/Envのテストには混ぜない。
- 最終既定coreスイートは成功（947ケース中946成功・1skip、176,324 assertion全成功、`core-default-final-test.log`）。追加でA2が発見したImageClsの修正は、その後に公開Agentの学習率テストへWARN無しのassertを加え、1件のREDを確認してから実装した（`red-imagecls-input-test.log`）。NN構築へ渡す仕様のコピーから教師ラベルvectorだけを外し、ImageCls・NN・出荷設定の関連テストを再実行中。元のラベルと学習率の数値検証は保った。
- A2はAtari・CartPole・DropMerge・MuZero・LunarLanderがWARN 0・正常終了で合格。GridMazeは未読2キー、ImageClsは教師ラベルのNN入力仕様と未選択main_const phaseで不合格を記録し、修正後の再実行を残す。NMRND/SERNDのコピー元8ファイルは、保存済みSHA-256との一致を再確認した。設定整理後の既定キー16テスト成功、静的監査4,022対象・誤り0、所有Python13ファイルは構文検査成功。
- GridMazeの合意済み整理後の再Runも、learn scalar 2,854件・WARN 0・終了コード0・保存/排水で合格（`gridmaze-cleanup-native.log`）。7 Envの再採取21 assertion成功。追加の全キー比較では旧body構造・未使用tau・ImageCls未選択phaseが削除され、残るキーの文字列値の変化は0（`resolved-comparison-a2-cleanup.json`）。RT30BとSE30の通常Runも合格し、現時点は8/27成功、ImageCls1件が修正後再Run待ち、残る18件は実行中/未了。
- 独立実装レビュー1回目: 計画OK、実装は固定キーの読み取りとPython冒頭説明の2件で差し戻し（`review-impl-r1.md`、原文保存）。blockのtype、参照時のprofile、Noisyの実行値をcpp内の局所DTOとANET_READ_CONFIGへ揃え、fresh scopeと必須endを保持した。4つの検証資材のdocstringも日本語へ修正。NN・profile・出荷設定の関連検証を再実行中。レビュー途中に指摘された受入のlearn step代替は、公開Run成果物の形式を使う3テスト（正例2条件のREDを含む）で修正を確認し、正式な不足からは除かれた。exp stepやcheckpointの値を学習到達には使わない。
- レビュー修正後はDebug coreビルドと、NN・profile・出荷設定の158ケース・5,125 assertion成功（`review-fixes-core-build.log`、`review-fixes-core-test.log`）。受入判定の3テストも成功（`review-fixes-acceptance-test.log`）。NMRNDの通常Runはlearn scalar 1,489件・WARN 0・終了コード0・保存/排水まで確認し、合格は9/27となった。
- 独立実装レビュー2回目は計画・実装ともOK（`review-impl-r2.md`、原文保存）。再確認で追加のImageCls設定のCRLFも検出し、BOMと設定値を保ってLFへ修正してから確認を受けた。全77項目に未解消の指摘は無い。A2の残りとImageClsの通常Runは未了で、このOKだけで工程全体の完了とはしない。

## 受入項目との対応（最終結果）

受入補完の進捗: 153ケースのうち151成功、失敗した2ケースは追加fixtureのbranch不足とprofile診断の期待文字列だった。入力を修正し、A3の両演算子/参照有無/type上書き、branch typo、PER代替、history/redirect、Observerなどを含む8ケース66 assertionが成功した（`acceptance-repetition-final-test.log`）。ImageCls cacheは2ケース16 assertion成功（`acceptance-image-cache-test.log`）。Atari表示は検証用SDL DLLを既存ソースから`.scratch`へビルドし、dummy driverで実際のALE生成を2回通してWARN 1件を確認（`atari-display/result.xml`、`result.json`）。PATHだけの初回失敗は受入成功に含めず保存した。最終的なdriver前提検査、Actorごとのskip設定通知のassertion、追加GUIは再ビルド後に確認する。対応表の入力と未再現箇所はtestdata READMEに保存した。

2026-11-12のユーザー判断: A1の起動時に一度だけ通る処理とメモリ確保失敗は、公開経路で再現可能な確認、通知部品の反復テスト、呼び出し箇所の照合で受け入れる。実際に再現していない分岐はその旨を記録する。検証用の失敗注入機構や実メモリ枯渇は追加しない。

2026-11-12の受入照合で、関連テストの存在だけでは指定入力・反復・通知件数を満たさない箇所を検出した。`review-acceptance-r1.md`を原文として保存した。A3のファイル両演算子・未参照block・type上書き・使用branch profile、A4の元キーを含む実WARN・実block構築後の上書き元除外、A5の両休止条件のINFO 1/WARN 0、A6のcheckpoint/1 lane/既知13キー/Observer、A10の欠落を2回ロードしたWARN件数を追加検証する。下表の既存成功は従来の範囲であり、これらを完了したとする記録ではない。

A1は共通通知部品と一部の実経路の反復が成功済み。未確認の各行は同レビューの一覧を正とする。特に起動時workspace代替、保存0 byte、GUI centre修復の再発、optimistic/TBO/learner停止、Actorの未使用skip設定、STOP、PER残り3条件/spatial/抽選代替、RND無視設定、InitRL、history/redirect、ResBlock、画像Observer/probe、採用episode不足、Env dump非対応、Atari表示、ImageDataset auto cache、Outlier RowのWARN 0を個別に照合・補完する。起動時だけの経路や資源確保失敗の再現を含め、未確認を共通部品の成功で置き換えない。

| 項目 | 入力・回数・観測と検証の所在 | 現時点 |
| --- | --- | --- |
| A1 通知 | 同一箇所/対象の反復、別対象、複数スレッドとRunリセットはlog_testとFile/Logs paneの統合テスト。INFOへの変更はDQNのSTOP・休止・optimizer、RND optimizer、InitRL、dormantの実経路で確認（`green-notification-sites-test.log`、GUI16ケース）。動画autoは公開Logからffmpeg代役で2 Runを実行し、codec6行と各Run WARN 1件。削除5文字列は本番ソースで0件（`a1-removed-notifications.log`、rg終了値1は一致無し）。§5.5の残す通知各行と反復テストの対応・未再現の分岐をtestdata READMEへ個別に記録した。公開経路と共通部品の反復・呼び出し照合による受入はユーザー合意済み | 成功。GUI19ケースと追加core反復試験を含む |
| A2 通常Run | matrixの27構成。学習scalarの実レコードまたは未宣言時のRunner learn_step、evalonlyはsession end。正常保存/排水と終了後ログのWARN件数。各試行のcommand/input/config/metrics/log/exitを保存する | 全27構成成功。`acceptance/summary.json`とtestdata READMEの個別表 |
| A3 閉じた項目 | blockの`?=`/`=`×typeだけのRun上書き有無×参照/未参照の8入力、使用するbranch profileの既読typoを公開読み口で検証。4 AgentのActor、NN実行設定、eval/dormant/schedule、scalar/trace/imageも含む。未参照profile/forwardとNoisy OFFの有効な省略は受理する | 補完8ケース66 assertionと最新の既定core953ケースで成功 |
| A4 未読一覧 | 両ファイル代入演算子・CLIの未知キーを含むWARNの全文と件数、0件/複数件、実block構築後の上書き元除外、在庫/休止/未宣言prefix、branch未使用profileの実WARN、構築を拒否しないことを検証 | 最新の既定core953ケースで成功。GUI起動の従来ケースも成功 |
| A5 dormant | interval 0とschedule無しの両方で同tagのscalar参照2件。skip INFOはtagごと1件、WARN 0 | `acceptance-exact-inputs-test.log`と最新の既定coreで成功 |
| A6 RND | RND checkpoint有り・learner停止・1 lane・batch数2で2回構築。既知13 scalarのNaN/未知nullopt、公開Observer経由のWARN 0・休止INFO 1。無効時、batch数超過の構築エラー、lane数と同数の学習件数も確認 | `acceptance-exact-inputs-test.log`と最新の既定coreで成功 |
| A7 構築時の失敗 | missing/malformed include、metrics event/axis/target/属性、正規化方式、Rainbow QR件数、画像tag重複、未知runner_type。キー/値/期待値とserial/pipelineの大小文字受理を確認 | config/metrics/image/trainer/DQNの関連テストと既定core成功 |
| A8 既存検証 | MSVC経由のDebugビルド、最新の既定core 953ケース中952成功/1skip・176770 assertion（`core-default-acceptance-final-test.log`）。skipはCUDA不可時の診断試験がCUDA利用可能のため除外されたもの。その後のassertion/include/定数整理は、出荷設定を含む19ケース4114 assertion成功（`acceptance-final-assertions-test.log`）。Atari表示1ケース7 assertion、ImageCls cache2ケース16 assertionも成功。GUI Logsと追加3ケースを含む全19ケース成功（`gui-acceptance-final-expanded-r2.log`）。全体DebugビルドとGUIハーネス修正後の再リンクも成功 | 成功 |
| A9 出荷カタログ | 7 Env、4カタログの全項目。block仮数値・module factory、profileは参照用Linear、forwardはNoisy OFF。未知out_channels/profile/forwardのfixtureで検査自体が拒否し、branch専用profileとOFFの省略形を受理。本数/名前/値は固定しない | 最新158ケースに含む |
| A10 checkpoint通知 | DQNとRNDの設定差はINFO・WARN 0。skip network/primary_skip momentsの欠けは同じcheckpointを2回ロードして文言ごとWARNレベル1件。moment/設定値も検証する | 補完の当該ケースと最新の既定coreで成功 |
| A11 用語 | 指定した追跡範囲へのgit grep。5複合語と単独の葉の残りは、AGENTSの避ける語表・CONTEXTのAvoid・wiki課題区画・言葉だけ（`a11-named-terms.log`、`a11-leaf-terms.log`） | 除外後0件 |

## 自己点検（2026-11-12）

実装品質確認観点の全77項目を、変更箇所と適用範囲に沿って走査した。機械検査は所有133ファイルを対象とし、BOM変更・ヘッダのusing namespace・productionのdynamic_cast・避ける語の追加は0件。CRLFを持った8ファイルをLFへ修正し、再検査で問題0件（`self-check-machine-final.log`）。重複リテラルの101候補は設定キー・既存のJSONキー・テストの設定と期待値が中心で、補間typeの重複、検証の環境変数名、includeと新規テストの内部リンケージも見直した。

- 1〜7・9〜13・15〜21・23〜47・50〜52・54〜57・59〜66・69・71〜76: コード/テスト/設定の読み口、元キーと所有権、既存型への局所検証、変更理由と検証記録、日英の内容とリンク、改名と歴史資料の保持を照合した。型を持たないカタログ走査は参照時に検証し、Runの乱数は消費しない。頻繁な通常stepに新しい同期転送を加えていない。通知の重複除去のmutexは通知を出すときだけで、動画autoの選択の固定はatomic。UIが学習更新の長いロックを待つ経路は加えていない。
- 8・22・31・53・58・67・68・70: 今回はskillの変更、互換層、新規ヘッダ、dynamic_castを避ける取り出し口、順序番号、新しいmetrics連鎖や機能設定キーを足していないため対象外。
- 14・48・49・77: A2全27構成、既定core953ケース、GUI19ケース、受入で不足した指定入力・通知反復の補完が成功した。GridMazeは回答に従って未使用2指定を削除し、実効tau0.001を維持した。学習scalarの実レコード、評価セッション終了、通常停止・保存・排水を個別成果物で確認した。
- 通知レベルはAGENTSの表へ対応させた。休止・checkpointの値の置換・モードの説明はINFO。未読一覧・ほかの設定の効き目が変わる構成・使用しない入力の診断はWARN。動画autoの代替と既存のPER抽選の代替は各該当行、checkpointの欠けの初期化は該当行、GUI/ファイル/画像/指標の部分的な異常は補助機能の行、clipboardの操作失敗だけは操作ごとの行に従う。全27構成の通常RunではこれらのWARNが出ないことを確認した。

## 最終受入の記録（2026-11-12）

- 全27構成が成功し、終了コード0・WARN 0・通常停止と排水を確認した（`acceptance/summary.json`、`acceptance-summary-final.log`）。学習26構成はlearn scalarの実レコード、評価専用Runは4セッションの終了を確認した。各構成の停止予算・最終train exp記録・learn scalar件数はtestdata READMEへ保存した。保存有効の構成は保存完了、MuZeroと評価専用Runは指定どおり保存無効を確認した。
- 評価専用Runの初回は停止予算1でセッションまで届かず不合格とし、失敗記録を保持した。最初の有効な評価間隔10とlane数1から停止予算11を求めるよう受入ツールを直し、学習・評価条件を変えず再実行して成功した。受入判定の3テストも再実行して成功した。
- 全体Debugビルド成功（`debug-final-build.log`）。初期通常Runのバイナリは`acceptance/debug-executable.json`、ImageClsと評価専用の最終再実行は`acceptance/debug-executable-final.json`へSHA-256を記録した。後段の固定キー読み取りDTO化は有効な構成の値を変えず、既定coreと全出荷カタログで再確認した。診断対象を絞ったImageClsは最終Runnerで再実行した。解決メタデータは`acceptance/prepared/06-ImageCls-cleanup.json`へ更新済み。
- NMRND/SERNDの受入用コピーだけを移行した。旧frozen設定・bat・exeを含む原本8ファイルは、保存済みSHA-256と一致することを再確認した。GridMazeは合意どおり実効tau0.001を保った。
- GUIの最初の追加受入で、同一プロセス内のRunner再起動と新タグの起動設定漏れによりEnvを二重登録し、2ケースが失敗した。製品側の重複検証は変えず、テスト起動を分離して新タグを起動対象へ登録した。GUI再ビルド成功（`gui-acceptance-harness-fix-build.log`）、再実行は全19ケース成功（`gui-acceptance-final-expanded-r2.log`）。最初の失敗XMLも保存した。
- 既定coreは953ケース中952成功・想定した1skip、176,770 assertion成功。その後のテスト内の件数assertion・include・定数整理は出荷カタログを含む19ケース4,114 assertionで確認した。Atari表示はdummy driverを要求する最終テストバイナリで1ケース7 assertion、ImageCls cacheは2ケース16 assertion成功。
- 所有対象は141ファイル。並行変更のPRD103 READMEは保持して対象から除外した。CONTEXT.mdのnn_traceの新設とcapture/traceの関連説明の置換も並行変更で、PRD117の所有差分には含めない。全77項目の自己点検、Python15ファイルの構文、BOM/LF、禁止表記、既定キー、日英の対応と翻訳元情報を確認した。最後の機械検査結果はレビュー依頼に添付する。

## 決まらなかったこと

ユーザー判断が必要な未決は無し。A1の一度だけ通る経路とメモリ確保失敗は、合意した方法で受け入れ、実行で再現していない分岐をtestdata READMEへ明記した。GridMazeの実効値と旧RNDファイルの保持も回答どおり扱った。最終レビューと課題6件のAI判定の記録まで完了した。記録する課題の候補はありません。

## 完了報告（2026-11-12）

実施内容: PRD117のD1〜D14を実装し、A1〜A11の検証を終えた。Debug全体ビルド、既定core953ケース中952成功・想定した1skip／176,770 assertion、追加19ケース／4,114 assertion、GUI19ケース、通常Run27構成が成功した。通常Runは全構成でWARN 0、終了コード0、通常停止と排水を確認した。

変更ファイル: coreの設定・NN・通知・Agent・メトリクス、Envの設定検証、Runnerと出荷設定、日英設計文書7組、用語・規約、検証資材、課題6件。レビュー時の所有一覧は`.scratch/prd117/diff-20261112-045342/manifest.json`の141ファイル。CONTEXT.mdのnn_trace関係は並行作業の部分所有として除外した。課題判定から生成した一覧とwiki地図8ファイルは、生成直前と直後を`issue-maps-final-r2/`へ保存した。

レビュー結果: 計画8回、実装・受入照合4回を経て、最終は計画・実装ともOK（不足・範囲外・開発者確認は無し）。原文は`.scratch/prd117/review-impl-r3.md`。再確認では指定入力・反復・対象別通知件数、A1の合意した未再現分岐、全27Runの個別結果と初期／最終バイナリの区別、GUI追加19ケース、141ファイルのLF・BOM・検索結果を照合した。根拠が無く見送った指摘は無い。全77項目の適用・対象外の内訳はレビュー原文に記録した。自己点検は141ファイル・問題0件（`self-check-final-owned.log`）、Python15ファイルの構文も成功した。

決まらなかったこと: なし。記録する課題の候補はありません。

判断してほしいこと: なし。

次にできること: 実装工程はここで終了する。課題6件のAI判定を残件無し・2026-11-12に更新し、statusは対応中のまま保持した。判定日の空欄を扱う記録補助を修正後、6件を照合し、`reports/tools/wiki_support.py maps`の最終実行が終了コード0で成功した。stage・commit・pushは行っていない。

## 追加の両モデルレビュー（2026-11-12）

- ユーザーが完了後の`cc-review`と、指摘の修正・再レビューを明示指示した。同じ依頼をCodexの継続担当とClaude Opusへ送り、原文を`review-20261112-050554-codex.md`／`review-20261112-050554-claude.md`へ保存した。基準HEADは`010a037b2b9f04de161f75c51daff15ecc58afca`、対象は従来141ファイルと課題判定の派生8ファイル。別作業によるステージ済み変更と並行コミットを保持した。
- 1回目はCodexが計画・実装ともOK、Claudeが計画OK・実装2件差し戻し。Fail-Fastの表で消えた行名を指す品質観点23の参照文と、動画codec名の直書き増加が対象だった。ユーザーの追加承認は求めず、指示の範囲で2件を修正した。ClaudeはRead/Glob/Grepに限定され、スクリプト検査は実行できなかったため、Codexと自己点検の結果を併記する。Claude側費用は約2.64 USD。
- 規約の参照は、現行表で`ANET_SYSTEM_ERROR`による停止を定めた状態を指す文へ合わせた。`docs/quality/prd.md`と日英設計文書も検索し、消えた行名へ依存する別の参照は無かった。表のERROR条件は変えていない。
- 動画codecはファイル内の定数`kSoftwareH264Codec`と`kNvencH264Codec`へ統一し、判定・選択・起動・ffmpeg引数・診断文で同じ値を使う。定数はファイルの範囲で内部リンケージを持ち、公開ヘッダや機能は増やしていない。codec名・引数・通知文言・fallbackの条件は保った。
- 修正後の全体Debugビルド成功（`cc-r1-fixes-build.log`）。既存の動画関連9ケース45 assertionと、公開MetricsLoggerからffmpeg代役を起動する1ケース5 assertionが成功（`cc-r1-video-unit-test.log`、`cc-r1-video-fallback-test.log`）。codecの6行も`libx264,h264_nvenc,libx264,libx264,h264_nvenc,libx264`で一致した。通常Runの学習・評価条件を変える修正ではなく、A2の全27Runは前のバイナリによる受入結果として保持する。
- 全77項目を変更3ファイルの適用範囲と前回結果に照合した。LF・既存BOM・C++禁止表記・旧参照は問題0。品質観点72と同じ差分・パターンによる文字列検査も候補0件、codec名のリテラルは各1件（`cc-r1-self-check.log`、同json）。修正2件をまとめて同じ両モデルの継続レビューへ出す。
- 2回目はCodex・Claudeとも計画と実装がOK（不足・範囲外・開発者確認は無し）。原文は`review-20261112-051952-codex-r2.md`／`review-20261112-051952-claude-r2.md`。2件の解消、現行表への参照、codec定数の使用、同じ引数と通知文言、Debugと関連10ケース50 assertion、LF/BOM・文字列検査の結果を再確認した。根拠が無く見送った指摘は無い。Claudeはスクリプトを実行できず、その検査はCodexと自己点検で確認した。Claude側の費用表示は初回2.6401291 USD、再レビュー3.9793763 USD。
- 追加のcc-reviewは2回で完了した。最終修正後にcore既定全件・GUI全件・A2全27構成を再実行したとは扱わず、codec識別子の定数化後に実行したDebug全体ビルドと動画関連検証を区別して記録した。課題6件の残件無し判定は維持する。未決・未解消の指摘・新たな課題候補は無し。PRD117の実装と追加の両モデルレビューを完了し、stage・commit・pushは行わず停止する。

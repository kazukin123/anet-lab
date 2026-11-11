# 未読設定と通知の検証

設定の解決・読み取り・カタログの閉じた検証と、Run単位の通知を確認する。設定の内容の本数・名前・値を固定するgolden比較は使わない。

## 自動テスト

- `[prd117]`: 元キーの追跡、コピー・上書き・在庫、入れ子のカタログ、未読一覧、NNとActorの読み口、通知のスレッド共有とRun開始のリセット。
- `[config_profile]`: 参照時の検証、補間値、branchごとの上書き、文字列の設定dump。
- `[shipped_config]`: 出荷設定の読み口による検証。未参照のblock、全体とbranch内のprofile、NN実行設定も対象にする。
- GUIの`[log_panel]`: ファイルとLogs paneでの件数とリセット用の非表示記録。

## 動画autoの代替

リポジトリルートのPowerShellから実行する。先にDebugのanet-core-testをビルドする。

```bash
.venv/Scripts/python.exe core/anet-core/testdata/prd117/run_video_fallback_test.py
```

`fake_ffmpeg.cpp`をMSVCで`.scratch/prd117/video-fallback/ffmpeg.exe`へビルドし、テストの子プロセスだけのPATHへ追加する。親のPATHは変えない。代役は要求されたcodecを`codecs.txt`へ記録し、NVENCだけ起動失敗させ、ほかはstdinを読む。製品の公開`MetricsLogger::Log`から、小画像、別タグの大画像2件、次のRunの大画像を渡す。

2026-11-12の実行で5 assertion成功。codecの順は`libx264, h264_nvenc, libx264, libx264, h264_nvenc, libx264`となり、WARNはRunごとに1件。小画像でのCPU選択だけでは固定せず、NVENC起動失敗後だけ固定し、次のRunでは再び選択することを確認した。

## 通常のRunで確認すること

RunManagerとアプリの画像登録・panel初期化を終えたあと、最初のstepより前に未読一覧を確定する。学習するRunは、解決済み定義で`@learn`を使うscalarを1件以上記録するまで実行する。該当scalarが無いRunはlearn stepが1以上になるまで実行する。評価専用Runは最初の評価sessionを完了させる。アプリの終了予算で到達させるため、最初の記録より後まで進む場合もある。強制終了せず、通常の停止・保存・Observerの排水・ログの終了まで確認する。

各Runについて、起動引数、実効設定、到達した条件、終了コード、Runのログ、WARNの件数を保存する。WARNの集計はログの`: [W] `を対象とし、終了処理の出力も含める。期待値は全構成で0件。例外や到達条件の未達を、WARNが0件であることだけで成功にしない。

Atariの評価専用Runには、同じ検証のAtari既定Runが通常終了時に保存したcheckpointを渡す。外部の古いRunのcheckpointで代用しない。ImageClsのdatasetファイルやAtariのROMが無ければ未達として記録する。

## 実行対象

既定の7 EnvはAtari、CartPole、DropMerge、GridMaze、GridMaze_muzero、ImageCls、LunarLander。それぞれ共通の`_main.txt`にEnvのファイルを重ね、既定の`run.$`を使う。

番号付きbatは`goto`の飛び先によらず全宣言を読み、SETの変数を展開した`run.$`の文字列で重複をまとめる。2026-11-11の集計は次の19構成で、既定7構成とAtariの評価専用1構成を合わせて27構成。実行時には現行ファイルから再集計する。

| 番号 | Env | 元の宣言 |
| --- | --- | --- |
| 1 | Atari | `11_batch_run.bat`のRT30B |
| 2 | Atari | `11_batch_run.bat`のSE30 |
| 3 | Atari | `11_batch_run.bat`のNMRND（RT30Bを含む） |
| 4 | Atari | `11_batch_run.bat`のSERND（SE30を含む） |
| 5 | Atari | `11_batch_run.bat`と`12_batch_run.bat`のTGCH |
| 6 | DropMerge | `12_batch_run.bat`のBASE |
| 7 | DropMerge | `12_batch_run.bat`のSKBASE |
| 8 | LunarLander | `12_batch_run.bat`のLLRND |
| 9 | Atari | `13_batch_run.bat`のRT |
| 10 | Atari | `13_batch_run.bat`のRT50 |
| 11 | Atari | `13_batch_run.bat`のSE |
| 12 | Atari | `13_batch_run.bat`のRT50NZ |
| 13 | Atari | `13_batch_run.bat`のRT30 |
| 14 | Atari | `13_batch_run.bat`のRT10 |
| 15 | Atari | `13_batch_run.bat`のRT10RND |
| 16 | Atari | `13_batch_run.bat`のRT10V |
| 17 | Atari | `13_batch_run.bat`のSENZ |
| 18 | Atari | `13_batch_run.bat`のSENZPA |
| 19 | Atari | `13_batch_run.bat`のSENZPASK |

## 実施状況

2026-11-12時点で、既定coreスイートは953ケース中952成功・1skip、176,770 assertion成功。skipはCUDA不可時の診断試験で、この環境ではCUDAを使えるための想定した除外である。その後のActorごとの件数assertionとinclude/定数の整理を再ビルドし、出荷設定を含む19ケース・4,114 assertionも成功した（出荷設定単独は15ケース・2,632 assertion）。ImageCls cacheは2ケース・16 assertion、Atari表示は1ケース・7 assertion成功。最終Debugビルドは成功し、GUIは保存0 byteとcentre異常の再発を含む19ケースすべて成功した。通常Runも全27構成で到達条件・WARN 0・終了コード0・通常停止と排水を確認した。経過と個別の検証ログは[実装メモ](../../../../docs/memo/117_warn_policy_and_unread_config_20impl.md)に記録する。

## A1の行ごとの確認

通知部品は`log_test.cpp`の`Run notifications occur once per source and target across threads`と、GUIの`[log_panel]`で、同対象・別対象・並行通知・Runリセットを確認する。下表の反復では途中にリセットを入れない。

下表は入力とassertionの対応であり、実施結果は「実施状況」と実装メモに記録する。追加GUI（保存0 byte/centre再発）も実行し、全19ケースが成功した。Atari表示は隔離したexeとSDL DLLによる実行が成功した。

| 通知・扱い | 実経路の入力と観測 |
| --- | --- |
| クリップボード操作の失敗 | LogPanelのcopy操作の失敗で通常のWARNを出す箇所を照合する。操作ごとの通知を保ち、Run単位の抑止を入れていない |
| 旧runs_dir/root_dir専用の通知削除 | GUIの`[prd117_startup]`で旧root_dirが未読一覧に載り、専用通知は無いことを確認。削除対象5文字列はproductionのcpp/hppで0件 |
| workspace代替 | `app_util_test.cpp`の`Unavailable last workspace can fall back...`で、隔離した履歴の先頭が失われた場合の公開Resolve失敗と`_default`構築を確認。RunnerApp自身の起動WARNは未再現で、OnInitの履歴先頭1件の分岐と`warn_once`呼び出しを照合する |
| 保存0 byte | GUIの`[prd117_save_zero]`で同じpathへSaveAgentを2回、戻り値0・WARN 1・実ファイル0 byte |
| centre無し・二重centre | GUIの`[prd110_fallback]`と`[prd110_redundant_show]`で修復後に同じ異常配置を再発、各WARN 1 |
| optimistic target・TBO・checkpoint無しのlearner停止 | `dqn_based_agent_test.cpp`の各`...warns...`で同じ設定を2回構築、WARN 1 |
| 未使用skip.max_length・STOP説明 | 同`Skip target windows validate...`で同じ設定を2回構築、対象ActorごとWARN 1、STOPはINFO 1 |
| shared時のskip.nn_forward_key | 同`Shared skip warns once...`で同じ設定を2回構築、各ActorのWARN 1 |
| RND batch超過 | `intrinsic_test.cpp`の構築時エラーと等lane数の件数確認 |
| RND休止 | 同`DQN checkpoint requires matching intrinsic presence...`でcheckpointを1 lane/learner停止で2回構築、INFO 1・全13キーNaN・未知nullopt・公開ObserverのWARN 0 |
| skip Network・primary_skip状態の欠落 | `dqn_based_agent_test.cpp`の`Skip checkpoint migration...`で同じcheckpointを2回ロード、欠落条件ごとWARN 1 |
| PERの4構成 | 同`DQN replay priority validation warns once...`と`DQN replay priority warnings identify...`で同じ検証を2回、各WARN 1 |
| spatialのlane不足 | 同`Spatial exploration reports...`で2種の方策を同じ2 laneで構築、共通通知箇所のWARN 1 |
| GNS・policy churn | 同`Gradient noise validates...`と`DQN policy churn warns once...`で同条件の購読を反復、WARN 1 |
| DQN optimizer設定差 | 同`DefaultDQNAgent creates the initial snapshot from an auto-loaded network`でauto_load後に再Load、INFO・WARN 0。`Learner checkpoint applies current optimizer options...`でも設定値とmomentの保持を確認 |
| RND無視設定 | `intrinsic_test.cpp`の`RND warns once per ignored resolved leaf`でfactory Createを2回、各対象設定のWARN 1 |
| RND optimizer設定差 | 同`RND checkpoint applies current optimizer options`でロードを反復、INFO・WARN 0 |
| CUDA_LAUNCH_BLOCKING置換 | `init_test.cpp`で1回目の後に環境変数の不一致を戻しInitRLを2回、INFO 1・WARN 0 |
| history読取失敗・不正entry | `app_util_test.cpp`で同じpath/entryを2回読む、各対象WARN 1 |
| stdout/stderr redirect失敗 | 同`Standard stream redirection failures...`で同名directoryを置き2回起動、各pathのWARN 1 |
| 重ね設定ファイル不在・include不在/構文不正 | `config_test.cpp`でProperties/ConfigManagerの構築エラー |
| ImageProvider tag重複 | `image_test.cpp`で公開登録時のエラー |
| 動画auto | 上記ffmpeg代役による2 Run、失敗後の選択固定・各Run WARN 1 |
| 未参照profileの個別通知削除 | `nn_test.cpp`の未参照共有定義・branch専用定義。前者はWARN 0、後者は元キーを含む未読一覧WARN 1 |
| 未使用NN入力 | 同`NetworkBodyBuilder warns once only for directly unused inputs`で同条件を反復、対象入力ごとWARN 1 |
| ResBlock BatchNormとdropout | 同`ResBlock exposes dropout config...`で同blockを2回構築、WARN 1 |
| HeatMap・Conv2d・GraphVizの異常入力 | `observers_test.cpp`の`Image observers suppress...`で同じサイズ不一致2回/未終了episode開始3回、各Observer WARN 1 |
| 値のないscalar | 同`Unknown non-DQN action scalar...`で同じ未知キーを反復、tagごとWARN 1。既知未成立値はWARN 0 |
| metrics未知識別子 | `observers_test.cpp`でevent/axis/target/属性の公開定義読み口のエラー |
| VectorProbe値なし | `probe_test.cpp`で同probeへ存在しないaction値を2回問い合わせ、WARN 1 |
| PER抽選代替 | `replay_buffer_test.cpp`の`[prd117_rejection]`で有効な明示priority 0を使い2回上限へ到達、WARN 1・一様抽選結果 |
| Rainbow QR不正・正規化方式不正 | Agent/normalizerの設定構築時エラー |
| Outlier Row | `scaler_test.cpp`の`Robust normalization admits...`で外れ値行を反復、WARN 0。DEBUG出力の有無はassertしない |
| dormant | `trainer_test.cpp`のinterval 0/schedule無しで同tagの参照2件、INFO 1・WARN 0 |
| 採用episode不足 | 同`RunManager decorates configured Eval...`の公開構築で各tag WARN 1。構築時に一意tagを一巡するため、同Run同tagの再構築は行わず呼び出し箇所を照合する |
| Env config dump非対応 | 同`RunManager does not warn when shared Eval adopts its only group`で公開構築の各Env名 WARN 1。公開CreateEvalRunnerも生成時に通るが、成功済みの同名Envは再生成を拒否する。dump後の生成失敗を挟む同名再試行は未再現 |
| Atari display_screen | `AtariEnv_test.cpp`のhidden `[prd117_display]`をSDLのdummy driverで実行、同対象を2回構築しWARN 1 |
| ImageDataset auto cache | `ImageClsEnv_test.cpp`のcache試験でcap超過とReset反復、WARN 1。確保失敗のstd::exception/その他の例外は未再現。両catchでdataset keyを対象に`warn_once`を呼び、noneへ選択固定する箇所を照合する |

2026-11-12のユーザー判断により、起動時に一度だけ通る処理とメモリ確保失敗は、公開経路で再現可能な確認、共通通知部品の反復テスト、呼び出し箇所照合を合わせて受け入れる。採用episode不足も構築時の一意tagごとの処理であり、Env dumpも成功した同名Envの再生成を許さないため、この確認方法を適用する。workspace・採用episode不足・Env dumpの同対象反復、およびImageDatasetの実メモリ確保失敗を、実際に再現した結果としては扱わない。

## 受入補助ツールの使い方

### Atariの表示通知

先にDebugの`AtariEnv-test`をビルドし、`ATARI_ROM_DIR`で`pong.bin`を用意する。SDL2 DLLを`.scratch/prd117/sdl2/SDL2.dll`へ用意して、リポジトリルートのPowerShellから実行する。別のDLLを使う場合は`--sdl-dll`でパスを指定する。

```bash
.venv/Scripts/python.exe core/anet-core/testdata/prd117/run_atari_display_test.py
```

wrapperは検証用exeとDLLだけを`.scratch/prd117/atari-display`へコピーし、その子プロセスだけで`SDL_VIDEODRIVER=dummy`を指定する。実際の窓は開かない。ALEの`LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`はPATHを探索しないため、SDL DLLをexeと同じdirectoryに置く。libtorch/wxWidgetsは元のtestのdirectoryから読み込む。JUnit・console・終了値・exe/DLLのSHA-256を同directoryへ保存する。

2026-11-12は既存の`C:/dev/vcpkg/buildtrees/sdl2/src/se-2.30.11-e9693b8e70.clean`から共有SDLを`.scratch/prd117/sdl2`へビルドした。依存ソースを変更せず、取得や製品への同梱はしていない。ビルド記録は`.scratch/prd117/acceptance-sdl-build.log`。最初のPATHだけの試行はDLLを読めず失敗し、exeとの同居後に成功した。dummy driverの前提検査を含む最終再ビルド後も1ケース・7 assertion成功した。

### 通常Run

`run_acceptance.py`は現行のbat宣言から構成を収集し、実際のConfigManagerで解決した設定・scalar定義を`.scratch/prd117/acceptance`へ保存する。NMRND/SERNDはユーザー判断に従い、旧frozen入力の受入用コピーだけを現行キーへ移行する。変換内容と元/先のSHA-256も保存し、元のファイル・bat・旧exeは変えない。

既に解決した構成は`--run --only 2`のようにIDを指定して実行できる。出力先・終了予算・画面の更新頻度はCLIで固定し、学習や評価の設定は維持する。既存の試行は上書きせず、次のattemptのディレクトリへ保存する。Atari evalonlyは同じ受入で合格した既定Atariのcheckpointを使う。

保存済み試行の最新結果を集計する。未完了や不合格がある場合は終了値1になる。

```bash
.venv/Scripts/python.exe core/anet-core/testdata/prd117/run_acceptance.py --summarize
```

`@learn`の定義が無い場合の代替判定は、trainの構造化ログか、定義済みのRunnerのlearn_step scalarで1以上を観測する。exp stepやcheckpointの値では代用しない。learnタグが定義されていれば、learn scalarの実際の記録が必要である。この判定はRunnerを起動せずに保存成果物の形式から検証できる。

```bash
.venv/Scripts/python.exe core/anet-core/testdata/prd117/run_acceptance_test.py
```

## 通常Runの結果（2026-11-12）

全27構成が成功した。各行は最新試行の結果で、失敗した初回試行も消さずに保存している。全行で終了コード0・WARN 0。学習Runはlearn側scalarの実レコードを確認し、評価専用Runは4つの評価セッションの終了を確認した。保存を有効にした構成は保存完了まで確認し、MuZeroと評価専用Runは指定どおり終了時保存を無効にした。すべて通常停止とObserverの排水を終えている。

「停止予算」は`app.exp_exit_step`で要求した値で、停止要求の確認とbatchの境界により実際の停止は後ろにずれることがある。「最終のtrain exp記録」はtrain側のexp軸を持つscalarの最終記録であり、評価側のexp軸や、停止時の厳密なcounterとは混ぜない。

| ID | Env / 構成 | 停止予算（exp step） | 最終のtrain exp記録 | learn scalar件数 | WARN | 終了コード | 結果 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 01 | Atari / 既定 | 208,320 | 208,128 | 1,396 | 0 | 0 | 成功 |
| 02 | CartPole / 既定 | 67,840 | 67,328 | 2,815 | 0 | 0 | 成功 |
| 03 | DropMerge / 既定 | 42,240 | 41,984 | 3,706 | 0 | 0 | 成功 |
| 04 | GridMaze / 既定 | 19,200 | 18,688 | 2,854 | 0 | 0 | 成功 |
| 05 | GridMaze_muzero / 既定 | 804 | 796 | 1,956 | 0 | 0 | 成功 |
| 06 | ImageCls / 既定 | 2,176 | 1,920 | 217 | 0 | 0 | 成功 |
| 07 | LunarLander / 既定 | 24,832 | 24,320 | 6,245 | 0 | 0 | 成功 |
| 08 | Atari / RT30B | 208,320 | 208,128 | 1,396 | 0 | 0 | 成功 |
| 09 | Atari / SE30 | 208,320 | 208,128 | 2,804 | 0 | 0 | 成功 |
| 10 | Atari / NMRND | 208,320 | 208,128 | 1,489 | 0 | 0 | 成功 |
| 11 | Atari / SERND | 208,320 | 208,128 | 3,371 | 0 | 0 | 成功 |
| 12 | Atari / TGCH | 208,320 | 208,128 | 3,623 | 0 | 0 | 成功 |
| 13 | DropMerge / BASE | 42,240 | 41,984 | 3,706 | 0 | 0 | 成功 |
| 14 | DropMerge / SKBASE | 42,240 | 41,984 | 4,021 | 0 | 0 | 成功 |
| 15 | LunarLander / LLRND | 24,832 | 24,320 | 7,028 | 0 | 0 | 成功 |
| 16 | Atari / RT | 208,320 | 208,128 | 1,396 | 0 | 0 | 成功 |
| 17 | Atari / RT50 | 208,320 | 208,128 | 1,396 | 0 | 0 | 成功 |
| 18 | Atari / SE | 208,320 | 208,128 | 2,804 | 0 | 0 | 成功 |
| 19 | Atari / RT50NZ | 208,320 | 208,128 | 1,402 | 0 | 0 | 成功 |
| 20 | Atari / RT30 | 208,320 | 208,128 | 1,396 | 0 | 0 | 成功 |
| 21 | Atari / RT10 | 208,320 | 208,128 | 1,396 | 0 | 0 | 成功 |
| 22 | Atari / RT10RND | 208,320 | 208,128 | 1,799 | 0 | 0 | 成功 |
| 23 | Atari / RT10V | 208,320 | 208,128 | 1,396 | 0 | 0 | 成功 |
| 24 | Atari / SENZ | 208,320 | 208,128 | 2,810 | 0 | 0 | 成功 |
| 25 | Atari / SENZPA | 208,320 | 208,128 | 2,810 | 0 | 0 | 成功 |
| 26 | Atari / SENZPASK | 208,320 | 208,128 | 3,434 | 0 | 0 | 成功 |
| 27 | Atari / evalonly | 11 | 9 | 135 | 0 | 0 | 成功 |

GridMazeは実効tauを0.001に保ち、未使用の旧指定だけを削除した。ImageClsは未使用phaseの削除と教師ラベルをNN入力仕様から除いた後、最終Debug Runnerで再実行した。評価専用Runの初回は停止予算1でセッションへ届かなかったため、評価間隔10を越える予算11で再実行した。評価条件は変えていない。NMRND/SERNDは受入用コピーだけを現行キーへ移行し、原本8ファイルのSHA-256が元と一致することを再確認した。

集計は`.scratch/prd117/acceptance/summary.json`（27/27、pending 0）、個別のコマンド・設定・ログ・終了値は同所`runs/*/acceptance.json`を参照する。初期Debug RunnerのSHA-256は`debug-executable.json`、ImageCls/評価専用の最終再実行用は`debug-executable-final.json`に保存した。後段の設定読み取りのDTO化は、有効な構成の値を変えないことを既定coreと全出荷カタログで再確認している。

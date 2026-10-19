# PRD093 検証

## 比較資料と現状

実装の正本は `docs/memo/093_attention_entropy_diagnostics_20impl.md`。ログ・実行体・設定基準は `.scratch/prd093/` に保存する。

- `before/`: 変更前の Debug / Release Runner・DLL・config と対象ソース。`before-manifest.json` に hash とサイズを記録する。
- `before-current/`: ConfigManager で解決した現用7環境の全キー・文字列値。過去の固定入力に依存しない。
- `02-rename-red.log`: 初期の改名テストの失敗記録。後続の検証でgetterの観測先が誤っていたと判明したため、改名behaviorの有効なRED証拠とは扱わない。DTOの公開メンバを確認するテストへ修正した。
- PRD061 の過去入力を現用設定へ当てる既存採取テストは、削除済み `run.@nature_dqn` により停止した。その出力は参考として保持し、今回の比較基準には使わない。

2026-10-17: GPU解放後の再開指示により、保留していた両本線smoke、Release OFF/ON、NVTX費用採取、Release関連テストまで完了した。結果は末尾の「GPU検証の最終結果」に集約する。過去段階の保留やTDD順序の未達は、その時点の記録として以下に残す。変更前Releaseの退避とユーザーの既存変更を保持し、commit/pushは行っていない。

## 検証する契約

- 改名単独の前後比較はキー名以外の全キー・文字列値が一致すること。PRD072固定入力・golden・manifestは変更しない。
- captureの完全一致・末尾 `/` の接頭辞、重複・未要求・未達、書込み順、入口Reset、位置付与、部分forward、detach。
- エントロピーの独立MHA参照、一様・集中・系列長1、pre/post-LN、SN、CPU FP32とCUDA autocast。
- 購読interval・更新前測定・不足時NaN・値失効・未購読時追加計算ゼロ・層番号検証。
- 学習用RNG、PER・sampled-once統計、正規化統計、network bufferに非干渉で、実行modeを例外時も復元すること。
- 両smokeは4行の実効定義とlearn_step503・1006の測定値を確認する。Atariのmunchausen・replay_fitも保持する。
- OFF/ON checksumはtag・step・点数・数値を比較し、性能・資源群と48群だけを除く。既存47群を除外せず、checkpoint SHAは判定に使わない。
- interval503の測定費用と実時間比を報告する。1%は目標。長期Runの分析は別タスク。

## 再現手順と残る検証

CPUの再現手順はGPUを不可視にして実行し、ユーザー許可により既存Runnerとの並走を認める。Debugだけをビルドし、Release実行体を更新しない。GPUの本線smokeと性能測定は既存Run終了後に別途実施し、Runスクリプトの排他ガードを維持する。既存のexecution.jsonは上書きしないため、再試行時は過去記録を保持した別の採取先へ分ける。

1. AGENTS.mdのVsDevCmd経由でDebug全体をビルドし、`anet-core-test.exe "[prd093]"`、関連`[plasticity]`・`[nn]`・ImageCls/Munchausenテスト、既定スイートを実行する。CUDA無しのskipをGPU受入成功と扱わない。
2. `run_validation.py rename-snapshot`で退避した現用設定を複製して改名だけを施し、現在のConfigManagerで7入力を解決し直して、変更前の全キー・文字列値と比較する。追加診断キーやprofileの差分を改名比較へ混ぜない。現在の設定と複製元の差分も別途点検する。
3. `run_validation.py before-cpu`、続いて`run_validation.py cpu`を実行する。保存済み旧Runner、変更後OFF、変更後ONの順に比較する。LunarLander CPU、seed931、8 env、FP32、B16、RR1、capacity4096、warmup256、16640経験、4層の小型Transformer、probe16件、Dropout有効を固定する。学習由来の全scalarをtag・step・float.hex・点数で比較し、性能・資源群と48群だけを除外する（47群は含む）。checkpoint SHAは等価性に使わない。
4. `validate_run_test.py`の合成artifact回帰テスト、PRD072のprepare/compare、既定葉検査を実行する。PRD072の固定入力・golden・manifestは変更せず、`prd072-hashes.json`とも比較する。
5. `run_validation.py smoke`でDropMerge本線とAtari既定チェーン＋`@vitva > @attention`をDebug実行する。保存した既定のB/RR/warmupから1006更新点を越える予算を計算し、既存学習スケジュールを縮めない。`inspect_run.py tags`の出力も保存し、4タグのsource key・learn event・learn_step軸・update_result source・interval503と503/1006両点の有限な0〜1を確認する。Atariではmunchausen・replay_fitの定義が残ることも検証する。
6. Release全体をビルドし、`run_validation.py performance`で各環境をOFF/ONの順に測定する。503→1006の同一更新区間の実時間、process実時間、CPU peak working set・private usage最大観測値、GPU device全体memory.usedの1秒サンプリング最大値を保存する。GPU値はallocator単体や瞬間ピークではない。1対の差を統計的な性能保証と扱わず、必要なら同一条件で反復する。
7. NVTX/Tracyで`Learner::CaptureAttentionEntropy`のinclusive range合計を採取し、同じRun区間の実時間に対する割合を報告する。`CustomTransformerEncoderLayer::MeasureAttentionEntropy`・`ComputeAttentionEntropy`で内訳も追える。GPU非同期処理・同期の影響を説明し、OFF/ON実時間差をrange合計の代用品にしない。自動レポートの`profile_capture_seconds`と`profile_capture_ratio`は採取まではnull（未測定）とする。1%は目標であり、超過だけで不合格にしない。

Pythonはリポジトリの `.venv/Scripts/python.exe` で実行する。GPU本線smoke・Release性能測定・NVTX採取も実施済み。以下の過去段階の保留は最新結果で更新された。CPU比較のRun実時間は性能評価には使用しない。

## テストの観測面

- `nn_test.cpp`: captureの操作、通常/準備済み/部分forward、繰返し位置、独立libtorch MHA参照、pre/post-LN、非ゼロbias、SN/spectral_cap、CPU FP32・CUDA BF16、no-grad。
- `dqn_based_agent_test.cpp`: `ConfigureScalarMetricSubscriptions`・`UpdateFromBatch`・`BatchUpdateResult::GetScalar`。異なる間隔の和集合、更新前値、結果の寿命、未測定・不足時NaN、購読ゼロ、番号不正、層数超過、Transformer無し、例外時mode復元。
- 実LearnerのTD/QR/IQN、各Munchausen方式、Noisy・PER・SN・Dropout・正規化を含むON/OFF比較。parameter・gradient・buffer、学習用RNG、既存loss/Q/TD/grad指標、正規化count、evictionの未sample統計を比較する。
- `config_baseline_test.cpp`: DQN/ImageClsの共用キー、常時検証するprobe.batch_size、DropMerge/Atariの既存metrics保持とattention4行の追加。
- `validate_run_test.py`: 合成metadata/metricsによる配線違反、欠損、値域外、Atari既定診断の欠落の検出。

## 編集段階の静的確認（追加許可以前の記録）

`git diff --check`で差分の空白エラーなし。追加Python 3ファイルはAST parseのみ通過し、モジュールやテストは実行していない。現用C++・設定・日本語設計文書・補助スクリプトの旧契約参照を検索し、PRD072固定資料41ファイルのhash不変とユーザーのgui.cpp差分保持を確認した。詳細は `.scratch/prd093/static-review.json`。これらをコンパイル成功・テスト成功の代わりに扱わない。

## 軽量検証結果（2026-10-17）

- MSVC初期化経由のDebug全体ビルド成功。`24-debug-build.log`。性能テスト・hiddenテストを除くCPU全体スイートは697件中688件成功・9件skip、91,715アサーション成功。`25-default-cpu-tests.log`。capture到達branchの追加REDもこのスイートでGREENを確認した。
- 改名単独の現用7入力は、キー名の置換以外の差分0。`rename-comparison.json`、`18-rename-comparison.log`。
- PRD072の固定17入力、156アサーションが成功。`19-prd072-comparison.log`。固定資料41ファイルのhashは保持する。
- 既定葉検査は25,981代入、3,463監査、エラー0。検査器の回帰12件も成功。`16-default-leaves.log`、`17-default-leaves-tests.log`。
- 合成artifactの検証器テスト3件成功。`14-validator-tests.log`。
- PRD093 CPUテストは12件中10件成功・CUDA対象2件skip、46,232アサーション成功。`22-prd093-cpu-tests.log`。CUDAの成功をCPU skipから推定しない。
- 変更前・変更後OFF・変更後ONのCPU Runは、既存Noisy診断を含む61タグ・31,694点が完全一致。性能/資源群と追加48群だけを除外し、tag・step・点数・float.hexを比較した。3本のSHA256は `cdfb115f6ae74a702c881d3527d76e33d48039ac0472e626d75b74a954216941`。`18-cpu-comparison.log`、`cpu-comparison.json`、各`*.normalized.summary.json`。
- CPU Runの4層すべてでlearn_step503・1006を含む有限な0〜1を確認（全採取値0.934948〜0.970297）。`cpu-attention-values.json`。この小型LunarLander経路は両本線smokeの代替としない。
- 通常forwardの未達エラーで到達branchが空になる点を追加テストでRED確認し、最後に実行したbranchを報告するよう修正。`23-capture-red.log`。既存全体テストで見つかった改名前のエラー文言期待1件も、新しい共用キーへ移行した。

記録した失敗試行は削除せず、後続の成功ログと区別して保持する。初期の編集段階で実行を保留したRED/GREENは、後から実行済みだったとは扱わない。

### CUDA検証と残項目

`small-cuda-tests.log` は12ケースをCPU/CUDAで実行し、TD/QR/IQN × Munchausenの4方式 × CPU/CUDAの24組合せの非干渉チェックが成功した。独立MHA参照は値・精度・buffer不変の検証が成功した一方、`cuda` と実Tensorの `cuda:0` の比較によるテスト側の8失敗があった。比較先を入力Tensorの実deviceへ修正し、`26-test-build.log` の再ビルド後に `[nn][attention][prd093]` を再実行した。`small-cuda-nn-green.log` で3ケース・225アサーションすべて成功し、CUDA BF16、pre/post-LN、bias、SN/spectral_cap、通常出力の一致を確認した。成功済みのLearner非干渉チェックは再実行していない。

小型CUDAテストはGPU全体使用量の増分2,048MiB・空き1,024MiBを境界に、こちらの子プロセスだけを止める監視付きで実行した。初回127.7秒・観測増分最大618MiB、再確認15.8秒・596MiBで、どちらも監視による停止なし。記録は `small-cuda-tests.json` と `small-cuda-nn-green.json`、監視資材は `.scratch/prd093/run_small_cuda.py`。device全体の観測値には既存Runの変動を含み、プロセス別allocationでも本線smokeの費用でもない。実行時間も性能比較には使用しない。

CPU Runのchecksum採取後のproduction修正は、capture未達エラーに最後の到達branchを表示する変更だけ。最終Debug全体ビルド・CPU全体スイートはその修正後に成功した。追加したNNテストのdevice比較修正後は該当NNテストをCPU/CUDA双方で再確認した。

残項目はDropMerge本線・Atari既定チェーン＋`@vitva > @attention`の大容量GPU smoke、およびReleaseでの時間/メモリとNVTX/Tracy範囲の費用報告。設定上の既存診断保持はテスト済みだが、両本線Runの503・1006測定の成功とは区別する。全体受入は未完了。既存Runを停止せず、Release実行体、既存gui.cpp、PRD072固定資料を保持し、commit/pushは行っていない。

## GPU本線検証の再開（2026-10-17）

ユーザーのGPU解放・再開指示を受け、Release全体ビルドを実施し成功（`28-release-build.log`）。上記の軽量検証時点の保留・Release保持はその時点の記録であり、現用Releaseは今回のソースで更新した。変更前の実行体は引き続き `before/` に保存する。

DropMerge本線smokeは正常終了し、4タグの実効定義、503・1006の有限な測定値を確認した（`smoke_dropmerge.summary.json`、`smoke_dropmerge.tags.txt`）。初回のAtari smokeでは、既定プロファイルが検証用Run名・batch終了予算・評価無効化を上書きしていた。今回起動した検証子だけを停止し、Run出力と失敗採取記録を `failed-attempts/atari-profile-overrides/` の説明とともに保持した。

採取器は検証条件を `run.@prd093_validation` として既定チェーンの末尾へ置くよう修正した。実際のPython生成入力をConfigManagerで読むhiddenテスト `[prd093-validation-config]` でRED（4失敗）を確認し、修正後は8アサーション成功（`43-config-contract-red.log`、`44-config-contract-green.log`）。Atariのみ再開する場合は `run_validation.py smoke --env Atari` を使用できる。再試行の実効dumpでも終了予算478528・全評価interval0・検証用Run名を確認した。

Nsight Systems 2026.3.1の反復range採取は接続エラーとなり、reportを採取できなかった（`29-nsight-preflight.log`）。2026.1.1の通常NVTX採取でreport生成とSQLite exportを確認し、`profile_run.py --nsys` でこの経路を使う。`profile_run_test.py` は同一更新区間への対応、capture欠落のfail-fast、不正時計・duration、親範囲と内訳の二重加算防止の3テストをRED/GREENで確認済み（`33`〜`42`の関連ログ）。

費用の実行順は、GPUを排他にしてRelease OFF/ONを各環境で直列実行し、その後別RunでNVTXを採取する。範囲の母数は同じprofile Runのclock(1006)-clock(503)。その区間に含まれるprobeは1006側であり、503側を重複計上しない。`Learner::CaptureAttentionEntropy`のhost inclusive時間は同期D2Hを含み、先行GPU処理の待ちも含み得る。CUDA kernel時間と同一視しない。内訳範囲は親へ加算しない。

## GPU検証の最終結果（2026-10-17）

- Debug本線smokeはDropMerge・Atariとも正常終了。4タグの定義と503・1006の有限な0〜1を確認した。DropMergeは287.369秒、Atari再試行は545.973秒。`smoke_dropmerge.summary.json`、`smoke_atari.summary.json`、各`*.tags.txt`・`*.execution.json`。
- Atariの既存Munchausen/replay-fit診断19行は、smoke・Release OFF・ON・profileの4 Runすべてで記録があり、全19行に有限値を確認した。`atari-existing-diagnostics.json`。
- Release OFF/ONの実効設定は、両環境ともRun名と追加attention4行以外の全キー・文字列値が一致。`gpu-effective-config-comparison.json`。測定4本は同じRelease SHA `78a1dcfc183d5c890332bd09fd23282f74bc36e77033d3c5275e94e356104680` を使用した。
- 最終Release全体ビルド成功（`48-final-release-build.log`）。Releaseの `[prd093]` はCUDAを含む12ケース・91,970アサーションすべて成功、skipなし（`49-release-prd093-tests.log`）。CPU側の既定スイート688成功・9skip、変更前/OFF/ONの61タグ・31,694点の厳密一致は前節の記録どおり。

### Release実時間・資源

| 環境 | 購読 | 503→1006秒 | process全体秒 | peak working set MiB | peak private MiB | GPU device観測最大 MiB |
|---|---|---:|---:|---:|---:|---:|
| DropMerge | OFF | 32.924 | 110.486 | 3201 | 25498 | 24075 |
| DropMerge | ON | 32.861 | 108.836 | 3218 | 25356 | 23913 |
| Atari | OFF | 22.946 | 99.768 | 5860 | 29742 | 12881 |
| Atari | ON | 22.972 | 98.042 | 5889 | 29744 | 15709 |

503→1006のON/OFF差はDropMerge −0.1913%、Atari +0.1133%。各1対の測定であり、統計的な性能保証・高速化の根拠とはしない。GPU値はdesktop等を含むdevice全体の1秒サンプリング最大で、allocation単体や瞬間ピークではない。working setとprivateも異なる指標として扱う。記録は `performance-summary.json`、`perf_*.execution.json` と対応するRun。

### NVTX host inclusive範囲

| 環境 | step0 ms | step503 ms | step1006 ms | 3回合計 ms | 503→1006区間比 | 初回含む記録Run区間比 |
|---|---:|---:|---:|---:|---:|---:|
| DropMerge | 317.513 | 23.021 | 22.782 | 363.316 | 0.0685% | 0.4394% |
| Atari | 1225.349 | 13.813 | 12.840 | 1252.002 | 0.0567% | 1.5036% |

503→1006区間比の分子は、その区間に含まれるstep1006のprobe。母数は同じprofile Runの33.271秒（DropMerge）・22.628秒（Atari）。初回を含む比率は3回合計を、同じRunの最終通常時計点までの82.692秒・83.268秒で割った。最終時計点は全probeを含むがprocess終了より前のため、process全体の正確な比率とは区別する。Nsight CLIはreport生成時間も含むため、その実時間をRunの母数には使わない。

定常側は両環境で1%未満。一方、初回を含むAtariの記録区間比は約1.50%で目標1%を超えた。計画どおり、超過だけを不合格理由としない。初回費用と、host範囲に含まれ得る初期化・先行GPU処理の同期待ちを残して報告する。head/query/sampleの平均、FP32・NoGrad等の仕様を性能のために変更していない。

`profile_dropmerge/`・`profile_atari/` にnsys report・SQLite・実行体/config hash・summaryを保存した。両Runで親probeは3回、層のentropy内訳範囲は12回。内訳を親へ二重加算しない。profile側もattention4行・503/1006・正常終了とAtari既存診断を検証済み。

NVTX採取の再現は `profile_run.py --nsys` に利用するnsys.exeの絶対パスを渡す。今回は `C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.1/target-windows-x64/nsys.exe` を使用した。`profile_run_test.py` は初回を含む記録Run時計比の追加RED/GREENも含め4テスト成功（`50-recorded-run-cost-red.log`、`51-recorded-run-cost-green.log`）。この時計比の追加は保存済みartifactの再集計であり、GPU Runの再実行や測定値の置換は行っていない。

保留していたGPU機能・費用の検証は完了。初期TDD順序の未達は前節の記録を維持し、後から満たしたとはしない。全ログは `.scratch/prd093/` に保存し、既存ユーザーRun・変更前実行体・PRD072固定資料・無関係なdirty変更を保持した。

最終補足: profiling付き/なしONの実効設定も両環境で比較し、Run名以外は全キー・文字列値が一致した（`profile-effective-config-comparison.json`）。検証ツールの最終集約7テストとRelease版の採取条件契約8アサーションも成功（`52-validation-tools-tests.log`、`53-release-config-contract.log`）。静的最終監査は `static-review.json`・`54-final-audit.log` を参照する。

## レビュー指摘の修正（2026-10-17）

`final_feature_branch_key`のDTO既定値を空文字へ戻した。`main_feature`はbaseline設定が指定する値とし、baseline未適用の設定・dumpでは未指定を保持する。これにより、関連診断の購読時に既存の必須検証が働く。旧キーの検出・alias・読み替えは追加しない。DQN/ImageClsの設定不在と、現用DropMerge/Atariのbaseline適用後の値を回帰テストで区別した。修正前は未指定を確認する2アサーションが失敗した（`55-review-default-red.log`）。

`dqn_based_agent.cpp`と`nn_impl.cpp`のUTF-8 BOMを復元し、`<charconv>`は自前ヘッダの後で標準ヘッダと並べた。設定txt 5本は全行LFの状態から、`core.autocrlf=true`に合わせて作業ツリーをCRLFへ統一した。BOMと改行を正規化した文字列内容は変更していない（`review-formatting.json`）。C++・MarkdownのLFは維持する。

既存のGPU smoke・等価性・費用記録は修正前の実行体による結果として保持する。これらのRunは共通境界キーをbaselineから明示的に解決しており、今回の修正はその値や診断の数式を変えない。今回のレビュー対応では性能Runを再測定しない。

レビュー修正後のDebug全体ビルドは成功（`56-review-debug-build.log`）。CPU既定スイートは697ケース中688成功・9スキップ、91,719アサーション成功（`57-review-cpu-suite.log`）。未指定値の回帰、baseline適用後の値、DQN/ImageClsの購読時検証を含む。

レビュー修正後のRelease全体ビルドも成功（`58-review-release-build.log`）。CPU指定のPRD093関連テストは12ケース中10成功・CUDA対象2スキップ、46,238アサーション成功（`59-review-release-tests.log`）。BOM・include順・改行、現用の旧契約参照なし、PRD072固定資料41本と既存gui.cppの保持を最終監査した（`review-audit.json`）。

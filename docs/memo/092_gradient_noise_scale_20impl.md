# PRD092 勾配ノイズスケール実装メモ

## 概要

2026-10-15 承認済み計画。PRD092 の現行仕様（15指標）を実装する。未決事項監査のユーザー判断ブロッカーは0件。実装の正本は本メモとし、PRDの現行裁定を優先する。

## 主な変更

- Optimize(loss, weighted_element_loss) に変更し、TD・QR・IQNと既存テストを移行する。要素損失と平均は既存のautocast内で計算する。TDのPER無効時の error.loss.mean() の経路を保つ。
- nn_util に小バッチ勾配取得、信号・分散推定、Adam前処理、EMA対の計算部品を置く。retain_graph=true、create_graph=false、allow_unused=true とし、zero_grad前に測る。乱数、.grad、optimizer stateを変更しない。
- Network にfeature依存閉包とreadoutの学習パラメータ集合を返すAPIを追加する。既存の重みノルムと分類規則を共有し、勾配計測には使用されたNoisyNetのsigmaも含める。既存の重みノルムの対象は維持する。
- 使用パラメータは今回の勾配から判定する。全小バッチで未使用なら除外、一部だけ未使用なら寄与0。Adamは更新前state、所属groupのbeta2、learner.adam_epsで前処理し、不足する群とallだけ未成立にする。
- 購読ゲート・EMA状態はLearnerが所有する。発火点の和集合で1回測り、購読された版・群の成立したEMA対を各1回更新する。各keyは自身の発火時だけ公開する。未成立回はEMAを進めず前回値を公開しない。有限の負の信号は保持し、EMA信号が非正ならscale・ratioはNaN。空群もNaN、未知keyだけnullopt。
- num_micro_batches=8、ema_alpha=0.05、全15行interval 503の metrics.scalar.@gradnoise とDropMerge用 run.@gradnoise を追加する。baseline外とする。購読時にk・B整除・alpha・FP16・必要なfeature keyを検証し、間隔混在は起動時1回WARN。未購読時は値域・組み合わせを検証しない。
- 公開ヘッダの変更はLearnerConfig、Networkの群取得API、nn_utilの計算部品に限定する。checkpointへEMA状態を追加しない。

## テスト

Public surface: ConfigureScalarMetricSubscriptions、UpdateFromBatch、BatchUpdateResult::GetScalar、Networkの群取得API、nn_utilの計算部品、設定読込、runner出力。

優先behaviorとTDD順序（各1テストのRED→最小実装のGREEN→必要なrefactor）:

1. 購読→UpdateFromBatch→GetScalarの縦スライス。素のraw値が測定更新だけで得られる。
2. 合成問題の解析値、小バッチ勾配平均、群の加法性、Adamの単位・非一様前処理、偏り補正・epsilon。
3. 残存.grad、未使用、部分的未使用、群ごとのstate不足、空群、非有限値、負の信号、未成立後の再成立。
4. 初回発火・bucket-crossing・異なる間隔・同時発火・共有EMA・key別公開・rawからのEMA再現・WARN回数・購読時fail-fast・未購読NoCare。
5. TD/QR/IQNのFP32とCUDA BF16でON/OFF非干渉。毎更新のparameter・gradient・buffer・optimizer state・Learner乱数状態・既存数値指標を比較。通常AdamWとFusedAdamW、NoisyNet・PER・Munchausenを含める。

コード編集前に変更前binary・DLL・設定を保存する。決定論LunarLander CPUで変更前対変更後OFF、変更後OFF対ONの既存数値指標を比較する。PRD084 compare_metrics.py は追加の除外prefix指定を加え既定動作を維持する。PRD092では49群と性能・資源群を除き、既存Noisy指標も比較する。

## 検証と文書同期

- VsDevCmd経由のDebugビルド、対象回帰、既定スイート。
- BF16・NoisyNet・B256/RR1のDropMerge smokeで15定義・購読・source keyを照合。Adam state形成後のraw8指標は有限、scale常時成立は不要。
- 同一Release・checkpoint・seed・warmup・比較learn step数・出力条件でOFF→ONを交互に各3回。既定k/interval/alphaを維持。warmup後wall-clock、中央値・最小最大・増加率、CPU/GPUピークと差分を報告。2%は目標。
- 正常完了Runのkey別予定発火点と有限値数から値の未成立率を集計。空群は予定点0、率未定義。
- 設計200、分析ガイド030、設定、CHANGELOG、core/anet-core/testdata/prd092/README.mdを同期。解釈限界・共有EMA・経験数をそろえたraw比較を説明する。設定は既定葉監査と実効値比較を行う。

## 前提と完了条件

無関係な変更を保持し、commit/pushしない。履歴の11指標・all先行完成は採用しない。自動調整、因果分岐実験、追加状態指標、BatchNorm専用検出、将来Agent拡張は対象外。15指標、受入検証、本線smoke、性能・メモリ報告、文書同期がそろって完成。CUDA BF16未検証なら当該受入は未完了とする。


## 実装・受入結果（2026-10-15）

15指標・計算部・群分類・購読/共有EMA・設定・文書同期を完了した。CPU FP32とCUDA BF16、通常/Fused AdamW、TD/QR/IQN、Noisy/PER/Munchausenを検証した。checkpointへEMAを追加していない。

- 変更前binary/DLL/configは `.scratch/prd092/before/` に109ファイル、5,919,117,968 bytesを複製。初回保存した本計画のSHA256は `55840291e937eb3857287a211deafdf3f0af075a7a195804d4e5b7bc585e3935`。
- TDDはraw公開、群別Adam計算、15指標購読、比較・未成立率ツールの順でRED→GREEN。最終対象テスト8 cases / 17,578 assertions成功。Debug全体とRelease Runnerのビルド成功。既定スイート684 passed・1 skipped / 55,539 assertions成功（CUDA利用不可の検査だけskip）。
- CPUの変更前・変更後OFF・15指標購読ONは既存74 tags / 35,951点が完全一致。Noisy指標を比較に含めた。既定葉監査errors=0、変更前対OFFの既存設定値差分なし。
- 本線BF16 smokeは `run_20261015-174306_prd092_smoke`、exit0、実更新510回。15行の実効定義・購読・source keyとraw8指標を確認。
- 同一Release・checkpoint・seed・warmup・1,006更新でOFF→ONを交互に各3回。時間中央値はOFF 68.764s（66.403〜71.776）、ON 69.106s（68.345〜69.424）、増加率+0.50%。既定k8・interval503・alpha0.05を維持。
- メモリ中央値差はCPU PeakWorkingSet +14.97MiB、PrivateUsage最大観測値 -161.62MiB、GPU device memory.usedピーク観測値 -17MiB。GPUは0.5秒samplingであり瞬間ピーク・allocator単体値ではない。
- 各keyの予定発火点・有限値数・未成立率、全6回の時間とメモリ、再現コマンド、証拠一覧は [検証README](../../core/anet-core/testdata/prd092/README.md) に記録した。受入Runと入力checkpoint元Runの実効configは、検証fixtureとしてtestdata/prd092/configへ逐語保存した。
- 設計200・分析ガイド030・設定3ファイル・CHANGELOG・検証READMEを同期した。因果を評価する長期分岐実験は範囲外、B/RR/学習率の成績効果は判定不能。

### 実行上の補足

既存Runnerの停止確認後に重い検証を再開し、GPUメモリ解放についてもユーザーの明示回答を受けた。既存プロセスは停止していない。既存Release実行体を上書きせず `.scratch/prd092/release/` へ分離ビルドし、CMakeの一時設定は復元した。
最初のCPU ONはprofile上書きで未購読だったため受入から除外し、15定義を必須検査する最終Runを再取得した。最初のsmokeは503実更新だったため、505回以上と2購読点を満たす最終Runを再取得した。試行Runも削除していない。
無関係な未コミット変更を保持し、staging・commit・pushは行っていない。mainへ未コミットのため、PRD/実装メモはdocs/memo直下に保持する。

### レビュー指摘の反映

030 §4.9の本文を操作手順と確認表へ整理し、理由・仕組みは補足枠へ分離した。config複製11本は実験記録用ではなく検証fixtureのため、testdata/prd092/configへ移動した。非分割購読でのfeatureフラグと、SetupOptimizerによるAdamW型の不変条件をコメントで明記した。計算・設定・測定結果は変更していない。

### レビュー修正（2026-10-15）

1. `TestLearner`に行ごとの損失係数を指定するテスト専用の口を足し、公開の購読→`UpdateFromBatch()`→`GetScalar()`で非正の平滑化信号を検証した。係数が前半+1・後半−1ならraw信号は負・分散は正、`variance_raw = -8 * signal_raw`、素/AdamのscaleとratioはNaN、Adam信号・分散は有限であることを確認する。全行+1ならscale・ratioが有限になる対照も同じテストで確認する。未指定の既存テストの損失は変えない。
2. `ComputeParameterNormSplit()`をHEADのbranch単位の加算→SN置換→最後にheadを加算する順へ戻した。本体は`ANET_PROFILE_FUNC()`以外HEADと同一。勾配計測用`GetParameterSplit()`と公開ヘッダは保持した。
3. 設計200 §10のPRD092の確認事項を番号付きリストの10番へ移した。
4. PRD092で追加したテスト5件、新しい負信号テスト、nn_utilテスト全体、計算部と購読設定の新設部分だけを周囲の書式へそろえた。既存5件と計算部・購読処理のtoken列一致を検査し、式・呼び出し順・識別子を維持した。
5. 030 §4.9の4段落を節末の`> [!NOTE]`枠の箇条書きへ移した。本文の手順・表と補足の文章は保持し、英訳は変更していない。

検証結果:

- MSVC初期化経由の`cmake --build --preset x64-Debug`成功。ログ: `.scratch/prd092/20-review-debug-build.log`。
- 追加テストは入力口未実装のコンパイルREDを確認後、最小のfixture変更でGREEN（1 test case / 94 assertions、正係数対照を含む）。ログ: `17-review-test-red.log`、`18-review-test-green-build.log`、`18-review-test-green.log`。
- [gradnoise]: All tests passed (17672 assertions in 9 test cases)。ログ: `.scratch/prd092/21-review-gradnoise.log`。
- [plasticity]: All tests passed (292 assertions in 29 test cases)。ログ: `.scratch/prd092/21-review-plasticity.log`。
- 既定スイート: test cases:   686 |   685 passed | 1 skipped / assertions: 55633 | 55633 passed。ログ: `.scratch/prd092/21-review-default.log`。
- 重みノルム本体のHEAD一致、整形前後のtoken一致、030の本文と4段落の保持、確認事項1〜10の順序を検査した。証拠: `.scratch/prd092/19-review-static-verification.json`。`git diff --check`も成功。
- ビルド終了確認前に開始した最初のテスト実行は破棄し、自分の検証だけを停止してビルド成功確認後に全3種類を再実行した。正式結果は実行体hash付きの`.scratch/prd092/21-review-tests.json`。`premature-*`ログは受入に使わない。
- 本線smoke・Release性能Runは初回受入時の記録を保持した。今回、公開key・設定・既定値・学習経路は変更していない。commit・pushは行っていない。

今回変更したファイル:

- `core/anet-core/src/dqn_based_agent_test.cpp`
- `core/anet-core/src/nn_util_test.cpp`
- `core/anet-core/src/nn_impl.cpp`
- `core/anet-core/src/nn_util.cpp`
- `core/anet-core/src/dqn_based_agent.cpp`
- `docs/design/200_dqn_agents.jp.md`
- `docs/design/030_user_guide_analysis.jp.md`
- `core/anet-core/testdata/prd092/README.md`
- `docs/memo/092_gradient_noise_scale_20impl.md`

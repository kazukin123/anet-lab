# RND による一生の新しさ 実装メモ

> 2026-11-06夜の公開API見直しと追加検証は [21impl](116_rnd_lifelong_novelty_21impl.md) に記録する。本書の当初の操作名・グラフ再利用・費用と検証結果は、変更前の実装記録として保持する。

## 概要

対象は [PRD116](116_rnd_lifelong_novelty_10prd.md)。2026-11-06 の実装依頼により開始。基準コミットは `c84da22701b6170d4e2fecff4e22637ddfd36d92`。PRD の暫定の既定を採り、公開の口は §5.1 に従う。

RND を独立した非公開実装とし、DefaultDQNAgent の facade でスケール済み外的報酬へ加算する。無効時の学習系列と checkpoint を維持する。Replay、Actor、NetworkModel、Learner の optimizer の役割表は変更しない。

## 主な変更

- `include/anet/intrinsic.hpp` に `LifelongNoveltyEstimator` と Factory を置く。倍率、補助学習、診断、Serializable の口だけを公開する。`src/intrinsic.cpp` は公開ヘッダの対とする。
- `src/intrinsic_rnd.hpp/.cpp` に設定 DTO、target/predictor、私有 AdamW、観測正規化、誤差の累積統計、採点回数、学習用乱数を置く。DQN は RND の型・設定の葉を参照しない。
- `default_dqn_agent.hpp/.cpp` に enabled/beta、生成、採点・done マスク・加算、診断、checkpoint の識別子と節を追加する。
- `scaler.hpp` と `scaler_impl.hpp/.cpp` に正規化状態の保存・復元を追加する。既存の変換計算・既定値は変えない。
- 正規化設定の局所的な値域は `ValidateObservationNormalizerConfig` に集約し、RND の設定読込境界から論理キー付きで呼ぶ。
- `agent.txt` と Atari/GridMaze/LunarLander の `run.@rnd` に既定・構造・指標を追加する。
- `intrinsic_test.cpp` と `testdata/prd116/` に検証を置く。CMake の CONFIGURE_DEPENDS により新規ソースを収集するため、登録だけの CMake 編集は不要。
- `200_dqn_agents.jp.md` と `.en.md` の §6.2・§7.2・§8.2・§9.1 と検証の案内を追随させる。索引 issue の対応内容と判定は実施結果に合わせて更新する。
- `212_rnd_lifelong_novelty.jp.md` を新設する。強化学習やRNDに詳しくない読者向けに、用語・基本概念、RND/NGUの背景、数式と処理順、数値例、設定・指標・保存再開を順に説明する。日英 `README` の §3 と日英200の §6.2 から案内する。ユーザーの依頼範囲は日本語の解説なので212の英訳は今回作らず、英語の索引・200からも日本語版と明記してリンクする。
- wikiの一覧再生成で新規RNDファイルが地図から漏れないよう、`reports/tools/categories.json` の Agent 共通基盤に `intrinsic` / `intrinsic_rnd` を登録する。分類名や既存ファイルの分類は変えない。

## 未決事項監査

ユーザー判断が必要なブロッカー: 0。

repo evidence で解決済み:

- `trainer.cpp` は CPU の観測・報酬から BatchExperience を組む。`default_dqn_agent.cpp` の既存 unique_lock の中で採点と加算ができる。
- `ObservationNormalizer` は Serializable ではなく、`RunningMeanStd` の統計も非公開である。保存要求には正規化機構自身の保存・復元対応が必要。
- `ObservationNormalizerConfig` の既定は use_centering=false、post_process_type=SymLog。RND の白色化では centering=true、post_process_type=None、robust_update=false を RND 側の既定とする。
- ADR0053 の optimizer options 再適用を RND にも適用する。ADR0062 の primary 出力を使う補助 Network とは入力と同期の責務が異なる。
- 新しい用語と ADR の確定は PRD §8 に従って 924 に残す。

2ximpl へ固定:

- PRD §5.3 の「新しい転送を足さない」は CPU 観測と CUDA Network の組み合わせでは成立しない。PRD 冒頭の暫定規定に基づき、必要な入力 H2D と結果 D2H を明示して実装する。既存ロック以外の同期機構は追加しない。転送と計算を計測区間に分ける。性能の合否は推定しない。
- 正規化は CPU 観測を既存部品で処理し、採点前の正規化結果を Update まで保持して使う。採点後に元観測から統計を更新する。更新済み統計で同じ観測を正規化し直さない。
- warm-up は採点回数で数え、期間中は統計のみ更新する。Update は warm-up 中に optimizer を進めない。
- train_fraction は独立した名前付き乱数による lane 単位の一様抽選。選択が空なら更新せず loss=NaN。
- predictor の採点グラフを Update まで保持し、選択 lane の誤差から逆伝播する。forward の追加を避けるため、逆伝播の計算幅は最大 B（PRD の B×fraction は損失に寄与する lane 数）になる。
- Atari の観測仕様は uint8 で、既存 Normalizer は整数を通過させる。RND 内で非カテゴリの byte 観測を float へ変換し、その仕様から専用正規化を作る。Q 入力の変換は変えない。
- NetworkBuilder の seed は ModuleRandomSource 用であり、通常の重み初期化には CPU の既定 generator が使われる。同じ named seed の初期値を再現するため、RND の構築中だけ CPU generator の状態を退避・seed 指定・例外時も復元し、CPU 構築後に目的 device へ移す。構築は既存の初期化順序で行い、採点中は既定 generator を切り替えない。
- facade の内側の Learner は差し替え口を持たない。テスト専用の production API を足さず、公開 CreateLearner 経由で gamma=0・初期Q=0・学習率0 の TD 誤差から投入報酬を確かめる。
- 無効時 checkpoint のファイル全体のバイト同一は、既存 libtorch optimizer がパラメータの C++ アドレスを保存するため、変更前の同じ実行体の再実行でも成立しない。実データで確認したため、識別子・Config・Network のバイト同一、optimizer の展開内容（宣言されたアドレスと、それに由来する serialization_id を除く）の一致を検証する。ファイルそのものは変更せず、raw SHA256 と非同一の事実も記録する。保存形式の変更はしない。
- 転送費用の事前見積もり（実測ではない）: Atari B=64・84×84 の float 入力は約1.8 MB/step。実効 12 GB/s と仮定したコピー成分は約0.15 ms/step。起動・同期を各10〜100 µsと仮定すると、入力H2Dと小さい誤差D2Hの合計は概ね0.17〜0.35 ms/step。補助学習時は抽選した index の H2D がさらに1回あり、B=64/fraction=.25 の期待量は128 bytes、概ね0.01〜0.10 ms/update。3回の合計は概ね0.18〜0.45 ms/step。index転送は warm-up 中・選択0件では行わず、laneループの外へまとめ、独立の selection_transfer 区間で測れるようにする。CPU正規化とNetwork計算は含まず、共有GPUの負荷下では保証できない。1 batch Env step ごとに実行し、性能の実測は指示により保留する。
- beta は StepCounts に従う ProfiledValue。無効時には beta と RND の葉を解釈しない。
- 既存の未コミット変更は保持する。着手時に変更されていた bat、wiki、索引等の変更を取り消さない。

PRD 範囲外: 924 の方策の家族、エピソード内の新しさ、Replay の注釈経路、学習効果の長期比較。

## テスト

Public interface / surface: Factory、Multiplier、Update、Diagnostics、Save/Load、DefaultDQNAgent の CreateLearner/UpdateFromBatch・GetScalar・Save/auto_load_file、設定 resolver。

TDD 順序（各 behavior ごとに RED → GREEN、GREEN 後にだけ整理）:

1. 小さいベクトル観測から Factory で作り、warm-up の倍率 1 と有限な診断を得る経路。
2. 同じ入力の補助学習による誤差低下、target 固定、倍率の値域、統計を更新する順序。
3. 正規化と RND の保存復元、採点回数、optimizer options の現在設定の適用。
4. enabled/beta/RND 設定の値域と無効時に他の葉を読まない契約。
5. facade の報酬加算、done と truncated、診断 key、評価専用の休止。
6. checkpoint の識別子と有無の一致・不一致、無効時の従来形式。
7. 設定プロファイル、CPU/CUDA、関連回帰。

## 検証

- 変更前に MSVC 経由で Debug をビルドし、実行体・DLL・設定・解決済み設定を `.scratch/prd116/` の新しい基準ディレクトリへ保存する。PRD098 の before/after/compare を今回の保存先で使い、過去の基準は変更しない。
- Debug ビルド: `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug -j 2'`。
- `[prd116]`、正規化、DQN checkpoint、`[shipped_config]` の関連テスト。設定変更は default leaves と解決済み全キーの比較も行う。
- 4 環境の intrinsic 無効で scalar checksum・defs を比較する。
- GridMaze と Atari KFM の smoke を実施し、warm-up 後の bonus・NaN・clip_ratio を確認する。短い smoke から一般的な効果を主張しない。
- 自己点検後に独立した `anet-review-agent` へ実装計画と実装を出し、差し戻しは修正して再レビューする。

## 前提

- 2026-11-06 の追加指示: ユーザー不在中も可能な限り進める。GPU 利用は許可。別の Run が動いているため性能測定は行わない。PRD §6.3 の ABBA と実時間・内訳測定は未実施として残す。
- 既定値は PRD §5.4。alpha の式は平均を引く NGU の形。done の bonus は 0、truncated は採点する。
- 評価専用では採点・更新を休止し、一度 WARN。checkpoint に必要な部品の生成と保存状態は有効設定に従う。
- コミット、ステージ、push は行わない。

## 経過と結果

- 2026-11-06: PRD・指定資料・既存コード・ADR・用語・wiki を照合。現行 Debug 実行体が古いため、変更前ビルドを開始。実装・検証・レビューは未完了。
- 変更前 Debug 全体ビルド成功。実行体・DLL・設定を `.scratch/prd116/baseline-20261106/` へ複製。`[prd093-config-snapshot]` は 15 assertions 成功。`testdata/prd116/run_validation.py before` で GridMaze・LunarLander・Atari・DropMerge が終了コード 0。Runner SHA256 は `80fb335199ce43dcc8b6a3618c5b60e99a591152f4d1b2a091eeb4d9426522d2`。
- TDD1: 公開ヘッダ未実装によるコンパイル失敗を確認後、Factory → Multiplier → Diagnostics の warm-up 経路を実装。`[prd116]` 1 test / 8 assertions 成功。
- TDD2: predictor 学習テストは loss=NaN、誤差が下がらないことの 2 assertions で RED。私有 AdamW と名前付き乱数を追加して GREEN を検証中。
- TDD2 GREEN: 2 tests / 310 assertions。TDD3 は保存未実装の例外で RED、正規化の Serializable と RND checkpoint を追加後、`[prd116],[scaler]` 6 tests / 399 assertions 成功。
- TDD4: 不正設定13件のうち未検証10件で RED。共通の正規化値域検証を含めて412 assertions 成功。
- byte 観測の同値テストで整数の正規化漏れ、CUDA device 配置漏れ、同じ seed の初期重み不一致を検出して修正。CPU/CUDA を含めて428 assertions 成功。
- facade は診断キー未実装と beta の負値を通す4件で RED。採点・done マスク・加算・補助学習・atomic 診断を追加後、Debug 全体ビルド成功、`[prd116],[scaler]` 11 tests / 445 assertions 成功（`.scratch/prd116/unit-tests.txt`）。
- `run.@rnd` を3環境に追加し、日本語・英語の設計書を追随。固定教師、過去統計による倍率、実学習報酬、Agent 保存再開の追加テストと、4環境の無効時比較を実行中。
- 固定 target・seed と既定 generator の非干渉・更新前誤差統計・実 Learner の TD 誤差・Agent checkpoint の有無と継続を追加。`[prd116],[scaler],[shipped_config]` 25 tests / 3060 assertions 成功（`.scratch/prd116/unit-tests-r3.txt`）。
- 現用7構成の全キー・文字列値を採取し、変更は `DefaultDQNAgent.intrinsic.*` の既定9葉の追加だけであることを確認。既存キーの変更・削除は0（`baseline-20261106/resolved-after-r1/`、snapshot 15 assertions）。既定葉検査は既存の `Atari.txt` の `app.online.eval_panel.fps` / `app.online.train_panel.fps` の2件を報告した。保存済み baseline-config と同じ行であり、本変更の追加行に違反は無い。今回の範囲外として保持する。
- 保存直後の累積診断を NaN にしていた2件と、非有限誤差を統計へ入れてしまう1件を追加テストで RED とし、修正後の全体ビルドを開始。
- `off-r1` の4環境の全 scalar 点が基準と完全一致: GridMaze 26,946点、LunarLander 24,399点、Atari 23,185点、DropMerge 31,152点。4環境すべての `metrics.scalar.defs.json` も一致。比較記録は `.scratch/prd116/baseline-20261106/off-r1_<Env>/comparison.json`。
- 無効 checkpoint の比較（warm-up 中の16 experience）: header/config と Network バイトは完全一致。optimizer は libtorch のアドレス metadata だけが異なる。基準同士の再実行でも raw は非同一、同じ比較結果となった。`compare_checkpoint.py` はアドレス9件だけを識別して比較し、tensor と options は除外しない。基準同士の再実行記録は `checkpoint-r1_repeat_before/execution.json`。
- 境界修正後の Debug 全体ビルド成功。`[prd116],[scaler]` は15 tests / 480 assertions。その後、CUDA 保存再開、skip 有無×intrinsic 有無の4識別子、現在の learning_rate / weight_decay 適用を追加し、16 tests / 570 assertions 成功（`.scratch/prd116/unit-tests-r5.txt`）。
- 最終候補の `checkpoint-r2` でも header/config、Network バイト、optimizer 展開内容が一致（`checkpoint-r2_checkpoint.json`）。既存11ファイルの BOM は基準と一致。変更対象はすべて LF。既存の staged/untracked 作業は維持し、stage/commit/push は行っていない。
- 自己点検: `docs/quality/impl.md` 全項目を適用範囲で走査。公開口は PRD §5.1、共通 normalizer の変更理由は本計画、所有は Agent → estimator → Resource/State。追加の UI ロック・スレッドは無し。改名・互換層は無し。ログは英語、コメントは日本語、診断は既知未成立 NaN / 無効 nullopt。`git diff --check`、BOM・LF、header using namespace、production dynamic_cast・TODO、共有契約の project 固有語の検査を実施。既定葉の範囲外2件と性能未実施以外の自己点検上の未決は無し。smoke の結果は終了後に追記する。
- 独立レビュー1往復目: 計画OK、実装差し戻し。原文は `.scratch/prd116/review-20261106-r1.md`。RND の `[B,F]` 特徴量の構築時検証、同一入力100回後のtarget出力・重み固定、未知checkpoint headerの拒否試験、集成体の指示付き初期化、テストの識別子定数化を指摘。すべて範囲内として対応し、判断が必要な事項と見送った指摘は無し。
- shape 回帰テストは、余分な軸を残す Body を公開 Factory が通してしまう1件だけで RED（17 tests / 577 assertions 中576成功、`unit-tests-r6-red.txt`）。RND 私有の HeadFactory が dummy features の rank2 を検証し、汎用 LinearHead の振る舞いを変えずに `[B, output_dim]` と laneごとの倍率を保証する。指摘された残りの試験・規約も同時に補い、全体ビルドと関連テストで再検証する。
- GridMaze の100k experience smoke は終了コード0。8指標を採取し非有限値無し。bonus_mean の最初/最後の四分の一の平均は `9.72e-5 → 5.93e-5`、error_step は `6.49e-6 → 1.84e-7`、predictor_loss は `1.89e-6 → 1.87e-7`。bonus_share は `[0, 0.578]`、clip_ratio は0。採点と補助学習の成立を確認し、学習性能の一般的な効果は主張しない。記録は `baseline-20261106/smoke-r1_smoke_GridMaze.json`。Atari KFMの100k experienceを継続中。
- `off-r2` も4環境すべて終了コード0、scalar checksumとdefsが変更前に完全一致。総点数105,682点（時刻・資源の90_perf群を除く）。レビュー後の変更はRND有効時だけの構築検証と計測区間・テストであり、無効経路の再変更は無し。
- レビュー修正後の Debug 全体ビルド成功。`anet-core-test.exe [prd116],[scaler],[shipped_config] --reporter compact --out .scratch/prd116/unit-tests-r6.txt` は28 tests / 3160 assertions 全成功。shape拒否、同一入力100回のtarget固定、未知header拒否も成功。
- 独立レビュー2往復目: 計画・実装ともOK、差し戻し0件。原文は `.scratch/prd116/review-20261106-r2.md`。再確認ポイントはAtari smoke継続中と性能未実施で、レビューOKはそれらの完了を意味しない。見送った指摘・開発者の判断が必要な事項は無し。
- 一覧の再生成前に新規RNDファイルの分類欠落を確認。`reports/tools/categories.json` に2ファイル名を登録し、`.venv/Scripts/python.exe -B reports/tools/wiki_support.py maps` は終了コード0、未解決PRD無し。生成変更は `321_agent_core.md` のファイルリンク3行だけ。既存のstaged変更を保持し、追加分を再レビューする。
- 独立レビュー3往復目: 分類追加と生成リンク、最新の計画を含めて計画・実装ともOK。原文は `.scratch/prd116/review-20261106-r3.md`。追加指摘・範囲外の検出・開発者への確認事項は無し。Atari smokeと性能測定の未了は引き続き区別されている。
- Atari KFMの100k experience smokeも終了コード0。公開された8指標に非有限値無し。clip_ratioは `[0, 0.125]` で1への張り付き無し。error_step の最初/最後の四分の一の平均は `8.33e-4 → 1.95e-4`、predictor_loss は `7.11e-4 → 1.87e-4`。bonus_mean は `3.66e-4 → 9.64e-4` で、Atariの減少を合否にはしていない。bonus_shareは分母が成立した1341点で `[0,1]`、分母0の未成立値は契約どおりNaNで公開対象外。warm-upや選択0件のpredictor_lossも未成立値である。記録は `baseline-20261106/smoke-r1_smoke_Atari.json`。両smokeともモデルを保存して正常終了した。

## 受入と証跡の対応

| PRD | 確かめたこと | 証跡 |
| --- | --- | --- |
| §6.1 無効時 | 4環境の105,682点と指標定義の完全一致。checkpointのheader/config・Networkバイト・optimizer内容一致 | `baseline-20261106/off-r2_<Env>/comparison.json`、`checkpoint-r2_checkpoint.json`。ファイル全体の非同一の理由と比較範囲は上記の前提 |
| §6.2 target / predictor / alpha / warm-up / 統計 | 同一入力100回後のtarget重み・出力固定、誤差低下、倍率値域、更新前統計による採点、CPU/CUDAのbyte正規化と保存再開 | `unit-tests-r6.txt` の RND tests |
| §6.2 facade / checkpoint / config / seed / metrics | 実LearnerのTD誤差で加算位置、doneとtruncated、4headerの保存復元と不一致・未知header拒否、現在optimizer設定、設定値域、名前付きseed・既定generator非干渉、GetScalar有効/無効 | `unit-tests-r6.txt` の DQN intrinsic / checkpoint tests。実Learnerを使う理由は上記の前提 |
| §6.3 性能 | 未実施 | 実行中Runとの競合によりユーザーが禁止。再開後にAtari B=64・GridMazeでABBA各3巡以上、Tracy内訳を採取する |
| §6.4 smoke | GridMaze・Atari KFMとも100k成功 | `baseline-20261106/smoke-r1_smoke_GridMaze.json`、`baseline-20261106/smoke-r1_smoke_Atari.json`、各`smoke-r1_tmp_smoke_<Env>/execution.json` |

smoke は保存済み実行体 `smoke-r1-bin/AnetRLRunner.exe`（SHA256 `c970096e4cacf066524a2094faaf0dd584656b5b07fef8ce732ff363ae6332b0`）で実行する。その後の本体変更は不正入力・構築shapeの拒否、Load直後の診断、selection_transferの計測区間であり、smokeの正常入力の採点・更新計算は変わらない。新しい境界の正しさは上記の現行ビルドのunit testsで確認する。

## 決まらなかったことと残す検証

- 実装判断の未決は無し。PRD §8の11件は元の方針どおり924の再グリルへ残し、今回確定しない。
- 性能測定はユーザーの指示で未実施。新たな起票はせず、索引issueの対応内容と本記録に残す。再開許可後、§6.3のABBAとTracy内訳を検証する。既存Runを停止・変更して測定環境を作ることはしない。
- 性能測定時の確認事項: `Multiplier` は更新前統計での `Normalize` に続き、統計更新のため `NormalizeAndUpdateStats` を呼び、その正規化結果を捨てている。画像観測ではCPU上の全画素の正規化が重複する。`normalize` と `statistics` の内訳を確認してから、共通normalizerへ統計更新専用APIを追加するか判断する。現時点では計算順序・APIを変えず、呼び出し箇所に `@todo` を残す。
- 範囲外の既存設定2件: `Atari.txt` の `app.online.eval_panel.fps ?= 14.5` と `app.online.train_panel.fps ?= 10` が既定葉検査に反する。基準にも存在し、本実装では保持した。工程末に提示した新規課題の候補をユーザーが選択し、2026-11-06 に [issue-20261106-ddd395](../management/issues/issue-20261106-ddd395.md) として起票した。起票時の再検査でも同じ2件を確認。既存へ追記・この場で直す・見送り推奨の追加候補は無し。

## 実装・検証の結果

RND本体、DefaultDQN facade、保存再開、診断、3環境の設定、200設計書の日英の追随、212解説文書の日本語版新設、検証資材、wiki分類を実装。Debug全体ビルド・28 tests /3160 assertions・無効時4環境の同一性・GridMaze/Atari KFM各100k smokeに成功。独立レビューは3往復、最後は計画・実装ともOK。最初の指摘はすべて修正し、見送った指摘は無し。レビューで未確認だったAtari smokeは上記のとおり完了し、性能だけは指示に従って未実施とした。

索引issueのAI判定を実装済みとし、statusと監理の項目は変えない。作業ツリーの既存変更を保持し、stage・commit・pushは実施していない。2026-11-06 にユーザーが選んだ範囲外の課題1件を起票し、未決の記録を完了した。性能測定の未実施理由を残したうえで、今回の実装工程を完了する。

## 追加指摘への対応（2026-11-06）

- ユーザーから、基本概念を丁寧に説明し、アルゴリズムの背景と実装の関係が分かる212番文書を求められ、上記の日本語解説を作成した。Markdownリンク28件中のローカル参照先、4laneの数値例、Mermaid 2図の16ラベルと用語表、UTF-8/LF、コードとの整合を確認した。日英索引と200日本語版からリンクし、後述のレビュー対応で200英語版からも案内した。
- PRD079 D1〜D4に合わせ、`intrinsic_rnd.cpp`、`scaler_impl.cpp`、`intrinsic_test.cpp` の定数・補助関数・`TEST_CASE` を無名namespaceの外へ移した。定数は無修飾の `constexpr`、補助関数は `static` とし、型定義だけの無名namespaceは中身をインデントした。計算処理と公開契約の変更は無し。
- `Multiplier` の二重正規化は上記の性能測定時の確認事項へ記録した。専用APIの追加判断は測定後に行い、今回性能測定は実施しない。
- 検証コマンド: `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug -j 2 && core\anet-core\bin\Debug\anet-core-test.exe [prd116],[scaler] --reporter compact --out .scratch/prd116/unit-tests-r7.txt'`。Debug全体ビルド成功、17 tests / 577 assertions 全成功。`docs/quality/impl.md` の全項目を適用範囲で自己点検し、型だけのnamespace・内部リンケージ、UTF-8/LF・既存BOM保持、`git diff --check` を確認した。

## 2モデルレビュー後の修正（2026-11-06）

- レビュー原文は `.scratch/prd116/review-20261106-2054-impl.md` と `review-20261106-2054-impl-claude.md`。Codexは計画・実装ともOK、Claudeは212の作成記録の不足とテストの2識別子の定数化漏れを指摘。ユーザーはPRDの修正を別枠とし、それ以外の修正を依頼した。
- 本計画の「主な変更」「実装・検証の結果」「追加指摘への対応」に212の新設・経緯・検証・日英のリンクを記録した。200英語版にも212日本語版の案内を追加し、翻訳元情報を現行日本語版へ更新した。
- `intrinsic_test.cpp` の `td_mean` と `skip_context` を定数化し、参照を統一した。
- ユーザーの追加指摘に従い、DefaultDQNAgentのseed生成とprobe用乱数初期化をひとまとまりに戻した。さらに共通基本処理から並べるため、RuntimeVars生成の後、RewardScaler生成の直前に内的報酬の推定器・診断の初期化を置いた。RuntimeVars生成との順序依存は無い。`MakeNamedSeed` は元seedと名前だけから値を求めるconst関数なので、この移動でseed値は変わらない。RNDの生成内容・条件は保持した。再配置後も下記と同じコマンド（出力先だけ `.scratch/prd116/unit-tests-r9.txt`）でDebug本体ライブラリ・テスト実行体のビルドと17 tests /577 assertions全成功を確認した。
- 検証コマンド: `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test -j 2 && core\anet-core\bin\Debug\anet-core-test.exe [prd116],[scaler] --reporter compact --out .scratch/prd116/unit-tests-r8.txt'`。Debugの本体ライブラリ・テスト実行体のビルド成功、17 tests /577 assertions全成功。実装品質確認観点を適用範囲で自己点検し、差分の空白、UTF-8/LF、既存BOM、英語版のリンクと翻訳元ハッシュも確認した。性能測定と2モデルの再レビューは今回実施していない。PRDはユーザーの別枠修正に委ねた。

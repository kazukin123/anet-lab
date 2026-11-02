# PRD097：通知後に終端エピソードをリセットする 実装メモ

## 概要

PRD: [097_env_auto_reset_after_notification_10prd.md](097_env_auto_reset_after_notification_10prd.md)。2026-10-20、ユーザーの `PLEASE IMPLEMENT THIS PLAN` により以下の計画が承認された。通知中の全laneを「遷移後・リセット前」にそろえ、学習・replayの遷移と完了値の系列を維持する。本書を実装の正本とする。

## 未決事項監査

- ユーザー判断が必要なブロッカー: 0。
- repo evidence: BatchEnvの2実装がStep内でリセットしている。Runner3種とEvalSessionEnvがcontinue_stateを使用する。採用episodeの通知はRunSessionがDoStepInternalの外で出している。
- 固定する前提: PRDのAPI・呼び出し順序・完了値の契約へクリーンブレークする。既存の無関係な変更は保持する。
- 範囲外: PRD094、開始状態のObserver通知、QValuePanelの手動操作改善。

## 主な変更

- BatchEnvへResetEndedEpisodes()を追加し、戻り値を既存のBatchResetResultにする。BatchStepResultと派生結果からcontinue_stateを削除し、現用の呼び出し元・テストを一括移行する。公開ヘッダrl.hpp、env.hpp、trainer.hppも対象。
- BatchEnvBaseに呼び出し順序検査を共通化する。未初期化の操作、段の省略・重複、Step直後の全体Resetは、env名・実際の前操作・期待する前操作を含むエラーで停止する。
- 逐次・ThreadPoolのBatchEnvはStepでは終端laneを保持する。段では終端laneだけリセットし、継続laneは直前のnext_stateを返す。結果は毎回新規に作り、auxは空のlaneも含めlane数を保つ。ThreadPoolは終端laneだけへ仕事を配り、既存のworker割り当てを維持する。
- エピソード構造検査を、Step結果の完了group検査と、段のepisode_start == (done || truncated)検査に分ける。全体Resetの検査は維持する。
- 逐次学習・通常評価は既存の通知をすべて出してから段を呼ぶ。評価セッションはRunSessionが採用episodeの完了通知を出してから段を呼ぶ。最後のstepも同じ。
- 非同期学習は次のDoStep冒頭で、前stepの学習待機・通知を済ませ、段を呼んでから推論する。最初の2回を含む既存の通知回数は変えない。終了時は学習待機後、残っている段だけを1回完了する。
- EvalSessionEnvは段の結果を保存する。次セッション開始時、全groupがfreshなら空auxで再利用し、それ以外は全体Resetする。セッション完了後も最後の段を呼べるようにする。
- ImageClsは先読みをStepに残し、開始フラグとstep数の切り替えを段へ移す。段では追加のデータ読み込みを行わない。
- Atariは観測・RGB・auxの作成後に残機喪失時のNOOPを移し、そこでゲームが終われば完了を確定する。Resetはゲーム再作成またはFIREを行う。completed_*を削除する。
- DropMerge・GridMazeは完了値の持ち越しを削除し、Reset後の完了キーをNaNにする。DropMergeのView用auxは残し、統計の計算式と浮動小数点演算順序を維持する。
- 主要処理に計測範囲と日本語の段階コメントを付ける。PRD §9の日本語設計文書と学習シーケンス図を同期する。英訳と過去の記録は変更しない。

## テスト

Public interface: Reset / Step / ResetEndedEpisodes / GetScalar / GetTensor、およびRunnerの通知から観測できる挙動。

以下を1件ずつRED→GREENで進め、GREEN後だけ整理する。

1. 最初の縦スライス: 実BatchEnvと逐次学習Runnerを通し、終端通知中はリセット前のscalar・tensorが読め、次の行動にはリセット後の状態が渡る。
2. 呼び出し順序違反6種類、終端なし・混在・全終端、doneとtruncated、SHARED、auxのlane対応、過去の結果が次の段で書き換わらない。
3. 非同期学習・通常評価・評価セッションの通知順序と、セッション終了・キャンセル・再開始・Runner終了時の段の過不足。
4. ImageClsの先読み回数、終端フラグ、padding、評価windowの系列。
5. Atari・DropMerge・GridMazeの完了値とReset後のNaN、DropMergeのView用aux保持。

### Atari受入2(e2)

AsterixのRAM操作で初期状態を作る。seed=1、内部frameskip=1、sticky=0、noop_max=0、fire_reset=false、episodic_life=true。最小action集合に対し初期値1の32bit LCG (1664525*x + 1013904223)で300フレーム進め、RAM 0x53=1、0x47=3を設定する。次のNOOPでは残機が減って未終了、その次のNOOPで終了する条件を使う。

RAM設定は内部ヘッダのテスト専用friendから行い、検証は公開APIで行う。通常実行用の設定キーや操作口は追加しない。Python版ALEでは成立を確認済みだが、リンク対象のC++版ALEで再確認する。Step直後に完了値が成立し、段の後にNaNになることを必須とする。再現できなければ合格扱いにせず、PRDの規定に従って停止する。

## 検証

資材はcore/anet-core/testdata/prd097、結果は.scratch/prd097/implementation-20261020へ保存する。変更前後の実行体、DLL、設定、実効値、ハッシュ、実行引数と結果を分離する。基準コミットと既存dirty一覧を経過へ記録する。

共通条件: seed=1、経験数512000、決定的backend、deterministic_warn_only=false。定期評価は同期実行として発火と完走を再現可能にし、変更前後で同じ条件を使う。

| ケース | 構成 |
|---|---|
| Atari 1 | Breakout、pipeline、run.@v5_iqn_impala_x2、Envは@classic > E1 |
| Atari 2 | Atari 1のEnvを@100k > E1へ変更 |
| DropMerge | 現行設定、pipeline |
| LunarLander | run.@cpu |
| GridMaze | 現行設定、pipeline |
| ImageCls | 現行設定、pipeline |
| DropMerge serial | DropMergeにrun.train.runner_type=serialを指定 |

- 変更前を各2回、変更後を各1回実行する。変更前どうしが一致しない場合は原因を解消するまで本体変更へ進まない。
- Atari評価はeval_target / eval / greedy_distを有効にし、各2lane・2episode、interval=1000。残機設定、学習開始、評価・traceの成立を実効設定と成果物で確認する。
- scalar全点とtrace全行を比較する。除外はPRD指定の90_perf/・91_resource/・92_resource/とtraceのtimestampだけ。Noisy指標を追加除外しない。
- 残機7タグは変更前どうしでは値も比較し、変更前後では点数とstepを比較する。それ以外の数値差は不合格。
- 性能はAtari 1を変更前／後の順で交互に各3回実行する。後半の90_perf/12_exp_step_per_secの中央値をRun値とし、3反復の中央値の低下が2%以内であること。
- Debug全体ビルドと隠し以外の全テストを実行する。C++ビルドはAGENTS.mdのVsDevCmd.bat経由、補助Pythonはリポジトリの.venvを使う。実行コマンドと結果を経過へ記録する。
- EvalPanelで最後の残機を失ったstepの画面が、ゲームオーバー時の画面になることを確認する。

### 実行コマンド

リポジトリルートで実行する。C++のビルドはMSVC初期化と同じcmdプロセスで実行する。全テストはCTestに登録された各Catch2実行体を引数なしで呼ぶ既定スイートであり、隠しタグは既定どおり除く。詳細ログ・終了コードは本書へ記録する。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```

```powershell
ctest --test-dir out/build/x64-Debug --output-on-failure -j 1
```

```powershell
.\.venv\Scripts\python.exe core/anet-core/testdata/prd097/compare_metrics_test.py
```

```powershell
.\.venv\Scripts\python.exe core/anet-core/testdata/prd097/run_validation.py before
```

```powershell
.\.venv\Scripts\python.exe core/anet-core/testdata/prd097/run_validation.py after
```

## 前提・完了条件

- 負荷のある検証は裏のRunが終了した状態で行う。今回、検証再開の回答は受領済み。
- 自己点検後、独立したan-reviewへ計画・実装・検証結果を渡し、修正と再レビューを行う。
- 本書に検証・レビュー結果と未実施項目を記録する。必須受入が残る場合は完了扱いにしない。
- PRD094の実装、開始状態のObserver通知、QValuePanelの手動操作改善は範囲外。コミット・pushは行わない。

## 経過

- 2026-10-20: 承認済み計画を保存。基準コミットはd3d87edda8642725b8faa8fd3c21f5e41e890347（保存後に現在値を追記して確認）。本体コードの変更前に変更前Runの2回一致を検証する。既存dirty（apps/11_batch_run.bat、apps/13_batch_run.bat、docs/anet.asta、docs/memo/921_agent_prev_action_obs_10prd.md等）は対象外として保持する。

- 現在の基準コミットを再確認: ef7862d4f62b1b01d0b17b9e3589f68a3ff340f2。上記d3d87edは計画調査時点であり、今回の変更前ビルドはef7862dの作業ツリーを使う。

- 基準Runの初回起動でRunプロファイル展開がAtariEnvの選択を上書きすることを実効env.train.txtで検出。自分が開始したPID24484だけを停止し、結果は不採用のまま保存。最終run.@prd097プロファイルへ検証指定をまとめ、protocol-fixed/へ基準を取り直す。本体変更なし。

- 変更前Releaseビルドexit0（.scratch/prd097/baseline-build.log）。比較器の契約テスト1件OK。修正後Atari 1の1本目は202.3秒・exit0、比較対象93027点。残機7タグすべて成立し、train14240行と評価3チャネル各4行のtraceを確認。
- C++版ALEの独立プローブexit0。承認済みAsterix RAM条件で、frame301: lives3→1・未終了、frame302: NOOP後にゲーム終了を確認（.scratch/prd097/ale_probe.cpp / ale_probe.log）。本体ソースは未変更。

- 変更前Atari 1: 2反復exit0、93027点完全一致。Atari 2: 2反復exit0、90028点完全一致。続いてDropMerge等を順次検証中。
- TDD最初の縦スライスとしてtrainer_test.cppに通知中のscalar/tensorと次の行動の観測を確認する1テストを追加。本体を変更せずDebugビルド中。

- tracer RED: Debugのanet-core-test.exe [prd097] --reporter compactはexit42、1 test / 7 assertions中4失敗。通知中のscalarとtensor（TrainEvent/EpisodeEndEvent）が期待1に対して0。遷移のnext_stateと次の判断の開始観測は成功。ログ: .scratch/prd097/tracer-red-test.log。GREEN実装は残る変更前Runの一致を確認してから開始する。

- 計画レビュー1往復目: 検証コマンドの記載不足1件で差し戻し。検証節へAGENTS準拠のDebug全体ビルドとCTest全テスト、PRD097補助ツールの実行コマンドを追記。原文: .scratch/prd097/review-20261020-0236-plan.md。

- 計画レビュー2往復目: OK。原文: .scratch/prd097/review-20261020-0237-plan.md。実装・検証資材と受入達成は今回の判定対象外。実装後に同じreview_prd097へ実装全体を依頼する。
- 基準Run: LunarLander CPU 63648点、GridMaze 90851点も2反復で完全一致。計5構成が通過し、ImageClsを実行中。

### 2026-10-20 日本語設計文書・図の同期

- 対象の日本語設計文書5件を通知後リセットの契約へ更新。英訳・履歴資料は保持。
- Archify学習シーケンスは通知後のResetEndedEpisodesを追加。記録I/Oはデータフロー図と重複するためこの代表経路から省略し、通知・段の時間順を明確化した。
- validate/deliverはshowcase全9検査・errors=0・warnings=0。visual-checkは1440x900、1600x1000、1920x1080、2048x1320でoverflowなし。日本語の説明、固定Viewer UIは英語。レシートは`.scratch/prd097/archify-*.json`、表示証跡は`.scratch/prd097/archify-visual/`。
- 改修前ImageClsの1回目は512000経験・628.2秒で正常終了。2回目を実行中。本体コードは未変更。

### 2026-10-20 改修前検証の続報

- ImageClsの2反復は628.2秒/617.7秒、scalar・traceの67160点が完全一致。6構成の一致確認が済み、最後のDropMerge serialの2反復を実行中。
- `audit_validation.py`（実効値・評価trace/学習成立・DLL/設定ハッシュ）と`run_performance.py`（Atari1の前後交互各3回）を検証資材へ追加。比較器の契約テストは再実行してOK。
- EvalPanelのGUI受入は未実施。computer-useの必須資料を読み、node_replを初期化したが`failed to write kernel assets: 指定されたパスが見つかりません。 (os error 3)`。kernel reset後も同じエラー。UI操作は未実行であり、画面の受入を合格扱いにしない。

### 2026-10-20 改修前ゲート通過（03:00）

- 7構成・14 Runが全て正常終了。各構成の2反復は全scalar・traceで完全一致（構成ごとの点数合計728534）。DropMerge serialは206.9秒/199.5秒・162210点。
- `audit_validation.py before`はaccepted=true、14 Run、241ファイルのハッシュを記録。Atariの指定残機設定、3評価系各2lane/2episode・同期interval=1000、学習lossと評価trace各4本の成立を確認。
- 変更前binのDLL50個をhardlinkから独立コピーへ固定した。以後は保存した変更前実行体を保持し、本体実装を開始する。
- 最初のREDは既に保存済み。pure virtual追加とcontinue_state削除に伴うBatchEnv実装・Runner・テストdoubleの機械的移行は、最初のGREENへ必要な同一API変更として行う。

### 2026-10-20 実装と境界テスト

- `BatchEnv::ResetEndedEpisodes()`をpure virtualで追加し、Step結果からcontinue_stateを削除。逐次・ThreadPoolはStepで終端を保持し、段は終端laneだけ同じworker割当てでResetする。BatchEnvBaseに呼出順序の共通検査を追加。
- Runner3種とRunSessionの通知後に段を接続し、Pipelineは次DoStep冒頭とShutdownで残りを完了する。ImageClsは先読みをStepに残し、段で開始flagとstep数だけを切り替える。
- 最初の縦スライス: 変更前RED（4 assertion失敗）からGREEN（7 assertion）。`.scratch/prd097/tracer-{red,green}-test.log`。
- 混在lane/結果保持: 追加テストのREDでは遅延auxが後続Resetを読んで14 assertion失敗。結果生成時にauxを固定してGREEN。逐次/ThreadPool、終端なし・全終端・混在、順序違反6種、done/truncated、aux lane対応と状態/aux保持を確認。最初のテストと合わせ91 assertion。`contract-{red,green}-test.log`。
- 既存の関連51テスト・731 assertionは全てOK（`core-boundary-regression.log`）。新API移行時に必要となったRunnerの追加回帰は44 assertionがOK（`runner-boundary-test.log`）。Pipeline最初2回の通知数、Shutdown二重呼出し、通常評価、session最終段、通知中キャンセル後の再開始を確認。
- ImageClsは15テスト・148 assertionがOK（`imagecls-boundary-test.log`）。先読み済み画像の参照を段で再利用し、Step後に画像を壊しても段ではI/Oしないことを公開APIで確認。既存のpadding・eval window・train全lane終端も維持。新API実装の回帰として追加。
- ImageClsの古いPCHでC1853が出たため、対象buildディレクトリ内の生成済みPCH/objectだけを再生成して解消。
- Asterixの必須C++ Envテストを追加し、REDビルド中。friendはRAM初期条件設定だけに限定し、検証は公開BatchEnv/Module APIから行う。

### 2026-10-20 具体Env検証の続報

- Atari必須受入2(e2): 実際のDebug C++ ALEをリンクしたEnv/BatchEnv経路でRED（Step直後の完了scoreがNaN）を確認後、NOOPをStep内の観測/RGB/aux作成後へ移動。GREENは24 assertion。auxはframe301、完了値はframe302、段後は完了キーNaN。`.scratch/prd097/atari-{red,green}-test.log`。
- Atari全非隠しテスト38件・1016 assertionは全てOK（`atari-regression-test.log`）。旧Reset後のfinite期待は新契約のNaNへ移行。
- RAM fixtureがALEヘッダを使うため、`core/envs/atari1/CMakeLists.txt`のテストtargetだけにALE includeを追加。通常実行用設定/操作口は追加していない。
- DropMergeの次のREDは19個の完了キーがReset後もfiniteになることを検出（`dropmerge-red-test.log`）。新テストの通常ログ捕捉漏れで自分のテストがdialog待ちになったため、実行体パスを確認してPID42760だけ停止し、LogCaptureGuardを入れてやり直した。GREEN実装/ビルド中。
- GridMazeには専用test targetが無い。計画の公開Reset/Step/ResetEndedEpisodes/GetScalar検証を実施するため、既存GridMaze機能グループ内に`src/GridMazeEnv_test.cpp`と`GridMazeEnv-test` targetを追加する。実行main/DLL配置は他Envの標準形にそろえ、全体CTestは6実行体になる。これは受入テストの実行基盤追加であり、公開契約/設定を追加しない。

### 2026-10-20 全体検証の起動条件是正

- DropMergeはREDの19個の持ち越しを修正して64 assertion GREEN。全28テスト・342 assertionもOK。GridMazeはdone/truncatedのRED（8 assertion失敗）後に現在episode状態へ移してGREEN。続けて公開設定によるhole終端も追加した。
- Debug全体ビルドは成功。古いLunarLander/CartPole PCHを対象build下で再生成して解消。
- 最初の全体CTestはEnv5実行体がOK、coreは704テスト中6ケース・11 assertion失敗。build/core/anet-coreをcwdにしたことで現用設定への相対パスが見つからず、config_baseline/config/observer設定テストで期待値が空になった。
- 全テスト実施の前提を満たすため、`core/anet-core/CMakeLists.txt`のCTestだけにrepoルートのWORKING_DIRECTORYを指定する。通常実行やテスト期待値は変えず、既存の直接実行と同じcwdへそろえる。全6実行体を再実行する。
- coreの条件付きskip1件はCUDA不在時エラー専用テストで、この環境はCUDAが利用可能なためskipされたもの（init_test.cpp:145）。隠しテストはCatchの既定どおり除外。

### 2026-10-20 自己点検（独立レビュー前）

ユーザー判断が必要な設計未決は0。以下は`docs/quality/impl.md`の55項目を走査した結果。必須受入の未完了を「実装完了」とは扱わない。

| 項番 | 自己点検結果・根拠 |
|---|---|
| 1–3 | 変更hppのusing namespace、production dynamic_cast、大文字TODOは0件。static-checks.json。 |
| 4 | protocol/レビューroleの共有契約にanet/PRD参照0件。 |
| 5–7 | 計画名と対のPRDを確認。変更対象をGit diffと新規ファイルで特定。30ファイルのCR改行0。元のCMake UTF-8 BOMは保持。 |
| 8 | 対象外。skill実体は変更していない。 |
| 9–13 | 前提・公開API/通知surface・縦スライス順・受入matrix・MSVC経由コマンドを計画へ固定。前回計画レビューOK。 |
| 14 | 未完了。変更後7 Run・性能各3反復・EvalPanel手動確認はまだ未実施。Debug全CTestはcwd是正後の再実行中。 |
| 15–17 | PRDの段・通知順・NOOP・完了値契約へ一致。GridMaze専用target、AtariテストALE include、CTest cwd是正の理由は上記に記録。無関係なdirty変更は保持。 |
| 18–21 | code/tests/対象JP設計文書のcontinue_stateと完了snapshot旧識別子を検索し現用参照0。旧互換層は追加せず、英訳・ADR・過去記録は明示された範囲外として保持。 |
| 22 | 対象外。互換コードなし。 |
| 23–28 | 順序違反は常時ANET_SYSTEM_ERRORでEnv名/前操作/期待前操作を出す。意図した停止設定やfallbackは変更なし。ThreadPoolのWaitAll自体は待機のみで例外を伝播しない。初回レビューで誤記を指摘されたため、Reset/Step/段のworker例外を捕捉し、全タスクを待ってから呼出元へ再送出する処理と公開APIテストを追加する。Pipeline shutdownは残ったfutureをgetする。 |
| 29・54 | production設定は変更なし。検証用profileは境界をcolonで区切り、明示選択は=。 |
| 30–34 | 新規C++は既存GridMaze機能グループ内のtest.cppだけ。新規hppなし。公開ヘッダのAPI変更は承認計画内。SingleState/ActorRequestの集成体は指示付き初期化。 |
| 35・55 | 新しい暗黙global状態なし。結果/cacheはshared_ptrで所有。EnvTaskBatchはBatchEnv所有poolを同期呼出しの間だけ借用し、メンバ行末へ借用元を明記。poolの生ポインタ引数は保持しない。 |
| 36–39 | 新しい段階コメントは日本語。移動した統計初期化/blocked runコメントを移動先へ保持。View専用履歴を明示。未解決のコード上の懸念はなし。 |
| 40–41 | 段・Runner状態更新・Atari NOOP/soft/hard resetを既存のProfileRange形式で計測。計測のためだけのscope、任意のFULL計測名は追加なし。 |
| 42 | Atari/DropMerge/GridMazeの既知の完了キーはReset後NaN。未知キーnulloptと分離。ImageClsのaccuracy/epoch系列は維持。 |
| 43–44 | Agent production moduleは変更なし。Runner/Env自身が更新するStateだけを追加。Policy→Learner依存なし。 |
| 45 | 新しいruntimeエラーは英語。Atari完了logの値/順序は維持。 |
| 46–47 | public Reset/Step/ResetEndedEpisodes/GetScalar/GetTensorとNotifierを使用。RAM friendは初期条件だけ。縦スライス/順序・lane/各Runner/画像先読み/具体Env完了値のテストへ対応。 |
| 48–49 | 個別RED/GREENと回帰のコマンド・結果を上記ログへ保存。全CTest再実行、変更後比較、性能測定、UI手動受入の未実施を区別して記録。 |
| 50–51 | JP設計文書5件とsequence図を同期。120章からprd097 READMEとGridMaze/DropMerge専用テストを参照。英訳と履歴は触らない。 |
| 52–53 | 対象外。production設定ファイル/skillの変更なし。 |

EvalPanelのコード経路も確認した。`EvalPanel.cpp:122–128`の通知callbackがViewBase::UpdateViewDataを呼び、AtariView::CreateViewDataはその場でRGB/auxを表示データへ変換する。DoStep後のCaptureViewDataは保存済み表示データを適用する。ただし実画面を見た証拠ではなく、手動受入は未確認のまま。

### 2026-10-20 Debug全体検証合格

- `VsDevCmd.bat`経由の`cmake --build --preset x64-Debug -j 2`は成功（`.scratch/prd097/full-debug-build.log`）。
- cwd是正後の`ctest --test-dir out/build/x64-Debug --output-on-failure -j 1`は6/6実行体が合格、合計306.05秒（`.scratch/prd097/full-debug-test-root.log`）。CoreのCUDA不在時専用1ケースだけはCUDA利用可能のため条件付きskip。AtariのAsterix必須受入は実行済みでskipなし。
- 自己点検14/48の全体Debug検証は解消。変更後7 Run、性能3反復、UI手動受入は未完了。既存計画レビューと同じan-reviewへ、計画/実装のレビューを続けて依頼する。


### 2026-10-20 初回実装レビューへの対応

- 原文 `.scratch/prd097/review-20261020-0355-impl.md`。WaitAllが例外伝播するという自己点検の誤記、worker失敗未回収、HTML改行混在、コメント/JP設計文書4箇所の移行漏れを修正した。
- EnvTaskBatchでReset/Step/段のworker例外をlane別に保存し、投入済み全タスクを待ってから呼出元へ再送出する。enqueue途中の失敗でもdestructorが待機し、参照captureが生きたまま返らない。既存のworker割当て・段の終端lane限定投入は維持。
- 公開APIでReset/Step/段と各aux取得の6種類を検証。REDは修正前に異常終了（exit=3、`worker-red-test.log`）、GREENは12 assertion成功（`worker-green-test.log`）。他laneを遅延させ、再送出時にそのlaneも完了していることを追加確認する。
- 最終HTMLはLFへ統一。`git ls-files --eol`はw/lf、SHA256は`5f597302e594c35cd20166a5d833b2a50813f8e712c195e190f4a22ddee3501b`。LF後のvisual-checkも4 viewportでpass、1440x900を目視。レシート`.scratch/prd097/archify-visual-lf.json`はこの最終hashを記録し、生成直後のdeliver receiptとは区別する。
- 範囲外の検出として、overview評価図の既存`final EpisodeEndEvent`表記が現在のSessionEndEventと異なる旨を受領。今回の修正要求とは別件として保持。
- Release初回全体ビルドは既存metrics_logger.cpp.objのLNK1163で失敗。該当Release生成objectだけを再生成し、受入用Runner/Atariテストtargetのビルドをやり直す。Debug全体は修正後に再ビルド/CTestを行う。


### 2026-10-20 レビュー修正後の検証

- 最終Debug全体ビルド成功（`final-debug-build.log`）。遅延peerを含むworker失敗テストは6経路12 assertionが成功（`worker-green-final-test.log`）。
- `ctest --test-dir out/build/x64-Debug --output-on-failure -j 1`は全6実行体が成功、310.48秒（`final-debug-test.log`）。CUDA不在時専用の条件付きskip以外は実行し、Asterix必須受入も成功。
- Release受入用targetはmetrics_logger object再生成後に成功（`after-release-rebuild.log`）。Release Asterixも24 assertion成功（`atari-release-test.log`）。
- 独立an-review再レビューは計画/実装ともOK。原文`review-20261020-0406-impl.md`。新たな範囲外検出は既存observers.cppのproduction dynamic_castで、今回のコメント変更の外として保留。
- 最終41ファイルのCR検査・追加コードの規約検査は違反0（`final-static-checks.json`）。前の「production dynamic_cast 0」は当初対象30ファイルについての記録であり、新しく対象に加えたobservers.cppには既存1件がある。
- 変更後snapshotは`protocol-fixed/after`。Release実行体SHA256=`9a68ea49bda3483187c0e9618baa1745d8161e531897a92f500985fd4bb0aec8`。実行体/DLL/設定を独立コピーし、対象40ソース/文書のhashをsource-state.jsonへ固定。変更後7構成の比較、監査、前後交互3反復の性能測定を順次実行する。


### 2026-10-20 変更後の数値比較合格

| 構成 | 比較点数 | 実行秒 | 結果 |
|---|---:|---:|---|
| atari1 | 93027 | 228.6 | 完全一致 |
| atari2 | 90028 | 225.1 | 完全一致 |
| dropmerge | 161610 | 193.3 | 完全一致 |
| lunarlander | 63648 | 124.7 | 完全一致 |
| gridmaze | 90851 | 45.6 | 完全一致 |
| imagecls | 67160 | 625.7 | 完全一致 |
| dropmerge_serial | 162210 | 208.6 | 完全一致 |

7構成すべて正常終了し、比較点数合計728534。Noisy指標の追加除外なし。残機7タグは変更前後では点数とstepを比較し、それ以外のscalar/trace数値は完全一致。実行引数・実行体/DLL・設定・実効値・比較結果はprotocol-fixed/afterへ保存。性能判定とEvalPanel手動受入は未完了。


### 2026-10-20 初回性能受入不合格と是正

- `protocol-fixed/after/performance.json`の交互各3反復は不合格。変更前5290.42919921875、変更後4851.57275390625経験/秒、比0.9170470998123731（8.2953%低下）。この世代は数値比較7/7・監査に合格したが、性能不合格として保存し、採用しない。
- 個別速度中央値は変更前5271.0078125 / 5290.42919921875 / 5305.0263671875、変更後4862.60791015625 / 4851.57275390625 / 4835.4267578125。Noisy指標を除外したり閾値を変えたりしない。
- 段の継続lane処理が毎laneでTensorDictを生成し、観測を直列コピーしていた。既存のBatchState::Cloneでバッチごとに新しいstorageへ複製し、終端laneだけを上書きする。CPU終端maskも一度取り出し、laneごとのTensor select/itemを避ける。状態・auxの独立性、終端だけのworker投入、worker割当ては保つ。
- copy_state/reset_endedの主要フェーズ計測と日本語の意図コメントを追加。新API/設定はなし。既存の公開API境界/結果保持/例外伝播テストと全体Debugを再実行し、Release再構築後はafter-v2へ分離する。先にAtari1比較と交互性能を確認し、その後に残る6構成の最終世代を実行する。


### 2026-10-20 性能是正後の再検証開始

- 独立an-reviewは是正後も計画/実装ともOK。原文`review-20261020-0504-impl.md`。Cloneによる別storage、終端maskの寿命、aux/lane対応、worker割当てと例外回収、計測規約を確認した。
- Debug全体ビルド、Release受入targetビルドが成功（`v2-debug-build.log`、`v2-release-build.log`）。Release Asterix24 assertionも成功（`v2-atari-release-test.log`）。全体CTestを再実行中。
- `after-v2`実行体SHA256=`4ffffbb8d14f82584a90eaa8cf1d4a820944bfa5e63a012fae2b3e409775b5ce`。旧afterの実行体・Run・不合格性能記録は保持。v2の41対象ファイルの改行検査も違反0（`v2-static-checks.json`）。


### 2026-10-20 ImageClsテストmainの非対話ログ是正

- 性能是正後の全体CTestではCore/LunarLander/DropMerge/GridMazeが成功。ImageClsはPID29536がCPU停止・MainWindowTitle=Messageで待機し、テストログはseed出力だけだった。実行体パスを照合して自分のImageClsテストだけを停止した。失敗したCTest記録は`v2-debug-test.log`へ保持。
- ImageClsの既存mainは通常wxLogのstderr固定がなく、wx初期化後の通常ログがGUIへ流れ得た。Core/Atariと同じStderrLogGuardをmainへ追加する。テストの期待値・productionは変更しない。Debug全体増分ビルド後に、失敗したImageCls実行体を再実行し、他5実行体の成功記録と合わせて全対象を確認する。
- Releaseの受入RunnerとAtari targetにはこのテストmain変更が入らないため、保存済みafter-v2実行体は変更しない。


- ImageCls mainへ初期化前にStderrLogGuardを置いた最初の修正では、15テスト148 assertionは成功したが終了時SEH例外（`v2-imagecls-debug-test.log`）。ローカルwxWidgets `src/common/init.cpp:410–416`でwx初期化が既存log targetをdeleteすることを確認。ガードが保持したtargetの二重削除だったため、初期化をmainで一度だけ完了してからガードを設置する順序へ修正する。遅延ログそのものがSEH例外原因という初期仮説は訂正する。元のprocess-lifetime wx初期化方針とテスト期待値は維持。


### 2026-10-20 性能是正後のDebug全対象確認

- `v2-debug-test.log`のCore/LunarLander/DropMerge/GridMaze/Atariは成功。ImageClsだけGUIログ待ちのため自分のprocessを停止した記録を残した。
- wx初期化後に標準ログガードを設置する修正後、Debug全体増分ビルドが成功（`v2-imagecls-init-build.log`）。ImageClsの全非隠しテストをCTestで再実行し1/1成功・2.86秒（`v2-imagecls-init-test.log`）。これで全6実行体を確認済み。
- 終了時例外が出たseed=2440781346を指定してImageCls全15テスト148 assertionを再実行し、exit=0。`v2-imagecls-fixed-seed-test.log`。期待した画像破損エラーログはstderrへ出るが、テスト終了後もGUI待ち・SEH例外は起きない。
- 現在after-v2のAtari1数値比較→交互3反復の性能→全7構成比較（Atari1は同一実行体の成功Runを再利用）→実効値監査を順次実行中。性能・最終世代の全数値・EvalPanel手動受入が残る間は完了扱いにしない。


### 2026-10-20 v2性能不合格とプロファイル診断

- `after-v2/performance.json`の交互104–106も不合格。変更前5230.98291015625、変更後4806.515625経験/秒、比0.9188551573487035（8.1145%低下）。Atari1の93027点は一致したが、バッチCloneだけでは性能低下を解消できなかった。この世代も性能受入済みとは扱わない。後続6構成は性能ゲートで未実行。
- diagnoseスキルの手順へ切り替え、既存の前後比較を再現ループとして維持。仮説は(1)ResetがGPU学習との重なりから外れた待ち時間、(2)aux固定/worker例外回収によるStep増、(3)状態の確保・複製・解放、(4)追加構造検査の走査。ソース確認だけでは確定しないため、既存NVTX/CUDAの時間で区別する。
- 正式受入測定の終了後に、既存Nsight Systems 2026.3.1で各128000経験の短い前後診断を開始。`.scratch/prd097/diagnostics/profile_run.py`、同`profile-run.log`。診断だけの設定/Run/プロファイルはcapture時刻別フォルダへ分離し、受入結果へ代用しない。追加のproductionデバッグログは入れていない。


### 2026-10-20 診断結果と実アプリ終了段の是正

- 実学習開始は200000経験。最初の128000経験profileはwarmup区間だったため、320000経験へ延長し、256000–320000経験の500 stepを比較した。実lossは前後各467個がfiniteで一致。元の512000経験受入Runのlossも1217個が全てfiniteであり、学習成立は確認済み。
- `capture-20261020-055308/steady-summary.json`: Pipeline DoStep平均は29.46→32.04ms、Env Stepは13.09→10.21ms、段は4.01ms（複製0.15ms、Reset待機3.86ms）。学習workerは17.01→17.11msでほぼ同じ。
- `host-learner-overlap-summary.json`: 変更前のAtari Reset時間の56.08%がLearner::UpdateFromBatchと重なり、変更後のReset/段は0%。確保・複製・構造検査より、通知後へ移したResetが並行区間から外れたことが支配的。
- Nsightはdriver13.4未対応/13.3ライブラリ利用を警告し、後半のCUDA活動がNVTXと揃わなかった。`gpu-overlap-summary.json`をvalid=falseと明記し、GPU重なり率は根拠に使わない。上記は同じNVTX時計で得た学習workerとの重なりである。
- 別の実装漏れ: 実RunnerAppの終了がRunner::Shutdownへ接続されておらず、最終段が1回欠けた。短い実アプリcaptureは1000 Step/999段/最終Step後0段。`check_shutdown_profile.py`でRED（`app-shutdown-red.log`）。
- `RunnerApp::OnRun`のイベントloop終了後に学習threadをjoinしてTrainRunner::Shutdownを呼び、既存OnUnhandledExceptionへ終了時例外を渡す。通常のStop/再開は変更しない。新APIなし。実終了経路を直すための計画内移行漏れ修正としてRunnerApp.cppを対象へ追加する。


### 2026-10-20 実アプリ終了のGREENと性能設計判断待ち

- RunnerApp終了経路の変更後、Debug全体/Release受入Runnerの増分ビルド成功（`v3-debug-build.log`、`v3-release-build.log`）。固定したafter-v3のSHA256は`87752cd8584c29263b7953b2b5b3474e98683444b447c042f7c0bb4f974d9329`。
- 実アプリ128000経験のNVTXを`check_shutdown_profile.py`で検査し、1000 Step/1000段/最後のStep後1段でGREEN（`app-shutdown-green.log`）。修正前1000/999/0のREDも保持。評価を含むcaptureはこの検査の入力にしない。
- Atari実学習の成立監査をタグ存在からfinite lossの存在へ強化。変更前14 Run/241 hashの再監査も成功（`before-finite-loss-audit.log`、`before/audit-finite-loss.json`）。元の監査結果を保持した。
- after-v3の7構成数値比較・監査を実行中。v2の性能不合格8.11%は未解決で、終了経路の修正によって性能合格したとは扱わない。
- production外のC++プロトタイプ`diagnostics/ale_spare_probe.cpp`で、現在ALEのsystem stateを別ALEへ複製し、別ALEだけreset_gameしてから差し替えた。Breakout/Asterix/Pong、sticky=0/0.25、各12000フレーム、毎500フレームまたはgame-over時のResetで、各行動の報酬・RAM・RGB・残機・game-over・総/episodeフレームが一致した。準備中の現在ALEも変更されないことを比較した。`ale-spare-result.log`は6条件すべてPASS。これはプロトタイプで、正式受入の代替ではない。
- clone/restore平均は0.65–0.86ms、reset_game平均3.80–4.90ms。追加32 ALEのprivate memory差分は22036480 bytes（約21 MiB、`ale-spare-memory.log`）。1個ずつの差分はヒープ再利用の影響があるため、資源見積りには採用しない。
- 具体案は、headless Atariに予備ALEを所有させ、Step内で予備だけ次ゲームを準備し、通知後のResetで差し替える。現在ALEは通知が済むまで終端のまま維持する。Env RNGを使うNOOP/FIREの実行時点・RAM統計の扱いも契約として固定する必要がある。PRD §4.3.6のReset内部という指定を変えるため、ユーザーへ計画追加の採否を質問した。回答まではこの設計に依存するproduction/PRD変更を行わない。
- EvalPanelの実画面受入は引き続き未実施（UI実行基盤のkernel assets初期化エラー）。性能2%・最終世代数値比較・実画面受入が残る間は完了扱いにしない。


### 2026-10-20 追加変更の独立レビュー

- 同じan-reviewへround4 manifest（計画を含む43ファイル）を渡し、前回対象を維持して終了経路・ImageClsテストmain・finite loss監査の追加変更を再確認した。計画/実装ともOK、修正要求なし。原文`.scratch/prd097/review-20261020-0626-impl.md`。
- after-v3 Atari1は93027点が一致。残る6構成と監査を継続中。性能2%とEvalPanel画面は未達。レビューOKは作業全体の完了を意味しない。別ALE案は回答待ちを維持し、採用に依存する変更は行っていない。


### 2026-10-20 ユーザー指定の性能再計測

- 別ALE案の採否への回答は「再計測」。採用の承認とは解釈せず、現行設計を維持してafter-v3の変更前/後交互各3反復を再測定する。予備ALEのproduction/PRD変更は行わない。
- after-v3の7構成の数値比較がすべて完了し、728534点が一致。実効値/学習/成果物の監査も7 Run・158ファイル合格（`v3-validation.log`、`v3-audit.log`、`after-v3/summary.json`、`after-v3/audit.json`）。これまでの不合格性能結果も保持する。
- 再計測は同一seed/経験数512000/決定的backend/同期評価/固定設定を使い、変更前/後の順を3回維持する。各Run後半の90_perf/12_exp_step_per_sec中央値と、3反復の中央値を用いる。基準2%は変更しない。


### 2026-10-20 再計測の確定結果と別ALE計測範囲の明確化

- ユーザー指定の再計測107–109（変更前/後交互各3回）が6 Runすべてexit=0で終了。変更前中央値5077.56396484375、変更後4864.73291015625経験/秒、比0.9580840229367649（4.1916%低下）。基準2%以内は不合格。`v3-performance-validation.log`、`after-v3/performance.json`。

| 反復 | 側 | 後半速度中央値 exp/s |
|---|---|---:|
| 107 | before | 5026.51171875 |
| 107 | after | 4864.73291015625 |
| 108 | before | 5188.73681640625 |
| 108 | after | 4866.85595703125 |
| 109 | before | 5077.56396484375 |
| 109 | after | 4761.234375 |

- これは別ALEなしの現行実装の再測定であり、8.11%から4.19%という差を別ALE案の効果とは扱わない。別ALE案は独立したC++プログラムでの状態同値性・処理時間・追加メモリの予備検証のみ。Runnerへは未導入で、Run全体が何%速くなるかは未計測。ユーザーから測定範囲が読み取りにくいとの指摘を受け、区別を明示した。
- GPUを使う再計測は終了した。別ALE導入の採否は未承認を維持する。性能受入とEvalPanel実画面受入が残るため、PRD097の完了とは扱わない。


### 2026-10-20 別ALE試作Runnerの速度・メモリ測定（ユーザー依頼）

- ユーザーが「別ALE試作版でメモリ消費や速度について測定よろ」と明示。正式採用の変更とは分け、`.scratch/prd097/spare-runner-prototype`にAtariEnv.hpp/cppのコピー・試作実行体・測定資材を置く。PRD本体と正式productionソースは変更しない。
- 現行after-v3を対照とし、headless/sound=falseのAtariの終端Stepで別ALEへsystem stateを複製してreset_gameだけを準備する。通知中は現在ALE・scalar/RGB/aux/RAM統計を保持する。Resetで差し替え、BeginRamMetrics後に既存のEnv RNGによるNOOP/FIREを実行する。Env RNGの消費タイミングは維持。soft resetは従来経路。表示/音声ありは準備しない。
- 既存のAtari全テスト（特にAsterix2(e2)）と、Atari1の512000経験scalar/trace完全一致を先に確認する。同一128lane/設定/seed/経験数で現行/試作を交互各3回実行する。速度は各Run後半の経験/秒中央値、3回の中央値。Win32 process private bytes/working setを一定間隔で取り、後半中央値と全体peakを保存する。GPUメモリは既存のRun resource指標で比較する。新しい通常実行用設定キーは追加しない。
- 既存のCMake compile/link条件とオブジェクトを使い、差し替えるAtariEnv.cppとその内部ヘッダだけを試作用へコンパイルし、別のstatic lib/Runnerをリンクする。元のソース/実行体を上書きしない。計測前にソース・バイナリ・設定と引数を保存する。


- 別ALE隔離試作のRelease全Atari38テスト/1016 assertionが成功。128lane/512000経験の検証Run spare_10も93027点のscalar/traceが現行after-v3と完全一致（残機7指標を含む、値の除外なし）。最初のspare_0はサンプラーが定義行をscalarと誤認して失敗したため、自分のPIDだけ停止し、不採用記録を保持。修正後の検証Runは速度3反復へ混ぜない。productionソース2ファイルと正式Release exeは試作前hashと一致。これから同一サンプラーで現行/試作を交互各3回測定する。


### 2026-10-20 別ALE試作Runnerの実測完了

- 現行after-v3/別ALE試作を128lane・512000経験で交互各3回測定。速度中央値は4906.65234375→5110.91845703125経験/秒、+4.1630%。CPU Private Bytes中央値は+85.227MiB、Working Setは+87.699MiB。
- Run全体所要時間の中央値は193.911→198.378秒、+2.3037%。学習中指標は改善したが、この512000経験の全体時間短縮は確認できない。増加フェーズの寄与は未分解で、原因を断定しない。
- GPUのNVML値はdevice全体の参考値で、プロセス固有VRAMではない。CPUはWin32 GetProcessMemoryInfoを0.5秒間隔で採取。全6Runが93027点のscalar/trace（残機7指標を含む）完全一致。
- 詳細と各回の値は .scratch/prd097/spare-runner-prototype/report.md、生集計は同summary.json、追加集計はanalysis.json。試作は正式ソース/実行体へ未反映で、試作前hash一致を確認。別ALE対元改修前の正式交互性能は未実施のため、元実装比2%基準の合格とはしない。GPU測定は終了。


### 2026-10-20 文書A〜Cの隔離W試作・追加測定の確定結果

- ユーザー回答「Bも含めて文書A〜Cを実施」に従い、依頼 .scratch/prd097/perf-measure-request-20261020.md のW試作、Atari4版比較、非Atari比較、低下時NVTX診断を実施した。採否は決めず、正式productionソース・設定・Release exeは監査対象299件で試作前後SHA256一致。
- Wは .scratch/prd097/snapshot-runner-prototype-20261020 に隔離。終端Stepのworkerで読み取りModuleを所有する写しとReset結果を保存し、通知中のBatchEnv GetScalar/GetTensor/GetTensorVectorは写しを参照する。段は保存Reset結果から次状態を作り、AtariのEnv仕事をしない。scalar評価関数は現用値と写しで共通化。非Atari、Runner、EvalSession、操作順序検査は現行経路を維持する。正式APIや設計の採用変更は行っていない。
- 公開APIのW固有テストで想定1件REDを確認後、隔離ReleaseのAtari全39テスト1041 assertionがGREEN（Asterix受入2(e2)を含む）。事前の512000経験Runも93027点完全一致、残機7指標の値も一致。
- 新しい測定世代 .scratch/prd097/perf-comparison-20261020 に、seed=1・512000経験・決定的backend・同期評価の正式40 Runと、別のhost NVTX診断11 Runを保存した。全51 Runがexit=0でscalar/trace比較に合格。残機7指標は指定どおり点数/stepで比較し、Noisy指標の追加除外はない。実引数・実効設定・実行体と入力設定のSHA256、0.5秒メモリ/経験数サンプル、各組の値と比を保存した。

| Atariの側 | 3本の速度中央値 exp/s | before比 | after-v3比 |
|---|---:|---:|---:|
| before | 4979.097412 | 1.000000000 | 1.055454627 |
| after-v3 | 4717.490723 | 0.947459014 | 1.000000000 |
| S1 | 4836.621094 | 0.971385111 | 1.025252911 |
| W | 4906.333984 | 0.985386221 | 1.040030447 |

- 今回の主値は依頼の「最終速度報告stepの半分超」。従来の256000超も別の参照集計へ残した。旧世代の測定値と混ぜていない。Wの今回の中央値比は0.985386で指定2%基準の数値内だが、隔離試作の結果であり正式受入合格や採用とは扱わない。現行正式after-v3の性能受入未達は維持する。
- after-v3との差はS1が速度+2.525%・Private +87.988MiB・WorkingSet +88.309MiB・全体時間 +4.536秒、Wが速度+4.003%・Private -31.430MiB・WorkingSet -3.871MiB・全体時間 -2.083秒。CPUプロセスメモリの値で、NVMLはdevice全体の参考値。
- S1の予備ALE構築はNVTXで135回・合計5.749秒。正式測定の起動観測差+5.525秒と近い。別captureの後半で同期評価がafter-v3 48.315/94.722秒、S1 47.897/91.252秒、W 47.849/89.967秒を占める。時間重みEMAの点の中央値と、評価等を含む壁時計は別量として示した。

| 非Atari構成 | 組数 | after-v3/before |
|---|---:|---:|
| gridmaze | 3 | 0.955050889 |
| lunarlander | 5 | 0.839397174 |
| dropmerge | 3 | 0.972238781 |
| dropmerge_serial | 3 | 0.977439402 |

- LunarLanderのみbefore最初の3本のmax/min−1が2%を超え、5組へ延長。他は3組。4構成とも0.98未満なので前後各1回をcaptureし、学習開始後の末尾500 training Stepを比較した。
- Wの段は平均0.180ms、copy_state 0.150ms、段内EnvTaskBatch::Wait 0件。after-v3は段3.432ms・Step12.588ms、WはStep14.188ms。GPU kernelの重なりを測ったとは扱わない。
- 段のReset臨界経路近似/全体はGridMaze 0.335/0.445ms、LunarLander 0.039/0.190ms、DropMerge非同期 0.089/0.328ms、逐次 0.069/0.354ms。GridMazeはReset側の比重が大きく、他は複製・queue往復・同期等を含む残余が大きい。ただしLunar/Drop非同期のDoStep差全体は段直接費だけでは説明できず、GridMaze/Drop逐次の1回のcaptureでは正式測定の低下方向は再現しなかった。主因を断定しない。
- 詳細は .scratch/prd097/perf-comparison-20261020/report.md、audit.json、atari-summary.json、non-atari-summary.json、capture-summary.json。DropMerge after-v3の学習・保存後にNsight PluginTearDown待ちを確認し、非侵入cdbでスタックを読み取り後detach。正常終了・export済みで、学習区間と正式速度測定には混ぜていない。
- W専用Debugビルドは未実施（正式資材を変えない隔離Release性能試作として検証、コマンドと理由はWのplan.md）。正式PRD097の性能・EvalPanel実画面受入を代替せず、PRD097は未完了を維持する。commit/pushは行っていない。
- 計画/実装およびA/B測定記録の独立レビューはOK。最終報告・監査結果の独立レビューをこの後に行う。


### 2026-10-20 A〜C測定報告の最終独立レビュー

- an-reviewが2計画の追記・報告/監査/集計/更新スクリプト/診断ノートを確認し、OK 3件・差し戻し0件。修正要求なし。原文は .scratch/prd097/perf-comparison-20261020/review-final-20261020.md。対象manifestのSHA256/LF、40正式Run+11診断Run、正式299資材の不変、計画のappend-onlyも確認された。
- 今回依頼された隔離W試作と文書A〜Cの測定・報告は完了。正式PRD097の採用判断・性能受入・EvalPanel実画面受入の未完了は維持する。W専用Debug未実施の範囲と理由も保持する。commit/pushは行っていない。

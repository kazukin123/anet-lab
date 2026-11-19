# PRD182 ソース配置の見直し 実装計画

## 概要

基準 `7b5b52ed` から作成した専用 worktree で、[PRD182](182_source_layout_10prd.md) D16 の順に実装する。ステージごとに受入を記録してから次へ進む。元の worktree には編集済みの wiki・課題一覧などがあるため、そこでの編集と Run 資産を分離する。

## 主な変更

1. ステージ1。移動前に Debug の Runner、GUI テスト実行体、設定、リンク切れ一覧、追跡ファイルの改行属性、配布物の構成を採る。基盤を `src/core/base`、6 Env を `src/env`、Runner と Metrics Viewer のソースを `src/apps`、道具を `apps/tools` に移す。D4 のターゲット名にそろえ、テスト実行体と共有 DLL を `out/test/<構成>` に置く。Runner の本体と GUI テストは `apps/runner/bin/<構成>`、Viewer の jar は `apps/metrics-viewer/bin`、Maven 中間物は `out` に出す。bat、道具、CI、配布、Doxygen、分類、現行文書と検証資材を移す。過去の記録、英訳、生成物は D23 のリンク規則に従う。変更記録には旧い生成物の片付け方を足す。
2. ステージ2。§5.3 の割り振りに従い、DQN の共通部品とテストを学習、行動、Network、組み立て、計算の5組に分ける。定義本体とテストケースを比較する道具を検証資材に置き、比較対象の例外を S2-1 に合わせる。現用の利用箇所と検証資材を追随させる。
3. ステージ3。§5.4・§5.8 に沿って DQN・MuZero・ImageCls を別の CMake ターゲットへ移す。基盤は Head の拡張と N-step の共通部品を公開し、Runner は各系統を登録する。テスト用の口を追加し、具象 Agent と系統固有の Head に触れる現行ケースの行き先をテストケースの実物から一覧化して記録する。ターゲットと include の依存を確かめる道具を検証資材に置く。AGENTS.md と設計文書へ拡張点、テスト、依存方向の規則を書く。
4. ステージ4。Runner、Metrics Viewer、道具の実行時データの基準を固定相対の `data/` へ切り替える。追跡している8本を移し、配布には workspace config 7本を入れる。§5.5 の改名ツールと移行手順を用意し、使い捨ての fixture で事前表示、件数の一致、同名衝突時の中止を検証する。各マシンの既存 workspace と appdata は人が移す。

## テスト

- ステージ1。CMake、実行体の出力パス、bat、道具の既存入口を対象とする。最初に、GUI テスト以外のテストを `out/test/<構成>` から実行できる経路を S1-3 の受入で検査し、旧配置で不合格を確かめてから CMake を移す。次に、道具の通常入口を既存の workspace/Optuna/分析テストで確かめる。Viewer は移動後の Maven プロジェクトから jar を作り、起動 bat が新しい jar を選ぶことを確かめる。Runner の短い Run を標準起動の入口から確認する。
- ステージ2。既存の公開挙動とテストを基準にする。定義・ケースの比較器は、定義本体を変えた fixture で不合格を確認し、次のサイクルで現物の5組を検査する。
- ステージ3。CMake の依存検査、系統の初期化で Agent を選べる入口、公開ヘッダとテスト用の口を順に確認する。境界を越える接続の検出と、DQN テストの単独ビルドを確かめる。
- ステージ4。最初に `[workspace]` の旧置き場だけにある fixture で、相対指定が旧ディレクトリを拾わないことを検証する。次に相対 Run パス、appdata の優先順、道具・Viewer の workspace 一覧、移行ツールを個別に検証する。

## 検証

- 各ステージで C1 の3構成を AGENTS.md の Windows/MSVC 手順でビルドし、C2 の既定テストと標準テストを実行する。GUI テスト実行体は明示ターゲットでもビルドする。
- C3 の34構成は PRD128 §6.4 の入力生成と比較を再利用する。基準は PRD182 着手前に作り、WARN と ERROR の比較から、比較時に限りソース位置も除く。
- C4 は [リンクの注意](README.md#リンクの注意) の検査で基準との差を取る。各ステージで §6 の検索・ビルド後の配置・ツールの受入を確認し、結果をこの計画へ追記する。
- C5 の動線マップと architecture atlas は人の skill 起動後に確認する。和文の変更は翻訳の skill の入力として報告する。リリースワークフローの手動実行と Viewer 実画面の結果も、実施した範囲を記録する。

## 前提

- 旧配置の `core/` と `viewers/` が現存するため、ステージ1から始める。PRD181 の設定の配置、PRD185 の梱包方式は、それぞれの PRD が扱う。
- ステージ1では `apps/runner/workspaces` が workspace の根。ステージ4で各パスを切り替える。稼働中の Run の停止や、元の worktree の workspace・appdata の移動は行わない。
- 検証で作る Run は専用 worktree の検証用パスへ出す。固定した実行体は workspace を絶対パスで指定する。

## 経過

- 2026-11-19。元の worktree にある既存編集を避けるため、専用 worktree とブランチ `codex/prd182-source-layout` を作成した。ユーザー判断を要する実装の未決は0件。

- 開発者の確認を受け、ステージ1から順に実装する。追跡583パスの移動元・移動先は専用 worktree 内で、移動先の衝突は0件だった。
- ステージ1の配置検査は移動前の CTest 登録6件で RED。比較用 Debug 実行体と GUI テストのビルドを通し、PRD128 の34構成を採取して比較用 Run を開始した。
- 移動後、workspace 解決と Python 道具の標準テスト8本が成功した。Metrics Viewer の Maven package が成功し、jar が新しい場所に出た。
- 変更前のリンク切れ236件から105件に減り、増分0件。移動したファイルを指す行番号付きリンク131件を有効なアンカーにした。

- S1-4 は開発者の確認により、ワークフローの梱包処理を隔離fixtureで実行したローカル比較を、次のステージへ進む判定に使う。GitHub Actions での手動実行と実物zipの確認は後に残す。

## ステージ1の検証経過

- C1。Debug（Runner の GUI テストを含む）、RelWithDebInfo、Release を Windows/MSVC でビルドし、終了値0。
- C2。workspace・道具8本、Atari 分析24件、Viewer 285件の再試験が成功。Viewer の初回は4 failures・13 errors。再試験では Maven テストの作業ディレクトリをビルド出力へ移し、初回がソース配下へ作った一時データを保管した。
- C2 の初回で設定例の fixture のパス連結が旧階層を向いていたため修正した。対象ケースと基盤スイートを再試験中。
- C2 の既存3件は開発者が残件扱いを選択。ImageCls の閉じた設定カタログのテストと Eval Trace の1行選択 GUI テストは移動前にも同じ箇所で失敗した。番号付き bat の検査は旧 `SET EXE` とプロセス終了を前提とし、現用 bat は `RUNNER` と集計後の常駐を採る。検査と現行 bat の差は本ステージの移動前から存在する。
- C3。変更前の基準を `.scratch/prd182/comparison` に保存。基準 Run の完了後にステージ1との比較が始まるよう実行を設定した。
- S1-3。Debug CTest の6つの登録済みテスト実行体が `out/test/Debug` にあることを確認した。GUI テストは `apps/runner/bin/Debug`。
- S1-4。旧版と現行のワークフローの梱包手順を同じ fixture に対して実行した。パッケージには実物の Viewer jar を使い、Runner・wx DLL・Doxygen はパス検査用 fixture。新旧 zip の差は道具と jar の置き場に限られた。GitHub Actions の実物の確認は後に残す。
- S1-5。`apps/10_run.bat` から絶対パスで隔離した GridMaze workspace、256 experience、`tmp` を含む名前で実行し、終了値0と Run フォルダを確認した。
- S1-6 と S1-7。wiki maps は終了値0。移動した583パスの改行属性差0件。C4 の追加リンク切れは0件。C5 の生成物と和文の英訳追随は人の起動を待つ。
- 独立レビューの実装計画は OK。実装の差し戻し5件に対し、既存の BOM と改行、Viewer の検証資材と起動手順、push レビューと atlas の検知、bat のタイトルを修正した。PRD090 の `new` は合成の2048点で実行し、`validate` も隔離先で終了値0。原文は `.scratch/prd182/review-20261119-0745.md`。

- ステージ1の再レビューは計画・実装とも OK。原文は `.scratch/prd182/review-20261119-0802.md`。

- ステージ1。設定例の修正後に1190 assertionsと基盤既定スイートが通った。変更前の再現性と移動後の前後比較は34/34構成で一致した。承認済みの既存3件と手動確認を記録してステージ2へ進む。

## ステージ2の実装

- 共通計算、Network、行動、学習、組み立てに、関数と型の本体を移す。scalar の Actor Q hint 行と decoder は初期優先度の計算も使うため共通計算に置く。NetworkModel のキーを使う skip 入出力の補助は Network の定義に置く。共用する診断の型とキーは共通計算に置く。
- 265のケースを5役割と具象Agentの計6本に分ける。従来から複数のテストが使う fixture は、このテスト群の内部ヘッダに置く。
- 一時領域で定義・型・定数329件、テスト本体265件の一致を確認し、製品5本とテスト6本のMSVCコンパイルを通した。関数本体を変えた合成fixtureは比較が不合格になり、同じ本体を移したfixtureは合格する。

- 独立レビューは実装計画 OK。実装で指摘された4項目について、Catch ヘッダの順序、補助の翻訳単位、無名 namespace、設計書から検証手順へのリンクを修正した。再レビューは OK。原文は `.scratch/prd182/stage2-review-1.md` と `.scratch/prd182/stage2-review-2.md`。
- C2 の既定の基盤・Env テストは5/6実行体が通った。ImageCls Env は既存の1件で失敗。Runner GUI は38/39件で、既存の Eval Trace 行選択の1件が失敗した。これらはレビュー修正前の実行体での結果。
- S2-1・S2-2・S2-3、C4。レビュー修正後も329定義と265ケースが一致し、現用の旧名は0件。新たなリンク切れ163件は、すべて D23 で認める分割元3本を指す。

- ステージ2の受入用実行体では、着手前に保存した実行体と34/34構成で scalar・設定・WARN/ERROR が完全一致した。レビュー修正後は3構成とも再ビルドに成功し、同じ比較は34/34で再度一致した。既定テストの再試験では既存のImageClsとEval Traceに加え、MRU保存で一度だけAccess is deniedが出た。同じ最終版で対象ケースを直列再実行し4 assertionsが通った。

## ステージ3の実装

基盤の公開面に Head と Replay の経験組み立てを置き、DQN の設定と具象 Agent を系統へ移す。Runner は系統ごとの初期化関数を呼ぶ。基盤のテスト用の口は別ターゲットとしてビルドし、内部ヘッダに届くのはその実装までとする。

各ケースの検証内容を保持する。複数系統の設定 DTO を一つのケースで作っていた検証は系統別に配り、関連する設定の検証を SECTION にまとめる。基盤の共通処理は基盤で確認し、Noisy Head を含む検証は利用先の DQN に置く。PRD の静的概数と、具象 Agent を構成する現行ケースの数に差があった。現行コードの主題と構成に基づく移行先を下表に記録する。

| 移動元 | ケースまたは確認内容 | 行き先 |
|---|---|---|
| intrinsic_test | RND shipped Atari networks and Taiga settings match their contracts | DQN intrinsic |
| intrinsic_test | DQN intrinsic validates beta only when enabled | DQN intrinsic |
| intrinsic_test | DQN intrinsic state scoring ignores next observations and terminal masks | DQN intrinsic |
| intrinsic_test | DQN intrinsic learning cadence follows experience count | DQN intrinsic |
| intrinsic_test | DQN intrinsic masks done but scores truncated lanes | DQN intrinsic |
| intrinsic_test | DQN intrinsic known scalar keys stay NaN while disabled or suspended | DQN intrinsic |
| intrinsic_test | DQN validates the selected intrinsic observation only when enabled | DQN intrinsic |
| intrinsic_test | DQN scales the intrinsic reward before combining it with external reward | DQN intrinsic |
| intrinsic_test | Intrinsic return scaling masks before statistics and stores components before beta | DQN intrinsic |
| intrinsic_test | DQN learner consumes the scaled external reward plus intrinsic bonus | DQN intrinsic |
| intrinsic_test | DQN intrinsic done mask controls the actual learning reward | DQN intrinsic |
| intrinsic_test | DQN checkpoint resumes dynamic reward scaler statistics | DQN intrinsic |
| intrinsic_test | DQN checkpoint requires matching intrinsic presence and resumes scoring | DQN intrinsic |
| intrinsic_test | DQN intrinsic checkpoint resumes fractional credit and selected observations | DQN intrinsic |
| intrinsic_test | DQN intrinsic consumes warmup credit and holds the last loss | DQN intrinsic |
| intrinsic_test | DQN intrinsic validates schedule settings only when enabled | DQN intrinsic |
| intrinsic_test | DQN intrinsic learns complete lane groups and rejects oversize batches | DQN intrinsic |
| intrinsic_test | Decomposed Q-only Trace avoids RND followers and scoring | DQN intrinsic |
| intrinsic_test | Decomposed AMP updates both heads with merge and split | DQN intrinsic |
| intrinsic_test | Decomposed Agent selects fixed-zero and learner beta per Actor | DQN intrinsic |
| intrinsic_test | DQN decomposed values learn both heads through Actor and Replay | DQN intrinsic |
| intrinsic_test | DQN trace scores its Actor snapshot after synchronization | DQN intrinsic |
| intrinsic_test | Trace forwards requested intrinsic keys and Actor lifetime controls | DQN intrinsic |
| intrinsic_test | Intrinsic diagnostics honor snapshot synchronization and suspension | DQN intrinsic |
| intrinsic_test | Intrinsic diagnostics wait only for an un-cloned requested estimator | DQN intrinsic |
| intrinsic_test | Trace bonus equals the reward consumed by the real learner | DQN intrinsic |
| intrinsic_test | Interleaved Trace leaves learner rewards and saved intrinsic state unchanged | DQN intrinsic |
| intrinsic_test | Checkpoint trace scores bonus before the first learning intake | DQN intrinsic |
| intrinsic_test | Disabled learning intake advances the Trace beta schedule | DQN intrinsic |
| intrinsic_test | Disabled learning intake preserves all intrinsic scalar diagnostics | DQN intrinsic |
| intrinsic_test | Read-only nonfinite errors remain unavailable and warn once | DQN intrinsic |
| intrinsic_test | Shipped RND profiles expose readable intrinsic Trace columns only when selected | DQN intrinsic |
| nn_test | Network dot view emits head outputs and optional head info | DQN Network |
| nn_test | Agent configs read nn_viz keys, ignore old keys, and reject invalid layout | DQN Network、MuZero config |
| nn_test | Network enumerates paired Noisy parameters with their owning branches | DQN Network |
| nn_test | Existing parameter norms exclude Noisy sigma independently of its magnitude | DQN Network |
| nn_test | Noisy statistics use paired means and effective spectral weights only | DQN Network |
| nn_test | Noisy parameters clone synchronize and serialize without resetting held draws | DQN Network |
| nn_test | CUDA autocast preserves Body overrides and FP32 Head and spectral calculation | DQN Network |
| nn_test | Unreferenced forward contents remain dormant | DQN Network |
| nn_test | Network-specific Noisy executors isolate synchronized draws for all Q Heads | DQN Network |
| config_baseline_test | Actor catalog validates networks and MuZero temperature and clone contracts | DQN config、MuZero config |
| config_baseline_test | Agent Actor catalogs reject unread keys independently of prior reads | DQN config、MuZero config、ImageCls config |
| config_baseline_test | Attention Run profiles preserve existing metric definitions | DQN config |
| config_baseline_test | Attention probe batch size validates even without a subscription | DQN config、ImageCls config |
| config_baseline_test | Final feature branch is a shared learner configuration | DQN config、ImageCls config |
| config_baseline_test | PRD061 captures resolved and typed configuration | DQN config、MuZero config、ImageCls config |
| config_baseline_test | PRD061 shipped configurations resolve every Actor reference without panel overrides | DQN config、MuZero config、ImageCls config |
| config_baseline_test | PRD061 target Actor inherits final eval policy with explicit override priority | DQN config |
| config_baseline_test | PRD103 skip profiles expose greedy session and padded metric tags | DQN config |
| config_baseline_test | PRD104 defaults and auxiliary arms preserve metrics and propagate Actor limits | DQN config |
| config_baseline_test | Value decomposition ships inactive defaults and optional metrics | DQN config |
| trainer_test | Runner selects an Actor catalog entry and exposes its epsilon | DQN Trainer |
| config_test | Config reads ProfiledValueConfig root fields and phases | 基盤。テスト専用の設定名を使用 |
| config_test | Config rejects a listed ProfiledValue phase without a definition | 基盤。テスト専用の設定名を使用 |
| config_test | Config ProfiledValueConfig override can switch type and keep dormant fields | 基盤。テスト専用の設定名を使用 |
| config_test | ConfigManager resolves nested selection copied from material | 基盤。テスト専用の設定名を使用 |
| config_test | PRD101 bare profiled values record only active fields | 基盤。テスト専用の設定名を使用 |
| config_test | Schedule の設定 DTO | 基盤。テスト専用の設定名を使用 |
| metrics_logger_test | MetricsLogger writes ConfigData text file | 基盤。設定キーは StubAgent |
| GridMazeEnv_test | RunManager shares GridMaze process seed across training evaluation panels and Runs | GridMaze Env。StubAgent |
| dqn_based_actor_test | MuZero Actor catalog owns temperature and exploration noise | MuZero Actor |

検証は Debug 全体、RelWithDebInfo、Release の build と、DQN テスト単独 build、各既定テスト、GridMaze の `[prd102_runner]`、3 系統の `[prd061-capture]`、旧名・ガード・ケース数・依存グラフを通す。34 構成を変更前の保存実行体と比較する。


### ケースの照合と受入

PRD の調査時点の57件は、内的報酬34、出荷設定11、設定6、NN 3、Trainer 1、MetricsLogger 1、GridMaze 1。ステージ2直後のソースでは、内的報酬の具象Agentとの構成を32ケース確認した。残りのRND単体検証は基盤で実行する。NNは具象Agentの設定とHeadの検証に加えて、Noisy Headを共同で使う機構の検証を含めると9ケースになった。DQN分割のActorテストに含まれていたMuZeroの1ケースも系統へ戻す。この差で表は62行となる。変更前の `src` にある `*test.cpp` の `TEST_CASE` と `TEMPLATE_TEST_CASE` の宣言を数えた総数は1288。ステージ3でもこの総数を照合し、移行元ごとの検証項目を表の行き先で確認する。

| 元の確認 | 移行先での確認 |
|---|---|
| Agent Actor catalogs reject unread keys independently of prior reads | DQN Actor catalogs validate fields and network choices、ImageCls configuration validates Actor and feature branch settings、MuZero shipped and explicit configuration covers Actor and Network visualization settings の各SECTION。同じownerで未知キーが独立に落ちることを確認する |
| PRD061 shipped configurations resolve every Actor reference without panel overrides | DQN shipped profiles provide default values, actors and metrics のSECTION、ImageClsの同名ケース、MuZero shipped and explicit configuration covers Actor and Network visualization settings のSECTION。7環境のonlineとbatchrunで解決済み参照とEvalPanel用Actorを照合する |
| PRD061 captures resolved and typed configuration | 3系統の同じ隠しタグから、固定17入力のIDに対して期待する系統のDTOで採取する。ステージ2と `agent`・`typed`・`values` を照合。現用設定で解決しない入力は旧実行体のエラーと照合し、有効な入力全件の出力を分けて採取する |
| Actor catalog validates networks and MuZero temperature and clone contracts | DQN Actor catalogs validate fields and network choices で不明networkを検証。MuZeroの同じ設定ケース内で温度とcloneの契約を検証 |
| Final feature branch is a shared learner configuration、Attention probe batch size validates even without a subscription | DQN feature diagnostics validate their learner settings の各SECTIONでDQNの既定値・明示値・範囲、ImageClsの設定ケースのSECTIONでImageClsの既定値・明示値を検証 |
| Agent configs read nn_viz keys, ignore old keys, and reject invalid layout | DQN Networkの同名ケースでDQNの新旧キーと無効layout、MuZero設定ケースのSECTIONでMuZeroの新旧キーを検証 |

最初に合成DOTで禁止した系統間の辺を入れた検査を失敗させ、基盤だけへの依存では検査が通ることを確認した。Runner の公開経路では各 `Init...` を呼んだ後に従来のAgent class IDを選び、総合比較34構成で挙動を確認する。DQNテストを単独ターゲットでビルドして実行する。

- S3-1・S3-2。Debug configureで `CMAKE_EXPORT_COMPILE_COMMANDS=ON` を設定し、`cmake --preset x64-Debug --graphviz=out/build/x64-Debug/prd182-targets.dot` のDOTと `compile_commands.json` を `check_stage3_dependencies.py` に渡す。依存と基盤内部のincludeパス違反が0件。基盤の内部ヘッダ名と相対親includeを公開ヘッダ・系統・Envで検索する。
- S3-3・S3-4・S3-5・S3-9。テスト用ヘッダのガードと製品ソースのinclude、Rainbow・DefaultDQN・共通部品の参照、`InitRL` とRunnerの初期化、PRDで列挙した旧名を検索する。
- S3-6・S3-7・S3-8。移行前後のケース宣言総数と上の検証対応を照合し、基盤・3系統・Envの既定テストと隠しケースを実行する。AGENTSの規約4点、010の拡張点7行、110・120・130の参照、011の新設を確認する。


### 検証経過

- 専用worktreeのDebugビルドは終了値0。依存グラフと基盤内部includeパスの違反0件。現用の旧ヘッダ参照、テストガード違反、DQN具象間の依存違反は各0件。ケース宣言総数1288。C4の増分は許容対象163件、許容外0件。
- 隔離ビルドの既定テストは8/9実行体が通った。既存のImageCls Envの1ケースだけが失敗した。GridMazeの `[prd102_runner]` はStubAgentで通過した。
- PRD061の固定17入力を隔離環境で1件ずつ前後比較した。15件は `agent`、`typed`、`values` がすべて一致した。現用のAtari設定では `run.@nature_dqn` と `run.@classic_iqn_impala_x2` が無く、これら2件の解決エラーも前後で一致した。
- 実装計画と実装は再レビューでOK。依存の初期検査からCMakeの自己参照を除き、系統テストで公開初期化を呼ぶ検証と、現用の検証スクリプトから各系統を呼ぶ変更を追加した。診断用の凍結コピーは当初の保存名を使用する。原文は `.scratch/prd182/stage3-{plan,impl}-review-{1,2}.md`。


## ステージ4の実装

実行体の配布位置を基準に `data/` を決め、workspace、相対 Run パス、portable appdata から使う。Runner の設定と起動時の作業ディレクトリは従来の配置を使う。Viewer とツールはそれぞれの入口の固定相対パスから `data/` に届く。

リポジトリでは追跡している 8 本だけを改名する。移行ツールは隔離 fixture のみで実行し、実機の workspace と appdata の改名手順を CHANGELOG に残す。配布用 ZIP には 7 本の設定を入れ、合成 fixture と通常起動で相対パスを確認する。

### 実機移行前の検証

隔離環境で、空のdataからworkspaceを作成する経路、旧配置だけにあるworkspaceを相対指定で使わない経路、portable appdataとユーザー設定フォルダへの代替を確かめる。共通設定テンプレートはRunnerのconfigから読む。

Runnerの短いGridMaze Runを3回流し、workspace指定、設定直接指定の相対出力、設定直接指定の既定出力が、それぞれdata/workspacesのruns、data配下の指定先、data/runsに成果物を作ることを確かめる。ViewerはRunnerと同じ作業ディレクトリで起動し、既定のdata/workspacesをローカルAPIで読み、data/logsへ書くことを確認する。実画面での一覧の確認は開発者が行う。

移行ツールは合成データで改名、件数一致、同名衝突とdry-runを確認する。梱包は実ワークフローの処理を隔離fixtureで動かし、配布設定7本と、タグセット・Run・旧配置の除外を確認する。GitHub Actionsの手動確認は配布時に行う。

短いRunはすべて名前に `tmp` を含め、`apps/` にRunフォルダが増えないことも確認する。ツールは `apps/tools/resolve_workspace_test.ps1`、`optuna_workspace_test.py`、`optuna_metrics_gzip_test.py`、`inspect_run_test.py` を実行する。S4-5は、PRD §6の現用範囲を対象に固定文字列 `apps/runner/workspaces`、`apps/runner/appdata`、`apps/runner/logs` のスラッシュと逆スラッシュの両方を検索し0件を確認する。旧配置を読み込まないケースのソースはこの検索から除く。

適用前に専用worktreeで稼働中のRunが無いことを確かめる。追跡対象8本の改名に合わせ、.gitignore、用語集のworkspace、AGENTSのRun規約、利用者向けの実行・分析手順を更新する。実機ではフォルダを同じドライブ内で改名し、最後に選んだworkspaceを新Runnerで選び直す手順をCHANGELOGに記録する。ローカルの配布ZIPとAPIの確認結果を記録し、GitHub Actionsの実物ZIPとViewer実画面の確認は人の確認として残す。

変更点の入口は `anet::GetDataRootDir()` とする。最初の受入は公開のworkspace解決で、旧配置だけに合成workspaceを作り、前段のRunnerが相対指定を受けて旧配置にRunを作ることを対照で確認する。新しい実行体では `[workspace]` の `WorkspaceService::IsResolvable` と `Resolve` で同じ置き方を拒否し、明示作成時だけdata側にworkspaceができることで、期待を満たすことを確認する。

### 隔離環境の結果

- 実装計画の再レビューはOK。原文は `.scratch/prd182/stage4-plan-review-{1,2}.md`。
- 旧Runnerは合成workspaceの相対名を旧配置から解決した。新しい配置ではworkspaceテスト22ケース・122 assertionsが通った。短いGridMaze Runはworkspace指定、相対出力の直接指定、既定出力の3件が完走し、各成果物がdata配下にできた。apps配下のRun増加は0件。
- 移行ツールの5ケースは、dry-run、旧ルートのタグセットを含む改名、同名停止、衝突のない各workspaceの改名、旧ルートの作業ファイルと既存dataが同居する場合の停止を確認した。実機の改名は人が行う。
- ローカルZIPは配布設定7本を収録し、旧配置・Run・タグセットは0件。番号付きbat 3本は、空白入りの合成workspaceで準備、絶対パスの起動引数、集計先の一致を確認した。
- Viewerのjarを作り、隔離環境のAPIからdata/workspacesの合成Runが返ることと、data/logsにログができることを確認した。

- ステージ3は実装worktreeの3構成のbuildと34/34構成の比較が通った。最終の既定CTestで8/9実行体が通り、ImageCls Envの既存1件は同じ失敗だった。隔離GUI再ビルドのEval Trace行選択は既存の失敗。ダンプ比較で時刻が秒をまたいだ1件は対象ケースの再試験で通った。DQNの単独ターゲットもbuildした。
- ステージ4を反映した。ステージ3の実行比較が終わり、専用worktreeのRunnerプロセスが無いことを確認してから、追跡対象8本をGitで改名した。

- ステージ4のC3は隔離Runnerで凍結入力を共有して34/34構成が完全一致。専用worktreeとの製品ソース194本を、コメントを除くトークン列で同一と確認した。C4の差分は許容対象163件、許容外0件。旧パス0件、追跡対象8本、`data/runs` がGit管理外になることを確認した。
- Viewerの初回285件のうち、6クラスに10 failures・13 errorsが出た。C++比較終了後に対象6クラスを再試験し79/79件が通った。実装レビューで既定値の注入をapplication.propertiesに集約し、更新後の専用worktreeのjarで、隔離Viewerの実APIとログ出力も再試験した。
- 実装の再レビューはOK。原文は `.scratch/prd182/stage4-impl-review-{1,2}.md`。

### 最終確認

- ステージ1〜4の比較はそれぞれ34/34構成で一致した。ステージ4のDebug、RelWithDebInfo、Releaseのbuildと、専用worktreeの `[workspace],[app_util]` 22ケース、移行ツールの合成5ケース、隔離した道具のテスト8本が通った。最終差分の空白検査は終了値0。
- 最終CTestはDQNを含む8/9実行体が通った。ImageCls Envは、初回に一時PNGの保存失敗と既知の閉じた設定カタログの失敗が出た。新しい一時先で再試験すると18/19件が通り、既知の1件だけが残った。Runner GUIは38/39件で、既知のEval Trace行選択だけが失敗した。番号付きbatの既存検査も従来と同じ `SET EXE` の前提で止まった。
- 人が行うものは、各マシンでのworkspaceとappdataの改名とworkspaceの再選択、GitHub Actionsの実物ZIP、Viewerの実画面、C5の生成物と英訳。実機移行は[CHANGELOG](../../CHANGELOG.md#ワークスペースと-appdata-の移行prd182-ステージ4)を参照。

# PRD101：探索・学習パラメータの ProfiledValue 化 実装メモ

## 概要
2026-10-25 レビュー後の PRD101 と承認済み計画を正本とし、段階1（探索側）、段階2（学習側）の順に実装・受入検証する。PER β は既存の適用時機を維持する。qbert の成績比較は後続実験であり完了条件に含めない。

## 主な変更
### 段階1
- ProfiledValue に exponential と time_constant を追加する。root は漸近し、phase は指定長で終わる。
- root の定数は bare で読む。root .value の読み取りを消し、旧記法専用の検出・WARN・fail-fast は足さない。phase 内 .value は維持する。
- 設定読み取りと記録を分け、dump は実効 type の使用欄だけを記録する。非 constant 下の bare は保持するが評価・記録しない。
- DefaultDQN・Rainbow・skip の ε・UQE ε・τ・spatial を新キーへ移行し、Greedy と target_policy のコピー・上書きも対応する。
- SpatialLadderSchedule は定数の両端では tensor を再利用し、変動時だけ更新する。lane 割当てと1 lane の既存挙動を保つ。
- log は max(端, 1e-4) の列を指定両端へ線形伸縮する。clamp 後の両端が等しければ線形補間、両端が下限以上なら従来の生成経路を使う。
- D11 の指標を追加し、Thompson の uqe_tau は常に NaN。ε・τ・β の linear 時間評価は従来 float 演算順とし、double・整数の既存挙動は維持する。

### 段階2
- 7値を ProfiledValue 化。β 以外の6値は UpdateFromBatch の先頭で評価する。β は step 0 から開始し各更新末尾で次回抽選用に評価し、prefetch の要求順序も保つ。
- 全 optimizer group の学習率・weight_decay を更新し、σ 専用 decay は維持する。soft update には現在値を渡す。
- Actor の評価済み γ・Munchausen 値を行動結果に保持し To・WithAction でも保持する。step診断は既存 GetScalar 経路で受け取る。
- Replay Push・Sample・診断用抽選へ γ を明示的に渡し、builder・extractor に永続保持しない。MuZero は定数を渡す。
- 定数 γ は保存済み収益、schedule γ は step 報酬を既存 builder と同じ逆順の和で再計算し bootstrap と一致させる。
- prefetch は抽選済み報酬列と長さを保持し、消費側 Sample の γ で確定する。上書き後の Replay は読み直さない。TempoRL窓と診断用抽選も対応する。
- checkpoint 比較・上書き、scaler、ホライズン表示は step 0 の値。値域・有限性・spectral norm は D21。

### 移行
- 変更前の実行体・DLL・設定・4環境出力と、段階1完了時の段階2用基準を保存する。
- 現用設定・bat・テスト・PRD072 golden/比較器を移行し、sync_interval.value と設定・ImageClsテストの root .value も bare にする。
- DropMerge と LunarLander の本体 last=0 は中間laneも変わる。両方の等価性検査だけ last=1e-4。
- 日本語設計100・150・200、ガイド020、CHANGELOG、必要なwikiを追随させる。既存の用語を再利用する。en・過去記録はPRDの除外規則を適用する。
- 残存検査から CHANGELOG 対応表と phase 内 .value を除く。
- 稼働用batと実行体の入替は D25 の Run/連鎖キュー終了後に同時に行う。無関係の変更を保持し、コミット・pushしない。

## テスト
Public interface / surface: Config読み取りとdump、方策 UpdateSchedule/GetScalar、Actor/WithAction/診断、Learner UpdateFromBatch、Replay Push/Sample/診断抽選、checkpoint、runner-facing設定とmetrics。
優先 behavior / TDD順序:
1. 設定読込→方策生成→UpdateSchedule→GetScalarで指数から0への切替を1テストで RED→GREEN。
2. bare・継承・dump・同期周期、指数/phase境界、不正値、ラダー0/微小端/下限連続性/1lane/旧値一致と方策・指標を1挙動ずつ RED→GREEN。
3. optimizer全group、β初回/次回時機、soft update、checkpoint、Actor hintとstep診断。
4. γ境界前Push/後Sample、done/truncated/wrap、TempoRL、CPU/CUDA prefetchと診断抽選。定数経路と値を維持。
GREEN後だけ整理する。計画外の公開挙動が必要なら計画へ戻る。

## 検証
MSVC VsDevCmd.bat を call する cmd /s /c 経由で Debug 全体をビルドする。関連テストとcore全体テストを行う。
段階ごとに7環境の全キー・文字列値、check_default_leaves.py errors=0、4環境のchecksum一致（PRD §6.1-1だけ除外）を確認する。
短い実Runでε停止と学習率段階切替を確認する。資材は core/anet-core/testdata/prd101/、結果は .scratch/prd101/。実測から期待値を自動承認しない。
自己点検は実装品質確認観点の全項目。独立 an-review に計画・実装を渡し、差し戻し対応を終える。未実施・未決を本書へ記録する。

## 前提
- 適用するPRD blob: a39f5b8ab807f76708ec43e3686718a53b894c69（LunarLander改訂後）。開始時の版は dd3db5b2aa11a926e69917f5320b64c802f8048a。
- 設定型は現用配置の include/anet/agent.hpp、実行部品は src/dqn_based_agent.* に置く。
- 頻繁な経路に新しい同期readbackを加えない。ラダー統計はCPU端点から計算して保持する。schedule γ は必要な報酬列だけバッチへ追加し、定数構成では追加のコピー・再計算をしない。
- 通常sandboxのMXC起動失敗のため、必要なコマンドは承認付き実行を使う。
- 開始時は AnetRLRunner のローカル稼働なし。稼働用入替の確認は別途 D25 に従う。

## 経過と結果
- 2026-10-25: 計画保存。コード・設定の既存変更はなし。wiki等の既存変更は保持。- 既定葉検査の開始時不備: GridMazeEnv.seed の実験用 = が既定扱いされていた。実値・演算子を変えず、分類器に明示的な実験seedとして登録し、RED→GREENで検証する。

- 2026-10-25: ユーザー回答により LunarLander も末端0を維持する例外へ追加した。G5・D14・§6.1-1を改訂し、実装レビューとは別にPRDレビューを行う。

## 検証の実行入口

リポジトリrootのPowerShellで実行する。各段階のログは `.scratch/prd101/` へ保存する。

標準入口は `core/anet-core/testdata/prd101/verify.ps1`。未使用の `-Label` を指定し、build / test-build / related / core / learning-series / epsilon-switch / learning-rate-switch の7検査を順次実行する。buildとtest-buildはMSVC初期化経由。relatedの標準フィルターは次のとおり。

`[prd101]~[.],[profiled_value]~[.],[config]~[.],[replay_buffer]~[.],[diagnostic_no_wait],[actor_catalog]`

今回の呼出しは `powershell -NoProfile -ExecutionPolicy Bypass -File core/anet-core/testdata/prd101/verify.ps1 -Label claude-r1-fixes-final`。標準7検査のJSON・ログにこのLabelを使う。補足検査は `.scratch/prd101/claude-r1-fixes-acceptance.ps1` が続けて行い、`claude-r1-fixes-final-extra.json` に別記する。固定17入力の通常compareと直接実行は `core/anet-core/testdata/prd072/README.md` の手順を使う。

7環境の全キー・文字列値は、各段階で新しい出力先を `ANET_PRD093_OUTPUT` に設定して `anet-core-test.exe "[prd093-config-snapshot]"` で採取する。開始前の `.scratch/prd101/resolved-before/` を原値とし、PRD D16の宣言に沿った置換・型宣言・spatial両端・新指標だけを期待差分として確認する。段階2は段階1の保存値も比較基準にする。比較資材の入口・詳細は `core/anet-core/testdata/prd101/README.md` に残す。

4環境の学習系列は `run_validation.py` が各Runのexecution.jsonとcomparison.json、段階summaryを保存する。段階1はbefore各環境、段階2はstage1各環境が基準。LunarLanderは `stage1_r2_LunarLander` の検査用末端1e-4を基準に採用する。DropMergeも検査用末端1e-4。本体の末端0は変えない。短い実Runは入力・出力・境界のassert結果を同じ検証ディレクトリへ保存する。

- 段階1: 4環境の全学習scalar一致。PRD072固定17入力157assertions一致。元宣言から期待値を移行し、変更後出力をgoldenに採用していない。
- 初期の関連テスト一括実行はリンク競合回避のため途中終了。その後のr2/r3で関連と全体を再実行した。

- 段階1の7環境全キー・文字列値比較は errors=0（Atari1309、CartPole716、DropMerge1403、GridMaze763、GridMaze_muzero706、ImageCls1065、LunarLander889 keys）。`check_resolved.py resolved-stage1-final` に入口を保存。
- 短い実Runのεは指数phaseから512で0へ切替し、254点の系列assertが成功。`.scratch/prd101/short_epsilon/assertions.json`。
- 計画レビューr1の2指摘を修正し、r2はOK。原文を `.scratch/prd101/review-20261025-impl-plan-r1.md`・`r2.md` に保存。
- 学習側7値、Actorの評価値carrier、GetScalar委譲、step診断、ReplayとMuZeroの明示γ渡し、JP文書・CHANGELOG・wiki移行を実装した。
- checkpointはstep0のoptions比較・上書きを確認。σを含む4optimizer groupの学習率とdecay、βの初回と次回要求、soft update、Actor hintを検証した。`boundary-coverage.log` は3331 assertions / 20 cases成功。
- CPU/CUDA prefetch×done/truncated/通常×ring wrap×TempoRL窓を確認。抽選済みsnapshotで確定し、定数は保存収益を使う。CUDA assayのCPU転送はblockingに修正した（非同期転送直後のCPU読取りを除去）。
- 段階2の7環境全キー・文字列表記は errors=0（Atari1318、CartPole724、DropMerge1412、GridMaze772、GridMaze_muzero708、ImageCls1067、LunarLander898）。`resolved-stage2-r1-comparison.json`。
- `static-validation-summary.json`: PRD072固定17入力157 assertions、既定葉 errors=0、既定葉分類テスト16件、旧契約残存328ファイル errors=0。設定/ProfiledValue回帰で旧spectralエラー文の期待値が1件残っていたため更新し、該当テストは成功。
- `final-r1-verification.json`: MSVC初期化経由のDebug全体ビルド成功。当時の広いフィルター `[prd101]~[.],[profiled_value]~[.],[config]~[.],[dqn]~[.],[replay_buffer]~[.]`（[dqn]全体を含む）による関連437件中2件で既存テストの移行漏れを検出し修正した（Actor catalogのlinear宣言、step診断の評価済みAction値）。その後の実行結果は最終受入欄に記録した。

## 実装判断と検証の対応

- scaleの許容列挙値はagent.hppの設定共通関数をReaderと単体Scheduleから呼ぶ。両端の[0,1]はReaderでは既存metadataとProfiledValue共通検証が保証する。単体ScheduleにはmetadataなしのLadderConfigを直接渡せるため、端点metadataを補って同じ共通検証を呼ぶ境界を残す（独自の端点検証は実装しない）。
- `MunchausenConfig::Resolve` は、`app_util.hpp` の `WorkspaceService::Resolve`、`env.hpp` の `WorkerThreadResolver::Resolve` / `BatchEnvBuilderConfig::ResolveWorkerThreads`、`metrics_logger.hpp` の `detail::ResolveVideoCodec` と同じく「宣言から実行時の実効値を確定する」意味で使う。時間の評価自体はProfiledValue::Evaluateが担い、Resolveは評価済みの数値と宣言flagからMunchausenValuesを確定する。

- `MunchausenConfig` はprofile宣言、`MunchausenValues` は評価済みの数値とflagを持つ。Actor hint・学習の数値計算へprofileを持ち込まず、毎stepの再構築を避けるために分けた。`ActorQHintValues`も同じ境界で使う。
- `SpatialLadderSchedule` は既存DQN実行部品と同じ `dqn_based_agent.*` に配置し、時間更新・tensor再利用・統計をまとめる。`prd101_ladder_test.cpp` は独立したラダー境界のfixtureとして分けた。`prd101_muzero_test.cpp` は既存のMuZero replay専用テストがないため、定数γ明示渡しのテストだけを独立させた。公開ヘッダは新設しない。
- Replayのscheduled収益はDQNのunroll_steps=0を対象とし、構成時に検証する。MuZeroは専用bufferに定数γを明示的に渡し、従来の保存値と計算順を保持する。
- 定数lr/decayは既存のSetup/Loadだけで適用し、変動するときだけ全groupを書き換える。σgroupの識別を初期化時に保持する。定数γは報酬snapshotのコピーと再計算を行わない。
- UIのstep診断はActionが保持するimmutableな評価値を読む。Agentの学習指標GetScalarは既存の学習worker側の収集経路へ委譲し、UIへ新しいlearnerロック待ちやdevice同期転送を追加しない。

| 優先behavior | RED→GREENと追加境界の証跡（`.scratch/prd101/`） |
| --- | --- |
| 設定→方策→指数phase停止→指標 | `red-test.log` / `green-test.log`、`short_epsilon/assertions.json` |
| bare・dump・継承・同期周期・指数 | `bare-red-test.log` / `bare-green-test.log`、`schedule_test.cpp` と `config_test.cpp` |
| ラダー0/微小/1lane/既存値/方策指標 | `spatial-red-test.log` / `spatial-green-test.log`、追加の下限通過・tensor再利用。指定の3 lane下限通過は `claude-r1-fixes-ladder-red.log` → `claude-r1-fixes-ladder-green.log`、4 laneの0.4→5e-5は `claude-r1-fixes-boundaries.log`（該当テスト成功）と最終再検証へ記録。 |
| lr/decay全group・β・soft update | `learner` / `optimizer` / `beta` / `soft-tau` の `red.log` / `green.log`、`boundary-coverage.log` |
| γ明示APIと学習時機・Munchausen | `gamma-red-build.log`（新APIのcompile RED）/ `gamma-green.log`、`gamma-learner` / `munchausen` の `red.log` / `green.log` |
| Actor値の引継ぎと診断・MuZero | `actor-values` / `step-diagnostics` の `red.log` / `green.log`、`muzero-red-build.log` / `muzero-green.log` |
| spectral norm全端点・Agent指標 | `spectral-profile` / `agent-scalars` の `red.log` / `green.log` |
| γ done/truncated/wrap・TempoRL・CPU/CUDA prefetch・診断抽選・定数経路 | `boundary-coverage.log` とReplayの `[prd101]` テスト |

compile REDのうちfixtureの構文誤りはbehaviorのREDに数えない。期待値は元の数式・宣言から構成し、新実装の出力からgoldenを自動更新しない。

## 最終受入と残る運用

Claude側r1指摘対応とユーザー追加の設定並べ替え後の最新受入。ログは `.scratch/prd101/`。関連の標準フィルターは `[prd101]~[.],[profiled_value]~[.],[config]~[.],[replay_buffer]~[.],[diagnostic_no_wait],[actor_catalog]`。

| 標準検査名（verify.ps1） | 終了コード | 秒 | ログ |
| --- | --- | --- | --- |
| build | 0 | 142.6 | `claude-r1-fixes-final-build.log` |
| test-build | 0 | 115.3 | `claude-r1-fixes-final-test-build.log` |
| related | 0 | 63.4 | `claude-r1-fixes-final-related.log` |
| core | 0 | 598.5 | `claude-r1-fixes-final-core.log` |
| learning-series | 0 | 619.1 | `claude-r1-fixes-final-learning-series.log` |
| epsilon-switch | 0 | 45.6 | `claude-r1-fixes-final-epsilon-switch.log` |
| learning-rate-switch | 0 | 26.6 | `claude-r1-fixes-final-learning-rate-switch.log` |

- 関連271 cases / 21076 assertions成功。core全体806 cases（805 passed / 1 skipped）、167465 assertions成功。skipはCUDA搭載時に省略するPRD070の非搭載環境用診断（init_test.cpp:145）。追加したラダー2場面はCPU/CUDAで成功。3 laneの最初の中間値はD8の倍精度計算で約5.000375e-5となる。
- 4環境の学習系列は全75116点が厳密一致。GridMaze18514、LunarLander15967、Atari17915、DropMerge22720点。各 `claude-r1-fixes-final_<Env>/comparison.json` と `stage2-summary.json` に保存。LunarLander/DropMergeの比較だけlast=1e-4、本体はlast=0。更新前summaryは `claude-r1-fixes-final-previous-stage2-summary.json` へ保持した。
- Atariのεは512 exp_stepで0、GridMaze学習率は512で0.001→0.0001へ切替。`claude-r1-fixes-final-epsilon/assertions.json` / `claude-r1-fixes-final-learning-rate/assertions.json`。
- 並べ替え後の7環境全キー・文字列表記はerrors=0（Atari1318, CartPole724, DropMerge1412, GridMaze772, GridMaze_muzero708, ImageCls1067, LunarLander898）。PRD072固定17入力はREADMEのprepare→親環境設定→直接実行と通常compareの双方で157 assertions成功。既定葉は `Assignments: 29099, audited: 3629, errors: 0`。分類器16件、旧契約329ファイルも成功。

| 補足検査名（別JSON） | 終了コード | ログ |
| --- | --- | --- |
| prd072-prepare | 0 | `claude-r1-fixes-final-prd072-prepare.log` |
| prd072-direct | 0 | `claude-r1-fixes-final-prd072-direct.log` |
| prd072-compare | 0 | `claude-r1-fixes-final-prd072-compare.log` |
| config-snapshot | 0 | `claude-r1-fixes-final-config-snapshot.log` |
| resolved-comparison | 0 | `claude-r1-fixes-final-resolved-comparison.log` |
| default-leaves | 0 | `claude-r1-fixes-final-default-leaves.log` |
| default-leaves-tests | 0 | `claude-r1-fixes-final-default-leaves-tests.log` |
| contract | 0 | `claude-r1-fixes-final-contract.log` |

標準7検査は `claude-r1-fixes-final-verification.json`、補足8検査は `claude-r1-fixes-final-extra.json`。BOM・改行・差分とPRD保持、設定行内容保持は `claude-r1-fixes-hygiene.json`。自己点検全66項目は `claude-r1-fixes-self-check.md`。設定の追加並べ替えは5ファイル11組で、行内容・値・演算子・コメントを保持。GridMaze系列Run開始と並べ替えの前後関係は末尾の追加依頼節に記録した。

D25の稼働環境切替とqbert成績実験は範囲外。今回の検証の未実施・仕様の未決なし。2026-10-25のユーザー報告により、Claude側の継続再レビューはOK。コミット・pushは行っていない。

### 以前の受入履歴

以下のfinal-r2/final-r3はClaude側r1指摘対応前の履歴。final-r2-related.logとfinal-r3-related.logの関連268件のフィルターは `[prd101]~[.],[profiled_value]~[.],[config]~[.],[replay_buffer]~[.],[diagnostic_no_wait],[actor_catalog]`。`final-r3-verification.json` の6検査（build / test-build / related / atari-epsilon / learning-rate / core）は補足実行であり、標準verify.ps1の7検査を実行した証跡とは区別する。最新受入結果は本節冒頭と末尾の指摘対応表に記録した。

- ラダー単体構築の値域は、負端・1超・負phaseの3条件でRED→GREENを確認した。tensor再利用と下限通過の連続性も成功。`ladder-bound-red.log` → `final-r2-related.log`。
- 自己点検は `docs/quality/impl.md` 全66項目を実施し、`.scratch/prd101/self-check.md` に保存。`final-hygiene.json` はBOM・LF/CRLF・CP932・C++機械観点・git diff --checkが全て合格。旧契約329ファイル errors=0。
- 段階1・2とも4環境の全学習系列が厳密一致。各段階の比較件数はGridMaze18514、LunarLander15967、Atari17915、DropMerge22720点（計75116点）。`stage1-summary.json` / `stage2-summary.json`。本体のLunarLander/DropMergeはlast=0、比較だけlast=1e-4。
- 最終の設定境界修正後も7環境の全キー・文字列表記比較は errors=0。`final-r3-config-snapshot.log` / `final-r3-resolved-comparison.log`。既定葉は29052 assignments / 3629 audited / errors=0（`final-r3-default-leaves.log`）。PRD072固定17入力157 assertions成功（`prd072-final-refactor.log`）。
- 最終Debug全体ビルドと明示的testターゲットビルドが成功。`final-r3-build.log` / `final-r3-test-build.log`。関連268 cases / 21032 assertions成功（`final-r3-related.log`）。この最終実行体のcore全体は803 cases、802 passed / 1 skipped、167421 assertions全成功（`final-r3-core.log`、589.8秒）。skipはCUDA利用可能時に非搭載環境用の診断テストを省略したもの。`final-r3-verification.json` の6検証（全体ビルド・testビルド・関連・Atari ε・学習率・core）は全て終了コード0。
- 最終実行体の短い実Run: Atariの指数εは512 exp_stepで0へ切替、254点のassertが成功（`final-r3-atari-epsilon/assertions.json`、before32点/after190点）。GridMazeの学習率は512で0.001→0.0001へ切替、111点のassertが成功（`final-r3-learning-rate/assertions.json`、before15点/after95点）。GridMazeでの初期ε確認とは区別する。

## 実装レビューの対応

r1の原文は `.scratch/prd101/review-20261025-0250-impl.md`（同内容 `review-impl-r1.md`）。4群をまとめて対応した。

| 指摘 | 対応と検証 |
| --- | --- |
| R1: Rainbowのspatial.enabled無視 | 公開CreateActor→MakeAction→GetScalarでon/offと動的端点を検証。`rainbow-spatial-red.log` は8 assertions中4失敗。生成時の引数1か所をcfgの値へ直し、`final-r3-related.log` で8 assertions成功。 |
| R2: Atariでのε停止実Runなし | `short_run.py`のεをAtariへ変更し、環境・切替step・境界前後の実測列をassertions.jsonへ保存。上記最終Atari Run成功。学習率はGridMazeを維持。 |
| R3: 設計150/200の追随漏れ | 一様probe・履歴probeの署名へgammaを追加。hard/soft共通の有限性・[0,1]と、SN soft時の全active端点[0,0.1]（root constant1だけ例外）を明記。 |
| R4: 設定3行のprofile境界 | @random/@baseline後を:へ変更。値と演算子を保持。7環境全キー文字列表記と既定葉の再検査が成功。 |

再現用 `verify.ps1` は全体ビルドに加えtestターゲットも明示ビルドし、失敗時に古いexeを検証しない。PowerShell構文検査 errors=0。

独立レビューr2はOK。R1〜R4充足・追加指摘なし、最終coreと全受入証跡の照合を完了。原文は `.scratch/prd101/review-20261025-0311-impl.md`（同内容 `review-impl-r2.md`）に保存した。この時点のCodex側の計画・PRD・実装レビューは全てOKだった。後続のClaude側独立レビューr1は別枠として末尾に記録する。

## 未実施・未決

- 仕様の未決なし。
- D25のサブ機キュー・メイン機の隔離実行体へは配布していない。稼働環境へのbatと実行体の同時入替は、対象Run/連鎖キューの終了後に行う運用作業。現地のキュー状態はこの作業から確認していない。
- qbert成績比較はPRDの除外した後続実験。実装受入の不足ではない。
- Claude側独立レビューr1の指摘対応と再検証は完了。2026-10-25にユーザーからClaude側の継続再レビューOKの報告を受けた。D25の稼働環境切替と後続実験は別途運用作業。
- コミット・pushは実施していない。無関係の未コミット変更は保持した。

## 独立レビュー（Claude側）r1：承認済み指摘対応計画

2026-10-25のレビュー原文は `.scratch/prd101/review-20261025-0412-impl.md`。依頼は `.scratch/prd101/codex-request-20261025-impl-review-fixes.md`。先のCodex側r1/r2とは別のレビューであり、以下を実施してClaude側の継続再レビューへ渡す。PRDを変更せず、GridMaze.txtはユーザーの意図したコメント・見出し移動として保持する。

- P1: 関連テストを `[prd101]~[.],[profiled_value]~[.],[config]~[.],[replay_buffer]~[.],[diagnostic_no_wait],[actor_catalog]` に統一。verify.ps1の7検査名（build / test-build / related / core / learning-series / epsilon-switch / learning-rate-switch）で再実行する。旧437件は[dqn]全体を含む別範囲、旧6検査は補足実行として残す。
- I1: D8の倍精度独立計算を期待値に、3 laneのfirst=1.0001e-4→1e-4→5e-5・last=0を追加してREDを確認する。下限未満の端を含む経路だけexpm1の正規化重みと両端の加重和にし、CPU/CUDAでGREENを確認する。その後4 laneの0.4→5e-5を追加する。端点・中間値・連続性をfloat丸め程度で検証し、既存の両端が下限以上の経路・clamp同値の線形補間・定数tensor再利用を保つ。最初の中間値は約5.000375e-5であり、レビューの5e-5は丸め表記。
- I3: scaleの検証をagent.hppの設定共通関数に集約しReaderと単体Scheduleから呼ぶ。DefaultDQNとActionPolicyの重複を消す。metadataなしの単体構築には既存ProfiledValue共通検証による[0,1]保証を残す。不正scale・負/1超/不正phase端点とエラー内容を確認する。
- I2/I6: PRD072 READMEをprofiled-<hash>出力と必須ANET_PRD072_ROOTへ追随させる。compareの子環境設定とprepare.ROOTを使う直接実行手順を区別し、C++/Python双方で名前つき定数にする。
- I4/I7: 報酬snapshotの確保・転記・Tensor化、discount、診断scalar、定数/schedule分岐とsequence保持に日本語の意図コメントを加える。MunchausenConfig::Resolveは既存のWorkspace、worker数、video codecと同じ「宣言から実行時の実効値を確定する」意味で維持し、根拠を記録する。
- 検証: MSVC Debug全体・testターゲット・関連・core全体、4環境学習系列、Atari ε停止、GridMaze学習率切替、PRD072固定17入力とREADME直接実行、7環境全キー/文字列表記、既定葉、旧契約、BOM/改行/差分を再実行する。期待値は実測から自動更新しない。LL/DropMergeの等価性検査だけlast=1e-4。
- 記録: 8件別の対応表・新しいログ/フィルター/件数/終了コードと自己点検を本書へ残す。範囲外3件、インデント整理、二重FinalizeReturns最適化は今回変更しない。Claude側へ再レビュー資料を渡し、コミット・pushはしない。

実施状況: 数値境界のRED→GREEN、P1/I1〜I7、追加の設定並べ替え、標準7検査と補足8検査を完了。2026-10-25のユーザー報告により、Claude側の継続再レビューはOK。開始時ファイルとPRD/GridMazeのhashは `.scratch/prd101/claude-r1-fixes-before/manifest.json` に保存した。

### r1対応中の検出と範囲

- 3 laneの下限通過はCPU/CUDAで4 assertionsのREDを確認し、expm1式で関連5ケース32 assertionsがGREEN。最初のテスト型指定のコンパイル誤りは挙動のREDに数えていない。
- 不正値メッセージの追加検査では、metadata付き端点をProfiledValueが先に検証して論理キーが出ない4 assertionsのREDを検出した。単体Scheduleの既存端点検証をProfiledValue構築前へ移した。検証ルール自体は既存共通処理を使う。ヘルパー移動時の名前空間修飾のコンパイル誤りも挙動のREDとは区別する。
- 任意の範囲外3件（float linear全体に対する計画の説明範囲、設計200のτ検証の説明範囲、既存IQN riskテスト名）は、今回承認されたP1/I1〜I7の修正範囲を保つため変更しない。PRDの変更は不要で、動作の追加修正も要求されていない。
- rainbow_agent.cpp / rl.cppのインデント整理は指摘対応と無関係なので変更しない。prefetchの二重FinalizeReturnsは現契約を満たしており、性能最適化を混ぜないため維持する。
- I5はユーザー判断を優先し、GridMazeの意図したコメント・見出し移動を保持した。追加依頼前のhash一致を確認後、最新依頼による記載順変更だけを適用した。全行内容の保持と並べ替え後の7環境比較errors=0、GridMaze学習率Run成功を確認した。

### r1の8件別対応表

| ID | 修正箇所・対応 | 検証・証跡（`.scratch/prd101/`） |
| --- | --- | --- |
| P1 | verify.ps1のrelatedフィルターを標準へ統一。検証入口と旧437/268件、標準7検査/旧補足6検査の区別を本書へ記録。 | `claude-r1-fixes-final-verification.json` の標準7検査が全て終了コード0。関連271件21076 assertions、core806件167465 assertions。 |
| I1 | prd101_ladder_test.cppに指定の3 lane/4 laneを追加。SpatialLadderSchedule::Updateの下限未満だけexpm1と正規化重みで安定化。両端が下限以上・同clamp端・1 lane・定数cacheは維持。 | `claude-r1-fixes-ladder-red.log`（CPU/CUDAで4失敗）→`claude-r1-fixes-ladder-green.log`。指定2場面と追加検証は`claude-r1-fixes-boundaries-r3.log`（7ケース56 assertions成功）。指定テスト名は `PRD101 spatial first crosses the log floor without cancellation` / `PRD101 spatial log ladder preserves a tiny last endpoint`。最終独立参照式は全演算doubleとし、標準related/coreでCPU/CUDAとも再検証成功。許容誤差はRED時から変更なし。 |
| I2 | PRD072 READMEの生成先をprofiled-<hash>へ更新。compareの子環境設定とprepare.ROOTを使う親PowerShellへの設定を明記。 | READMEどおりのprepareは成功（`claude-r1-fixes-prd072-prepare.log`）。最終direct/compareも各157 assertions成功（`claude-r1-fixes-final-prd072-direct.log` / `claude-r1-fixes-final-prd072-compare.log`）。 |
| I3 | agent.hppのValidateSpatialScaleへ許容値検証を集約。Reader/単体Scheduleから呼び、DefaultDQN/ActionPolicyの重複を削除。単体端点検証は構築前へ移し論理キーを保証。 | `PRD101 spatial validation reports keys values and expected ranges` がReader/単体の計10 assertions成功。metadataなしの既存3境界も成功。`claude-r1-fixes-boundaries.log`のメッセージRED→`claude-r1-fixes-boundaries-r3.log`のGREEN。 |
| I4 | DefaultSampleExtractor::ExtractSamples、DefaultReplayBuffer::ProcessQueue、DqnAgentStepDiagnosticsProvider::Observeの指定された各段に日本語の意図コメント。 | 差分を読み、snapshot/discount/診断scalar/定数分岐/sequence保持の理由を確認。標準core成功、4環境の75116点の学習系列も厳密一致。 |
| I5 | ユーザー意図のコメント/見出し移動を保持。最新の追加依頼により設定の記載順だけは変更し、全行内容を保持する。 | `config-order-report.json`でGridMazeを含む5設定の行内容一致。最終7環境全キー/文字列比較errors=0、並べ替え後のGridMaze学習率Runも成功。 |
| I6 | ANET_PRD072_ROOTをC++ kPrd072RootEnv/Python PRD072_ROOT_ENVに定数化し、READMEに同じ識別子であることを明記。 | 最終PRD072直接実行/compareの双方で157 assertions成功。 |
| I7 | MunchausenConfig::Resolveは維持。「実装判断と検証の対応」にWorkspace・worker数・video codecの既存同義のResolveを列挙。 | 既存3ファイルと意味を照合。「宣言から実行時の実効値を確定する」で一致。 |

### ユーザー追加依頼：設定の記載順

2026-10-25、start/endとspatial first/lastが離れて読みにくいという追加依頼に従い、Atari / DropMerge / GridMaze / LunarLander / agent.txtの11組を、時間側とspatial側それぞれのまとまりへ並べ替えた。値・キー・演算子・コメントを含む行の多重集合は前後で完全一致する。変更前を `.scratch/prd101/config-order-before/`、検査結果を `config-order-report.json` に保存。I5のユーザー意図のコメント・見出し移動は保持するが、最新依頼による記載順変更だけはGridMazeにも適用する。PRDは変更しない。標準検証のGridMaze学習系列は並べ替えの直前に開始していたため、cpp修正後・旧記載順の設定を使う。後続の3環境、Atari ε停止、GridMaze学習率切替、補足7環境比較は並べ替え後の設定を使う。追加依頼の内容等価性は行内容の完全一致と、並べ替え後の7環境全キー・文字列値比較で検証する。

### Claude側の継続再レビュー完了

2026-10-25、ユーザーから再レビューOKの報告を受けた。同じメッセージで許可されたとおり、DefaultSampleExtractor::ExtractSamplesのprefetch収益再確定コメントの参照をD19からD20へ訂正した。実行コード・設定・PRDの変更はない。差分、BOM/LF、訂正前後の実行コード不変を確認した。コメント参照と記録だけの変更のためビルド・テストは再実行せず、直前の標準7検査・補足8検査の成功記録を維持する。全66項目の自己点検について今回の差分を照合し、新しい未対応事項なし。コミット・pushなし。

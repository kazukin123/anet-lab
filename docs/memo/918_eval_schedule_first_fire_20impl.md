# PRD918 eval スケジュールの発火の列 実装メモ

## 概要

PRD918 の D1〜D10、§5、A1〜A8 を実装する。`run.eval_schedule.[tag]` に `offset`（既定 0）と `warmup_count`（既定 1）を加える。`IntervalGate` を変えず、`EpisodeEvalObserver` で offset 到達前の問い合わせを止め、到達後は offset を引いて問い合わせ、先頭の指定回数の発火では評価を走らせない。

## 主な変更

- `core/anet-core/include/anet/observers.hpp`、`core/anet-core/src/observers.cpp`: observer のコンストラクタを `(eval_runner, eval_interval, eval_offset, eval_warmup_count, use_background, wait_on_exit)` に移す。offset と残る warmup 発火回数を保持する。背景の例外回収は発火判定より前のままにする。抑止した発火でもゲートを進め、抑止ごとのログは足さない。
- `core/anet-core/src/trainer.cpp`: `interval` と同じ `int` の読み口で新しい値を読み、Env 生成前に負数を拒否する。休止でも検証する。scheduled の行を D6 の順序にして observer へ渡す。
- `core/anet-core/src/{observers_test,episode_end_test,trainer_test,config_baseline_test}.cpp`: 発火列、設定拒否、休止、scheduled の出力、設定の解決結果を検証する。observer を直接作る既存 13 か所は offset 0・warmup_count 0 を明示する。
- `apps/runner/src/RunnerApp_test.cpp`: hidden の `[runner_close_drain]` に warmup_count 0 を明示し、学習イベントが 1 回だけの fixture でも評価セッションの排水を確認する。
- `apps/runner/config/{Atari,LunarLander,GridMaze,DropMerge}.txt`、`apps/{11,12,13}_batch_run.bat`: PRD §5.3 の値へ移す。12 の bat の既存の変更は保持する。bat は CP932・CRLF のまま扱う。
- `core/anet-core/testdata/prd084/{run_forward_refactor.ps1,run_review_validation.ps1,README.md}`、`prd094/run_validation.py`、`prd097/{run_validation.py,audit_validation.py}`、`prd103/run_current.py`、`prd104/run_experiments.py`: PRD §5.4 の対象タグに warmup_count 0 を足す。比較資材で offset を継承して学習直後の発火がずれるタグは、D9 と比較維持の目的に従い offset 0 も指定する。
- `docs/design/{010_framework_overview,020_user_guide_run,100_runtime_and_configuration,140_observability,200_dqn_agents}.jp.md` と対応する英語版の関連部分、`CHANGELOG.md`: 新しい発火列、既定、評価専用 Run の指定、Learner の warmup との違いを反映する。英語版を変える前に翻訳元情報の規則を読み、確認した範囲だけを記録する。
- `docs/wiki/mechanisms/{evaluation,diagnostic_cost}.md`、関連する目録・実装の地図・log: 実装結果と正本に合わせる。
- `docs/management/issues/issue-20261103-a86dcc.md`: 計画保存時に対応中、レビュー前に対応と検証結果を追記、完了前に残件無しと判断日を記録する。解決にはしない。派生する一覧は標準の道具で再生成する。
- `core/anet-core/testdata/prd918/README.md`: PRD 固有の検証コマンドと証跡の場所をまとめ、140 のテストの節から参照する。検証の補助スクリプトが必要なら同じディレクトリに置く。

## テスト

- Public interface / surface: `EpisodeEvalObserver::OnLearn` とセッション完了通知、`RunManager(ConfigData)`、`ConfigManager` の設定解決、Runner の batchrun と起動・セッションログ。
- 最初の behavior: interval 10 の observer に learn step 0〜20 を送り、既定では 10・20 にだけ評価セッションが始まることを確認する。既存のテスト用 Env・Agent、実際の EvalRunner、SessionEndObserver を使い、production API にテスト用の取り出し口は足さない。
- TDD 順序: (1) 既定の抑止を 1 テストで RED→GREEN。(2) offset と warmup_count 0・複数回を RED→GREEN。(3) interval 1000、offset 0・500・2500、warmup_count 0・1・3、刻み 1・7・1500 の行列、初回 2700、抑止回数過多を確認する。(4) 設定の負数拒否・休止・interval 必須・起動ログを各 behavior ごとに RED→GREEN。(5) 出荷設定の解決結果を RED→GREEN で移す。GREEN のあとだけ整理する。
- `EvalRunner` が final であるため、PRD A2 の fake runner は既存のテスト用 Env と実際の runner で代替し、公開通知で開始した learn step を確かめる。
- 既存の排水・背景例外のテストを、新しいコンストラクタへ移して維持する。hidden の `[runner_close_drain]` は既定で確認し、必要な場合だけ warmup_count 0 を指定する。

## 検証

- 基準コミット: `670f6d75ebb129207324cd29b38f9c2205931889`。着手時の未コミット変更を保持する。PRD918 の計画は本書が最初である。
- 実装前の Debug ビルドと関連テスト、既定スイートを記録し、PRD093 の attention の既知の失敗があれば前後で照合する。
- ビルドは `AGENTS.md` の Windows/MSVC 手順で VsDevCmd を call してから `cmake --build --preset x64-Debug` を実行する。関連対象は `anet-core-test`、`AnetRLRunner`、`AnetRLRunner-gui-test`。
- `[interval_gate]`、新しい `[prd918]`、`[eval_schedule]`、`[eval_session]`、`[eval_runner]`、設定検査、既定のテストスイート、hidden の `[runner_close_drain]` を実行する。
- bat の CP932・CRLF と、§5.3 の各変数の値を検査する。新しい検証資材の構文と関連する解決値の監査を確認する。
- Runner 起動前に Run 実行ユーザーガイドの指定の節を読む。LunarLander の既定を 300000 exp step まで実行し、予算完走、scheduled の値、初回セッションの下限、Greedy とほかのタグの発火時点の不一致を確認する（A6）。
- PRD A8 の 3 検索、util.hpp の差分が空であること、品質確認観点の全項目、BOM・LF・bat の CP932 を確認する。ログは `.scratch/prd918/` に保存し、応答へは要約を出す。
- 自己点検のあと、計画と実装を独立した `anet-review-agent` にまとめて依頼する。指摘を仕分け、直せるものを直し、同じエージェントへ再依頼する。結果の原文と本書の経過を保存する。

## 実行した検証コマンドと結果

以下はリポジトリルートの PowerShell で実行した履歴である。長いビルド・スイート・Run は exec_command の実行セッションで開始し、標準出力・標準エラーを `> ... 2>&1` でログへ保存した。write_stdin で終了を待ち、終了コードとログ末尾を照合した。途中はログ更新、CPU 時間、Run の進行を必要なときにまとめて確認した。ログ全文は応答へ繰り返していない。ログ名の基点は `.scratch/prd918/`。

### 実装前の Debug ビルド

終了コード 0。baseline-build.log。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test > .scratch\prd918\baseline-build.log 2>&1'
```

### Debug 全体・GUI ビルド

両方終了コード 0。debug-build.log、gui-build.log。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug > .scratch\prd918\debug-build.log 2>&1 && cmake --build --preset x64-Debug --target AnetRLRunner-gui-test > .scratch\prd918\gui-build.log 2>&1'
```

### GUI fixture 修正後のビルド

終了コード 0。gui-warmup-build.log。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target AnetRLRunner-gui-test > .scratch\prd918\gui-warmup-build.log 2>&1'
```

### テスト識別子の定数化後のビルド

終了コード 0。refactor-test-build.log。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test > .scratch\prd918\refactor-test-build.log 2>&1'
```

### A6 用の最適化ビルド

終了コード 0。optimized-build.log。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-RelWithDebInfo --target AnetRLRunner > .scratch\prd918\optimized-build.log 2>&1'
```

### 実装前の既定スイート

追加の環境変数なし。終了コード 0、885 成功・1 skip、173853 assertions 成功。baseline-tests.log。

```bash
& core\anet-core\bin\Debug\anet-core-test.exe > .scratch\prd918\baseline-tests.log 2>&1
```

### 変更後の既定スイート

終了コード 0、888 成功・1 skip、173908 assertions 成功。final-tests.log。このあと行ったテスト識別子の定数化は、次の関連検査で再確認した。

```bash
$env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; & core\anet-core\bin\Debug\anet-core-test.exe > .scratch\prd918\final-tests.log 2>&1
```

### 整理後の関連検査

OMP/MKL の上書きなし。終了コード 0、38 件・564 assertions 成功。refactor-tests.log。

```bash
$env:ANET_PRD918_OUTPUT = Join-Path (Get-Location) '.scratch/prd918/config-check'; & core\anet-core\bin\Debug\anet-core-test.exe '[prd918],[prd918-config],[eval_schedule],[interval_gate],[eval_session],[eval_runner]' > .scratch\prd918\refactor-tests.log 2>&1
```

### 出荷規則

終了コード 0、12 件・2604 assertions 成功。shipped-rules.log。まとめて走らせた初回の oneDNN assertion は green-shipped.log に残した。

```bash
$env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; & core\anet-core\bin\Debug\anet-core-test.exe '[shipped_config]' > .scratch\prd918\shipped-rules.log 2>&1
```

### 旧 hidden 検査との同時実行

終了コード 42。2 件中 PRD918 は成功、PRD103 は指標不在の 4 assertions で失敗。schedule の断定は成功。green-config-only.log。

```bash
$env:ANET_PRD918_OUTPUT = Join-Path (Get-Location) '.scratch/prd918/config-check'; $env:ANET_PRD103_OUTPUT = Join-Path (Get-Location) '.scratch/prd918/prd103-config'; & core\anet-core\bin\Debug\anet-core-test.exe '[prd918-config],[prd103-config]' > .scratch\prd918\green-config-only.log 2>&1
```

### GUI 排水

終了コード 0。GUI fixture が標準ストリームを Run 側へ移すため、このログは filter・seed の行までである。親プロセスで終了コード 0 を確認した。runner-close-drain-green.log。

```bash
& apps\runner\bin\Debug\AnetRLRunner-gui-test.exe '[runner_close_drain]' > .scratch\prd918\runner-close-drain-green.log 2>&1
```

### 変更前の全キー採取

終了コード 0、15 assertions 成功。config-before.log と config-before/。

```bash
$env:ANET_PRD093_OUTPUT = Join-Path (Get-Location) '.scratch/prd918/config-before'; & core\anet-core\bin\Debug\anet-core-test.exe '[prd093-config-snapshot]' > .scratch\prd918\config-before.log 2>&1
```

### 変更後の全キー採取

終了コード 0、15 assertions 成功。config-after.log と config-after/。

```bash
$env:ANET_PRD093_OUTPUT = Join-Path (Get-Location) '.scratch/prd918/config-after'; & core\anet-core\bin\Debug\anet-core-test.exe '[prd093-config-snapshot]' > .scratch\prd918\config-after.log 2>&1
```

### 全キー比較

終了コード 0。7 環境とも eval_schedule 以外の差なし。config-diff.json。

```bash
.\.venv\Scripts\python.exe core\anet-core\testdata\prd918\validate.py config-diff
```

### bat と A1・A8 の静的受入

終了コード 0。CP932・CRLF、各変数の葉、共通ゲート無変更、旧記述の検索を確認。

```bash
.\.venv\Scripts\python.exe core\anet-core\testdata\prd918\validate.py static
```

### 既定葉検査

終了コード 0。32106 代入・4007 監査・エラー 0。default-leaves.log。

```bash
.\.venv\Scripts\python.exe core\anet-core\testdata\prd072\check_default_leaves.py > .scratch\prd918\default-leaves.log 2>&1
```

### A6 の完走した試行

終了コード 0。最終 300032 exp step。optimized-run-validation.log と RelWithDebInfo/short-run.json。補助スクリプトは Runner の stdout/stderr を RelWithDebInfo/short-run-console.log にまとめ、Run 自身の logger は Run 内へ保存。subprocess.run で終了を待ち、上限 3600 秒、予算と発火は終了後に断定する。

```bash
.\.venv\Scripts\python.exe core\anet-core\testdata\prd918\validate.py short-run --configuration RelWithDebInfo > .scratch\prd918\optimized-run-validation.log 2>&1
```

### 最初の Debug 試行（受入には使わない）

最適化した同一予算の試行が成功したあと、重複処理として停止。非 0 終了は short-run-validation.log に記録。Run run_20261110-223848_prd918_ll は途中試行として保持。この起動時の上限は 1200 秒で、後続の版では 3600 秒へ改めた。

```bash
.\.venv\Scripts\python.exe core\anet-core\testdata\prd918\validate.py short-run > .scratch\prd918\short-run-validation.log 2>&1
```

Python 資材の構文・BOM・LF/CRLF・リンク・日英の翻訳元情報は `.scratch/prd918/check_files.py`、差分の語と空白は `check_diff.py` で確認した。PowerShell 2 本は System.Management.Automation.Language.Parser.ParseFile で構文エラー 0。レビューでも既存 39 ファイルの BOM 保持と新規 3 ファイル、改行適合を確認した。文字列候補は設定キーの例外だけで、直訳語と空白エラーは是正後 0 件。

### 2 モデルレビューの修正後

Debug 全体ビルドは終了コード 0（cc-fix-build.log）。テストの定数参照だけを変更し、production コード・設定・挙動は変えていない。標準出力・標準エラーをログへ保存し、exec_command のセッションを write_stdin で終了まで待った。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug > .scratch\prd918\cc-fix-build.log 2>&1'
```

関連検査は終了コード 0、38 件・564 assertions 成功（cc-fix-tests.log）。旧 PRD103 hidden の指標名の不整合は今回も修正対象に含めない。

```bash
$env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; $env:ANET_PRD918_OUTPUT = Join-Path (Get-Location) '.scratch/prd918/cc-fix-config'; & core\anet-core\bin\Debug\anet-core-test.exe '[prd918],[prd918-config],[eval_schedule],[interval_gate],[eval_session],[eval_runner]' > .scratch\prd918\cc-fix-tests.log 2>&1
```

ファイル検査・差分検査はともに終了コード 0（cc-fix-file-check.log、cc-fix-diff-check.log）。42 ファイルの BOM・改行・リンク・Python 構文・翻訳元情報に問題なし。差分の文字列候補は設定キーの例外で、変更した PRD103/918 の環境名の比較と skip プロファイルの直書きを別途読み直して解消を確認した。追加差分について実装品質確認観点の全項目を自己点検した。

```bash
$env:PYTHONUTF8 = '1'; .\.venv\Scripts\python.exe .scratch\prd918\check_files.py > .scratch\prd918\cc-fix-file-check.log 2>&1; .\.venv\Scripts\python.exe .scratch\prd918\check_diff.py > .scratch\prd918\cc-fix-diff-check.log 2>&1
```

## 前提

- ユーザー判断が必要なブロッカーは 0。PRD の暫定の印は外れており、未決もない。D1〜D10 と §5.3 の値を追加判断なしで採用する。
- 新しい 2 キーの型は interval と同じ。独自の上限や丸め、互換 overload、旧挙動専用の分岐、抑止ごとの診断は足さない。
- warmup_count は発火回数であり、飛ばした bucket の数ではない。残数がある間だけ減らし、全発火を累積する整数の桁あふれを避ける。
- 既存の背景セッションの待ち方、排水、乱数の共有、メトリクス用 IntervalGate は変えない。
- ADR 0070 と ADR 0028 の注記は PRD の工程で作成済みなので、重ねて起こさない。CONTEXT.md の用語定義にもキー一覧を足さない。
- 過去の Run・実験記録、PRD §5.4 の触らない検証資材は保持する。同じ間隔の組全般への offset、背景待機方式、学習軌跡の新旧一致は範囲外。

## 決めたこと

- PRD の発火列と移行表を正本にし、共通ゲートを変更せず observer の状態だけで実装する。
- 観測可能な通知と設定解決を検証し、テストのための API 拡張や新しいスケジュール抽象化を作らない。

## 経過

- 2026-11-10 PRD、関連するコード・テスト・設定・CONTEXT・ADR・wiki を照合し、未決事項監査を終えた。実装計画の観点と文書の検証・報告の規約を読んだ。
- 実装前の Debug ビルド成功（`baseline-build.log`）。既定スイートは 886 件中 885 成功・1 skip、173853 assertions 成功（`baseline-tests.log`）。PRD093 attention の既知の失敗は再現しなかった。
- 7 環境の全キー・文字列値を既存 hidden `[prd093-config-snapshot]` で採取した（`.scratch/prd918/config-before/`）。
- 最初の発火テストは旧実装で 0・10・20 を観測して RED（`red-default.log`）、先頭 1 回抑止の最小実装で 10・20 となり GREEN（`green-default.log`）。
- 日英 5 組の設計文書は、英語版が持つ翻訳元 blob と変更前の日本語版がすべて一致することを確認した。今回の差を両言語へ反映し、翻訳元情報を更新した。
- offset・抑止回数の行列と既存の評価・排水・背景例外のテストは成功（26 件・440 assertions）。設定読み込み・休止・ログを加えた関連検査は成功（37 件・526 assertions）。
- 出荷設定の通常検査は OMP/MKL 各 1 スレッドで成功（12 件・2604 assertions）。まとめて走らせた最初の試行は oneDNN 内部の `nthr_ == nthr` assertion で中断したため、設定解決の hidden 検査とはプロセスを分けた。
- PRD918 の hidden 設定検査は成功。旧 `[prd103-config]` は、新旧で触っていない `metrics.scalar.[50_agent_skip/19_primary_clip_ratio_ema]` の不在で 4 環境とも失敗する（`green-config-only.log`）。この指標の定義は HEAD の metrics_scalar.txt にもなく、本 PRD の発火列とは別の既存検査の不整合である。
- `[runner_close_drain]` の最初の実行は失敗した。fixture の Learner は学習イベントを 1 回しか出さないため、既定の抑止では probe に届かない。PRD §5.4 の条件に従い、このテストだけ warmup_count 0 を明示して再確認する。
- Debug 全体・GUI ターゲットのビルドは成功。変更後の既定スイートは 889 件中 888 成功・1 skip、173908 assertions 成功（`final-tests.log`）。GUI の排水テストも warmup_count 0 で成功した。
- 自己点検後の識別子の定数化を経た関連検査は 38 件・564 assertions 成功（`refactor-tests.log`）。7 環境の全キー比較は eval_schedule の差だけで、既定葉検査は 32106 代入・4007 監査・エラー 0。
- A6 は実行中。現行既定は replay batch 256、刻み 8 で、PRD の概算（1024・刻み 2）より更新が多い。Debug 実行は約 220 exp step/s のため、同じ既定・同じ 300000 exp step の受入を最適化ビルド（RelWithDebInfo）でも実行する。最初の Debug 試行の記録は保持し、完走した試行だけを受入とする。
- A6 は RelWithDebInfo で成功。正常終了コード 0、最後の TrainEvent の exp step 299776 と完了した batch 256 から最終 300032 を確認した。eval_target・eval の最初は learn step 200、Greedy は 1056 で、その後も 2056〜9056 と他タグの発火座標とは一致しなかった。証跡: `.scratch/prd918/RelWithDebInfo/short-run.json`、`optimized-run-validation.log`、Run `run_20261110-224902_prd918_ll`。重複する Debug 試行は専用 config と PID を照合し、受入成功後に停止した。途中の Debug 試行は受入の根拠に使わない。
- 独立レビュー 1 往復目: 計画は OK、実装は検証コマンド・必要な環境変数・ログと結果の対応の記録不足で差し戻し。成功した生ログと A1〜A8 は確認された。原文は `.scratch/prd918/review-20261110-2258.md`。
- 指摘への対応として「実行した検証コマンドと結果」に 18 件の実行履歴、環境変数、stdout/stderr の保存、待機・終了確認の方法を追記した。Run 名の転記誤りも証跡と照合して訂正した。親の見直しで 200 の learner.enabled と warmup_count の既定を説明する文の順序を日英で整え、翻訳元 blob を更新した。コード・設定・テストは前回の依頼以降変えていない。
- 独立レビュー 2 往復目: 計画・実装とも OK、不足・修正要求・開発者確認はなし。原文は `.scratch/prd918/review-20261110-2305.md`。A6 は完走した RelWithDebInfo の試行を根拠にし、途中停止した Debug 試行を含めないことも確認された。
- 対象課題 issue-20261103-a86dcc の対応内容を確定し、ai_judge を残件無し、ai_judged を 2026-11-10 とした。status は対応中のまま、解決・コミットは行っていない。
- 2026-11-10 工程末の新規起票候補を開発者が選び、旧 hidden 検査の不整合を [issue-20261110-546bdb](../management/issues/issue-20261110-546bdb.md) として記録した。

- 2026-11-10 2 モデルレビュー: Codex は計画・実装とも OK、Claude は計画 1 件・実装 2 件で差し戻し。計画の変更一覧に GUI テストを追記し、変更した PRD103/918 テストの環境名と skip プロファイルの参照を定数へ揃え、wiki の履歴へ日付を付けて末尾へ移した。原文は `.scratch/prd918/review-20261110-2314-impl.md` と `review-20261110-2314-impl-claude.md`。Claude の費用は約 2.2072 USD。
- 2 モデルの再レビュー: Codex・Claude とも計画・実装が OK、不足・修正要求・開発者確認なし。原文は `.scratch/prd918/review-20261110-2324-impl.md` と `review-20261110-2324-impl-claude.md`。Claude の再レビューのログの概算費用は 3.6589719 USD、終了コード 0。Claude はスクリプトを実行できず、Codex と親の実行記録で Debug 全体ビルド、関連 38 件・564 assertions、BOM・改行等を確認した。

## 決まらなかったこと

- 範囲外の検出（Claude）: PRD918 §2.1 と A6 の説明は replay batch 1024・刻み 2 を前提とし、現用の 256・刻み 8 と食い違う。A6 の受入閾値には影響せず、実測と差は経過に記録済み。入力元 PRD の記述の整理は未対応。
- 範囲外の検出（Claude）: `docs/wiki/mechanisms/diagnostic_cost.md` の既存行に、arm を直訳した避ける語が残る。今回の追加行ではなく、未対応。
- [issue-20261110-546bdb](../management/issues/issue-20261110-546bdb.md): 旧 hidden `[prd103-config]` が現用定義と異なる `19_primary_clip_ratio_ema` を要求し、4環境で失敗する。開発者の選択で新規起票済み。PRD918 の残件には含めない。

## 完了報告

- 実施内容: 既定の先頭 1 回抑止と offset、Env 生成前の検証、scheduled のログを実装した。評価専用 Run と比較資材の初回評価を明示し、出荷設定・bat・日英 5 組の文書・wiki を移行した。共通 IntervalGate は変更していない。
- 変更ファイル: 「主な変更」の範囲。新規は本書と prd918 の README・検証スクリプト。既存のユーザー差分を保持した。
- 検証: Debug 全体・GUI と最適化 Runner のビルド成功。既定スイート 888 成功・1 skip、整理後の関連検査 38 件成功、出荷規則 12 件成功、GUI 排水成功。7 環境の全キー比較、既定葉、文字コード・改行・リンク・翻訳元情報の確認も成功。A6 は 300032 exp step まで正常完走した。
- レビュー結果: 独立レビューの実行条件の記録不足と、続く 2 モデルレビューの 3 件を修正した。再レビューは Codex・Claude とも計画・実装が OK。A1〜A8 の確認を受け、開発者の確認事項はなし。
- 決まらなかったこと: 実装の未決はなし。範囲外の旧 hidden 検査の不整合は issue-20261110-546bdb として起票した。Claude が検出した入力元 PRD の刻みの概算と現用設定の差、wiki の既存行の避ける語は「決まらなかったこと」に未対応として記録した。
- 判断してほしいこと: なし。

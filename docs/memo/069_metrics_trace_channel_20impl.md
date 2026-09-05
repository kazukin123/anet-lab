# PRD069：メトリクス trace チャネル実装計画

実装・受入検証完了（2026-09-05）。未解決の比較差異は0件。

## 概要

`069_metrics_trace_channel_10prd.md` の D1〜D16 と受入条件を実装する。eval の採用エピソードごとの値を trace に保存し、既存 scalar のセッション集約を維持する。ユーザー承認済み計画の正本。未決ブロッカーは0件。

## 実装変更

- SessionEndEvent、Observer、RunnerScoped wrapper、Notifier の登録・解除・通知を追加。EvalSessionEnv::LastAdoptedGroups() は直前 Step の採用完了 group を昇順で返し、次の Step と Reset で消去する。EvalRunner は Step 直後に採用 episode のみ通知し、return 集約確定後に SessionEnd を1回通知。SHARED は group 0 を env_index=-1 へ変換。
- eval scalar と RunManager の ENV 購読を @session_end へ移行。PRD §4.5 の検証行列と旧指定への置換案内付きエラーを実装。他の scalar パース、集約、EMA、interval を維持。
- LogTrace(tag, step, lane, data) を JSONL backend 直書きで追加。共通 episode observer が callback 内で宣言順に GetScalar(key, env_index) を呼ぶ。未知キーは tag/key/lane/target 付き fail-fast、非有限値はキーを残して null。行ごとの side file や個体値蓄積は作らない。
- 共通トークン分類を純粋 helper に切り出し、指定順と出現情報を保持。trace は event/target 明示、1個以上の裸キーを要求し、重複・未知・禁止指定を後勝ち処理前に拒否する。TraceMetricDef と JSON 変換を追加。attach 済み定義だけをチャネル別に記録し、同名 scalar/trace tag を許可。Agent 購読ヒントは scalar のみ。
- scalar 定義を metrics.scalar.defs、trace 定義を metrics.trace.defs とする。inspect_run の master/cache は新名優先・新名不在時だけ旧 metrics.defs を読む。session_end の設定導出も対応。
- 現用 eval scalar 設定、名指しされた Git 管理外 atari-live/config/atari_base.txt、テスト、現行設計文書、AGENTS.md を同期。Atari に eval trace 2宣言とコメントアウトした train 例を追加。

## TDD と回帰検証

各 behavior を1テスト → RED確認 → 最小実装 → GREENの順で進め、GREEN 後だけ refactor する。

1. EvalRunner::RunSession() から Notifier まで通し、採用 N 件の EpisodeEnd と最後の SessionEnd 1件、既存集約値を検証。
2. N>G、N<G、N=1、SHARED、同時完了、非採用完了、Reset、次 Step で消える確定値の callback 内取得を検証。
3. 設定から observer と実際の JSONL 行まで検証。DSL 全セル、別表記の重複、禁止指定の上書き、宣言順、非有限値、未知キー、train 共通経路、同名 tag を検証。
4. dormant eval の定義除外、定義ミラー、新旧 reader、cache 未構築時の selector と tags --no-observed を検証。background は制御したテストで配送・行数・observer 例外の再送出を確認。
5. Metrics Viewer の既存 ingest 統合テストに trace/scalar 同名 tag を追加し、READY と scalar 利用を確認。

検証は AGENTS.md の VsDevCmd.bat 経由 Debug ビルド、関連 C++ テスト、anet-core-test.exe 全体、.venv の inspect_run_test.py、Maven の MetricsIngestorIntegrationTest。既知失敗・未実施は根拠付きで記録。

## 前後 Run と受入条件

- コード・設定編集前に現行ソースから Release をビルドして baseline 採取。Breakout、run.@evalN10、app.$=app.batchrun、seed=1、backend.@deterministic、eval1/eval2 の use_background=false を固定。
- 初期予算は前後とも 2.5M exp step。各 eval 最低3セッション完了を実測。未達なら延長して同条件の比較を取り直す。Run 名は PRD 指定に従い tmp を含め、別出力先と実効設定・コマンド・終了情報を記録。
- 設定差はイベント移行・trace追加・出力先に限定。時間依存 source key の除外一覧を明記し、他 scalar の件数・step・値を比較。差異未解決、baseline 不在は検証不足。
- 変更後は各 session/各 eval trace tag が10行、同一 step、lane 0〜9各1回。score 平均と既存 scalar が float 丸め範囲内で一致。score×len 同時分布、p10、score >=432 率を記録。
- 変更後 Run を Viewer と inspect_run runs/tags に通し、既存 scalar の利用を確認。

## 固定した前提

- コード・現用設定・文書が一つの完成単位。イベント分離だけの独立導入はしない。
- 互換読み取りは明示された過去 Run の定義名に限定。過去 artifact は変更しない。
- trace 専用 reader、追加イベント、episode_id、model_version、DropMerge 宣言、TensorBoard 対応は範囲外。
- 無関係な未コミット変更を保持。staging/commit/push は人間が実施。


## 実装・検証記録（2026-09-05）

### 編集前 baseline

コード・設定編集前に HEAD `9ca5d616d238c845423d9605b5c0ef1bcc3dbf37` の Release 全体をビルドし、exit 0 を確認した。実行体は `apps/runner/bin/Release/AnetRLRunner-prd069-before.exe` に保持した（SHA256 `24EA55A672598FA97984A4B6A5B80AEE8CCF45EDE2622FE17F435986AB3A0D29`）。

- Run: `out/test-tmp/prd069/runs/run_20260905-154712_tmp_prd069_before`
- 条件: `before.txt` に保存した Breakout / evalN10 / batchrun / seed=1 / deterministic backend / foreground eval / 2,500,000 exp step。
- 終了: exit 0。eval1 / eval2 はそれぞれ9セッション、scalar は107 tag。
- 実効設定: `before.resolved.txt`。標準出力・標準エラー・終了情報・binary hash を同ディレクトリへ保存。
- 編集前 reader 回帰: inspect_run 53テスト成功、MetricsIngestorIntegrationTest 22テスト成功。

### TDD の証跡

証跡の保存先は `out/test-tmp/prd069/`。段階ごとに次の失敗と成功を確認した。

| 段階 | RED | GREEN |
|---|---|---|
| 採用 episode→session 通知 | SessionEndObserver / Event が未定義（red1.build.log） | 1ケース・14 assertions |
| eval scalar の session 束縛 | GetSessionEndObservers が未定義（red2.build.log） | 14ケース・231 assertions |
| trace 宣言→実 JSONL | trace observer が0件（red3） | 15ケース・249 assertions |
| DSL 検証行列・重複・禁止指定 | 89 assertions が失敗（red4） | 20ケース・433 assertions |
| チャネル別定義とミラー | metrics.scalar.defs のミラー不在（red5） | 21ケース・458 assertions |

GREEN 後、採用 group の寿命、N>G / N<G / N=1 / SHARED、foreground / background の実 JSONL、通知順序、同名 tag、宣言順取得、NaN/Inf の null 化、background observer 例外、dormant のみの場合の定義除外を回帰テストへ追加した。

Release の途中で `observers.cpp.obj` に LNK1163（COMDAT）が1回発生した。該当生成 object のみを削除して再ビルドし解消した。ソース変更による回避はしていない。ビルドログは `green2.build.log` と `green2.retry.build.log`。

Python の最終 reader 回帰は57テスト成功（`inspect-test-final.log`）。Java の同名 tag 統合テスト追加後は23テスト成功（`viewer-test.log`）。SQLite loader の古い一時 DLL を削除できない診断はテスト失敗を伴わない。

### 実装範囲

イベント分離、scalar の session 束縛、共通 token 分類、trace observer / JSONL / 定義、RunManager の購読・attach・定義出力、inspect_run の新旧定義名と session_end 導出を実装した。現用設定、名指しされた Git 管理外 atari-live 設定、現行設計文書、AGENTS.md を同期した。

作業中に見えた `apps/11_batch_run.bat` の変更は本作業では編集していない。既存の未追跡文書も保持し、staging / commit / push は実施しない。


### 比較で検出した既存 scalar の未初期化

最初の変更後試行 `run_20260905-162919_tmp_prd069_after` の途中比較で、`40_agent_rs/02_clip_ratio` だけが baseline と異なった。baseline は0、試行は `3.730604447582664e-08`。他の scalar は、その時点までの全出力 prefix が一致した。

原因は `ConstantRewardScaler::last_clip_ratio_` が初期化されず、クリッピング無効時の `Scale()` でも書き込まれない既存不具合。型・constructor・Scale・GetScalar の経路を確認し、公開 RewardScalerFactory 経由の回帰テストで初回と Scale 後の不定値を再現した（`scaler.red.test.log`）。同ログのスケール値 assertion は fixture の auto post scale 指定漏れも含み、fixture 側を明示的な非 auto 設定へ直した。

この試行は停止し、Run artifact をそのまま保持した（`after.uninitialized-trial.txt`、`after.uninitialized.exit`）。初回 Scale 前は NaN、非クリップ Scale 後は実際のクリップ率0とする最小修正を追加した。報酬 Tensor の計算は変更しない。時間依存メトリクス以外の除外を増やさず、元の編集前 baseline との厳密比較を取り直す。

PRD069 関連の Debug 回帰は、この追加修正前の時点で26テスト・594 assertions が成功（`debug.related.test.log`）。Debug 全体ビルドも成功（`debug.build.log`）。


### 最終ビルド・回帰

- Debug 全体ビルド: exit 0（`debug.final.build.log`）。
- Release 全体ビルド: exit 0（`release.final.build.log`）。
- Debug 関連テスト: `[scaler],[trace],[eval_session],[observer_factory],[metrics_defs],[episode_end]`、27ケース・598 assertions 成功（`debug.final.related.log`）。
- Debug `anet-core-test.exe` 全体: seed `2386836949`、exit 0、552ケース中550成功・2期待済み失敗、5850 assertions 中5848成功・2期待済み失敗（`debug.full.test.log` / `.exit`）。
- 期待済み失敗は既存 `[!shouldfail]` の `ReplayBuffer n-step returns stop at episode_start without done` と `ReplayBuffer frame stacking starts a new stack at episode_start without done`。該当テスト・production は変更していない。
- Python: `.\.venv\Scripts\python.exe viewers/metrics-tools/inspect_run_test.py`、57テスト成功。
- Java: `apps/metrics-viewer` で `mvn -q -Dtest=MetricsIngestorIntegrationTest test`、23テスト成功。

ビルドは両構成とも次の MSVC 初期化経由で実行した（preset を Debug / Release に切替）。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
.\core\anet-core\bin\Debug\anet-core-test.exe
```

最終変更後 Run は `run_20260905-164207_tmp_prd069_after`。起動は `apps/runner` を working directory とし、`bin/Release/AnetRLRunner.exe --config C:/dev/anet-lab/out/test-tmp/prd069/after.txt`。標準出力・標準エラーを保存して終了コードを回収する。


### 最終 Run 受入結果

前後とも 2,500,000 exp step の終了条件で exit 0。eval1 / eval2 は前後とも各9セッション完了した。実効設定差はイベント移行60項目、trace追加2項目、Run名1項目のみで、想定外差分は0件（`config-diff.json` / `config-validation.json`）。最終 Release SHA256 は `B13E9B6873F3430BDC39F04FFCC0A2ECB97042E4D15F034DF16CA6AA7A53DCD0`。

- scalar は前後とも107観測 tag・413,799行。全107 tag の件数が一致。
- 時間依存キー `train_step_per_sec` / `exp_step_per_sec` / `elapse_hour` を値比較から除外。この Run の該当 tag は `90_perf/12_exp_step_per_sec`、`90_perf/22_exp_step_per_sec_ema`、`90_perf/90_elapse_hour`。
- 残る104 tag・412,299行の `(step, value)` が厳密一致。許容差による scalar 差異の吸収はしていない。
- trace は eval1 / eval2 各90行、合計180行。全18セッションで各10行、同一 step、lane 0〜9が各1回。
- 全セッションの score 平均が同 step の既存 scalar と float 丸め範囲内で一致（比較基準 rel=1e-6 / abs=1e-5）。
- p10 は線形補間、閾値率は `game_score >= 432`。score×len の全180組は [比較結果 JSON](../../out/test-tmp/prd069/comparison.json) の `trace_sessions[].score_len` に保存。

| eval | exp step | score平均 | p10 | score ≥432率 |
|---|---:|---:|---:|---:|
| 51_eval1 | 200320 | 2.2 | 0 | 0% |
| 51_eval1 | 456320 | 1.7 | 0 | 0% |
| 51_eval1 | 712320 | 1 | 0 | 0% |
| 51_eval1 | 968320 | 1 | 0 | 0% |
| 51_eval1 | 1224320 | 1.2 | 0 | 0% |
| 51_eval1 | 1480320 | 0.3 | 0 | 0% |
| 51_eval1 | 1736320 | 0.1 | 0 | 0% |
| 51_eval1 | 1992320 | 2.3 | 0 | 0% |
| 51_eval1 | 2248320 | 0.4 | 0 | 0% |
| 52_eval2 | 200320 | 0.7 | 0 | 0% |
| 52_eval2 | 458624 | 0.3 | 0 | 0% |
| 52_eval2 | 716928 | 1.3 | 0 | 0% |
| 52_eval2 | 975232 | 0.8 | 0 | 0% |
| 52_eval2 | 1233536 | 0.8 | 0 | 0% |
| 52_eval2 | 1491840 | 3 | 0 | 0% |
| 52_eval2 | 1750144 | 2.2 | 0 | 0% |
| 52_eval2 | 2008448 | 2.2 | 0 | 0% |
| 52_eval2 | 2266752 | 0.4 | 0 | 0% |

### 実 Run の reader 検証

最終 Run を Viewer の production `MetricsIngestor` に通し、`READY scalars=413799 traces=180 quarantined_tags=0` を確認した（`viewer-actual.log`）。補助実行コードは `IngestSmoke.java`、classpath は `viewer-classpath.txt` に保存した。

SQLite loader が既存の古い一時 DLL を削除できない `AccessDeniedException` を ERROR レベルで出力し、JDK が native access の注意を出した。いずれも Run ingest の ERROR 状態ではなく、処理は exit 0 / READY で完了した。既存の一時 DLL は削除していない。

`inspect_run.py runs` と `tags` は master / cache の両方で exit 0、warnings空、stderr空。`tags` は両経路とも `def_source=metrics_defs`、128定義・107観測 tag。未観測定義も列挙する既存契約を維持し、trace tag を scalar tag に混ぜていない（`reader-validation.json`、`inspect-*-master.json`、`inspect-*-cache.json`）。

```powershell
.\.venv\Scripts\python.exe out/test-tmp/prd069/compare.py
.\.venv\Scripts\python.exe viewers/metrics-tools/inspect_run.py runs out/test-tmp/prd069/runs/run_20260905-164207_tmp_prd069_after --format json
.\.venv\Scripts\python.exe viewers/metrics-tools/inspect_run.py tags out/test-tmp/prd069/runs/run_20260905-164207_tmp_prd069_after --format json
```

現用 eval scalar の旧イベント指定は移行済み。残る旧表記は拒否テスト、trace、座標系説明、過去資料のもの。`git diff --check` は成功。計画で指定したビルド・回帰・前後 Run・reader の未実施項目はなく、期待済み失敗と環境依存診断は上記に区別して記録した。

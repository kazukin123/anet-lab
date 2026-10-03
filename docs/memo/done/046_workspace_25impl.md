# Workspace 機構 PH3 Optuna 統合 実装メモ

## 概要

Optuna ハーネスを `runs_optuna` 固定方式から workspace 自己完結方式へクリーンブレークする。`--workspace` から `<workspace>/runs` と `<workspace>/optuna` を導出し、全入力の preflight が成功するまで DB、artifact、Run、harness.log を生成しない。Dashboard launcher と正本ドキュメントも同じ契約へ移行する。

## 主な変更

- `optuna_common.py` に副作用のない workspace／SQLite URI／artifact path 解決と、source・target の存在／型／包含検証を追加する。workspace 入力の trim、`#`、`//`、末尾 `;`、UNC 拒否、および `resolve(strict=False)` 相当の component 単位 containment で sibling-prefix、`..`、symlink／junction escape、wrong-bucket を拒否する。
- dry-run／run-trial／run-study は `--workspace`（既定 `_default`）から runs dir を `<workspace>/runs` に固定し、`--runs-dir` と placeholder 展開を削除する。`--storage`／`--optuna-artifact-dir` の override は `<workspace>/optuna` 配下だけを許可する。
- cleanup-running と summarize-study に workspace 導出既定値を追加する。明示 source だけで完結する呼び出しは無関係な workspace config を検証せず、summarize-study の省略 target は対応する source を継承する。
- command ごとに「path 解決 → 全 source／target 検証 → workspace/config 検証 → target 作成」を分離する。run 系の artifact store は trial 採番や DB 接続より先に必須初期化し、失敗時は WARN 継続せず fail-fast する。harness.log は run-trial／run-study だけ `<workspace>/optuna/harness.log` に生成する。
- trial context、採番走査、manifest、Study User Attributes、`00_last_run_study_args` を workspace 契約へ更新する。`last_runs_dir` を廃止して `last_workspace` を保存し、再開 args から `--runs-dir` を除去して `--workspace` を含める。
- DropMerge の生成 config は共通 main、workspace config、extra config、trial override の順に include／出力する。Runner へ `--workspace` は渡さず、`--config` の完全自己記述モードを維持する。
- `23_optuna_dashboard.bat` は workspace 引数必須、bat 位置基準の相対解決、禁止 path 拒否、workspace／optuna.db／artifacts の既存必須へ変更し、不足時に何も生成しない。既に削除済みの `22_metrics_viewer_java_optuna.bat` は復活させない。
- `docs/design/optuna.md`、application／analysis の関連ページ、標準検証手順を workspace layout、手動分割 migration、旧 study 再開時の clean-break 引数へ更新する。`CONTEXT.md` と ADR 0021/0022 は変更しない。

## テスト

- Public interface / surface: `dropmerge_optuna.py` の subcommand parser／`main()`、生成 config・manifest・Study attrs、filesystem の生成有無、`23_optuna_dashboard.bat` の終了値と診断。
- 優先 behavior: workspace dry-run が `<workspace>/runs` に workspace include 付き config を生成する経路を tracer bullet とする。続いて `--runs-dir` 未知引数、workspace 不在／禁止 path、storage URI 等価性、包含／wrong-bucket、cleanup／summarize の source-only、target 継承と型違反、run 系 artifact fail-fast、harness.log の生成境界、Dashboard の無引数／不在／別 cwd 解決を縦スライスで追加する。
- TDD 順序: `apps/runner/tools/optuna_workspace_test.py` と launcher test に behavior を 1 件ずつ追加し、各 RED を確認してから最小実装で GREEN にする。private helper の構造ではなく CLI 終了値、出力 JSON／config、生成された／されない path を検証する。

## 検証

```powershell
.\.venv\Scripts\python.exe apps\runner\tools\optuna_workspace_test.py
.\.venv\Scripts\python.exe apps\runner\tools\optuna_metrics_gzip_test.py
powershell -NoProfile -ExecutionPolicy Bypass -File apps\runner\tools\resolve_workspace_test.ps1
.\.venv\Scripts\python.exe apps\runner\tools\dropmerge_optuna.py dry-run --workspace _default --study-name workspacePh3Smoke --trial-name t00000 --budget small
git diff --check
```

## 前提

- 対象は PRD046 PH3 のみとし、PH1／PH2 の現行 workspace 契約を変更しない。
- `summarize` は明示された metrics ファイルだけを扱うため workspace 対象外のまま維持する。
- 既存 `runs_optuna` の自動 migration は実装せず、Run／DB／artifact／WAL／SHM は PRD §5 の静止条件で手動移動する。
- 無関係な dirty ファイルと既存 `20impl`〜`24impl` は変更せず、stage、commit、push は行わない。

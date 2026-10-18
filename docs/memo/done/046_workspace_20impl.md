# Workspace 機構 PH1 実装メモ

## 概要

PRD046 の PH1 として、Runner を常時 workspace 対応にし、config、Run、履歴、補助ツールを選択中 workspace へ統合する。workspace 入力は外側空白を trim して採用し、`#`、`//`、末尾 `;`、UNC root を拒否する。PH2 Metrics Viewer、PH3 Optuna、既存 Run の物理移動は今回実装しない。

## 主な変更

- `ConfigData` に後勝ち merge、Properties 文字列化、replace-existing 保存を追加し、`ConfigManager` の合成順を main → 導出値 → workspace config → CLI → AutoMerge → CLI にする。
- `GetAppDataDir()` を追加し、`apps/runner/appdata/` の存在で portable mode、未存在なら OS user-data directory を使う。
- Runner に `--workspace` / `--select-workspace`、workspace 解決、MRU、選好、選択ダイアログ、`_default` 生成を追加する。`--config` は workspace 選択・履歴・`app.runs_dir` 導出を行わない完全自己記述モードにする。
- workspace 確定時に解決済み絶対パスを plain UTF-8 の `GetAppDataDir()/last_workspace.txt` に保存する。補助 launcher は第1引数、`last_workspace.txt`、`_default` の順で workspace を選ぶ。
- workspace モードでは最終的な `app.runs_dir` が導出文字列と完全一致することを検証する。
- 現在 dirty な `_main.txt` の DropMerge 選択を `_workspace_template.txt` へそのまま移し、共通 `_main.txt` から env block を除く。`common.txt` から `app.runs_dir` を除く。
- DOT/MP4/TensorBoard/MLflow launcher を workspace 対応にする。MLflow DB は合意済み例外として `<workspace>/runs/mlflow.db` に置き、新 ADR に理由を記録する。
- PRD046、Runner/config/MLflow/launcher の現行 design docs、`apps/README.md` を新契約へ更新する。`CONTEXT.md` は編集しない。

## テスト

- Public interface / surface: `ConfigData`、`ConfigManagerOptions`、`GetAppDataDir()`、Runner workspace resolver、Runner CLI/config 合成、補助 launcher、MLflow bridge。
- 優先 behavior: Properties round-trip と2回目保存、workspace overlay の後勝ち、portable/user app-data、相対/絶対 workspace、MRU/prefs/last workspace、CLI競合、`--config` bypass、`app.runs_dir`直接/間接改変拒否、launcher fallback、MLflow DB path。
- TDD 順序: tracer bullet となる1テストを RED にし、最小実装で GREEN にしてから次の behavior へ進む。refactor は GREEN 後だけ行う。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
core\anet-core\bin\Debug\anet-core-test.exe
apps\runner\bin\Debug\AnetRLRunner-test.exe
.\.venv\Scripts\python.exe viewers\metrics-tools\mlflow_bridge_test.py
ctest --preset x64-Debug --output-on-failure
```

## 前提

- レビュー裁定として、D21、ADR 0022、`last_workspace.txt`、`<workspace>/runs/mlflow.db` は実装前に明示承認済みであり維持する。
- dirty な `_main.txt` から移した新規 workspace の既定 env は DropMerge とする。
- PH1〜PH3 は同一クリーンブレークの途中であり、PH1 単独ではリリースしない。
- Viewer/Optuna の workspace 移行は後続フェーズで行う。
- 既存 Run、Optuna DB、MLflow DB の自動 migration は追加しない。
- 無関係な dirty ファイルは変更せず、stage、commit、push は行わない。

# Workspace 基盤 core app_util 移管 実装メモ

## 概要

workspace の非 GUI 機能を `core/anet-core` の `app_util` module へ移し、Runner GUI、将来の CLI application、test executable が同じ interface を利用できるようにする。`WorkspaceDialog.hpp/.cpp` は維持し、`Workspace.hpp/.cpp` と workspace 専用 Runner test target は廃止する。

## 主な変更

- `WorkspacePaths`、`WorkspaceService`、workspace config 合成と `app.runs_dir` 検証を `anet/app_util.hpp` と `app_util.cpp` へ移し、namespace を `anet` にする。
- config、workspace、強制 workspace 選択の競合判定を `AppConfigMode` / `DetermineAppConfigMode()` として core interface に置く。
- MRU と `last_workspace.txt` は core workspace 基盤が所有し、GUI 固有の `workspace.dialog_skip` は `WorkspaceDialog` module が所有する。
- Runner を core interface へ切り替え、`Workspace.hpp/.cpp`、`Workspace_test.cpp`、空になる `AnetRLRunner-test` target を削除する。
- `app_util.hpp` の namespace 内を 4 space 字下げへ統一し、設計資料と標準テスト手順を新しい所有関係へ更新する。
- 補助 launcher は手作業で整理した workspace も参照できるよう、既存 root と `runs/` だけを要求し、`config/_main.txt` を解決条件にしない。
- `apps` 配下の batch file は `cmd.exe` の既定環境で日本語コメントを誤解析しないよう、Shift_JIS（CP932）・CRLF で保存し、resolver test で検査する。
- workspace resolver とテストはユーザー操作用launcher面から分離し、`apps/runner/tools/resolve_workspace.bat` と隣接する `resolve_workspace_test.ps1` に置く。
- `41_mlflow_bridge.bat` はversion確認でMLflow本体を二重importせず、package metadataで検査してから起動待ちをINFO表示する。

## テスト

- Public interface / surface: `anet::WorkspaceService`、`WorkspacePaths`、`AppConfigMode`、`CreateWorkspaceConfigManager()`、`ValidateWorkspaceRunsDir()`。
- 優先 behavior: 相対 workspace の作成・解決を tracer bullet とし、path 検証、補完可能性、UTF-8、MRU、履歴、config 合成、起動 source 競合を既存テストの観測結果のまま core test へ移す。
- TDD 順序: 移管前 `AnetRLRunner-test` を GREEN baseline とし、public interface と 1 behavior を core へ移して GREEN、残りの既存 behavior を順に移し、最後に Runner 側の参照と test target を削除する。

## 検証

```powershell
core\anet-core\bin\Debug\anet-core-test.exe "[workspace]"
core\anet-core\bin\Debug\anet-core-test.exe "[app_util]"
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test AnetRLRunner -j1'
powershell -NoProfile -ExecutionPolicy Bypass -File apps\runner\tools\resolve_workspace_test.ps1
git diff --check
```

## 前提

- workspace directory、template、履歴、`last_workspace.txt`、config 合成順の契約は変更しない。
- core 化は現在の filesystem と `ConfigManager` 依存をそのまま移し、将来用 adapter や追加設定 interface は導入しない。
- `WorkspaceDialog.hpp/.cpp` 以外の新しい Runner workspace ファイルは作らず、既存 `20impl` / `21impl` は変更しない。

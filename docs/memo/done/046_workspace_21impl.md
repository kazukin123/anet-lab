# 不完全な既存 workspace の選択・自動初期化 実装メモ

## 概要

`apps/runner/workspaces/` 直下の全ディレクトリを選択肢へ表示する。Runner が選択・明示指定した既存ディレクトリに `config/_main.txt` が無い場合は、追跡対象の `_workspace_template.txt` をコピーして初期化し、既存の Run や他のファイルを保持したまま起動する。

## 主な変更

- `ScanLocalWorkspaces()` は `config/_main.txt` の有無にかかわらず直下の全ディレクトリを名前順で返す。
- `Resolve(..., true)` は相対・絶対・Browse の全経路で、既存ディレクトリに不足する `config/_main.txt` だけをテンプレートから補完し、その後に `runs/` を作る。既存 `_main.txt` は上書きせず、不正な型は fail-fast する。
- `Resolve(..., false)` は副作用なしを維持し、非存在の絶対・多階層パスも引き続き作成しない。
- `IsResolvable()` は完成済みに加え、テンプレートから補完可能な既存ディレクトリを選択可能と判定する。補助 launcher は自動初期化しない。
- PRD046 と Runner 利用ガイドへ、過去アーカイブを含む既存フォルダの初回選択時補完を追記する。`CONTEXT.md` と ADR は変更しない。

## テスト

- Public interface / surface: `WorkspaceService::ScanLocalWorkspaces()`、`Resolve()`、`IsResolvable()`。
- 優先 behavior: 不完全な相対 workspace の列挙・補完・既存ファイル保持を tracer bullet とし、絶対パス補完、既存 config 非上書き、副作用なし失敗、不正型・テンプレート不在を順に確認する。
- TDD 順序: behavior ごとに 1 テストを RED にし、最小実装で GREEN にしてから次へ進む。refactor は全対象が GREEN になった後だけ行う。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target AnetRLRunner AnetRLRunner-test -j1'
apps\runner\bin\Debug\AnetRLRunner-test.exe
git diff --check
```

## 前提

- 自動補完は確認ダイアログなしで行い、現在の `_workspace_template.txt` をそのまま使う。
- 過去 Run の移動・変換・設定推測は行わず、不足した workspace config だけを補う。
- 無関係な dirty ファイルは変更せず、stage、commit、push は行わない。

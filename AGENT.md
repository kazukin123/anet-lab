# AGENT.md

## Viewing UTF-8 Japanese Text in Codex on Windows

This file is encoded as UTF-8. If Japanese text appears as mojibake in the Codex
PowerShell terminal, the file is usually still correct; the terminal output
encoding is the problem. Before reading or printing this file, switch the
console/output encoding to UTF-8:

```powershell
chcp 65001
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
Get-Content -Encoding UTF8 AGENT.md
```

`git diff -- AGENT.md` may render Japanese correctly even when plain
`Get-Content AGENT.md` does not, because Git and PowerShell use different
output paths and encodings in this environment.

このドキュメントは、anet-lab を編集する AI エージェントおよび開発支援ツール向けの作業規約です。
人間が読む開発メモとしても使えるように、リポジトリ構成、ビルド手順、コーディング方針をまとめます。

## プロジェクト概要

anet-lab は libtorch を基盤とした C++20 の強化学習実験プロジェクトです。

- `core/anet-core`: 強化学習フレームワーク、Agent、NN、設定、メトリクスなどの共通実装
- `core/envs`: 環境実装
- `apps/runner`: 実行アプリケーションと設定
- `viewers`: メトリクス表示・可視化ツール
- `docs`: 設計資料、Doxygen 設定、実行結果
- `third_party`: 外部依存ライブラリ

このプロジェクトは実験・学習目的のコードを含みます。
変更時は、広範なリファクタリングよりも、目的に対して局所的で読みやすい変更を優先してください。

## 基本方針

- 既存の設計、命名、ファイル配置を尊重する。
- 変更範囲は要求された目的に必要な範囲へ絞る。
- 公開ヘッダへの変更は慎重に行う。
- 依存方向を単純に保ち、循環依存を避ける。
- 暗黙のグローバル状態を増やさない。
- 生成物やローカル環境ファイルを不用意に編集・追加しない。

## コーディング規約

C++ コードは Google C++ スタイルガイドを前提とします。
ただし、既存コードに明確なローカル規約がある場合は、無理に一括変更せず、周辺コードとの一貫性を優先してください。

特に以下を意識してください。

- C++20 を前提とする。
- 読みやすく、責務が明確な実装にする。
- include は必要最小限にする。
- C++ の `.hpp` 側では `using namespace` を使用しない。
- C++ の `.cpp` 側では、`namespace ... {}` で全体を囲むのではなく `using namespace ...;` を使用。
- 同じ名前空間で省略可能な名前空間修飾は省略する。
- `const` を適切かつ積極的に使う。
- 例外、安全性、境界条件を意識する。
- 大規模な整形変更や無関係なリネームは避ける。
- 改行コードはCRLFで統一。特に1つのファイル内に LF と CRLF を混在させない。

## コメント・TODO ルール

- 日本語でコメントを入れる。
- リファクタリング時も、処理段階を説明する既存コメントは原則残す。
- 共通化や関数切り出しでコメントが消える場合は、移動先の関数に同等の意図コメントを移す。
- コメントは「この段階で何をしているか」「なぜこの順序なのか」を後から追える粒度にする。
- 行単位で自明な説明を増やすのではなく、アルゴリズム、同期境界、副作用境界、メトリクス算出意図を優先して説明する。
- 実装中に設計上の懸念、未対応の分岐、将来修正が必要な点に気づき、その場で解決しない場合は Doxygen 形式の TODO コメントを残す。
- TODO コメントは `/// @todo ...` または `///< @todo ...` の形式を使い、理由と残作業が追える短い内容にする。

## Agent 系実装の所有権ルール

Agent 関連の変数・オブジェクト追加時は、必ず以下の資料に従ってください。

- `docs/ownership_guideline.md`

要点:

- State は、それを更新するモジュールが所有する。
- Resource は Agent が所有する。
- Policy が Learner に依存する構造は禁止する。
- Agent モジュール間の循環依存を避ける。

例:

- epsilon、EMA 統計、warmup counter などは、それを更新するモジュールの State。
- policy net、target net、optimizer、ReplayBuffer、RNG、Config などは Agent 所有の Resource。

## ビルド

主な想定環境は Windows x64 です。
CMake Presets を使ってビルドします。

```powershell
cmake --preset x64-Debug
cmake --build --preset x64-Debug
```

利用可能な preset:

- `x64-Debug`
- `x64-RelWithDebInfo`
- `x64-Release`

主な依存:

- libtorch
- wxWidgets
- Ninja
- MSVC toolchain

libtorch のパスは以下の環境変数で指定できます。

- `Torch_DIR_DEBUG`
- `Torch_DIR_RELEASE`
- `Torch_DIR`

## 検証

コード変更後は、可能な限り少なくとも Debug ビルドを実行してください。

```powershell
cmake --build --preset x64-Debug
```

Doxygen ドキュメントを確認する場合:

```powershell
cmake --build --preset x64-Debug --target doc
```

テストが追加された場合は、このドキュメントに標準のテスト実行手順を追記してください。

## 編集しない・慎重に扱う領域

以下は生成物、ローカル環境、または外部依存として扱います。

- `out/`
- `.vs/`
- `docs/runs/`
- `third_party/`

これらは明示的な依頼がない限り編集しないでください。
特に `third_party/` 配下のコード変更は、依存ライブラリ修正が目的である場合に限ります。

## AI エージェントの作業ルール

編集前:

- 近い実装と既存ドキュメントを確認する。
- 変更の目的と影響範囲を把握する。
- ユーザーの未コミット変更を勝手に戻さない。

編集中:

- 要求された目的に直接関係する変更だけを行う。
- 既存の命名・責務分割・依存方向を壊さない。
- 迷った場合は、局所変更を優先し、設計変更は明示的に説明する。
- テストのために production 本体 API や設計を歪めない。
- `static` は純粋計算・状態非依存の helper に限定する。
- テストは必要なら fixture、test-only subclass、public 経路で組み、本体に test-only API を増やさない。

編集後:

- 変更したファイルを要約する。
- 実行したビルド・検証コマンドを報告する。
- 実行できなかった検証があれば理由を明記する。

## Codex でのビルド注意事項 (Windows/MSVC)

Codex の標準 PowerShell 環境では `cl.exe` が見えていても、MSVC 標準ヘッダの
include パスが `INCLUDE` に入っていない場合があります。この状態で C++ ターゲットを
ビルドすると、次のようなエラーで失敗します。

```text
fatal error C1083: Cannot open include file: 'type_traits': No such file or directory
```

Codex から確実にビルドする場合は、Visual Studio Developer Command Prompt の
初期化バッチを経由して CMake ビルドを実行してください。

```powershell
cmd /s /c "`"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test"
```

他のターゲットをビルドする場合も同じ形式を使います。

```powershell
cmd /s /c "`"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug"
```

Codex の PowerShell から `Launch-VsDevShell.ps1` を使う方法には依存しないでください。
PowerShell の実行ポリシーでブロックされることがあり、この環境では Visual Studio の
インストール情報を解析する段階でも失敗しました。

Codex からビルドする場合、MSVC、Windows SDK、CUDA、libtorch、vcpkg が
ワークスペース外にあるため、サンドボックス外実行の承認が必要になることがあります。

`anet-core-test` をビルドした後は、リポジトリルートから次のように実行します。

```powershell
core\anet-core\bin\Debug\anet-core-test.exe
```

テスト実行ファイルは意図的に `core/anet-core/bin/<Config>` 配下へ出力します。
CMake の post-build 処理で libtorch の DLL を実行ファイルの隣へコピーし、
runner アプリと同じ実行時配置に揃えています。

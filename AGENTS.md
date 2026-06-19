# AGENTS.md

## Viewing UTF-8 Japanese Text in AI Agent Terminals on Windows

This file is encoded as UTF-8. If Japanese text appears as mojibake in an AI
agent's PowerShell terminal, the file is usually still correct; the terminal output
encoding is the problem. Before reading or printing this file, switch the
console/output encoding to UTF-8:

```powershell
chcp 65001
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
Get-Content -Encoding UTF8 AGENTS.md
```

`git diff -- AGENTS.md` may render Japanese correctly even when plain
`Get-Content AGENTS.md` does not, because Git and PowerShell use different
output paths and encodings in this environment.

このドキュメントは、anet-lab を編集する AI エージェントおよび開発支援ツール向けの作業規約です。
`CLAUDE.md` などから参照される場合もあるため、特定の AI エージェント実装に依存しない共通ルールとして扱います。
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

## 設定値の扱い

設定ファイル、コマンドライン引数、実験パラメータなど、ユーザーが指定する値は明示的に検証してください。

- 人間が明示的に指定した設定値はできるだけ尊重し、勝手に丸めたり別の値へ置き換えたりしない。
- 非推奨だが動作可能な設定値は、可能な限り指定どおりに動かし、WARN ログでキー、指定値、非推奨理由、推奨される代替設定を知らせる。
- 不正値、未知の列挙値、範囲外の値、互換性のない組み合わせなど、動作不能または意味が破綻する設定を検出した場合は、`ANET_SYSTEM_ERROR` で明示的に落とす。
- 動作不能な設定を暗黙に既定値へフォールバックしない。ユーザーが設定ミスに気づき、修正できる失敗にする。
- 後方互換性などの理由でフォールバックを許容する場合は、その理由、条件、通知方法をコードまたはドキュメント上で明確にする。
- エラーメッセージには、問題のあるキー、指定値、期待される値または範囲を含める。

## コーディング規約

C++ コードは Google C++ スタイルガイドを前提とします。
ただし、既存コードに明確なローカル規約がある場合は、無理に一括変更せず、周辺コードとの一貫性を優先してください。

個別の指示・合意が無い限りにおいて、以下を意識してください。

- C++20 を前提とする。
- 性能影響に対しては妥協しない姿勢を基本とし、可読性・保守性との折り合いを付ける。
- 読みやすく、責務が明確な実装にする。
- ANET では、原則として 1 クラス 1 ファイルではなく、機能グループ単位で `.hpp` / `.cpp` を作る。
  例: `dqn_based_agent.*` は DQN 系の Policy、Actor、Learner をまとめ、`nn_modules.*` は NN module 群をまとめ、`observers.*` は Observer 群をまとめる。
- 新しいクラスを追加する場合も、まず既存の機能グループに属するかを確認し、独立したサブシステムとして分ける理由がある場合だけ新規ファイルを作る。
- C++20 の指示付き初期化（Designated Initializers）は原則使用し、可読性が極端に低下する場合のみ通常のコンストラクタ呼び出しや段階的な代入を優先する。
- include は必要最小限にする。
- C++ の `.hpp` 側では `using namespace` を使用しない。
- C++ の `.cpp` 側では、`namespace ... {}` で全体を囲むのではなく `using namespace ...;` を使用。
- 同じ名前空間で省略可能な名前空間修飾は省略する。
- `const` を適切かつ積極的に使う。
- `dynamic_cast` はテストコード内に限定し、production code では使用しない。型ごとの分岐が必要な場合は、仮想関数、明示的な interface、既存 API の拡張、または型情報を持つデータ構造で表現する。
- 例外、安全性、境界条件を意識する。
- 大規模な整形変更や無関係なリネームは避ける。
- 改行コードは LF で統一。
- `.hpp`、`.cpp`、`.java`、`.md`、`.editorconfig`、`.gitattributes` は `.editorconfig` と `.gitattributes` の LF 指定に従う。
- `third_party/` 配下は外部依存として改行コード統一の一括対象から除外する。

## コメント・TODO ルール

- 日本語でコメントを入れる。
- リファクタリング時も、処理段階を説明する既存コメントは原則残す。
- 共通化や関数切り出しでコメントが消える場合は、移動先の関数に同等の意図コメントを移す。
- コメントは「この段階で何をしているか」「なぜこの順序なのか」を後から追える粒度にする。
- 行単位で自明な説明を増やすのではなく、アルゴリズム、同期境界、副作用境界、メトリクス算出意図を優先して説明する。
- 実装中に設計上の懸念、未対応の分岐、将来修正が必要な点に気づき、その場で解決しない場合は Doxygen 形式の TODO コメントを残す。
- TODO コメントは `/// @todo ...` または `///< @todo ...` の形式を使い、理由と残作業が追える短い内容にする。

## 性能測定・ProfileRange ルール

性能測定が必要な処理には `ANET_PROFILE_SCOPE` 系のマクロを入れ、後から実測できる状態を保ってください。

特に以下の処理を追加・変更する場合は、計測範囲を入れることを優先してください。

- 学習・評価・実行ループから頻繁に呼ばれる処理。
- `Step`、`Reset`、`MakeAction`、`UpdateFromBatch`、`Forward`、`Sample`、`Push` など、実行時間の主要因になりやすい境界。
- batch size、env 数、action 数、node 数、画像サイズ、ログ量などに応じて処理量が増える処理。
- Tensor 変換、device 転送、同期、queue 処理、ReplayBuffer、NN forward、loss 計算、可視化、画像・動画・GraphViz 出力など、CPU/GPU/I/O の負荷が読みにくい処理。
- 既存の計測済み処理を分割・共通化する場合の、分割後の主要フェーズ。

計測名は既存コードに合わせて `ClassName::FunctionName` または `ClassName::FunctionName.phase` のように安定した名前にしてください。
細かすぎる getter、単純な分岐、軽量な per-element 内側ループには原則として入れず、測定のノイズにならない意味のある処理境界を選んでください。
計測用のブロックを作ってスコープを不自然に狭めないでください。
フェーズ全体を測る場合は、原則として `ANET_PROFILE_SCOPE(phase)` を対象フェーズの先頭に置き、そのフェーズで使うローカル変数の構築・初期化も測定対象に含めてください。
測定対象から意図的に外したい重い初期化がある場合だけ、別スコープへ分けてください。
連続フェーズを測る場合は、同じ可視 lifetime 内で `ANET_PROFILE_SCOPE_NEXT(phase)` または `ANET_PROFILE_SCOPE_NEXT_FROM(phase, prev_phase)` を使ってください。

`ANET_PROFILE_SCOPE_FULL(var_name, full_name_literal)` は例外用途です。
lambda / callback / timer / async worker / prefetch など、自動生成される `ClassName::FunctionName.phase` が profiler 上の論理処理名として不適切な場合、または既存の計測名を維持して before/after 比較を可能にしたい場合だけ使ってください。
`full_name_literal` は `ClassName::FunctionName.phase` またはそれに準じる stable な完全名にしてください。
通常のメンバ関数・自由関数内のフェーズ計測では `ANET_PROFILE_SCOPE(phase)` を優先し、単に任意の名前を付けたいという理由では `ANET_PROFILE_SCOPE_FULL` を使わないでください。

推奨:

```cpp
ANET_PROFILE_SCOPE(sample);
ExperienceSamples cpu_samples;
inner_->Sample(cpu_samples, minibatch_size, beta);
```

避ける例:

```cpp
ExperienceSamples cpu_samples;
{
    ANET_PROFILE_SCOPE(sample);
    inner_->Sample(cpu_samples, minibatch_size, beta);
}
```

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
MSVC 環境が初期化済みのシェルでは CMake Presets を使ってビルドします。
AI エージェントの実行シェルや通常の PowerShell からビルドを試す場合は、素の
`cmake --build` を一度試すのではなく、後述の Windows/MSVC 注意事項に従い、最初から
`VsDevCmd.bat` を `call` してから CMake を実行してください。

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
AI エージェントが検証する場合は、素の PowerShell から次の `cmake --build` を直接実行せず、
Windows/MSVC 注意事項の `VsDevCmd.bat` 経由コマンドを使ってください。

```powershell
cmake --build --preset x64-Debug
```

Doxygen ドキュメントを確認する場合:

```powershell
cmake --build --preset x64-Debug --target doc
```

テストが追加された場合は、このドキュメントに標準のテスト実行手順を追記してください。

## Python 補助ツールの実行

AI エージェントがリポジトリ内の Python 補助ツールを実行する場合は、素の `python` ではなく
リポジトリルートの `.\.venv\Scripts\python.exe` を優先してください。
user site やグローバル Python に入っているパッケージは、AI エージェントのサンドボックスから
見えない場合があるため、前提にしないでください。

`.venv` が存在しない場合は、リポジトリルートで次のように作成してください。

```powershell
C:\Python314\python.exe -m venv .venv
```

依存パッケージは、必ず `.venv` 側の Python に対してインストールしてください。

```powershell
.\.venv\Scripts\python.exe -m pip install ...
```

Python 補助ツールの検証も、同じ `.venv` の Python で行ってください。
`.venv` はローカル実行環境として扱い、Git 管理対象にしないでください。

## 編集しない・慎重に扱う領域

以下は生成物、ローカル環境、または外部依存として扱います。

- `out/`
- `.vs/`
- `docs/runs/`
- `third_party/`

これらは明示的な依頼がない限り編集しないでください。
特に `third_party/` 配下のコード変更は、依存ライブラリ修正が目的である場合に限ります。

## AI エージェントの応答言語ルール

このリポジトリで作業する AI エージェントは、ユーザーが明示的に別言語を指定しない限り日本語で書いてください。

`<proposed_plan>` を出す場合も、本文と見出しは日本語にしてください。

コード、コマンド、ファイルパス、API 名、設定キー、エラーメッセージ、外部仕様名、引用は、原文または既存表記を保持してかまいません。

実装内のログは後述のログ出力ルールに従い、UI 文言、出力ファイルの文言は、周辺コード、既存仕様、個別指示を優先してください。
この応答言語ルールだけを理由に、実装内の英語文言を一括で日本語化しないでください。

## ログ出力ルール

実装内のログ出力は、ユーザーから個別に指定がない限り基本英語で書いてください。

- `LOG::info()`、`LOG::warn()`、`LOG::error()`、`anet::log::warn()`、`ANET_LOG_DEBUG`、`ANET_SYSTEM_ERROR` などで出すメッセージは英語を既定にする。
- 設定キー、ファイルパス、型名、関数名、例外メッセージ、外部ツールの出力は、原文または既存表記を保持してよい。
- 既存の日本語ログを、このルールだけを理由に一括で英語へ翻訳しない。触った箇所、新規追加箇所、修正対象の近傍から適用する。
- コメントやユーザー向け応答は、それぞれの日本語ルールを優先し、ログ出力ルールと混同しない。

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
- ビルドを試す場合は、素の PowerShell から `cmake --build` せず、PowerShell-safe な `cmd /s /c 'call "...VsDevCmd.bat" ... && cmake --build ...'` 形式で MSVC 環境を初期化する。
- 実行できなかった検証があれば理由を明記する。

## AI エージェントでのビルド注意事項 (Windows/MSVC)

AI エージェントの標準 PowerShell 環境では `cl.exe` が見えていても、MSVC 標準ヘッダの
include パスが `INCLUDE` に入っていない場合があります。この状態で C++ ターゲットを
ビルドすると、次のようなエラーで失敗します。

```text
fatal error C1083: Cannot open include file: 'type_traits': No such file or directory
```

AI エージェントから C++ ビルドを試す場合は、PowerShell では外側を単一引用符にし、必ず
`cmd /s /c 'call "...\VsDevCmd.bat" -arch=x64 -host_arch=x64 && ...'` の形で実行してください。
`VsDevCmd.bat` のパスは内側の二重引用符で囲み、MSVC 初期化と CMake ビルドを同じ `cmd`
プロセスで実行してください。`cl.exe` だけが見えていても
`INCLUDE`、`LIB`、Windows SDK などの環境が不足することがあるため、素の PowerShell から
`cmake --build` を実行しないでください。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
```

他のターゲットをビルドする場合も同じ形式を使います。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```

次のように外側と内側の二重引用符が衝突する形は使わないでください。

```powershell
cmd /s /c "call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug"
```

AI エージェントの PowerShell から `Launch-VsDevShell.ps1` を使う方法には依存しないでください。
PowerShell の実行ポリシーでブロックされることがあり、この環境では Visual Studio の
インストール情報を解析する段階でも失敗しました。

AI エージェントからビルドする場合、MSVC、Windows SDK、CUDA、libtorch、vcpkg が
ワークスペース外にあるため、サンドボックス外実行の承認が必要になることがあります。

`anet-core-test` をビルドした後は、リポジトリルートから次のように実行します。

```powershell
core\anet-core\bin\Debug\anet-core-test.exe
```

テスト実行ファイルは意図的に `core/anet-core/bin/<Config>` 配下へ出力します。
CMake の post-build 処理で libtorch の DLL を実行ファイルの隣へコピーし、
runner アプリと同じ実行時配置に揃えています。

## Agent skills

### Issue tracker

Issues live in the GitHub repo `kazukin123/anet-lab`. Use the `gh` CLI.
See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical roles map 1:1 to label strings (no overrides).
See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root.
See `docs/agents/domain.md`.

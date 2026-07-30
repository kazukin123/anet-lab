# ANET 開発環境構築ガイド

> 主たる観点: 行程単位（前提確認、依存準備、ビルド、テスト）

## 1. はじめに

### 1.1 目的

この文書は、ANET の C++ 本体、Runner、Metrics Viewer、Python 補助ツールを変更し、ビルドとテストを行うための開発環境を準備する手順を示します。

### 1.2 対象読者

- ANET 本体、Agent、Env、Runner を変更する開発者
- Metrics Viewer や Python 補助ツールを変更する開発者
- CI と同等のビルド・テストをローカルで再現したい開発者

### 1.3 記載範囲

Windows 上の開発環境、外部依存の準備、CMake/Maven/Python のビルド・テスト手順を扱います。ビルド済み Runner の設定と操作は [Run 実行ガイド](020_user_guide_run.jp.md)で扱います。

## 2. 検証済み環境と前提

現時点でフレームワーク全体の開発・実行を確認しているのは、Windows 11 x64、MSVC、NVIDIA GPU、CUDA 対応版 libtorch の組み合わせです。

| 構成 | 状態 |
|---|---|
| Windows 11 x64 + NVIDIA CUDA | 検証済みの標準構成 |
| CPU-only | 個別の CPU 経路はあるが、開発環境から Runner 実行までの一式は未検証 |
| Linux / macOS | 未検証 |
| Visual Studio | Visual Studio 2022 の MSVC toolchain を使用 |
| C++ | C++20 必須 |
| CMake | 3.20 以上を要求 |

ローカル開発で利用する libtorch、CUDA Toolkit、NVIDIA ドライバーは、相互に互換性のある組み合わせを選んでください。特定バージョンをフレームワークの普遍的な要件とはしていません。CI で現在使用している組み合わせは `.github/workflows/windows-ci.yml` で確認できます。

## 3. 必要なツールとライブラリ

### 3.1 C++ 本体と Runner

| 項目 | 用途・要件 |
|---|---|
| Visual Studio 2022 | MSVC x64 compiler と Windows SDK。C++ デスクトップ開発に必要な構成を導入する |
| CMake | configure と build。リポジトリの最低要求は 3.20 |
| Ninja | `CMakePresets.json` が使用する generator |
| libtorch | Tensor、NN、自動微分。Debug 用と Release 用を分けて用意する |
| wxWidgets | Runner GUI。`core`、`base`、`gl`、`aui` component が必要 |
| NVIDIA driver / CUDA Toolkit | CUDA 対応版 libtorch と GPU を利用するために必要 |

Box2D、Catch2、nlohmann/json、Tracy、NVTX header は `third_party/` に含まれています。Doxygen、Graphviz、ffmpeg、Tracy server、NVIDIA Nsight は、それぞれ文書生成、図・動画生成、性能分析を行う場合に追加します。

### 3.2 Metrics Viewer

`viewers/metrics-viewer/pom.xml` は Java 17 を target とします。JDK 17 以上と Maven を用意してください。JavaScript 表示テストは、ローカルに Microsoft Edge がない場合は skip されます。

### 3.3 Python 補助ツール

Python は Optuna harness、Python 版 Metrics Viewer、TensorBoard bridge、MLflow bridge などで使用します。C++ 本体と Java 版 Metrics Viewer のビルドだけを行う場合は必須ではありません。

リポジトリには補助ツール全体をまとめた固定 requirements file がないため、利用するツールに応じて依存を導入します。代表例は `optuna`、`pandas`、`pyarrow`、`plotly`、`dash`、`torch`、`tensorboard`、`mlflow` です。

## 4. 外部依存の準備

### 4.1 libtorch

MSVC の Debug/Release runtime と合わせるため、libtorch は Debug 用と Release/RelWithDebInfo 用を別に用意します。環境変数には `libtorch` ルートではなく `share/cmake/Torch` を指定します。

```powershell
$env:Torch_DIR_DEBUG = 'C:\path\to\libtorch-debug\share\cmake\Torch'
$env:Torch_DIR_RELEASE = 'C:\path\to\libtorch-release\share\cmake\Torch'
```

`Torch_DIR_DEBUG` または `Torch_DIR_RELEASE` がない場合は `Torch_DIR` を共通の代替値として参照します。それもない場合、CMake は次のリポジトリ内パスへフォールバックします。

```text
third_party/libtorch/debug/share/cmake/Torch
third_party/libtorch/release/share/cmake/Torch
```

CUDA 対応版 libtorch を使う場合は、配布物が要求する CUDA 世代とドライバーの互換性も確認します。

### 4.2 wxWidgets

ルート `CMakeLists.txt` は `find_package(wxWidgets REQUIRED COMPONENTS core base gl aui)` を使用し、Config mode と Module mode の両方を受け付けます。wxWidgets をビルド・インストールするか、vcpkg などで x64 向けに導入し、CMake から発見できるようにします。

Config package を使う場合の configure 例:

```powershell
cmake --preset x64-Debug -DwxWidgets_DIR=C:/path/to/wxWidgets/lib/cmake/wxWidgets
```

Module mode の配置を使う場合は、configure 時に `wxWidgets_ROOT_DIR` を渡せます。

```powershell
cmake --preset x64-Debug -DwxWidgets_ROOT_DIR=C:/path/to/wxWidgets
```

このリポジトリには `vcpkg.json` がないため、vcpkg の package 導入と toolchain/検索パスの指定は開発環境側で行います。

### 4.3 CUDA

NVIDIA driver、CUDA Toolkit、CUDA 対応版 libtorch を互換性のある組み合わせで導入します。セットアップ後に、少なくとも次を確認します。

```powershell
nvidia-smi
nvcc --version
$env:CUDA_PATH
```

`agent.device_type` や評価 device の設定を CUDA にしたとき、Runner は libtorch の CUDA backend を利用します。CUDA の同期デバッグや決定性は `backend.*` 設定で制御しますが、通常の開発環境構築では既存設定を起点にしてください。

## 5. C++ プロジェクトの構成とビルド

### 5.1 MSVC 環境の初期化

通常の PowerShell では `cl.exe` が見えても、MSVC 標準 header、library、Windows SDK の環境変数が不足することがあります。configure と build は、`VsDevCmd.bat` を呼び出した同じ `cmd` process 内で実行します。

Visual Studio Community の標準的な配置例:

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --preset x64-Debug'
```

Edition やインストール先が異なる場合は `VsDevCmd.bat` のパスを読み替えます。`Launch-VsDevShell.ps1` には依存しません。

### 5.2 CMake Preset

| Preset | 用途 | 出力ディレクトリ |
|---|---|---|
| `x64-Debug` | assert と debug 情報を有効化 | `out/build/x64-Debug` |
| `x64-RelWithDebInfo` | 最適化と debug 情報 | `out/build/x64-RelWithDebInfo` |
| `x64-Release` | 最適化、assert 無効 | `out/build/x64-Release` |

最初は Debug を configure・build します。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --preset x64-Debug'
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```

主な生成物は次の場所へ出力されます。

| 生成物 | Debug の出力先 |
|---|---|
| Runner | `apps/runner/bin/Debug/AnetRLRunner.exe` |
| core test | `core/anet-core/bin/Debug/anet-core-test.exe` |
| LunarLander test | `core/envs/lunarlander1/bin/Debug/LunarLanderEnv-test.exe` |
| ImageCls test | `core/envs/imagecls1/bin/Debug/ImageClsEnv-test.exe` |

MSVC build では post-build 処理が libtorch DLL を実行ファイルの隣へコピーします。

### 5.3 C++ テスト

CMake に登録されているテストをまとめて実行します。

```powershell
ctest --preset x64-Debug --output-on-failure
```

対象を絞る場合は、先に target をビルドしてから実行ファイルをリポジトリルートで起動します。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe
```

同様に `LunarLanderEnv-test` と `ImageClsEnv-test` を個別実行できます。

## 6. Metrics Viewer のビルドとテスト

Java 版 Metrics Viewer は Maven project です。

```powershell
cd viewers\metrics-viewer
mvn -B test
mvn -B package
```

package 後の実行可能 JAR は `viewers/metrics-viewer/target/metrics-viewer.jar` です。起動確認の例:

```powershell
java -Xmx1g -jar target\metrics-viewer.jar --server.port=8082
```

Run ディレクトリなどの起動引数は [Run 分析ガイド](030_user_guide_analysis.jp.md)を参照してください。

## 7. Python 補助ツールの環境

Python package を user site や global 環境へ混在させず、リポジトリルートの `.venv` を使用します。

```powershell
C:\Python314\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

上の Python パスは現在の開発環境に合わせた例です。別の Python を使う場合も、作成後は必ず `.\.venv\Scripts\python.exe` を指定して package の導入と補助ツールの実行を行います。

```powershell
.\.venv\Scripts\python.exe -m pip install <必要なパッケージ>
.\.venv\Scripts\python.exe apps\runner\tools\dropmerge_optuna.py --help
```

`.venv` はローカル開発環境であり、Git 管理対象にしません。

### 7.1 MLflow bridge

MLflow bridge と MLflow server は、リポジトリルートの `.venv` と `apps/runner/runs/mlflow.db` を共有します。
依存 package は次のコマンドで導入してください。

```powershell
.\.venv\Scripts\python.exe -m pip install -r viewers\metrics-tools\requirements.txt
```

`requirements.txt` は MLflow を `3.13.0` に固定しています。
MLflow 3.14.0 の server は Python 3.14 で削除された `importlib.abc.Traversable` を import するため、この組み合わせでは起動できません。
`apps/runner/41_mlflow_bridge.bat` と `apps/runner/42_start_mlflow.bat` は、要求 version が導入されていない場合に fail-fast します。

MLflow bridge は対象 Run の `config/config_data.txt` を読み、各 `key = value` を MLflow parameter として記録します。
MLflow parameter 名で使用できない `[` と `]` は除去し、その他の使用不可文字は `_` へ置換します。
変換後の parameter 名が衝突する場合は、値を上書きせず fail-fast します。

`apps/runner/41_mlflow_bridge.bat` は `apps/runner/runs/run_*/metrics.jsonl` を列挙し、起動時点で存在するすべての直下 Run を MLflow へ変換します。
監視中に追加された直下 Run も自動的に対象へ追加します。
更新時刻が最も新しい Run は、保存済み offset が現在の末尾へ追いつくまで優先して変換します。
最新 Run の処理中も、10 batchごとに過去 Runを1 batch処理し、対象はround-robinで交代します。
追いついた後は、他の Run を 1 batch ずつ変換します。
監視中は、前回表示後にmetricsの処理が進んだRunに限り、最大10秒間隔で最新Runと過去Runを区別して、Run名、処理offset、末尾までの遅延量をconsoleへ表示します。
`runs/group/run_*/metrics.jsonl` のように別 directory の下へネストされた Run は対象外です。
MLflow の `Status=RUNNING` は学習 process の生存状態ではなく、bridge が継続監視する対象として登録したことを示します。
`--once` で変換した場合だけ、現在の末尾まで取り込んだ後に `Status=FINISHED` とします。

## 8. トラブルシューティング

### 8.1 `type_traits` などの標準 header が見つからない

MSVC 環境が不完全です。通常の PowerShell から直接 `cmake --build` せず、`VsDevCmd.bat` と build を同じ `cmd` process で実行してください。

### 8.2 CMake が Torch を見つけない

`Torch_DIR_DEBUG` / `Torch_DIR_RELEASE` が、それぞれ対象配布物の `share/cmake/Torch` を指しているか確認します。Debug build と Release 版 libtorch の混在も避けてください。

### 8.3 CMake が wxWidgets を見つけない

導入した wxWidgets が x64 向けであることと、`core`、`base`、`gl`、`aui` を含むことを確認します。その後、Config mode なら `wxWidgets_DIR`、Module mode なら `wxWidgets_ROOT_DIR` または CMake toolchain/search path を設定します。

### 8.4 CUDA または DLL の読み込みに失敗する

libtorch の CUDA 世代、NVIDIA driver、CUDA Toolkit の互換性を確認します。実行ファイル隣の DLL が古い場合は、対象 configuration を再ビルドして post-build copy をやり直します。

### 8.5 Maven が Java release 17 を扱えない

`java -version` と `mvn -version` が同じ JDK 17 以上を参照しているか確認します。必要に応じて `JAVA_HOME` と `PATH` を修正します。

## 9. 関連文書

- [ドキュメント一覧](README.jp.md)
- [ANET フレームワーク全体概要](010_framework_overview.jp.md)
- [Run 実行ガイド](020_user_guide_run.jp.md)
- [開発・作業規約](../../AGENTS.md)
- [ルート CMakeLists](../../CMakeLists.txt)
- [CMake Presets](../../CMakePresets.json)
- [Windows CI](../../.github/workflows/windows-ci.yml)
- [Metrics Viewer Maven 設定](../../viewers/metrics-viewer/pom.xml)

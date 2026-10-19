# ANET 開発環境構築ガイド

> 主たる観点: 初期環境構築から IDE 上のビルド完了まで

## 1. はじめに

### 1.1 目的

この文書は、ANET に興味を持った開発者が上から順に作業し、C++ 本体と Runner、必要に応じて Metrics Viewer をビルドできる状態になるまでの手順を示します。

C++ 本体と Runner は Visual Studio、Metrics Viewer は Eclipse を使う GUI 開発を標準経路とします。CLI は AI エージェント、CI 相当の検証、GUI で問題が起きた場合の診断に使用する補助経路として分離します。

### 1.2 対象読者

- ANET 本体、Agent、Env、Runner を変更する開発者
- Metrics Viewer や Python 補助ツールを変更する開発者
- CI と同等のビルド・テストをローカルで再現したい開発者または AI エージェント

### 1.3 この文書の完了地点

| 読み終える章 | 完了状態 |
|---|---|
| 3章 | Visual Studio、libtorch、wxWidgets、CUDA など、ビルドに必要な初期環境が揃っている |
| 4章 | Visual Studio 上で C++ 本体と Runner を Debug ビルドできる |
| 5章 | Metrics Viewer をビルドして起動できる。変更する場合は Eclipse 上でビルドできる |

C++ 本体と Runner だけを扱う場合は4章までで初期構築は完了です。Metrics Viewer を使う場合は5.1節まで、Metrics Viewer を変更する場合は5章の最後まで進めてください。ビルド済み Runner の設定と操作は [Run 実行ガイド](020_user_guide_run.jp.md)、Run の分析は [Run 分析ガイド](030_user_guide_analysis.jp.md)で扱います。

## 2. 検証済み環境と推奨 IDE

### 2.1 検証済み環境

現時点でフレームワーク全体の開発・実行を確認しているのは、Windows 11 x64、MSVC、NVIDIA GPU、CUDA 対応版 libtorch の組み合わせです。

| 構成 | 状態 |
|---|---|
| Windows 11 x64 + NVIDIA CUDA | 検証済みの標準構成 |
| CPU-only | 個別の CPU 経路はあるが、開発環境から Runner 実行までの一式は未検証 |
| Linux / macOS | 未検証 |
| Visual Studio | Visual Studio Community 2022 と MSVC v143 で検証 |
| C++ | C++20 必須 |
| CMake | リポジトリの最低要求は 3.20 |
| Java | Metrics Viewer は Java 17 を target とする |

libtorch、CUDA Toolkit、NVIDIA driver は相互に互換性のある組み合わせを選びます。このリポジトリの CI が現在使用している version は [Windows CI](../../.github/workflows/windows-ci.yml)を正本として確認してください。

### 2.2 C++ 本体と Runner の推奨 IDE

C++ 本体と Runner のコーディング、CMake configure、ビルド、デバッグには Visual Studio を推奨します。現在の標準開発環境は Visual Studio Community 2022 です。

Visual Studio Community の利用条件に合わない組織では、Visual Studio Professional または Enterprise を使用してください。使用する edition が異なっても、この文書の「フォルダーを開く」形式の CMake 開発手順は同じです。Community の最新の利用条件は [Visual Studio Community](https://visualstudio.microsoft.com/vs/community/)を確認してください。

Visual Studio はリポジトリルートの `CMakeLists.txt` と `CMakePresets.json` を直接読みます。`.sln` を生成して開く手順は使用しません。

### 2.3 Metrics Viewer の推奨 IDE

Java 版 Metrics Viewer のコーディング、Maven build、testには Eclipse を推奨します。JDK 17 以上と Maven integration for Eclipse（m2e）を使用します。

### 2.4 CLI の位置付け

通常の人間の開発は Visual Studio または Eclipse 上で行います。CLI 手順は次の用途に限定し、6章にまとめます。

- AI エージェントによるビルドと検証
- CI 相当の手順をローカルで再現する場合
- IDE の configure や build が失敗した場合の切り分け

## 3. 初期環境構築

### 3.1 導入するソフトウェア

利用者が導入を意識する単位は次のとおりです。

| 導入単位 | 必要になる開発 |
|---|---|
| Git for Windows | リポジトリと ALE の取得。PowerShell から `git` を実行できる場合は追加不要 |
| Visual Studio Community 2022 または適切な edition | C++ 本体、Env、Agent、Runner |
| CUDA 対応版 libtorch の Debug 版と Release 版 | C++ 本体、Runner |
| wxWidgets（sourceからbuild） | Runner GUI |
| ALE v0.12.0（sourceからbuild） | Atari env（使う場合だけ） |
| NVIDIA driver と CUDA Toolkit | CUDA 対応版 libtorch を使う C++ 開発 |
| JDK 17 以上 | Metrics Viewer |
| Eclipse | Metrics Viewer の変更（必要な場合だけ） |
| Python | Optuna、Python viewer、TensorBoard・MLflow bridgeなどの補助ツール |

CMake と Ninja は Visual Studio Installer の「C++ CMake tools for Windows」に含まれます。Visual Studio による標準開発では、CMake と Ninja を個別にインストールする必要はありません。

Box2D、Catch2、nlohmann/json、Tracy、NVTX header は `third_party/` に含まれています。Doxygen、Graphviz、ffmpeg、Tracy server、NVIDIA Nsight は、それぞれ文書生成、図・動画生成、性能分析を行う場合に追加します。最初のビルドには不要です。

### 3.2 Visual Studio の導入

1. [Visual Studio Community 2022 Installer](https://aka.ms/vs/17/release/vs_community.exe)から Visual Studio Community 2022 を導入します。
2. Visual Studio Installer で「C++ によるデスクトップ開発」workload を選択します。
3. インストールの詳細で、少なくとも次の component が選択されていることを確認します。

| Component | 用途 |
|---|---|
| MSVC v143 - VS 2022 C++ x64/x86 build tools | C++ compiler と linker |
| Windows 11 SDK | Windows header と library |
| C++ CMake tools for Windows | Visual Studio の CMake integration、CMake、Ninja |
| vcpkg package manager | ALE が使う package（zlib、SDL2）の取得。Atari env を使う場合だけ |

既に Visual Studio を導入済みの場合は、Visual Studio Installer の「変更」から component を追加できます。Visual Studio の CMake integration は [CMake projects in Visual Studio](https://learn.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio)も参照してください。

### 3.3 libtorch の準備

MSVC の Debug/Release runtime と合わせるため、libtorch は Debug 版と Release 版を両方用意します。CUDA 世代は4章のビルド時点で CI と同じものを選ぶのが基準です。

現在の Windows CI は libtorch `2.12.0+cu130` を使用しています。次の2種類を取得します。

- Release: [`libtorch-win-shared-with-deps-2.12.0+cu130.zip`](https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-2.12.0%2Bcu130.zip)
- Debug: [`libtorch-win-shared-with-deps-debug-2.12.0+cu130.zip`](https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-debug-2.12.0%2Bcu130.zip)

この文書では、展開先を次のようにします。別の場所へ展開しても構いませんが、3.8節の環境変数には実際のパスを指定してください。

```text
C:\dev\libtorch-win-shared-with-deps-2.12.0+cu130\libtorch
C:\dev\libtorch-win-shared-with-deps-debug-2.12.0+cu130\libtorch
```

それぞれのディレクトリに `share/cmake/Torch/TorchConfig.cmake` が存在することを確認します。

### 3.4 wxWidgets の準備

wxWidgets は vcpkg を使わず、sourceからbuild・installします。Runner GUI は `core`、`base`、`gl`、`aui` componentを使用し、`find_package` は Config modeで解決するため、CMake の package configを含む install treeが必要です。公式の pre-built binaryは package configを含まないため使用できません。

[wxWidgets の release page](https://github.com/wxWidgets/wxWidgets/releases)から source archiveを取得して `C:\dev\wxWidgets-3.3.3` へ展開します。Windowsのスタートmenuから `Developer PowerShell for VS 2022` を起動し、次を実行します。

```powershell
$src = "C:\dev\wxWidgets-3.3.3"
cmake -S $src -B "$src\build" -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_INSTALL_PREFIX="$src\install" -DwxBUILD_SHARED=ON -DwxBUILD_PRECOMP=ON
cmake --build "$src\build" --config Debug --parallel
cmake --install "$src\build" --config Debug
cmake --build "$src\build" --config Release --parallel
cmake --install "$src\build" --config Release
```

generatorには anet-lab 本体と同じ toolsetを指定します。異なる toolsetでbuildすると ABI が一致しません。Debug と Release を同じ prefixへinstallすると、library名で区別されるため共存します。build treeは install 後に削除できます。

install した package configの位置は、3.8節で環境変数 `wxWidgets_DIR` に設定します。

### 3.5 NVIDIA driver と CUDA Toolkit の準備

CUDA 対応版 libtorchを使用するため、NVIDIA driver と libtorch の CUDA 世代に対応する [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)を導入します。`cu130` の libtorchを使う場合は CUDA Toolkit 13.0 を基準とします。

CUDA Toolkit Installer が設定する既定の `CUDA_PATH` を使用します。導入後、新しく開いた通常の PowerShell で次を確認します。`Developer PowerShell for VS 2022` では `nvidia-smi` が見つかりません。

```powershell
nvidia-smi
nvcc --version
$env:CUDA_PATH
```

少なくとも、NVIDIA GPU と driver が `nvidia-smi` に表示され、`nvcc --version` が選択した CUDA Toolkit を示すことを確認します。

> [!NOTE]
> `Developer PowerShell for VS 2022` は32bit版の PowerShell を起動します。32bit の processからは `C:\Windows\System32` が別の folderに読み替えられるため、そこにある `nvidia-smi` が見つかりません。また、CUDA Toolkit の導入前から開いていた terminalには、Installer が設定した `PATH` と `CUDA_PATH` が反映されません。

### 3.6 Atari env 用の ALE の準備

Atari env を使わない場合、この節は飛ばしてください。

Windowsのスタートmenuから `Developer PowerShell for VS 2022` を起動し、次を実行します。

```powershell
$ale = "C:\dev\ALE"
git clone --depth 1 --branch v0.12.0 https://github.com/Farama-Foundation/Arcade-Learning-Environment.git $ale
cmake -S $ale -B "$ale\build" -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows-static-md `
  -DBUILD_PYTHON_LIB=OFF -DSDL_SUPPORT=ON -DSDL_DYNLOAD=ON
cmake --build "$ale\build" --config Debug --parallel
cmake --build "$ale\build" --config Release --parallel
```

build後、次のファイルが存在することを確認します。build treeは削除しないでください。

```text
C:\dev\ALE\build\src\ale\Debug\ale.lib
C:\dev\ALE\build\src\ale\Release\ale.lib
```

ROM（`.bin`）を用意して1つの folderにまとめます。ファイル名は `pong.bin` のような小文字の snake_caseです。ale-py 0.12.0 を pip で導入すると、`site-packages\ale_py\roms` にこの形で入ります。

Atari env の `display_screen` か `sound` を使う場合は、`SDL2.dll` を4.1節で cloneした repositoryの `third_party\runtime_dlls\` に置き、configureし直します。`SDL2.dll` は ale-py の `ale_py\SDL2.dll` を使えます。

4.2節の configureで、CMake 出力に `Atari/ALE env enabled` が表示されることを確認します。

> [!NOTE]
> - Atari env は ALE（Arcade Learning Environment）v0.12.0 を C++ から直接使います。ALE は GPL-2.0 のため repositoryに含めず、リリースパッケージも Atari env を含めずにビルドします。構成の詳細は [Atari Env](220_atari_env.jp.md)の3章にあります。
> - ALE は Visual Studio 同梱の vcpkg を manifest modeで使い、zlib と SDL2 を build treeの中へ取得します。Developer PowerShell は環境変数 `VCPKG_ROOT` に同梱の vcpkg の位置を設定します。anet-lab の Atari env も、この build treeの中の zlib を使います。
> - ALE は installせず、anet-lab が build treeを直接参照します。build treeを削除できないのはこのためです。
> - anet-lab の configureは、`ALE_ROOT` の下に `src\ale\ale_interface.hpp` があれば Atari env を自動で有効にします（CMake 変数 `ANET_ENABLE_ATARI` の既定値 `AUTO`）。ALE がなくても、Atari env を除く C++ 本体と Runner はビルドできます。
> - ROM がなくてもビルドはでき、ROM を使うテストは skipされます。
> - `third_party\runtime_dlls\` に置いた DLL は、ビルド時に Runner の出力先へコピーされます。

### 3.7 Metrics Viewer 用の JDK

Metrics Viewer を使わない場合、この節は飛ばしてください。

JDK を導入します。推奨は Amazon Corretto 25 です。[Corretto 25 のダウンロードページ](https://docs.aws.amazon.com/corretto/latest/corretto-25-ug/downloads-list.html)から Windows x64 用の `.msi` を取得して実行します。既定の導入先は `C:\Program Files\Amazon Corretto\` の下です。

導入した JDK の位置は、3.8節で環境変数 `JAVA_HOME` と `Path` に設定します。

> [!NOTE]
> - Metrics Viewer は Java 17 を targetにしているため（`pom.xml` の `maven.compiler.release`）、JDK 17 以上ならどの版でもビルドできます。Corretto 25 を推奨するのは、現行の LTS の中で支援期間が最も長い（2032年10月まで）ためです。Metrics Viewer が使う Spring Boot 3.5 は Java 25 まで対応しています。
> - `JAVA_HOME` は5.1節の `build.bat` が、`Path` から見つかる `java` は Metrics Viewer の起動 bat が使います。

### 3.8 環境変数の設定

Windows の「システム環境変数の編集」から「環境変数」を開き、次をユーザー環境変数として設定します。

| 変数 | 設定例 |
|---|---|
| `Torch_DIR_DEBUG` | `C:\dev\libtorch-win-shared-with-deps-debug-2.12.0+cu130\libtorch\share\cmake\Torch` |
| `Torch_DIR_RELEASE` | `C:\dev\libtorch-win-shared-with-deps-2.12.0+cu130\libtorch\share\cmake\Torch` |
| `wxWidgets_DIR` | `C:\dev\wxWidgets-3.3.3\install\lib\cmake\wxWidgets-3.3` |
| `ALE_ROOT` | `C:\dev\ALE` |
| `ATARI_ROM_DIR` | `C:\dev\atari-roms` |
| `JAVA_HOME` | `C:\Program Files\Amazon Corretto\jdk25.0.4_10` |

`wxWidgets_DIR` は3.4節で install した package configの位置です。directory名は versionを含みます。

`ALE_ROOT` と `ATARI_ROM_DIR` は Atari env を使う場合だけ設定します。`ALE_ROOT` は3.6節で cloneした ALE の folder、`ATARI_ROM_DIR` は ROM を置いた folderです。

`JAVA_HOME` は Metrics Viewer を使う場合だけ設定します。3.7節で導入した JDK の folderで、folder名は versionを含みます。あわせて、ユーザー環境変数 `Path` に `%JAVA_HOME%\bin` を追加します。

`CUDA_PATH` は通常 CUDA Toolkit Installer が設定します。独自の配置を使用する場合だけ手動で設定してください。

環境変数を追加・変更した後は、起動中の Visual Studio をすべて終了してから起動し直します。Visual Studio は起動時の環境変数を使って CMake を configure します。

## 4. Visual Studio による C++ 開発

この章を上から順に行い、最後の完了確認まで通れば、C++ 本体と Runner の初期構築は完了です。

### 4.1 リポジトリをcloneして開く

最初に、任意のGit toolを使って次のrepositoryをローカルへcloneします。既にclone済みの場合は、clone手順を飛ばしてrepository rootをVisual Studioで開いてください。

```text
https://github.com/kazukin123/anet-lab.git
```

Git commandを使う例:

```powershell
git clone https://github.com/kazukin123/anet-lab.git C:\dev\anet-lab
```

Visual Studioを使う例:

1. Visual Studioの開始画面で「リポジトリのクローン」を選択します。
2. 「リポジトリの場所」に `https://github.com/kazukin123/anet-lab.git` を入力します。
3. 「パス」にclone先を指定し、「クローン」を実行します。

Visual Studio以外のGit toolでcloneした場合や、既存のcloneを開く場合は、Visual Studioの `ファイル > 開く > フォルダー`から `CMakeLists.txt` と `CMakePresets.json` があるrepository rootを選択します。

Visual Studioがrepositoryを開くと、Solution ExplorerにファイルとCMake targetが表示され、CMake configureとsource indexingが始まります。進行状況とerrorは「出力」windowのCMake出力で確認できます。

### 4.2 Debug preset で configure する

Visual Studio 上部の configure presetで `x64-Debug` を選択します。現在の `CMakePresets.json` には次の presetがあります。

| Preset | 用途 | 出力ディレクトリ |
|---|---|---|
| `x64-Debug` | assert と debug 情報を有効化 | `out/build/x64-Debug` |
| `x64-RelWithDebInfo` | 最適化と debug 情報 | `out/build/x64-RelWithDebInfo` |
| `x64-Release` | 最適化、assert 無効 | `out/build/x64-Release` |

初回は `x64-Debug` を使用します。自動 configure が始まらない場合は `プロジェクト > Configure anet-lab` を実行します。configure 完了時に CMake 出力に error がないことを確認してください。

環境変数や依存 library のパスを修正した場合は、`プロジェクト > キャッシュを削除して再構成`を実行します。

### 4.3 C++ 本体と Runner をビルドする

`ビルド > すべてビルド`を実行します。個別 targetだけをビルドする場合は、Solution Explorerを CMake Targets Viewへ切り替え、対象 targetを右クリックして「ビルド」を選択します。

初回ビルドは libtorch を含むため時間がかかる場合があります。ビルドが完了すると、主な Debug 生成物は次の場所に作成されます。

| 生成物 | 出力先 |
|---|---|
| Runner | `apps/runner/bin/Debug/AnetRLRunner.exe` |
| core test | `core/anet-core/bin/Debug/anet-core-test.exe` |
| LunarLander test | `core/envs/lunarlander1/bin/Debug/LunarLanderEnv-test.exe` |
| ImageCls test | `core/envs/imagecls1/bin/Debug/ImageClsEnv-test.exe` |
| DropMerge test | `core/envs/dropmerge1/bin/Debug/DropMergeEnv-test.exe` |
| Atari test（3.6節で ALE を用意した場合） | `core/envs/atari1/bin/Debug/AtariEnv-test.exe` |

post-build処理により、libtorchとwxWidgetsの実行時DLLも実行ファイルの隣へコピーされます。

### 4.4 テストを実行する

Visual Studio の `テスト` menuから CTest を実行します。すべてを実行する前に対象を絞る場合は、CMake Targets Viewから `anet-core-test` などのtest targetをビルドし、Test Explorerまたは生成された実行ファイルから実行します。

intentional failureを確認する場合を除き、通常の自動検証では失敗時ダイアログを無効にします。AIエージェント向けの具体的な実行方法は [開発・作業規約](../../AGENTS.md)に従います。

### 4.5 Runner を起動する

Visual Studio 上部の startup itemから Debug の `AnetRLRunner.exe` を選択し、`デバッグ > デバッグなしで開始`または `F5` で起動します。

Runner windowが表示されれば、Visual StudioによるC++開発環境の初期構築は完了です。Runnerの設定と操作は [Run 実行ガイド](020_user_guide_run.jp.md)へ進んでください。

## 5. Metrics Viewer のビルドと開発

Metrics Viewer を使わない場合、この章は飛ばして構いません。

### 5.1 bat でビルドして起動する

`apps\metrics-viewer\build.bat` を実行します。起動中の Metrics Viewer がある場合は、先に止めてから実行します。成功すると、次の JAR が生成されます。

```text
apps/metrics-viewer/target/metrics-viewer.jar
```

`apps\22_metrics_viewer_java.bat` を実行すると Metrics Viewer が起動し、ブラウザで `http://localhost:8082` が開きます。起動引数と Run directory の指定は [Run 分析ガイド](030_user_guide_analysis.jp.md)を参照してください。

Metrics Viewer を変更しない場合は、ここで完了です。

> [!NOTE]
> `build.bat` は同じ folderの Maven Wrapper（`mvnw.cmd`）でビルドするため、Maven の導入は不要です。初回は Maven を `%USERPROFILE%\.m2\wrapper` へ取得するため時間がかかります。テストは実行しません。

### 5.2 Eclipse を導入する

Metrics Viewer を変更する場合だけ、5.2節以降を行います。

1. [Eclipse IDE](https://eclipseide.org/)を導入します。
2. Maven integration for Eclipse（m2e）が使用できることを確認します。
3. Eclipse の `Window > Preferences > Java > Installed JREs` で、3.7節で導入した JDK を選択します。一覧にない場合は `Add...` から導入先の folderを追加します。

### 5.3 Maven projectをimportする

1. Eclipseを起動します。
2. `File > Import`を開きます。
3. `Maven > Existing Maven Projects`を選択します。
4. Root Directoryに `apps/metrics-viewer` を指定します。
5. 検出された `pom.xml` を選択してimportします。

`pom.xml` は Java 17 をtargetとします。projectのJREが JDK 17 より古い場合は、project propertiesまたはEclipseのInstalled JREsで3.7節の JDK へ変更します。

### 5.4 Eclipse上でtestする

Projectを右クリックし、`Run As > Maven test`を実行します。JavaScript表示testは、ローカルにMicrosoft Edgeがない場合はskipされます。

testが失敗した場合は、Consoleの最初のerrorと、その直前に実行されたtestを確認します。

### 5.5 Eclipse上でpackageする

Projectを右クリックし、`Run As > Maven build...`を開き、Goalsに次を指定して実行します。

```text
clean package
```

成功すると、次のJARが生成されます。

```text
apps/metrics-viewer/target/metrics-viewer.jar
```

JARが生成されれば完了です。起動は5.1節と同じく `apps\22_metrics_viewer_java.bat` で行います。

## 6. CLI によるビルド・テスト（AI・CI・診断用）

この章は、Visual StudioやEclipseによる通常の開発手順ではありません。AIエージェント、CI相当の検証、IDEの問題の切り分けに使用します。

### 6.1 C++ のconfigureとbuild

人間がCLIでbuildする場合は、Windowsのスタートmenuから次のいずれかを起動します。

- `Developer Command Prompt for VS 2022`
- `Developer PowerShell for VS 2022`

これらのterminalは、MSVC標準header、library、Windows SDKを使うための環境変数を初期化します。起動後、使用するterminalに合わせてrepository rootへ移動します。

Developer Command Promptの場合:

```bat
cd /d C:\dev\anet-lab
```

Developer PowerShellの場合:

```powershell
Set-Location C:\dev\anet-lab
```

その後、次のコマンドでconfigureとbuildを実行します。

```powershell
cmake --preset x64-Debug
cmake --build --preset x64-Debug
```

通常のPowerShellやAIエージェントから実行する場合は、`VsDevCmd.bat`とCMakeを同じ`cmd` process内で実行しても構いません。Visual Studio 2022 Communityの標準配置例:

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --preset x64-Debug'
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```

Editionやインストール先が異なる場合は `VsDevCmd.bat` のパスを読み替えます。

### 6.2 C++ test

CMakeに登録されているtestをまとめて実行します。

```powershell
ctest --preset x64-Debug --output-on-failure
```

対象を絞る場合は、先にtargetをビルドしてから実行ファイルをリポジトリルートで起動します。

```powershell
cmake --build --preset x64-Debug --target anet-core-test
core\anet-core\bin\Debug\anet-core-test.exe
```

同様に `LunarLanderEnv-test`、`ImageClsEnv-test`、`DropMergeEnv-test` を個別実行できます。ALE を用意した場合は `AtariEnv-test` も同様です。AIエージェントはdialog抑止、MSVC初期化、対象を絞ったtest実行について [開発・作業規約](../../AGENTS.md)にも従います。

### 6.3 Metrics Viewer

```powershell
cd apps\metrics-viewer
.\mvnw.cmd -B test
.\mvnw.cmd -B package
```

起動確認の例:

```powershell
java -Xmx1g -jar target\metrics-viewer.jar --server.port=8082
```

## 7. Python 補助ツール

PythonはOptuna harness、Python版Metrics Viewer、TensorBoard bridge、MLflow bridgeなどで使用します。C++本体とJava版Metrics Viewerのビルドだけを行う場合は不要です。

### 7.1 仮想環境

Python packageをuser siteやglobal環境へ混在させず、リポジトリルートの `.venv` を使用します。

```powershell
C:\Python314\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

上のPython pathは例です。別のPythonを使う場合も、作成後は必ず `.\.venv\Scripts\python.exe` を指定してpackageの導入と補助ツールの実行を行います。

```powershell
.\.venv\Scripts\python.exe -m pip install <必要なパッケージ>
.\.venv\Scripts\python.exe apps\runner\tools\dropmerge_optuna.py --help
```

`.venv` はローカル開発環境であり、Git管理対象にしません。

### 7.2 workspace metrics圧縮

完了Runの`metrics.jsonl`をまとめてgzipへ移行する場合は、次のlauncherを起動する。

```powershell
apps\70_compress_workspace_metrics.bat
apps\70_compress_workspace_metrics.bat dm-iqn
apps\70_compress_workspace_metrics.bat dm-iqn --dry-run --no-pause
```

workspace指定を省略すると、`apps/runner/workspaces/`直下で`runs/`を持つ候補から番号選択する。
`[0] EXIT`は固定の終了選択肢である。各workspaceの処理結果を表示した後は、dry-runや取消を含めて
pauseし、キー入力後にworkspace選択へ戻るため、複数workspaceを続けて処理できる。
`--no-pause`指定時はpauseを省略する。workspaceを引数指定した場合は1回で終了する。
preflight後は`YES`で実行、`NO`で取消、`DRY-RUN`で変更なしの確認を選ぶ。
自動実行では`--dry-run`でpromptを省略でき、`--no-pause`で終了時のキー入力待ちを無効化できる。
workspaceを引数指定して1回実行した場合の終了codeは、成功またはblockerなしのdry-runが0、
未処理Runありが1、利用者取消が2である。

### 7.3 MLflow bridge

MLflow bridgeとMLflow serverは、リポジトリルートの `.venv` と選択workspaceの`runs/mlflow.db`を共有します。launcherの第1引数でworkspaceを指定し、省略時はRunnerが保存した`last_workspace.txt`、取得不能時は`_default`を使います。依存packageは次のコマンドで導入します。

```powershell
.\.venv\Scripts\python.exe -m pip install -r viewers\metrics-tools\requirements.txt
```

`requirements.txt` はMLflowを `3.13.0` に固定しています。MLflow 3.14.0のserverはPython 3.14で削除された `importlib.abc.Traversable` をimportするため、この組み合わせでは起動できません。`apps/41_mlflow_bridge.bat` と `apps/42_start_mlflow.bat` は、要求versionが導入されていない場合にfail-fastします。`41_mlflow_bridge.bat`のversion確認はMLflow本体をimportせずpackage metadataを参照し、本体importの開始前に進捗をconsoleへ表示します。

MLflow bridgeは対象Runの `config/config_data.txt` を読み、各 `key = value` をMLflow parameterとして記録します。parameter名で使用できない `[` と `]` は除去し、その他の使用不可文字は `_` へ置換します。変換後のparameter名が衝突する場合は、値を上書きせずfail-fastします。

`apps/41_mlflow_bridge.bat` は選択workspaceの`runs/run_*/metrics.jsonl(.gz)`を列挙し、起動時点で存在するすべての直下RunをMLflowへ変換します。両方がある場合はrawを優先し、監視中に追加された直下Runも自動的に対象へ追加します。rawからgzipへ移行しても同じMLflow Runと展開後byte offsetを継続します。

更新時刻が最も新しいRunは、保存済みoffsetが現在の末尾へ追いつくまで優先して変換します。最新Runの処理中も、10 batchごとに過去Runを1 batch処理し、対象はround-robinで交代します。追いついた後は、他のRunを1 batchずつ変換します。

監視中は、前回表示後にmetricsの処理が進んだRunに限り、最大10秒間隔で最新Runと過去Runを区別して、Run名、処理offset、末尾までの遅延量をconsoleへ表示します。`runs/group/run_*/metrics.jsonl(.gz)` のように別directoryの下へネストされたRunは対象外です。

MLflowの `Status=RUNNING` は学習processの生存状態ではなく、bridgeが継続監視する対象として登録したことを示します。`--once` で変換した場合だけ、現在の末尾まで取り込んだ後に `Status=FINISHED` とします。

## 8. トラブルシューティング

### 8.1 Visual Studio に CMake preset が表示されない

Visual Studio Installerで「C++ CMake tools for Windows」が導入されていることを確認します。その後、リポジトリルートをフォルダーとして開き直します。`CMakePresets.json` を編集した直後は、Visual Studioの再起動またはCMake cacheの再構成が必要な場合があります。

### 8.2 環境変数を変更してもconfigure結果が変わらない

Visual Studioをすべて終了して起動し直し、`プロジェクト > キャッシュを削除して再構成`を実行します。CMake出力の `Final Torch_DIR` が3.8節の設定を指していることを確認します。

### 8.3 `type_traits` などの標準headerが見つからない

CLIのMSVC環境が不完全です。通常のPowerShellから直接 `cmake --build` せず、6.1節のDeveloper Command PromptまたはDeveloper PowerShellを使用してください。自動化では、補足のように `VsDevCmd.bat` とbuildを同じ`cmd` processで実行します。Visual Studio GUIからのbuildでは、選択したMSVC toolsetとWindows SDKを確認します。

### 8.4 CMake が Torch を見つけない

`Torch_DIR_DEBUG` / `Torch_DIR_RELEASE` が、それぞれ対象配布物の `share/cmake/Torch` を指しているか確認します。Debug buildとRelease版libtorchの混在も避けてください。変更後はVisual StudioのCMake cacheを削除して再構成します。

### 8.5 CMake が wxWidgets を見つけない

環境変数 `wxWidgets_DIR` が install treeの `lib/cmake/wxWidgets-<major>.<minor>` を指しているか確認します。directory名は versionを含むため、wxWidgets を更新したときは設定も変更します。

`find_package` は Config modeで解決するため、指定先に `wxWidgetsConfig.cmake` が存在する必要があります。`core`、`base`、`gl`、`aui` componentを含む x64 buildであることも確認します。

環境変数を変更しても解決先が変わらない場合は、次の2つを疑います。

- CMake cacheの `wxWidgets_DIR` が優先される。cache変数は環境変数より優先されるため、Visual Studio の CMake cacheを削除して再構成するか、`-DwxWidgets_DIR=...` で上書きします
- vcpkg に `wxwidgets` が installされている。vcpkg の toolchainは installed treeを `CMAKE_PREFIX_PATH` へ追加し、これも環境変数より優先されます。`vcpkg remove wxwidgets:x64-windows` で削除します

`wx/wx.h` が見つからないというerrorは、libraryの解決先とheaderの解決先が食い違うときに出ます。vcpkg の installed に wxWidgets の headerが残っていると、ZLIB など他の package経由で include search pathへ入り、headerだけが vcpkg 側へ解決されます。

### 8.6 CUDA または DLL の読み込みに失敗する

libtorchのCUDA世代、NVIDIA driver、CUDA Toolkitの互換性を確認します。実行ファイル隣のDLLが古い場合は、対象configurationを再ビルドしてpost-build copyをやり直します。

### 8.7 Maven が Java release 17 を扱えない

EclipseのInstalled JREs、projectのJRE System Library、Maven runtimeがJDK 17以上を参照しているか確認します。CLIでは `java -version` と `apps\metrics-viewer\mvnw.cmd -version` が同じJDKを参照しているか確認し、必要に応じて `JAVA_HOME` と `PATH` を修正します。

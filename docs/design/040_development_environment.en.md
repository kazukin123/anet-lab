<!-- translated-from: 040_development_environment.jp.md blob:9985ac57cd924bede906140f8faa16f094fef04a date:2026-10-07 progress:done -->
# ANET Development Environment Setup Guide

> Primary perspective: from initial setup to a completed IDE build

## 1. Introduction

### 1.1 Purpose

This document guides developers interested in ANET through the steps, in order, to build the C++ core and Runner, and optionally Metrics Viewer.

GUI development with Visual Studio for the C++ core and Runner, and Eclipse for Metrics Viewer, is the standard path. CLI procedures are separated as a supporting path for AI agents, CI-equivalent validation, and diagnosing GUI problems.

### 1.2 Audience

- Developers modifying ANET core, Agents, Envs, or Runner
- Developers modifying Metrics Viewer or Python supporting tools
- Developers or AI agents reproducing CI-equivalent builds and tests locally

### 1.3 Completion Milestones

| Chapter completed | Result |
|---|---|
| Chapter 3 | Initial build prerequisites are installed, including Visual Studio, libtorch, wxWidgets, and CUDA |
| Chapter 4 | The C++ core and Runner can be built in Debug in Visual Studio |
| Chapter 5 | Metrics Viewer can be built and started. When modifying it, it can be built in Eclipse |

If working only with the C++ core and Runner, initial setup ends at Chapter 4. To use Metrics Viewer, continue through Section 5.1; to modify Metrics Viewer, continue to the end of Chapter 5. See the [Run Execution Guide](020_user_guide_run.en.md) for configuring and operating a built Runner, and the [Run Analysis Guide](030_user_guide_analysis.en.md) for analyzing Runs.

## 2. Verified Environment and Recommended IDEs

### 2.1 Verified Environment

The currently verified environment for development and execution of the whole framework combines Windows 11 x64, MSVC, an NVIDIA GPU, and CUDA-enabled libtorch.

| Configuration | Status |
|---|---|
| Windows 11 x64 + NVIDIA CUDA | Verified standard configuration |
| CPU-only | Individual CPU paths exist, but the full setup from development environment to Runner execution is unverified |
| Linux / macOS | Unverified |
| Visual Studio | Verified with Visual Studio Community 2022 and MSVC v143 |
| C++ | C++20 required |
| CMake | Repository minimum is 3.20 |
| Java | Metrics Viewer targets Java 17 |

Select mutually compatible versions of libtorch, CUDA Toolkit, and the NVIDIA driver. Consult [Windows CI](../../.github/workflows/windows-ci.yml) as the authoritative source for versions currently used by repository CI.

### 2.2 Recommended IDE for the C++ Core and Runner

Visual Studio is recommended for coding, CMake configuration, building, and debugging the C++ core and Runner. Visual Studio Community 2022 is the current standard development environment.

Organizations ineligible for Visual Studio Community should use Visual Studio Professional or Enterprise. The Open Folder CMake workflow in this document is the same across editions. Check [Visual Studio Community](https://visualstudio.microsoft.com/vs/community/) for current eligibility terms.

Visual Studio reads `CMakeLists.txt` and `CMakePresets.json` directly from the repository root. This workflow does not generate and open a `.sln`.

### 2.3 Recommended IDE for Metrics Viewer

Eclipse is recommended for coding, Maven builds, and tests for the Java Metrics Viewer, using JDK 17 or later and Maven integration for Eclipse (m2e).

### 2.4 Role of the CLI

Normal human development takes place in Visual Studio or Eclipse. CLI procedures are limited to these uses and collected in Chapter 6:

- Builds and validation by AI agents
- Local reproduction of CI-equivalent procedures
- Diagnosing IDE configuration or build failures

## 3. Initial Setup

### 3.1 Software to Install

The installation units users need to consider are listed below.

| Installation unit | Required for |
|---|---|
| Git for Windows | Cloning the repository and ALE. Unnecessary if `git` already runs from PowerShell |
| Visual Studio Community 2022 or an appropriate edition | C++ core, Envs, Agents, Runner |
| Debug and Release distributions of CUDA-enabled libtorch | C++ core, Runner |
| wxWidgets (built from source) | Runner GUI |
| ALE v0.12.0 (built from source) | Atari env (only when used) |
| NVIDIA driver and CUDA Toolkit | C++ development using CUDA-enabled libtorch |
| JDK 17 or later | Metrics Viewer |
| Eclipse | Modifying Metrics Viewer (only when needed) |
| Python | Supporting tools such as Optuna, Python viewer, and TensorBoard/MLflow bridges |

CMake and Ninja are included in Visual Studio Installer's C++ CMake tools for Windows. Standard Visual Studio development does not require separate installations.

Box2D, Catch2, nlohmann/json, Tracy, and NVTX headers are included in `third_party/`. Add Doxygen, Graphviz, ffmpeg, Tracy server, or NVIDIA Nsight when generating documentation, diagrams/videos, or analyzing performance. They are unnecessary for the first build.

### 3.2 Installing Visual Studio

1. Install Visual Studio Community 2022 using the [Visual Studio Community 2022 Installer](https://aka.ms/vs/17/release/vs_community.exe).
2. Select the Desktop development with C++ workload in Visual Studio Installer.
3. In installation details, ensure at least these components are selected.

| Component | Purpose |
|---|---|
| MSVC v143 - VS 2022 C++ x64/x86 build tools | C++ compiler and linker |
| Windows 11 SDK | Windows headers and libraries |
| C++ CMake tools for Windows | Visual Studio CMake integration, CMake, Ninja |
| vcpkg package manager | Fetching packages used by ALE (zlib, SDL2). Only when using the Atari env |

If Visual Studio is already installed, add components through Modify in Visual Studio Installer. See also [CMake projects in Visual Studio](https://learn.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio) for CMake integration.

### 3.3 Preparing libtorch

Install both Debug and Release libtorch distributions to match the MSVC Debug/Release runtimes. As a baseline, choose the CUDA generation used by CI at the time of the Chapter 4 build.

Current Windows CI uses libtorch `2.12.0+cu130`. Obtain both distributions:

- Release: [`libtorch-win-shared-with-deps-2.12.0+cu130.zip`](https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-2.12.0%2Bcu130.zip)
- Debug: [`libtorch-win-shared-with-deps-debug-2.12.0+cu130.zip`](https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-debug-2.12.0%2Bcu130.zip)

This document uses these extraction locations. Other locations are acceptable; use their actual paths in Section 3.8's environment variables.

```text
C:\dev\libtorch-win-shared-with-deps-2.12.0+cu130\libtorch
C:\dev\libtorch-win-shared-with-deps-debug-2.12.0+cu130\libtorch
```

Verify that each directory contains `share/cmake/Torch/TorchConfig.cmake`.

### 3.4 Preparing wxWidgets

wxWidgets is built and installed from source without vcpkg. Runner GUI uses the wxWidgets `core`, `base`, `gl`, and `aui` components, and `find_package` resolves them in Config mode, so an install tree containing the CMake package config is required. The official pre-built binaries do not include the package config and cannot be used.

Download the source archive from the [wxWidgets release page](https://github.com/wxWidgets/wxWidgets/releases), and extract it to `C:\dev\wxWidgets-3.3.3`. Launch `Developer PowerShell for VS 2022` from the Windows Start menu and run the following.

```powershell
$src = "C:\dev\wxWidgets-3.3.3"
cmake -S $src -B "$src\build" -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_INSTALL_PREFIX="$src\install" -DwxBUILD_SHARED=ON -DwxBUILD_PRECOMP=ON
cmake --build "$src\build" --config Debug --parallel
cmake --install "$src\build" --config Debug
cmake --build "$src\build" --config Release --parallel
cmake --install "$src\build" --config Release
```

Specify the same toolset as the anet-lab main build for the generator. Building with a different toolset results in a mismatched ABI. Debug and Release installed to the same prefix coexist because their library names differ. The build tree can be deleted after installation.

The location of the installed package config is set in the environment variable `wxWidgets_DIR` in Section 3.8.

### 3.5 Preparing the NVIDIA Driver and CUDA Toolkit

To use CUDA-enabled libtorch, install an NVIDIA driver and a [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) compatible with libtorch's CUDA generation. CUDA Toolkit 13.0 is the baseline for `cu130` libtorch.

Use the default `CUDA_PATH` set by CUDA Toolkit Installer. After installation, check the following in a newly opened ordinary PowerShell. `nvidia-smi` is not found in `Developer PowerShell for VS 2022`.

```powershell
nvidia-smi
nvcc --version
$env:CUDA_PATH
```

At minimum, confirm that `nvidia-smi` lists the NVIDIA GPU and driver and that `nvcc --version` reports the selected CUDA Toolkit.

> [!NOTE]
> `Developer PowerShell for VS 2022` launches the 32-bit PowerShell. A 32-bit process sees `C:\Windows\System32` redirected to another folder, so `nvidia-smi` located there is not found. Also, terminals opened before installing CUDA Toolkit do not reflect the `PATH` and `CUDA_PATH` set by the installer.

### 3.6 Preparing ALE for the Atari Env

Skip this section if you will not use the Atari env.

Launch `Developer PowerShell for VS 2022` from the Windows Start menu and run the following.

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

After the build, confirm that the following files exist. Do not delete the build tree.

```text
C:\dev\ALE\build\src\ale\Debug\ale.lib
C:\dev\ALE\build\src\ale\Release\ale.lib
```

Prepare ROMs (`.bin`) and place them in one folder. File names are lowercase snake_case, such as `pong.bin`. Installing ale-py 0.12.0 with pip places them in this form in `site-packages\ale_py\roms`.

When using the Atari env's `display_screen` or `sound`, place `SDL2.dll` in `third_party\runtime_dlls\` of the repository cloned in Section 4.1 and reconfigure. `ale_py\SDL2.dll` from ale-py can be used as `SDL2.dll`.

During the configure in Section 4.2, confirm that CMake output shows `Atari/ALE env enabled`.

> [!NOTE]
> - The Atari env uses ALE (Arcade Learning Environment) v0.12.0 directly from C++. Because ALE is GPL-2.0, it is not included in the repository, and release packages are built without the Atari env. See Chapter 3 of [Atari Env](220_atari_env.en.md) for configuration details.
> - ALE uses the vcpkg bundled with Visual Studio in manifest mode to fetch zlib and SDL2 into its build tree. Developer PowerShell sets the environment variable `VCPKG_ROOT` to the location of the bundled vcpkg. The anet-lab Atari env also uses the zlib in this build tree.
> - ALE is not installed; anet-lab references its build tree directly. This is why the build tree cannot be deleted.
> - anet-lab's configure enables the Atari env automatically when `src\ale\ale_interface.hpp` exists under `ALE_ROOT` (the default value `AUTO` of the CMake variable `ANET_ENABLE_ATARI`). Without ALE, the C++ core and Runner still build, excluding the Atari env.
> - The build works without ROMs, and tests that use ROMs are skipped.
> - DLLs placed in `third_party\runtime_dlls\` are copied to the Runner output directory during the build.

### 3.7 JDK for Metrics Viewer

Skip this section if you will not use Metrics Viewer.

Install a JDK. Amazon Corretto 25 is recommended. Download the Windows x64 `.msi` from the [Corretto 25 downloads page](https://docs.aws.amazon.com/corretto/latest/corretto-25-ug/downloads-list.html) and run it. The default installation location is under `C:\Program Files\Amazon Corretto\`.

The location of the installed JDK is set in the environment variables `JAVA_HOME` and `Path` in Section 3.8.

> [!NOTE]
> - Metrics Viewer targets Java 17 (`maven.compiler.release` in `pom.xml`), so any JDK 17 or later can build it. Corretto 25 is recommended because it has the longest support period among the current LTS releases (until October 2032). Spring Boot 3.5, used by Metrics Viewer, supports up to Java 25.
> - `JAVA_HOME` is used by `build.bat` in Section 5.1, and the `java` found through `Path` is used by the Metrics Viewer launcher bat.

### 3.8 Setting Environment Variables

Open Environment Variables from Edit the system environment variables in Windows, and set these user variables.

| Variable | Example |
|---|---|
| `Torch_DIR_DEBUG` | `C:\dev\libtorch-win-shared-with-deps-debug-2.12.0+cu130\libtorch\share\cmake\Torch` |
| `Torch_DIR_RELEASE` | `C:\dev\libtorch-win-shared-with-deps-2.12.0+cu130\libtorch\share\cmake\Torch` |
| `wxWidgets_DIR` | `C:\dev\wxWidgets-3.3.3\install\lib\cmake\wxWidgets-3.3` |
| `ALE_ROOT` | `C:\dev\ALE` |
| `ATARI_ROM_DIR` | `C:\dev\atari-roms` |
| `JAVA_HOME` | `C:\Program Files\Amazon Corretto\jdk25.0.4_10` |

`wxWidgets_DIR` is the location of the package config installed in Section 3.4. The directory name includes the version.

Set `ALE_ROOT` and `ATARI_ROM_DIR` only when using the Atari env. `ALE_ROOT` is the ALE folder cloned in Section 3.6, and `ATARI_ROM_DIR` is the folder containing the ROMs.

Set `JAVA_HOME` only when using Metrics Viewer. It is the folder of the JDK installed in Section 3.7, and the folder name includes the version. Also add `%JAVA_HOME%\bin` to the user environment variable `Path`.

CUDA Toolkit Installer normally sets `CUDA_PATH`. Set it manually only for a custom installation location.

After adding or changing environment variables, close all running Visual Studio instances and restart them. Visual Studio uses the environment inherited at startup when configuring CMake.

## 4. C++ Development with Visual Studio

Follow this chapter in order through its final verification to complete initial setup for the C++ core and Runner.

### 4.1 Clone and Open the Repository

First clone the following repository locally using any Git tool. If already cloned, skip cloning and open the repository root in Visual Studio.

```text
https://github.com/kazukin123/anet-lab.git
```

Example using Git commands:

```powershell
git clone https://github.com/kazukin123/anet-lab.git C:\dev\anet-lab
```

Example using Visual Studio:

1. Select Clone a repository on the Visual Studio start screen.
2. Enter `https://github.com/kazukin123/anet-lab.git` in Repository location.
3. Set the destination in Path and select Clone.

If you cloned using another Git tool, or are opening an existing clone, select the repository root containing `CMakeLists.txt` and `CMakePresets.json` through `File > Open > Folder` in Visual Studio.

Once Visual Studio opens the repository, Solution Explorer displays files and CMake targets, and CMake configuration and source indexing begin. Check progress and errors in the CMake output of the Output window.

### 4.2 Configure with the Debug Preset

Select `x64-Debug` in the configure preset selector at the top of Visual Studio. Current `CMakePresets.json` provides these presets.

| Preset | Purpose | Output directory |
|---|---|---|
| `x64-Debug` | Enables assertions and debug information | `out/build/x64-Debug` |
| `x64-RelWithDebInfo` | Optimization and debug information | `out/build/x64-RelWithDebInfo` |
| `x64-Release` | Optimization, assertions disabled | `out/build/x64-Release` |

Use `x64-Debug` initially. If automatic configuration does not start, select `Project > Configure anet-lab`. Confirm that the CMake output contains no errors after configuration completes.

After correcting environment variables or dependency paths, select `Project > Delete Cache and Reconfigure`.

### 4.3 Build the C++ Core and Runner

Select `Build > Build All`. To build only a particular target, switch Solution Explorer to CMake Targets View, right-click the target, and select Build.

The first build may take time because it includes libtorch. The main Debug outputs are created at these locations.

| Artifact | Output path |
|---|---|
| Runner | `apps/runner/bin/Debug/AnetRLRunner.exe` |
| Core test | `core/anet-core/bin/Debug/anet-core-test.exe` |
| LunarLander test | `core/envs/lunarlander1/bin/Debug/LunarLanderEnv-test.exe` |
| ImageCls test | `core/envs/imagecls1/bin/Debug/ImageClsEnv-test.exe` |
| DropMerge test | `core/envs/dropmerge1/bin/Debug/DropMergeEnv-test.exe` |
| Atari test (when ALE is prepared in Section 3.6) | `core/envs/atari1/bin/Debug/AtariEnv-test.exe` |

Post-build processing also copies libtorch and wxWidgets runtime DLLs beside the executables.

### 4.4 Run Tests

Run CTest from Visual Studio's Test menu. To narrow the scope before running everything, build a test target such as `anet-core-test` from CMake Targets View, then run it from Test Explorer or its generated executable.

Disable failure dialogs during ordinary automated validation, except when checking intentional failures. AI agents must follow the specific execution instructions in the [Development and Work Rules](../../AGENTS.md).

### 4.5 Start Runner

Select the Debug `AnetRLRunner.exe` from the startup item selector at the top of Visual Studio, then launch with `Debug > Start Without Debugging` or `F5`.

When the Runner window appears, initial setup for C++ development in Visual Studio is complete. Continue to the [Run Execution Guide](020_user_guide_run.en.md) for Runner configuration and operation.

## 5. Building and Developing Metrics Viewer

Skip this chapter if you will not use Metrics Viewer.

### 5.1 Build and Start with bat Files

Run `apps\metrics-viewer\build.bat`. If Metrics Viewer is running, stop it first. On success, this JAR is generated.

```text
apps/metrics-viewer/target/metrics-viewer.jar
```

Running `apps\22_metrics_viewer_java.bat` starts Metrics Viewer and opens `http://localhost:8082` in the browser. See the [Run Analysis Guide](030_user_guide_analysis.en.md) for startup arguments and Run directory selection.

If you will not modify Metrics Viewer, setup is complete here.

> [!NOTE]
> `build.bat` builds with the Maven Wrapper (`mvnw.cmd`) in the same folder, so installing Maven is unnecessary. The first run takes longer because it downloads Maven to `%USERPROFILE%\.m2\wrapper`. Tests are not run.

### 5.2 Install Eclipse

Perform Section 5.2 and later only when modifying Metrics Viewer.

1. Install [Eclipse IDE](https://eclipseide.org/).
2. Confirm that Maven integration for Eclipse (m2e) is available.
3. Under `Window > Preferences > Java > Installed JREs` in Eclipse, select the JDK installed in Section 3.7. If it is not listed, add its installation folder with `Add...`.

### 5.3 Import the Maven Project

1. Start Eclipse.
2. Open `File > Import`.
3. Select `Maven > Existing Maven Projects`.
4. Set Root Directory to `apps/metrics-viewer`.
5. Select the detected `pom.xml` and import it.

`pom.xml` targets Java 17. If the project uses a JRE older than JDK 17, change it to the JDK from Section 3.7 in project properties or Eclipse's Installed JREs.

### 5.4 Test in Eclipse

Right-click the project and select `Run As > Maven test`. JavaScript display tests are skipped if Microsoft Edge is not installed locally.

If tests fail, inspect the first error in Console and the test executed immediately before it.

### 5.5 Package in Eclipse

Right-click the project, open `Run As > Maven build...`, and run with these Goals.

```text
clean package
```

On success, this JAR is generated.

```text
apps/metrics-viewer/target/metrics-viewer.jar
```

Setup is complete once the JAR is generated. Start it with `apps\22_metrics_viewer_java.bat`, as in Section 5.1.

## 6. CLI Builds and Tests (AI, CI, and Diagnostics)

This chapter is for AI agents, CI-equivalent validation, and diagnosing IDE problems, rather than normal Visual Studio or Eclipse development.

### 6.1 C++ Configuration and Build

For manual CLI builds, open either of these from the Windows Start menu:

- `Developer Command Prompt for VS 2022`
- `Developer PowerShell for VS 2022`

These terminals initialize environment variables for MSVC standard headers, libraries, and the Windows SDK. After startup, change to the repository root using the command appropriate to the terminal.

For Developer Command Prompt:

```bat
cd /d C:\dev\anet-lab
```

For Developer PowerShell:

```powershell
Set-Location C:\dev\anet-lab
```

Then configure and build with:

```powershell
cmake --preset x64-Debug
cmake --build --preset x64-Debug
```

From ordinary PowerShell or an AI agent, `VsDevCmd.bat` and CMake can also run in the same `cmd` process. Example for the standard Visual Studio 2022 Community location:

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --preset x64-Debug'
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```

Adjust the `VsDevCmd.bat` path for other editions or installation locations.

### 6.2 C++ Tests

Run all tests registered with CMake:

```powershell
ctest --preset x64-Debug --output-on-failure
```

For a narrower scope, build the target first, then launch its executable from the repository root.

```powershell
cmake --build --preset x64-Debug --target anet-core-test
core\anet-core\bin\Debug\anet-core-test.exe
```

Likewise, `LunarLanderEnv-test`, `ImageClsEnv-test`, and `DropMergeEnv-test` can run individually. When ALE is prepared, `AtariEnv-test` can run the same way. AI agents must also follow [Development and Work Rules](../../AGENTS.md) for dialog suppression, MSVC initialization, and focused test execution.

### 6.3 Metrics Viewer

```powershell
cd apps\metrics-viewer
.\mvnw.cmd -B test
.\mvnw.cmd -B package
```

Example startup check:

```powershell
java -Xmx1g -jar target\metrics-viewer.jar --server.port=8082
```

## 7. Python Supporting Tools

Python is used by the Optuna harness, Python Metrics Viewer, TensorBoard bridge, MLflow bridge, and similar tools. It is unnecessary if only building the C++ core and Java Metrics Viewer.

### 7.1 Virtual Environment

Use `.venv` at the repository root rather than mixing packages into user-site or global environments.

```powershell
C:\Python314\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

The Python path above is an example. Even when creating the environment with another Python, always use `.\.venv\Scripts\python.exe` afterward for package installation and supporting tools.

```powershell
.\.venv\Scripts\python.exe -m pip install <必要なパッケージ>
.\.venv\Scripts\python.exe apps\runner\tools\dropmerge_optuna.py --help
```

`.venv` is a local development environment and must not be tracked by Git.

### 7.2 Workspace Metrics Compression

To migrate completed Runs' `metrics.jsonl` files to gzip in bulk, start this launcher.

```powershell
apps\70_compress_workspace_metrics.bat
apps\70_compress_workspace_metrics.bat dm-iqn
apps\70_compress_workspace_metrics.bat dm-iqn --dry-run --no-pause
```

Without a workspace argument, select a numbered candidate containing `runs/` directly under `apps/runner/workspaces/`. `[0] EXIT` is the fixed exit choice. After displaying each workspace's result, including dry-run or cancellation, the launcher pauses and returns to workspace selection after a keypress, allowing multiple workspaces to be processed. `--no-pause` skips the pause. A workspace argument makes it exit after one pass. After preflight, select `YES` to execute, `NO` to cancel, or `DRY-RUN` to inspect without changes. Automation can skip the prompt with `--dry-run` and disable the final keypress wait with `--no-pause`. For a single pass with an explicit workspace, exit codes are 0 for success or a blocker-free dry-run, 1 for unprocessed Runs, and 2 for user cancellation.

### 7.3 MLflow Bridge

MLflow bridge and server share the repository-root `.venv` and the selected workspace's `runs/mlflow.db`. The launcher's first argument selects the workspace; if omitted, it uses Runner's saved `last_workspace.txt`, falling back to `_default` if unavailable. Install dependencies with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r viewers\metrics-tools\requirements.txt
```

`requirements.txt` pins MLflow to `3.13.0`. The MLflow 3.14.0 server imports `importlib.abc.Traversable`, removed in Python 3.14, so that combination cannot start. `apps/41_mlflow_bridge.bat` and `apps/42_start_mlflow.bat` fail fast if the required version is absent. The version check in `41_mlflow_bridge.bat` uses package metadata without importing MLflow itself and displays console progress before starting the import.

MLflow bridge reads each Run's `config/config_data.txt` and records every `key = value` as an MLflow parameter. It removes unsupported `[` and `]` from parameter names and replaces other unsupported characters with `_`. If transformed names collide, it fails fast instead of overwriting values.

`apps/41_mlflow_bridge.bat` enumerates `runs/run_*/metrics.jsonl(.gz)` in the selected workspace and converts all directly contained Runs existing at startup to MLflow. Raw files take precedence when both exist, and new directly contained Runs are automatically added during monitoring. Migration from raw to gzip preserves the same MLflow Run and decompressed byte offset.

The most recently modified Run is processed preferentially until its saved offset reaches the current end. While processing the latest Run, every 10 batches it processes one batch from an older Run, rotating older Runs round-robin. After catching up, it processes the other Runs one batch at a time.

During monitoring, only Runs whose metrics processing advanced since the previous display produce console updates, at most once every 10 seconds. Updates distinguish latest and older Runs and show Run name, processed offset, and lag to the end. Runs nested under another directory, such as `runs/group/run_*/metrics.jsonl(.gz)`, are excluded.

MLflow `Status=RUNNING` means the bridge registered the Run for continued monitoring, not that its training process is alive. Only `--once` conversion changes it to `Status=FINISHED` after ingesting through the current end.

## 8. Troubleshooting

### 8.1 CMake Presets Missing in Visual Studio

Confirm that C++ CMake tools for Windows is installed through Visual Studio Installer, then reopen the repository root as a folder. After editing `CMakePresets.json`, restarting Visual Studio or reconfiguring the CMake cache may be necessary.

### 8.2 Environment Variable Changes Do Not Affect Configuration

Close all Visual Studio instances, restart, and select `Project > Delete Cache and Reconfigure`. Check that `Final Torch_DIR` in CMake output matches Section 3.8.

### 8.3 Standard Headers Such as `type_traits` Are Missing

The CLI MSVC environment is incomplete. Use the Developer Command Prompt or Developer PowerShell in Section 6.1 rather than running `cmake --build` directly from ordinary PowerShell. For automation, run `VsDevCmd.bat` and the build in the same `cmd` process as described there. For Visual Studio GUI builds, check the selected MSVC toolset and Windows SDK.

### 8.4 CMake Cannot Find Torch

Check that `Torch_DIR_DEBUG` / `Torch_DIR_RELEASE` point to `share/cmake/Torch` in the corresponding distributions. Avoid mixing a Debug build with Release libtorch. After changes, delete Visual Studio's CMake cache and reconfigure.

### 8.5 CMake Cannot Find wxWidgets

Check that the environment variable `wxWidgets_DIR` points to `lib/cmake/wxWidgets-<major>.<minor>` in the install tree. The directory name includes the version, so change the setting when you update wxWidgets.

Because `find_package` resolves in Config mode, `wxWidgetsConfig.cmake` must exist at the specified location. Also verify that it is an x64 build containing the `core`, `base`, `gl`, and `aui` components.

If the resolved location does not change after you change the environment variable, suspect the following two causes.

- The CMake cache's `wxWidgets_DIR` takes precedence. Cache variables take precedence over environment variables, so delete Visual Studio's CMake cache and reconfigure, or override it with `-DwxWidgets_DIR=...`
- `wxwidgets` is installed in vcpkg. The vcpkg toolchain adds the installed tree to `CMAKE_PREFIX_PATH`, and this also takes precedence over the environment variable. Remove it with `vcpkg remove wxwidgets:x64-windows`

An error saying that `wx/wx.h` cannot be found occurs when the library and the headers resolve to different locations. If wxWidgets headers remain in vcpkg's installed tree, they enter the include search path through other packages such as ZLIB, and only the headers resolve to the vcpkg side.

### 8.6 CUDA or DLL Loading Fails

Check compatibility between libtorch's CUDA generation, the NVIDIA driver, and CUDA Toolkit. If DLLs beside the executable are old, rebuild the relevant configuration to rerun post-build copying.

### 8.7 Maven Cannot Handle Java Release 17

Check that Eclipse's Installed JREs, the project's JRE System Library, and Maven runtime point to JDK 17 or later. On the CLI, verify that `java -version` and `apps\metrics-viewer\mvnw.cmd -version` use the same JDK; update `JAVA_HOME` and `PATH` as needed.

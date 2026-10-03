<!-- translated-from: 040_development_environment.jp.md blob:480cdffefde00cd7ee990d73015c16fa1137846c date:2026-09-19 progress:done -->
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
| Chapter 5 | Metrics Viewer can be built in Eclipse |

If modifying only the C++ core and Runner, initial setup ends at Chapter 4. Continue through Chapter 5 to modify Metrics Viewer too. See the [Run Execution Guide](020_user_guide_run.en.md) for configuring and operating a built Runner, and the [Run Analysis Guide](030_user_guide_analysis.en.md) for analyzing Runs.

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

Eclipse is recommended for coding, Maven builds, and tests for the Java Metrics Viewer, using JDK 17 and Maven integration for Eclipse (m2e).

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
| Git for Windows | Cloning the repository and vcpkg. Unnecessary if `git` already runs from PowerShell |
| Visual Studio Community 2022 or an appropriate edition | C++ core, Envs, Agents, Runner |
| Debug and Release distributions of CUDA-enabled libtorch | C++ core, Runner |
| vcpkg and wxWidgets | Runner GUI |
| NVIDIA driver and CUDA Toolkit | C++ development using CUDA-enabled libtorch |
| Eclipse, JDK 17 | Metrics Viewer |
| Python | Supporting tools such as Optuna, Python viewer, and TensorBoard/MLflow bridges |

CMake and Ninja are included in Visual Studio Installer's C++ CMake tools for Windows. Standard Visual Studio development does not require separate installations. Only when running Chapter 6 commands from an ordinary terminal outside Visual Studio must CMake and Ninja be accessible through that terminal's `PATH`.

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

If Visual Studio is already installed, add components through Modify in Visual Studio Installer. See also [CMake projects in Visual Studio](https://learn.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio) for CMake integration.

### 3.3 Preparing libtorch

Install both Debug and Release libtorch distributions to match the MSVC Debug/Release runtimes. As a baseline, choose the CUDA generation used by CI at the time of the Chapter 4 build.

Current Windows CI uses libtorch `2.12.0+cu130`. Obtain both distributions:

- Release: [`libtorch-win-shared-with-deps-2.12.0+cu130.zip`](https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-2.12.0%2Bcu130.zip)
- Debug: [`libtorch-win-shared-with-deps-debug-2.12.0+cu130.zip`](https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-debug-2.12.0%2Bcu130.zip)

This document uses these extraction locations. Other locations are acceptable; use their actual paths in Section 3.6's environment variables.

```text
C:\dev\libtorch-win-shared-with-deps-2.12.0+cu130\libtorch
C:\dev\libtorch-win-shared-with-deps-debug-2.12.0+cu130\libtorch
```

Verify that each directory contains `share/cmake/Torch/TorchConfig.cmake`.

### 3.4 Preparing vcpkg and wxWidgets

Runner GUI uses wxWidgets `core`, `base`, `gl`, and `aui` components. Standard local development uses the vcpkg `x64-windows` triplet.

Run the following in PowerShell.

```powershell
New-Item -ItemType Directory -Force C:\dev
git clone https://github.com/microsoft/vcpkg C:\dev\vcpkg
C:\dev\vcpkg\bootstrap-vcpkg.bat
C:\dev\vcpkg\vcpkg.exe install wxwidgets:x64-windows
```

After installation, confirm that this file exists.

```text
C:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake
```

This repository has no `vcpkg.json`, so package installation and vcpkg placement are managed by the development environment. Manually built and installed wxWidgets is also supported, but vcpkg is recommended for initial setup.

### 3.5 Preparing the NVIDIA Driver and CUDA Toolkit

To use CUDA-enabled libtorch, install an NVIDIA driver and a [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) compatible with libtorch's CUDA generation. CUDA Toolkit 13.0 is the baseline for `cu130` libtorch.

Use the default `CUDA_PATH` set by CUDA Toolkit Installer. After installation, check the following in PowerShell.

```powershell
nvidia-smi
nvcc --version
$env:CUDA_PATH
```

At minimum, confirm that `nvidia-smi` lists the NVIDIA GPU and driver and that `nvcc --version` reports the selected CUDA Toolkit.

### 3.6 Setting Environment Variables

Open Environment Variables from Edit the system environment variables in Windows, and set these user variables.

| Variable | Example |
|---|---|
| `Torch_DIR_DEBUG` | `C:\dev\libtorch-win-shared-with-deps-debug-2.12.0+cu130\libtorch\share\cmake\Torch` |
| `Torch_DIR_RELEASE` | `C:\dev\libtorch-win-shared-with-deps-2.12.0+cu130\libtorch\share\cmake\Torch` |
| `VCPKG_ROOT` | `C:\dev\vcpkg` |
| `CMAKE_TOOLCHAIN_FILE` | `C:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake` |

CUDA Toolkit Installer normally sets `CUDA_PATH`. Set it manually only for a custom installation location.

After adding or changing environment variables, close all running Visual Studio instances and restart them. Visual Studio uses the environment inherited at startup when configuring CMake.

### 3.7 Eclipse and JDK for Metrics Viewer

Skip this section and continue to Chapter 4 if you will not modify Metrics Viewer.

1. Install JDK 17 or later.
2. Install [Eclipse IDE](https://eclipseide.org/).
3. Confirm that Maven integration for Eclipse (m2e) is available.
4. Select JDK 17 under `Window > Preferences > Java > Installed JREs` in Eclipse.

CLI Maven is required for Chapter 6 commands. Standard Eclipse development runs Maven builds through m2e.

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

Post-build processing also copies libtorch and wxWidgets runtime DLLs beside the executables.

### 4.4 Run Tests

Run CTest from Visual Studio's Test menu. To narrow the scope before running everything, build a test target such as `anet-core-test` from CMake Targets View, then run it from Test Explorer or its generated executable.

Disable failure dialogs during ordinary automated validation, except when checking intentional failures. AI agents must follow the specific execution instructions in the [Development and Work Rules](../../AGENTS.md).

### 4.5 Start Runner

Select the Debug `AnetRLRunner.exe` from the startup item selector at the top of Visual Studio, then launch with `Debug > Start Without Debugging` or `F5`.

When the Runner window appears, initial setup for C++ development in Visual Studio is complete. Continue to the [Run Execution Guide](020_user_guide_run.en.md) for Runner configuration and operation.

## 5. Metrics Viewer Development with Eclipse

Skip this chapter if you will not modify Metrics Viewer.

### 5.1 Import the Maven Project

1. Start Eclipse.
2. Open `File > Import`.
3. Select `Maven > Existing Maven Projects`.
4. Set Root Directory to `apps/metrics-viewer`.
5. Select the detected `pom.xml` and import it.

`pom.xml` targets Java 17. If the project uses another JRE, change it to JDK 17 in project properties or Eclipse's Installed JREs.

### 5.2 Test in Eclipse

Right-click the project and select `Run As > Maven test`. JavaScript display tests are skipped if Microsoft Edge is not installed locally.

If tests fail, inspect the first error in Console and the test executed immediately before it.

### 5.3 Package in Eclipse

Right-click the project, open `Run As > Maven build...`, and run with these Goals.

```text
clean package
```

On success, this JAR is generated.

```text
apps/metrics-viewer/target/metrics-viewer.jar
```

JAR generation completes initial Metrics Viewer setup. See the [Run Analysis Guide](030_user_guide_analysis.en.md) for startup arguments and Run directory selection.

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

Likewise, `LunarLanderEnv-test`, `ImageClsEnv-test`, and `DropMergeEnv-test` can run individually. AI agents must also follow [Development and Work Rules](../../AGENTS.md) for dialog suppression, MSVC initialization, and focused test execution.

### 6.3 Metrics Viewer

```powershell
cd apps\metrics-viewer
mvn -B test
mvn -B package
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

Close all Visual Studio instances, restart, and select `Project > Delete Cache and Reconfigure`. Check that `Final Torch_DIR` and the vcpkg toolchain path in CMake output match Section 3.6.

### 8.3 Standard Headers Such as `type_traits` Are Missing

The CLI MSVC environment is incomplete. Use the Developer Command Prompt or Developer PowerShell in Section 6.1 rather than running `cmake --build` directly from ordinary PowerShell. For automation, run `VsDevCmd.bat` and the build in the same `cmd` process as described there. For Visual Studio GUI builds, check the selected MSVC toolset and Windows SDK.

### 8.4 CMake Cannot Find Torch

Check that `Torch_DIR_DEBUG` / `Torch_DIR_RELEASE` point to `share/cmake/Torch` in the corresponding distributions. Avoid mixing a Debug build with Release libtorch. After changes, delete Visual Studio's CMake cache and reconfigure.

### 8.5 CMake Cannot Find wxWidgets

Confirm that `CMAKE_TOOLCHAIN_FILE` points to `C:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake` and that `wxwidgets:x64-windows` is installed. Substitute the actual vcpkg path for other layouts.

For manually installed wxWidgets, set `wxWidgets_DIR` in Config mode or `wxWidgets_ROOT_DIR` in Module mode, and verify an x64 build containing `core`, `base`, `gl`, and `aui` components.

### 8.6 CUDA or DLL Loading Fails

Check compatibility between libtorch's CUDA generation, the NVIDIA driver, and CUDA Toolkit. If DLLs beside the executable are old, rebuild the relevant configuration to rerun post-build copying.

### 8.7 Maven Cannot Handle Java Release 17

Check that Eclipse's Installed JREs, the project's JRE System Library, and Maven runtime point to JDK 17 or later. On the CLI, verify that `java -version` and `mvn -version` use the same JDK; update `JAVA_HOME` and `PATH` as needed.

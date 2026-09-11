@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "DASHBOARD_HOST=0.0.0.0"
set "DASHBOARD_PORT=8088"
set "WORKSPACE_INPUT=%~1"

if not defined WORKSPACE_INPUT (
    echo Usage: %~nx0 ^<workspace_path^>
    exit /b 1
)

call :trim WORKSPACE_INPUT
call :validate "%WORKSPACE_INPUT%"
if errorlevel 1 (
    echo [ERROR] Invalid workspace path: "%WORKSPACE_INPUT%"
    exit /b 1
)

for %%I in ("%~dp0runner") do set "RUNNER_ROOT=%%~fI"
set "IS_ABSOLUTE="
if "%WORKSPACE_INPUT:~1,2%"==":\" set "IS_ABSOLUTE=1"
if "%WORKSPACE_INPUT:~1,2%"==":/" set "IS_ABSOLUTE=1"
if defined IS_ABSOLUTE (
    for %%I in ("%WORKSPACE_INPUT%") do set "WORKSPACE_ROOT=%%~fI"
) else (
    for %%I in ("%RUNNER_ROOT%\workspaces\%WORKSPACE_INPUT%") do set "WORKSPACE_ROOT=%%~fI"
)

if not exist "%WORKSPACE_ROOT%\" (
    echo [ERROR] Workspace directory was not found: "%WORKSPACE_ROOT%"
    exit /b 1
)
set "OPTUNA_DIR=%WORKSPACE_ROOT%\optuna"
set "STORAGE_PATH=%OPTUNA_DIR%\optuna.db"
set "ARTIFACT_DIR=%OPTUNA_DIR%\artifacts"
if not exist "%STORAGE_PATH%" (
    echo [ERROR] Optuna storage was not found: "%STORAGE_PATH%"
    exit /b 1
)
if not exist "%ARTIFACT_DIR%\" (
    echo [ERROR] Optuna artifact directory was not found: "%ARTIFACT_DIR%"
    exit /b 1
)

cd /d "%OPTUNA_DIR%"
echo.
echo Starting Optuna Dashboard...
echo Workspace: %WORKSPACE_ROOT%
echo DB: %STORAGE_PATH%
echo Artifacts: %ARTIFACT_DIR%
echo URL: http://127.0.0.1:%DASHBOARD_PORT%
echo.

start "OptunaDashboard" cmd /k "optuna-dashboard sqlite:///optuna.db --port %DASHBOARD_PORT% --host %DASHBOARD_HOST% --allow-unsafe --artifact-dir artifacts"
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:%DASHBOARD_PORT%"

endlocal
exit /b 0

:trim
setlocal EnableDelayedExpansion
set "TAB=	"
for %%V in (%1) do set "VALUE=!%%V!"
for /f "tokens=*" %%A in ("!VALUE!") do set "VALUE=%%A"
:trim_tail
if "!VALUE:~-1!"==" " set "VALUE=!VALUE:~0,-1!" & goto :trim_tail
if "!VALUE:~-1!"=="!TAB!" set "VALUE=!VALUE:~0,-1!" & goto :trim_tail
endlocal & set "%1=%VALUE%"
exit /b 0

:validate
set "VALUE=%~1"
if not defined VALUE exit /b 1
if not "%VALUE:#=%"=="%VALUE%" exit /b 1
if not "%VALUE://=%"=="%VALUE%" exit /b 1
if "%VALUE:~-1%"==";" exit /b 1
set "NORMALIZED_VALUE=%VALUE:/=\%"
if "%NORMALIZED_VALUE:~0,2%"=="\\" exit /b 1
exit /b 0

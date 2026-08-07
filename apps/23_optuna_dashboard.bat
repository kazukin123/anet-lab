@echo off
setlocal

rem === batファイルの場所を基準に Optuna DB の場所へ移動 ===
set "RUNS_OPTUNA=%~dp0runner\runs_optuna"
set "DASHBOARD_HOST=0.0.0.0"
set "DASHBOARD_PORT=8088"

if not exist "%RUNS_OPTUNA%" (
    mkdir "%RUNS_OPTUNA%"
)

if not exist "%RUNS_OPTUNA%\artifacts" (
    mkdir "%RUNS_OPTUNA%\artifacts"
)

cd /d "%RUNS_OPTUNA%"

if not exist "optuna.db" (
    echo [WARN] optuna.db not found: %RUNS_OPTUNA%\optuna.db
    echo        Start a run-study first, or check the storage path.
    echo.
)

echo.
echo Starting Optuna Dashboard...
echo DB:  %RUNS_OPTUNA%\optuna.db
echo Artifacts: %RUNS_OPTUNA%\artifacts
echo URL: http://127.0.0.1:%DASHBOARD_PORT%
echo.

start "OptunaDashboard" cmd /k "optuna-dashboard sqlite:///optuna.db --port %DASHBOARD_PORT% --host %DASHBOARD_HOST% --allow-unsafe --artifact-dir artifacts"

timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:%DASHBOARD_PORT%"

echo.
echo Optuna Dashboard launched. Press any key to close this window.
rem pause >nul
endlocal
exit /b

@echo off
REM ============================================================
REM tb_bridge 起動スクリプト
REM ============================================================

setlocal

REM ---- 設定 ----
set RUNS_PATH=logs
set PORT=8000
set VENV_PYTHON=..\..\.venv\Scripts\python.exe
set MLFLOW_REQUIREMENTS=..\..\viewers\metrics-tools\requirements.txt

REM ---- 作業ディレクトリをプロジェクトルートへ移動 ----
cd /d "%~dp0"

REM ---- リポジトリの仮想環境を検証 ----
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Python virtual environment was not found: "%VENV_PYTHON%"
    echo [INFO] Create it from the repository root: C:\Python314\python.exe -m venv .venv
    exit /b 1
)

"%VENV_PYTHON%" -c "import mlflow, sys; sys.exit(0 if mlflow.__version__ == '3.13.0' else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Required MLflow version 3.13.0 is not installed in: "%VENV_PYTHON%"
    echo [INFO] Install it with: "%VENV_PYTHON%" -m pip install -r "%MLFLOW_REQUIREMENTS%"
    exit /b 1
)

REM ---- ビューワー起動 ----
pwd
"%VENV_PYTHON%" ..\..\viewers\metrics-tools\mlflow_bridge.py

REM ---- 自動ブラウザオープン ----
REM start http://127.0.0.1:%PORT%

pause
endlocal

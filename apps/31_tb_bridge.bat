@echo off
title %~nx0
REM ============================================================
REM tb_bridge 起動スクリプト
REM ============================================================

setlocal

call "%~dp0tools\resolve_workspace.bat" "%~1"
if errorlevel 1 exit /b 1

REM ---- 設定 ----
set PORT=8050
set VENV_PATH=..\..\.venv

REM ---- 作業ディレクトリをプロジェクトルートへ移動 ----
cd /d "%~dp0runner"

REM ---- 仮想環境が存在すれば有効化 ----
if not exist "%VENV_PATH%\Scripts\python.exe" (
    echo [ERROR] Python virtual environment was not found: "%VENV_PATH%"
    exit /b 1
)

REM ---- ビューワー起動 ----
"%VENV_PATH%\Scripts\python.exe" ..\tools\tb_bridge.py --runsdir "%RUNS_DIR%"

REM ---- 自動ブラウザオープン ----
REM start http://127.0.0.1:%PORT%

pause
endlocal

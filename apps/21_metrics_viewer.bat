@echo off
title %~nx0
REM ============================================================
REM Metrics Viewer 起動スクリプト
REM ============================================================

setlocal

REM ---- 設定 ----
REM set PORT=8050
set VENV_PATH=..\..\.venv

REM ---- 作業ディレクトリをプロジェクトルートへ移動 ----
cd /d "%~dp0runner"

REM ---- 仮想環境が存在すれば有効化 ----
if not exist "%VENV_PATH%\Scripts\python.exe" (
    echo [ERROR] Python virtual environment was not found: "%VENV_PATH%"
    exit /b 1
)

REM ---- ビューワー起動 ----
"%VENV_PATH%\Scripts\python.exe" ..\tools\metrics_viewer.py

REM ---- 自動ブラウザオープン ----
REM start http://127.0.0.1:%PORT%

pause
endlocal

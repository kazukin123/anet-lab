@echo off
REM ============================================================
REM Metrics Viewer 起動スクリプト  for cartpole2
REM ============================================================

setlocal

REM ---- 設定 ----
set RUNS_PATH=logs
REM set PORT=8050
set VENV_PATH=..\..\viewers\metrics-tools\.venv

REM ---- 作業ディレクトリをプロジェクトルートへ移動 ----
cd /d "%~dp0"

REM ---- 仮想環境が存在すれば有効化 ----
if exist "%VENV_PATH%\Scripts\activate.bat" (
    call "%VENV_PATH%\Scripts\activate.bat"
)

REM ---- ビューワー起動 ----
pwd
python ..\..\viewers\metrics-tools\metrics_viewer.py

REM ---- 自動ブラウザオープン ----
REM start http://127.0.0.1:%PORT%

pause
endlocal

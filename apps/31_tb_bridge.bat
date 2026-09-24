@echo off
REM ============================================================
REM tb_bridge 起動スクリプト
REM ============================================================

setlocal

call "%~dp0runner\tools\resolve_workspace.bat" "%~1"
if errorlevel 1 exit /b 1

REM ---- 設定 ----
set PORT=8050
set VENV_PATH=..\..\viewers\metrics-tools\.venv

REM ---- 作業ディレクトリをプロジェクトルートへ移動 ----
cd /d "%~dp0runner"

REM ---- 仮想環境が存在すれば有効化 ----
if exist "%VENV_PATH%\Scripts\activate.bat" (
    call "%VENV_PATH%\Scripts\activate.bat"
)

REM ---- ビューワー起動 ----
pwd
python ..\..\viewers\metrics-tools\tb_bridge.py --runsdir "%RUNS_DIR%"

REM ---- 自動ブラウザオープン ----
REM start http://127.0.0.1:%PORT%

pause
endlocal

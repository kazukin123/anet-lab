@echo off
setlocal enabledelayedexpansion

pushd "%~dp0runner" >nul || exit /b 1
set "RUNS_DIR=%CD%\runs"

if not exist "%RUNS_DIR%" (
    echo [ERROR] No runs directory found: "%RUNS_DIR%"
    popd >nul
    exit /b 1
)

:: 最新のrunフォルダを取得
set "run="
for /f "delims=" %%A in ('dir "%RUNS_DIR%" /b /o:n /ad 2^>nul') do (
    set "run=%%A"
)

if "%run%"=="" (
    echo [ERROR] No run directory found.
    popd >nul
    exit /b 1
)

set "run_dir=%RUNS_DIR%\%run%"
echo RUN: %run%

call :convert_run "%run_dir%"
if errorlevel 1 (
    popd >nul
    exit /b 1
)

echo DONE.

:: 変換が終わったら最新のdot_pngフォルダを自動で開く
explorer "%run_dir%\dot_png"

popd >nul
exit /b


:: ==========================================
:: run配下のdotを階層に依存せずPNGへ変換
:: ==========================================
:convert_run
set "run_dir=%~1"
set "dot_dir=%run_dir%\dot"
set "png_dir=%run_dir%\dot_png"

if not exist "%dot_dir%" (
    echo [INFO] No 'dot' directory found in "%run_dir%".
    exit /b 1
)

set "found_dot="
for /r "%dot_dir%" %%F in (*.dot) do (
    set "found_dot=1"
    call :to_png "%%~fF" "%dot_dir%" "%png_dir%"
)

if not defined found_dot (
    echo [INFO] No .dot files found in "%dot_dir%".
)
exit /b 0


:: ==========================================
:: DOTからPNGへの変換処理
:: ==========================================
:to_png
:: %~1 = 入力dotファイル
:: %~2 = dotルートディレクトリ
:: %~3 = pngルートディレクトリ

set "in_file=%~1"
set "dot_root=%~2"
set "png_root=%~3"
set "file_dir=%~dp1"
set "rel_dir=!file_dir:%dot_root%\=!"
set "out_dir=%png_root%\!rel_dir!"
set "out_file=!out_dir!%~n1.png"

if not exist "!out_dir!" mkdir "!out_dir!"

echo   Converting: !in_file:%dot_root%\=! -^> !out_file:%png_root%\=!
dot -Tpng:cairo -Gdpi=192 "!in_file!" -o "!out_file!"

if errorlevel 1 (
    echo   [ERROR] Failed to convert "%in_file%"
)
exit /b 0

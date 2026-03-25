@echo off
setlocal enabledelayedexpansion

:: 最新のrunフォルダを取得
set "run="
for /f "delims=" %%A in ('dir runs /b /o:n /ad') do (
    set "run=%%A"
)

if "%run%"=="" (
    echo [ERROR] No run directory found.
    exit /b
)
echo RUN: %run%

:: runフォルダ内に移動
cd runs\%run%

:: dotフォルダが存在するか確認
if not exist "dot" (
    echo [INFO] No 'dot' directory found in %run%.
    cd ../..
    exit /b
)

:: dotフォルダ内の各タグ(サブディレクトリ)を走査
for /d %%T in (dot\*) do (
    set "tag_name=%%~nxT"
    echo ========================================
    echo Processing tag: !tag_name!
    echo ========================================

    :: 出力先の dot_png/(tag) フォルダを作成
    if not exist "dot_png\!tag_name!" mkdir "dot_png\!tag_name!"

    :: タグフォルダ内の全ての .dot ファイルを走査
    for %%F in ("%%T\*.dot") do (
        call :to_png "%%F" "!tag_name!" "%%~nF"
    )
)

:: 元のディレクトリに戻る
cd ../..
echo DONE.

:: =========================================================
:: 変換が終わったら最新の dot_png フォルダを自動で開く
:: =========================================================
explorer "runs\%run%\dot_png"

exit /b


:: ==========================================
:: サブルーチン: DOTからPNGへの変換処理
:: ==========================================
:to_png
:: %~1 = 入力ファイルパス 
:: %~2 = タグ名 
:: %~3 = ファイル名拡張子なし 

set "in_file=%~1"
set "out_file=dot_png\%~2\%~3.png"

:: 既にPNGが存在する場合は変換をスキップ (差分更新用)
if exist "%out_file%" (
    echo   Skip: %~3.png already exists.
    exit /b
)

echo   Converting: %~3.dot -^> %~3.png
dot -Tpng "%in_file%" -o "%out_file%"

:: 変換失敗時のエラーハンドリング
if %ERRORLEVEL% neq 0 (
    echo   [ERROR] Failed to convert %~3.dot
)
exit /b
@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Processing ALL runs for DOT to PNG conversion
echo ========================================

:: runs フォルダ配下のすべてのディレクトリを走査
for /d %%R in (runs\*) do (
    set "run_dir=%%R"
    echo.
    echo ----------------------------------------
    echo RUN: !run_dir!
    echo ----------------------------------------

    :: dotフォルダが存在するか確認
    if exist "!run_dir!\dot" (
        
        :: dotフォルダ内の各タグ(サブディレクトリ)を走査
        for /d %%T in ("!run_dir!\dot\*") do (
            set "tag_name=%%~nxT"
            echo   Processing tag: !tag_name!

            :: 出力先の dot_png/(tag) フォルダを作成
            if not exist "!run_dir!\dot_png\!tag_name!" mkdir "!run_dir!\dot_png\!tag_name!"

            :: タグフォルダ内の全ての .dot ファイルを走査
            for %%F in ("%%T\*.dot") do (
                call :to_png "%%F" "!tag_name!" "%%~nF" "!run_dir!"
            )
        )
    ) else (
        echo   [INFO] No 'dot' directory found, skipping.
    )
)

echo.
echo ========================================
echo ALL DONE.
echo ========================================
pause
exit /b


:: ==========================================
:: サブルーチン: DOTからPNGへの変換処理
:: ==========================================
:to_png
:: %~1 = 入力ファイルパス (例: runs\run_name\dot\mcts_tree\file.dot)
:: %~2 = タグ名 (例: mcts_tree)
:: %~3 = ファイル名拡張子なし (例: file)
:: %~4 = runディレクトリパス (例: runs\run_name)

set "in_file=%~1"
set "out_file=%~4\dot_png\%~2\%~3.png"

:: 既にPNGが存在する場合は変換をスキップ (差分更新用)
if exist "%out_file%" (
    rem echo     Skip: %~3.png already exists.
    exit /b
)

echo     Converting: %~3.dot -^> %~3.png
dot -Tpng "%in_file%" -o "%out_file%"

:: 変換失敗時のエラーハンドリング
if %ERRORLEVEL% neq 0 (
    echo     [ERROR] Failed to convert %~3.dot
)
exit /b


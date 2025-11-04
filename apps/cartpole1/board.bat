@echo off
setlocal

:: =========================================
:: 設定
:: =========================================
set BUILD_DIR=out\build\x64-Debug
set INPUT_TSV=%BUILD_DIR%\runs\cart1\scalars.tsv
set OUTPUT_DIR=runs_converted\cart1_tb
set PYTHON=python

echo [1/3] TensorBoardログ変換を開始します...

if not exist "%INPUT_TSV%" (
    echo エラー: ログファイルが見つかりません。
    echo %INPUT_TSV%
    pause
    exit /b 1
)

%PYTHON% visualize_tb.py
if errorlevel 1 (
    echo Pythonスクリプト実行中にエラーが発生しました。
    pause
    exit /b 1
)

echo [2/3] TensorBoardを起動します...
start "" cmd /c "tensorboard --logdir %OUTPUT_DIR% --port 6006"

:: 少し待ってからブラウザを開く
timeout /t 3 >nul

echo [3/3] ブラウザを開きます...
start http://localhost:6006

echo =========================================
echo ? TensorBoard起動完了！
echo ブラウザが開かない場合は手動でアクセス：
echo   http://localhost:6006
echo =========================================

pause
endlocal

@echo off
title %~nx0
setlocal
call "%~dp0tools\resolve_workspace.bat" "%~1"
if errorlevel 1 exit /b 1
cd /d "%~dp0runner"

set "TENSORBOARD_EXE=%~dp0..\.venv\Scripts\tensorboard.exe"
if not exist "%TENSORBOARD_EXE%" (
    echo [ERROR] Python tool was not found: "%TENSORBOARD_EXE%"
    exit /b 1
)

start "" "%TENSORBOARD_EXE%" --logdir "%RUNS_DIR%"
timeout /t 10 /nobreak >nul
start "" "http://localhost:6006/"
endlocal

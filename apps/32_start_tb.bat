@echo off
setlocal
call "%~dp0runner\tools\resolve_workspace.bat" "%~1"
if errorlevel 1 exit /b 1
cd /d "%~dp0runner"

start "" tensorboard --logdir "%RUNS_DIR%"
timeout /t 10 /nobreak >nul
start "" "http://localhost:6006/"
endlocal

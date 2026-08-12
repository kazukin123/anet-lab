@echo off
setlocal
call "%~dp0runner\tools\resolve_workspace.bat" "%~1"
if errorlevel 1 exit /b 1

set "run="
for /f "delims=" %%A in ('dir "%RUNS_DIR%" /b /o:-n /ad 2^>nul') do (
    echo RUN: %%A
    call:each_run "%RUNS_DIR%\%%A"
    echo ==========
)
pause
exit /b

:each_run
pushd "%~1" || exit /b 1
mkdir videos-mp4
for /f "delims=" %%A in ('dir videos /b /o:n') do (
    REM echo   FILE: %%A
    call:to_mp4 %%A
)
popd
exit /b


:to_mp4
echo   %DATE% %TIME% START %*
ffmpeg -loglevel warning -y -r 30 -i videos\%1 -vcodec libx264 -r 30 videos-mp4\%*.mp4 
exit /b


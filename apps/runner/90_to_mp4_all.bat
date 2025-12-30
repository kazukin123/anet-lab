@echo off

set "run="
for /f "delims=" %%A in ('dir runs /b /o:-n') do (
    echo RUN: %%A
    call:each_run "%%A"
    echo ==========
)
pause
exit /b

:each_run
cd runs
cd "%*"
mkdir videos-mp4
for /f "delims=" %%A in ('dir videos /b /o:n') do (
    REM echo   FILE: %%A
    call:to_mp4 %%A
)
cd ..\..
exit /b


:to_mp4
echo   %DATE% %TIME% START %*
ffmpeg -loglevel warning -y -r 30 -i videos\%1 -vcodec libx264 -r 30 videos-mp4\%*.mp4 
exit /b


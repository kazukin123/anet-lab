@echo off

set "run="
for /f "delims=" %%A in ('dir logs /b /o:n') do (
    echo RUN: %%A
    call:each_run %%A
)
pause
exit /b

:each_run
cd logs\%1
mkdir videos-mp4
for /f "delims=" %%A in ('dir videos /b /o:n') do (
    echo FILE: %%A
    call:to_mp4 %%A
)
cd ../..
exit /b


:to_mp4
echo %DATE% %TIME% START %1
ffmpeg -loglevel warning -y -r 30 -i videos\%1 -vcodec libx264 -r 30 videos-mp4\%1.mp4 
exit /b


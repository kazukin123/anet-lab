@echo off

set "run="
for /f "delims=" %%A in ('dir logs /b /o:n') do (
    echo RUN: %run%
    call:each_run %%A
)

:each_run
cd logs\%1
for /f "delims=" %%A in ('dir images /b /o:n') do (
    call:to_mp4 %%A
)
cd ../..
exit /b


:to_mp4
echo %DATE% %TIME% START %1
ffmpeg -loglevel warning -y -r 60 -i images\%1\%1_%%06d.png -vcodec libx264 -pix_fmt yuv420p -r 30 %1.mp4 
exit /b


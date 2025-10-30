@echo off

set "run="
for /f "delims=" %%A in ('dir logs /b /o:n') do (
    set "run=%%A"
)
echo RUN: %run%

cd logs\%run%
for /f "delims=" %%A in ('dir images /b /o:n') do (
    call:to_mp4 %%A
)
cd ../..
exit /b


:to_mp4
echo %DATE% %TIME% START %1
ffmpeg -loglevel warning -y -r 60 -i images\%1\%1_%%06d.png -vcodec libx264 -pix_fmt yuv420p -r 30 %1.mp4 
start %1.mp4 
exit /b


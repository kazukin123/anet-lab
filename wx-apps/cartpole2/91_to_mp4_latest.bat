@echo off

set "run="
for /f "delims=" %%A in ('dir logs /b /o:n') do (
    set "run=%%A"
)
echo RUN: %run%

cd logs\%run%
mkdir videos-mp4
for /f "delims=" %%A in ('dir videos /b /o:n') do (
   call:to_mp4 %%A
)
cd ../..
exit /b


:to_mp4
echo %DATE% %TIME% START %1
echo IN: %1
ffmpeg -loglevel warning -y -r 30 -i videos\%1 -vcodec libx264 -r 30 videos-mp4\%1.mp4 
start videos-mp4\%1.mp4 
exit /b


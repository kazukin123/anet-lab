@echo off
setlocal
call "%~dp0runner\tools\resolve_workspace.bat" "%~1"
if errorlevel 1 exit /b 1

set "run="
for /f "delims=" %%A in ('dir "%RUNS_DIR%" /b /o:n /ad 2^>nul') do (
    set "run=%%A"
)
echo RUN: %run%

if "%run%"=="" (
    echo [ERROR] No run directory found.
    exit /b 1
)
pushd "%RUNS_DIR%\%run%" || exit /b 1
mkdir videos-mp4
for /f "delims=" %%A in ('dir videos /b /o:n') do (
   call:to_mp4 %%A
)
popd
exit /b


:to_mp4
echo %DATE% %TIME% START %1
echo IN: %1
ffmpeg -loglevel warning -y -r 30 -i videos\%1 -vcodec libx264 -r 30 videos-mp4\%1.mp4 
start videos-mp4\%1.mp4 
exit /b


@echo off
cd /d "%~dp0runner"
REM SET EXE="bin\RelWithDebInfo\AnetRLRunner.exe" app.$=app.batchrun
REM SET EXE="bin\RelWithDebInfo\AnetRLRunner.exe"
SET EXE="bin\Release\AnetRLRunner.exe" app.$=app.batchrun
REM SET EXE="bin\Release\AnetRLRunner.exe"


call:run_target_m 32
call:run_target_m 16
call:run_target_m 64
call:run_target_m 8

pause
exit /b


:run_target_m
call:run_exe app.run_name=run_{t}_dm_iqn-k32-n32-m%1-stratified
exit /b


:run_exe
echo %DATE% %TIME% START %*
%EXE% %*
echo   %DATE% %TIME% END   %*
exit /b

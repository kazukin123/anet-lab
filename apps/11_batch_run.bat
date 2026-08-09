@echo off
cd /d "%~dp0runner"
SET EXE="bin\RelWithDebInfo\AnetRLRunner.exe" app.$=app.batchrun
REM SET EXE="bin\Release\AnetRLRunner.exe" app.$=app.batchrun
REM SET EXE="bin\Release\AnetRLRunner.exe"


REM call:run_exe app.run_name=run_{t}_base train.seed=1
REM call:run_exe app.run_name=run_{t}_base train.seed=1
REM call:run_exe app.run_name=run_{t}_base train.seed=1

call:run_exe train.seed=2
call:run_exe train.seed=3

pause
exit /b


:run_exe
echo %DATE% %TIME% START %*
%EXE% %*
echo   %DATE% %TIME% END   %*
exit /b

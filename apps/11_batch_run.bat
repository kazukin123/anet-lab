@echo off
title %~nx0

cd /d "%~dp0runner"

SET "BUILD=Release"
if not exist "bin\%BUILD%\AnetRLRunner.exe" goto :no_exe
copy /Y "bin\%BUILD%\AnetRLRunner.exe" "bin\%BUILD%\AnetRLRunner_ab.exe" >nul
if errorlevel 1 goto :no_exe
if not exist "bin\%BUILD%\AnetRLRunner_ab.exe" goto :no_exe

SET RUNNER="bin\%BUILD%\AnetRLRunner_ab.exe"

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET "EVD=C:/dev/anet-lab/.scratch/evaldiag_noise_tau"

echo === 1. no-skip s2 50M: NoisyNet sample + tau random (train-like, eps 0) ===
call :run_cfg off2_nt
echo === 2. no-skip s2 50M: NoisyNet off + tau fixed (current greedy) ===
call :run_cfg off2_a

if "%FAILED_RUNS%"=="0" goto :all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED, %FAILED_RUNS% FAILED ===
pause
exit /b 1

:all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED ===
pause
exit /b 0


:run_cfg
echo %DATE% %TIME% START %~1
%RUNNER% --config "%EVD%/cases/%~1/run.txt" "app.runs_dir=%EVD%/runs" "app.run_name=run_{t}_evnt_%~1"
SET "RUN_EXIT_CODE=%ERRORLEVEL%"
if "%RUN_EXIT_CODE%"=="0" goto :run_succeeded
echo %DATE% %TIME% [ERROR] RUN FAILED exit_code=%RUN_EXIT_CODE% case=%~1
SET /A FAILED_RUNS+=1
exit /b 0

:run_succeeded
SET /A SUCCEEDED_RUNS+=1
echo   %DATE% %TIME% END   %~1
exit /b 0

:no_exe
echo *** bin\%BUILD%\AnetRLRunner.exe not found or copy failed. Nothing was run.
pause
exit /b 1

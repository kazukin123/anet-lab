@echo off

cd /d "%~dp0runner"

SET "BUILD=RelWithDebInfo"
REM SET "BUILD=Release"

if not exist "bin\%BUILD%\AnetRLRunner.exe" goto :no_exe
copy /Y "bin\%BUILD%\AnetRLRunner.exe" "bin\%BUILD%\AnetRLRunner_ab.exe" >nul
if errorlevel 1 goto :no_exe

SET EXE="bin\%BUILD%\AnetRLRunner_ab.exe" --workspace atari-2nd

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET "FIX1=backend.$=backend.@non-deterministic"
SET "FIX2=E1.game=breakout"

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex"
SET "EV=run.@evalN10"
SET "BM=run.@breakout_metrics"

echo === 0. wiring check - ge432 metrics, train side (2 min) ===
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_soft008>%BM%>%EV%>run.@pl_check"

echo === 0b. wiring check - ge432 metrics, eval side (7 min) ===
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_soft008>%BM%>%EV%>run.@eval_smoke"

echo === 1. soft tau 0.008 - BASELINE, seed 1, 50M (2.9h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_soft008>%BM%>%EV%"

echo === 2. soft tau 0.008 at 100M - budget extension (5.8h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_soft008>run.@to_100m>%BM%>%EV%"

if "%FAILED_RUNS%"=="0" goto :all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED, %FAILED_RUNS% FAILED ===
pause
exit /b 1

:all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED, 0 FAILED ===
pause
exit /b 0


:run_exe
echo %DATE% %TIME% START %*
%EXE% %* %FIX1% %FIX2%
SET "RUN_EXIT_CODE=%ERRORLEVEL%"
if "%RUN_EXIT_CODE%"=="0" goto :run_succeeded
echo %DATE% %TIME% [ERROR] RUN FAILED exit_code=%RUN_EXIT_CODE% args=%*
SET /A FAILED_RUNS+=1
exit /b 0

:run_succeeded
SET /A SUCCEEDED_RUNS+=1
echo   %DATE% %TIME% END   %*
exit /b 0


:no_exe
echo *** bin\%BUILD%\AnetRLRunner.exe not found or copy failed. Nothing was run.
pause
exit /b 1

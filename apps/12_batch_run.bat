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

echo === 0. wiring check x5 (15 min) ===
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_soft020>%EV%>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_soft050>%EV%>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_btr_default>%EV%>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_soft008_btr>%EV%>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_btrstruct_taurelu>run.@seed2>%EV%>run.@pl_check"

echo === 1. soft tau 0.02 - sweep, 50M (2.9h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_soft020>%EV%"

echo === 2. soft tau 0.05 - sweep upper end, 50M (2.9h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_soft050>%EV%"

echo === 3. BTR NN full match - btrstruct + tau ReLU + default init, 50M (2.9h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_btr_default>%EV%"

echo === 4. soft tau 0.008 + BTR structure - does tau mask structure, 50M (2.9h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_soft008_btr>%EV%"

echo === 5. btrstruct + tau ReLU, SEED 2 - replicate and first seed variance, 50M (2.9h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_btrstruct_taurelu>run.@seed2>%EV%"

echo === 6. soft tau 0.008, SEED 2 - replicate of the best score, 50M (2.9h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_soft008>run.@seed2>%EV%"

echo === 7. soft tau 0.008 at 100M - budget extension (5.8h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_soft008>run.@to_100m>%EV%"

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

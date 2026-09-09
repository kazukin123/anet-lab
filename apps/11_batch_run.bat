@echo off

cd /d "%~dp0runner"

SET "BUILD=RelWithDebInfo"
REM SET "BUILD=Release"

if not exist "bin\%BUILD%\AnetRLRunner.exe" goto :no_exe
copy /Y "bin\%BUILD%\AnetRLRunner.exe" "bin\%BUILD%\AnetRLRunner_ab.exe" >nul
if errorlevel 1 goto :no_exe

SET EXE="bin\%BUILD%\AnetRLRunner_ab.exe" --workspace atari-3rd

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET "BK=backend.$=backend.@non-deterministic"
SET "FIX2=E1.game=breakout"

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base"
SET "EV=run.@evalN10"
SET "RR4=run.@hard500>run.@rr4>run.@munch"

echo === 0. wiring: dp010 / alpha06 / wd001 (100k) ===
call:run_exe "run.$=%A5%>%RR4%>run.@dp010>%EV%>run.@pl_check" "app.run_name=run_{t}_tmp_wiring_dp010"
call:run_exe "run.$=%A5%>%RR4%>run.@alpha06>%EV%>run.@pl_check" "app.run_name=run_{t}_tmp_wiring_alpha06"
call:run_exe "run.$=%A5%>%RR4%>run.@wd001>%EV%>run.@pl_check" "app.run_name=run_{t}_tmp_wiring_wd001"

echo === 1. RR4 + DropPath 0.1 50M (8h) ===
call:run_exe "run.$=%A5%>%RR4%>run.@dp010>%EV%" "app.run_name=run_{t}_rr4_dp010_munch"

echo === 2. RR4 + per_alpha 0.6 50M (8h) ===
call:run_exe "run.$=%A5%>%RR4%>run.@alpha06>%EV%" "app.run_name=run_{t}_rr4_alpha06_munch"

echo === 3. RR4 + weight_decay 0.01 50M (8h) ===
call:run_exe "run.$=%A5%>%RR4%>run.@wd001>%EV%" "app.run_name=run_{t}_rr4_wd001_munch"

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
%EXE% %* %BK% %FIX2%
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

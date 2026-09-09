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
SET "RR4B512=run.@hard250>run.@rr4>run.@b512>run.@munch"
SET "RR4B1024=run.@hard125>run.@rr4>run.@b1024>run.@munch"
SET "RR4A06=run.@hard500>run.@rr4>run.@munch>run.@alpha06"

echo === 0. wiring: rr4+b512 / rr4+b1024 / rr4+alpha06 (100k) ===
call:run_exe "run.$=%A5%>%RR4B512%>%EV%>run.@pl_check" "app.run_name=run_{t}_tmp_wiring_rr4b512"
call:run_exe "run.$=%A5%>%RR4B1024%>%EV%>run.@pl_check" "app.run_name=run_{t}_tmp_wiring_rr4b1024"
call:run_exe "run.$=%A5%>%RR4A06%>%EV%>run.@pl_check" "app.run_name=run_{t}_tmp_wiring_rr4a06"

echo === 1. RR4 + B512 50M (8h) ===
call:run_exe "run.$=%A5%>%RR4B512%>%EV%" "app.run_name=run_{t}_rr4_b512_munch"

echo === 2. RR4 + per_alpha 0.6 50M (8h) ===
call:run_exe "run.$=%A5%>%RR4A06%>%EV%" "app.run_name=run_{t}_rr4_alpha06_munch"

echo === 3. RR4 + B1024 50M (8h) ===
call:run_exe "run.$=%A5%>%RR4B1024%>%EV%" "app.run_name=run_{t}_rr4_b1024_munch"

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

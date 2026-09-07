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
SET "ARM=run.@hard125>run.@munch"
SET "UQE=A3.train_policy.policy_type=UQE"
SET "WS=workspaces/atari-3rd/runs"
SET "CK100=A3.auto_load_file=workspaces/atari-2nd/runs/run_20260906-161947_hard125_munch_resume50m/agent_close.anet"

SET "RR4=run.@hard500>run.@rr4>run.@munch"

echo === 0. wiring: %RR4% (400k) ===
call:run_exe "run.$=%A5%>%RR4%>%EV%>run.@to_400k" "app.run_name=run_{t}_tmp_wiring_rr4"

echo === 1. %RR4% 50M (8-9h) ===
call:run_exe "run.$=%A5%>%RR4%>%EV%" "app.run_name=run_{t}_rr4_munch"

echo === 2. baseline r3: %ARM% 50M (2.8h) ===
call:run_exe "run.$=%A5%>%ARM%>%EV%" "app.run_name=run_{t}_munch_r3"

echo === 3. baseline r4: %ARM% 50M (2.8h) ===
call:run_exe "run.$=%A5%>%ARM%>%EV%" "app.run_name=run_{t}_munch_r4"

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

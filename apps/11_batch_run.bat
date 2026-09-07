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
SET "RR4=run.@hard500>run.@rr4>run.@munch"
SET "ARM=run.@hard125>run.@munch"
SET "EV0=run.@evalonly>run.@greedy_eval>run.@to_50"
SET "EV1=run.@evalonly>run.@to_50"

SET "CK_RR4=A3.auto_load_file=workspaces/atari-3rd/runs/run_20260907-121338_rr4_munch/agent_close.anet"
SET "CK_RR1R1=A3.auto_load_file=workspaces/atari-2nd/runs/run_20260905-221731_mu1_hard125_munch_breakout/agent_close.anet"

echo === 1. RR4 50M eps=0 (18min) ===
call:run_exe "run.$=%A5%>%RR4%>%EV0%" "%CK_RR4%" "app.run_name=run_{t}_ev_rr4_50m_greedy"

echo === 2. RR4 50M eps=0.01 (18min) ===
call:run_exe "run.$=%A5%>%RR4%>%EV1%" "%CK_RR4%" "app.run_name=run_{t}_ev_rr4_50m_eps001"

echo === 3. RR1 r1 50M eps=0 (18min) ===
call:run_exe "run.$=%A5%>%ARM%>%EV0%" "%CK_RR1R1%" "app.run_name=run_{t}_ev_rr1r1_50m_greedy"

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

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

SET "BK=backend.$=backend.@non-deterministic"
SET "FIX2=E1.game=breakout"

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base"
SET "ARM=run.@hard125>run.@munch"
SET "EO=run.@evalonly>run.@to_50"
SET "CK100=A3.auto_load_file=workspaces/atari-2nd/runs/run_20260906-161947_hard125_munch_resume50m/agent_close.anet"
SET "CK50=A3.auto_load_file=workspaces/atari-2nd/runs/run_20260906-034637_mu1_hard125_munch_breakout/agent_close.anet"

echo === 0. NoLearn check: 100M ckpt, eps=0.01 (vs run_20260906-205239) ===
call:run_exe "run.$=%A5%>%ARM%>%EO%" "%CK100%" "app.run_name=run_{t}_nolearn_100m_eps001"

echo === 1. NoLearn check: 100M ckpt, eps=0 (vs run_20260906-200907) ===
call:run_exe "run.$=%A5%>%ARM%>%EO%>run.@greedy_eval" "%CK100%" "app.run_name=run_{t}_nolearn_100m_greedy"

echo === 2. 50M ckpt, eps=0 ===
call:run_exe "run.$=%A5%>%ARM%>%EO%>run.@greedy_eval" "%CK50%" "app.run_name=run_{t}_nolearn_50m_greedy"

echo === 3. 50M ckpt, eps=0.01 ===
call:run_exe "run.$=%A5%>%ARM%>%EO%" "%CK50%" "app.run_name=run_{t}_nolearn_50m_eps001"

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

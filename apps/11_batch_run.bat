@echo off

cd /d "%~dp0runner"

SET "BUILD=RelWithDebInfo"

:waitprev
tasklist /FI "IMAGENAME eq AnetRLRunner_ab.exe" 2>nul | find /I "AnetRLRunner_ab.exe" >nul
if not errorlevel 1 (
  ping -n 31 127.0.0.1 >nul
  goto :waitprev
)

if not exist "bin\%BUILD%\AnetRLRunner.exe" goto :no_exe
copy /Y "bin\%BUILD%\AnetRLRunner.exe" "bin\%BUILD%\AnetRLRunner_ab.exe" >nul
if errorlevel 1 goto :no_exe

SET RUNNER="bin\%BUILD%\AnetRLRunner_ab.exe"

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base"
SET "RR1=run.@hard125>run.@munch"
SET "RF=run.@rfit>run.@a5_metrics"
SET "BASE=run.$=%A5%>%RR1%>%RF%>run.@cap2m>run.@eval2ch_r1"
SET "ONCE=run.eval_schedule.[eval_target].interval=30 run.eval_schedule.[eval].interval=31 run.eval_schedule.[greedy_dist].interval=32 run.eval.[eval_target].eval_episodes=16"
SET "KFM=A3.auto_load_file=workspaces/AtariTube-01/runs/run_20260923-095708_atari_kung_fu_master/agent_close.anet"
SET "QB=A3.auto_load_file=workspaces/atari5-01/runs/run_20260923-163113_a5_qbert_rr1_100m_seed2/agent_close.anet"

SET "WS=--workspace atari5-01"
echo === 1. tmp ram: kung_fu_master seed2 eval ===
call :run_exe "%BASE%>run.@evalonly>run.@to_50>run.@batch" %ONCE% E1.game=kung_fu_master "%KFM%" "app.run_name=run_{t}_tmp_ram_kfm_eval"
echo === 2. tmp ram: qbert seed2 eval ===
call :run_exe "%BASE%>run.@evalonly>run.@to_50>run.@batch" %ONCE% E1.game=qbert "%QB%" "app.run_name=run_{t}_tmp_ram_qbert_eval"
echo === 3. tmp ram: qbert seed2 train 400k ===
call :run_exe "%BASE%>run.@to_400k>run.@batch" E1.game=qbert "%QB%" "app.run_name=run_{t}_tmp_ram_qbert_train"

if "%FAILED_RUNS%"=="0" goto :all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED, %FAILED_RUNS% FAILED ===
pause
exit /b 1

:all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED ===
pause
exit /b 0


:run_exe
echo %DATE% %TIME% START %WS% %*
%RUNNER% %WS% %*
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

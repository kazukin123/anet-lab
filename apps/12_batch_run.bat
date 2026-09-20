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

SET "FIX2=E1.game=breakout"

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base"
SET "RR4=run.@hard500>run.@rr4>run.@munch"
SET "RR1=run.@hard125>run.@munch"
SET "RF=run.@rfit"
SET "NET12=run.@btrnet>run.@btrsn12"
SET "NETVIT=run.@btrnet>run.@vitbtr"
SET "NETP4=run.@btrnet>run.@btrpool4"
SET "EV=run.@evalonly>run.@eval2only>run.@greedy_eval>run.@to_50"

SET "S_VIT=workspaces/atari-04/runs/run_20260918-123929_rr4_vitbtr_envs128_cap2m_50m/agent_close.anet"
SET "S_CAP4M=workspaces/atari-04/runs/run_20260919-093925_rr4_btrsn12_envs128_cap4m_50m/agent_close.anet"
SET "S_CAP15=workspaces/atari-04/runs/run_20260919-190647_rr4_btrsn12_envs128_cap1536k_50m/agent_close.anet"

SET "WS=--workspace atari-04"
echo === 1. ev: vitbtr envs128 cap2M 50M eps=0 (25m) ===
call :run_exe "run.$=%A5%>run.@munch>%NETVIT%>%EV%" "A3.auto_load_file=%S_VIT%" "app.run_name=run_{t}_ev_vitbtr_envs128_cap2m_50m"
echo === 2. ev: sn12 envs128 cap4M lane 32768 50M eps=0 (25m) ===
call :run_exe "run.$=%A5%>run.@munch>%NET12%>%EV%" "A3.auto_load_file=%S_CAP4M%" "app.run_name=run_{t}_ev_sn12_envs128_cap4m_50m"
echo === 3. ev: sn12 envs128 cap1536k lane 12288 50M eps=0 (25m) ===
call :run_exe "run.$=%A5%>run.@munch>%NET12%>%EV%" "A3.auto_load_file=%S_CAP15%" "app.run_name=run_{t}_ev_sn12_envs128_cap1536k_50m"
echo === 4. wiring: btrpool4 iqn1024 100k (2m) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>%NETP4%>run.@cap2m>run.@pl_check"
echo === 5. walltime lane: RR1 kyu-net cap2M 100M (6h) ===
call :run_exe "run.$=%A5%>%RR1%>%RF%>run.@cap2m>run.@to_100m>run.@eval2ch_r1" "app.run_name=run_{t}_rr1_va_hard125_cap2m_100m"
echo === 6. pool: RR4 btrpool4 envs128 cap2M 50M (8h) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>%NETP4%>run.@cap2m>run.@eval2ch" "app.run_name=run_{t}_rr4_btrpool4_envs128_cap2m_50m"
echo === 7. lane floor: RR4 sn12 envs128 cap512k lane 4096 50M (8h) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>%NET12%>run.@cap512k>run.@eval2ch" "app.run_name=run_{t}_rr4_btrsn12_envs128_cap512k_50m"

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
%RUNNER% %WS% %* %FIX2%
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

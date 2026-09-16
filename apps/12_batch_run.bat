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

SET EXE="bin\%BUILD%\AnetRLRunner_ab.exe" --workspace atari-3rd

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET "FIX2=E1.game=breakout"

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base"
SET "RR4=run.@hard500>run.@rr4>run.@munch"
SET "RF=run.@rfit"
SET "NET=run.@btrnet>run.@btrsn"
SET "EV=run.@evalonly>run.@eval2only>run.@greedy_eval>run.@to_50"

SET "S_R2=workspaces/atari-3rd/runs/run_20260914-143610_rr4_btrfull_r2_20m/agent_close.anet"
SET "S_SN12=workspaces/atari-3rd/runs/run_20260915-155133_rr4_btrfull_sn12_20m/agent_close.anet"
SET "S_AAI=workspaces/atari-3rd/runs/run_20260915-114835_rr4_btrfull_aainit_20m/agent_close.anet"
SET "S_NOSN=workspaces/atari-3rd/runs/run_20260913-025232_rr4_btrnosn_20m/agent_close.anet"

echo === 1. ev: btrfull r2 20M eps=0 (12m) ===
call :run_exe "run.$=%A5%>run.@munch>%NET%>%EV%" "A3.auto_load_file=%S_R2%" "app.run_name=run_{t}_ev_btrfull_r2_20m"
echo === 2. ev: sn12 20M eps=0 (12m) ===
call :run_exe "run.$=%A5%>run.@munch>run.@btrnet>run.@btrsn12>%EV%" "A3.auto_load_file=%S_SN12%" "app.run_name=run_{t}_ev_btrfull_sn12_20m"
echo === 3. ev: aainit 20M eps=0 (12m) ===
call :run_exe "run.$=%A5%>run.@munch>%NET%>%EV%" "A3.auto_load_file=%S_AAI%" "app.run_name=run_{t}_ev_btrfull_aainit_20m"
echo === 4. ev: btrnosn 20M eps=0 (12m) ===
call :run_exe "run.$=%A5%>run.@munch>run.@btrnet>%EV%" "A3.auto_load_file=%S_NOSN%" "app.run_name=run_{t}_ev_btrnosn_20m"
echo === 5. envs128 + replay_capacity 2M (lane 16,384) 50M (11h) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>%NET%>run.@cap2m>run.@btreval_r4" "app.run_name=run_{t}_rr4_btrfull_envs128_cap2m_50m"
echo === 6. walltime preset: RR1 + kyu-net 100M (5-6h) ===
call :run_exe "run.$=%A5%>run.@hard125>run.@munch>%RF%>run.@to_100m>run.@btreval_r1" "app.run_name=run_{t}_rr1_va_hard125_100m"

if "%FAILED_RUNS%"=="0" goto :all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED, %FAILED_RUNS% FAILED ===
pause
exit /b 1

:all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED ===
pause
exit /b 0


:run_exe
echo %DATE% %TIME% START %*
%EXE% %* %FIX2%
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

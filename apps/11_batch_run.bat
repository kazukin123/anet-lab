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
SET "TP=run.@evaloff>run.@to_10m"

SET "S_CAP2M=workspaces/atari-3rd/runs/run_20260915-203706_rr4_btrfull_envs128_cap2m_50m/agent_close.anet"

echo === 1. ev: envs128 cap2M 50M eps=0 (12m) ===
call :run_exe "run.$=%A5%>run.@munch>%NET%>%EV%" "A3.auto_load_file=%S_CAP2M%" "app.run_name=run_{t}_ev_btrfull_envs128_cap2m_50m"
echo === 2. throughput: RR1 kyu-net, eval off, 10M (0.5h) ===
call :run_exe "run.$=%A5%>run.@hard125>run.@munch>%RF%>%TP%" "app.run_name=run_{t}_tp_rr1_va_evaloff_10m"
echo === 3. throughput: RR4 btrfull SN 15/15, eval off, 10M (1.5h) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>%NET%>run.@envs64>%TP%" "app.run_name=run_{t}_tp_rr4_btrfull_evaloff_10m"
echo === 4. throughput: RR4 SN 12/15, eval off, 10M (1.5h) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>run.@btrnet>run.@btrsn12>run.@envs64>%TP%" "app.run_name=run_{t}_tp_rr4_sn12_evaloff_10m"
echo === 5. RR4 + BTR net + SN, envs64 + cap2M (lane 32,768) 50M (10h) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>%NET%>run.@cap2m>run.@envs64>run.@btreval_r4" "app.run_name=run_{t}_rr4_btrfull_envs64_cap2m_50m"

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

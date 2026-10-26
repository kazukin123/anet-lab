@echo off

cd /d "%~dp0runner"

SET "BUILD=Release"

:waitprev
tasklist /FI "IMAGENAME eq AnetRLRunner*" 2>nul | find /I "AnetRLRunner" >nul
if not errorlevel 1 (
  ping -n 31 127.0.0.1 >nul
  goto :waitprev
)
REM Another bat may start its next stage within a second; confirm that no runner appears for 15 s.
ping -n 16 127.0.0.1 >nul
tasklist /FI "IMAGENAME eq AnetRLRunner*" 2>nul | find /I "AnetRLRunner" >nul
if not errorlevel 1 goto :waitprev

if not exist "bin\%BUILD%\AnetRLRunner_ab.exe" goto :no_exe
if not exist "bin\%BUILD%\AnetRLRunner.exe" goto :no_exe

SET RUNNER="bin\%BUILD%\AnetRLRunner_ab.exe"

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET RUNNER="bin\%BUILD%\AnetRLRunner.exe"
SET "WS=--workspace Atari5-01"
SET "RUNS=workspaces\Atari5-01\runs"
SET "RTOFF=run.$=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base>run.@hard125>run.@munch>run.@rfit>run.@a5_metrics>run.@cap2m>run.@eval2ch_r1>run.@eval2ch_r1_50m>run.@batch"
SET "RTSKIP=run.$=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base>run.@hard125>run.@munch>run.@rfit>run.@a5_metrics>run.@cap2m>run.@eval2ch_r1>run.@eval2ch_r1_50m>run.@skip>run.@batch"
SET "B10M=app.batchrun.exp_exit_step=10000000"
SET "SKIPSTOP=DefaultDQNAgent.learner.skip.feature_gradient=stop"
SET "SKIPIN=DefaultDQNAgent.skip.network_mode=action_in"
SET "SKIPINBIND=DefaultDQNAgent.skip.net.branch.[skip_feature].bind=skip_context, action"
:wait_disk_a0
SET "FREE_GB=0"
for /f %%f in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive C).Free/1GB)"') do SET "FREE_GB=%%f"
if %FREE_GB% LSS 15 (
  echo %DATE% %TIME% waiting for disk space: C: has %FREE_GB% GB free, need 15 GB
  ping -n 301 127.0.0.1 >nul
  goto :wait_disk_a0
)
echo === t98 stage A: qbert realtime profile 10M, off / sep-merge / sep-stop / in-stop, seeds 1-2 ===
for %%s in (1 2) do (
  call :run_exe "%RTOFF%" E1.game=qbert run.seed=%%s %B10M% "app.run_name=run_{t}_a5_qb_rt_t98off_s%%s"
  call :run_exe "%RTSKIP%" E1.game=qbert run.seed=%%s %B10M% "app.run_name=run_{t}_a5_qb_rt_t98mrg_s%%s"
  call :run_exe "%RTSKIP%" E1.game=qbert run.seed=%%s %B10M% %SKIPSTOP% "app.run_name=run_{t}_a5_qb_rt_t98sep_s%%s"
  call :run_exe "%RTSKIP%" E1.game=qbert run.seed=%%s %B10M% %SKIPSTOP% %SKIPIN% "%SKIPINBIND%" "app.run_name=run_{t}_a5_qb_rt_t98in_s%%s"
)

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

:find_ckpt
SET "CKPT="
SET "CKPT_DIR="
if not "%RUN_EXIT_CODE%"=="0" exit /b 0
for /f "delims=" %%d in ('dir /b /ad /o-n "%RUNS%\%~1" 2^>nul') do if not defined CKPT_DIR SET "CKPT_DIR=%%d"
if not defined CKPT_DIR goto :find_ckpt_missing
if exist "%RUNS%\%CKPT_DIR%\agent_close.anet" SET "CKPT=%RUNS:\=/%/%CKPT_DIR%/agent_close.anet"
if defined CKPT exit /b 0
:find_ckpt_missing
echo %DATE% %TIME% [ERROR] agent_close.anet not found: %RUNS%\%~1 newest=%CKPT_DIR%
SET /A FAILED_RUNS+=1
exit /b 0

:no_exe
echo *** bin\%BUILD%\AnetRLRunner_ab.exe or AnetRLRunner.exe not found. Nothing was run.
pause
exit /b 1

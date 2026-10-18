@echo off

cd /d "%~dp0runner"

SET "BUILD=Release"

:waitprev
tasklist /FI "IMAGENAME eq AnetRLRunner_ab.exe" 2>nul | find /I "AnetRLRunner_ab.exe" >nul
if not errorlevel 1 (
  ping -n 31 127.0.0.1 >nul
  goto :waitprev
)

if not exist "bin\%BUILD%\AnetRLRunner_ab.exe" goto :no_exe

SET RUNNER="bin\%BUILD%\AnetRLRunner_ab.exe"

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base"
SET "RR4=run.@hard500>run.@rr4>run.@munch"
SET "RF=run.@rfit>run.@a5_metrics"
SET "SE=run.$=%A5%>%RR4%>%RF%>run.@btrnet>run.@btrsn12>run.@cap2m>run.@eval2ch>run.@batch"

SET "WS=--workspace atari5-01"
SET "RUNS=workspaces\atari5-01\runs"
SET "EPS005=A2.actor.[train].policy.eps_start=0.05 A2.actor.[train].policy.eps_end=0"
SET "S1=A2.actor.[train].policy.eps_start=1.0 A2.actor.[train].policy.eps_end=0.06 app.batchrun.exp_exit_step=6000000"
SET "S2=A2.actor.[train].policy.eps_start=0.01 A2.actor.[train].policy.eps_end=0.01 app.batchrun.exp_exit_step=19000000"
SET "S3=%EPS005% app.batchrun.exp_exit_step=25000000"

echo === 1. a5 qbert btr schedule 0-6M: eps ladder 0.06-1.0 ===
call :run_exe "%SE%" E1.game=qbert run.seed=1 %S1% "app.run_name=run_{t}_a5_qbert_btrsched_s1"
call :find_ckpt "run_*_a5_qbert_btrsched_s1"
if not defined CKPT goto :qbert_skip
echo === 2. a5 qbert btr schedule 6-25M: eps 0.01 ===
call :run_exe "%SE%" E1.game=qbert run.seed=1 %S2% "A3.auto_load_file=%CKPT%" "app.run_name=run_{t}_a5_qbert_btrsched_s2"
call :find_ckpt "run_*_a5_qbert_btrsched_s2"
if not defined CKPT goto :qbert_skip
echo === 3. a5 qbert btr schedule 25-50M: eps ladder 0.05-0 ===
call :run_exe "%SE%" E1.game=qbert run.seed=1 %S3% "A3.auto_load_file=%CKPT%" "app.run_name=run_{t}_a5_qbert_btrsched_s3"
goto :qbert_done
:qbert_skip
echo %DATE% %TIME% [SKIP] qbert btr schedule: remaining stages skipped
:qbert_done

echo === 4. a5 battle_zone btr 50M eps ladder 0.05-0 ===
call :run_exe "%SE%" E1.game=battle_zone run.seed=1 %EPS005% "app.run_name=run_{t}_a5_battle_zone_btr50m_eps005_end0"

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
echo *** bin\%BUILD%\AnetRLRunner_ab.exe not found. Nothing was run.
pause
exit /b 1

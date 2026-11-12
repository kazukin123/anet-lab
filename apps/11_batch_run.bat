@echo off
title %~nx0

cd /d "%~dp0runner"

SET "BUILD=Release"

:waitprev
tasklist /FI "IMAGENAME eq AnetRLRunner_ab.exe" 2>nul | find /I "AnetRLRunner_ab.exe" >nul
if not errorlevel 1 (
  ping -n 31 127.0.0.1 >nul
  goto :waitprev
)
tasklist /FI "IMAGENAME eq AnetRLRunner_ab12.exe" 2>nul | find /I "AnetRLRunner_ab12.exe" >nul
if not errorlevel 1 (
  ping -n 31 127.0.0.1 >nul
  goto :waitprev
)
ping -n 16 127.0.0.1 >nul
tasklist /FI "IMAGENAME eq AnetRLRunner_ab.exe" 2>nul | find /I "AnetRLRunner_ab.exe" >nul
if not errorlevel 1 goto :waitprev

SET "SRC_EXE=%~dp0..\.scratch\r123-20261112\bin\AnetRLRunner_r123_20261112.exe"
if not exist "%SRC_EXE%" goto :no_exe
copy /Y "%SRC_EXE%" "bin\%BUILD%\AnetRLRunner_ab.exe" >nul
if errorlevel 1 goto :no_exe
if not exist "bin\%BUILD%\AnetRLRunner_ab.exe" goto :no_exe

SET RUNNER="bin\%BUILD%\AnetRLRunner_ab.exe"

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0
SET "RUN_EXIT_CODE=0"

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base"
SET "RR4=run.@hard500>run.@rr4>run.@munch"
SET "RF=run.@rfit>run.@a5_metrics"
SET "K32=DefaultDQNAgent.actor.[train].policy.tau_rule.num_taus=32 DefaultDQNAgent.actor.[eval].policy.tau_rule.num_taus=32"
SET "ALNINIT=net.block.[AtariImpalaConv32].init.mode=default net.block.[AtariImpalaConv64].init.mode=default net.block.[AtariImpalaRes32SN].init1.mode=default net.block.[AtariImpalaRes32SN].init2.mode=default net.block.[AtariImpalaRes64SN].init1.mode=default net.block.[AtariImpalaRes64SN].init2.mode=default net.block.[AtariIQNTauProj2304].init.mode=default"
SET "TGCH=run.$=%A5%>%RR4%>%RF%>run.@btrnet>run.@btrsn12>run.@cap2m>run.@eval2ch>run.@to_30m>run.@rnd>run.@batch"
SET "TGQ=DefaultDQNAgent.learner.replay_ratio=8 DefaultDQNAgent.model.hard_update_interval=250 DefaultDQNAgent.learner.per_alpha=0.5 DefaultDQNAgent.learner.per_beta.start=0.5 DefaultDQNAgent.learner.per_beta.end=0.5"
SET "TGEV=run.eval_schedule.[eval].interval=62528 run.eval_schedule.[greedy].interval=31250 run.eval_schedule.[greedy_single].interval=31250"
SET "TGRND=DefaultDQNAgent.intrinsic.learn_ratio=1 DefaultDQNAgent.intrinsic.learn_batch_size=128"
SET "MZT=E1.game=montezuma_revenge run.seed=1 run.eval.[greedy].eval_episodes=16"
SET "NMDD=DefaultDQNAgent.learner.munchausen.enabled=false DefaultDQNAgent.learner.use_double_dqn=true"
SET "R123=DefaultDQNAgent.intrinsic.rnd.warmup_steps=200000"

call :use_ws AtariRnd-01 Atari.txt

echo === 20. montezuma_revenge 30m seed 1: b001 recipe (Munchausen off, Double DQN on) with the PRD123 RND default (warm-up 200k), beta 0.01 ===
call :find_done mz_tg_r123_b001_s1
if not defined DONE_DIR call :run_ws "%TGCH%" %MZT% %K32% %ALNINIT% %TGQ% %TGRND% %TGEV% %NMDD% %R123% DefaultDQNAgent.intrinsic.beta=0.01 app.batchrun.exp_exit_step=30000000 "app.run_name=run_{t}_mz_tg_r123_b001_s1"

echo === 21. montezuma_revenge 30m seed 1: same as 20 but beta 0.003 ===
call :find_done mz_tg_r123_b0003_s1
if not defined DONE_DIR call :run_ws "%TGCH%" %MZT% %K32% %ALNINIT% %TGQ% %TGRND% %TGEV% %NMDD% %R123% DefaultDQNAgent.intrinsic.beta=0.003 app.batchrun.exp_exit_step=30000000 "app.run_name=run_{t}_mz_tg_r123_b0003_s1"

echo === 22. montezuma_revenge 30m seed 1: same as 20 but beta 0.001 ===
call :find_done mz_tg_r123_b0001_s1
if not defined DONE_DIR call :run_ws "%TGCH%" %MZT% %K32% %ALNINIT% %TGQ% %TGRND% %TGEV% %NMDD% %R123% DefaultDQNAgent.intrinsic.beta=0.001 app.batchrun.exp_exit_step=30000000 "app.run_name=run_{t}_mz_tg_r123_b0001_s1"

if "%FAILED_RUNS%"=="0" goto :all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED, %FAILED_RUNS% FAILED ===
goto :beep_forever

:all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED ===
goto :beep_forever

:use_ws
SET "WS=--workspace %~1"
SET "RUNS=workspaces\%~1\runs"
if exist "workspaces\%~1\config\_main.txt" exit /b 0
mkdir "workspaces\%~1\config" 2>nul
>"workspaces\%~1\config\_main.txt" echo $include ^<%~2^>
exit /b 0

:find_done
SET "DONE_DIR="
for /f "delims=" %%d in ('dir /b /ad "%RUNS%\run_*_%~1" 2^>nul') do (
  findstr /m /c:"Stop training requested" "%RUNS%\%%d\*.log" >nul 2>&1 && SET "DONE_DIR=%%d"
)
if defined DONE_DIR echo %DATE% %TIME% SKIP %~1 already finished in %DONE_DIR%
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

:run_exe
echo %DATE% %TIME% START --config "%CFG%" "app.runs_dir=%RUNS%" %*
%RUNNER% --config "%CFG%" "app.runs_dir=%RUNS%" %*
SET "RUN_EXIT_CODE=%ERRORLEVEL%"
if "%RUN_EXIT_CODE%"=="0" goto :run_succeeded
echo %DATE% %TIME% [ERROR] RUN FAILED exit_code=%RUN_EXIT_CODE% args=%*
SET /A FAILED_RUNS+=1
exit /b 0

:run_ws
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
echo *** exe or config not found, or the copy failed. Nothing was run.
goto :beep_forever

:beep_forever
powershell -NoProfile -Command "[console]::Beep(1000,800)" >nul 2>&1
ping -n 301 127.0.0.1 >nul
goto :beep_forever

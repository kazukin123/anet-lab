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
ping -n 16 127.0.0.1 >nul
tasklist /FI "IMAGENAME eq AnetRLRunner_ab.exe" 2>nul | find /I "AnetRLRunner_ab.exe" >nul
if not errorlevel 1 goto :waitprev

SET "SRC_EXE=%~dp0..\.scratch\rnd-explore-20261107\bin\AnetRLRunner_rnd_20261107.exe"
if not exist "%SRC_EXE%" goto :no_exe
copy /Y "%SRC_EXE%" "bin\%BUILD%\AnetRLRunner_ab.exe" >nul
if errorlevel 1 goto :no_exe
if not exist "bin\%BUILD%\AnetRLRunner_ab.exe" goto :no_exe

SET RUNNER="bin\%BUILD%\AnetRLRunner_ab.exe"

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0
SET "RUN_EXIT_CODE=0"

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base"
SET "RR1=run.@hard125>run.@munch"
SET "RR4=run.@hard500>run.@rr4>run.@munch"
SET "RF=run.@rfit>run.@a5_metrics"
SET "RT30B=run.$=%A5%>%RR1%>%RF%>run.@cap2m>run.@eval2ch_r1>run.@eval2ch_r1_50m>run.@to_30m"
SET "SE30=run.$=%A5%>%RR4%>%RF%>run.@btrnet>run.@btrsn12>run.@cap2m>run.@eval2ch>run.@to_30m"
SET "CFG=%~dp0runner\workspaces\AtariRnd-01\config\rnd_explore.txt"
if not exist "%CFG%" goto :no_exe
SET "NMRND=%RT30B%>run.@rnd>run.@rnd_nn>run.@no_munch>run.@batch"
SET "SERND=%SE30%>run.@rnd>run.@rnd_nn>run.@batch"
SET "K32=DefaultDQNAgent.actor.[train].policy.tau_rule.num_taus=32 DefaultDQNAgent.actor.[eval].policy.tau_rule.num_taus=32"
SET "ALNINIT=net.block.[AtariImpalaConv32].init.mode=default net.block.[AtariImpalaConv64].init.mode=default net.block.[AtariImpalaRes32SN].init1.mode=default net.block.[AtariImpalaRes32SN].init2.mode=default net.block.[AtariImpalaRes64SN].init1.mode=default net.block.[AtariImpalaRes64SN].init2.mode=default net.block.[AtariIQNTauProj2304].init.mode=default"
SET "MZ=E1.game=montezuma_revenge run.seed=1 DefaultDQNAgent.intrinsic.beta=0.1 run.eval.[greedy].eval_episodes=16"

call :use_ws AtariRnd-01 Atari.txt
echo === 1. montezuma_revenge 10m seed 1: RND, K32, taus fixed in train and greedy ===
call :find_done mz_k32_taufix_s1
if not defined DONE_DIR call :run_exe "%NMRND%" %MZ% %K32% DefaultDQNAgent.actor.[train].policy.tau_rule.sample_mode=fixed run.eval_schedule.[eval].interval=7816 app.batchrun.exp_exit_step=10000000 "app.run_name=run_{t}_mz_k32_taufix_s1"

echo === 2. montezuma_revenge 30m seed 1: sample-efficiency profile with RND and K32 ===
call :find_done mz_se_rnd_k32_s1
if not defined DONE_DIR call :run_exe "%SERND%" %MZ% %K32% %ALNINIT% run.eval_schedule.[eval].interval=31264 app.batchrun.exp_exit_step=30000000 "app.run_name=run_{t}_mz_se_rnd_k32_s1"

SET "EV2X2=A3.learner.enabled=false app.save_agent_on_close=false run.train.num_envs=1 DefaultDQNAgent.actor.[train].policy.spatial.enabled=false app.batchrun.exp_exit_step=40 run.eval.[greedy].eval_episodes=32 run.eval.[greedy].eval_batch_size=32 run.eval_schedule.[greedy].interval=1000 run.eval.[greedy_single].eval_episodes=32 run.eval.[greedy_single].eval_batch_size=32 run.eval_schedule.[greedy_single].interval=1000 run.eval.[eval].eval_episodes=32 run.eval.[eval].eval_batch_size=32 run.eval_schedule.[eval].interval=1000 run.eval.[eval_target].eval_episodes=32 run.eval.[eval_target].eval_batch_size=32 run.eval_schedule.[eval_target].interval=1000 DefaultDQNAgent.actor.[greedy_single].policy.tau_rule.sample_mode=random DefaultDQNAgent.actor.[eval_target].network=online DefaultDQNAgent.actor.[eval_target].policy.tau_rule.sample_mode=fixed"
SET "SEDIR=%RUNS:\=/%/run_20261108-003056_mz_se_rnd_k32_s1"
if not exist "%RUNS%\run_20261108-003056_mz_se_rnd_k32_s1\agent_close.anet" goto :no_ckpt
if not exist "%RUNS%\run_20261108-003056_mz_se_rnd_k32_s1\agent_20066688.anet" goto :no_ckpt

echo === 3. montezuma_revenge eval only: SE+RND 30m weights, eps and tau as a 2x2 ===
call :find_done mz_se_ev30m_2x2
if not defined DONE_DIR call :run_exe "%SERND%" %MZ% %K32% %ALNINIT% %EV2X2% "A3.auto_load_file=%SEDIR%/agent_close.anet" "app.run_name=run_{t}_mz_se_ev30m_2x2"

echo === 4. montezuma_revenge eval only: SE+RND 20m weights, eps and tau as a 2x2 ===
call :find_done mz_se_ev20m_2x2
if not defined DONE_DIR call :run_exe "%SERND%" %MZ% %K32% %ALNINIT% %EV2X2% "A3.auto_load_file=%SEDIR%/agent_20066688.anet" "app.run_name=run_{t}_mz_se_ev20m_2x2"

if not exist "%RUNS%\run_20261108-003056_mz_se_rnd_k32_s1\agent_14119296.anet" goto :no_ckpt
if not exist "%RUNS%\run_20261108-003056_mz_se_rnd_k32_s1\agent_15675776.anet" goto :no_ckpt

echo === 5. montezuma_revenge eval only: SE+RND 14.1m weights, eps and tau as a 2x2 ===
call :find_done mz_se_ev14m_2x2
if not defined DONE_DIR call :run_exe "%SERND%" %MZ% %K32% %ALNINIT% %EV2X2% "A3.auto_load_file=%SEDIR%/agent_14119296.anet" "app.run_name=run_{t}_mz_se_ev14m_2x2"

echo === 6. montezuma_revenge eval only: SE+RND 15.7m weights, eps and tau as a 2x2 ===
call :find_done mz_se_ev16m_2x2
if not defined DONE_DIR call :run_exe "%SERND%" %MZ% %K32% %ALNINIT% %EV2X2% "A3.auto_load_file=%SEDIR%/agent_15675776.anet" "app.run_name=run_{t}_mz_se_ev16m_2x2"

echo === 7. montezuma_revenge 30m seed 1: sample-efficiency profile with RND, K32 and TempoRL ===
call :find_done mz_se_rnd_skip_s1
if not defined DONE_DIR call :run_exe "%SERND%>run.@skip" %MZ% %K32% %ALNINIT% run.eval_schedule.[eval].interval=31264 app.batchrun.exp_exit_step=30000000 "app.run_name=run_{t}_mz_se_rnd_skip_s1"

if "%FAILED_RUNS%"=="0" goto :all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED, %FAILED_RUNS% FAILED ===
pause
exit /b 1

:all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED ===
pause
exit /b 0


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

:run_succeeded
SET /A SUCCEEDED_RUNS+=1
echo   %DATE% %TIME% END   %*
exit /b 0

:no_exe
echo *** exe or config not found, or the copy failed. Nothing was run.
pause
exit /b 1

:no_ckpt
echo *** checkpoint not found. Stages 3 and 4 were not run.
pause
exit /b 1
